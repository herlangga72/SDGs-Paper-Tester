//! web — Rust HTTP server for the SDG paper matcher (migration of web/app.py).
//!
//! Zero-framework: std::net threads, hand-rolled request/response parsing,
//! multipart + urlencoded form parsing. Matching runs on the SIMD engine
//! (rust/src/simd.rs: AVX2 lowercasing and substring search, scalar
//! fallback), so per-paper matching and keyword highlighting are much faster
//! than the Python regex version.
//!
//! Endpoints (identical to the old Python server):
//!     GET  /                       UI page
//!     GET  /static/<file>          CSS / JS
//!     GET  /samples                JSON list of sample papers (name, title, year)
//!     GET  /sample?name=&format=   raw markdown or parsed JSON fields
//!     GET  /doi?doi=...            Crossref lookup -> JSON fields
//!     POST /match                  fields -> HTML report
//!     GET  /health                 liveness
//!
//! Usage:
//!     cargo run --bin web --release [--host 127.0.0.1] [--port 8000] [--no-browser]

use sdg_tools::cache;
use sdg_tools::matcher::{self, Pattern};
use sdg_tools::paper::{self, Meta, Paper, F_ANY};
use sdg_tools::query::{self, Query};
use sdg_tools::simd::find;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// SDG metadata (official UN short names + brand colors)
// ---------------------------------------------------------------------------

const SDGS: [(&str, &str, &str); 17] = [
    ("01", "No Poverty", "#E5243B"),
    ("02", "Zero Hunger", "#DDA63A"),
    ("03", "Good Health and Well-being", "#4C9F38"),
    ("04", "Quality Education", "#C5192D"),
    ("05", "Gender Equality", "#FF3A21"),
    ("06", "Clean Water and Sanitation", "#26BDE2"),
    ("07", "Affordable and Clean Energy", "#FCC30B"),
    ("08", "Decent Work and Economic Growth", "#A21942"),
    ("09", "Industry, Innovation and Infrastructure", "#FD6925"),
    ("10", "Reduced Inequalities", "#DD1367"),
    ("11", "Sustainable Cities and Communities", "#FD9D24"),
    ("12", "Responsible Consumption and Production", "#BF8B2E"),
    ("13", "Climate Action", "#3F7E44"),
    ("14", "Life Below Water", "#0A97D9"),
    ("15", "Life on Land", "#56C02B"),
    ("16", "Peace, Justice and Strong Institutions", "#00689D"),
    ("17", "Partnerships for the Goals", "#19486A"),
];

fn sdg_name(no: &str) -> &str {
    SDGS.iter().find(|(n, _, _)| *n == no).map(|(_, name, _)| *name).unwrap_or("")
}

fn sdg_color(no: &str) -> &str {
    SDGS.iter().find(|(n, _, _)| *n == no).map(|(_, _, c)| *c).unwrap_or("#555")
}

// ---------------------------------------------------------------------------
// Paths (repo root detection: SDG_ROOT env, cwd, parent, or /app in Docker)
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    if let Ok(r) = std::env::var("SDG_ROOT") {
        if !r.is_empty() {
            return PathBuf::from(r);
        }
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    for p in [cwd.clone(), cwd.join(".."), PathBuf::from("/app")] {
        if p.join("web").join("static").join("index.html").is_file() {
            return p;
        }
    }
    cwd
}

fn static_dir() -> PathBuf {
    repo_root().join("web").join("static")
}

fn papers_dir() -> PathBuf {
    repo_root().join("papers")
}

fn queries_dir() -> PathBuf {
    repo_root().join("engine").join("data").join("queries")
}

// ---------------------------------------------------------------------------
// Query cache (loaded once; matching itself is ~1-2 s per paper in Python,
// a few ms here thanks to the SIMD matcher)
// ---------------------------------------------------------------------------

static APP: OnceLock<(
    Vec<Query>,
    &'static [Pattern],
    Vec<Vec<matcher::FlatBlock>>,
    Vec<matcher::SdgDict>,
    Vec<&'static [matcher::LeafDesc]>,
)> = OnceLock::new();

/// Per-SDG unique include-leaf list for the Advanced keyword browser: every
/// distinct (pattern pid, field mask) that appears on the include side of
/// any block of the SDG, with its resolved memo slot. `Memo::leaf_hit` over
/// this list answers "which keywords are already in the paper?" in one pass
/// per (pid, mask) - no per-block boolean VM, no hits/misses/excluded list
/// materialization, and excluded (NOT) leaves are not evaluated at all.
fn build_present_tables(flats: &[Vec<matcher::FlatBlock>]) -> Vec<&'static [matcher::LeafDesc]> {
    flats
        .iter()
        .map(|sdg| {
            let mut seen: HashSet<(u32, u8)> = HashSet::with_capacity(sdg.len().min(4096) * 2);
            let mut out: Vec<matcher::LeafDesc> = Vec::new();
            for flat in sdg {
                for l in flat.leaves {
                    if l.excluded {
                        continue;
                    }
                    if seen.insert((l.pid, l.mask)) {
                        out.push(l.clone());
                    }
                }
            }
            let leak: &'static mut [matcher::LeafDesc] = Box::leak(out.into_boxed_slice());
            let shrink: &'static [matcher::LeafDesc] = leak;
            shrink
        })
        .collect()
}

fn app() -> &'static (
    Vec<Query>,
    &'static [Pattern],
    Vec<Vec<matcher::FlatBlock>>,
    Vec<matcher::SdgDict>,
    Vec<&'static [matcher::LeafDesc]>,
) {
    APP.get_or_init(|| {
        let qdir = queries_dir();
        let t = Instant::now();
        // Boot cache: the parsed+resolved query ASTs, pattern table,
        // flattened blocks, pretokenized SDG dictionaries and per-SDG
        // present tables are persisted to sdg_cache.bin (validated by query
        // mtimes), so a restart skips the Scopus-file parse AND the
        // ~21k-keyword recompile.
        let cached = cache::read_cached(&qdir);
        let (queries, table, flats, dicts) = match cached {
            Some(data) => {
                matcher::rebuild_first_quads(data.patterns);
                eprintln!(
                    "[web] mmap'd boot cache ({} queries, {} patterns)",
                    data.queries.len(),
                    data.patterns.len()
                );
                (data.queries, data.patterns, data.flats, data.dicts)
            }
            None => {
                let mut queries = match query::load_queries(&qdir) {
                    Ok(q) => q,
                    Err(e) => {
                        eprintln!("[web] warning: could not load queries: {e}");
                        Vec::new()
                    }
                };
                // Precompile every keyword once into a dense table and stamp
                // each AST leaf with its pattern index; matching then never
                // hashes keyword strings.
                let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
                let mut nslots = 0u32;
                for q in &mut queries {
                    matcher::resolve_blocks(&mut q.blocks, &table, &mut nslots);
                }
                // Flatten every block to a postfix program once, so a request
                // never re-walks the AST (tree dispatch was ~40% of the
                // per-request time).
                let flats: Vec<Vec<matcher::FlatBlock>> = queries
                    .iter()
                    .map(|q| q.blocks.iter().map(|b| matcher::flatten_block(b, &table)).collect())
                    .collect();
                // Keyword dictionaries (unique include keywords + excluded
                // set) for suggestions and the Advanced tab.
                let dicts: Vec<matcher::SdgDict> =
                    queries.iter().map(|q| matcher::collect_sdg_dict(&q.blocks)).collect();
                if let Err(e) = cache::write_cache(
                    &qdir,
                    matcher::blob_slice(),
                    &queries,
                    &table,
                    &flats,
                    &dicts,
                ) {
                    eprintln!("[web] warning: could not write boot cache: {e}");
                }
                let table: &'static [Pattern] = Box::leak(table.into_boxed_slice());
                (queries, table, flats, dicts)
            }
        };
        let present = build_present_tables(&flats);
        eprintln!(
            "[web] ready in {:.1} ms ({} patterns, {} present leaves)",
            t.elapsed().as_secs_f64() * 1000.0,
            table.len(),
            present.iter().map(|p| p.len()).sum::<usize>()
        );
        (queries, table, flats, dicts, present)
    })
}

fn get_queries() -> &'static Vec<Query> {
    &app().0
}

fn get_patterns() -> &'static [Pattern] {
    app().1
}

fn get_flats() -> &'static Vec<Vec<matcher::FlatBlock>> {
    &app().2
}

fn get_dicts() -> &'static Vec<matcher::SdgDict> {
    &app().3
}

fn get_present() -> &'static Vec<&'static [matcher::LeafDesc]> {
    &app().4
}

// ---------------------------------------------------------------------------
// Matching (identical semantics to engine/match_paper.py)
// ---------------------------------------------------------------------------

/// One non-matching block that can still qualify, ranked by `cost`
/// (minimum keywords to add). `need` holds the missing-tag groups: any ONE
/// keyword from each group qualifies the block.
struct NearBlock {
    bno: usize,
    /// Include keywords already hit (deduped).
    n_hit: usize,
    /// Minimum keywords to add to qualify (never INF_COST here).
    cost: usize,
    /// Candidate keywords per group (any one per group).
    need: Vec<Vec<&'static str>>,
    /// The include keywords of this block already present in the paper
    /// (rendered as green "already in your text" chips).
    hits: Vec<(&'static str, u8)>,
}

struct SdgReport {
    sdg: String,
    // Keywords are borrowed from the global FlatBlocks ('static).
    matched: Vec<(usize, Vec<(&'static str, u8)>)>,
    near: Vec<NearBlock>,
    near_total: usize,
    excluded: Vec<&'static str>,
    max_kw: usize,
    /// Deterministic best-fit keyword suggestions (no LLM).
    suggestions: Vec<matcher::Suggestion>,
    /// Keywords that alone qualify the SDG: they appear in the single missing
    /// group (need.len()==1, cost==1) of some near-miss block.
    solo: HashSet<&'static str>,
    /// Per keyword that appears in a near-miss block but does NOT qualify
    /// alone: the minimum number of additional keywords still needed
    /// (cost of the cheapest block containing it, minus one).
    extra: HashMap<&'static str, usize>,
}

/// Full report: one entry per SDG. The pattern cache is global (precompiled
/// once at boot), and each block is scanned in a single traversal that also
/// yields the boolean verdict.
fn match_report(paper: &Paper, top: usize, max_kw: usize) -> Vec<SdgReport> {
    let table = get_patterns();
    // One memo per request: keywords repeated across SDG blocks (~4.4x in
    // the corpus) are evaluated once instead of once per occurrence.
    let mut memo = matcher::Memo::new(paper, 0);
    let mut out = Vec::new();
    // Paper word set, built once and reused for every SDG's suggestion
    // scoring (allocation-free after this point).
    let paper_text = String::from_utf8_lossy(paper.text_lower(paper::F_ANY));
    let words = matcher::text_words(&paper_text);
    // Scratch vectors reused across blocks (clear keeps their capacity).
    let mut hits: Vec<(&'static str, u8)> = Vec::new();
    let mut misses: Vec<(&'static str, u8)> = Vec::new();
    let mut ex_hits: Vec<&'static str> = Vec::new();
    let mut mscr = matcher::MinAddScratch::default();
    for (qi, q) in get_queries().iter().enumerate() {
        let mut matched = Vec::new();
        // (block_no, keywords already hit, min keywords to add, need groups,
        //  already-hit keyword entries)
        let mut near: Vec<(usize, usize, usize, Vec<Vec<&'static str>>, Vec<(&'static str, u8)>)> = Vec::new();
        let mut ex: Vec<&'static str> = Vec::new();
        let mut present: HashSet<&'static str> = HashSet::new();
        // Keywords that qualify the SDG on their own / still need N more.
        let mut solo: HashSet<&'static str> = HashSet::new();
        let mut extra: HashMap<&'static str, usize> = HashMap::new();
        for (bno, flat) in get_flats()[qi].iter().enumerate() {
            hits.clear();
            misses.clear();
            ex_hits.clear();
            let is_match = matcher::scan_flat_into(flat, table, &mut memo, &mut hits, &mut misses, &mut ex_hits);
            // Every include-leaf hit counts as "present" for the keyword
            // suggestions (even when its block did not match overall).
            for (kw, _) in hits.iter() {
                present.insert(*kw);
            }
            // The Scopus query files repeat terms across AND sub-groups, so
            // dedupe by (keyword identity, field mask) before rendering.
            let hits = dedupe_kw(std::mem::take(&mut hits));
            if is_match {
                matched.push((bno, hits));
            } else {
                // Cost-only near-miss analysis (zero allocation): exact
                // minimum keywords to add. INF_COST means a required-path
                // NOT is already true - the block is disqualified by an
                // excluded term, so it is NOT a near miss. The candidate
                // keyword groups are materialized below only for the blocks
                // that make the displayed list.
                let (_, cost) = matcher::min_add_flat_cost(flat, table, &mut memo, &mut mscr);
                if cost == matcher::INF_COST as u32 {
                    // Only report excluded terms when the positive side alone
                    // would have matched - i.e. the NOT genuinely blocked a
                    // near-qualifying block (off-topic blocks are dropped).
                    if matcher::eval_ignore_not_block(&get_queries()[qi].blocks[bno], table, &mut memo) {
                        ex.extend(ex_hits.iter().cloned());
                    }
                } else {
                    // Materialize the missing-tag groups for every finite-cost
                    // block: they feed both the AND-clause visualization and
                    // the per-chip "qualifies alone / needs N more" badges.
                    let ma = matcher::min_add_flat(flat, table, &mut memo, &mut mscr);
                    let cost_us = cost as usize;
                    if cost_us == 1 && ma.need.len() == 1 {
                        solo.extend(ma.need[0].iter().copied());
                    } else {
                        let need_more = cost_us.saturating_sub(1);
                        for g in &ma.need {
                            for k in g {
                                let cur = extra.get(k).copied();
                                if cur.is_none() || need_more < cur.unwrap() {
                                    extra.insert(*k, need_more);
                                }
                            }
                        }
                    }
                    near.push((bno, hits.len(), cost_us, ma.need, hits));
                }
            }
        }
        // Rerank: fewest keywords to add first; ties go to the block that
        // already hit more keywords.
        near.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.1.cmp(&a.1)));
        let near_total = near.len();
        let near_blocks: Vec<NearBlock> = near
            .into_iter()
            .take(top)
            .map(|(bno, n_hit, cost, need, hits)| NearBlock { bno, n_hit, cost, need, hits })
            .collect();
        let mut exu = ex.clone();
        exu.sort_unstable();
        exu.dedup();
        // Best-fit keywords to add: rank the SDG's include keywords by
        // word-token overlap with the paper text (pure math, no LLM).
        let suggestions = matcher::suggest_keywords(&words, &get_dicts()[qi], &present, 10);
        out.push(SdgReport {
            sdg: q.sdg.clone(),
            matched,
            near: near_blocks,
            near_total,
            excluded: exu,
            max_kw,
            suggestions,
            solo,
            extra,
        });
    }
    out
}

fn paper_from_fields(f: &HashMap<String, String>) -> (Paper, Meta) {
    let mut sections: [Option<String>; 4] = [None, None, None, None];
    let mut meta = Meta::default();
    if let Some(v) = f.get("title").map(|v| v.trim()).filter(|v| !v.is_empty()) {
        sections[0] = Some(v.to_string());
        meta.title = Some(v.to_string());
    }
    if let Some(v) = f.get("abstract").map(|v| v.trim()).filter(|v| !v.is_empty()) {
        sections[1] = Some(v.to_string());
        meta.abstract_text = Some(v.to_string());
    }
    if let Some(v) = f.get("keywords").map(|v| v.trim()).filter(|v| !v.is_empty()) {
        let kws: Vec<String> = v
            .split(|c: char| c == ';' || c == ',')
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .collect();
        let joined = kws.join(", ");
        sections[2] = Some(joined.clone());
        sections[3] = Some(joined);
        meta.keywords = kws;
    }
    if let Some(v) = f.get("authors").map(|v| v.trim()).filter(|v| !v.is_empty()) {
        meta.authors = v
            .split(|c: char| c == ';' || c == ',')
            .map(|a| a.trim().to_string())
            .filter(|a| !a.is_empty())
            .collect();
    }
    if let Some(v) = f.get("year").map(|v| v.trim()).filter(|v| !v.is_empty()) {
        meta.year = Some(v.to_string());
    }
    if let Some(v) = f.get("journal").map(|v| v.trim()).filter(|v| !v.is_empty()) {
        meta.journal = Some(v.to_string());
    }
    if let Some(v) = f.get("doi").map(|v| v.trim()).filter(|v| !v.is_empty()) {
        meta.doi = Some(v.to_string());
    }
    (Paper::from_sections(sections), meta)
}

// ---------------------------------------------------------------------------
// Keyword highlighting in the paper text (span-based, HTML-safe)
// ---------------------------------------------------------------------------

#[inline]
fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Every occurrence of a plain term, word-bounded (SIMD substring search).
fn find_all_boundary(hay: &[u8], needle: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(p) = find(hay, needle, from) {
        let before = p == 0 || !is_word(hay[p - 1]);
        let e = p + needle.len();
        let after = e >= hay.len() || !is_word(hay[e]);
        if before && after {
            out.push((p, e));
        }
        from = p + 1;
    }
    out
}

/// Non-greedy glob match from one start position: `*` = any run of bytes
/// (shortest first), `?` = exactly one byte. Returns the match end.
/// Line-local and non-greedy, mirroring the Python highlighter's
/// `[^\n]*?` pattern. `budget` bounds worst-case backtracking on
/// pathological single-line inputs.
fn glob_from(pat: &[u8], pi: usize, line: &[u8], pos: usize, budget: &mut usize) -> Option<usize> {
    if *budget == 0 {
        return None;
    }
    *budget -= 1;
    if pi == pat.len() {
        return Some(pos);
    }
    match pat[pi] {
        b'*' => {
            for k in pos..=line.len() {
                if let Some(e) = glob_from(pat, pi + 1, line, k, budget) {
                    return Some(e);
                }
            }
            None
        }
        b'?' => {
            if pos < line.len() {
                glob_from(pat, pi + 1, line, pos + 1, budget)
            } else {
                None
            }
        }
        c => {
            if pos < line.len() && line[pos] == c {
                glob_from(pat, pi + 1, line, pos + 1, budget)
            } else {
                None
            }
        }
    }
}

/// Leftmost, shortest match of a glob pattern within a single line.
fn glob_match_span(pat: &[u8], line: &[u8], budget: &mut usize) -> Option<(usize, usize)> {
    for s in 0..=line.len() {
        if let Some(e) = glob_from(pat, 0, line, s, budget) {
            return Some((s, e));
        }
    }
    None
}

/// All spans a keyword matches in the lowercased text (positions map 1:1 to
/// the original text since ASCII lowercasing preserves byte length).
fn hl_term_spans(lower: &[u8], kw: &str) -> Vec<(usize, usize)> {
    let lk = kw.to_ascii_lowercase();
    let has_star = lk.contains('*');
    let has_q = lk.contains('?');
    if !has_star && !has_q {
        return find_all_boundary(lower, lk.as_bytes());
    }
    let pat = lk.as_bytes();
    let mut spans = Vec::new();
    let mut budget = 2_000_000usize;
    let mut line_start = 0usize;
    for line in lower.split(|&b| b == b'\n') {
        if line.len() > 32 * 1024 {
            line_start += line.len() + 1;
            continue; // degenerate single-line input: skip wildcard spans
        }
        let mut from = 0usize;
        loop {
            match glob_match_span(pat, &line[from..], &mut budget) {
                Some((s, e)) => {
                    spans.push((line_start + from + s, line_start + from + e));
                    from += if e > 0 { e } else { 1 };
                    if from > line.len() {
                        break;
                    }
                }
                None => break,
            }
        }
        line_start += line.len() + 1;
    }
    spans
}

fn highlight(lower: &[u8], orig: &str, terms: &[impl AsRef<str>]) -> String {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for t in terms {
        spans.extend(hl_term_spans(lower, t.as_ref()));
    }
    if spans.is_empty() {
        return esc(orig);
    }
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in spans {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }
    let len = orig.len();
    let worst = merged.iter().map(|(s, e)| e - s).max().unwrap_or(0);
    // one degenerate catch-all term (e.g. '*') -> skip highlighting
    if worst as f64 > len as f64 * 0.8 {
        return esc(orig);
    }
    let mut out = String::new();
    let mut pos = 0usize;
    for (s, e) in merged {
        out.push_str(&esc(&orig[pos..s]));
        out.push_str("<mark>");
        out.push_str(&esc(&orig[s..e]));
        out.push_str("</mark>");
        pos = e;
    }
    out.push_str(&esc(&orig[pos..]));
    out
}

// ---------------------------------------------------------------------------
// Escaping (HTML + JSON)
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(c),
        }
    }
    out
}

fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Small HTML entity unescaper (for Crossref titles/abstracts, which ship
/// JATS/HTML entities).
fn html_unescape(s: &str) -> String {
    fn entity(e: &str) -> Option<char> {
        if let Some(num) = e.strip_prefix('#') {
            let v = if let Some(hex) = num.strip_prefix('x').or_else(|| num.strip_prefix('X')) {
                u32::from_str_radix(hex, 16).ok()?
            } else {
                num.parse::<u32>().ok()?
            };
            return char::from_u32(v);
        }
        Some(match e {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            "nbsp" => '\u{a0}',
            "ndash" => '–',
            "mdash" => '—',
            "hellip" => '…',
            "ldquo" => '\u{201c}',
            "rdquo" => '\u{201d}',
            "lsquo" => '\u{2018}',
            "rsquo" => '\u{2019}',
            _ => return None,
        })
    }
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let n = chars.len();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < n {
        let (_, c) = chars[i];
        if c == '&' {
            let mut j = i + 1;
            while j < n && j <= i + 12 && chars[j].1 != ';' {
                j += 1;
            }
            if j < n && j <= i + 12 {
                let ent: String = chars[i + 1..j].iter().map(|x| x.1).collect();
                if let Some(c) = entity(&ent) {
                    out.push(c);
                    i = j + 1;
                    continue;
                }
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Strip JATS tags and collapse whitespace (Crossref abstracts).
fn strip_tags_collapse(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    let mut pending_space = false;
    for c in s.chars() {
        if c == '<' {
            in_tag = true;
            pending_space = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag {
            if c.is_whitespace() {
                pending_space = true;
            } else {
                if pending_space && !out.is_empty() {
                    out.push(' ');
                }
                pending_space = false;
                out.push(c);
            }
        }
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// DOI lookup (Crossref REST API — free, no key; requires internet)
// ---------------------------------------------------------------------------

const CROSSREF_UA: &str = "sdg-paper-matcher/2.0 (local paper-matching app, Rust)";
const LANDING_UA: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/120.0 Safari/537.36";

fn normalize_doi(doi: &str) -> String {
    let mut d = doi.trim();
    for p in [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ] {
        if let Some(rest) = d.strip_prefix(p) {
            d = rest;
        }
    }
    d.to_string()
}

fn valid_doi(doi: &str) -> bool {
    let b = doi.as_bytes();
    if !b.starts_with(b"10.") {
        return false;
    }
    let mut i = 3;
    let mut nd = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        nd += 1;
        i += 1;
    }
    if !(4..=9).contains(&nd) || i >= b.len() || b[i] != b'/' {
        return false;
    }
    i += 1;
    i < b.len() && b[i..].iter().all(|c| !c.is_ascii_whitespace())
}

/// Value of the `key` attribute inside a single tag, with either quote
/// style (or bare); `None` when absent. The value keeps its original case
/// (byte offsets are identical in the lowercased copy).
fn attr_value_of(tag: &str, key: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let mut from = 0usize;
    while let Some(rel) = lower[from..].find(key) {
        let mut j = from + rel + key.len();
        while j < lower.len() && (lower.as_bytes()[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= lower.len() || lower.as_bytes()[j] != b'=' {
            from = j + 1;
            continue;
        }
        j += 1;
        while j < lower.len() && (lower.as_bytes()[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= lower.len() {
            return None;
        }
        let (s, e) = match lower.as_bytes()[j] {
            b'"' => {
                let s = j + 1;
                (s, lower[s..].find('"').map(|x| s + x).unwrap_or(lower.len()))
            }
            b'\'' => {
                let s = j + 1;
                (s, lower[s..].find('\'').map(|x| s + x).unwrap_or(lower.len()))
            }
            _ => {
                let s = j;
                (
                    s,
                    lower[s..]
                        .find(|c: char| c.is_whitespace() || c == '>')
                        .map(|x| s + x)
                        .unwrap_or(lower.len()),
                )
            }
        };
        if e > s {
            return Some(tag[s..e].to_string());
        }
        from = e + 1;
    }
    None
}

/// Value of `<meta name="NAME" content="...">` (name matched case-insensitively).
fn meta_tag_content(html: &str, name: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase(); // same byte length as `html`
    let mut from = 0usize;
    while let Some(rel) = lower[from..].find("<meta") {
        let start = from + rel;
        let tag_end = lower[start..].find('>').map(|x| start + x).unwrap_or(lower.len());
        if attr_value_of(&lower[start..tag_end], "name").as_deref() == Some(name) {
            if let Some(v) = attr_value_of(&html[start..tag_end], "content") {
                let v = html_unescape(&v).trim().to_string();
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
        from = start + 1;
    }
    None
}

/// Split a keywords string on commas/semicolons: trim, drop empties, dedupe.
fn split_keywords(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for part in s.split([',', ';']) {
        let t = part.trim();
        if !t.is_empty() && seen.insert(t.to_string()) {
            out.push(t.to_string());
        }
    }
    out
}

/// Text of the `<div class="txt">` that follows a `Keywords` label
/// (Business Perspectives renders author keywords that way).
fn keywords_label_content(html: &str) -> Option<String> {
    let mut from = 0usize;
    while let Some(rel) = html[from..].find("Keywords") {
        let i = from + rel;
        let tail = &html[i + "Keywords".len()..];
        if let Some(d) = tail.find("<div class=\"txt\">") {
            let content = &tail[d + "<div class=\"txt\">".len()..];
            let end = content.find("</div>").unwrap_or(content.len());
            let text = strip_tags_collapse(&content[..end]);
            if !text.is_empty() {
                return Some(text);
            }
        }
        from = i + 1;
    }
    None
}

/// Best-effort author keywords from the DOI landing page, used when
/// Crossref has no `subject` (the closest thing it usually has to
/// keywords). Tries the Google-Scholar `citation_keywords` meta tag first
/// (Springer/Elsevier/IEEE/…), then the `Keywords</strong><div class="txt">`
/// markup used by Business Perspectives. Never fails the DOI lookup: on any
/// network error, timeout or parse miss it returns an empty list.
fn fetch_landing_keywords(doi: &str) -> Vec<String> {
    let url = format!("https://doi.org/{}", percent_encode(doi));
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(12)))
        .max_redirects(10)
        .build()
        .new_agent();
    let mut resp = match agent
        .get(&url)
        .header("User-Agent", LANDING_UA)
        .header("Accept", "text/html")
        .call()
    {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let html = match resp.body_mut().read_to_string() {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };
    if let Some(v) = meta_tag_content(&html, "citation_keywords") {
        let kws = split_keywords(&v);
        if !kws.is_empty() {
            return kws;
        }
    }
    if let Some(v) = keywords_label_content(&html) {
        return split_keywords(&v);
    }
    Vec::new()
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Crossref JSON message -> the same shape as the sample endpoint
/// (title, authors, year, journal, doi, abstract, keywords).
fn crossref_json(v: &serde_json::Value, doi: &str) -> String {
    let msg = &v["message"];
    let mut parts = vec![format!("\"doi\":{}", jstr(doi))];
    if let Some(t) = msg["title"].as_array().and_then(|a| a.first()).and_then(|x| x.as_str()) {
        let t = html_unescape(t).trim().to_string();
        if !t.is_empty() {
            parts.push(format!("\"title\":{}", jstr(&t)));
        }
    }
    let mut authors: Vec<String> = Vec::new();
    if let Some(arr) = msg["author"].as_array() {
        for a in arr {
            let given = a["given"].as_str().unwrap_or("");
            let family = a["family"].as_str().unwrap_or("");
            let name = a["name"].as_str().unwrap_or("");
            let n = if given.is_empty() && family.is_empty() {
                name.to_string()
            } else {
                format!("{given} {family}").trim().to_string()
            };
            if !n.is_empty() {
                authors.push(html_unescape(&n));
            }
        }
    }
    if !authors.is_empty() {
        let arr: Vec<String> = authors.iter().map(|a| jstr(a)).collect();
        parts.push(format!("\"authors\":[{}]", arr.join(",")));
    }
    if let Some(t) = msg["container-title"].as_array().and_then(|a| a.first()).and_then(|x| x.as_str()) {
        let t = html_unescape(t).trim().to_string();
        if !t.is_empty() {
            parts.push(format!("\"journal\":{}", jstr(&t)));
        }
    }
    for key in ["issued", "published-print", "published-online"] {
        if let Some(y) = msg[key]["date-parts"][0][0].as_i64() {
            parts.push(format!("\"year\":{}", jstr(&y.to_string())));
            break;
        }
    }
    if let Some(abs) = msg["abstract"].as_str() {
        let abs = strip_tags_collapse(&html_unescape(abs));
        if !abs.is_empty() {
            parts.push(format!("\"abstract\":{}", jstr(&abs)));
        }
    }
    let mut subjects: Vec<String> = Vec::new();
    if let Some(arr) = msg["subject"].as_array() {
        for s in arr {
            if let Some(s) = s.as_str() {
                let s = html_unescape(s).trim().to_string();
                if !s.is_empty() {
                    subjects.push(s);
                }
            }
        }
    }
    if !subjects.is_empty() {
        let arr: Vec<String> = subjects.iter().map(|s| jstr(s)).collect();
        parts.push(format!("\"keywords\":[{}]", arr.join(",")));
    }
    format!("{{{}}}", parts.join(","))
}

fn fetch_doi(doi: &str) -> Result<String, (u16, String)> {
    let d = normalize_doi(doi);
    if !valid_doi(&d) {
        return Err((
            400,
            format!("not a valid DOI: {doi:?} (expected e.g. 10.1257/jep.28.4.99)"),
        ));
    }
    let url = format!("https://api.crossref.org/works/{}", percent_encode(&d));
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(12)))
        .build()
        .new_agent();
    let mut resp = match agent
        .get(&url)
        .header("User-Agent", CROSSREF_UA)
        .header("Accept", "application/json")
        .call()
    {
        Ok(r) => r,
        Err(ureq::Error::StatusCode(code)) => {
            return if code == 404 {
                Err((400, format!("DOI not found in Crossref: {d}")))
            } else {
                Err((502, format!("Crossref API error {code}")))
            };
        }
        Err(e) => return Err((502, format!("network error: {e}"))),
    };
    let body = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| (502, format!("network error: {e}")))?;
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| (502, format!("bad Crossref response: {e}")))?;
    let mut body = crossref_json(&v, &d);
    if !body.contains("\"keywords\"") {
        // Crossref has no subject for this DOI: fall back to the DOI
        // landing page's author keywords (best-effort, never fails).
        let kws = fetch_landing_keywords(&d);
        if !kws.is_empty() {
            let arr: Vec<String> = kws.iter().map(|s| jstr(s)).collect();
            body = format!("{},\"keywords\":[{}]}}", &body[..body.len() - 1], arr.join(","));
        }
    }
    Ok(body)
}

// ---------------------------------------------------------------------------
// HTML rendering of the match report
// ---------------------------------------------------------------------------

fn chip(kw: &str, mask: u8, cls: &str) -> String {
    let field = if mask == 0 || mask == field_mask_all() {
        String::new()
    } else {
        format!("<span class=\"field\">[{}]</span>", matcher::field_names(mask))
    };
    format!("<span class=\"kw {cls}\">{}{field}</span>", esc(kw))
}

/// One keyword chip list: the first `max_kw` chips plus a "Show all N"
/// toggle button revealing the full list (see app.js `.kw-toggle` handler).
fn kw_tags<K: AsRef<str>>(entries: &[(K, u8)], cls: &str, max_kw: usize) -> String {
    if entries.is_empty() {
        return "<span class=\"none\">none</span>".to_string();
    }
    let n = entries.len();
    let take = max_kw.min(n);
    let mut out = String::new();
    for (kw, mask) in entries.iter().take(take) {
        out.push_str(&chip(kw.as_ref(), *mask, cls));
    }
    if n > take {
        let rest: String = entries
            .iter()
            .skip(take)
            .map(|(kw, mask)| chip(kw.as_ref(), *mask, cls))
            .collect();
        out.push_str(&format!(
            "<span class=\"kw-more\" hidden>{rest}</span>\
             <button type=\"button\" class=\"kw-toggle\" data-all=\"Show all {n}\" \
             data-few=\"Show fewer\">Show all {n}</button>"
        ));
    }
    out
}

/// Render one near-miss path as AND-joined "pick any one" boxes.
fn render_need_chain(need: &[Vec<&'static str>], max_kw: usize, body: &mut String) {
    body.push_str("<div class=\"and-chain\">");
    let n_groups = need.len();
    for (gi, g) in need.iter().take(3).enumerate() {
        if gi > 0 {
            body.push_str("<div class=\"and-op\">AND</div>");
        }
        body.push_str(&format!(
            "<div class=\"and-group\"><div class=\"and-group-label\">Box {} — pick any ONE:</div>",
            gi + 1
        ));
        let entries: Vec<(&'static str, u8)> = g.iter().map(|k| (*k, 0)).collect();
        body.push_str(&kw_tags(&entries, "missing", max_kw));
        body.push_str("</div>");
    }
    if n_groups > 3 {
        let mut rest = String::new();
        for (gi, g) in need.iter().enumerate().skip(3) {
            rest.push_str("<div class=\"and-op\">AND</div>");
            rest.push_str(&format!(
                "<div class=\"and-group\"><div class=\"and-group-label\">Box {} — pick any ONE:</div>",
                gi + 1
            ));
            let entries: Vec<(&'static str, u8)> = g.iter().map(|k| (*k, 0)).collect();
            rest.push_str(&kw_tags(&entries, "missing", max_kw));
            rest.push_str("</div>");
        }
        body.push_str(&format!(
            "<div class=\"kw-more kw-more-block\" hidden>{rest}</div>\
             <button type=\"button\" class=\"kw-toggle\" data-all=\"Show all {n_groups} boxes\" \
             data-few=\"Show fewer boxes\">Show all {n_groups} boxes</button>"
        ));
    }
    body.push_str("</div>");
}

/// Mask of the default TITLE-ABS-KEY search (all four section fields).
fn field_mask_all() -> u8 {
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
}

/// Drop repeated (keyword, field-mask) entries, keyed on the Arc string
/// identity + mask so the same keyword in different field contexts stays.
fn dedupe_kw(entries: Vec<(&'static str, u8)>) -> Vec<(&'static str, u8)> {
    let mut seen: HashSet<(usize, u8)> = HashSet::with_capacity(entries.len());
    entries
        .into_iter()
        .filter(|(kw, m)| seen.insert((kw.as_ptr() as *const u8 as usize, *m)))
        .collect()
}

fn render_results(report: &[SdgReport], paper: &Paper, meta: &Meta, ms: f64) -> String {
    let mut meta_parts: Vec<String> = Vec::new();
    if let Some(t) = &meta.title {
        meta_parts.push(format!("<b>{}</b>", esc(t)));
    }
    if !meta.authors.is_empty() {
        meta_parts.push(esc(&meta.authors.join(", ")));
    }
    for v in [&meta.year, &meta.journal, &meta.doi] {
        if let Some(v) = v {
            meta_parts.push(esc(v));
        }
    }
    let info_html = if meta_parts.is_empty() {
        String::new()
    } else {
        format!("<div class=\"card paper-info\">{}</div>", meta_parts.join(" · "))
    };

    let matched_sdgs: Vec<&SdgReport> = report.iter().filter(|r| !r.matched.is_empty()).collect();
    let near_sdgs: Vec<&SdgReport> =
        report.iter().filter(|r| r.matched.is_empty() && !r.near.is_empty()).collect();
    let ex_sdgs: Vec<&SdgReport> = report.iter().filter(|r| !r.excluded.is_empty()).collect();

    let mut chips = String::new();
    for r in &matched_sdgs {
        let color = sdg_color(&r.sdg);
        chips.push_str(&format!(
            "<span class=\"chip matched\" style=\"color:{color}\"><span class=\"dot\" \
             style=\"background:{color}\"></span>SDG {} ✓</span>",
            r.sdg
        ));
    }
    for r in &near_sdgs {
        let color = sdg_color(&r.sdg);
        chips.push_str(&format!(
            "<span class=\"chip near\" style=\"color:{color}\"><span class=\"dot\" \
             style=\"background:{color}\"></span>SDG {} near</span>",
            r.sdg
        ));
    }
    for r in &ex_sdgs {
        let color = sdg_color(&r.sdg);
        chips.push_str(&format!(
            "<span class=\"chip\" style=\"color:{color}\"><span class=\"dot\" \
             style=\"background:{color}\"></span>SDG {} ⚠ excluded terms</span>",
            r.sdg
        ));
    }
    let chips_html = if chips.is_empty() {
        "<div class=\"chips\"><span class=muted-text>no SDG signal found</span></div>".to_string()
    } else {
        format!("<div class=\"chips\">{chips}</div>")
    };

    let stat = format!(
        "<div class=\"stat\"><b>{}</b> of <b>17</b> SDGs matched · <b>{}</b> near misses · \
         <b>{}</b> with excluded terms found · processed in <b>{:.1}</b> ms</div>",
        matched_sdgs.len(),
        near_sdgs.len(),
        ex_sdgs.len(),
        ms
    );

    let mut cards = String::new();
    for r in report {
        if r.matched.is_empty() && r.near.is_empty() && r.excluded.is_empty() {
            continue;
        }
        let color = sdg_color(&r.sdg);
        let mut badges = String::new();
        if !r.matched.is_empty() {
            badges.push_str(&format!("<span class=\"badge ok\">✓ {} block(s) matched</span>", r.matched.len()));
        }
        if !r.near.is_empty() {
            badges.push_str(&format!("<span class=\"badge miss\">{} near miss(es)</span>", r.near.len()));
        }
        if !r.excluded.is_empty() {
            badges.push_str("<span class=\"badge ex\">excluded terms found</span>");
        }

        let mut body = String::new();
        if !r.matched.is_empty() {
            body.push_str("<div class=\"block\"><h4>Matched blocks — keywords that hit</h4>");
            for (bno, hits) in &r.matched {
                body.push_str(&format!(
                    "<div class=\"muted-text\" style=\"margin:4px 0 2px\">block {bno}</div>"
                ));
                body.push_str(&kw_tags(hits, "hit", r.max_kw));
            }
            body.push_str("</div>");
        }
        if !r.near.is_empty() {
            body.push_str("<div class=\"block\"><h4>How to qualify this SDG</h4>");
            let (first, rest) = r.near.split_first().unwrap();
            let word = if first.cost == 1 { "keyword" } else { "keywords" };
            body.push_str(&format!(
                "<div class=\"status-line\">Your text is <b>{} {}</b> short of qualifying for <b>SDG {} — {}</b>.</div>",
                first.cost,
                word,
                r.sdg,
                esc(sdg_name(&r.sdg))
            ));
            // Fastest (cheapest) path, green callout.
            body.push_str("<div class=\"fastest\">");
            body.push_str(&format!(
                "<div class=\"fastest-head\">⚡ Fastest — add {} {} (pick any one from each box):</div>",
                first.cost, word
            ));
            render_need_chain(&first.need, r.max_kw, &mut body);
            body.push_str(
                "<div class=\"min-hint\">1 keyword from each box = the SDG qualifies. Click a chip to add it to your Keywords field.</div>",
            );
            body.push_str("<div class=\"have-line\"><b>Already in your text:</b> ");
            if first.hits.is_empty() {
                body.push_str("<span class=\"none\">none yet</span>");
            } else {
                body.push_str(&kw_tags(&first.hits, "hit", r.max_kw));
            }
            body.push_str("</div>");
            body.push_str("</div>");
            // Other paths, collapsed.
            if !rest.is_empty() {
                let mut alts = String::new();
                for (ai, nb) in rest.iter().enumerate() {
                    let w = if nb.cost == 1 { "keyword" } else { "keywords" };
                    alts.push_str(&format!(
                        "<div class=\"way-head\">Way {} — add {} {}:</div>",
                        ai + 2,
                        nb.cost,
                        w
                    ));
                    render_need_chain(&nb.need, r.max_kw, &mut alts);
                }
                body.push_str(&format!(
                    "<div class=\"kw-more kw-more-block\" hidden>{alts}</div>\
                     <button type=\"button\" class=\"kw-toggle\" data-all=\"Show {} other ways to qualify\" \
                     data-few=\"Hide other ways\">Show {} other ways to qualify</button>",
                    rest.len(),
                    rest.len()
                ));
            }
            if r.near_total > r.near.len() {
                body.push_str(&format!(
                    "<div class=\"muted-text\">… {} more ways not shown</div>",
                    r.near_total - r.near.len()
                ));
            }
            body.push_str("</div>");
        }
        if !r.excluded.is_empty() {
            body.push_str("<div class=\"block\"><h4>Excluded terms that blocked a near match — remove them from the text to qualify</h4>");
            let entries: Vec<(&'static str, u8)> = r.excluded.iter().map(|k| (*k, 0)).collect();
            body.push_str(&kw_tags(&entries, "ex", r.max_kw));
            body.push_str("</div>");
        }
        if !r.suggestions.is_empty() {
            let heading = if r.matched.is_empty() {
                "Best-fit keywords to add (click to copy)"
            } else {
                "Related keywords from this SDG — best fit to your text (click to copy)"
            };
            body.push_str(&format!("<div class=\"block\"><h4>{heading}</h4>"));
            if r.matched.is_empty() {
                body.push_str(
                    "<div class=\"sug-legend\"><span class=\"sug-badge solo\">✓ alone</span> qualifies by itself · \
                     <span class=\"sug-badge more\">+N</span> still needs N more keyword(s) · \
                     <span class=\"sug-badge block\">⚠ blocked</span> excluded term — adding it blocks a match</div>",
                );
            }
            body.push_str("<div class=\"sug-row\">");
            for s in r.suggestions.iter().take(r.max_kw.min(10)) {
                let pct = (s.score * 100.0).round() as u32;
                let badge = if s.excluded_in_sdg {
                    "<span class=\"sug-badge block\" title=\"also an excluded (NOT) term in this SDG — adding it can block a match\">⚠ blocked</span>".to_string()
                } else if r.matched.is_empty() && r.solo.contains(s.keyword.as_ref()) {
                    "<span class=\"sug-badge solo\" title=\"this keyword alone qualifies the SDG\">✓ alone</span>".to_string()
                } else if r.matched.is_empty() {
                    match r.extra.get(s.keyword.as_ref()) {
                        Some(n) => format!("<span class=\"sug-badge more\" title=\"still needs {n} more keyword(s) — see the near-miss boxes above\">+{n} more</span>"),
                        None => "<span class=\"sug-badge more\" title=\"does not qualify by itself — see the near-miss boxes above\">+ more</span>".to_string(),
                    }
                } else {
                    String::new()
                };
                body.push_str(&format!(
                    "<button type=\"button\" class=\"kw sug\" data-kw=\"{}\">{}<span class=\"score\">{pct}%</span>{badge}</button>",
                    esc(&s.keyword),
                    esc(&s.keyword)
                ));
            }
            body.push_str("</div>");
            body.push_str(&format!(
                "<div class=\"muted-text\">Auto-ranked by word overlap with your text — no AI. Open the <b>Advanced</b> tab for the full SDG {} keyword list.</div>",
                r.sdg
            ));
            body.push_str("</div>");
        }

        let open = if r.matched.is_empty() { "" } else { " open" };
        cards.push_str(&format!(
            "\n<details class=\"card sdg-card\"{open}>\n  <summary class=\"sdg-head\">\n    \
             <span class=\"num\" style=\"background:{color}\">{}</span>\n    <span class=\"name\">{}</span>\n    \
             <span class=\"badges\">{badges}</span>\n    <span class=\"chev\">▼</span>\n  </summary>\n  \
             <div class=\"sdg-body\">{body}</div>\n</details>",
            r.sdg,
            esc(sdg_name(&r.sdg))
        ));
    }

    // highlight all matched keywords in the full paper text
    let mut terms: Vec<&'static str> = Vec::new();
    for r in &matched_sdgs {
        for (_, hits) in &r.matched {
            for (kw, _) in hits {
                terms.push(*kw);
            }
        }
    }
    terms.sort_unstable();
    terms.dedup();
    let mut hl = String::new();
    if !terms.is_empty() {
        let text = paper.full_text().trim();
        if !text.is_empty() {
            // Pass the lowercased buffer by reference: no need to copy the
            // whole text just to highlight matching substrings.
            let hl_text = highlight(paper.text_lower(F_ANY), text, &terms);
            hl = format!(
                "<div class=\"card highlight-card\">\n  <h3>Matched keywords highlighted in the \
                 paper text ({})</h3>\n  <div class=\"papertext\">{hl_text}</div>\n</div>",
                terms.len()
            );
        }
    }

    let explainer = "<details class=\"card explainer\"><summary>How SDG matching works (30 seconds)</summary>\
        <p>Each SDG is made of several <b>keyword paths</b>. A paper qualifies for an SDG as soon as <b>one full path</b> \
        is present in its text. For every SDG you are close to, we show the <b>shortest missing path</b> first: \
        pick <b>one keyword from each box</b> and the SDG qualifies. Click any suggested keyword to add it to your \
        Keywords field (it is copied to the clipboard too).</p></details>";

    format!(
        "<div id=\"results-inner\"><h2 class=\"section\">Results</h2>{info_html}{stat}{chips_html}{explainer}\
         <div id=\"cards\">{cards}</div>{hl}</div>"
    )
}

fn error_box(msg: &str) -> String {
    format!("<div class=\"error-box\">{}</div>", esc(msg))
}

// ---------------------------------------------------------------------------
// Form parsing (application/x-www-form-urlencoded + multipart/form-data)
// ---------------------------------------------------------------------------

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn percent_decode(b: &[u8]) -> String {
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                if let (Some(h), Some(l)) = (hex_val(b[i + 1]), hex_val(b[i + 2])) {
                    out.push(h * 16 + l);
                    i += 3;
                    continue;
                }
                out.push(b'%');
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parse_urlencoded(body: &[u8]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for pair in body.split(|&b| b == b'&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.iter().position(|&b| b == b'=') {
            Some(i) => (&pair[..i], &pair[i + 1..]),
            None => (pair, &b""[..]),
        };
        let k = percent_decode(k);
        let v = percent_decode(v);
        out.entry(k).or_insert(v);
    }
    out
}

fn trim_ascii(mut b: &[u8]) -> &[u8] {
    while let Some((&c, rest)) = b.split_first() {
        if c.is_ascii_whitespace() {
            b = rest;
        } else {
            break;
        }
    }
    while let Some((&c, rest)) = b.split_last() {
        if c.is_ascii_whitespace() {
            b = rest;
        } else {
            break;
        }
    }
    b
}

fn unquote_header(v: &[u8]) -> String {
    let mut s = String::from_utf8_lossy(v).into_owned();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s = s[1..s.len() - 1].replace("\\\"", "\"");
    }
    s
}

fn boundary_of(ctype: &str) -> Option<String> {
    for part in ctype.split(';') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("boundary=") {
            let rest = rest.trim().trim_matches('"');
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    None
}

/// Parse a multipart/form-data body into (fields, files).
fn parse_multipart(body: &[u8], boundary: &str) -> (HashMap<String, String>, HashMap<String, Vec<u8>>) {
    let mut fields: HashMap<String, String> = HashMap::new();
    let mut files: HashMap<String, Vec<u8>> = HashMap::new();
    let delim = format!("--{boundary}");
    let db = delim.as_bytes();
    let mut pos = 0usize;
    while let Some(d) = find(body, db, pos) {
        let mut p = d + db.len();
        if p < body.len() && body[p] == b'\r' {
            p += 1;
        }
        if p < body.len() && body[p] == b'\n' {
            p += 1;
        }
        if p + 1 < body.len() && body[p] == b'-' && body[p + 1] == b'-' {
            break; // closing delimiter
        }
        let Some(h_end) = find(body, b"\r\n\r\n", p) else { break };
        let Some(next) = find(body, db, h_end + 4) else { break };
        let mut cend = next;
        if cend > h_end + 4 && body[cend - 1] == b'\n' {
            cend -= 1;
        }
        if cend > h_end + 4 && body[cend - 1] == b'\r' {
            cend -= 1;
        }
        let content = &body[h_end + 4..cend];
        let headers = &body[p..h_end];

        let mut name: Option<String> = None;
        let mut filename: Option<String> = None;
        for line in headers.split(|&c| c == b'\n') {
            let line = if line.ends_with(b"\r") { &line[..line.len() - 1] } else { line };
            let t = trim_ascii(line);
            if t.len() >= 20 && t[..20].eq_ignore_ascii_case(b"content-disposition:") {
                let params = trim_ascii(&t[20..]);
                for seg in params.split(|&c| c == b';') {
                    let seg = trim_ascii(seg);
                    if let Some(v) = seg.strip_prefix(b"name=") {
                        name = Some(unquote_header(v));
                    } else if let Some(v) = seg.strip_prefix(b"filename=") {
                        filename = Some(unquote_header(v));
                    }
                }
            }
        }
        if let Some(name) = name {
            if filename.is_some() {
                files.insert(name, content.to_vec());
            } else {
                fields.insert(name, String::from_utf8_lossy(content).into_owned());
            }
        }
        pos = d + db.len(); // continue right after the delimiter we just consumed
    }
    (fields, files)
}

fn clamp_int(v: Option<&str>, default: usize, lo: usize, hi: usize) -> usize {
    let n = v
        .and_then(|s| if s.is_empty() { None } else { s.trim().parse::<usize>().ok() })
        .unwrap_or(default);
    n.clamp(lo, hi)
}

// ---------------------------------------------------------------------------
// Sample papers
// ---------------------------------------------------------------------------

fn sample_meta(name: &str) -> Option<(Meta, String)> {
    let fname = Path::new(name).file_name()?.to_string_lossy().into_owned();
    if Path::new(&fname).extension().and_then(|e| e.to_str()) != Some("md") {
        return None;
    }
    let p = papers_dir().join(&fname);
    if !p.is_file() {
        return None;
    }
    let text = fs::read_to_string(&p).ok()?;
    let (pairs, _) = paper::parse_frontmatter(&text);
    Some((Meta::from_pairs(&pairs), text))
}

fn meta_json(meta: &Meta, raw: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(t) = &meta.title {
        parts.push(format!("\"title\":{}", jstr(t)));
    }
    if !meta.authors.is_empty() {
        let arr: Vec<String> = meta.authors.iter().map(|a| jstr(a)).collect();
        parts.push(format!("\"authors\":[{}]", arr.join(",")));
    }
    if let Some(y) = &meta.year {
        parts.push(format!("\"year\":{}", jstr(y)));
    }
    if let Some(j) = &meta.journal {
        parts.push(format!("\"journal\":{}", jstr(j)));
    }
    if let Some(d) = &meta.doi {
        parts.push(format!("\"doi\":{}", jstr(d)));
    }
    if !meta.keywords.is_empty() {
        let arr: Vec<String> = meta.keywords.iter().map(|k| jstr(k)).collect();
        parts.push(format!("\"keywords\":[{}]", arr.join(",")));
    }
    if let Some(a) = &meta.abstract_text {
        parts.push(format!("\"abstract\":{}", jstr(a)));
    }
    parts.push(format!("\"raw\":{}", jstr(raw)));
    format!("{{{}}}", parts.join(","))
}

fn samples_json() -> String {
    let mut names: Vec<PathBuf> = match fs::read_dir(papers_dir()) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "md"))
            .collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    let mut arr: Vec<String> = Vec::new();
    for p in names {
        let text = fs::read_to_string(&p).unwrap_or_default();
        let (pairs, _) = paper::parse_frontmatter(&text);
        let meta = Meta::from_pairs(&pairs);
        let fname = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let title = meta.title.clone().unwrap_or_else(|| fname.clone());
        let year = meta.year.clone().unwrap_or_default();
        arr.push(format!(
            "{{\"name\":{},\"title\":{},\"year\":{}}}",
            jstr(&fname),
            jstr(&title),
            jstr(&year)
        ));
    }
    format!("[{}]", arr.join(","))
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

fn mime_for(name: &str) -> &'static str {
    match Path::new(name).extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("md") => "text/markdown; charset=utf-8",
        Some("txt") => "text/plain; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

struct Resp {
    code: u16,
    reason: &'static str,
    ctype: String,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
}

impl Resp {
    fn html(code: u16, body: String) -> Resp {
        Resp { code, reason: "OK", ctype: "text/html; charset=utf-8".into(), body: body.into_bytes(), headers: Vec::new() }
    }
    fn json(code: u16, body: String) -> Resp {
        Resp { code, reason: "OK", ctype: "application/json; charset=utf-8".into(), body: body.into_bytes(), headers: Vec::new() }
    }
    fn text(code: u16, body: &str) -> Resp {
        Resp { code, reason: "OK", ctype: "text/plain; charset=utf-8".into(), body: body.as_bytes().to_vec(), headers: Vec::new() }
    }
    fn bytes(code: u16, body: Vec<u8>, ctype: &str) -> Resp {
        Resp { code, reason: "OK", ctype: ctype.into(), body, headers: Vec::new() }
    }
    fn not_found() -> Resp {
        Resp { code: 404, reason: "Not Found", ctype: "text/plain; charset=utf-8".into(), body: b"not found".to_vec(), headers: Vec::new() }
    }
    fn with_header(mut self, name: &str, value: &str) -> Resp {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

fn route_get(path: &str, qs: &str) -> Resp {
    match path {
        "/" | "/index.html" => {
            let p = static_dir().join("index.html");
            match fs::read(&p) {
                Ok(b) => Resp::bytes(200, b, "text/html; charset=utf-8"),
                Err(_) => Resp::not_found(),
            }
        }
        "/health" => Resp::text(200, "ok"),
        "/samples" => Resp::json(200, samples_json()),
        "/sample" => {
            let params = parse_urlencoded(qs.as_bytes());
            let name = params.get("name").cloned().unwrap_or_default();
            let fmt = params.get("format").cloned().unwrap_or_default();
            let Some((meta, raw)) = sample_meta(&name) else {
                return Resp::text(404, "sample not found");
            };
            if fmt == "json" {
                Resp::json(200, meta_json(&meta, &raw))
            } else {
                Resp::bytes(200, raw.into_bytes(), "text/markdown; charset=utf-8")
            }
        }
        "/doi" => {
            let params = parse_urlencoded(qs.as_bytes());
            let doi = params.get("doi").cloned().unwrap_or_default();
            match fetch_doi(&doi) {
                Ok(body) => Resp::json(200, body),
                Err((code, msg)) => Resp::json(code, format!("{{\"error\":{}}}", jstr(&msg))),
            }
        }
        _ if path.starts_with("/static/") => {
            // basename only, no traversal
            let rel = &path["/static/".len()..];
            let fname = Path::new(rel)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if fname.is_empty() {
                return Resp::not_found();
            }
            let p = static_dir().join(&fname);
            match fs::read(&p) {
                Ok(b) => Resp::bytes(200, b, mime_for(&fname)),
                Err(_) => Resp::not_found(),
            }
        }
        _ => Resp::not_found(),
    }
}

struct MatchOutcome {
    paper: Paper,
    meta: Meta,
    report: Vec<SdgReport>,
    ms: f64,
}

/// Parse the /match form (urlencoded or multipart), build the paper, and run
/// the report. Shared by the HTML endpoint and the JSON API.
/// Build a Paper from the request fields (form fields when present, else
/// raw pasted text / uploaded file). Shared by /match and /api/keywords.
fn paper_from_request(
    fields: &mut HashMap<String, String>,
    files: &HashMap<String, Vec<u8>>,
) -> Result<(Paper, Meta), String> {
    let form_keys = ["title", "abstract", "keywords", "authors", "year", "journal", "doi"];
    let any = form_keys
        .iter()
        .any(|k| fields.get(*k).map_or(false, |v| !v.trim().is_empty()));
    if any {
        Ok(paper_from_fields(fields))
    } else {
        // raw pasted text, or uploaded file
        let mut text = fields.remove("paper").unwrap_or_default();
        if text.trim().is_empty() {
            for fname in ["file", "paper"] {
                if let Some(data) = files.get(fname) {
                    text = String::from_utf8_lossy(data).into_owned();
                    break;
                }
            }
        }
        if text.trim().is_empty() {
            return Err(
                "No paper entered — fill in the form (Title / Abstract / Keywords), \
                 paste raw text, or upload a file."
                    .to_string(),
            );
        }
        Ok(Paper::from_owned_with_meta(text))
    }
}

fn run_match(headers: &[(String, String)], body: &[u8]) -> Result<MatchOutcome, String> {
    let t0 = Instant::now();
    let ctype = headers
        .iter()
        .find(|(k, _)| k == "content-type")
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    let (mut fields, files) = if ctype.contains("multipart/form-data") {
        match boundary_of(&ctype) {
            Some(b) => parse_multipart(body, &b),
            None => (HashMap::new(), HashMap::new()),
        }
    } else {
        (parse_urlencoded(body), HashMap::new())
    };

    let (paper, meta) = paper_from_request(&mut fields, &files)?;

    let top = clamp_int(fields.get("top").map(String::as_str), 30, 1, 30);
    let max_kw = clamp_int(fields.get("maxkw").map(String::as_str), 10, 1, 50);
    let report = match_report(&paper, top, max_kw);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(MatchOutcome { paper, meta, report, ms })
}

fn route_match(headers: &[(String, String)], body: &[u8]) -> Resp {
    match run_match(headers, body) {
        Err(msg) => Resp::html(200, error_box(&msg)),
        Ok(m) => Resp::html(200, render_results(&m.report, &m.paper, &m.meta, m.ms))
            .with_header("X-Processing-Time", &format!("{:.1} ms", m.ms)),
    }
}

/// POST /api/match — same input as /match, JSON report out (for scripts/CLI).
fn api_match(headers: &[(String, String)], body: &[u8]) -> Resp {
    match run_match(headers, body) {
        Err(msg) => Resp::json(400, format!("{{\"error\":{}}}", jstr(&msg))),
        Ok(m) => {
            let out = serde_json::json!({
                "ms": m.ms,
                "sdgs": m.report.iter().map(|r| {
                    let matched: Vec<serde_json::Value> = r.matched.iter().map(|(bno, hits)| {
                        serde_json::json!({
                            "block": bno,
                            "keywords": hits.iter().map(|(kw, f)| serde_json::json!({"keyword": kw, "fields": matcher::field_names(*f)})).collect::<Vec<_>>(),
                        })
                    }).collect();
                    let near: Vec<serde_json::Value> = r.near.iter().map(|nb| {
                        let need: Vec<serde_json::Value> = nb.need.iter().map(|g| {
                            serde_json::json!({
                                "group": g.iter().map(|k| *k).collect::<Vec<&str>>(),
                            })
                        }).collect();
                        let missing: Vec<serde_json::Value> = nb.need.iter().flatten()
                            .map(|k| serde_json::json!({"keyword": *k, "fields": ""}))
                            .collect();
                        serde_json::json!({
                            "block": nb.bno,
                            "hits": nb.n_hit,
                            "cost": nb.cost,
                            "need": need,
                            "missing": missing,
                        })
                    }).collect();
                    let suggestions: Vec<serde_json::Value> = r.suggestions.iter().map(|s| {
                        serde_json::json!({
                            "keyword": s.keyword.as_ref(),
                            "score": (s.score * 100.0).round() as u32,
                            "excluded": s.excluded_in_sdg,
                            "qualifies_alone": r.solo.contains(s.keyword.as_ref()),
                            "extra_needed": r.extra.get(s.keyword.as_ref()).copied(),
                        })
                    }).collect();
                    serde_json::json!({
                        "sdg": r.sdg,
                        "matched": matched,
                        "near": near,
                        "near_total": r.near_total,
                        "excluded": r.excluded.iter().map(|e| e).collect::<Vec<_>>(),
                        "suggestions": suggestions,
                    })
                }).collect::<Vec<_>>(),
            });
            Resp::json(200, out.to_string())
                .with_header("X-Processing-Time", &format!("{:.1} ms", m.ms))
        }
    }
}

/// POST /api/keywords — full keyword list of one SDG, scored against the
/// paper's text (Advanced tab). Same form fields as /api/match plus `sdg`
/// and optional `limit`. Deterministic token-overlap ranking, no LLM.
fn api_keywords(headers: &[(String, String)], body: &[u8]) -> Resp {
    let ctype = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-type"));
    let (fields, files) = match ctype.and_then(|(_, v)| boundary_of(v)) {
        Some(b) => parse_multipart(body, &b),
        None => (parse_urlencoded(body), HashMap::new()),
    };
    let sdg = fields.get("sdg").map(String::as_str).unwrap_or("10").trim().to_string();
    let qi = match get_queries().iter().position(|q| q.sdg == sdg) {
        Some(i) => i,
        None => {
            return Resp::json(400, format!("{{\"error\":{}}}", jstr(&format!("unknown sdg {sdg}"))));
        }
    };
    let limit = clamp_int(fields.get("limit").map(String::as_str), 300, 1, 2000);
    let mut fields = fields;
    let (paper, _meta) = match paper_from_request(&mut fields, &files) {
        Ok(p) => p,
        Err(msg) => return Resp::json(400, format!("{{\"error\":{}}}", jstr(&msg))),
    };
    let table = get_patterns();
    // This endpoint only needs to know which keywords of the SDG are ALREADY
    // in the paper text ("present" chips). Evaluate each distinct include
    // (pattern, field-mask) exactly once through the memoized leaf_hit over
    // the boot-time present table (build_present_tables): no per-block
    // boolean-VM scan, no hits/misses/excluded vector pushes, and excluded
    // (NOT) leaves are never searched, so fewer real SIMD searches run.
    let mut memo = matcher::Memo::new(&paper, 0);
    let mut present: HashSet<&'static str> = HashSet::new();
    for l in get_present()[qi] {
        if memo.leaf_hit(&table[l.pid as usize], l.mask, l.slot) {
            present.insert(l.raw());
        }
    }
    let paper_text = String::from_utf8_lossy(paper.text_lower(paper::F_ANY));
    let words = matcher::text_words(&paper_text);
    let scored = matcher::score_keywords(&words, &get_dicts()[qi], &present, limit);
    let n_present = present.len();
    let total = get_dicts()[qi].len();
    let kws: Vec<serde_json::Value> = scored
        .iter()
        .map(|s| {
            serde_json::json!({
                "keyword": s.keyword.as_ref(),
                "score": (s.score * 100.0).round() as u32,
                "present": s.present,
                "excluded": s.excluded_in_sdg,
            })
        })
        .collect();
    let out = serde_json::json!({
        "sdg": sdg,
        "sdg_name": sdg_name(&sdg),
        "total": total,
        "present": n_present,
        "keywords": kws,
        "limit": limit,
    });
    Resp::json(200, out.to_string())
}

fn route(method: &str, target: &str, headers: &[(String, String)], body: &[u8]) -> Resp {
    let (path, qs) = match target.find('?') {
        Some(i) => (&target[..i], &target[i + 1..]),
        None => (target, ""),
    };
    match method {
        "GET" => route_get(path, qs),
        "POST" if path == "/match" => route_match(headers, body),
        "POST" if path == "/api/match" => api_match(headers, body),
        "POST" if path == "/api/keywords" => api_keywords(headers, body),
        _ => Resp::not_found(),
    }
}

fn accepts_gzip(headers: &[(String, String)]) -> bool {
    headers
        .iter()
        .find(|(k, _)| k == "accept-encoding")
        .map_or(false, |(_, v)| v.to_ascii_lowercase().contains("gzip"))
}

fn compressible(ctype: &str) -> bool {
    ctype.starts_with("text/")
        || ctype.contains("json")
        || ctype.contains("javascript")
        || ctype.contains("svg")
}

fn maybe_gzip(resp: &mut Resp, headers: &[(String, String)]) {
    if resp.body.len() < 512 || !accepts_gzip(headers) || !compressible(&resp.ctype) {
        return;
    }
    use std::io::Write;
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    if enc.write_all(&resp.body).is_ok() {
        if let Ok(gz) = enc.finish() {
            resp.body = gz;
            resp.headers.push(("Content-Encoding".into(), "gzip".into()));
            resp.headers.push(("Vary".into(), "Accept-Encoding".into()));
        }
    }
}

fn handle_conn(stream: &mut TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    let header_end = loop {
        match stream.read(&mut tmp) {
            Ok(0) => return,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if let Some(p) = find(&buf, b"\r\n\r\n", 0) {
                    break p;
                }
                if buf.len() > 1 << 20 {
                    return;
                }
            }
            Err(_) => return,
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]);
    let mut lines = head.split("\r\n");
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut content_length = 0usize;
    for l in lines {
        if let Some(ci) = l.find(':') {
            let k = l[..ci].trim().to_ascii_lowercase();
            let v = l[ci + 1..].trim().to_string();
            if k == "content-length" {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }

    let body_start = header_end + 4;
    let mut body = buf[body_start..].to_vec();
    while body.len() < content_length {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
            Err(_) => break,
        }
    }
    body.truncate(content_length);

    let mut resp = route(&method, &target, &headers, &body);
    maybe_gzip(&mut resp, &headers);
    let mut out = Vec::with_capacity(resp.body.len() + 256);
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\
         Cache-Control: no-store\r\nX-SIMD-Usage: {}\r\n",
        resp.code,
        resp.reason,
        resp.ctype,
        resp.body.len(),
        sdg_tools::simd::dispatch_name()
    );
    for (k, v) in &resp.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(&resp.body);
    let _ = stream.write_all(&out);
    eprintln!("[web] {method} {target} -> {}", resp.code);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn self_check(url: &str) -> i32 {
    let rest = url.strip_prefix("http://").unwrap_or(url);
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match hostport.rfind(':') {
        Some(i) => (&hostport[..i], hostport[i + 1..].parse().unwrap_or(80)),
        None => (hostport, 80),
    };
    match TcpStream::connect((host, port)) {
        Ok(mut s) => {
            let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
            let req = format!("GET {path} HTTP/1.0\r\nHost: {hostport}\r\nConnection: close\r\n\r\n");
            if s.write_all(req.as_bytes()).is_err() {
                return 1;
            }
            let mut buf = [0u8; 512];
            let n = s.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]);
            if head.starts_with("HTTP/1.0 200") || head.starts_with("HTTP/1.1 200") {
                0
            } else {
                1
            }
        }
        Err(_) => 1,
    }
}

#[cfg(target_os = "linux")]
fn open_browser(url: &str) {
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

#[cfg(target_os = "macos")]
fn open_browser(url: &str) {
    let _ = std::process::Command::new("open").arg(url).spawn();
}

#[cfg(target_os = "windows")]
fn open_browser(url: &str) {
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn open_browser(_url: &str) {}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // `--self-check URL` runs one health request and exits (Docker HEALTHCHECK)
    if let Some(pos) = args.iter().position(|a| a == "--self-check") {
        let url = args
            .get(pos + 1)
            .cloned()
            .unwrap_or_else(|| "http://127.0.0.1:7860/health".to_string());
        std::process::exit(self_check(&url));
    }

    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8000;
    let mut no_browser = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--host" => {
                if let Some(v) = it.next() {
                    host = v.clone();
                }
            }
            "--port" => {
                if let Some(v) = it.next() {
                    if let Ok(p) = v.parse() {
                        port = p;
                    }
                }
            }
            "--no-browser" => no_browser = true,
            other => eprintln!("[web] unknown option {other}"),
        }
    }

    let n = get_queries().len();
    eprintln!("[web] loaded {n} SDG query sets");

    let addr = format!("{host}:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[web] cannot bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    let url = format!("http://{host}:{port}/");
    eprintln!("[web] SDG Paper Matcher (Rust + SIMD) running at {url}  (Ctrl-C to stop)");
    if !no_browser {
        open_browser(&url);
    }

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                std::thread::spawn(move || {
                    let mut s = s;
                    handle_conn(&mut s);
                });
            }
            Err(e) => eprintln!("[web] accept error: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoded_basic() {
        let f = parse_urlencoded(b"title=a+b&abstract=x%20y&doi=10.1/abc");
        assert_eq!(f.get("title").unwrap(), "a b");
        assert_eq!(f.get("abstract").unwrap(), "x y");
        assert_eq!(f.get("doi").unwrap(), "10.1/abc");
    }

    #[test]
    fn multipart_basic() {
        let boundary = "XxBoundaryXx";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nMy Paper\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"p.md\"\r\n\
             Content-Type: text/markdown\r\n\r\n# body\r\n--{boundary}--\r\n"
        );
        let (fields, files) = parse_multipart(body.as_bytes(), boundary);
        assert_eq!(fields.get("title").unwrap(), "My Paper");
        assert_eq!(files.get("file").unwrap(), b"# body");
    }

    #[test]
    fn boundary_and_doi_validation() {
        assert_eq!(boundary_of("multipart/form-data; boundary=abc").unwrap(), "abc");
        assert_eq!(boundary_of("multipart/form-data; boundary=\"a b\"").unwrap(), "a b");
        assert!(valid_doi("10.1257/jep.28.4.99"));
        assert!(valid_doi("10.1000/182"));
        assert!(!valid_doi("10.123/xy"));
        assert!(!valid_doi("10.12345xy"));
        assert!(!valid_doi("jep.28.4.99"));
        assert_eq!(normalize_doi("https://doi.org/10.1/a b"), "10.1/a b");
    }

    #[test]
    fn html_unescape_and_tags() {
        assert_eq!(html_unescape("a &amp; b &lt;x&gt; &#39;c&#39;"), "a & b <x> 'c'");
        assert_eq!(strip_tags_collapse("<p>Hello   <i>world</i></p>  now"), "Hello world now");
    }

    #[test]
    fn highlight_plain_and_wildcard() {
        // plain term: word-bounded occurrences
        let text = "The coral reef and a coral reef system. Coral!";
        let lower = text.to_ascii_lowercase();
        let out = highlight(lower.as_bytes(), text, &["coral reef".to_string()]);
        assert!(out.contains("<mark>coral reef</mark>"));
        assert!(out.contains("<mark>coral reef</mark> system"));
        assert!(!out.contains("<mark>Coral!</mark>")); // no trailing boundary

        // wildcard: line-local, non-greedy (matches Python's `[^\n]*?`
        // semantics: the span ends at the first " countr", same as the
        // Python highlighter), case-insensitive
        let text2 = "We study developing countries and developing country policies.";
        let lower2 = text2.to_ascii_lowercase();
        let out2 = highlight(lower2.as_bytes(), text2, &["developing* countr*".to_string()]);
        assert!(out2.contains("<mark>developing countr</mark>ies"), "{out2}");
        assert!(out2.contains("<mark>developing countr</mark>y policies"), "{out2}");
    }

    #[test]
    fn glob_leftmost_shortest() {
        let mut budget = 1000;
        // leftmost
        assert_eq!(glob_match_span(b"a*b", b"xxaab", &mut budget), Some((2, 5)));
        // shortest expansion of the star
        assert_eq!(glob_match_span(b"a*b", b"ab", &mut budget), Some((0, 2)));
        // '?' matches exactly one byte
        assert_eq!(glob_match_span(b"a?b", b"axb", &mut budget), Some((0, 3)));
        assert_eq!(glob_match_span(b"a?b", b"ab", &mut budget), None);
        // star matches zero bytes
        assert_eq!(glob_match_span(b"ab*", b"ab", &mut budget), Some((0, 2)));
    }
}

#[cfg(test)]
mod keyword_parse_tests {
    use super::*;

    #[test]
    fn meta_citation_keywords() {
        let html = "<html><head>\
            <meta name=\"citation_keywords\" content=\"climate change; green finance\">\
            <meta name=\"keywords\" content=\"junk site keywords\">\
            </head></html>";
        assert_eq!(meta_tag_content(html, "citation_keywords").as_deref(),
                   Some("climate change; green finance"));
        assert_eq!(meta_tag_content(html, "keywords").as_deref(), Some("junk site keywords"));
        assert_eq!(meta_tag_content("<meta name='citation_keywords' content='a; b'>", "citation_keywords"),
                   Some("a; b".to_string()));
        assert_eq!(meta_tag_content("<meta name=\"keywords\">", "keywords"), None);
    }

    #[test]
    fn keywords_label_div_txt() {
        let html = "<li><strong class=\"title\">Keywords</strong>\
            <div class=\"txt\"><a href=\"/tag/a\">descriptive analysis</a>, \
            <a href=\"/tag/b\">Islamic finance</a></div></li>";
        let txt = keywords_label_content(html).expect("label keywords");
        assert_eq!(split_keywords(&txt), vec!["descriptive analysis", "Islamic finance"]);
        // no label -> nothing
        assert_eq!(keywords_label_content("<p>no keywords here</p>"), None);
    }

    #[test]
    fn split_keywords_commas_and_semicolons() {
        assert_eq!(split_keywords("a, b; c , d"), vec!["a", "b", "c", "d"]);
        assert_eq!(split_keywords("a, a, b"), vec!["a", "b"]);
        assert_eq!(split_keywords(", ; "), Vec::<String>::new());
    }

    #[test]
    fn meta_tag_not_found_returns_none() {
        assert_eq!(meta_tag_content("<meta name=\"description\" content=\"x\">", "citation_keywords"), None);
    }
}
