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

use sdg_tools::matcher::{self, Pattern};
use sdg_tools::paper::{self, Meta, Paper, F_ANY};
use sdg_tools::query::{self, Query};
use sdg_tools::simd::find;

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
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

static APP: OnceLock<(Vec<Query>, Vec<Pattern>)> = OnceLock::new();

fn app() -> &'static (Vec<Query>, Vec<Pattern>) {
    APP.get_or_init(|| {
        let mut queries = match query::load_queries(&queries_dir()) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("[web] warning: could not load queries: {e}");
                Vec::new()
            }
        };
        let t = Instant::now();
        // Precompile every keyword once into a dense table and stamp each
        // AST leaf with its pattern index; matching then never hashes
        // keyword strings. (Previously ~21k patterns were recompiled per
        // request, which dominated the per-request cost.)
        let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
        for q in &mut queries {
            matcher::resolve_blocks(&mut q.blocks, &table);
        }
        eprintln!(
            "[web] precompiled {} patterns in {:.1} ms",
            table.len(),
            t.elapsed().as_secs_f64() * 1000.0
        );
        (queries, table)
    })
}

fn get_queries() -> &'static Vec<Query> {
    &app().0
}

fn get_patterns() -> &'static Vec<Pattern> {
    &app().1
}

// ---------------------------------------------------------------------------
// Matching (identical semantics to engine/match_paper.py)
// ---------------------------------------------------------------------------

struct SdgReport {
    sdg: String,
    matched: Vec<(usize, Vec<(Arc<str>, u8)>)>,
    near: Vec<(usize, Vec<(Arc<str>, u8)>, usize)>,
    near_total: usize,
    excluded: Vec<Arc<str>>,
    max_kw: usize,
}

/// Full report: one entry per SDG. The pattern cache is global (precompiled
/// once at boot), and each block is scanned in a single traversal that also
/// yields the boolean verdict.
fn match_report(paper: &Paper, top: usize, max_kw: usize) -> Vec<SdgReport> {
    let table = get_patterns();
    // One memo per request: keywords repeated across SDG blocks (~4.4x in
    // the corpus) are evaluated once instead of once per occurrence.
    let mut memo = matcher::Memo::new();
    let mut out = Vec::new();
    for q in get_queries() {
        let mut matched = Vec::new();
        let mut near = Vec::new();
        let mut ex: Vec<Arc<str>> = Vec::new();
        for (bno, block) in q.blocks.iter().enumerate() {
            let (hits, misses, ex_hits, is_match) = matcher::scan_with_fields(block, paper, table, &mut memo);
            ex.extend(ex_hits);
            if is_match {
                matched.push((bno, hits));
            } else {
                near.push((bno, misses, hits.len()));
            }
        }
        near.sort_by_key(|t| t.1.len()); // fewest missing keywords first
        let near_total = near.len();
        let mut exu = ex.clone();
        exu.sort_unstable();
        exu.dedup();
        out.push(SdgReport {
            sdg: q.sdg.clone(),
            matched,
            near: near.into_iter().take(top).collect(),
            near_total,
            excluded: exu,
            max_kw,
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
    Ok(crossref_json(&v, &d))
}

// ---------------------------------------------------------------------------
// HTML rendering of the match report
// ---------------------------------------------------------------------------

fn kw_tags<K: AsRef<str>>(entries: &[(K, u8)], cls: &str, max_kw: usize) -> String {
    if entries.is_empty() {
        return "<span class=\"none\">none</span>".to_string();
    }
    let mut out = String::new();
    for (kw, mask) in entries.iter().take(max_kw) {
        let field = if *mask == 0 || *mask == field_mask_all() {
            String::new()
        } else {
            format!("<span class=\"field\">[{}]</span>", matcher::field_names(*mask))
        };
        out.push_str(&format!("<span class=\"kw {cls}\">{}{field}</span>", esc(kw.as_ref())));
    }
    if entries.len() > max_kw {
        out.push_str(&format!(
            "<span class=\"muted-text\">… +{} more</span>",
            entries.len() - max_kw
        ));
    }
    out
}

/// Mask of the default TITLE-ABS-KEY search (all four section fields).
fn field_mask_all() -> u8 {
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
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
            body.push_str("<div class=\"block\"><h4>Near misses — add any of these keywords to qualify</h4>");
            for (bno, misses, n_hit) in &r.near {
                body.push_str(&format!(
                    "<div class=\"muted-text\" style=\"margin:4px 0 2px\">block {bno}: \
                     {n_hit} keyword(s) already hit</div>"
                ));
                body.push_str(&kw_tags(misses, "missing", r.max_kw));
            }
            if r.near_total > r.near.len() {
                body.push_str(&format!(
                    "<div class=\"muted-text\">… {} more near-miss blocks not shown</div>",
                    r.near_total - r.near.len()
                ));
            }
            body.push_str("</div>");
        }
        if !r.excluded.is_empty() {
            body.push_str("<div class=\"block\"><h4>Excluded terms found in the text (can disqualify a match)</h4>");
            let entries: Vec<(Arc<str>, u8)> = r.excluded.iter().map(|k| (k.clone(), 0)).collect();
            body.push_str(&kw_tags(&entries, "ex", r.max_kw));
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
    let mut terms: Vec<Arc<str>> = Vec::new();
    for r in &matched_sdgs {
        for (_, hits) in &r.matched {
            for (kw, _) in hits {
                terms.push(kw.clone());
            }
        }
    }
    terms.sort_unstable();
    terms.dedup();
    let mut hl = String::new();
    if !terms.is_empty() {
        let text = paper.full_text().trim();
        if !text.is_empty() {
            let lower = paper.text_lower(F_ANY).to_vec();
            let hl_text = highlight(&lower, text, &terms);
            hl = format!(
                "<div class=\"card highlight-card\">\n  <h3>Matched keywords highlighted in the \
                 paper text ({})</h3>\n  <div class=\"papertext\">{hl_text}</div>\n</div>",
                terms.len()
            );
        }
    }

    format!(
        "<div id=\"results-inner\"><h2 class=\"section\">Results</h2>{info_html}{stat}{chips_html}\
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

    let form_keys = ["title", "abstract", "keywords", "authors", "year", "journal", "doi"];
    let any = form_keys
        .iter()
        .any(|k| fields.get(*k).map_or(false, |v| !v.trim().is_empty()));

    let (paper, meta) = if any {
        paper_from_fields(&fields)
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
        Paper::from_text_with_meta(&text)
    };

    let top = clamp_int(fields.get("top").map(String::as_str), 3, 1, 20);
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
                            "keywords": hits.iter().map(|(kw, f)| serde_json::json!({"keyword": kw.as_ref(), "fields": matcher::field_names(*f)})).collect::<Vec<_>>(),
                        })
                    }).collect();
                    let near: Vec<serde_json::Value> = r.near.iter().map(|(bno, misses, nh)| {
                        serde_json::json!({
                            "block": bno,
                            "missing": misses.iter().map(|(kw, f)| serde_json::json!({"keyword": kw.as_ref(), "fields": matcher::field_names(*f)})).collect::<Vec<_>>(),
                            "hits": nh,
                        })
                    }).collect();
                    serde_json::json!({
                        "sdg": r.sdg,
                        "matched": matched,
                        "near": near,
                        "near_total": r.near_total,
                        "excluded": r.excluded.iter().map(|e| e.as_ref()).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<_>>(),
            });
            Resp::json(200, out.to_string())
                .with_header("X-Processing-Time", &format!("{:.1} ms", m.ms))
        }
    }
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
         Cache-Control: no-store\r\n",
        resp.code,
        resp.reason,
        resp.ctype,
        resp.body.len()
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
