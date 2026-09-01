//! sdg_tools — parse Scopus SDG query files into a keyword table, and check
//! papers against them. SIMD (AVX2) with scalar fallback, zero dependencies.
//!
//! Usage:
//!   sdg_tools parse [DIR] [-o out.csv] [--dedup] [--quiet]
//!   sdg_tools match <paper.md> [--dir DIR] [--top N] [--max-kw M]
//!
//! The web server lives in src/bin/web.rs (build with `cargo build --bin web`).

use sdg_tools::ast::Node;
use sdg_tools::matcher;

use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::Path;

fn usage() -> ! {
    eprintln!(
        "sdg_tools — Scopus SDG query parser and paper matcher (Rust, SIMD)\n\
         \n\
         usage:\n\
         \x20 sdg_tools parse [DIR] [-o out.csv] [--dedup] [--quiet]\n\
         \x20 sdg_tools match <paper.md|-> [--dir DIR] [--top N] [--max-kw M]\n\
         \n\
         parse: read SDG*.txt query files -> keyword table CSV\n\
         match: check a paper (YAML frontmatter) against the SDG queries"
    );
    std::process::exit(2);
}

fn main() {
    // Detect the best SIMD route once at startup; every SIMD helper then
    // dispatches off this single cached decision (see simd::best_level).
    eprintln!("sdg_tools: using {} SIMD route", sdg_tools::simd::dispatch_name());

    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(AsRef::<str>::as_ref).unwrap_or("");
    let rest = &args[1..];
    let res = match cmd {
        "parse" => cmd_parse(rest),
        "match" => cmd_match(rest),
        _ => usage(),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// parse
// ---------------------------------------------------------------------------

struct Row {
    sdg: String,
    inc_exc: String,
    where_to_look: String,
    keyword: String,
    block_no: usize,
    logic: String,
    exact: bool,
}

fn walk(node: &Node, excluded: bool, fields: &[String], sdg: &str, block_no: usize, logic: &[String], rows: &mut Vec<Row>) {
    match node {
        Node::Leaf { keyword, exact, .. } => rows.push(Row {
            sdg: sdg.to_string(),
            inc_exc: if excluded { "exclude".to_string() } else { "include".to_string() },
            where_to_look: if fields.is_empty() { String::new() } else { fields.join("-") },
            keyword: keyword.clone(),
            block_no,
            logic: logic.join(">"),
            exact: *exact,
        }),
        Node::Field { fields: fs, child } => walk(child, excluded, fs, sdg, block_no, logic, rows),
        Node::Not { child } => walk(child, !excluded, fields, sdg, block_no, logic, rows),
        Node::Group { op, children } => {
            let mut nl = logic.to_vec();
            nl.push(op.clone());
            for c in children {
                walk(c, excluded, fields, sdg, block_no, &nl, rows);
            }
        }
    }
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn write_csv(path: &str, rows: &[Row]) -> Result<(), String> {
    let mut s = String::from("sdgs_no,include_or_exclude,where_to_look,keyword,block_no,logic,exact\n");
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            csv_field(&r.sdg),
            csv_field(&r.inc_exc),
            csv_field(&r.where_to_look),
            csv_field(&r.keyword),
            r.block_no,
            r.logic,
            if r.exact { 1 } else { 0 }
        ));
    }
    fs::write(path, s).map_err(|e| format!("cannot write {path}: {e}"))
}

fn cmd_parse(args: &[String]) -> Result<(), String> {
    let mut dir = ".".to_string();
    let mut out = "sdg_keywords.csv".to_string();
    let mut dedup = false;
    let mut quiet = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--output" => out = it.next().ok_or("missing value for -o")?.clone(),
            "--dedup" => dedup = true,
            "-q" | "--quiet" => quiet = true,
            s if s.starts_with('-') && s != "-" => return Err(format!("unknown option {s}")),
            s => dir = s.to_string(),
        }
    }

    let queries = load_queries(Path::new(&dir))?;
    let mut rows: Vec<Row> = Vec::new();
    for q in &queries {
        for (bno, block) in q.blocks.iter().enumerate() {
            walk(block, false, &[], &q.sdg, bno, &["OR".to_string()], &mut rows);
        }
    }

    if dedup {
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        rows.retain(|r| seen.insert((r.sdg.clone(), r.inc_exc.clone(), r.keyword.clone())));
    }

    write_csv(&out, &rows)?;

    if !quiet {
        println!("wrote {} rows -> {out}", rows.len());
        for q in &queries {
            let rs: Vec<&Row> = rows.iter().filter(|r| r.sdg == q.sdg).collect();
            let n_inc = rs.iter().filter(|r| r.inc_exc == "include").count();
            let n_exc = rs.len() - n_inc;
            let mut uniq: HashSet<(&str, &str)> = HashSet::new();
            for r in &rs {
                uniq.insert((r.inc_exc.as_str(), r.keyword.as_str()));
            }
            println!(
                "  SDG {:>2}: {:>6} rows ({} include, {} exclude, {} unique keyword/field combos)",
                q.sdg,
                rs.len(),
                n_inc,
                n_exc,
                uniq.len()
            );
        }
        println!("  total: {} rows", rows.len());
        let mut w2l: Vec<&str> = rows.iter().map(|r| r.where_to_look.as_str()).collect();
        w2l.sort_unstable();
        w2l.dedup();
        println!("  where_to_look values: {w2l:?}");
        let mut excl: Vec<&str> = rows.iter().filter(|r| r.inc_exc == "exclude").map(|r| r.keyword.as_str()).collect();
        excl.sort_unstable();
        excl.dedup();
        println!("  excluded keywords ({}): {excl:?}", excl.len());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// match
// ---------------------------------------------------------------------------

fn cmd_match(args: &[String]) -> Result<(), String> {
    let mut dir = ".".to_string();
    let mut top = 3usize;
    let mut max_kw = 10usize;
    let mut paper_path: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dir" => dir = it.next().ok_or("missing value for --dir")?.clone(),
            "--top" => top = it.next().ok_or("missing value for --top")?.parse().map_err(|_| "invalid --top")?,
            "--max-kw" => max_kw = it.next().ok_or("missing value for --max-kw")?.parse().map_err(|_| "invalid --max-kw")?,
            s if s.starts_with('-') && s != "-" => return Err(format!("unknown option {s}")),
            s => paper_path = Some(s.to_string()),
        }
    }
    let path = paper_path.ok_or("no paper file given (use '-' for stdin)")?;

    let text = if path == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).map_err(|e| format!("stdin: {e}"))?;
        buf
    } else {
        fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?
    };

    let paper = Paper::from_owned(text);
    let title = paper
        .title
        .clone()
        .unwrap_or_else(|| Path::new(&path).file_name().map_or(path.clone(), |n| n.to_string_lossy().into_owned()));
    println!("Paper: {title}\n");

    let qdir = Path::new(&dir);
    // Boot cache: parsed+resolved ASTs, patterns, flats and pretokenized
    // dicts are persisted in sdg_cache.bin and validated by query mtimes, so
    // every CLI invocation skips the Scopus-file parse and the ~70-80 ms
    // keyword recompile.
    let (mut queries, table, flats) = match sdg_tools::cache::read_cached(qdir) {
        Some(data) => {
            matcher::rebuild_first_quads(data.patterns);
            (data.queries, data.patterns, data.flats)
        }
        None => {
            let mut queries = load_queries(qdir)?;
            // Precompile every keyword once into a dense table and resolve
            // each leaf to its pattern index; scanning is then read-only.
            let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
            let mut nslots = 0u32;
            for q in &mut queries {
                matcher::resolve_blocks(&mut q.blocks, &table, &mut nslots);
            }
            // Flatten every block to a postfix program (min-add runs on the
            // flat program with SoA stacks - cache-friendly, no per-node
            // allocations).
            let flats: Vec<Vec<matcher::FlatBlock>> = queries
                .iter()
                .map(|q| q.blocks.iter().map(|b| matcher::flatten_block(b, &table)).collect())
                .collect();
            let dicts: Vec<matcher::SdgDict> =
                queries.iter().map(|q| matcher::collect_sdg_dict(&q.blocks)).collect();
            if let Err(e) = sdg_tools::cache::write_cache(
                qdir,
                matcher::blob_slice(),
                &queries,
                &table,
                &flats,
                &dicts,
            ) {
                eprintln!("[sdg_tools] warning: could not write boot cache: {e}");
            }
            let table: &'static [matcher::Pattern] = Box::leak(table.into_boxed_slice());
            (queries, table, flats)
        }
    };

    // One shared memo for the whole scan: buffers are indexed once and each
    // distinct (pattern, field-mask) is searched once. Without this the CLI
    // rebuilt a TextIndex and re-evaluated every term for every block, which
    // is ~60x slower on large papers.
    let mut memo = matcher::Memo::new(&paper, 0); // 0 -> memo grows on demand
    let mut mscr = matcher::MinAddScratch::default();

    for (qi, q) in queries.iter().enumerate() {
        println!("=== SDG {} ===", q.sdg);
        let mut matched: Vec<(usize, Vec<&'static str>)> = Vec::new();
        // (block_no, keywords already hit, min keywords to add, need groups)
        let mut near: Vec<(usize, usize, usize, Vec<Vec<&'static str>>)> = Vec::new();
        let mut disqualified: Vec<&'static str> = Vec::new();

        for (bno, block) in q.blocks.iter().enumerate() {
            let (scan, is_match) = matcher::scan_block_shared(block, &paper, &table, &mut memo);
            let n_hit = scan.hits.len();
            if is_match {
                matched.push((bno, scan.hits));
            } else {
                // Exact minimum keywords to add (no LLM): a block whose
                // required-path NOT is already true cannot qualify by adding
                // keywords -> report its excluded terms instead. Cost-only
                // flat pass: sequential, zero-allocation.
                let (_, cost) = matcher::min_add_flat_cost(&flats[qi][bno], &table, &mut memo, &mut mscr);
                if cost == matcher::INF_COST as u32 {
                    // Only report excluded terms when the positive side alone
                    // would have matched - i.e. the NOT genuinely blocked a
                    // near-qualifying block (off-topic blocks are dropped).
                    if matcher::eval_ignore_not_block(block, &table, &mut memo) {
                        disqualified.extend(scan.excluded_hits.iter().cloned());
                    }
                } else {
                    near.push((bno, n_hit, cost as usize, Vec::new()));
                }
            }
        }
        // Rerank by fewest keywords to add, then most keywords already hit.
        near.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| need_total(&a.3).cmp(&need_total(&b.3)))
        });
        // Materialize the missing-tag groups only for the displayed blocks.
        let near_shown: Vec<(usize, usize, usize, Vec<Vec<&'static str>>)> = near
            .into_iter()
            .take(top)
            .map(|(bno, n_hit, cost, _)| {
                let ma = matcher::min_add_flat(&flats[qi][bno], &table, &mut memo, &mut mscr);
                (bno, n_hit, cost, ma.need)
            })
            .collect();

        if matched.is_empty() {
            println!("  MATCHED  none");
        } else {
            for (bno, hits) in &matched {
                let shown: Vec<&str> = hits.iter().take(max_kw).map(AsRef::<str>::as_ref).collect();
                let more = hits.len() - shown.len();
                println!(
                    "  MATCHED  block {bno}: {} keyword(s) hit: {}{}",
                    hits.len(),
                    shown.join(", "),
                    if more > 0 { " ..." } else { "" }
                );
            }
        }

        if near_shown.is_empty() {
            println!("  NEAR MISSES none");
        } else {
            println!("  NEAR MISSES (add any 1 keyword from each group to qualify):");
            for (bno, n_hit, cost, need) in near_shown.iter() {
                let mut shown: Vec<String> = Vec::new();
                for g in need.iter().take(3) {
                    shown.extend(g.iter().take(max_kw).map(AsRef::<str>::as_ref).map(String::from));
                }
                let n_groups = need.len();
                let more_groups = n_groups.saturating_sub(3);
                let mut line = format!(
                    "    block {bno}: {n_hit} hit, add {cost} keyword(s) from {n_groups} group(s): {}",
                    shown.join(", ")
                );
                if more_groups > 0 {
                    line.push_str(&format!(" ... +{more_groups} more group(s)"));
                }
                println!("{line}");
            }
        }

        if !disqualified.is_empty() {
            let mut u: Vec<&str> = disqualified.iter().map(AsRef::<str>::as_ref).collect();
            u.sort_unstable();
            u.dedup();
            println!(
                "  DISQUALIFIED by excluded term(s) in text (remove them to qualify): {}",
                u.join(", ")
            );
        }
        println!();
    }
    Ok(())
}

/// Total candidate keywords across a near-miss block's need groups.
fn need_total(need: &[Vec<&'static str>]) -> usize {
    need.iter().map(|g| g.len()).sum()
}
