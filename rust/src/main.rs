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

    let paper = Paper::from_text(&text);
    let title = paper
        .title
        .clone()
        .unwrap_or_else(|| Path::new(&path).file_name().map_or(path.clone(), |n| n.to_string_lossy().into_owned()));
    println!("Paper: {title}\n");

    let mut queries = load_queries(Path::new(&dir))?;
    // Precompile every keyword once into a dense table and resolve each
    // leaf to its pattern index; scanning is then read-only.
    let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
    let mut nslots = 0u32;
    for q in &mut queries {
        matcher::resolve_blocks(&mut q.blocks, &table, &mut nslots);
    }

    // One shared memo for the whole scan: buffers are indexed once and each
    // distinct (pattern, field-mask) is searched once. Without this the CLI
    // rebuilt a TextIndex and re-evaluated every term for every block, which
    // is ~60x slower on large papers.
    let mut memo = matcher::Memo::new(&paper, nslots);

    for q in &queries {
        println!("=== SDG {} ===", q.sdg);
        let mut matched: Vec<(usize, Vec<std::sync::Arc<str>>)> = Vec::new();
        let mut near: Vec<(usize, Vec<std::sync::Arc<str>>, usize, usize)> = Vec::new();
        let mut excluded_hits: Vec<std::sync::Arc<str>> = Vec::new();

        for (bno, block) in q.blocks.iter().enumerate() {
            let (scan, is_match) = matcher::scan_block_shared(block, &paper, &table, &mut memo);
            excluded_hits.extend(scan.excluded_hits.iter().cloned());
            let n_hit = scan.hits.len();
            let n_miss = scan.misses.len();
            if is_match {
                matched.push((bno, scan.hits));
            } else {
                near.push((bno, scan.misses, n_hit, n_miss));
            }
        }
        near.sort_by_key(|x| x.1.len());

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

        if near.is_empty() {
            println!("  NEAR MISSES none");
        } else {
            println!("  NEAR MISSES (missing keywords -> add any of these to qualify):");
            for (bno, misses, n_hit, n_miss) in near.iter().take(top) {
                let shown: Vec<&str> = misses.iter().take(max_kw).map(AsRef::<str>::as_ref).collect();
                let more = n_miss - shown.len();
                println!(
                    "    block {bno}: {n_hit} hit, missing {n_miss} of {}: {}{}",
                    n_hit + n_miss,
                    shown.join(", "),
                    if more > 0 { " ..." } else { "" }
                );
            }
        }

        if !excluded_hits.is_empty() {
            let mut u: Vec<&str> = excluded_hits.iter().map(AsRef::<str>::as_ref).collect();
            u.sort_unstable();
            u.dedup();
            println!("  EXCLUDED terms found in text (can disqualify a match): {}", u.join(", "));
        }
        println!();
    }
    Ok(())
}
