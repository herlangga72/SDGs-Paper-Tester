//! Dev-only A/B: AST tree-walk path vs flattened postfix path, in one
//! process so the CPU clock state is identical for both. Run:
//!   cargo run --release --example ab_path -- <paper.md> [rounds]
use sdg_tools::matcher::{self};
use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let rounds: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(7);

    let paper = Paper::from_text(&std::fs::read_to_string(&path).unwrap());
    let mut queries =
        load_queries(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/data/queries")).unwrap();
    let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
    let mut nslots = 0u32;
    for q in &mut queries {
        matcher::resolve_blocks(&mut q.blocks, &table, &mut nslots);
    }
    let flats: Vec<Vec<matcher::FlatBlock>> = queries
        .iter()
        .map(|q| q.blocks.iter().map(|b| matcher::flatten_block(b, &table)).collect())
        .collect();
    let nblocks: usize = flats.iter().map(|q| q.len()).sum();

    let mut ast_times = Vec::new();
    let mut flat_times = Vec::new();
    let mut ast_matched = 0usize;
    let mut flat_matched = 0usize;

    for r in 0..rounds {
        // AST path
        let mut memo = matcher::Memo::new(&paper, nslots);
        let t0 = Instant::now();
        for q in &queries {
            for b in &q.blocks {
                if matcher::scan_with_fields(b, &paper, &table, &mut memo).3 {
                    ast_matched += 1;
                }
            }
        }
        ast_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        // Flat path (server behavior: scratch output vectors reused across
        // blocks, so no per-block allocations)
        let mut memo = matcher::Memo::new(&paper, nslots);
        let mut hits: Vec<(&str, u8)> = Vec::new();
        let mut misses: Vec<(&str, u8)> = Vec::new();
        let mut ex_hits: Vec<&str> = Vec::new();
        let t0 = Instant::now();
        for q in &flats {
            for f in q {
                hits.clear();
                misses.clear();
                ex_hits.clear();
                if matcher::scan_flat_into(f, &table, &mut memo, &mut hits, &mut misses, &mut ex_hits) {
                    flat_matched += 1;
                }
            }
        }
        flat_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        println!(
            "round {r}: ast={:.2} ms  flat={:.2} ms",
            ast_times[r], flat_times[r]
        );
    }
    ast_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    flat_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "median over {nblocks} blocks: ast={:.3} ms  flat={:.3} ms  (matched {ast_matched}/{flat_matched})",
        ast_times[rounds / 2],
        flat_times[rounds / 2]
    );
}
