//! Hot-path benchmark (dev-only): queries are loaded and compiled ONCE, then
//! we time repeated matching of a paper, reusing a fresh shared memo per
//! request (exactly what the web server does per request). Run:
//!   cargo run --release --example hotbench -- <paper.md> [iterations]
use sdg_tools::matcher::{self};
use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let iters: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(2000);

    let raw = std::fs::read_to_string(&path).unwrap();
    let paper = Paper::from_text(&raw);

    // --- one-time warm-up of the SIMD dispatch and query compile ---
    let _ = sdg_tools::simd::dispatch_name();

    let mut queries =
        load_queries(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/data/queries")).unwrap();
    let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
    let mut nslots = 0u32;
    for q in &mut queries {
        matcher::resolve_blocks(&mut q.blocks, &table, &mut nslots);
    }
    // Flatten every block once (the web server does this at boot).
    let flats: Vec<Vec<matcher::FlatBlock>> = queries
        .iter()
        .map(|q| q.blocks.iter().map(|b| matcher::flatten_block(b, &table)).collect())
        .collect();

    // warm the CPU + caches, verify correctness once
    let mut warm = matcher::Memo::new(&paper, nslots);
    let mut wmatched = 0;
    for q in &flats {
        for f in q {
            if matcher::scan_flat(f, &table, &mut warm).3 {
                wmatched += 1;
            }
        }
    }

    // steady-state loop: fresh memo per request (server behavior), reusing
    // the loaded+compiled queries and the already-parsed paper index, with
    // scratch output vectors reused across blocks (as web.rs does).
    let t0 = Instant::now();
    let mut matched = 0usize;
    for _ in 0..iters {
        let mut memo = matcher::Memo::new(&paper, nslots);
        let mut hits: Vec<(&str, u8)> = Vec::new();
        let mut misses: Vec<(&str, u8)> = Vec::new();
        let mut ex_hits: Vec<&str> = Vec::new();
        for q in &flats {
            for f in q {
                hits.clear();
                misses.clear();
                ex_hits.clear();
                if matcher::scan_flat_into(f, &table, &mut memo, &mut hits, &mut misses, &mut ex_hits) {
                    matched += 1;
                }
            }
        }
    }
    let dt = t0.elapsed();
    let per = dt.as_secs_f64() / iters as f64;
    println!(
        "hot: {} iters over {} blocks, {} matched, avg {:.3} ms/request ({:.1} req/s)",
        iters,
        queries.iter().map(|q| q.blocks.len()).sum::<usize>(),
        matched,
        per * 1e3,
        iters as f64 / dt.as_secs_f64()
    );
}
