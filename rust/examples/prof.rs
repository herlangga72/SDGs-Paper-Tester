//! Dev-only profiling harness (feature `prof`). Prints per-request counters
//! and wall time, with experiment toggles:
//!   PROF_SKIP_FILTER=1  -> skip the TextIndex pre-filter (SIMD runs more)
//!   PROF_SKIP_FIND=1    -> `matches` always true (filter + traversal only)
//! Run:  cargo run --release --features prof --example prof -- <paper.md> [iters]
#[cfg(feature = "prof")]
use sdg_tools::matcher::{self, prof};
#[cfg(feature = "prof")]
use sdg_tools::paper::Paper;
#[cfg(feature = "prof")]
use sdg_tools::query::load_queries;
#[cfg(feature = "prof")]
use std::path::Path;
#[cfg(feature = "prof")]
use std::time::Instant;

fn main() {
    #[cfg(not(feature = "prof"))]
    {
        eprintln!("prof harness: build with `--features prof`");
        return;
    }
    #[cfg(feature = "prof")]
    run();
}

#[cfg(feature = "prof")]
fn run() {
    let path = std::env::args().nth(1).unwrap();
    let iters: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(1000);

    let raw = std::fs::read_to_string(&path).unwrap();
    let paper = Paper::from_text(&raw);

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
    println!(
        "paper={path} blocks={nblocks} patterns={} slots={nslots} text={}B iters={iters}",
        table.len(),
        paper.full_text().len()
    );

    // warm-up + correctness sanity
    let mut warm = matcher::Memo::new(&paper, nslots);
    let mut wmatched = 0;
    for q in &flats {
        for f in q {
            if matcher::scan_flat(f, &table, &mut warm).3 {
                wmatched += 1;
            }
        }
    }
    println!("warm: {wmatched} blocks matched (of {nblocks})");

    prof::reset();
    let t0 = Instant::now();
    let mut matched = 0usize;
    for _ in 0..iters {
        let mut memo = matcher::Memo::new(&paper, nslots);
        for q in &flats {
            for f in q {
                if matcher::scan_flat(f, &table, &mut memo).3 {
                    matched += 1;
                }
            }
        }
    }
    let dt = t0.elapsed();
    println!("total: {:.3} ms/request ({:.1} req/s), matched={}", dt.as_secs_f64() / iters as f64 * 1000.0, iters as f64 / dt.as_secs_f64(), matched);
    println!(
        "index_builds={} index_bytes={} could_calls={} could_part_bytes={} matches_calls={}",
        prof::INDEX_BUILDS.load(std::sync::atomic::Ordering::Relaxed) / iters as u64,
        prof::INDEX_BYTES.load(std::sync::atomic::Ordering::Relaxed) / iters as u64,
        prof::COULD_CALLS.load(std::sync::atomic::Ordering::Relaxed) / iters as u64,
        prof::COULD_PARTS.load(std::sync::atomic::Ordering::Relaxed) / iters as u64,
        prof::MATCHES_CALLS.load(std::sync::atomic::Ordering::Relaxed) / iters as u64
    );
    println!(
        "term_computes={} term_cache_hits={} leaf_evals={} report_pushes={}",
        prof::TERM_COMPUTES.load(std::sync::atomic::Ordering::Relaxed) / iters as u64,
        prof::TERM_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed) / iters as u64,
        prof::LEAF_EVALS.load(std::sync::atomic::Ordering::Relaxed) / iters as u64,
        prof::REPORT_PUSHES.load(std::sync::atomic::Ordering::Relaxed) / iters as u64
    );
}
