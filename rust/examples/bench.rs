//! Benchmark harness (dev-only, not shipped): times the shared-memo matcher
//! against one paper. Run:  cargo run --release --example bench -- <paper.md>
use sdg_tools::matcher::{self};
use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
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
    let n_blocks: usize = flats.iter().map(|q| q.len()).sum();
    for round in 0..3 {
        let mut memo = matcher::Memo::new(&paper, nslots);
        let t0 = Instant::now();
        let mut matched = 0;
        for q in &flats {
            for b in q {
                if matcher::scan_flat(b, &table, &mut memo).3 {
                    matched += 1;
                }
            }
        }
        println!(
            "round {round}: {:.1} ms across {n_blocks} blocks ({matched} matched)",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}
