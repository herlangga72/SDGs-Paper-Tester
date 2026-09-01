//! Per-SDG need-materialization timing.
use sdg_tools::matcher::{self};
use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::path::Path;
use std::time::Instant;

fn main() {
    let raw = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
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

    let mut memo = matcher::Memo::new(&paper, nslots);
    let mut mscr = matcher::MinAddScratch::default();
    let mut total = 0.0;
    for (qi, q) in flats.iter().enumerate() {
        let mut near: Vec<(usize, usize)> = Vec::new();
        for (bi, b) in q.iter().enumerate() {
            let (_, _, _, matched) = matcher::scan_flat(b, &table, &mut memo);
            if !matched {
                let (_, cost) = matcher::min_add_flat_cost(b, &table, &mut memo, &mut mscr);
                if cost != matcher::INF_COST as u32 {
                    near.push((bi, cost as usize));
                }
            }
        }
        near.sort_by_key(|x| x.1);
        let t = Instant::now();
        let mut groups = 0usize;
        for (bi, _) in near.into_iter().take(30) {
            let ma = matcher::min_add_flat(&flats[qi][bi], &table, &mut memo, &mut mscr);
            groups += ma.need.len();
        }
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        total += dt;
        if dt > 0.1 {
            println!("SDG {:>2}: materialize 30 = {dt:6.2} ms (groups={groups})", queries[qi].sdg);
        }
    }
    println!("total materialize: {total:.2} ms");
}
