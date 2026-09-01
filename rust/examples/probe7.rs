//! Time min_add_flat per SDG07 block.
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
    let q7 = queries.iter().find(|q| q.sdg == "07").unwrap();
    let flats: Vec<matcher::FlatBlock> =
        q7.blocks.iter().map(|b| matcher::flatten_block(b, &table)).collect();
    let mut memo = matcher::Memo::new(&paper, nslots);
    let mut scr = matcher::MinAddScratch::default();

    for (bi, f) in flats.iter().enumerate() {
        let _ = matcher::scan_flat(f, &table, &mut memo);
        let (_, cost) = matcher::min_add_flat_cost(f, &table, &mut memo, &mut scr);
        use std::sync::atomic::Ordering;
        matcher::DBG_OPS.store(0, Ordering::Relaxed);
        matcher::DBG_UNION_KW.store(0, Ordering::Relaxed);
        matcher::DBG_UNIONS.store(0, Ordering::Relaxed);
        matcher::DBG_AND_GROUPS.store(0, Ordering::Relaxed);
        matcher::DBG_FP_HITS.store(0, Ordering::Relaxed);
        let t = Instant::now();
        let ma = matcher::min_add_flat(f, &table, &mut memo, &mut scr);
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        let nkw: usize = ma.need.iter().map(|g| g.len()).sum();
        println!("SDG07 block {bi}: prog={} cost={} need={:.2}ms groups={} kw={nkw} ops={} unions={} union_kw={} and_groups={} fp_hits={}",
                 f.prog.len(), cost, dt, ma.need.len(),
                 matcher::DBG_OPS.load(Ordering::Relaxed),
                 matcher::DBG_UNIONS.load(Ordering::Relaxed),
                 matcher::DBG_UNION_KW.load(Ordering::Relaxed),
                 matcher::DBG_AND_GROUPS.load(Ordering::Relaxed),
                 matcher::DBG_FP_HITS.load(Ordering::Relaxed));
    }
}
