//! Dev probe: per-stage timing of the per-request pipeline.
use sdg_tools::matcher::{self};
use sdg_tools::paper::{F_ANY, Paper};
use sdg_tools::query::load_queries;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
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
    let dicts: Vec<matcher::SdgDict> = queries.iter().map(|q| matcher::collect_sdg_dict(&q.blocks)).collect();
    let paper_text = String::from_utf8_lossy(paper.text_lower(F_ANY));
    let words = matcher::text_words(&paper_text);

    for round in 0..3 {
        // A: scan only
        let mut memo = matcher::Memo::new(&paper, nslots);
        let t0 = Instant::now();
        for q in &flats { for b in q { let _ = matcher::scan_flat(b, &table, &mut memo); } }
        let ta = t0.elapsed().as_secs_f64() * 1000.0;

        // B: scan + present set + cost-only pass (all blocks)
        let mut memo = matcher::Memo::new(&paper, nslots);
        let mut mscr = matcher::MinAddScratch::default();
        let t1 = Instant::now();
        let mut n_near = 0usize;
        for (qi, q) in flats.iter().enumerate() {
            let mut present: HashSet<String> = HashSet::new();
            for (bi, b) in q.iter().enumerate() {
                let (h, _, _, matched) = matcher::scan_flat(b, &table, &mut memo);
                for (kw, _) in h { present.insert(kw.to_string()); }
                if !matched {
                    let (_, cost) = matcher::min_add_flat_cost(b, &table, &mut memo, &mut mscr);
                    if cost != matcher::INF_COST as u32 { n_near += 1; }
                }
            }
            let _ = present;
        }
        let tb = t1.elapsed().as_secs_f64() * 1000.0;

        // C: B + materialize need for top-30
        let mut memo = matcher::Memo::new(&paper, nslots);
        let mut mscr = matcher::MinAddScratch::default();
        let t2 = Instant::now();
        let mut n_groups = 0usize;
        for (qi, q) in flats.iter().enumerate() {
            let mut near: Vec<(usize, usize)> = Vec::new();
            for (bi, b) in q.iter().enumerate() {
                let (_, _, _, matched) = matcher::scan_flat(b, &table, &mut memo);
                if !matched {
                    let (_, cost) = matcher::min_add_flat_cost(b, &table, &mut memo, &mut mscr);
                    if cost != matcher::INF_COST as u32 { near.push((bi, cost as usize)); }
                }
            }
            near.sort_by_key(|x| x.1);
            for (bi, _) in near.into_iter().take(30) {
                let ma = matcher::min_add_flat(&flats[qi][bi], &table, &mut memo, &mut mscr);
                n_groups += ma.need.len();
            }
        }
        let tc = t2.elapsed().as_secs_f64() * 1000.0;

        // D: C + suggestions
        let mut memo = matcher::Memo::new(&paper, nslots);
        let mut mscr = matcher::MinAddScratch::default();
        let t3 = Instant::now();
        let mut n_sug = 0usize;
        for (qi, q) in flats.iter().enumerate() {
            let mut present: HashSet<String> = HashSet::new();
            let mut near: Vec<(usize, usize)> = Vec::new();
            for (bi, b) in q.iter().enumerate() {
                let (h, _, _, matched) = matcher::scan_flat(b, &table, &mut memo);
                for (kw, _) in h { present.insert(kw.to_string()); }
                if !matched {
                    let (_, cost) = matcher::min_add_flat_cost(b, &table, &mut memo, &mut mscr);
                    if cost != matcher::INF_COST as u32 { near.push((bi, cost as usize)); }
                }
            }
            near.sort_by_key(|x| x.1);
            for (bi, _) in near.into_iter().take(30) {
                let ma = matcher::min_add_flat(&flats[qi][bi], &table, &mut memo, &mut mscr);
                let _ = ma;
            }
            let present_ref: HashSet<&str> = present.iter().map(String::as_str).collect();
            n_sug += matcher::suggest_keywords(&words, &dicts[qi], &present_ref, 10).len();
        }
        let td = t3.elapsed().as_secs_f64() * 1000.0;

        println!("round {round}: scan={ta:.2}  B(+present+cost)={tb:.2}  C(+need)={tc:.2}  D(+suggest)={td:.2} ms  near={n_near} groups={n_groups} sugg={n_sug}");
    }
}
