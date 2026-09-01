//! Compare AST min_add vs flat min_add (cost + need) per SDG07 block.
use sdg_tools::matcher::{self};
use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::path::Path;

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

    for (bi, f) in flats.iter().enumerate() {
        let mut memo = matcher::Memo::new(&paper, nslots);
        let _ = matcher::scan_flat(f, &table, &mut memo);
        let ast = matcher::min_add_block(&q7.blocks[bi], &table, &mut memo);
        let mut memo = matcher::Memo::new(&paper, nslots);
        let _ = matcher::scan_flat(f, &table, &mut memo);
        let mut scr = matcher::MinAddScratch::default();
        let fl = matcher::min_add_flat(f, &table, &mut memo, &mut scr);
        let ast_set: std::collections::HashSet<&str> =
            ast.need.iter().flatten().map(|k| k.as_ref()).collect();
        let fl_set: std::collections::HashSet<&str> =
            fl.need.iter().flatten().map(|k| k.as_ref()).collect();
        let only_ast: Vec<&str> = ast_set.difference(&fl_set).take(5).map(|s| *s).collect();
        let only_flat: Vec<&str> = fl_set.difference(&ast_set).take(5).map(|s| *s).collect();
        println!(
            "block {bi}: ast cost={} kw={} groups={} | flat cost={} kw={} groups={} | only_ast={only_ast:?} only_flat={only_flat:?}",
            ast.cost, ast_set.len(), ast.need.len(), fl.cost, fl_set.len(), fl.need.len()
        );
    }
}
