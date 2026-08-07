use sdg_tools::matcher;
use sdg_tools::query::load_queries;
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn collect(n: &sdg_tools::ast::Node, pairs: &mut HashSet<(u32, u8)>, field: u8, distinct_masks: &mut HashSet<u8>) {
    match n {
        sdg_tools::ast::Node::Leaf { pid, .. } => {
            let m = if field == 0 { 0x0f } else { field }; // default TITLE-ABS-KEY
            pairs.insert((*pid, m));
            distinct_masks.insert(m);
        }
        sdg_tools::ast::Node::Not { child } => collect(child, pairs, field, distinct_masks),
        sdg_tools::ast::Node::Field { fields, child } => {
            let mut m = 0u8;
            for f in fields {
                for c in f.bytes() {
                    m |= 1 << ((c - b'0') & 7);
                }
            }
            collect(child, pairs, m, distinct_masks);
        }
        sdg_tools::ast::Node::Group { children, .. } => {
            for c in children {
                collect(c, pairs, field, distinct_masks)
            }
        }
    }
}

fn main() {
    let mut q =
        load_queries(Path::new("/home/server/Downloads/sdg-paper-matcher/engine/data/queries")).unwrap();
    let t = matcher::compile_all(q.iter().flat_map(|x| x.blocks.iter()));
    for x in &mut q {
        matcher::resolve_blocks(&mut x.blocks, &t);
    }
    let mut pairs = HashSet::new();
    let mut masks = HashSet::new();
    for qs in &q {
        for b in &qs.blocks {
            collect(b, &mut pairs, 0, &mut masks);
        }
    }
    let mut ms: Vec<u8> = masks.into_iter().collect();
    ms.sort_unstable();
    println!("table={} patterns, distinct (pid,mask) pairs={}", t.len(), pairs.len());
    println!("distinct masks ({}): {:?}", ms.len(), ms);
}
