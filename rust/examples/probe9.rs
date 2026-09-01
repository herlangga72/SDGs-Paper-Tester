use sdg_tools::ast::Node;
use sdg_tools::cache;
use sdg_tools::matcher::{self};
use sdg_tools::paper::Paper;
use std::path::Path;

fn main() {
    let qdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/data/queries");
    let data = sdg_tools::cache::read_cached(&qdir).expect("cache");
    let paper = Paper::from_text(&std::fs::read_to_string("/home/server/sdgs/SDGs-Paper-Tester/papers/real_justice_2020.md").unwrap());
    matcher::rebuild_first_quads(data.patterns);
    let q1 = data.queries.iter().find(|q| q.sdg == "01").unwrap();
    let mut memo = matcher::Memo::new(&paper, 0);
    let (scan, matched) = matcher::scan_block_shared(&q1.blocks[3], &paper, data.patterns, &mut memo);
    println!("scan: matched={matched} hits={}", scan.hits.len());
    // walk leaves: node keyword, pid, pattern raw, term verdict (fresh eval)
    let mut leaves: Vec<(String, u32, String, bool)> = Vec::new();
    fn walk(n: &Node, table: &[matcher::Pattern], paper: &Paper, out: &mut Vec<(String, u32, String, bool)>) {
        match n {
            Node::Leaf { keyword, pid, mask, slot, .. } => {
                let leaf = Node::Leaf { keyword: keyword.clone(), exact: false, pid: *pid, mask: *mask, slot: *slot };
                let v = matcher::eval(&leaf, None, paper, table);
                out.push((keyword.clone(), *pid, table[*pid as usize].raw().to_string(), v));
            }
            Node::Field { child, .. } => walk(child, table, paper, out),
            Node::Not { child } => walk(child, table, paper, out),
            Node::Group { children, .. } => {
                for c in children {
                    walk(c, table, paper, out);
                }
            }
        }
    }
    walk(&q1.blocks[3], data.patterns, &paper, &mut leaves);
    println!("leaves: {}", leaves.len());
    let mism = leaves.iter().filter(|(k, _, p, _)| k != p).count();
    println!("node_kw != pattern_raw: {mism}");
    let acc_hit = leaves.iter().filter(|(k, _, _, v)| k == "access" && *v).count();
    println!("walk access hit leaves: {acc_hit}");
    // pid uniqueness
    let mut pids: std::collections::HashMap<u32, Vec<&str>> = std::collections::HashMap::new();
    for (k, pid, _, _) in &leaves {
        pids.entry(*pid).or_default().push(k);
    }
    let dup = pids.iter().filter(|(_, v)| v.len() > 1 && v.iter().any(|a| *a != v[0])).count();
    println!("pids with different keywords: {dup}");
    for (pid, v) in pids.iter().filter(|(_, v)| v.len() > 1 && v.iter().any(|a| *a != v[0])).take(5) {
        println!("  pid {pid}: {v:?}");
    }
}
