//! Dev benchmark: /api/keywords "present keywords" phase - old (per-block
//! scan_flat_into with report vectors) vs new (boot-time present table +
//! memoized leaf_hit). Also verifies both produce identical keyword sets.
//!
//! Usage: cargo run --release --example kw_present_bench -- <paper.md> [iters]

use sdg_tools::matcher::{self, FlatBlock, LeafDesc, Pattern};
use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

fn load() -> (Vec<sdg_tools::query::Query>, &'static [Pattern], Vec<Vec<FlatBlock>>, Vec<matcher::SdgDict>) {
    let qdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/data/queries");
    let mut queries = load_queries(&qdir).unwrap();
    let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
    let mut nslots = 0u32;
    for q in &mut queries {
        matcher::resolve_blocks(&mut q.blocks, &table, &mut nslots);
    }
    let flats: Vec<Vec<FlatBlock>> = queries
        .iter()
        .map(|q| q.blocks.iter().map(|b| matcher::flatten_block(b, &table)).collect())
        .collect();
    let dicts: Vec<matcher::SdgDict> =
        queries.iter().map(|q| matcher::collect_sdg_dict(&q.blocks)).collect();
    let table: &'static [Pattern] = Box::leak(table.into_boxed_slice());
    (queries, table, flats, dicts)
}

/// Old /api/keywords algorithm: one full scan_flat_into per block, keeping
/// the hit keywords of the SDG. Takes 'static block slices exactly like the
/// web server's boot cache does.
fn present_old(
    flats: &'static [FlatBlock],
    table: &'static [Pattern],
    paper: &Paper,
) -> (HashSet<&'static str>, u64) {
    let t0 = Instant::now();
    let mut memo = matcher::Memo::new(paper, 0);
    let mut present: HashSet<&'static str> = HashSet::new();
    let mut hits: Vec<(&'static str, u8)> = Vec::new();
    let mut misses: Vec<(&'static str, u8)> = Vec::new();
    let mut ex: Vec<&'static str> = Vec::new();
    for flat in flats {
        hits.clear();
        misses.clear();
        ex.clear();
        matcher::scan_flat_into(flat, table, &mut memo, &mut hits, &mut misses, &mut ex);
        for (kw, _) in hits.drain(..) {
            present.insert(kw);
        }
    }
    (present, t0.elapsed().as_nanos() as u64)
}

/// Present table exactly as web.rs build_present_tables.
fn build_present(flats: &[FlatBlock]) -> Vec<LeafDesc> {
    let mut seen: HashSet<(u32, u8)> = HashSet::new();
    let mut out: Vec<LeafDesc> = Vec::new();
    for flat in flats {
        for l in flat.leaves {
            if l.excluded {
                continue;
            }
            if seen.insert((l.pid, l.mask)) {
                out.push(l.clone());
            }
        }
    }
    out
}

/// New /api/keywords algorithm: memoized leaf_hit over the present table.
fn present_new(
    present: &[LeafDesc],
    table: &'static [Pattern],
    paper: &Paper,
) -> (HashSet<&'static str>, u64) {
    let t0 = Instant::now();
    let mut memo = matcher::Memo::new(paper, 0);
    let mut out: HashSet<&'static str> = HashSet::new();
    for l in present {
        if memo.leaf_hit(&table[l.pid as usize], l.pid, l.mask, l.slot) {
            out.insert(table[l.pid as usize].raw());
        }
    }
    (out, t0.elapsed().as_nanos() as u64)
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let iters: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(200);
    let raw = std::fs::read_to_string(&path).unwrap();
    let paper = Paper::from_text(&raw);
    let (queries, table, flats, dicts) = load();

    // snapshot counts, build present tables, then leak each SDG's block
    // list to 'static slices exactly like the web boot cache provides
    let blk_counts: Vec<usize> = flats.iter().map(|f| f.len()).collect();
    let occ_counts: Vec<usize> =
        flats.iter().map(|f| f.iter().map(|b| b.leaves.len()).sum::<usize>()).collect();
    let present_tables: Vec<Vec<LeafDesc>> = flats.iter().map(|f| build_present(f)).collect();
    let total_leaves: usize = present_tables.iter().map(|p| p.len()).sum();
    let flats_st: Vec<&'static [FlatBlock]> = flats
        .into_iter()
        .map(|v| {
            let leak: &'static mut [FlatBlock] = Box::leak(v.into_boxed_slice());
            let shrink: &'static [FlatBlock] = leak;
            shrink
        })
        .collect();
    println!(
        "paper={path} text={}B iters={iters} | present leaves total={total_leaves}",
        paper.full_text().len()
    );

    // correctness: identical sets for every SDG
    for (qi, q) in queries.iter().enumerate() {
        let (old, _) = present_old(flats_st[qi], table, &paper);
        let (new, _) = present_new(&present_tables[qi], table, &paper);
        assert_eq!(
            old.len(),
            new.len(),
            "SDG {}: present count mismatch (old {} vs new {})",
            q.sdg,
            old.len(),
            new.len()
        );
        for k in &old {
            assert!(new.contains(k), "SDG {}: {} only in old", q.sdg, k);
        }
        for k in &new {
            assert!(old.contains(k), "SDG {}: {} only in new", q.sdg, k);
        }
    }
    println!("equivalence OK for all 17 SDGs\n");

    // scoring phase (identical in both): measure for reference
    let t0 = Instant::now();
    let text = String::from_utf8_lossy(paper.text_lower(sdg_tools::paper::F_ANY));
    let words = matcher::text_words(&text);
    let words_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut score_ms = 0.0f64;
    for (qi, d) in dicts.iter().enumerate() {
        let (old, _) = present_old(flats_st[qi], table, &paper);
        for _ in 0..10 {
            let t1 = Instant::now();
            let o: HashSet<&str, matcher::FastHasher> = old.iter().copied().collect();
            let s = matcher::score_keywords(&words, d, &o, 300);
            score_ms += t1.elapsed().as_secs_f64() * 1000.0 / 10.0;
            assert!(!s.is_empty() || d.is_empty());
        }
    }

    println!(
        "{:>4} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}",
        "sdg", "blks", "occ", "uniq", "old_us", "new_us", "speedup"
    );
    let mut tot_old = 0.0f64;
    let mut tot_new = 0.0f64;
    for (qi, q) in queries.iter().enumerate() {
        let mut old_ns = 0u64;
        let mut new_ns = 0u64;
        // warm-up each path (memo/index caches are per-call, so this is
        // just branch/allocator warm-up)
        present_old(flats_st[qi], table, &paper);
        present_new(&present_tables[qi], table, &paper);
        for _ in 0..iters {
            let (_, t) = present_old(flats_st[qi], table, &paper);
            old_ns += t;
            let (_, t) = present_new(&present_tables[qi], table, &paper);
            new_ns += t;
        }
        let old_us = old_ns as f64 / iters as f64 / 1e3;
        let new_us = new_ns as f64 / iters as f64 / 1e3;
        tot_old += old_us;
        tot_new += new_us;
        println!(
            "{:>4} {:>7} {:>7} {:>7} {:>8.1} {:>8.1} {:>7.2}x",
            q.sdg,
            blk_counts[qi],
            occ_counts[qi],
            present_tables[qi].len(),
            old_us,
            new_us,
            old_us / new_us.max(1e-6)
        );
    }
    println!(
        "\nTOTAL present phase (all 17 SDGs, 1 request): old {:.0} us vs new {:.0} us ({:.2}x)\n\
         text_words {:.2} us/req, score_keywords all-17-SDGs {:.2} us/req",
        tot_old,
        tot_new,
        tot_old / tot_new.max(1e-6),
        words_ms * 1000.0,
        score_ms * 1000.0
    );
}

