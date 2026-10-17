//! Stage-attribution + parallel-scaling benchmark (dev-only, not shipped):
//! replicates the web server's `match_report` pipeline using only library
//! entry points. Reports per-stage timings, per-SDG cost distribution (the
//! parallel ceiling) and serial-vs-parallel wall time.
//!
//! Run: cargo run --release --example report_bench -- <paper.md> [iters] [par]

use sdg_tools::matcher::{self, FlatBlock, MinAddScratch, SdgDict};
use sdg_tools::paper::Paper;
use sdg_tools::query::load_queries;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

struct SdgOut {
    matched: usize,
    near_total: usize,
    suggestions: usize,
    need_calls: u64,
    need_ops: u64,
    badge_ops: u64,
}

/// One SDG's full pipeline with its own memo/scratch (what a worker runs).
#[allow(clippy::too_many_arguments)]
fn run_sdg(
    qi: usize,
    paper: &Paper,
    table: &[matcher::Pattern],
    flats: &'static [Vec<FlatBlock>],
    dicts: &[SdgDict],
    sdg_pid: &[HashMap<u32, Vec<u32>, matcher::FastHasher>],
    kw_pid: &HashMap<&str, u32, matcher::FastHasher>,
    words: &matcher::PaperWords,
    top: usize,
    memo: &matcher::Memo,
    mscr: &mut MinAddScratch,
    t_scan: &mut f64,
    t_cost: &mut f64,
    t_sug: &mut f64,
    t_need: &mut f64,
) -> SdgOut {
    let mut hits: Vec<(&'static str, u8)> = Vec::new();
    let mut misses: Vec<(&'static str, u8)> = Vec::new();
    let mut ex_hits: Vec<&'static str> = Vec::new();

    let mut out = SdgOut { matched: 0, near_total: 0, suggestions: 0, need_calls: 0, need_ops: 0, badge_ops: 0 };

    // scan
    let t0 = Instant::now();
    let mut nonmatch: Vec<(usize, Vec<(&'static str, u8)>)> = Vec::new();
    let mut present: HashSet<&'static str, matcher::FastHasher> =
        HashSet::with_hasher(matcher::FastHasher::default());
    for (bno, flat) in flats[qi].iter().enumerate() {
        hits.clear();
        misses.clear();
        ex_hits.clear();
        let is_match = matcher::scan_flat_into(flat, table, memo, &mut hits, &mut misses, &mut ex_hits);
        for (kw, _) in hits.iter() {
            present.insert(*kw);
        }
        let h = std::mem::take(&mut hits);
        if is_match {
            out.matched += 1;
        } else {
            nonmatch.push((bno, h));
        }
    }
    *t_scan += t0.elapsed().as_secs_f64();

    // cost-only min-add
    let t1 = Instant::now();
    let mut costs = vec![matcher::INF_COST as u32; flats[qi].len()];
    let mut near: Vec<(usize, u32, Vec<(&'static str, u8)>)> = Vec::new();
    for (bno, h) in nonmatch {
        let (_, cost) = matcher::min_add_flat_cost(&flats[qi][bno], table, memo, mscr);
        if cost != matcher::INF_COST as u32 {
            costs[bno] = cost;
            near.push((bno, cost, h));
        }
    }
    *t_cost += t1.elapsed().as_secs_f64();
    out.near_total = near.len();

    // suggestions
    let t2 = Instant::now();
    let sug = matcher::suggest_keywords(words, &dicts[qi], &present, 10);
    *t_sug += t2.elapsed().as_secs_f64();
    out.suggestions = sug.len();

    // full min-add for displayed + badge-candidate blocks
    let t3 = Instant::now();
    let mut ns: Vec<&(usize, u32, Vec<(&'static str, u8)>)> = near.iter().collect();
    ns.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.2.len().cmp(&a.2.len())));
    let mut computed: HashSet<u32, matcher::FastHasher> =
        HashSet::with_hasher(matcher::FastHasher::default());
    for (bno, _, _) in ns.iter().take(top) {
        out.need_calls += 1;
        out.need_ops += flats[qi][*bno].prog.len() as u64;
        std::hint::black_box(matcher::min_add_flat(&flats[qi][*bno], table, memo, mscr));
        computed.insert(*bno as u32);
    }
    for s in &sug {
        let Some(&pid) = kw_pid.get(s.keyword.as_ref()) else { continue };
        let Some(blocks) = sdg_pid[qi].get(&pid) else { continue };
        for &bno in blocks {
            if costs[bno as usize] == matcher::INF_COST as u32 || computed.contains(&bno) {
                continue;
            }
            std::hint::black_box(matcher::min_add_flat(
                &flats[qi][bno as usize],
                table,
                memo,
                mscr,
            ));
            out.badge_ops += flats[qi][bno as usize].prog.len() as u64;
            computed.insert(bno);
        }
    }
    *t_need += t3.elapsed().as_secs_f64();
    out
}

struct Setup {
    flats: &'static [Vec<FlatBlock>],
    dicts: Vec<SdgDict>,
    sdg_pid: Vec<HashMap<u32, Vec<u32>, matcher::FastHasher>>,
    kw_pid: HashMap<&'static str, u32, matcher::FastHasher>,
    table: Vec<matcher::Pattern>,
    nslots: u32,
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let iters: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(300);
    let par = std::env::args().nth(3).map(|s| s == "par").unwrap_or(false);
    let top = 30usize;

    let paper = Paper::from_text(&std::fs::read_to_string(&path).unwrap());
    let mut queries =
        load_queries(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/data/queries")).unwrap();
    let table = matcher::compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
    let mut nslots = 0u32;
    for q in &mut queries {
        matcher::resolve_blocks(&mut q.blocks, &table, &mut nslots);
    }
    let flats: &'static [Vec<FlatBlock>] = Box::leak(
        queries
            .iter()
            .map(|q| q.blocks.iter().map(|b| matcher::flatten_block(b, &table)).collect())
            .collect::<Vec<Vec<FlatBlock>>>()
            .into_boxed_slice(),
    );
    let dicts: Vec<SdgDict> = queries.iter().map(|q| matcher::collect_sdg_dict(&q.blocks)).collect();
    let sdg_pid = invert_pid_blocks(flats);
    let mut kw_pid: HashMap<&'static str, u32, matcher::FastHasher> =
        HashMap::with_hasher(matcher::FastHasher::default());
    for (i, p) in table.iter().enumerate() {
        kw_pid.insert(p.raw(), i as u32);
    }
    let setup = Setup { flats, dicts, sdg_pid, kw_pid, table, nslots };

    let n = flats.len();
    let threads = std::env::var("RB_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1))
        .min(n);
    println!(
        "paper={path} sdgs={n} patterns={} text={}B iters={iters} par={par} threads={threads}",
        setup.table.len(),
        paper.full_text().len()
    );

    let (mut t_scan, mut t_cost, mut t_sug, mut t_need) = (0f64, 0f64, 0f64, 0f64);
    let paper_text = String::from_utf8_lossy(paper.text_lower(sdg_tools::paper::F_ANY));
    let words = matcher::text_words(&paper_text);
    let mut per_sdg = vec![0f64; n];
    let wall = Instant::now();
    let (mut need_calls, mut need_ops, mut badge_ops) = (0u64, 0u64, 0u64);
    for _ in 0..iters {
        if par && threads > 1 {
            let next = AtomicUsize::new(0);
            let slots: Vec<Mutex<Option<(SdgOut, [f64; 4], f64)>>> =
                (0..n).map(|_| Mutex::new(None)).collect();
            let base_memo = matcher::Memo::new(&paper, setup.nslots);
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let memo = base_memo.fork();
                        let mut mscr = MinAddScratch::default();
                        loop {
                            let qi = next.fetch_add(1, Ordering::Relaxed);
                            if qi >= n {
                                break;
                            }
                            let (mut a, mut b, mut c, mut d) = (0f64, 0f64, 0f64, 0f64);
                            let ti = Instant::now();
                            let o = run_sdg(
                                qi, &paper, &setup.table, setup.flats, &setup.dicts, &setup.sdg_pid,
                                &setup.kw_pid, &words, top, &memo, &mut mscr, &mut a, &mut b, &mut c,
                                &mut d,
                            );
                            let dt = ti.elapsed().as_secs_f64();
                            *slots[qi].lock().unwrap() = Some((o, [a, b, c, d], dt));
                        }
                    });
                }
            });
            let _ = t0;
            for (qi, m) in slots.into_iter().enumerate() {
                let (o, [a, b, c, d], dt) = m.into_inner().unwrap().unwrap();
                t_scan += a;
                t_cost += b;
                t_sug += c;
                t_need += d;
                per_sdg[qi] += dt;
                need_calls += o.need_calls;
                need_ops += o.need_ops;
                badge_ops += o.badge_ops;
                std::hint::black_box((o.matched, o.near_total, o.suggestions));
            }
        } else {
            let memo = matcher::Memo::new(&paper, setup.nslots);
            let mut mscr = MinAddScratch::default();
            for qi in 0..n {
                let ti = Instant::now();
                let o = run_sdg(
                    qi, &paper, &setup.table, setup.flats, &setup.dicts, &setup.sdg_pid, &setup.kw_pid,
                    &words, top, &memo, &mut mscr, &mut t_scan, &mut t_cost, &mut t_sug, &mut t_need,
                );
                per_sdg[qi] += ti.elapsed().as_secs_f64();
                need_calls += o.need_calls;
                need_ops += o.need_ops;
                badge_ops += o.badge_ops;
            }
        }
    }
    let wall = wall.elapsed().as_secs_f64();

    let ms = |t: f64| t / iters as f64 * 1000.0;
    println!("scan        {:>8.3} ms/req", ms(t_scan));
    println!("minadd_cost {:>8.3} ms/req", ms(t_cost));
    println!("suggest     {:>8.3} ms/req", ms(t_sug));
    println!("minadd_need {:>8.3} ms/req", ms(t_need));
    println!("sum stages  {:>8.3} ms/req", ms(t_scan + t_cost + t_sug + t_need));
    println!("WALL        {:>8.3} ms/req", wall / iters as f64 * 1000.0);
    println!(
        "need_calls={:.1} need_ops={:.0} badge_ops={:.0}",
        need_calls as f64 / iters as f64,
        need_ops as f64 / iters as f64,
        badge_ops as f64 / iters as f64
    );
    let mut ps: Vec<(usize, f64)> = per_sdg.iter().map(|v| v / iters as f64 * 1000.0).enumerate().collect();
    ps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let tot: f64 = ps.iter().map(|x| x.1).sum();
    println!("per-SDG ms (top 6): {:?}", &ps[..6.min(ps.len())]);
    println!("heaviest SDG shares: {:.1}% {:.1}% {:.1}% {:.1}%", 100.0 * ps[0].1 / tot, 100.0 * ps[1].1 / tot, 100.0 * ps[2].1 / tot, 100.0 * ps[3].1 / tot);
    // LPT lower bound for `threads` workers.
    let mut load = vec![0f64; threads];
    for (_, v) in &ps {
        let i = (0..threads).min_by(|&a, &b| load[a].partial_cmp(&load[b]).unwrap()).unwrap();
        load[i] += v;
    }
    println!("ideal makespan (LPT, {} threads): {:.3} ms", threads, load.iter().cloned().fold(0f64, f64::max));
}

fn invert_pid_blocks(flats: &[Vec<FlatBlock>]) -> Vec<HashMap<u32, Vec<u32>, matcher::FastHasher>> {
    flats
        .iter()
        .map(|sdg| {
            let mut m: HashMap<u32, Vec<u32>, matcher::FastHasher> =
                HashMap::with_hasher(matcher::FastHasher::default());
            for (bno, flat) in sdg.iter().enumerate() {
                for l in flat.leaves {
                    if l.excluded {
                        continue;
                    }
                    let v = m.entry(l.pid).or_default();
                    if v.last() != Some(&(bno as u32)) {
                        v.push(bno as u32);
                    }
                }
            }
            m
        })
        .collect()
}
