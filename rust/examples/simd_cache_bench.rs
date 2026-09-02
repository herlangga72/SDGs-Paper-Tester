//! Dev benchmark: SIMD/cache behavior of the two hottest data transforms.
//!   a) lower_ascii - case folding (copy+fold fused vs old copy-then-fold)
//!   b) TextIndex::build - presence index (single streaming pass vs the old
//!      3-4 separate window/byte passes over the same buffer)
//! Run before and after a change; report both runs.
//!
//! Usage: cargo run --release --example simd_cache_bench

use sdg_tools::matcher::TextIndex;
use sdg_tools::simd::{dispatch_name, lower_ascii};
use std::time::Instant;

fn gen(size: usize, seed: u64) -> Vec<u8> {
    // pseudo-random printable ASCII, plenty of A-Z so folding does work
    let mut x = seed;
    let mut out = Vec::with_capacity(size);
    while out.len() < size {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = (x >> 33) as u8;
        let c = match b % 5 {
            0 => b' ',
            1 => b'a' + (b % 26),
            2 => b'A' + (b % 26),
            3 => b'0' + (b % 10),
            _ => b'z' - (b % 26),
        };
        out.push(c);
    }
    out
}

fn bench_lower(buf: &[u8], iters: usize) -> f64 {
    // warm
    let _ = lower_ascii(buf);
    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        let v = lower_ascii(buf);
        sink = sink.wrapping_add(v.len());
    }
    let secs = t0.elapsed().as_secs_f64() / iters as f64;
    // verify
    let v = lower_ascii(buf);
    assert_eq!(v.len(), buf.len());
    for (a, b) in buf.iter().zip(v.iter()) {
        assert_eq!(*b, a.to_ascii_lowercase());
    }
    let _ = sink;
    secs
}

fn bench_index(buf: &[u8], iters: usize) -> f64 {
    let _ = TextIndex::build(buf); // warm
    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        let idx = TextIndex::build(buf);
        sink = sink.wrapping_add(idx.positions(0).map_or(0, |p| p.len()));
    }
    let secs = t0.elapsed().as_secs_f64() / iters as f64;
    let _ = sink;
    secs
}

fn main() {
    println!("SIMD route: {}", dispatch_name());
    let cases: [(usize, &str, usize); 5] = [
        (2_000, "2 kB   ", 200_000),
        (64_000, "64 kB  ", 20_000),
        (512_000, "512 kB ", 3_000),
        (2_000_000, "2 MB   ", 800),
        (8_000_000, "8 MB   ", 200),
    ];
    println!("{:>8} {:>10} {:>12} {:>12}", "size", "lower_ms", "index_ms", "index GB/s");
    for (size, name, iters) in cases {
        let buf = gen(size, 0x5EED);
        let lo = bench_lower(&buf, iters);
        let idx = bench_index(&buf, iters);
        println!(
            "{} {:>9.3} {:>11.3} {:>11.1}",
            name,
            lo * 1e3,
            idx * 1e3,
            size as f64 / idx / 1e9
        );
    }
    // find full-scan (absent needle) throughput for reference
    let buf = gen(2_000_000, 42);
    let needle = b"zzzz_absent_needle_zzzz";
    let _ = sdg_tools::simd::find(&buf, needle, 0);
    let t0 = Instant::now();
    let mut n = 0usize;
    for _ in 0..50 {
        if sdg_tools::simd::find(&buf, needle, 0).is_some() {
            n += 1;
        }
    }
    let secs = t0.elapsed().as_secs_f64() / 50.0;
    println!("\nfind absent 26B needle over 2 MB: {:.1} GB/s, found={}", buf.len() as f64 / secs / 1e9, n);
}
