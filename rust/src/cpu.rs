//! Boot-time CPU cache specifications.
//!
//! Detected exactly once per process and cached (same pattern as
//! `simd::best_level`): cache-line size comes from CPUID leaf 1
//! (CLFLUSH line size, universal on x86_64); L1d/L2/L3 sizes come from the
//! AMD extended leaves (also populated by most Intel CPUs) with conservative
//! fallbacks and clamps, so wrong vendor reporting can never produce absurd
//! tuning values. Everything is advisory: correctness never depends on it.
//!
//! Consumers:
//!   - matcher: dynamic chunk size for the parallel TextIndex build
//!     (chunk + its segment-local Bloom must fit the detected L2), worker
//!     count capped at `cores`, chunk boundaries rounded to the cache line
//!   - future: aligned allocations for hot per-request buffers

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct CpuSpec {
    /// Cache line (CLFLUSH granularity), bytes. 64 on all mainstream x86_64.
    pub cache_line: usize,
    /// L1 data cache, bytes (best effort).
    pub l1d: usize,
    /// L2 cache, bytes (best effort; the value the TextIndex chunking uses).
    pub l2: usize,
    /// L3 / last-level cache, bytes (best effort).
    pub l3: usize,
    /// Logical cores available to this process (std::thread
    /// available_parallelism). 0 means unknown.
    pub cores: usize,
}

impl CpuSpec {
    pub fn log_summary(&self) -> String {
        format!(
            "cache line {}B, L1d {}KiB, L2 {}KiB, L3 {}MiB, {} cores",
            self.cache_line,
            self.l1d / 1024,
            self.l2 / 1024,
            self.l3 / (1024 * 1024),
            self.cores
        )
    }
}

static SPEC: OnceLock<CpuSpec> = OnceLock::new();

/// The detected CPU cache spec (computed once; repeated calls are a load).
pub fn best() -> &'static CpuSpec {
    SPEC.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    // EBX cannot be an asm operand (LLVM keeps it reserved under PIC), so
    // the result is routed through a scratch register with push-less
    // save/restore:  mov scratch, rbx ; cpuid ; xchg scratch, rbx
    let mut a = leaf;
    let mut c = subleaf;
    let mut b: u32;
    let mut d: u32;
    std::arch::asm!(
        "mov {b:r}, rbx",
        "cpuid",
        "xchg {b:r}, rbx",
        b = out(reg) b,
        inout("eax") a,
        inout("ecx") c,
        out("edx") d,
        options(nostack)
    );
    (a, b, c, d)
}

fn clamp(v: usize, lo: usize, hi: usize) -> usize {
    v.clamp(lo, hi)
}

/// Parse a Linux sysfs cache size like "32K", "512K", "16M", "3G".
fn parse_size(s: &str) -> Option<usize> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let (num, mult) = match t.as_bytes()[t.len() - 1] {
        b'k' | b'K' => (&t[..t.len() - 1], 1024usize),
        b'm' | b'M' => (&t[..t.len() - 1], 1024 * 1024),
        b'g' | b'G' => (&t[..t.len() - 1], 1024 * 1024 * 1024),
        _ => (t, 1),
    };
    num.trim().parse::<usize>().ok().map(|v| v.saturating_mul(mult))
}

/// Linux sysfs cache topology: /sys/devices/system/cpu/cpu0/cache/indexN
/// with files {level,type,size,coherency_line_size}. Returns
/// (l1d, l2, l3, line) in bytes for cpu0 when readable.
#[cfg(target_os = "linux")]
fn linux_sysfs() -> Option<(usize, usize, usize, usize)> {
    let mut out = [0usize; 4]; // idx0=l1d,1=l2,2=l3,3=line
    for idx in 0..16 {
        let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        // cache indices are contiguous from 0: a missing indexN ends the scan
        let Ok(level_s) = std::fs::read_to_string(format!("{dir}/level")) else {
            break;
        };
        let Ok(typ_s) = std::fs::read_to_string(format!("{dir}/type")) else {
            continue;
        };
        let Ok(size_s) = std::fs::read_to_string(format!("{dir}/size")) else {
            continue;
        };
        let Ok(level) = level_s.trim().parse::<usize>() else {
            continue;
        };
        if !(1..=3).contains(&level) {
            continue;
        }
        let typ = typ_s.trim();
        if level == 1 && typ != "Data" {
            continue; // instruction/unified L1 is not the data cache we size for
        }
        if let Some(size) = parse_size(&size_s) {
            if size > out[level - 1] {
                out[level - 1] = size;
            }
        }
        if level == 1 && out[3] == 0 {
            out[3] = std::fs::read_to_string(format!("{dir}/coherency_line_size"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
        }
    }
    if out[0] == 0 && out[1] == 0 && out[2] == 0 {
        None
    } else {
        Some((out[0], out[1], out[2], out[3]))
    }
}

fn detect() -> CpuSpec {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);

    #[cfg(target_os = "linux")]
    let (l1d, l2, l3, line_sysfs) = linux_sysfs()
        .unwrap_or((32 * 1024, 512 * 1024, 8 * 1024 * 1024, 0));

    #[cfg(not(target_os = "linux"))]
    let (l1d, l2, l3, line_sysfs) = (32 * 1024, 512 * 1024, 8 * 1024 * 1024, 0usize);

    #[cfg(target_arch = "x86_64")]
    let cache_line = {
        // CPUID leaf 1: EBX[15:8] = CLFLUSH line size in 8-byte units
        // (universal on x86_64; reliable even where sysfs is hidden).
        let line = unsafe {
            let (_, ebx, _, _) = cpuid(1, 0);
            (((ebx >> 8) & 0xFF) as usize) * 8
        };
        if (16..=256).contains(&line) && line.is_power_of_two() {
            line
        } else {
            64
        }
    };

    #[cfg(not(target_arch = "x86_64"))]
    let cache_line = 64;

    CpuSpec {
        cache_line,
        l1d: clamp(l1d.max(line_sysfs * 4), 16 * 1024, 512 * 1024),
        l2: clamp(l2, 128 * 1024, 64 * 1024 * 1024),
        l3: clamp(l3, 1024 * 1024, 512 * 1024 * 1024),
        cores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_is_sane() {
        let s = best();
        assert!(s.cores >= 1, "cores = {}", s.cores);
        assert!(
            (16..=256).contains(&s.cache_line) && s.cache_line.is_power_of_two(),
            "cache line {}",
            s.cache_line
        );
        assert!((16 * 1024..=512 * 1024).contains(&s.l1d), "l1d {}", s.l1d);
        assert!((128 * 1024..=32 * 1024 * 1024).contains(&s.l2), "l2 {}", s.l2);
        assert!(s.l3 >= 1024 * 1024, "l3 {}", s.l3);
        eprintln!("[cpu] {}", s.log_summary());
    }
}
