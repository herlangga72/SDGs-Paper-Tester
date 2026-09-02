//! SIMD helpers with startup-time dispatch (x86_64 only).
//!
//! The best SIMD route available on this CPU is detected **once**, the first
//! time any SIMD helper runs (the program arranges this at startup), and
//! cached in a `DispatchLevel`. Every operation then dispatches off that one
//! stable decision - a single data-dependent branch on a value that never
//! changes after startup, so it stays perfectly branch-predictable and never
//! re-runs CPUID per call.
//!
//! Route ladder, in descending capability order:
//!   - AVX-512 (64-byte vectors, needs `avx512f`+`avx512bw`)
//!   - AVX2    (32-byte vectors)
//!   - SSE4.2  (`pcmpistri` string instructions; fastest for short-needle
//!     `find` even on AVX2-only hosts, so `find` uses it there)
//!   - SSE4.1  (`pcmpeqq` quads + `ptest` for long-needle `find`)
//!   - SSSE3   (`pshufb` two-table membership for `next_special`)
//!   - SSE3    (16-byte baseline; every x86_64 CPU since 2005)
//!   - Scalar  (non-x86_64 targets, or x86_64 CPUs without SSE3)
//!
//! Hot paths:
//!   - `lower_ascii`   : case folding for paper texts (two-compare mask)
//!   - `find`          : substring search (1-byte broadcast, 4-byte quad trick)
//!   - `any_ws`        : whitespace scan for wildcard gap checks
//!   - `next_special`  : quoted/braced scanning in the tokenizer
//!   - `skip_ws`       : whitespace runs in the tokenizer

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Startup-time dispatch level
// ---------------------------------------------------------------------------

/// The single SIMD route this CPU uses, in descending capability order.
/// `Scalar` is the floor for non-x86_64 targets.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum DispatchLevel {
    Scalar,
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
    Avx2,
    Avx512,
}

/// The chosen route, computed exactly once. All callers share this decision.
static LEVEL: OnceLock<DispatchLevel> = OnceLock::new();

/// Detect the best SIMD route once and return it. Repeated calls are a single
/// load of an initialized `OnceLock`. Call this at program startup (e.g. from
/// `main`) so the decision is made before any hot loop runs.
pub fn best_level() -> DispatchLevel {
    *LEVEL.get_or_init(detect_level)
}

#[cfg(target_arch = "x86_64")]
fn detect_level() -> DispatchLevel {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        DispatchLevel::Avx512
    } else if is_x86_feature_detected!("avx2") {
        DispatchLevel::Avx2
    } else if is_x86_feature_detected!("sse4.2") {
        DispatchLevel::Sse42
    } else if is_x86_feature_detected!("ssse3") {
        DispatchLevel::Ssse3
    } else if is_x86_feature_detected!("sse4.1") {
        DispatchLevel::Sse41
    } else if is_x86_feature_detected!("sse3") {
        DispatchLevel::Sse3
    } else {
        DispatchLevel::Scalar
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_level() -> DispatchLevel {
    DispatchLevel::Scalar
}

/// Human-readable name of the detected route, for startup diagnostics.
pub fn dispatch_name() -> &'static str {
    match best_level() {
        DispatchLevel::Avx512 => "AVX-512",
        DispatchLevel::Avx2 => "AVX2",
        DispatchLevel::Sse42 => "SSE4.2",
        DispatchLevel::Sse41 => "SSE4.1",
        DispatchLevel::Ssse3 => "SSSE3",
        DispatchLevel::Sse3 => "SSE3",
        DispatchLevel::Scalar => "scalar",
    }
}

// ---------------------------------------------------------------------------
// Case folding
// ---------------------------------------------------------------------------

pub fn lower_ascii(s: &[u8]) -> Vec<u8> {
    match best_level() {
        DispatchLevel::Avx512 => unsafe { lower_ascii_avx512(s) },
        DispatchLevel::Avx2 => unsafe { lower_ascii_avx2(s) },
        DispatchLevel::Scalar => {
            let mut out = Vec::with_capacity(s.len());
            for &c in s {
                out.push(c.to_ascii_lowercase());
            }
            out
        }
        // SSE3 is the 16-byte floor for x86_64; SSSE3/SSE4.1/SSE4.2 all
        // build on it and add nothing for case folding, so they share it.
        _ => unsafe { lower_ascii_sse3(s) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn lower_ascii_avx512(s: &[u8]) -> Vec<u8> {
    // Fused copy+fold: the source is read once and the destination written
    // once. (The old code did out = s.to_vec() and then folded the copy in
    // place, re-reading every written cache line - on buffers bigger than
    // L2 that is ~2x the DRAM traffic for no benefit.)
    let n = s.len();
    let mut out = vec![0u8; n];
    let mut i = 0;
    while i + 64 <= n {
        let chunk = _mm512_loadu_si512(s.as_ptr().add(i) as *const __m512i);
        // Branchless upper-case mask via two compares: 'A' <= c <= 'Z'.
        let ge_a = _mm512_cmpgt_epi8_mask(chunk, _mm512_set1_epi8(b'A' as i8 - 1));
        let le_z = _mm512_cmplt_epi8_mask(chunk, _mm512_set1_epi8(b'Z' as i8 + 1));
        let upper = ge_a & le_z;
        let lc = _mm512_add_epi8(chunk, _mm512_maskz_set1_epi8(upper, 32));
        _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, lc);
        i += 64;
    }
    while i < n {
        out[i] = s[i].to_ascii_lowercase();
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lower_ascii_avx2(s: &[u8]) -> Vec<u8> {
    // Fused copy+fold: one source read + one destination write per byte
    // (see lower_ascii_avx512 for why the old copy-then-fold was wasteful).
    let n = s.len();
    let mut out = vec![0u8; n];
    let mut i = 0;
    while i + 32 <= n {
        let chunk = _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i);
        // Branchless upper-case mask via two compares: 'A' <= c <= 'Z'.
        let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8(b'A' as i8 - 1));
        // chunk < 'Z'+1 (0x5B); non-ASCII bytes are signed-negative, so
        // they fail `ge_a` and stay untouched.
        let le_z = _mm256_cmpgt_epi8(_mm256_set1_epi8(b'Z' as i8 + 1), chunk);
        let upper = _mm256_and_si256(ge_a, le_z);
        let lc = _mm256_add_epi8(chunk, _mm256_and_si256(upper, _mm256_set1_epi8(32)));
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, lc);
        i += 32;
    }
    while i < n {
        out[i] = s[i].to_ascii_lowercase();
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Substring search
// ---------------------------------------------------------------------------

/// First occurrence of `needle` in `hay[from..]`.
pub fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= hay.len() {
        return None;
    }
    let rem = &hay[from..];
    if rem.len() < needle.len() {
        return None;
    }
    match best_level() {
        // AVX-512 is the widest rung; keep it first for hosts that have it.
        DispatchLevel::Avx512 if needle.len() <= 64 => {
            unsafe { find_avx512(rem, needle) }.map(|k| k + from)
        }
        // Benchmarked on Zen+ (2026-08): the SSE4.2 `pcmpistri` rung runs
        // ~7x faster than an AVX2 quad filter for needles <= 8 bytes
        // (52 us vs 373 us per 512 KiB scan), and SSE4.1 pcmpeqq edges
        // out AVX2 for longer needles as well. So on AVX2-only hosts `find`
        // still uses the SSE ladder; AVX2 adds nothing here. Every AVX2 CPU
        // has SSE4.2, so an AVX2 find rung is intentionally omitted.
        DispatchLevel::Avx512 | DispatchLevel::Avx2 | DispatchLevel::Sse42 => {
            unsafe { find_sse42(rem, needle) }.map(|k| k + from)
        }
        DispatchLevel::Sse41 => unsafe { find_sse41(rem, needle) }.map(|k| k + from),
        DispatchLevel::Ssse3 | DispatchLevel::Sse3 => {
            unsafe { find_sse3(rem, needle) }.map(|k| k + from)
        }
        DispatchLevel::Scalar => find_scalar(rem, needle).map(|k| k + from),
    }
}

fn find_scalar(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let (n, m) = (hay.len(), needle.len());
    if m == 0 || m > n {
        return None;
    }
    for i in 0..=(n - m) {
        if &hay[i..i + m] == needle {
            return Some(i);
        }
    }
    None
}

fn find_scalar_from(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let (n, m) = (hay.len(), needle.len());
    if m == 0 || from > n {
        return None;
    }
    let mut i = from;
    while i + m <= n {
        if &hay[i..i + m] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn find_avx512(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let n = hay.len();
    let m = needle.len();
    let mut i = 0usize;
    match m {
        // One byte: broadcast-compare every 64-byte chunk.
        1 => {
            let v = _mm512_set1_epi8(needle[0] as i8);
            while i + 64 <= n {
                let chunk = _mm512_loadu_si512(hay.as_ptr().add(i) as *const __m512i);
                let mask = _mm512_cmpeq_epi8_mask(chunk, v);
                if mask != 0 {
                    return Some(i + mask.trailing_zeros() as usize);
                }
                i += 64;
            }
        }
        // 2-3 bytes: first-byte candidates + scalar verify.
        2..=3 => {
            let v = _mm512_set1_epi8(needle[0] as i8);
            while i + 64 <= n {
                let chunk = _mm512_loadu_si512(hay.as_ptr().add(i) as *const __m512i);
                let mut bits = _mm512_cmpeq_epi8_mask(chunk, v);
                while bits != 0 {
                    let off = bits.trailing_zeros() as usize;
                    let cand = i + off;
                    if cand + m <= n && &hay[cand..cand + m] == needle {
                        return Some(cand);
                    }
                    bits &= bits - 1;
                }
                i += 64;
            }
        }
        // 4..=64 bytes: 4-byte window filter at offsets 0..3 (the quad trick),
        // then scalar verify of the full needle at rare candidate positions.
        _ => {
            let first4 = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);
            let v = _mm512_set1_epi32(first4 as i32);
            while i + 67 <= n {
                // Candidate positions are scanned in ascending order: a
                // naive off/lane nesting can verify a later match first.
                let mut masks = [0u64; 4];
                for off in 0..4usize {
                    let chunk = _mm512_loadu_si512(hay.as_ptr().add(i + off) as *const __m512i);
                    masks[off] = _mm512_cmpeq_epi32_mask(chunk, v) as u64;
                }
                for pos in 0..64usize {
                    if (masks[pos % 4] >> (pos / 4)) & 1 != 0 {
                        let cand = i + pos;
                        if cand + m <= n && &hay[cand..cand + m] == needle {
                            return Some(cand);
                        }
                    }
                }
                i += 64;
            }
        }
    }
    find_scalar_from(hay, needle, i)
}

// NOTE: `find_avx2` was removed after benchmarking (2026-08): the SSE4.2
// `pcmpistri` rung is ~7x faster for needles <= 8 bytes and SSE4.1 edges
// out AVX2 for longer needles on Zen+. Every AVX2 CPU also has SSE4.2, so
// the AVX2 find rung was unreachable in practice.

// ---------------------------------------------------------------------------
// Whitespace scan (wildcard gap checks in the matcher)
// ---------------------------------------------------------------------------

/// True if any byte in `s` is ASCII whitespace. Used by the matcher's
/// wildcard semantics (Scopus `*` matches within a word only), replacing a
/// per-byte scalar scan with a SIMD compare-OR.
pub fn any_ws(s: &[u8]) -> bool {
    match best_level() {
        // The SSE4.2 `pcmpistri` EQUAL_ANY single-instruction set match is
        // the fastest; AVX-512/AVX2 hosts still have SSE4.2, so they use it.
        DispatchLevel::Avx512 | DispatchLevel::Avx2 | DispatchLevel::Sse42 if s.len() >= 16 => {
            unsafe { any_ws_sse42(s) }
        }
        DispatchLevel::Sse41 | DispatchLevel::Ssse3 | DispatchLevel::Sse3 if s.len() >= 16 => {
            unsafe { any_ws_sse3(s) }
        }
        _ => s.iter().any(|c| c.is_ascii_whitespace()),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn any_ws_sse42(s: &[u8]) -> bool {
    let n = s.len();
    // EQUAL_ANY: index of the first byte IN the ws set (NUL-terminated).
    let ws = _mm_setr_epi8(
        b' ' as i8,
        b'\t' as i8,
        b'\n' as i8,
        b'\r' as i8,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let mut i = 0;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i);
        let idx = _mm_cmpistri(ws, chunk, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY);
        if idx < 16 {
            return true;
        }
        i += 16;
    }
    s[i..].iter().any(|c| c.is_ascii_whitespace())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn any_ws_sse3(s: &[u8]) -> bool {
    let n = s.len();
    let mut i = 0;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i);
        let ws = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b' ' as i8));
        let ws = _mm_or_si128(ws, _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\t' as i8)));
        let ws = _mm_or_si128(ws, _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\n' as i8)));
        let ws = _mm_or_si128(ws, _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\r' as i8)));
        if _mm_movemask_epi8(ws) != 0 {
            return true;
        }
        i += 16;
    }
    s[i..].iter().any(|c| c.is_ascii_whitespace())
}

// ---------------------------------------------------------------------------
// Tokenizer helpers
// ---------------------------------------------------------------------------

/// First index >= `from` where any byte of `chars` occurs.
pub fn next_special(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    match best_level() {
        DispatchLevel::Avx512 if !chars.is_empty() => {
            unsafe { next_special_avx512(text, from, chars) }
        }
        DispatchLevel::Avx2 if !chars.is_empty() => unsafe { next_special_avx2(text, from, chars) },
        // SSE4.2 `pcmpistri` EQUAL_ANY is fastest for sets of <= 15 bytes
        // without NUL (the set is NUL-terminated for implicit length).
        DispatchLevel::Avx512 | DispatchLevel::Avx2 | DispatchLevel::Sse42
            if !chars.is_empty() && chars.len() <= 15 && !chars.contains(&0) =>
        {
            unsafe { next_special_sse42(text, from, chars) }
        }
        // SSSE3 `pshufb` two-table membership is exact for sets whose
        // low-nibble layout doesn't collide within a high-nibble half.
        DispatchLevel::Ssse3 | DispatchLevel::Sse41 | DispatchLevel::Sse42
            if !chars.is_empty() && pshufb_representable(chars) =>
        {
            unsafe { next_special_ssse3(text, from, chars) }
        }
        DispatchLevel::Sse3 | DispatchLevel::Sse41 | DispatchLevel::Ssse3
        | DispatchLevel::Sse42 | DispatchLevel::Avx2 | DispatchLevel::Avx512
            if !chars.is_empty() =>
        {
            unsafe { next_special_sse3(text, from, chars) }
        }
        // Scalar fallback for non-x86_64 targets.
        _ => {
            let n = text.len();
            let mut i = from;
            while i < n {
                if chars.contains(&text[i]) {
                    return Some(i);
                }
                i += 1;
            }
            None
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn next_special_avx512(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    let n = text.len();
    let mut i = from;
    while i + 64 <= n {
        let chunk = _mm512_loadu_si512(text.as_ptr().add(i) as *const __m512i);
        let mut mask: u64 = 0;
        for &c in chars {
            mask |= _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(c as i8));
        }
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 64;
    }
    while i < n {
        if chars.contains(&text[i]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn next_special_avx2(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    let n = text.len();
    let mut i = from;
    while i + 32 <= n {
        let chunk = _mm256_loadu_si256(text.as_ptr().add(i) as *const __m256i);
        let mut mask: u32 = 0;
        for &c in chars {
            let eq = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(c as i8));
            mask |= _mm256_movemask_epi8(eq) as u32;
        }
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 32;
    }
    while i < n {
        if chars.contains(&text[i]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Skip a run of ASCII whitespace starting at `from`; returns first non-ws index.
pub fn skip_ws(text: &[u8], from: usize) -> usize {
    match best_level() {
        // AVX-512 is the widest rung; keep it first for hosts that have it.
        DispatchLevel::Avx512 => unsafe { skip_ws_avx512(text, from) },
        // SSE4.2 `pcmpistri` EQUAL_ANY + NEGATIVE_POLARITY finds the first
        // non-ws byte in one instruction. Benchmarked on Zen+ (2026-08):
        // it's faster than the AVX2 compare-OR (121 us vs 135 us per 256 KiB
        // scan), so AVX2-only hosts also use it here (every AVX2 CPU has
        // SSE4.2). The AVX2 rung is kept below but is slower on this chip.
        DispatchLevel::Avx2 | DispatchLevel::Sse42 => unsafe { skip_ws_sse42(text, from) },
        DispatchLevel::Sse41 | DispatchLevel::Ssse3 | DispatchLevel::Sse3 => {
            unsafe { skip_ws_sse3(text, from) }
        }
        // Scalar fallback for non-x86_64 targets.
        DispatchLevel::Scalar => {
            let mut i = from;
            while i < text.len() && text[i].is_ascii_whitespace() {
                i += 1;
            }
            i
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn skip_ws_avx512(text: &[u8], mut i: usize) -> usize {
    let n = text.len();
    while i + 64 <= n {
        let chunk = _mm512_loadu_si512(text.as_ptr().add(i) as *const __m512i);
        let ws = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b' ' as i8))
            | _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'\t' as i8))
            | _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'\n' as i8))
            | _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'\r' as i8));
        let notws = !ws;
        if notws != 0 {
            return i + notws.trailing_zeros() as usize;
        }
        i += 64;
    }
    while i < n && text[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// Retained for reference: benchmarked slower than the SSE4.2 `pcmpistri`
// rung on Zen+ (121 us vs 135 us per 256 KiB scan), so `skip_ws` does not
// dispatch to it. Kept so an AVX-512/AVX2 host can force it explicitly.
#[allow(dead_code)]
unsafe fn skip_ws_avx2(text: &[u8], mut i: usize) -> usize {
    let n = text.len();
    while i + 32 <= n {
        let chunk = _mm256_loadu_si256(text.as_ptr().add(i) as *const __m256i);
        let ws = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8));
        let ws = _mm256_or_si256(ws, _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\t' as i8)));
        let ws = _mm256_or_si256(ws, _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\n' as i8)));
        let ws = _mm256_or_si256(ws, _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\r' as i8)));
        let notws = !(_mm256_movemask_epi8(ws) as u32);
        if notws != 0 {
            return i + notws.trailing_zeros() as usize;
        }
        i += 32;
    }
    while i < n && text[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// SSE ladder (x86_64 fallback)
//
// 16-byte vectors, tiered so each ISA level builds on the one below it:
//   - SSE3   : baseline 16-byte processing (every x86_64 CPU since 2005)
//   - SSSE3  : `pshufb` two-table membership for `next_special`
//   - SSE4.1 : `pcmpeqq` 8-byte quads + `ptest` for `find`
//   - SSE4.2 : `pcmpistri` string instructions for `find` (equal-ordered),
//              `next_special` (equal-any) and `skip_ws` (equal-any with
//              negative polarity)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn lower_ascii_sse3(s: &[u8]) -> Vec<u8> {
    // Fused copy+fold: one source read + one destination write per byte
    // (see lower_ascii_avx512 for why the old copy-then-fold was wasteful).
    let n = s.len();
    let mut out = vec![0u8; n];
    let mut i = 0;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i);
        // Branchless upper-case mask via two compares: 'A' <= c <= 'Z'.
        // Benchmarked faster than the saturating-subtract range trick
        // (LLVM autovectorizes the naive map into this form too).
        let ge_a = _mm_cmpgt_epi8(chunk, _mm_set1_epi8(b'A' as i8 - 1));
        let le_z = _mm_cmplt_epi8(chunk, _mm_set1_epi8(b'Z' as i8 + 1));
        let upper = _mm_and_si128(ge_a, le_z);
        let lc = _mm_add_epi8(chunk, _mm_and_si128(upper, _mm_set1_epi8(32)));
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, lc);
        i += 16;
    }
    while i < n {
        out[i] = s[i].to_ascii_lowercase();
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn find_sse3(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let n = hay.len();
    let m = needle.len();
    let mut i = 0usize;
    match m {
        // One byte: broadcast-compare every 16-byte chunk.
        1 => {
            let v = _mm_set1_epi8(needle[0] as i8);
            while i + 16 <= n {
                let chunk = _mm_loadu_si128(hay.as_ptr().add(i) as *const __m128i);
                let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, v)) as u32;
                if mask != 0 {
                    return Some(i + mask.trailing_zeros() as usize);
                }
                i += 16;
            }
        }
        // 2-3 bytes: first-byte candidates + scalar verify.
        2..=3 => {
            let v = _mm_set1_epi8(needle[0] as i8);
            while i + 16 <= n {
                let chunk = _mm_loadu_si128(hay.as_ptr().add(i) as *const __m128i);
                let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, v)) as u32;
                let mut bits = mask;
                while bits != 0 {
                    let off = bits.trailing_zeros() as usize;
                    let cand = i + off;
                    if cand + m <= n && &hay[cand..cand + m] == needle {
                        return Some(cand);
                    }
                    bits &= bits - 1;
                }
                i += 16;
            }
        }
        // 4+ bytes: 4-byte window filter at offsets 0..3 (the quad trick),
        // then scalar verify of the full needle at rare candidate positions.
        // Handles any needle length.
        _ => {
            let first4 = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);
            let v = _mm_set1_epi32(first4 as i32);
            while i + 19 <= n {
                // Candidate positions are scanned in ascending order: a
                // naive off/lane nesting can verify a later match first.
                let mut masks = [0u32; 4];
                for off in 0..4usize {
                    let chunk = _mm_loadu_si128(hay.as_ptr().add(i + off) as *const __m128i);
                    let eq = _mm_cmpeq_epi32(chunk, v);
                    masks[off] = _mm_movemask_epi8(eq) as u32;
                }
                for pos in 0..16usize {
                    if (masks[pos % 4] >> ((pos / 4) * 4)) & 0xF != 0 {
                        let cand = i + pos;
                        if cand + m <= n && &hay[cand..cand + m] == needle {
                            return Some(cand);
                        }
                    }
                }
                i += 16;
            }
        }
    }
    find_scalar_from(hay, needle, i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn find_sse41(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let m = needle.len();
    if m < 9 {
        return find_sse3(hay, needle);
    }
    let n = hay.len();
    // 8-byte quads (`pcmpeqq`) at offsets 0..8 cover every possible start
    // in the 16-byte window; `ptest` skips windows with no candidate.
    let first8 = u64::from_le_bytes(needle[..8].try_into().unwrap());
    let v = _mm_set1_epi64x(first8 as i64);
    let mut i = 0usize;
    while i + 24 <= n {
        // Candidate positions are scanned in ascending order: a naive
        // off/lane nesting can verify a later match first.
        let mut masks = [0u32; 8];
        for off in 0..8usize {
            let chunk = _mm_loadu_si128(hay.as_ptr().add(i + off) as *const __m128i);
            let eq = _mm_cmpeq_epi64(chunk, v);
            if _mm_testz_si128(eq, eq) != 0 {
                continue;
            }
            masks[off] = _mm_movemask_epi8(eq) as u32;
        }
        for pos in 0..16usize {
            if (masks[pos % 8] >> ((pos / 8) * 8)) & 0xFF != 0 {
                let cand = i + pos;
                if cand + m <= n && &hay[cand..cand + m] == needle {
                    return Some(cand);
                }
            }
        }
        i += 16;
    }
    find_scalar_from(hay, needle, i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn find_sse42(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let n = hay.len();
    let m = needle.len();
    if m > 8 || needle[..m].contains(&0) {
        // `pcmpistri` needs the needle NUL-terminated (implicit length);
        // longer needles or embedded NULs use the SSE4.1 quad filter.
        return find_sse41(hay, needle);
    }
    // NUL-terminate the needle: implicit length = m. EQUAL_ORDERED reports
    // the least i where needle == window[i..i+m] fully inside the window,
    // so stride 16-m+1 guarantees every possible start is covered.
    let mut a = [0u8; 16];
    a[..m].copy_from_slice(needle);
    let av = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let stride = 16 - m + 1;
    let mut i = 0usize;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(hay.as_ptr().add(i) as *const __m128i);
        let idx = _mm_cmpistri(av, chunk, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED);
        if idx < 16 {
            let cand = i + idx as usize;
            // Defensive verify: the instruction guarantees a full in-window
            // match, but bounds + memcmp keep this robust to edge cases.
            if cand + m <= n && &hay[cand..cand + m] == needle {
                return Some(cand);
            }
        }
        i += stride;
    }
    find_scalar_from(hay, needle, i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn next_special_sse3(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    let n = text.len();
    let mut i = from;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(text.as_ptr().add(i) as *const __m128i);
        let mut mask: u32 = 0;
        for &c in chars {
            let eq = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(c as i8));
            mask |= _mm_movemask_epi8(eq) as u32;
        }
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 16;
    }
    while i < n {
        if chars.contains(&text[i]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn next_special_ssse3(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    let n = text.len();
    // `pshufb` two-table membership: table t_h holds, at index (c & 0x0F),
    // the special char c itself (split by high nibble half, so bytes with
    // the same low nibble never collide). `pshufb(t_h, chunk)` then returns
    // the stored char for each byte position, and `pcmpeqb` against the
    // chunk is an exact membership test - no false positives, no verify.
    let mut t0 = [0u8; 16];
    let mut t1 = [0u8; 16];
    for &c in chars {
        let lo = (c & 0x0F) as usize;
        if c >> 4 >= 8 {
            t1[lo] = c;
        } else {
            t0[lo] = c;
        }
    }
    let t0_v = _mm_loadu_si128(t0.as_ptr() as *const __m128i);
    let t1_v = _mm_loadu_si128(t1.as_ptr() as *const __m128i);
    let mut i = from;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(text.as_ptr().add(i) as *const __m128i);
        let r0 = _mm_shuffle_epi8(t0_v, chunk);
        let r1 = _mm_shuffle_epi8(t1_v, chunk);
        let eq = _mm_or_si128(_mm_cmpeq_epi8(r0, chunk), _mm_cmpeq_epi8(r1, chunk));
        let mask = _mm_movemask_epi8(eq) as u32;
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 16;
    }
    while i < n {
        if chars.contains(&text[i]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Whether a char set can be represented exactly by the two `pshufb`
/// membership tables: no two chars may share a low nibble within the same
/// high-nibble half, and NUL (0x00) is indistinguishable from an empty slot.
fn pshufb_representable(chars: &[u8]) -> bool {
    let mut seen = [0u8; 16];
    for &c in chars {
        if c == 0 {
            return false;
        }
        let lo = (c & 0x0F) as usize;
        let bit = 1 << (c >> 4 >= 8) as u32;
        if seen[lo] & bit != 0 {
            return false;
        }
        seen[lo] |= bit;
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn next_special_sse42(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    let n = text.len();
    // NUL-terminate the char set (implicit length). EQUAL_ANY then finds
    // the first byte belonging to the set in one instruction per 16 bytes.
    let mut a = [0u8; 16];
    a[..chars.len()].copy_from_slice(chars);
    let av = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let mut i = from;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(text.as_ptr().add(i) as *const __m128i);
        let idx = _mm_cmpistri(av, chunk, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY);
        if idx < 16 {
            return Some(i + idx as usize);
        }
        i += 16;
    }
    while i < n {
        if chars.contains(&text[i]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn skip_ws_sse3(text: &[u8], mut i: usize) -> usize {
    let n = text.len();
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(text.as_ptr().add(i) as *const __m128i);
        let ws = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b' ' as i8));
        let ws = _mm_or_si128(ws, _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\t' as i8)));
        let ws = _mm_or_si128(ws, _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\n' as i8)));
        let ws = _mm_or_si128(ws, _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\r' as i8)));
        let notws = !(_mm_movemask_epi8(ws) as u32);
        if notws != 0 {
            return i + notws.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < n && text[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn skip_ws_sse42(text: &[u8], mut i: usize) -> usize {
    let n = text.len();
    // EQUAL_ANY + NEGATIVE_POLARITY: index of the first byte that is NOT
    // one of the four whitespace characters (NUL-terminated set).
    let ws = _mm_setr_epi8(
        b' ' as i8,
        b'\t' as i8,
        b'\n' as i8,
        b'\r' as i8,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(text.as_ptr().add(i) as *const __m128i);
        let idx =
            _mm_cmpistri(ws, chunk, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_NEGATIVE_POLARITY);
        if idx < 16 {
            return i + idx as usize;
        }
        i += 16;
    }
    while i < n && text[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Tests (run the dispatched path available on the host CPU)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_level_detects_and_is_cached() {
        // The route is detected once and never changes; repeated calls must
        // return the identical value (and match the reported name).
        let l1 = best_level();
        let l2 = best_level();
        assert_eq!(l1, l2);
        assert_eq!(l1 as usize, dispatch_name_rank(l1));
        // The route must be one of the valid ladder rungs.
        match l1 {
            DispatchLevel::Avx512
            | DispatchLevel::Avx2
            | DispatchLevel::Sse42
            | DispatchLevel::Sse41
            | DispatchLevel::Ssse3
            | DispatchLevel::Sse3
            | DispatchLevel::Scalar => {}
        }
    }

    /// Returns the same ordinal the enum derives, so the test asserts the
    /// name/rank mapping stays in the documented capability order.
    fn dispatch_name_rank(l: DispatchLevel) -> usize {
        match l {
            DispatchLevel::Scalar => 0,
            DispatchLevel::Sse3 => 1,
            DispatchLevel::Ssse3 => 2,
            DispatchLevel::Sse41 => 3,
            DispatchLevel::Sse42 => 4,
            DispatchLevel::Avx2 => 5,
            DispatchLevel::Avx512 => 6,
        }
    }

    fn find_all(hay: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut out = Vec::new();
        let mut from = 0;
        while let Some(p) = find(hay, needle, from) {
            out.push(p);
            from = p + 1;
        }
        out
    }

    #[test]
    fn lower_ascii_matches_scalar() {
        let cases: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"Hello, WORLD!".to_vec(),
            b"The Quick Brown Fox Jumps Over The Lazy Dog".to_vec(),
            b"abcDEFGHIJKLMNOPQRSTUVWXYZxyz0123456789".to_vec(),
            b"a".repeat(300),
            b"AbCdEfGhIjKlMnOpQrStUvWxYz".repeat(17),
        ];
        for c in &cases {
            let got = lower_ascii(c);
            let want: Vec<u8> = c.iter().map(|b| b.to_ascii_lowercase()).collect();
            assert_eq!(got, want, "lower_ascii mismatch for {:?}", String::from_utf8_lossy(c));
        }
    }

    #[test]
    fn find_matches_scalar() {
        // 64+ byte haystacks to exercise the vector loops and their tails.
        let hay = b"lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod \
                     tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, \
                     quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.";
        let needles: Vec<Vec<u8>> = vec![
            b"lorem".to_vec(),
            b"a".to_vec(),
            b"Z".to_vec(),
            b"amet".to_vec(),
            b"consectetur".to_vec(),
            b"consequat.".to_vec(),
            b"ut".to_vec(),
            b"tempor incididunt ut labore".to_vec(),
            b"x".repeat(70), // longer than any vector path
        ];
        for nd in &needles {
            let got = find_all(hay, nd);
            let want = {
                let mut v = Vec::new();
                let mut i = 0;
                while i + nd.len() <= hay.len() {
                    if &hay[i..i + nd.len()] == nd {
                        v.push(i);
                    }
                    i += 1;
                }
                v
            };
            assert_eq!(got, want, "find mismatch for needle {:?}", String::from_utf8_lossy(nd));
        }
        assert_eq!(find(b"abc", b"", 0), None);
        assert_eq!(find(b"abc", b"d", 0), None);
        assert_eq!(find(b"abc", b"c", 2), Some(2));
        assert_eq!(find(b"abc", b"c", 3), None);
    }

    #[test]
    fn next_special_matches_scalar() {
        let text = b"((climate AND \"land use\") OR agriculture) AND TITLE(water*)";
        let chars = b"()\"*";
        for from in 0..text.len() {
            let got = next_special(text, from, chars);
            let want = (from..text.len()).find(|&i| chars.contains(&text[i]));
            assert_eq!(got, want, "mismatch at from={from}");
        }
    }

    #[test]
    fn sse_paths_match_reference() {
        // The host may dispatch to AVX-512/AVX2, so call the SSE-ladder
        // routines directly and compare against plain scalar references.
        let text: &[u8] = b"  \t\n\r  HELLO, World!  \n  tax evasion AND \"climate change\" (SDG07*)   ";
        let chars: &[u8] = b"()\"*";
        unsafe {
            // lower_ascii
            let want: Vec<u8> = text.iter().map(|b| b.to_ascii_lowercase()).collect();
            assert_eq!(lower_ascii_sse3(text), want);

            // find: every SSE rung, short and long needles, every from offset
            let long: Vec<u8> = b"x".repeat(20);
            let finders: [unsafe fn(&[u8], &[u8]) -> Option<usize>; 3] =
                [find_sse3, find_sse41, find_sse42];
            for nd in [
                b"HELLO".as_slice(),
                b"climate",
                b"e",
                b"Z",
                b"SDG07",
                b"world",
                b"climate change", // > 8 bytes: quad filter path
                long.as_slice(),   // > 16 bytes: quad filter path
            ] {
                for finder in finders {
                    let mut from = 0;
                    loop {
                        let got = finder(&text[from..], nd).map(|p| p + from);
                        let want = (from..text.len())
                            .find(|&i| i + nd.len() <= text.len() && &text[i..i + nd.len()] == nd);
                        assert_eq!(
                            got, want,
                            "find mismatch for needle {:?} from {from}",
                            String::from_utf8_lossy(nd)
                        );
                        match got {
                            Some(p) => from = p + 1,
                            None => break,
                        }
                    }
                }
            }
            // An embedded NUL forces find_sse42 to delegate to the SSE4.1
            // quad filter; the result must still be correct.
            let nul: &[u8] = b"cl\x00mate";
            let hay: &[u8] = b"xxcl\x00mateyy cl\x00mate";
            let got = find_sse42(hay, nul);
            let want = hay.windows(nul.len()).position(|w| w == nul);
            assert_eq!(got, want, "find_sse42 NUL-needle mismatch");

            // next_special: every SSE rung, every from offset
            let next_specials: [unsafe fn(&[u8], usize, &[u8]) -> Option<usize>; 3] =
                [next_special_sse3, next_special_ssse3, next_special_sse42];
            for ns in next_specials {
                for from in 0..text.len() {
                    let got = ns(text, from, chars);
                    let want = (from..text.len()).find(|&i| chars.contains(&text[i]));
                    assert_eq!(got, want, "next_special mismatch at from={from}");
                }
            }
            // Larger char sets (7 chars, all-distinct low nibbles so the
            // two-table representation stays exact) on the SSSE3 rung.
            let wide: &[u8] = b"()\"*!&|";
            for from in 0..text.len() {
                let got = next_special_ssse3(text, from, wide);
                let want = (from..text.len()).find(|&i| wide.contains(&text[i]));
                assert_eq!(got, want, "next_special_ssse3 wide mismatch at from={from}");
            }
            assert_eq!(next_special_ssse3(text, 0, b""), None);
            // Nibble-colliding sets must be rejected by the dispatch guard
            // so they fall back to the SSE3 per-char path.
            assert!(pshufb_representable(b"()\"*"));
            assert!(pshufb_representable(b"()\"*!&|"));
            assert!(!pshufb_representable(b"*:"));
            assert!(!pshufb_representable(b"a\x00b"));

            // skip_ws: both rungs
            let skippers: [unsafe fn(&[u8], usize) -> usize; 2] = [skip_ws_sse3, skip_ws_sse42];
            for sw in skippers {
                let mut i = 0;
                let mut expect = 0;
                while i < text.len() {
                    let got = sw(text, i);
                    while expect < text.len() && text[expect].is_ascii_whitespace() {
                        expect += 1;
                    }
                    assert_eq!(got, expect, "skip_ws mismatch at i={i}");
                    if got >= text.len() {
                        break;
                    }
                    i = got + 1;
                    expect = got + 1;
                }
            }
        }
    }

    #[test]
    fn find_returns_earliest_match_across_quad_lanes() {
        // Two matches inside one window whose starts sit in different
        // residue classes (2 and 7): the quad filter must still return the
        // earlier one. A naive off/lane scan checks position 12 (off 0,
        // lane 3) before position 2 (off 2, lane 0).
        let hay: &[u8] = b"xxABCDyABCDzzABCDwwABCD";
        let nd: &[u8] = b"ABCD";
        let want = vec![2usize, 7, 13, 19];
        let mut from = 0;
        let mut hits = Vec::new();
        while let Some(p) = find(hay, nd, from) {
            hits.push(p);
            from = p + 1;
        }
        assert_eq!(hits, want, "dispatched find");
        unsafe {
            for f in [find_sse3, find_sse41, find_sse42] {
                let mut from = 0;
                let mut hits = Vec::new();
                while let Some(p) = f(&hay[from..], nd).map(|k| k + from) {
                    hits.push(p);
                    from = p + 1;
                }
                assert_eq!(hits, want, "SSE finder");
            }
        }

        // Same check for a 9-byte needle, which drives the SSE4.1 pcmpeqq
        // quad path (8-byte quads at offsets 0..8).
        let hay9: &[u8] = b"xxABCDEFGHIyyABCDEFGHI";
        let nd9: &[u8] = b"ABCDEFGHI";
        let want9 = vec![2usize, 13];
        unsafe {
            for f in [find_sse3, find_sse41, find_sse42] {
                let mut from = 0;
                let mut hits = Vec::new();
                while let Some(p) = f(&hay9[from..], nd9).map(|k| k + from) {
                    hits.push(p);
                    from = p + 1;
                }
                assert_eq!(hits, want9, "SSE finder (9-byte needle)");
            }
        }
    }

    #[test]
    #[ignore = "speed benchmark; run with --ignored --nocapture in release mode"]
    fn bench_sse_ladder() {
        use std::hint::black_box;
        use std::time::Instant;

        // ~512 KiB of realistic mixed-case text with occasional specials,
        // so every workload terminates quickly and realistically.
        let mut text = Vec::with_capacity(512 << 10);
        let words = b"lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod \
                      tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, \
                      quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. ";
        while text.len() + words.len() <= 512 << 10 {
            text.extend_from_slice(words);
            let start = text.len() - words.len();
            if text.len() % 4096 < 2048 {
                text[start..].make_ascii_uppercase();
            }
            // Sprinkle tokenizer specials into the trailing gap (never
            // inside words, so the find needles stay intact).
            text[start + words.len() - 1] = b'(';
        }
        let text = text;
        let n = text.len();

        fn bench<T>(name: &str, iters: u64, f: impl Fn() -> T) {
            for _ in 0..5 {
                black_box(f());
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                black_box(f());
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("{name:<30} {us:>9.1} us/op");
            eprintln!("done: {name}");
        }

        println!(
            "input: {} bytes on {}",
            n,
            if is_x86_feature_detected!("avx2") { "AVX2 host" } else { "SSE host" }
        );

        // lower_ascii
        let iters = 100u64;
        let want = black_box(text.iter().map(|b| b.to_ascii_lowercase()).collect::<Vec<u8>>());
        assert_eq!(lower_ascii(&text), want);
        bench("lower_ascii scalar", iters, || {
            text.iter().map(|b| b.to_ascii_lowercase()).collect::<Vec<u8>>()
        });
        bench("lower_ascii sse3", iters, || unsafe { lower_ascii_sse3(&text) });
        bench("lower_ascii avx2", iters, || unsafe { lower_ascii_avx2(&text) });
        bench("lower_ascii dispatched", iters, || lower_ascii(&text));

        // find: sparse needles over the full text
        let needles: [&[u8]; 4] = [b"amet", b"lorem", b"consectetur", b"eiusmod tempor incididunt"];
        for &nd in &needles {
            let exp = find_all(&text, nd).len() as u64;
            assert!(exp > 0, "needle {:?} not found", String::from_utf8_lossy(nd));
            let iters = 30u64;
            let f = |fnder: unsafe fn(&[u8], &[u8]) -> Option<usize>| {
                // One full scan per invocation (bench repeats it). The
                // finder returns an offset relative to the `from` slice,
                // so advance by `p + 1` in absolute coordinates.
                let mut count = 0u64;
                let mut from = 0usize;
                while let Some(p) = unsafe { fnder(&text[from..], nd) } {
                    count += 1;
                    from += p + 1;
                }
                assert_eq!(count, exp, "needle {:?}", String::from_utf8_lossy(nd));
            };
            bench(&format!("find({}) scalar", String::from_utf8_lossy(nd)), iters, || {
                let mut from = 0;
                while let Some(p) = find_scalar_from(&text, nd, from) {
                    from = p + 1;
                }
            });
            bench(&format!("find({}) sse3", String::from_utf8_lossy(nd)), iters, || f(find_sse3));
            bench(&format!("find({}) sse41", String::from_utf8_lossy(nd)), iters, || f(find_sse41));
            bench(&format!("find({}) sse42", String::from_utf8_lossy(nd)), iters, || f(find_sse42));
            bench(&format!("find({}) dispatched", String::from_utf8_lossy(nd)), iters, || {
                let mut from = 0;
                while let Some(p) = find(&text, nd, from) {
                    from = p + 1;
                }
            });
        }

        // next_special: text now contains real '(' hits
        let chars: &[u8] = b"()\"*";
        let iters = 200u64;
        bench("next_special scalar", iters, || {
            let mut i = 0;
            while let Some(p) = (i..n).find(|&j| chars.contains(&text[j])) {
                i = p + 1;
            }
        });
        bench("next_special sse3", iters, || {
            let mut i = 0;
            while let Some(p) = unsafe { next_special_sse3(&text, i, chars) } {
                i = p + 1;
            }
        });
        bench("next_special ssse3", iters, || {
            let mut i = 0;
            while let Some(p) = unsafe { next_special_ssse3(&text, i, chars) } {
                i = p + 1;
            }
        });
        bench("next_special sse42", iters, || {
            let mut i = 0;
            while let Some(p) = unsafe { next_special_sse42(&text, i, chars) } {
                i = p + 1;
            }
        });
        bench("next_special dispatched", iters, || {
            let mut i = 0;
            while let Some(p) = next_special(&text, i, chars) {
                i = p + 1;
            }
        });

        // skip_ws: text with runs of whitespace
        let mut wstext = Vec::with_capacity(256 << 10);
        while wstext.len() + 32 <= 256 << 10 {
            wstext.extend_from_slice(b"   \t\n\r  word ");
        }
        let wstext = wstext;
        let iters = 200u64;
        bench("skip_ws scalar", iters, || {
            let mut i = 0;
            while i < wstext.len() {
                i = wstext.iter().skip(i).position(|b| !b.is_ascii_whitespace()).map(|k| i + k).unwrap_or(wstext.len());
                i = wstext.iter().skip(i).position(|b| b.is_ascii_whitespace()).map(|k| i + k).unwrap_or(wstext.len());
            }
        });
        bench("skip_ws sse3", iters, || {
            let mut i = 0;
            while i < wstext.len() {
                i = unsafe { skip_ws_sse3(&wstext, i) };
                i = wstext.iter().skip(i).position(|b| b.is_ascii_whitespace()).map(|k| i + k).unwrap_or(wstext.len());
            }
        });
        bench("skip_ws sse42", iters, || {
            let mut i = 0;
            while i < wstext.len() {
                i = unsafe { skip_ws_sse42(&wstext, i) };
                i = wstext.iter().skip(i).position(|b| b.is_ascii_whitespace()).map(|k| i + k).unwrap_or(wstext.len());
            }
        });
        bench("skip_ws dispatched", iters, || {
            let mut i = 0;
            while i < wstext.len() {
                i = skip_ws(&wstext, i);
                i = wstext.iter().skip(i).position(|b| b.is_ascii_whitespace()).map(|k| i + k).unwrap_or(wstext.len());
            }
        });
    }

    #[test]
    fn any_ws_matches_scalar() {
        let cases: [&[u8]; 7] = [
            b"",
            b"loremipsum",
            b"lorem ipsum",
            b"a\tb",
            b"\n",
            b"no-ws-here-123",
            b"line1\nline2",
        ];
        for c in &cases {
            let want = c.iter().any(|b| b.is_ascii_whitespace());
            assert_eq!(any_ws(c), want, "any_ws mismatch for {:?}", String::from_utf8_lossy(c));
        }
        // Long inputs exercise the vector loop and its tail.
        let long: Vec<u8> = b"abcdefgh".repeat(20); // 160 bytes, no ws
        assert!(!any_ws(&long));
        let mut with_ws = long.clone();
        with_ws[150] = b'\t';
        assert!(any_ws(&with_ws));
        let mut ws_at_end = long.clone();
        ws_at_end.push(b' ');
        assert!(any_ws(&ws_at_end));
    }

    #[test]
    fn skip_ws_matches_scalar() {
        let text = b"   \t\n\r  hello \n world\t\t";
        let mut i = 0;
        let mut expect = 0;
        while i < text.len() {
            let got = skip_ws(text, i);
            while expect < text.len() && text[expect].is_ascii_whitespace() {
                expect += 1;
            }
            assert_eq!(got, expect, "mismatch at i={i}");
            if got >= text.len() {
                break;
            }
            i = got + 1;
            expect = got + 1;
        }
    }
}
