//! SIMD helpers with runtime dispatch:
//!   - x86_64: AVX-512 (64-byte vectors, `avx512f`+`avx512bw`, runtime-detected),
//!     then AVX2 (32-byte vectors, runtime-detected), then the SSE ladder
//!     (16-byte vectors, runtime-detected): SSE4.2 (`pcmpistri` string
//!     instructions), SSE4.1 (`pcmpeqq` quads + `ptest`), SSSE3 (`pshufb`
//!     nibble lookups), SSE3 (16-byte baseline), then scalar
//!   - aarch64: NEON (16-byte vectors; baseline on ARMv8, no detection needed)
//!   - everything else: portable scalar
//!
//! Hot paths:
//!   - `lower_ascii`   : case folding for paper texts (range trick)
//!   - `find`          : substring search (1-byte broadcast, 4-byte quad trick)
//!   - `next_special`  : quoted/braced scanning in the tokenizer
//!   - `skip_ws`       : whitespace runs in the tokenizer

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Case folding
// ---------------------------------------------------------------------------

pub fn lower_ascii(s: &[u8]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { lower_ascii_avx512(s) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { lower_ascii_avx2(s) };
        }
        // SSE3 is the 16-byte floor for x86_64 (every CPU made since 2005);
        // CPUs without it fall through to the scalar path below.
        if is_x86_feature_detected!("sse3") {
            return unsafe { lower_ascii_sse3(s) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { lower_ascii_neon(s) };
    }
    // Scalar fallback: non-SIMD targets, and x86_64 CPUs without SSE3.
    {
        let mut out = Vec::with_capacity(s.len());
        for &c in s {
            out.push(c.to_ascii_lowercase());
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn lower_ascii_avx512(s: &[u8]) -> Vec<u8> {
    let mut out = s.to_vec();
    let n = out.len();
    let mut i = 0;
    while i + 64 <= n {
        let chunk = _mm512_loadu_si512(out.as_ptr().add(i) as *const __m512i);
        // Branchless upper-case mask via two compares: 'A' <= c <= 'Z'.
        let ge_a = _mm512_cmpgt_epi8_mask(chunk, _mm512_set1_epi8(b'A' as i8 - 1));
        let le_z = _mm512_cmplt_epi8_mask(chunk, _mm512_set1_epi8(b'Z' as i8 + 1));
        let upper = ge_a & le_z;
        let lc = _mm512_add_epi8(chunk, _mm512_maskz_set1_epi8(upper, 32));
        _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, lc);
        i += 64;
    }
    for c in out[i..].iter_mut() {
        *c = c.to_ascii_lowercase();
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lower_ascii_avx2(s: &[u8]) -> Vec<u8> {
    let mut out = s.to_vec();
    let n = out.len();
    let mut i = 0;
    while i + 32 <= n {
        let chunk = _mm256_loadu_si256(out.as_ptr().add(i) as *const __m256i);
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
    for c in out[i..].iter_mut() {
        *c = c.to_ascii_lowercase();
    }
    out
}

#[cfg(target_arch = "aarch64")]
unsafe fn lower_ascii_neon(s: &[u8]) -> Vec<u8> {
    let mut out = s.to_vec();
    let n = out.len();
    let mut i = 0;
    while i + 16 <= n {
        let chunk = vld1q_u8(out.as_ptr().add(i));
        // t = saturating(chunk - 'A' + 1); upper-case iff 1 <= t <= 26
        let t = vqsubq_u8(chunk, vdupq_n_u8(b'A' - 1));
        let le26 = vceqq_u8(vminq_u8(t, vdupq_n_u8(26)), t);
        let pos = vcgtq_s8(vreinterpretq_s8_u8(t), vdupq_n_s8(0));
        let upper = vandq_u8(le26, pos);
        let lc = vaddq_u8(chunk, vandq_u8(upper, vdupq_n_u8(32)));
        vst1q_u8(out.as_mut_ptr().add(i), lc);
        i += 16;
    }
    for c in out[i..].iter_mut() {
        *c = c.to_ascii_lowercase();
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
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 is the widest rung; keep it first for hosts that have it.
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            && needle.len() <= 64
        {
            return unsafe { find_avx512(rem, needle).map(|k| k + from) };
        }
        // Benchmarked on Zen+ (2026-08): the SSE4.2 `pcmpistri` rung runs
        // ~7x faster than the AVX2 quad filter for needles <= 8 bytes
        // (52 us vs 373 us per 512 KiB scan), and SSE4.1 pcmpeqq edges
        // out AVX2 for longer needles as well. Prefer the SSE ladder;
        // AVX2 stays for the bulk transform paths (lower_ascii,
        // next_special, skip_ws). Every CPU with AVX2 also has SSE4.2,
        // so the AVX2 find rung is unreachable and intentionally omitted.
        if is_x86_feature_detected!("sse4.2") {
            return unsafe { find_sse42(rem, needle).map(|k| k + from) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { find_sse41(rem, needle).map(|k| k + from) };
        }
        if is_x86_feature_detected!("sse3") {
            return unsafe { find_sse3(rem, needle).map(|k| k + from) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON handles every needle length (quad filter + scalar verify).
        return unsafe { find_neon(rem, needle).map(|k| k + from) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        find_scalar(rem, needle).map(|k| k + from)
    }
}

#[cfg(not(target_arch = "aarch64"))]
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

#[cfg(target_arch = "aarch64")]
unsafe fn find_neon(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let n = hay.len();
    let m = needle.len();
    let mut i = 0usize;
    match m {
        // One byte: broadcast-compare every 16-byte chunk.
        1 => {
            let v = vdupq_n_u8(needle[0]);
            while i + 16 <= n {
                let chunk = vld1q_u8(hay.as_ptr().add(i));
                let mask = neon_movemask(vceqq_u8(chunk, v));
                if mask != 0 {
                    return Some(i + (mask.trailing_zeros() / 4) as usize);
                }
                i += 16;
            }
        }
        // 2-3 bytes: first-byte candidates + scalar verify.
        2..=3 => {
            let v = vdupq_n_u8(needle[0]);
            while i + 16 <= n {
                let chunk = vld1q_u8(hay.as_ptr().add(i));
                let mut bits = neon_movemask(vceqq_u8(chunk, v));
                while bits != 0 {
                    let off = (bits.trailing_zeros() / 4) as usize;
                    let cand = i + off;
                    if cand + m <= n && &hay[cand..cand + m] == needle {
                        return Some(cand);
                    }
                    // NEON masks use a 4-bit window per byte; clear the
                    // whole window so the loop terminates.
                    bits &= !(0xF << bits.trailing_zeros());
                }
                i += 16;
            }
        }
        // 4+ bytes: 4-byte window filter at offsets 0..3 (the quad trick),
        // then scalar verify of the full needle at rare candidate positions.
        _ => {
            let first4 = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);
            let v = vdupq_n_u32(first4);
            while i + 19 <= n {
                // Candidate positions are scanned in ascending order: a
                // naive off/lane nesting can verify a later match first.
                let mut masks = [0u64; 4];
                for off in 0..4usize {
                    let chunk = vld1q_u8(hay.as_ptr().add(i + off));
                    let eq = vceqq_u32(vreinterpretq_u32_u8(chunk), v);
                    // 0xFFFF per matching u32 lane, packed into 64 bits
                    masks[off] = vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(eq, 4)), 0);
                }
                for pos in 0..16usize {
                    if (masks[pos % 4] >> ((pos / 4) * 16)) & 0xFFFF != 0 {
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

// ---------------------------------------------------------------------------
// Tokenizer helpers
// ---------------------------------------------------------------------------

/// First index >= `from` where any byte of `chars` occurs.
pub fn next_special(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            && !chars.is_empty()
        {
            return unsafe { next_special_avx512(text, from, chars) };
        }
        if is_x86_feature_detected!("avx2") && !chars.is_empty() {
            return unsafe { next_special_avx2(text, from, chars) };
        }
        // SSE ladder: EQUAL_ANY via `pcmpistri` (SSE4.2, sets of <= 15
        // bytes without NUL), `pshufb` two-table membership (SSSE3, sets
        // without nibble collisions), per-char compares (SSE3).
        if is_x86_feature_detected!("sse4.2")
            && !chars.is_empty()
            && chars.len() <= 15
            && !chars.contains(&0)
        {
            return unsafe { next_special_sse42(text, from, chars) };
        }
        if is_x86_feature_detected!("ssse3") && !chars.is_empty() && pshufb_representable(chars) {
            return unsafe { next_special_ssse3(text, from, chars) };
        }
        if is_x86_feature_detected!("sse3") && !chars.is_empty() {
            return unsafe { next_special_sse3(text, from, chars) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { next_special_neon(text, from, chars) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
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

#[cfg(target_arch = "aarch64")]
unsafe fn next_special_neon(text: &[u8], from: usize, chars: &[u8]) -> Option<usize> {
    let n = text.len();
    let mut i = from;
    while i + 16 <= n {
        let chunk = vld1q_u8(text.as_ptr().add(i));
        let mut mask: u64 = 0;
        for &c in chars {
            mask |= neon_movemask(vceqq_u8(chunk, vdupq_n_u8(c)));
        }
        if mask != 0 {
            return Some(i + (mask.trailing_zeros() / 4) as usize);
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

/// Skip a run of ASCII whitespace starting at `from`; returns first non-ws index.
pub fn skip_ws(text: &[u8], from: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { skip_ws_avx512(text, from) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { skip_ws_avx2(text, from) };
        }
        // SSE ladder: EQUAL_ANY + NEGATIVE_POLARITY via `pcmpistri`
        // (SSE4.2), compare-OR of the four whitespace bytes (SSE3).
        if is_x86_feature_detected!("sse4.2") {
            return unsafe { skip_ws_sse42(text, from) };
        }
        if is_x86_feature_detected!("sse3") {
            return unsafe { skip_ws_sse3(text, from) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { skip_ws_neon(text, from) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut i = from;
        while i < text.len() && text[i].is_ascii_whitespace() {
            i += 1;
        }
        i
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

#[cfg(target_arch = "aarch64")]
unsafe fn skip_ws_neon(text: &[u8], mut i: usize) -> usize {
    let n = text.len();
    while i + 16 <= n {
        let chunk = vld1q_u8(text.as_ptr().add(i));
        let ws = vorrq_u8(
            vorrq_u8(
                vorrq_u8(vceqq_u8(chunk, vdupq_n_u8(b' ')), vceqq_u8(chunk, vdupq_n_u8(b'\t'))),
                vceqq_u8(chunk, vdupq_n_u8(b'\n')),
            ),
            vceqq_u8(chunk, vdupq_n_u8(b'\r')),
        );
        let notws = !neon_movemask(ws);
        if notws != 0 {
            return i + (notws.trailing_zeros() / 4) as usize;
        }
        i += 16;
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
    let mut out = s.to_vec();
    let n = out.len();
    let mut i = 0;
    while i + 16 <= n {
        let chunk = _mm_loadu_si128(out.as_ptr().add(i) as *const __m128i);
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
    for c in out[i..].iter_mut() {
        *c = c.to_ascii_lowercase();
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
// aarch64 helpers
// ---------------------------------------------------------------------------

/// Compact 64-bit match mask for a 16-byte NEON compare result.
///
/// Each original byte occupies a 4-bit window: byte 2k -> bits [4k, 4k+4),
/// byte 2k+1 -> bits [4k+4, 4k+8). `trailing_zeros() / 4` recovers the
/// index of the first matching byte.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_movemask(v: uint8x16_t) -> u64 {
    let v = vshrn_n_u16(vreinterpretq_u16_u8(v), 4);
    vget_lane_u64(vreinterpret_u64_u8(v), 0)
}

// ---------------------------------------------------------------------------
// Tests (run the dispatched path available on the host CPU)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
