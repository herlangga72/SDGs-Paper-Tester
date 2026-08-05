//! SIMD helpers: AVX2 on x86_64 (runtime-detected), portable scalar fallback.
//!
//! Hot paths:
//!   - `lower_ascii`   : case folding for paper texts (AVX2 range trick)
//!   - `find`          : substring search (1-byte broadcast, 4-byte quad trick)
//!   - `next_special`  : quoted/braced scanning in the tokenizer
//!   - `skip_ws`       : whitespace runs in the tokenizer

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Case folding
// ---------------------------------------------------------------------------

pub fn lower_ascii(s: &[u8]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { lower_ascii_avx2(s) };
    }
    let mut out = Vec::with_capacity(s.len());
    for &c in s {
        out.push(c.to_ascii_lowercase());
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
        // t = saturating(chunk - 'A' + 1); upper-case iff 1 <= t <= 26
        let t = _mm256_subs_epu8(chunk, _mm256_set1_epi8(b'A' as i8 - 1));
        let le26 = _mm256_cmpeq_epi8(_mm256_min_epu8(t, _mm256_set1_epi8(26)), t);
        let pos = _mm256_cmpgt_epi8(t, _mm256_set1_epi8(0));
        let upper = _mm256_and_si256(le26, pos);
        let lc = _mm256_add_epi8(chunk, _mm256_and_si256(upper, _mm256_set1_epi8(32)));
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, lc);
        i += 32;
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
    if is_x86_feature_detected!("avx2") && needle.len() <= 32 {
        return unsafe { find_avx2(rem, needle).map(|k| k + from) };
    }
    find_scalar(rem, needle).map(|k| k + from)
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
#[target_feature(enable = "avx2")]
unsafe fn find_avx2(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let n = hay.len();
    let m = needle.len();
    let mut i = 0usize;
    match m {
        // One byte: broadcast-compare every 32-byte chunk.
        1 => {
            let v = _mm256_set1_epi8(needle[0] as i8);
            while i + 32 <= n {
                let chunk = _mm256_loadu_si256(hay.as_ptr().add(i) as *const __m256i);
                let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v)) as u32;
                if mask != 0 {
                    return Some(i + mask.trailing_zeros() as usize);
                }
                i += 32;
            }
        }
        // 2-3 bytes: first-byte candidates + scalar verify.
        2..=3 => {
            let v = _mm256_set1_epi8(needle[0] as i8);
            while i + 32 <= n {
                let chunk = _mm256_loadu_si256(hay.as_ptr().add(i) as *const __m256i);
                let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v)) as u32;
                let mut bits = mask;
                while bits != 0 {
                    let off = bits.trailing_zeros() as usize;
                    let cand = i + off;
                    if cand + m <= n && &hay[cand..cand + m] == needle {
                        return Some(cand);
                    }
                    bits &= bits - 1;
                }
                i += 32;
            }
        }
        // 4..=32 bytes: 4-byte window filter at offsets 0..3 (the quad trick),
        // then scalar verify of the full needle at rare candidate positions.
        _ => {
            let first4 = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);
            let v = _mm256_set1_epi32(first4 as i32);
            while i + 35 <= n {
                let mut best: Option<usize> = None;
                'offs: for off in 0..4usize {
                    let chunk = _mm256_loadu_si256(hay.as_ptr().add(i + off) as *const __m256i);
                    let eq = _mm256_cmpeq_epi32(chunk, v);
                    let mask = _mm256_movemask_epi8(eq) as u32;
                    for lane in 0..8usize {
                        // dword lanes are 4 bytes apart
                        if (mask >> (lane * 4)) & 0xF != 0 {
                            let cand = i + off + lane * 4;
                            if cand + m <= n && &hay[cand..cand + m] == needle {
                                best = Some(cand);
                                break 'offs;
                            }
                        }
                    }
                }
                if let Some(p) = best {
                    return Some(p);
                }
                i += 32;
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
    let n = text.len();
    let mut i = from;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && !chars.is_empty() {
        return unsafe { next_special_avx2(text, from, chars) };
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
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { skip_ws_avx2(text, from) };
    }
    let mut i = from;
    while i < text.len() && text[i].is_ascii_whitespace() {
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
