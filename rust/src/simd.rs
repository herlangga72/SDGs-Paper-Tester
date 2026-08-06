//! SIMD helpers with runtime dispatch:
//!   - x86_64: AVX-512 (64-byte vectors, `avx512f`+`avx512bw`, runtime-detected),
//!     then AVX2 (32-byte vectors, runtime-detected), then scalar
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
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { lower_ascii_neon(s) };
    }
    #[cfg(not(target_arch = "aarch64"))]
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
        // t = saturating(chunk - 'A' + 1); upper-case iff 1 <= t <= 26
        let t = _mm512_subs_epu8(chunk, _mm512_set1_epi8(b'A' as i8 - 1));
        let le26 = _mm512_cmpeq_epi8_mask(_mm512_min_epu8(t, _mm512_set1_epi8(26)), t);
        let pos = _mm512_cmpgt_epi8_mask(t, _mm512_set1_epi8(0));
        let upper = le26 & pos;
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
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            && needle.len() <= 64
        {
            return unsafe { find_avx512(rem, needle).map(|k| k + from) };
        }
        if is_x86_feature_detected!("avx2") && needle.len() <= 32 {
            return unsafe { find_avx2(rem, needle).map(|k| k + from) };
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
                let mut best: Option<usize> = None;
                'offs: for off in 0..4usize {
                    let chunk = _mm512_loadu_si512(hay.as_ptr().add(i + off) as *const __m512i);
                    let mask = _mm512_cmpeq_epi32_mask(chunk, v);
                    for lane in 0..16usize {
                        // dword lanes are 4 bytes apart
                        if (mask >> lane) & 1 != 0 {
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
                i += 64;
            }
        }
    }
    find_scalar_from(hay, needle, i)
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
                let mut best: Option<usize> = None;
                'offs: for off in 0..4usize {
                    let chunk = vld1q_u8(hay.as_ptr().add(i + off));
                    let eq = vceqq_u32(vreinterpretq_u32_u8(chunk), v);
                    // 0xFFFF per matching u32 lane, packed into 64 bits
                    let mask = vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(eq, 4)), 0);
                    for lane in 0..4usize {
                        if (mask >> (lane * 16)) & 0xFFFF != 0 {
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
