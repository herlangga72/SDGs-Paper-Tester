//! Aho-Corasick multi-pattern exact-substring scanner (dependency-free).
//!
//! Finds every occurrence of any pattern in one linear pass over the text.
//! The web highlighter uses it to highlight all matched plain keywords
//! without a per-keyword full-text scan.
//!
//! Semantics: a pattern occurrence at `[start, end)` is reported for every
//! position where the pattern's bytes appear verbatim, including overlaps
//! (`scan` reports both `"he"` at `1..3` and `"she"` at `0..3` inside
//! `"she"`), exactly like repeated `memmem` searches. Patterns are matched
//! case- and boundary-blind here; callers apply their own boundary rules.
//!
//! The trie keeps sparse per-state edges (keyword sets are small), so memory
//! is O(total pattern length). The classic goto/failure scan is amortized
//! linear over the text for typical sets; the theoretical worst case is
//! O(text x depth) when one long failure chain is re-walked per byte, which
//! does not occur for keyword highlighting.

/// A built Aho-Corasick automaton. Reusable: `scan` may be called any number
/// of times over different texts.
pub struct Aho {
    /// Sorted `(byte, next state)` edges per state (sparse goto table).
    go: Vec<Vec<(u8, u32)>>,
    /// Failure link per state: longest proper suffix that is itself a
    /// pattern prefix (root fails to itself).
    fail: Vec<u32>,
    /// Pattern ids that END at this state.
    ends: Vec<Vec<u32>>,
    /// Dictionary-suffix link: nearest strict ancestor in the failure chain
    /// that ends at least one pattern, or 0. Emitting at a terminal position
    /// walks this chain so shorter patterns sharing the suffix are reported
    /// at the same end position.
    dsuf: Vec<u32>,
    /// Byte length of each accepted pattern (id order, empties skipped).
    lens: Vec<usize>,
}

impl Aho {
    /// Build the automaton. Empty patterns are skipped (they would match at
    /// every position); the caller can detect this by comparing `len()` to
    /// the input count.
    pub fn new(patterns: &[impl AsRef<[u8]>]) -> Aho {
        let mut go = vec![Vec::new()];
        let mut ends: Vec<Vec<u32>> = vec![Vec::new()];
        let mut lens: Vec<usize> = Vec::new();
        for pat in patterns {
            let bytes = pat.as_ref();
            if bytes.is_empty() {
                continue;
            }
            let pid = lens.len() as u32;
            lens.push(bytes.len());
            let mut s = 0usize;
            for &b in bytes {
                match go[s].iter().position(|&(c, _)| c == b) {
                    Some(k) => s = go[s][k].1 as usize,
                    None => {
                        let n = go.len() as u32;
                        go[s].push((b, n));
                        go.push(Vec::new());
                        ends.push(Vec::new());
                        s = n as usize;
                    }
                }
            }
            ends[s].push(pid);
        }
        for v in &mut go {
            v.sort_unstable();
        }
        let n = go.len();
        let mut fail = vec![0u32; n];
        let mut dsuf = vec![0u32; n];
        // BFS in depth order. Children of the root fail to the root. For a
        // deeper state u reached from r over byte b, u's failure is the
        // result of following r's failure chain to the first state that has
        // an edge b (root has none -> fail to root).
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut head = 0usize;
        for &(_, c) in &go[0] {
            order.push(c);
        }
        while head < order.len() {
            let r = order[head] as usize;
            head += 1;
            for &(b, u) in &go[r] {
                order.push(u);
                let mut f = fail[r] as usize;
                loop {
                    if let Some(k) = go[f].iter().position(|&(c, _)| c == b) {
                        let v = go[f][k].1;
                        fail[u as usize] = v;
                        // If v itself ends a pattern it is the nearest
                        // pattern-ending ancestor; otherwise inherit v's.
                        dsuf[u as usize] = if ends[v as usize].is_empty() { dsuf[v as usize] } else { v };
                        break;
                    }
                    if f == 0 {
                        fail[u as usize] = 0;
                        dsuf[u as usize] = 0;
                        break;
                    }
                    f = fail[f] as usize;
                }
            }
        }
        Aho { go, fail, ends, dsuf, lens }
    }

    /// Number of accepted (non-empty) patterns.
    pub fn len(&self) -> usize {
        self.lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }

    /// Byte length of the accepted pattern `pid`.
    #[inline]
    pub fn pattern_len(&self, pid: usize) -> usize {
        self.lens[pid]
    }

    /// Scan `text` once, invoking `emit(pid, start, end)` for every
    /// occurrence of every pattern. Occurrences are reported in ascending
    /// end position; the order of different patterns ending at the same
    /// position is unspecified (but deterministic for a given automaton).
    pub fn scan(&self, text: &[u8], mut emit: impl FnMut(u32, usize, usize)) {
        if self.go.len() == 1 {
            return; // no non-empty patterns
        }
        let mut s = 0usize;
        for (i, &b) in text.iter().enumerate() {
            // Consume the byte: follow failure links until the current state
            // has an edge for it (or we are back at the root).
            loop {
                if let Ok(k) = self.go[s].binary_search_by_key(&b, |&(c, _)| c) {
                    s = self.go[s][k].1 as usize;
                    break;
                }
                if s == 0 {
                    break;
                }
                s = self.fail[s] as usize;
            }
            // Emit every pattern that ends at position i+1: patterns whose
            // terminal state is the current state, plus the ones reached via
            // the dictionary-suffix chain (shorter suffix patterns).
            let end = i + 1;
            let mut t = s as usize;
            loop {
                for &pid in &self.ends[t] {
                    let start = end - self.lens[pid as usize];
                    emit(pid, start, end);
                }
                t = self.dsuf[t] as usize;
                if t == 0 {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn naive(hay: &[u8], pat: &[u8]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if pat.is_empty() || pat.len() > hay.len() {
            return out;
        }
        for s in 0..=hay.len() - pat.len() {
            if &hay[s..s + pat.len()] == pat {
                out.push((s, s + pat.len()));
            }
        }
        out
    }

    #[test]
    fn ac_matches_naive() {
        let patterns: Vec<&[u8]> = vec![
            b"he", b"she", b"his", b"hers", b"a", b"aa", b"coral reef", b"ab", b"b",
            b"\xc3\xa9", // multi-byte "é"
            b"",
            b"x",
        ];
        let text = b"ushers she his hers a aa aaa ab abcd coral reef r\xc3\xa9sum\xc3\xa9 x";
        let ac = Aho::new(&patterns);
        let accepted: Vec<&[u8]> = patterns.iter().copied().filter(|p| !p.is_empty()).collect();
        let mut expect: HashSet<(u32, usize, usize)> = HashSet::new();
        for (pid, p) in accepted.iter().enumerate() {
            for (s, e) in naive(text, p) {
                expect.insert((pid as u32, s, e));
            }
        }
        let mut got = HashSet::new();
        ac.scan(text, |pid, s, e| {
            got.insert((pid, s, e));
        });
        assert_eq!(got.len(), expect.len());
        for g in &got {
            assert!(expect.contains(g), "unexpected {g:?}");
        }
        for e in &expect {
            assert!(got.contains(e), "missing {e:?}");
        }
    }

    /// Randomized parity with per-pattern naive substring search.
    #[test]
    fn ac_random_matches_naive() {
        let alphabet: Vec<u8> = b"abcde ".to_vec();
        let mut rng = 0xC0FF_EE00_1234_5678u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for trial in 0..40 {
            let mut patterns: Vec<Vec<u8>> = Vec::new();
            for _ in 0..(1 + next() % 8) {
                let len = 1 + (next() % 6) as usize;
                let p: Vec<u8> = (0..len).map(|_| alphabet[(next() % alphabet.len() as u64) as usize]).collect();
                patterns.push(p);
            }
            let text: Vec<u8> = (0..(next() % 200))
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                .collect();
            let ac = Aho::new(&patterns);
            let mut expect = HashSet::new();
            for (pid, p) in patterns.iter().enumerate() {
                for (s, e) in naive(&text, p) {
                    expect.insert((pid as u32, s, e));
                }
            }
            let mut got = HashSet::new();
            ac.scan(&text, |pid, s, e| {
                got.insert((pid, s, e));
            });
            assert_eq!(got, expect, "trial {trial} patterns {patterns:?} text {:?}", String::from_utf8_lossy(&text));
        }
    }

    #[test]
    fn scan_requires_length_parity() {
        let ac = Aho::new(&["abc".as_bytes(), "b".as_bytes(), "z".as_bytes()]);
        assert_eq!(ac.len(), 3);
        assert_eq!(ac.pattern_len(0), 3);
        assert_eq!(ac.pattern_len(1), 1);
        assert_eq!(ac.pattern_len(2), 1);
        // duplicate / empty patterns are skipped
        let ac2 = Aho::new(&["".as_bytes(), "".as_bytes()]);
        assert!(ac2.is_empty());
    }
}
