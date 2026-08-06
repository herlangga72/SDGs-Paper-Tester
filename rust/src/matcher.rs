//! Paper matching: evaluate the AST against a paper, with SIMD pattern search.

use crate::ast::Node;
use crate::paper::{Paper, ALL_FIELDS, F_ABS, F_ANY, F_AUTHKEY, F_KEY, F_TITLE};
use crate::simd::find;
use std::collections::HashMap;

pub struct Pattern {
    pub raw: String,
    lower_raw: Vec<u8>,
    parts: Vec<Vec<u8>>, // literal parts split on '*' (empty if the term has '?')
    no_wildcard: bool,
}

fn compile_pattern(kw: &str) -> Pattern {
    // Leading/trailing whitespace in a keyword is a data artifact
    // (SDG07 contains `TITLE-ABS(" international") AND TITLE-ABS(" cooperation")`
    // for "international cooperation"); Scopus ignores it in phrases too.
    let kw = kw.trim();
    let lower = kw.to_ascii_lowercase();
    let has_star = lower.contains('*');
    let has_q = lower.contains('?');
    if !has_star && !has_q {
        Pattern {
            raw: kw.to_string(),
            lower_raw: lower.clone().into_bytes(),
            parts: vec![lower.into_bytes()],
            no_wildcard: true,
        }
    } else if !has_q {
        Pattern {
            raw: kw.to_string(),
            lower_raw: lower.clone().into_bytes(),
            parts: lower.split('*').filter(|p| !p.is_empty()).map(|p| p.as_bytes().to_vec()).collect(),
            no_wildcard: false,
        }
    } else {
        Pattern { raw: kw.to_string(), lower_raw: lower.into_bytes(), parts: Vec::new(), no_wildcard: false }
    }
}

fn collect_leaves<'a>(node: &'a Node, out: &mut Vec<&'a str>) {
    match node {
        Node::Leaf { keyword, .. } => out.push(keyword),
        Node::Field { child, .. } => collect_leaves(child, out),
        Node::Not { child } => collect_leaves(child, out),
        Node::Group { children, .. } => {
            for c in children {
                collect_leaves(c, out);
            }
        }
    }
}

/// Precompile every keyword in a set of AST blocks. The returned map is
/// immutable after construction, so it can be shared across requests:
/// matching a paper used to recompile ~21k patterns per request, which
/// dominated the per-request cost (~60 ms of the ~70 ms fixed time).
pub fn compile_all<'a>(blocks: impl Iterator<Item = &'a Node>) -> HashMap<String, Pattern> {
    let mut out = HashMap::new();
    let mut leaves = Vec::new();
    for b in blocks {
        leaves.clear();
        collect_leaves(b, &mut leaves);
        for kw in leaves.drain(..) {
            out.entry(kw.to_string()).or_insert_with(|| compile_pattern(kw));
        }
    }
    out
}

#[inline]
fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Whole-word search (Scopus phrase semantics for terms without wildcards).
fn find_boundary(hay: &[u8], needle: &[u8]) -> bool {
    let mut from = 0usize;
    loop {
        match find(hay, needle, from) {
            Some(p) => {
                let before = p == 0 || !is_word(hay[p - 1]);
                let after_p = p + needle.len();
                let after = after_p >= hay.len() || !is_word(hay[after_p]);
                if before && after {
                    return true;
                }
                from = p + 1;
            }
            None => return false,
        }
    }
}

/// Classic glob match: `*` = any sequence, `?` = one byte.
fn glob_match(pat: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while t < text.len() {
        if p < pat.len() && (pat[p] == b'?' || pat[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pat.len() && pat[p] == b'*' {
            star = Some(p);
            mark = t;
            p += 1;
        } else if let Some(sp) = star {
            p = sp + 1;
            mark += 1;
            t = mark;
        } else {
            return false;
        }
    }
    while p < pat.len() && pat[p] == b'*' {
        p += 1;
    }
    p == pat.len()
}

impl Pattern {
    pub fn matches(&self, text: &[u8]) -> bool {
        if self.no_wildcard {
            find_boundary(text, &self.parts[0])
        } else if self.parts.is_empty() {
            glob_match(&self.lower_raw, text)
        } else {
            let mut from = 0usize;
            for (k, part) in self.parts.iter().enumerate() {
                if k == 0 {
                    // the first literal may start anywhere
                    match find(text, part, 0) {
                        Some(p) => from = p + part.len(),
                        None => return false,
                    }
                } else {
                    // subsequent literals must follow with only non-space
                    // characters in between: Scopus `*` matches within a
                    // word only, so `foreign* trad*` requires "foreign
                    // trade", not "foreign aid ... trade" (SDG10).
                    let mut p = from;
                    loop {
                        match find(text, part, p) {
                            Some(x) if !crate::simd::any_ws(&text[from..x]) => {
                                from = x + part.len();
                                break;
                            }
                            Some(x) => p = x + 1,
                            None => return false,
                        }
                    }
                }
            }
            true
        }
    }
}

pub fn field_ids(fields: &[String]) -> Vec<u8> {
    fields
        .iter()
        .map(|f| match f.as_str() {
            "TITLE" => 0,
            "ABS" => 1,
            "KEY" => 2,
            "AUTHKEY" => 3,
            _ => F_ANY,
        })
        .collect()
}

/// Same underlying buffer (pointer + length), used to avoid scanning the
/// same text multiple times when several fields fall back to the full text.
fn same_buf(a: &[u8], b: &[u8]) -> bool {
    a.as_ptr() == b.as_ptr() && a.len() == b.len()
}

fn term_hit(paper: &Paper, pat: &Pattern, fields: &[u8]) -> bool {
    // Multiple fields often resolve to the same buffer (missing sections
    // fall back to the full text), so scan each distinct buffer once.
    let mut bufs: [&[u8]; 4] = [&[]; 4];
    let mut nb = 0usize;
    for &f in fields {
        let t = paper.text_lower(f);
        if !bufs[..nb].iter().any(|&b| same_buf(b, t)) {
            bufs[nb] = t;
            nb += 1;
        }
    }
    bufs[..nb].iter().any(|&t| pat.matches(t))
}

/// Per-request memo of term results, keyed by (pattern address, field mask).
/// The same keyword appears ~4.4x across the 17 SDG query sets, so memoizing
/// avoids re-searching the text for every duplicated leaf.
pub struct Memo(HashMap<(usize, u8), bool>);

impl Memo {
    pub fn new() -> Memo {
        Memo(HashMap::new())
    }

    fn term_hit(&mut self, paper: &Paper, pat: &Pattern, fields: &[u8]) -> bool {
        let mut mask = 0u8;
        for &f in fields {
            mask |= 1 << (f & 7);
        }
        let key = (pat as *const Pattern as usize, mask);
        if let Some(&v) = self.0.get(&key) {
            return v;
        }
        let v = term_hit(paper, pat, fields);
        self.0.insert(key, v);
        v
    }
}

fn pat<'a>(cache: &'a HashMap<String, Pattern>, kw: &str) -> &'a Pattern {
    // Callers are expected to precompile via `compile_all` (the web server
    // does this once at boot); a missing keyword is a programming error.
    cache
        .get(kw)
        .expect("keyword not found in precompiled pattern cache")
}

/// Boolean evaluation of the AST against the paper (Scopus semantics:
/// NOT > AND/W-n > OR; W/n requires presence only).
pub fn eval(node: &Node, fields: Option<&[u8]>, paper: &Paper, cache: &HashMap<String, Pattern>) -> bool {
    match node {
        Node::Leaf { keyword, .. } => {
            let p = pat(cache, keyword);
            term_hit(paper, p, fields.unwrap_or(&ALL_FIELDS))
        }
        Node::Field { fields: fs, child } => eval(child, Some(&field_ids(fs)), paper, cache),
        Node::Not { child } => !eval(child, fields, paper, cache),
        Node::Group { op, children } => {
            if op == "OR" {
                children.iter().any(|c| eval(c, fields, paper, cache))
            } else {
                children.iter().all(|c| eval(c, fields, paper, cache))
            }
        }
    }
}

pub struct BlockScan {
    pub hits: Vec<String>,
    pub misses: Vec<String>,
    pub excluded_hits: Vec<String>,
}

/// Per-keyword detail for a block: include terms hit/missed, excluded terms hit.
pub fn scan_block(block: &Node, paper: &Paper, cache: &HashMap<String, Pattern>) -> BlockScan {
    let mut out = BlockScan { hits: Vec::new(), misses: Vec::new(), excluded_hits: Vec::new() };
    rec(block, None, false, paper, cache, &mut out);
    out
}

fn field_name(f: u8) -> &'static str {
    match f {
        F_TITLE => "TITLE",
        F_ABS => "ABS",
        F_KEY => "KEY",
        F_AUTHKEY => "AUTHKEY",
        _ => "TITLE-ABS-KEY",
    }
}

/// Like `scan_block`, but each hit/miss also carries the field(s) the term is
/// searched in ('' -> the default TITLE-ABS-KEY). Used by the web server,
/// which renders `[TITLE,ABS]` chips next to every keyword.
/// `memo` is the per-request term-result cache (pass the same Memo across all
/// blocks of a request so duplicated keywords are searched once).
/// Returns (hits, misses, excluded_hits, matched): `matched` is the block's
/// boolean verdict, computed in the same single traversal (previously the
/// web server ran a separate `eval` pass over the whole AST per block).
pub fn scan_with_fields(
    block: &Node,
    paper: &Paper,
    cache: &HashMap<String, Pattern>,
    memo: &mut Memo,
) -> (Vec<(String, String)>, Vec<(String, String)>, Vec<String>, bool) {
    let mut hits = Vec::new();
    let mut misses = Vec::new();
    let mut ex_hits = Vec::new();
    let matched = rec_fields(block, None, false, paper, cache, memo, &mut hits, &mut misses, &mut ex_hits);
    (hits, misses, ex_hits, matched)
}

fn rec_fields(
    node: &Node,
    fields: Option<&[u8]>,
    excluded: bool,
    paper: &Paper,
    cache: &HashMap<String, Pattern>,
    memo: &mut Memo,
    hits: &mut Vec<(String, String)>,
    misses: &mut Vec<(String, String)>,
    ex_hits: &mut Vec<String>,
) -> bool {
    match node {
        Node::Leaf { keyword, .. } => {
            let p = pat(cache, keyword);
            let found = memo.term_hit(paper, p, fields.unwrap_or(&ALL_FIELDS));
            let fname = fields.map_or(String::new(), |f| {
                f.iter().map(|&x| field_name(x)).collect::<Vec<_>>().join(",")
            });
            if excluded {
                if found {
                    ex_hits.push(keyword.clone());
                }
            } else if found {
                hits.push((keyword.clone(), fname));
            } else {
                misses.push((keyword.clone(), fname));
            }
            found
        }
        Node::Field { fields: fs, child } => {
            rec_fields(child, Some(&field_ids(fs)), excluded, paper, cache, memo, hits, misses, ex_hits)
        }
        Node::Not { child } => {
            !rec_fields(child, fields, !excluded, paper, cache, memo, hits, misses, ex_hits)
        }
        Node::Group { op, children } => {
            if op == "OR" {
                let mut acc = false;
                for c in children {
                    acc |= rec_fields(c, fields, excluded, paper, cache, memo, hits, misses, ex_hits);
                }
                acc
            } else {
                let mut acc = true;
                for c in children {
                    acc &= rec_fields(c, fields, excluded, paper, cache, memo, hits, misses, ex_hits);
                }
                acc
            }
        }
    }
}

fn rec(
    node: &Node,
    fields: Option<&[u8]>,
    excluded: bool,
    paper: &Paper,
    cache: &HashMap<String, Pattern>,
    out: &mut BlockScan,
) {
    match node {
        Node::Leaf { keyword, .. } => {
            let p = pat(cache, keyword);
            let found = term_hit(paper, p, fields.unwrap_or(&ALL_FIELDS));
            if excluded {
                if found {
                    out.excluded_hits.push(keyword.clone());
                }
            } else if found {
                out.hits.push(keyword.clone());
            } else {
                out.misses.push(keyword.clone());
            }
        }
        Node::Field { fields: fs, child } => rec(child, Some(&field_ids(fs)), excluded, paper, cache, out),
        Node::Not { child } => rec(child, fields, !excluded, paper, cache, out),
        Node::Group { children, .. } => {
            for c in children {
                rec(c, fields, excluded, paper, cache, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_matches_plain_term() {
        let p = compile_pattern("foreign aid");
        assert!(p.matches(b"tax evasion and foreign aid in developing countries"));
        assert!(!p.matches(b"foreign-aid policies"));
    }

    #[test]
    fn pattern_matches_wildcards() {
        let p = compile_pattern("developing* countr*");
        assert!(p.matches(b"studies in developing countries"));
        let p = compile_pattern("poverty*-reducing*");
        assert!(p.matches(b"poverty-reducing policies"));
        assert!(!p.matches(b"povertyreducing policies")); // dash is required
        let p = compile_pattern("povertyreducing*");
        assert!(p.matches(b"povertyreducing policies"));
    }

    #[test]
    fn find_plain() {
        assert_eq!(find(b"abc foreign aid xyz", b"foreign aid", 0), Some(4));
        assert_eq!(find(b"aaaa", b"aa", 0), Some(0));
        assert_eq!(find(b"aaaa", b"aa", 1), Some(1));
        assert_eq!(find(b"aaaa", b"aa", 2), Some(2));
        assert_eq!(find(b"aaaa", b"aa", 3), None);
        // needles of every length in a haystack > 35 bytes (exercises the
        // AVX2 4-byte quad branch that used to compute `lane * 32`)
        let hay = b"the quick brown fox jumps over the lazy dog and foreign aid matters";
        for m in 4..=12 {
            let needle = &hay[10..10 + m];
            assert_eq!(find(hay, needle, 0), Some(10), "m={m}");
        }
        assert_eq!(find(hay, b"foreign aid", 0), Some(48));
    }
}
