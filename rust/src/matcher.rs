//! Paper matching: evaluate the AST against a paper, with SIMD pattern search.

use crate::ast::Node;
use crate::paper::{Paper, ALL_FIELDS, F_ANY};
use crate::simd::find;
use std::collections::HashMap;
use std::sync::Arc;

pub struct Pattern {
    pub raw: Arc<str>,
    lower_raw: Vec<u8>,
    parts: Vec<Vec<u8>>, // literal parts split on '*' (empty if the term has '?')
    no_wildcard: bool,
}

/// Per-text presence index: every 4-byte window (quad), every 2-byte window
/// (bigram) and every byte of a buffer. Built once per request per distinct
/// buffer, then each pattern does O(1) set lookups to prove it *cannot*
/// match (skipping the SIMD search entirely). Exact: no false negatives.
pub struct TextIndex {
    quads: std::collections::HashSet<u32>,
    bigrams: std::collections::HashSet<u16>,
    bytes: [bool; 256],
}

impl TextIndex {
    pub fn build(text: &[u8]) -> TextIndex {
        let n = text.len();
        let mut quads = std::collections::HashSet::with_capacity(n.saturating_sub(3));
        let mut bigrams = std::collections::HashSet::with_capacity(n.saturating_sub(1));
        let mut bytes = [false; 256];
        for w in text.windows(4) {
            quads.insert(u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
        }
        for w in text.windows(2) {
            bigrams.insert(u16::from_le_bytes([w[0], w[1]]));
        }
        for &b in text {
            bytes[b as usize] = true;
        }
        TextIndex { quads, bigrams, bytes }
    }

    /// True if the literal part *could* appear in the indexed text. Uses the
    /// first 4 bytes (or fewer for short parts) which any occurrence must
    /// contain, so a false return is a hard no.
    pub fn could_contain(&self, part: &[u8]) -> bool {
        match part.len() {
            0 => true,
            1 => self.bytes[part[0] as usize],
            2..=3 => self.bigrams.contains(&u16::from_le_bytes([part[0], part[1]])),
            _ => self.quads.contains(&u32::from_le_bytes([part[0], part[1], part[2], part[3]])),
        }
    }
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
            raw: Arc::from(kw),
            lower_raw: lower.clone().into_bytes(),
            parts: vec![lower.into_bytes()],
            no_wildcard: true,
        }
    } else if !has_q {
        Pattern {
            raw: Arc::from(kw),
            lower_raw: lower.clone().into_bytes(),
            parts: lower.split('*').filter(|p| !p.is_empty()).map(|p| p.as_bytes().to_vec()).collect(),
            no_wildcard: false,
        }
    } else {
        Pattern { raw: Arc::from(kw), lower_raw: lower.into_bytes(), parts: Vec::new(), no_wildcard: false }
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

/// Precompile every keyword in a set of AST blocks into a dense table.
/// The table is immutable after construction and shared across requests;
/// leaves are then resolved to table indices (`resolve_blocks`) so matching
/// never hashes keyword strings. Matching a paper used to recompile ~21k
/// patterns per request, which dominated the per-request cost.
pub fn compile_all<'a>(blocks: impl Iterator<Item = &'a Node>) -> Vec<Pattern> {
    let mut table: Vec<Pattern> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut leaves = Vec::new();
    for b in blocks {
        leaves.clear();
        collect_leaves(b, &mut leaves);
        for kw in leaves.drain(..) {
            if !seen.contains_key(kw) {
                let idx = table.len();
                table.push(compile_pattern(kw));
                seen.insert(kw.to_string(), idx);
            }
        }
    }
    table
}

/// Stamp every leaf with its pattern index in `table`. Call once at boot
/// after `compile_all`; matching then resolves leaves by array indexing.
pub fn resolve_blocks(blocks: &mut [Node], table: &[Pattern]) {
    let mut map: HashMap<&str, usize> = HashMap::with_capacity(table.len());
    for (i, p) in table.iter().enumerate() {
        map.insert(p.raw.as_ref(), i);
    }
    for b in blocks {
        resolve_node(b, &map);
    }
}

fn resolve_node(node: &mut Node, map: &HashMap<&str, usize>) {
    match node {
        Node::Leaf { keyword, pid, .. } => {
            // compile_pattern trims keywords (data artifact in SDG07), so
            // the lookup must trim too.
            *pid = *map
                .get(keyword.trim())
                .expect("leaf keyword missing from pattern table") as u32;
        }
        Node::Field { child, .. } => resolve_node(child, map),
        Node::Not { child } => resolve_node(child, map),
        Node::Group { children, .. } => {
            for c in children {
                resolve_node(c, map);
            }
        }
    }
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
    /// Cheap pre-filter: every literal part of the pattern must be present
    /// in the indexed text, otherwise `matches` cannot succeed. Glob ('?')
    /// patterns have no literal parts and always pass.
    pub fn could_match(&self, idx: &TextIndex) -> bool {
        if self.parts.is_empty() {
            return true;
        }
        self.parts.iter().all(|p| idx.could_contain(p))
    }

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

/// Per-request memo of term results, keyed by (pattern address, field mask).
/// The same keyword appears ~4.4x across the 17 SDG query sets, so memoizing
/// avoids re-searching the text for every duplicated leaf. Also caches the
/// per-buffer `TextIndex` (built once per distinct buffer) used to prove
/// most patterns cannot match before running a SIMD search.
pub struct Memo {
    terms: HashMap<(usize, u8), bool>,
    indexes: HashMap<(usize, usize), TextIndex>,
}

impl Memo {
    pub fn new() -> Memo {
        Memo { terms: HashMap::new(), indexes: HashMap::new() }
    }

    fn index(&mut self, buf: &[u8]) -> &TextIndex {
        let key = (buf.as_ptr() as usize, buf.len());
        self.indexes.entry(key).or_insert_with(|| TextIndex::build(buf))
    }

    fn term_hit(&mut self, paper: &Paper, pat: &Pattern, fields: &[u8]) -> bool {
        let mask = field_mask(fields);
        let key = (pat as *const Pattern as usize, mask);
        if let Some(&v) = self.terms.get(&key) {
            return v;
        }
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
        let mut v = false;
        for &b in &bufs[..nb] {
            // Filter first: if a literal part of the pattern is absent from
            // this buffer, the SIMD search is guaranteed to fail.
            if pat.could_match(self.index(b)) && pat.matches(b) {
                v = true;
                break;
            }
        }
        self.terms.insert(key, v);
        v
    }
}

fn pat<'a>(table: &'a [Pattern], node: &Node) -> &'a Pattern {
    match node {
        Node::Leaf { pid, .. } => &table[*pid as usize],
        _ => unreachable!("pat called on non-leaf"),
    }
}

/// Boolean evaluation of the AST against the paper (Scopus semantics:
/// NOT > AND/W-n > OR; W/n requires presence only).
pub fn eval(node: &Node, fields: Option<&[u8]>, paper: &Paper, table: &[Pattern]) -> bool {
    let mut memo = Memo::new();
    eval_memo(node, fields, paper, table, &mut memo)
}

fn eval_memo(
    node: &Node,
    fields: Option<&[u8]>,
    paper: &Paper,
    table: &[Pattern],
    memo: &mut Memo,
) -> bool {
    match node {
        Node::Leaf { .. } => {
            let p = pat(table, node);
            memo.term_hit(paper, p, fields.unwrap_or(&ALL_FIELDS))
        }
        Node::Field { fields: fs, child } => eval_memo(child, Some(&field_ids(fs)), paper, table, memo),
        Node::Not { child } => !eval_memo(child, fields, paper, table, memo),
        Node::Group { op, children } => {
            if op == "OR" {
                children.iter().any(|c| eval_memo(c, fields, paper, table, memo))
            } else {
                children.iter().all(|c| eval_memo(c, fields, paper, table, memo))
            }
        }
    }
}

pub struct BlockScan {
    pub hits: Vec<Arc<str>>,
    pub misses: Vec<Arc<str>>,
    pub excluded_hits: Vec<Arc<str>>,
}

/// Compact mask of a field list (bit 0-3 = TITLE/ABS/KEY/AUTHKEY, bit 7 = ANY).
fn field_mask(fields: &[u8]) -> u8 {
    let mut mask = 0u8;
    for &f in fields {
        mask |= 1 << (f & 7);
    }
    mask
}

/// Per-keyword detail for a block: include terms hit/missed, excluded terms hit.
pub fn scan_block(block: &Node, paper: &Paper, table: &[Pattern]) -> BlockScan {
    let mut out = BlockScan { hits: Vec::new(), misses: Vec::new(), excluded_hits: Vec::new() };
    let mut memo = Memo::new();
    rec(block, None, false, paper, table, &mut memo, &mut out);
    out
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
    table: &[Pattern],
    memo: &mut Memo,
) -> (Vec<(Arc<str>, u8)>, Vec<(Arc<str>, u8)>, Vec<Arc<str>>, bool) {
    let mut hits = Vec::new();
    let mut misses = Vec::new();
    let mut ex_hits = Vec::new();
    let matched = rec_fields(block, None, false, paper, table, memo, &mut hits, &mut misses, &mut ex_hits);
    (hits, misses, ex_hits, matched)
}

fn rec_fields(
    node: &Node,
    fields: Option<&[u8]>,
    excluded: bool,
    paper: &Paper,
    table: &[Pattern],
    memo: &mut Memo,
    hits: &mut Vec<(Arc<str>, u8)>,
    misses: &mut Vec<(Arc<str>, u8)>,
    ex_hits: &mut Vec<Arc<str>>,
) -> bool {
    match node {
        Node::Leaf { .. } => {
            let p = pat(table, node);
            let found = memo.term_hit(paper, p, fields.unwrap_or(&ALL_FIELDS));
            let mask = field_mask(fields.unwrap_or(&ALL_FIELDS));
            if excluded {
                if found {
                    ex_hits.push(p.raw.clone());
                }
            } else if found {
                hits.push((p.raw.clone(), mask));
            } else {
                misses.push((p.raw.clone(), mask));
            }
            found
        }
        Node::Field { fields: fs, child } => {
            rec_fields(child, Some(&field_ids(fs)), excluded, paper, table, memo, hits, misses, ex_hits)
        }
        Node::Not { child } => {
            !rec_fields(child, fields, !excluded, paper, table, memo, hits, misses, ex_hits)
        }
        Node::Group { op, children } => {
            if op == "OR" {
                let mut acc = false;
                for c in children {
                    acc |= rec_fields(c, fields, excluded, paper, table, memo, hits, misses, ex_hits);
                }
                acc
            } else {
                let mut acc = true;
                for c in children {
                    acc &= rec_fields(c, fields, excluded, paper, table, memo, hits, misses, ex_hits);
                }
                acc
            }
        }
    }
}

/// Human-readable field names for a mask ('' = default TITLE-ABS-KEY).
pub fn field_names(mask: u8) -> String {
    if mask == 0 || mask == field_mask(&ALL_FIELDS) {
        return String::new();
    }
    if mask == 1 << (F_ANY & 7) {
        return "TITLE-ABS-KEY".to_string();
    }
    let mut out = String::new();
    for (bit, name) in [(0u8, "TITLE"), (1, "ABS"), (2, "KEY"), (3, "AUTHKEY")] {
        if mask & (1 << bit) != 0 {
            if !out.is_empty() {
                out.push(',');
            }
            out.push_str(name);
        }
    }
    out
}


fn rec(
    node: &Node,
    fields: Option<&[u8]>,
    excluded: bool,
    paper: &Paper,
    table: &[Pattern],
    memo: &mut Memo,
    out: &mut BlockScan,
) {
    match node {
        Node::Leaf { .. } => {
            let p = pat(table, node);
            let found = memo.term_hit(paper, p, fields.unwrap_or(&ALL_FIELDS));
            if excluded {
                if found {
                    out.excluded_hits.push(p.raw.clone());
                }
            } else if found {
                out.hits.push(p.raw.clone());
            } else {
                out.misses.push(p.raw.clone());
            }
        }
        Node::Field { fields: fs, child } => rec(child, Some(&field_ids(fs)), excluded, paper, table, memo, out),
        Node::Not { child } => rec(child, fields, !excluded, paper, table, memo, out),
        Node::Group { children, .. } => {
            for c in children {
                rec(c, fields, excluded, paper, table, memo, out);
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
