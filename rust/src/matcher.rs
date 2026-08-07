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
///
/// Memory layout is chosen to be branch-predictable, cache-resident and
/// SIMD-friendly rather than hash-table based:
///   - `bytes`   : one bit per byte value (256 B).
///   - `bigrams` : dense 256x256 bit matrix (8 KiB) indexed by `a<<8|b` for
///                 the first two bytes of a part. A single load + shift +
///                 mask, no probing.
///   - `quads`   : a two-hash Bloom filter over the first four bytes of a
///                 part, sized ~16 bits per window. Two independent
///                 multiplicative hashes into a power-of-two bit array keep
///                 the false-positive rate low while remaining a handful of
///                 branch-free bit operations. Bloom filters have no false
///                 negatives, which is all this pre-filter needs.
/// The old `HashSet`-based index paid for hash-table allocation/rehashing to
/// build (lots of transient memory movement) and walked probe chains on every
/// lookup (unpredictable branches). Dense bit arrays are built in a single
/// sequential pass and answer with two loads at most.
pub struct TextIndex {
    bytes: [bool; 256],
    bigrams: [u64; 1024],
    quads: Vec<u64>,
    quad_mask: u32, // power-of-two capacity minus one
}

/// Two independent multiplicative hashes into a `mask`-bounded bit array.
/// Deterministic (a quad always maps to the same two bits), so the index has
/// no false negatives; collisions are harmless false positives.
#[inline]
fn bloom_hashes(q: u32, mask: u32) -> (u32, u32) {
    let h1 = q.wrapping_mul(0x9E37_79B1) & mask;
    let h2 = (q.wrapping_mul(0x85EB_CA6B) ^ q.rotate_left(11)) & mask;
    (h1, h2)
}

impl TextIndex {
    pub fn build(text: &[u8]) -> TextIndex {
        let n = text.len();
        let mut bytes = [false; 256];
        let mut bigrams = [0u64; 1024];
        // ~16 bits of Bloom per window bounds the combined false-positive
        // rate to ~(1/16)^2 even when every window is distinct. Minimum 64K
        // bits (8 KiB) keeps tiny inputs cheap.
        let qbits = (n.saturating_sub(3).saturating_mul(16)).next_power_of_two().max(1 << 16);
        let mut quads = vec![0u64; (qbits >> 6) as usize];
        let quad_mask = (qbits as u32) - 1;

        for w in text.windows(2) {
            let a = w[0] as usize;
            let b = w[1] as usize;
            // bit index = a*256 + b
            bigrams[(a << 2) | (b >> 6)] |= 1u64 << (b & 63);
        }
        for w in text.windows(4) {
            let q = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            let (h1, h2) = bloom_hashes(q, quad_mask);
            quads[(h1 >> 6) as usize] |= 1u64 << (h1 & 63);
            quads[(h2 >> 6) as usize] |= 1u64 << (h2 & 63);
        }
        for &b in text {
            bytes[b as usize] = true;
        }
        TextIndex { bytes, bigrams, quads, quad_mask }
    }

    /// True if the literal part *could* appear in the indexed text. Any
    /// occurrence of `part` must contain every one of its internal 4-byte
    /// windows, so we require all of them to be present. Checking all quads
    /// (not just the first) rejects e.g. a part whose first 4 bytes appear
    /// but whose full word does not, which is the common false-positive
    /// driving wasted SIMD scans. A false return is still a hard no.
    pub fn could_contain(&self, part: &[u8]) -> bool {
        match part.len() {
            0 => true,
            1 => self.bytes[part[0] as usize],
            2..=3 => {
                let a = part[0] as usize;
                let b = part[1] as usize;
                (self.bigrams[(a << 2) | (b >> 6)] >> (b & 63)) & 1 != 0
            }
            _ => {
                let mut w = 0;
                while w + 4 <= part.len() {
                    let q = u32::from_le_bytes([part[w], part[w + 1], part[w + 2], part[w + 3]]);
                    let (h1, h2) = bloom_hashes(q, self.quad_mask);
                    if ((self.quads[(h1 >> 6) as usize] >> (h1 & 63)) & 1) == 0
                        || ((self.quads[(h2 >> 6) as usize] >> (h2 & 63)) & 1) == 0
                    {
                        return false;
                    }
                    w += 1;
                }
                true
            }
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

/// Stamp every leaf with its pattern index, effective field mask, and dense
/// memo slot. Call once at boot after `compile_all`; matching then resolves
/// leaves by array indexing and never hashes keyword strings or
/// `(pattern, mask)` keys.
///
/// The effective mask is static per leaf (determined by its enclosing `Field`
/// nodes), so it is threaded down here rather than recomputed at match time.
/// `slot` is a dense index over the live `(pid, mask)` pairs; the per-request
/// memo is a `Vec<u8>` of `nslots` entries (see `Memo::with_slots`).
/// Returns the number of distinct `(pid, mask)` slots assigned.
pub fn resolve_blocks(blocks: &mut [Node], table: &[Pattern]) -> usize {
    let mut map: HashMap<&str, usize> = HashMap::with_capacity(table.len());
    for (i, p) in table.iter().enumerate() {
        map.insert(p.raw.as_ref(), i);
    }
    // Slot ids: dense over the seen (pid, mask) pairs, assigned in first-seen
    // order. `pid` is already dense over the table, so keying the memo array
    // on `pid` alone would be too coarse (the same pattern under different
    // field masks can give different results) — hence a (pid, mask) slot.
    let mut slot_of: HashMap<(u32, u8), u32> = HashMap::new();
    let mut nslots = 0u32;
    fn assign(
        node: &mut Node,
        mask: u8,
        map: &HashMap<&str, usize>,
        slot_of: &mut HashMap<(u32, u8), u32>,
        nslots: &mut u32,
    ) {
        match node {
            Node::Leaf { keyword, pid, mask: lm, slot, .. } => {
                let pidv = *map.get(keyword.trim()).expect("leaf keyword missing from pattern table") as u32;
                *pid = pidv;
                *lm = mask;
                *slot = match slot_of.get(&(pidv, mask)) {
                    Some(&s) => s,
                    None => {
                        let s = *nslots;
                        *nslots += 1;
                        slot_of.insert((pidv, mask), s);
                        s
                    }
                };
            }
            Node::Field { fields, child } => {
                assign(child, field_mask_from_strings(fields), map, slot_of, nslots)
            }
            Node::Not { child } => assign(child, mask, map, slot_of, nslots),
            Node::Group { children, .. } => {
                for c in children {
                    assign(c, mask, map, slot_of, nslots);
                }
            }
        }
    }
    let default_mask = field_mask(&ALL_FIELDS);
    for b in blocks {
        assign(b, default_mask, &map, &mut slot_of, &mut nslots);
    }
    nslots as usize
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

/// Compact mask of a field-name list (bit 0-3 = TITLE/ABS/KEY/AUTHKEY, bit 7
/// = ANY). Computed once at each `Field` node instead of allocating a `Vec`
/// of field ids per evaluation.
fn field_mask_from_strings(fields: &[String]) -> u8 {
    let mut mask = 0u8;
    for f in fields {
        let id = match f.as_str() {
            "TITLE" => 0,
            "ABS" => 1,
            "KEY" => 2,
            "AUTHKEY" => 3,
            _ => F_ANY,
        };
        mask |= 1 << (id & 7);
    }
    mask
}

/// Same underlying buffer (pointer + length), used to avoid scanning the
/// same text multiple times when several fields fall back to the full text.
fn same_buf(a: &[u8], b: &[u8]) -> bool {
    a.as_ptr() == b.as_ptr() && a.len() == b.len()
}

/// Append `t` to `bufs` if it is not already present (dedup by identity).
fn push_buf<'a>(bufs: &mut [&'a [u8]; 4], nb: &mut usize, t: &'a [u8]) {
    if !bufs[..*nb].iter().any(|&b| same_buf(b, t)) {
        bufs[*nb] = t;
        *nb += 1;
    }
}

/// Fast non-cryptographic hasher for the per-request caches. The hot path
/// does ~10^5 HashMap lookups per request; std's default SipHash (RandomState)
/// is deliberately DoS-resistant but ~10x slower than a multiplicative hash.
/// Keys here are only (pattern address, field mask) tuples - not user-supplied
/// adversarial strings - so a trivial hash is safe. Zero-dependency.
struct FastHasher(u64);

impl std::hash::Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01B3);
        }
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n).wrapping_mul(0x100_0000_01B3);
    }
    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.write_u64(u64::from(n));
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

impl std::hash::BuildHasher for FastHasher {
    type Hasher = FastHasher;
    #[inline]
    fn build_hasher(&self) -> FastHasher {
        FastHasher(0xcbf2_9ce4_8422_2325)
    }
}

type FastMap<K, V> = HashMap<K, V, FastHasher>;

fn new_fast_map<K, V>() -> FastMap<K, V> {
    HashMap::with_capacity_and_hasher(16, FastHasher(0xcbf2_9ce4_8422_2325))
}

/// Per-request memo of term results, keyed by the leaf's precomputed dense
/// `slot` (see `resolve_blocks`): `terms[slot]` is 0 (unset), 1 (false) or
/// 2 (true). Slots are dense over the live `(pattern, mask)` pairs, so the
/// hot path is a single `Vec<u8>` read/write instead of a hashed
/// `(pattern*, mask)` lookup. Also caches the per-buffer `TextIndex` (built
/// once per distinct buffer) used to prove most patterns cannot match before
/// running a SIMD search.
pub struct Memo {
    terms: Vec<u8>,
    indexes: FastMap<(usize, usize), TextIndex>,
}

impl Memo {
    pub fn new() -> Memo {
        Memo { terms: Vec::new(), indexes: new_fast_map() }
    }

    fn index(&mut self, buf: &[u8]) -> &TextIndex {
        let key = (buf.as_ptr() as usize, buf.len());
        self.indexes.entry(key).or_insert_with(|| TextIndex::build(buf))
    }

    /// Evaluate `pat` under field `mask`, memoized at dense `slot`. The mask
    /// and slot are both precomputed and stamped on the leaf by
    /// `resolve_blocks`, so the caller passes them straight through.
    fn term_hit(&mut self, paper: &Paper, pat: &Pattern, mask: u8, slot: usize) -> bool {
        if let Some(&v) = self.terms.get(slot) {
            // 0 = unset; anything else is the cached verdict.
            return v == 2;
        }
        let v = self.compute(paper, pat, mask);
        // Grow on demand so `Memo::new()` needs no slot count (resizes once
        // to the max slot of the request, then is a single write below).
        if slot >= self.terms.len() {
            self.terms.resize(slot + 1, 0);
        }
        self.terms[slot] = if v { 2 } else { 1 };
        v
    }

    /// Actual search for `pat` under `mask` (uncached). Decodes the mask into
    /// the section buffers (bits 0-3) plus the full text (bit 7). Multiple
    /// fields often resolve to the same buffer (missing sections fall back to
    /// the full text), so scan each distinct buffer once.
    fn compute(&mut self, paper: &Paper, pat: &Pattern, mask: u8) -> bool {
        // A zero mask means "no field scoping" -> the default TITLE-ABS-KEY.
        let mask = if mask == 0 { field_mask(&ALL_FIELDS) } else { mask };
        let mut bufs: [&[u8]; 4] = [&[]; 4];
        let mut nb = 0usize;
        for f in 0..4u8 {
            if mask & (1 << f) != 0 {
                push_buf(&mut bufs, &mut nb, paper.text_lower(f));
            }
        }
        if mask & (1 << (F_ANY & 7)) != 0 {
            push_buf(&mut bufs, &mut nb, paper.text_lower(F_ANY));
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
    eval_memo(node, field_mask(fields.unwrap_or(&ALL_FIELDS)), paper, table, &mut memo)
}

fn eval_memo(node: &Node, mask: u8, paper: &Paper, table: &[Pattern], memo: &mut Memo) -> bool {
    match node {
        Node::Leaf { mask, slot, .. } => {
            let p = pat(table, node);
            memo.term_hit(paper, p, *mask, *slot as usize)
        }
        Node::Field { fields: fs, child } => eval_memo(child, field_mask_from_strings(fs), paper, table, memo),
        Node::Not { child } => !eval_memo(child, mask, paper, table, memo),
        Node::Group { op, children } => {
            if op == "OR" {
                children.iter().any(|c| eval_memo(c, mask, paper, table, memo))
            } else {
                children.iter().all(|c| eval_memo(c, mask, paper, table, memo))
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
    let mut memo = Memo::new();
    scan_block_shared(block, paper, table, &mut memo).0
}

/// `scan_block` with a caller-owned `Memo` (share it across all blocks of a
/// request so each distinct (pattern, field-mask) is searched once) and the
/// block's boolean verdict computed in the same single traversal.
pub fn scan_block_shared(
    block: &Node,
    paper: &Paper,
    table: &[Pattern],
    memo: &mut Memo,
) -> (BlockScan, bool) {
    let mut out = BlockScan { hits: Vec::new(), misses: Vec::new(), excluded_hits: Vec::new() };
    let matched = rec(block, field_mask(&ALL_FIELDS), false, paper, table, memo, &mut out);
    (out, matched)
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
    let matched = rec_fields(block, field_mask(&ALL_FIELDS), false, paper, table, memo, &mut hits, &mut misses, &mut ex_hits);
    (hits, misses, ex_hits, matched)
}

fn rec_fields(
    node: &Node,
    mask: u8,
    excluded: bool,
    paper: &Paper,
    table: &[Pattern],
    memo: &mut Memo,
    hits: &mut Vec<(Arc<str>, u8)>,
    misses: &mut Vec<(Arc<str>, u8)>,
    ex_hits: &mut Vec<Arc<str>>,
) -> bool {
    match node {
        Node::Leaf { mask, slot, .. } => {
            let p = pat(table, node);
            let found = memo.term_hit(paper, p, *mask, *slot as usize);
            if excluded {
                if found {
                    ex_hits.push(p.raw.clone());
                }
            } else if found {
                hits.push((p.raw.clone(), *mask));
            } else {
                misses.push((p.raw.clone(), *mask));
            }
            found
        }
        Node::Field { fields: fs, child } => {
            rec_fields(child, field_mask_from_strings(fs), excluded, paper, table, memo, hits, misses, ex_hits)
        }
        Node::Not { child } => {
            !rec_fields(child, mask, !excluded, paper, table, memo, hits, misses, ex_hits)
        }
        Node::Group { op, children } => {
            // Accumulate without short-circuiting so every leaf is still
            // reported as a hit/miss/excluded term.
            if op == "OR" {
                let mut acc = false;
                for c in children {
                    acc |= rec_fields(c, mask, excluded, paper, table, memo, hits, misses, ex_hits);
                }
                acc
            } else {
                let mut acc = true;
                for c in children {
                    acc &= rec_fields(c, mask, excluded, paper, table, memo, hits, misses, ex_hits);
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
    mask: u8,
    excluded: bool,
    paper: &Paper,
    table: &[Pattern],
    memo: &mut Memo,
    out: &mut BlockScan,
) -> bool {
    match node {
        Node::Leaf { mask, slot, .. } => {
            let p = pat(table, node);
            let found = memo.term_hit(paper, p, *mask, *slot as usize);
            if excluded {
                if found {
                    out.excluded_hits.push(p.raw.clone());
                }
            } else if found {
                out.hits.push(p.raw.clone());
            } else {
                out.misses.push(p.raw.clone());
            }
            found
        }
        Node::Field { fields: fs, child } => rec(child, field_mask_from_strings(fs), excluded, paper, table, memo, out),
        Node::Not { child } => !rec(child, mask, !excluded, paper, table, memo, out),
        Node::Group { op, children } => {
            // Accumulate without short-circuiting so every leaf is still
            // reported in `out`.
            if op == "OR" {
                let mut acc = false;
                for c in children {
                    acc |= rec(c, mask, excluded, paper, table, memo, out);
                }
                acc
            } else {
                let mut acc = true;
                for c in children {
                    acc &= rec(c, mask, excluded, paper, table, memo, out);
                }
                acc
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
