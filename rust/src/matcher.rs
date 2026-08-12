//! Paper matching: evaluate the AST against a paper, with SIMD pattern search.

use crate::ast::Node;
use crate::paper::{Paper, ALL_FIELDS, F_ANY};
use crate::simd::find;
use std::collections::HashMap;
use std::sync::Arc;

/// Dev-only counting instrumentation and experiment toggles (feature `prof`).
#[cfg(feature = "prof")]
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub static INDEX_BUILDS: AtomicU64 = AtomicU64::new(0);
    pub static INDEX_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static COULD_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static COULD_PARTS: AtomicU64 = AtomicU64::new(0);
    pub static MATCHES_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static TERM_COMPUTES: AtomicU64 = AtomicU64::new(0);
    pub static TERM_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
    pub static LEAF_EVALS: AtomicU64 = AtomicU64::new(0);
    pub static REPORT_PUSHES: AtomicU64 = AtomicU64::new(0);

    fn flag(name: &str) -> bool {
        static CACHE: OnceLock<Vec<(&'static str, bool)>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| {
            vec![
                ("PROF_SKIP_FILTER", std::env::var("PROF_SKIP_FILTER").is_ok()),
                ("PROF_SKIP_FIND", std::env::var("PROF_SKIP_FIND").is_ok()),
                ("PROF_SKIP_REPORT", std::env::var("PROF_SKIP_REPORT").is_ok()),
            ]
        });
        cache.iter().any(|(n, v)| *n == name && *v)
    }
    /// Skip the TextIndex pre-filter (always run the SIMD search).
    pub fn skip_filter() -> bool {
        flag("PROF_SKIP_FILTER")
    }
    /// Make `matches` always succeed (isolate pre-filter + traversal cost).
    pub fn skip_find() -> bool {
        flag("PROF_SKIP_FIND")
    }
    /// Skip materializing hits/misses/excluded lists (verdicts still computed).
    pub fn skip_report() -> bool {
        flag("PROF_SKIP_REPORT")
    }
    pub fn reset() {
        for c in [
            &INDEX_BUILDS, &INDEX_BYTES, &COULD_CALLS, &COULD_PARTS, &MATCHES_CALLS,
            &TERM_COMPUTES, &TERM_CACHE_HITS, &LEAF_EVALS, &REPORT_PUSHES,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

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
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::INDEX_BUILDS.fetch_add(1, Ordering::Relaxed);
            prof::INDEX_BYTES.fetch_add(text.len() as u64, Ordering::Relaxed);
        }
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
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::COULD_CALLS.fetch_add(1, Ordering::Relaxed);
            prof::COULD_PARTS.fetch_add(part.len() as u64, Ordering::Relaxed);
        }
        match part.len() {
            0 => true,
            1 => self.bytes[part[0] as usize],
            2..=3 => {
                let a = part[0] as usize;
                let b = part[1] as usize;
                (self.bigrams[(a << 2) | (b >> 6)] >> (b & 63)) & 1 != 0
            }
            _ => {
                // Check the FIRST and LAST 4-byte windows only. A part can
                // occur only where its first quad occurs (hard necessary
                // condition, no false negatives), and requiring the last
                // quad too restores the all-quads rejection power at a
                // fraction of the cost: the per-quad bloom false-positive
                // rate is ~0.06%, so two quads cut SIMD runs from ~4500 to
                // ~400 per request (measured on the SDG corpus, 2026-08)
                // for just 4 hash ops per part vs ~26 for all quads.
                let first = u32::from_le_bytes([part[0], part[1], part[2], part[3]]);
                let (h1, h2) = bloom_hashes(first, self.quad_mask);
                if ((self.quads[(h1 >> 6) as usize] >> (h1 & 63)) & 1) == 0
                    || ((self.quads[(h2 >> 6) as usize] >> (h2 & 63)) & 1) == 0
                {
                    return false;
                }
                let n = part.len();
                let last = u32::from_le_bytes([part[n - 4], part[n - 3], part[n - 2], part[n - 1]]);
                let (h1, h2) = bloom_hashes(last, self.quad_mask);
                ((self.quads[(h1 >> 6) as usize] >> (h1 & 63)) & 1) != 0
                    && ((self.quads[(h2 >> 6) as usize] >> (h2 & 63)) & 1) != 0
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
///
/// The slot space is GLOBAL across every `resolve_blocks` call: pass a shared
/// `&mut u32` counter and increment it as new `(pid, mask)` pairs are seen.
/// Callers that scan multiple queries with a single shared `Memo` MUST use one
/// counter for the whole loop (web.rs, main.rs) — otherwise leaves in different
/// queries get colliding slot ids and share memo entries, corrupting results.
///
/// Returns the total number of distinct `(pid, mask)` slots assigned so far
/// (after this call).
pub fn resolve_blocks(blocks: &mut [Node], table: &[Pattern], nslots: &mut u32) -> usize {
    let mut map: HashMap<&str, usize> = HashMap::with_capacity(table.len());
    for (i, p) in table.iter().enumerate() {
        map.insert(p.raw.as_ref(), i);
    }
    // Slot ids: dense over the seen (pid, mask) pairs, assigned in first-seen
    // order. `pid` is already dense over the table, so keying the memo array
    // on `pid` alone would be too coarse (the same pattern under different
    // field masks can give different results) — hence a (pid, mask) slot.
    // `slot_of` is local per call, but `nslots` is the GLOBAL running counter
    // so slots never collide across queries that share one Memo.
    let mut slot_of: HashMap<(u32, u8), u32> = HashMap::new();
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
        assign(b, default_mask, &map, &mut slot_of, &mut *nslots);
    }
    *nslots as usize
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
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::MATCHES_CALLS.fetch_add(1, Ordering::Relaxed);
        }
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

/// Per-request memo of term results, keyed by the leaf's precomputed dense
/// `slot` (see `resolve_blocks`): `terms[slot]` is 0 (unset), 1 (false) or
/// 2 (true). Slots are dense over the live `(pattern, mask)` pairs, so the
/// hot path is a single `Vec<u8>` read/write instead of a hashed
/// `(pattern*, mask)` lookup.
///
/// Buffers are also resolved once per request: the effective text of each
/// field (its section, or the full text when the section is missing) is
/// deduplicated by identity into a dense id space (<= 5 ids), and each
/// distinct field mask is FOLDED into a single joined buffer (fields
/// concatenated with '\n' separators) on first use. Every term is then
/// searched once per mask against one buffer instead of up to four. The
/// join separator is not a word character and is whitespace, so phrase,
/// wildcard-`*` and boundary checks behave exactly as per-field searches;
/// `?`-glob patterns run per segment so they can never match across fields.
pub struct Memo<'a> {
    terms: Vec<u8>,
    bufs: [&'a [u8]; 5],
    field_buf: [u8; 4],
    full_id: u8,
    full_covers_sections: bool,
    full_segs: [(usize, usize); 4],
    joined: Vec<JoinedEntry>,
    /// O(1) mask -> joined-buffer index (i32 for a -1 "unset" sentinel).
    mask_cache: [i32; 256],
}

/// A mask's folded buffer plus its per-field segments (for `?` globs) and a
/// lazily built `TextIndex` pre-filter.
struct JoinedEntry {
    buf: Vec<u8>,
    idx: Option<TextIndex>,
    nsegs: u8,
    segs: [(usize, usize); 5],
}

/// Append `t` to `bufs` if no identical (pointer+length) buffer is present;
/// returns the deduplicated id of `t`.
fn push_dedup<'b>(bufs: &mut [&'b [u8]; 5], nb: &mut usize, t: &'b [u8]) -> u8 {
    for (k, b) in bufs[..*nb].iter().enumerate() {
        if b.as_ptr() == t.as_ptr() && b.len() == t.len() {
            return k as u8;
        }
    }
    bufs[*nb] = t;
    *nb += 1;
    (*nb - 1) as u8
}

impl<'a> Memo<'a> {
    /// `nslots` is the global slot count returned by `resolve_blocks` (0 if
    /// unknown; the terms vec then grows on demand).
    pub fn new(paper: &'a Paper, nslots: u32) -> Memo<'a> {
        let mut bufs: [&'a [u8]; 5] = [&[]; 5];
        let mut nb = 0usize;
        let mut field_buf = [0u8; 4];
        let full_id = push_dedup(&mut bufs, &mut nb, paper.text_lower(F_ANY));
        for f in 0..4u8 {
            field_buf[f as usize] = push_dedup(&mut bufs, &mut nb, paper.text_lower(f));
        }
        Memo {
            terms: vec![0; nslots as usize],
            bufs,
            field_buf,
            full_id,
            full_covers_sections: paper.full_covers_sections,
            full_segs: paper.full_section_ranges(),
            joined: Vec::new(),
            mask_cache: [-1; 256],
        }
    }

    /// Get (building on first use) the folded buffer for `mask`. Returns its
    /// index in `self.joined`.
    fn joined_for(&mut self, mask: u8) -> usize {
        let cached = self.mask_cache[mask as usize];
        if cached >= 0 {
            return cached as usize;
        }
        // Decode the mask into deduplicated buffer ids, then concatenate.
        let mut ids = [0u8; 5];
        let mut n = 0usize;
        for f in 0..4u8 {
            if mask & (1 << f) != 0 {
                let id = self.field_buf[f as usize];
                if !ids[..n].contains(&id) {
                    ids[n] = id;
                    n += 1;
                }
            }
        }
        if mask & (1 << (F_ANY & 7)) != 0 {
            let id = self.full_id;
            if !ids[..n].contains(&id) {
                ids[n] = id;
                n += 1;
            }
        }
        if self.full_covers_sections && (mask & 0x0F) == 0x0F {
            // Default TITLE-ABS-KEY: the full text IS the sections' join, so
            // use it directly (one buffer instead of four, ~2x fewer bytes).
            // Its per-section ranges make '?'-globs keep per-field semantics.
            let full = self.bufs[self.full_id as usize];
            let mut segs = [(0usize, 0usize); 5];
            let mut nsegs = 0usize;
            for (s, e) in self.full_segs.iter() {
                if s != e {
                    segs[nsegs] = (*s, *e);
                    nsegs += 1;
                }
            }
            self.joined.push(JoinedEntry {
                buf: full.to_vec(),
                idx: None,
                nsegs: nsegs as u8,
                segs,
            });
            let idx = self.joined.len() - 1;
            self.mask_cache[mask as usize] = idx as i32;
            return idx;
        }
        let mut cap = 0usize;
        for k in 0..n {
            cap += self.bufs[ids[k] as usize].len();
        }
        let mut buf = Vec::with_capacity(cap + n);
        let mut segs = [(0usize, 0usize); 5];
        let mut nsegs = 0usize;
        for k in 0..n {
            let b = self.bufs[ids[k] as usize];
            if k > 0 {
                buf.push(b'\n');
            }
            let start = buf.len();
            buf.extend_from_slice(b);
            segs[nsegs] = (start, buf.len());
            nsegs += 1;
        }
        self.joined.push(JoinedEntry { buf, idx: None, nsegs: nsegs as u8, segs });
        let idx = self.joined.len() - 1;
        self.mask_cache[mask as usize] = idx as i32;
        idx
    }

    /// Evaluate `pat` under field `mask`, memoized at dense `slot`. The mask
    /// and slot are both precomputed and stamped on the leaf by
    /// `resolve_blocks`, so the caller passes them straight through.
    fn term_hit(&mut self, pat: &Pattern, mask: u8, slot: usize) -> bool {
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::LEAF_EVALS.fetch_add(1, Ordering::Relaxed);
        }
        // 0 = unset. A slot present but still 0 is NOT a cached verdict, so
        // recompute (slots are global across all blocks, and traversal may
        // visit a high slot before a lower one).
        if slot < self.terms.len() {
            match self.terms[slot] {
                1 => {
                    #[cfg(feature = "prof")]
                    {
                        use std::sync::atomic::Ordering;
                        prof::TERM_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                    }
                    return false;
                }
                2 => {
                    #[cfg(feature = "prof")]
                    {
                        use std::sync::atomic::Ordering;
                        prof::TERM_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                    }
                    return true;
                }
                _ => {}
            }
        }
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::TERM_COMPUTES.fetch_add(1, Ordering::Relaxed);
        }
        let v = self.compute(pat, mask);
        if slot >= self.terms.len() {
            self.terms.resize(slot + 1, 0);
        }
        self.terms[slot] = if v { 2 } else { 1 };
        v
    }

    /// Actual search for `pat` under `mask` (uncached), against the mask's
    /// single folded buffer. `?`-glob patterns (no literal parts) run per
    /// segment so a `?` can never consume the join separator; everything
    /// else uses the TextIndex pre-filter then one SIMD search.
    fn compute(&mut self, pat: &Pattern, mask: u8) -> bool {
        // A zero mask means "no field scoping" -> the default TITLE-ABS-KEY.
        let mask = if mask == 0 { field_mask(&ALL_FIELDS) } else { mask };
        let jidx = self.joined_for(mask);
        if pat.parts.is_empty() {
            // '?' glob: run per segment so patterns cannot span fields.
            let j = &self.joined[jidx];
            if j.nsegs > 1 {
                for s in &j.segs[..j.nsegs as usize] {
                    if glob_match(&pat.lower_raw, &j.buf[s.0..s.1]) {
                        return true;
                    }
                }
                return false;
            }
            return glob_match(&pat.lower_raw, &j.buf);
        }
        let entry = &mut self.joined[jidx];
        if entry.idx.is_none() {
            entry.idx = Some(TextIndex::build(&entry.buf));
        }
        let idx = entry.idx.as_ref().unwrap();
        // Filter first: if a literal part of the pattern is absent from the
        // buffer, the SIMD search is guaranteed to fail.
        #[cfg(feature = "prof")]
        let filter = !prof::skip_filter() && pat.could_match(idx);
        #[cfg(not(feature = "prof"))]
        let filter = pat.could_match(idx);
        // SIMD search runs only when the pre-filter passes (short-circuit).
        #[cfg(feature = "prof")]
        if filter && (prof::skip_find() || pat.matches(&entry.buf)) {
            return true;
        }
        #[cfg(not(feature = "prof"))]
        if filter && pat.matches(&entry.buf) {
            return true;
        }
        false
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
    let mut memo = Memo::new(paper, 0);
    eval_memo(node, field_mask(fields.unwrap_or(&ALL_FIELDS)), table, &mut memo)
}

fn eval_memo(node: &Node, mask: u8, table: &[Pattern], memo: &mut Memo) -> bool {
    match node {
        Node::Leaf { mask, slot, .. } => {
            let p = pat(table, node);
            memo.term_hit(p, *mask, *slot as usize)
        }
        Node::Field { fields: fs, child } => eval_memo(child, field_mask_from_strings(fs), table, memo),
        Node::Not { child } => !eval_memo(child, mask, table, memo),
        Node::Group { op, children } => {
            if op == "OR" {
                children.iter().any(|c| eval_memo(c, mask, table, memo))
            } else {
                children.iter().all(|c| eval_memo(c, mask, table, memo))
            }
        }
    }
}

pub struct BlockScan {
    pub hits: Vec<Arc<str>>,
    pub misses: Vec<Arc<str>>,
    pub excluded_hits: Vec<Arc<str>>,
}

// ---------------------------------------------------------------------------
// Flattened blocks
//
// A request re-walks the AST of every block and re-dispatches per node
// (~93k leaf evals + ~60k group/field/not nodes per paper). Flattening each
// block ONCE at boot into a postfix program over leaf indices plus a flat
// leaf list turns the hot path into two linear loops (program eval + leaf
// classification) with no tree dispatch.
// ---------------------------------------------------------------------------

/// Postfix operator over leaf indices (`Push(i)` evaluates `leaves[i]`).
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Push(u32),
    /// Push a constant (identity of an empty group: AND -> true, OR -> false).
    True,
    False,
    Not,
    And,
    Or,
}

/// One keyword occurrence inside a flattened block.
#[derive(Clone, Debug)]
pub struct LeafDesc {
    pub pid: u32,
    pub slot: u32,
    pub mask: u8,
    pub excluded: bool,
    pub raw: Arc<str>,
}

/// A block compiled to a postfix program + flat leaf list. Built once at
/// boot, AFTER `resolve_blocks` has stamped slots onto the AST.
pub struct FlatBlock {
    pub prog: Vec<Op>,
    pub leaves: Vec<LeafDesc>,
}

/// Flatten one block (call AFTER `resolve_blocks`). Leaf order and exclusion
/// parity match the AST traversal exactly, so `scan_flat` produces the same
/// hits/misses/excluded lists (including duplicates) as the tree walk.
pub fn flatten_block(block: &Node, table: &[Pattern]) -> FlatBlock {
    fn emit(node: &Node, excluded: bool, table: &[Pattern], fb: &mut FlatBlock) {
        match node {
            Node::Leaf { pid, mask, slot, .. } => {
                let i = fb.leaves.len() as u32;
                fb.leaves.push(LeafDesc {
                    pid: *pid,
                    slot: *slot,
                    mask: *mask,
                    excluded,
                    raw: table[*pid as usize].raw.clone(),
                });
                fb.prog.push(Op::Push(i));
            }
            Node::Field { child, .. } => emit(child, excluded, table, fb),
            Node::Not { child } => {
                emit(child, !excluded, table, fb);
                fb.prog.push(Op::Not);
            }
            Node::Group { op, children } => {
                // A group with k children needs k-1 binary ops; fold eagerly
                // (op after every child past the first) so the eval stack
                // stays ~2 deep instead of k deep. AND/OR are associative,
                // so the fold order does not matter. A 1-child group
                // evaluates to the child itself (no extra op, since `And`
                // would pop an empty stack -> false), and an empty group
                // pushes its identity (AND -> true, OR -> false).
                match children.len() {
                    0 => fb.prog.push(if op == "OR" { Op::False } else { Op::True }),
                    1 => emit(&children[0], excluded, table, fb),
                    _ => {
                        let opcode = if op == "OR" { Op::Or } else { Op::And };
                        let mut it = children.iter();
                        emit(it.next().unwrap(), excluded, table, fb);
                        for c in it {
                            emit(c, excluded, table, fb);
                            fb.prog.push(opcode);
                        }
                    }
                }
            }
        }
    }
    let mut fb = FlatBlock { prog: Vec::new(), leaves: Vec::new() };
    emit(block, false, table, &mut fb);
    fb
}

/// Scan a flattened block: evaluate the postfix program for the verdict and
/// classify every leaf occurrence. Same output contract as `scan_with_fields`
/// (hits/misses/excluded lists with per-leaf field masks, duplicates kept),
/// except keywords are borrowed from the `FlatBlock` (which must outlive the
/// returned lists) instead of cloned `Arc`s.
pub fn scan_flat<'a, 'b>(
    flat: &'b FlatBlock,
    table: &[Pattern],
    memo: &mut Memo<'a>,
) -> (Vec<(&'b str, u8)>, Vec<(&'b str, u8)>, Vec<&'b str>, bool) {
    let mut hits = Vec::new();
    let mut misses = Vec::new();
    let mut ex_hits = Vec::new();
    let matched = scan_flat_into(flat, table, memo, &mut hits, &mut misses, &mut ex_hits);
    (hits, misses, ex_hits, matched)
}

/// `scan_flat` writing into caller-owned vectors, which may be reused across
/// blocks (clear + retain capacity between calls).
pub fn scan_flat_into<'a, 'b>(
    flat: &'b FlatBlock,
    table: &[Pattern],
    memo: &mut Memo<'a>,
    hits: &mut Vec<(&'b str, u8)>,
    misses: &mut Vec<(&'b str, u8)>,
    ex_hits: &mut Vec<&'b str>,
) -> bool {
    // Single pass: `Push(i)` evaluates leaf i (memoized per slot) AND
    // classifies it immediately - Push order is leaf order, so hits/misses/
    // excluded keep the AST traversal order. The stack evaluates the
    // boolean program; the verdict is the final stack value.
    let mut stack: Vec<bool> = Vec::with_capacity(8);
    for op in &flat.prog {
        match *op {
            Op::Push(i) => {
                let l = &flat.leaves[i as usize];
                let v = memo.term_hit(&table[l.pid as usize], l.mask, l.slot as usize);
                if l.excluded {
                    if v {
                        ex_hits.push(&*l.raw);
                    }
                } else if v {
                    hits.push((&*l.raw, l.mask));
                } else {
                    misses.push((&*l.raw, l.mask));
                }
                stack.push(v);
            }
            Op::True => stack.push(true),
            Op::False => stack.push(false),
            Op::Not => {
                let t = stack.pop().unwrap_or(false);
                stack.push(!t);
            }
            Op::And => {
                let b = stack.pop().unwrap_or(false);
                let a = stack.pop().unwrap_or(false);
                stack.push(a && b);
            }
            Op::Or => {
                let b = stack.pop().unwrap_or(false);
                let a = stack.pop().unwrap_or(false);
                stack.push(a || b);
            }
        }
    }
    stack.pop().unwrap_or(false)
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
    let mut memo = Memo::new(paper, 0);
    scan_block_shared(block, paper, table, &mut memo).0
}

/// `scan_block` with a caller-owned `Memo` (share it across all blocks of a
/// request so each distinct (pattern, field-mask) is searched once) and the
/// block's boolean verdict computed in the same single traversal.
pub fn scan_block_shared<'a>(
    block: &Node,
    _paper: &'a Paper,
    table: &[Pattern],
    memo: &mut Memo<'a>,
) -> (BlockScan, bool) {
    let mut out = BlockScan { hits: Vec::new(), misses: Vec::new(), excluded_hits: Vec::new() };
    let matched = rec(block, field_mask(&ALL_FIELDS), false, table, memo, &mut out);
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
pub fn scan_with_fields<'a>(
    block: &Node,
    _paper: &'a Paper,
    table: &[Pattern],
    memo: &mut Memo<'a>,
) -> (Vec<(Arc<str>, u8)>, Vec<(Arc<str>, u8)>, Vec<Arc<str>>, bool) {
    let mut hits = Vec::new();
    let mut misses = Vec::new();
    let mut ex_hits = Vec::new();
    let matched = rec_fields(block, field_mask(&ALL_FIELDS), false, table, memo, &mut hits, &mut misses, &mut ex_hits);
    (hits, misses, ex_hits, matched)
}

fn rec_fields(
    node: &Node,
    mask: u8,
    excluded: bool,
    table: &[Pattern],
    memo: &mut Memo,
    hits: &mut Vec<(Arc<str>, u8)>,
    misses: &mut Vec<(Arc<str>, u8)>,
    ex_hits: &mut Vec<Arc<str>>,
) -> bool {
    match node {
        Node::Leaf { mask, slot, .. } => {
            let p = pat(table, node);
            let found = memo.term_hit(p, *mask, *slot as usize);
            #[cfg(feature = "prof")]
            let report = !prof::skip_report();
            #[cfg(not(feature = "prof"))]
            let report = true;
            if excluded {
                if found {
                    #[cfg(feature = "prof")]
                    {
                        use std::sync::atomic::Ordering;
                        prof::REPORT_PUSHES.fetch_add(1, Ordering::Relaxed);
                    }
                    if report {
                        ex_hits.push(p.raw.clone());
                    }
                }
            } else if found {
                #[cfg(feature = "prof")]
                {
                    use std::sync::atomic::Ordering;
                    prof::REPORT_PUSHES.fetch_add(1, Ordering::Relaxed);
                }
                if report {
                    hits.push((p.raw.clone(), *mask));
                }
            } else {
                #[cfg(feature = "prof")]
                {
                    use std::sync::atomic::Ordering;
                    prof::REPORT_PUSHES.fetch_add(1, Ordering::Relaxed);
                }
                if report {
                    misses.push((p.raw.clone(), *mask));
                }
            }
            found
        }
        Node::Field { fields: fs, child } => {
            rec_fields(child, field_mask_from_strings(fs), excluded, table, memo, hits, misses, ex_hits)
        }
        Node::Not { child } => {
            !rec_fields(child, mask, !excluded, table, memo, hits, misses, ex_hits)
        }
        Node::Group { op, children } => {
            // Accumulate without short-circuiting so every leaf is still
            // reported as a hit/miss/excluded term.
            if op == "OR" {
                let mut acc = false;
                for c in children {
                    acc |= rec_fields(c, mask, excluded, table, memo, hits, misses, ex_hits);
                }
                acc
            } else {
                let mut acc = true;
                for c in children {
                    acc &= rec_fields(c, mask, excluded, table, memo, hits, misses, ex_hits);
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
    table: &[Pattern],
    memo: &mut Memo,
    out: &mut BlockScan,
) -> bool {
    match node {
        Node::Leaf { mask, slot, .. } => {
            let p = pat(table, node);
            let found = memo.term_hit(p, *mask, *slot as usize);
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
        Node::Field { fields: fs, child } => rec(child, field_mask_from_strings(fs), excluded, table, memo, out),
        Node::Not { child } => !rec(child, mask, !excluded, table, memo, out),
        Node::Group { op, children } => {
            // Accumulate without short-circuiting so every leaf is still
            // reported in `out`.
            if op == "OR" {
                let mut acc = false;
                for c in children {
                    acc |= rec(c, mask, excluded, table, memo, out);
                }
                acc
            } else {
                let mut acc = true;
                for c in children {
                    acc &= rec(c, mask, excluded, table, memo, out);
                }
                acc
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flattened-block path must produce byte-identical reports (verdict,
    /// hits, misses, excluded) to the AST tree walk, for every block of every
    /// SDG query and every paper in the repo.
    #[test]
    fn flat_matches_ast_scan() {
        use crate::query::load_queries;
        use std::path::Path;

        let qdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/data/queries");
        let mut queries = load_queries(&qdir).unwrap();
        let table = compile_all(queries.iter().flat_map(|q| q.blocks.iter()));
        let mut nslots = 0u32;
        for q in &mut queries {
            resolve_blocks(&mut q.blocks, &table, &mut nslots);
        }
        let flats: Vec<Vec<FlatBlock>> = queries
            .iter()
            .map(|q| q.blocks.iter().map(|b| flatten_block(b, &table)).collect())
            .collect();

        for p in ["sample_paper.md", "besley_persson_2014.md", "hughes_2003_coral.md"] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../papers/{p}"));
            let paper = Paper::from_text(&std::fs::read_to_string(&path).unwrap());
            let mut memo = Memo::new(&paper, 0);
            for (qi, q) in queries.iter().enumerate() {
                for (bi, b) in q.blocks.iter().enumerate() {
                    let ast = scan_with_fields(b, &paper, &table, &mut memo);
                    let flat = scan_flat(&flats[qi][bi], &table, &mut memo);
                    let owned = |v: Vec<(&str, u8)>| v.into_iter().map(|(s, m)| (s.to_owned(), m)).collect::<Vec<_>>();
                    let owned_ast = |v: Vec<(Arc<str>, u8)>| v.into_iter().map(|(s, m)| (s.to_string(), m)).collect::<Vec<_>>();
                    assert_eq!(flat.3, ast.3, "verdict mismatch {p} q{qi} b{bi}");
                    assert_eq!(owned(flat.0), owned_ast(ast.0), "hits mismatch {p} q{qi} b{bi}");
                    assert_eq!(owned(flat.1), owned_ast(ast.1), "misses mismatch {p} q{qi} b{bi}");
                    assert_eq!(
                        flat.2.into_iter().map(String::from).collect::<Vec<_>>(),
                        ast.2.into_iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                        "excluded mismatch {p} q{qi} b{bi}"
                    );
                }
            }
        }
    }

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
