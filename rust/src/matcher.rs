//! Paper matching: evaluate the AST against a paper, with SIMD pattern search.

use crate::ast::Node;
use crate::paper::{Paper, ALL_FIELDS, F_ANY};
use crate::simd::find;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

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

/// The process-wide string blob. Every `Pattern`/`LeafDesc`/`SdgDict`
/// record stores (offset, len) pairs into this blob. At boot the blob is
/// either the compiled+leaked keyword data or the mmap'd cache region, so
/// the hot path reads strings straight out of mapped memory (zero copy).
/// Stored as atomics so tests can replace it and the hot path reads it
/// without locking.
static BLOB_PTR: std::sync::atomic::AtomicPtr<u8> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
static BLOB_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Current blob (for the boot-cache writer to persist it).
pub fn blob_slice() -> &'static [u8] {
    blob()
}

/// Install the string blob (boot: compiled data or the mmap'd cache).
pub fn set_blob(b: &'static [u8]) {
    use std::sync::atomic::Ordering;
    BLOB_PTR.store(b.as_ptr() as *mut u8, Ordering::Relaxed);
    BLOB_LEN.store(b.len(), Ordering::Relaxed);
}

#[inline]
fn blob() -> &'static [u8] {
    use std::sync::atomic::Ordering;
    let p = BLOB_PTR.load(Ordering::Relaxed);
    if p.is_null() {
        return &[];
    }
    let len = BLOB_LEN.load(Ordering::Relaxed);
    unsafe { std::slice::from_raw_parts(p, len) }
}

/// A keyword pattern compiled to blob-offset records. `parts` is a table of
/// (offset, len) u32 pairs inside the blob (empty for '?'-glob terms).
/// `#[repr(C)]` + fixed size: the boot cache stores the raw records and the
/// runtime can view the mmap'd region as `&[Pattern]` directly.
#[repr(C)]
pub struct Pattern {
    raw_off: u32,
    raw_len: u32,
    lower_off: u32,
    lower_len: u32,
    parts_off: u32,
    parts_len: u32, // number of parts (u32 pairs at parts_off)
    no_wildcard: bool,
}

impl Pattern {
    /// The keyword as written in the query file.
    #[inline]
    pub fn raw(&self) -> &'static str {
        let b = blob();
        unsafe { std::str::from_utf8_unchecked(&b[self.raw_off as usize..(self.raw_off + self.raw_len) as usize]) }
    }

    /// Full lowercased text (only '?'-glob patterns keep it).
    #[inline]
    fn lower_raw(&self) -> &'static [u8] {
        let b = blob();
        &b[self.lower_off as usize..(self.lower_off + self.lower_len) as usize]
    }

    #[inline]
    fn n_parts(&self) -> usize {
        self.parts_len as usize
    }

    #[inline]
    fn part(&self, i: usize) -> &'static [u8] {
        let b = blob();
        let t = self.parts_off as usize + i * 8;
        let off = u32::from_ne_bytes([b[t], b[t + 1], b[t + 2], b[t + 3]]) as usize;
        let len = u32::from_ne_bytes([b[t + 4], b[t + 5], b[t + 6], b[t + 7]]) as usize;
        &b[off..off + len]
    }

    fn parts(&self) -> impl Iterator<Item = &'static [u8]> + '_ {
        (0..self.n_parts()).map(move |i| self.part(i))
    }

    pub fn no_wildcard(&self) -> bool {
        self.no_wildcard
    }

    /// Blob offsets of the keyword text (for LeafDesc records).
    #[inline]
    pub fn raw_off_len(&self) -> (u32, u32) {
        (self.raw_off, self.raw_len)
    }
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
/// One segment's quad Bloom: `mask` is the segment-local power-of-two
/// capacity minus one and `bits` the bit array. Sized from the CPU cache
/// spec so chunk text + Bloom + dense arrays fit the detected cache level
/// (see `chunk_plan`); random Bloom writes then hit cache instead of DRAM.
struct QuadSeg {
    bits: Vec<u64>,
    mask: u32,
}

pub struct TextIndex {
    bytes: [bool; 256],
    bigrams: [u64; 1024],
    /// Segment-local quad Bloom tables. Each parallel chunk hashes only its
    /// own windows into its own power-of-two table sized by the CPU cache
    /// spec (chunk bytes, `cpu::best()`), so total Bloom memory is ~1 bit
    /// per text byte with NO per-chunk duplication of a full-size table and
    /// every worker's writes stay L2/L3 resident (the old layout allocated
    /// `ceil(n/chunk) x n` bytes of Bloom tables and OR-merged them all).
    /// A pattern quad is "present" when ANY segment holds both of its Bloom
    /// bits; every window's start belongs to exactly one chunk, so an
    /// occurrence whose internal windows straddle a boundary is still found
    /// (its windows live in the neighbour segment) - no false negatives.
    quads: Vec<QuadSeg>,
    /// quad -> byte positions, recorded only for pattern first-quads
    /// (`FIRST_QUADS`). A missing entry means the quad is absent from the
    /// text: an exact, false-negative-free gate that lets pattern search
    /// verify candidate starts instead of scanning the whole buffer.
    pos: FastMap<u32, Vec<u32>>,
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

/// Fast non-cryptographic hasher for the quad-position map and the
/// pattern-first-quad set. Keys are compile-time/derived u32s, never
/// adversarial input, so a multiplicative hash is safe (std's SipHash is
/// ~10x slower on the per-window positions pass).
#[derive(Clone, Copy)]
pub struct FastHasher(u64);

impl Default for FastHasher {
    fn default() -> Self {
        FastHasher(0xcbf2_9ce4_8422_2325)
    }
}

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

pub type FastMap<K, V> = HashMap<K, V, FastHasher>;

fn new_fast_map<K, V>() -> FastMap<K, V> {
    HashMap::with_capacity_and_hasher(16, FastHasher::default())
}

/// First-4-bytes of every literal pattern part (len >= 4), unioned across
/// every `compile_all` call. TextIndexes record byte positions only for
/// these quads, so pattern search verifies a handful of candidate starts
/// instead of scanning the whole buffer (the bloom-only prefilter used to
/// fall through to a full SIMD scan per term, which on repetitive text
/// where every quad is present made scan cost ~memory-bandwidth-bound:
/// measured 570 ms on a 1.6 MB repeated paper vs ~10 ms with positions). A
/// quad's absence from the recorded positions is a hard no-match, never a
/// fallback to scan.
static FIRST_QUADS: Mutex<Option<Arc<HashSet<u32, FastHasher>>>> = Mutex::new(None);

/// Shared snapshot of the pattern first-quads for one index build (the
/// positions pass does a contains per window; the Arc avoids holding the
/// global lock and cloning ~25k entries per request).
fn first_quads() -> Arc<HashSet<u32, FastHasher>> {
    let mut g = FIRST_QUADS.lock().unwrap();
    g.get_or_insert_with(|| Arc::new(HashSet::with_hasher(FastHasher::default()))).clone()
}


/// CPU-spec-driven chunk plan for the parallel TextIndex build.
///
/// The detected cache geometry (cpu.rs, once per boot) decides everything:
///   - texts whose whole working set fits ~L2/3 are built serially (thread
///     split + merge would only add overhead),
///   - larger texts are split so each worker's slice + its segment-local
///     Bloom + dense arrays stay L2-resident (chunk ~= L2/3),
///   - the number of chunks never exceeds the detected core count.
/// Chunk boundaries are rounded to the detected cache line.
fn chunk_plan(n: usize) -> Vec<(usize, usize)> {
    let spec = crate::cpu::best();
    let line = spec.cache_line.max(16);
    let serial_limit = (spec.l2 / 3).max(32 * 1024); // fits cache: no threads
    if n <= serial_limit {
        return vec![(0, n)];
    }
    let ideal = (spec.l2 / 3).clamp(line * 64, 1 << 20); // chunk + bloom ~ L2
    let cores = spec.cores.max(1);
    let need = n.div_ceil(ideal);
    let workers = need.min(cores);
    let chunk = (n.div_ceil(workers)).max(line);
    let mut out = Vec::with_capacity(workers);
    let mut base = 0usize;
    while base < n {
        let end = (base + chunk).min(n);
        out.push((base, end));
        base = end;
    }
    out
}

/// Partial index of one text slice (dense arrays are OR-merged, position
/// lists are concatenated in ascending slice order; the quad Bloom needs no
/// merge - each slice owns its segment).
struct BuildPart {
    bytes: [bool; 256],
    bigrams: [u64; 1024],
    quads: QuadSeg,
    pos: FastMap<u32, Vec<u32>>,
}

fn build_serial(text: &[u8], needed: &HashSet<u32, FastHasher>) -> TextIndex {
    let n = text.len();
    merge_parts(vec![build_chunk(text, needed, 0, n)], n)
}

/// Index the slice `text[base..end]` (reads up to `end + 3` bytes so 2- and
/// 4-byte windows straddling the chunk boundary are still seen; the bytes
/// recorded and the windows hashed into THIS segment are exactly those whose
/// start lies in `[base, end)`, i.e. every window is owned by exactly one
/// chunk). The Bloom is sized from the windows owned by this chunk, so its
/// bitset is ~chunk bytes - L2 sized by construction (`chunk_plan`).
fn build_chunk(
    text: &[u8],
    needed: &HashSet<u32, FastHasher>,
    base: usize,
    end: usize,
) -> BuildPart {
    let n = text.len();
    let chunk_len = end - base;
    let slice = &text[base..(end + 3).min(n)];
    let slen = slice.len();

    let mut bytes = [false; 256];
    let mut bigrams = [0u64; 1024];
    // windows hashed into this segment: starts i with i < chunk_len and
    // i + 3 < slen  ->  at most chunk_len of them
    let wc = chunk_len.min(slen.saturating_sub(3));
    let qbits = (wc.saturating_mul(8)).next_power_of_two().max(1 << 16);
    let quad_mask = (qbits as u32) - 1;
    let mut quads = vec![0u64; (qbits >> 6) as usize];
    let mut pos: FastMap<u32, Vec<u32>> = new_fast_map();

    // Optional 64K-bit direct-mapped filter over the needed quads: one load
    // + shift per window; the exact HashSet probe runs only on hits (a few
    // per thousand windows). No false negatives.
    let nf: Option<[u64; 1024]> = if !needed.is_empty() {
        let mut nf = [0u64; 1024];
        for &q in needed {
            let h = ((u64::from(q).wrapping_mul(0x9E37_79B1)) >> 16) & 0xFFFF;
            nf[(h >> 6) as usize] |= 1u64 << (h & 63);
        }
        Some(nf)
    } else {
        None
    };

    // ONE streaming pass over the slice instead of four separate ones: at
    // position i we emit the byte bit (i < chunk_len), the bigram bit
    // (i, i+1) anywhere in the +3 overlap, the quad Bloom bits for windows
    // owned by this segment (i < chunk_len && i + 3 < slen), and the
    // first-quad position (only for owned windows that also fully fit the
    // chunk: i + 4 <= chunk_len; the straddling tail is the next chunk's).
    let mut i = 0usize;
    while i < slen {
        if i < chunk_len {
            bytes[slice[i] as usize] = true;
        }
        if i + 1 < slen {
            let a = slice[i] as usize;
            let b = slice[i + 1] as usize;
            bigrams[(a << 2) | (b >> 6)] |= 1u64 << (b & 63);
        }
        if i < chunk_len && i + 3 < slen {
            let q = u32::from_le_bytes([slice[i], slice[i + 1], slice[i + 2], slice[i + 3]]);
            let (h1, h2) = bloom_hashes(q, quad_mask);
            quads[(h1 >> 6) as usize] |= 1u64 << (h1 & 63);
            quads[(h2 >> 6) as usize] |= 1u64 << (h2 & 63);
            if i + 4 <= chunk_len {
                if let Some(nf) = nf {
                    let h = ((u64::from(q).wrapping_mul(0x9E37_79B1)) >> 16) & 0xFFFF;
                    if (nf[(h >> 6) as usize] >> (h & 63)) & 1 != 0 && needed.contains(&q) {
                        let v = pos.entry(q).or_default();
                        // Bounded per chunk: verification reads at most 1024
                        // candidates; the merge keeps the first 1024 overall.
                        if v.len() < 1024 {
                            v.push((base + i) as u32);
                        }
                    }
                }
            }
        }
        i += 1;
    }
    BuildPart { bytes, bigrams, quads: QuadSeg { bits: quads, mask: quad_mask }, pos }
}

fn merge_parts(parts: Vec<BuildPart>, _n: usize) -> TextIndex {
    let mut out_bytes = [false; 256];
    let mut out_bigrams = [0u64; 1024];
    let mut out_quads: Vec<QuadSeg> = Vec::with_capacity(parts.len());
    let mut out_pos: FastMap<u32, Vec<u32>> = new_fast_map();
    for p in parts {
        for i in 0..256 {
            out_bytes[i] |= p.bytes[i];
        }
        for i in 0..1024 {
            out_bigrams[i] |= p.bigrams[i];
        }
        // segment order == chunk order: push, do not OR (each chunk owns a
        // disjoint window range, so no cross-segment duplicates exist)
        out_quads.push(p.quads);
        for (q, v) in p.pos {
            let dst = out_pos.entry(q).or_default();
            // Keep the first 1024 candidates overall (ascending order).
            let room = 1024usize.saturating_sub(dst.len());
            dst.extend(v.into_iter().take(room));
        }
    }
    TextIndex { bytes: out_bytes, bigrams: out_bigrams, quads: out_quads, pos: out_pos }
}

impl TextIndex {
    pub fn build(text: &[u8]) -> TextIndex {
        TextIndex::build_with(text, &first_quads())
    }

    /// `build`, but records quad positions only for the quads in `needed`
    /// (the pattern first-quads). Pass an empty set when only the presence
    /// filters are wanted (benchmarks, tests).
    ///
    /// Texts larger than ~L2/3 are split into cache-shaped chunks (see
    /// `chunk_plan`: chunk ~= L2/3, worker count <= detected cores) built on
    /// scoped threads (no rayon dependency) and merged: the positions pass
    /// (one filter probe per 4-byte window, plus a push per recorded
    /// occurrence) dominates the build at MB scale, and it parallelizes
    /// cleanly.
    pub fn build_with(text: &[u8], needed: &HashSet<u32, FastHasher>) -> TextIndex {
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::INDEX_BUILDS.fetch_add(1, Ordering::Relaxed);
            prof::INDEX_BYTES.fetch_add(text.len() as u64, Ordering::Relaxed);
        }
        let plan = chunk_plan(text.len());
        if plan.len() == 1 {
            return build_serial(text, needed);
        }
        let mut parts: Vec<Option<BuildPart>> = (0..plan.len()).map(|_| None).collect();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(plan.len());
            for (ci, slot) in parts.iter_mut().enumerate() {
                let (base, end) = plan[ci];
                handles.push(s.spawn(move || {
                    *slot = Some(build_chunk(text, needed, base, end));
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        merge_parts(parts.into_iter().map(|p| p.unwrap()).collect(), text.len())
    }

    /// Byte positions of `q` (first-4-bytes of a pattern part), if the index
    /// was built with `q` in its `needed` set. Every occurrence of `q` is
    /// present (windows enumerated step 1), so a missing entry is a hard
    /// "quad absent from this text".
    #[inline]
    pub fn positions(&self, q: u32) -> Option<&Vec<u32>> {
        self.pos.get(&q)
    }

    /// True if the literal part *could* appear in the indexed text. Any
    /// occurrence of `part` must contain every one of its internal 4-byte
    /// windows, so we require all of them to be present. Checking all quads
    /// (not just the first) rejects e.g. a part whose first 4 bytes appear
    /// but whose full word does not, which is the common false-positive
    /// driving wasted SIMD scans. A false return is still a hard no.
    /// One Bloom probe over the segment tables. Exact presence bit test:
    /// `q` is definitely absent only when no segment holds both of its bits.
    #[inline]
    fn quad_present(&self, q: u32) -> bool {
        self.quads.iter().any(|seg| {
            let (h1, h2) = bloom_hashes(q, seg.mask);
            let bits = &seg.bits;
            ((bits[(h1 >> 6) as usize] >> (h1 & 63)) & 1) != 0
                && ((bits[(h2 >> 6) as usize] >> (h2 & 63)) & 1) != 0
        })
    }

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
                // Segment-local tables: the first and last quad may live in
                // different segments when the part would straddle a chunk
                // boundary - probing each quad independently keeps this an
                // exact "absent" gate with no false negatives.
                let first = u32::from_le_bytes([part[0], part[1], part[2], part[3]]);
                if !self.quad_present(first) {
                    return false;
                }
                let n = part.len();
                let last = u32::from_le_bytes([part[n - 4], part[n - 3], part[n - 2], part[n - 1]]);
                self.quad_present(last)
            }
        }
    }

    /// All-quads variant: every internal 4-byte window must hit the bloom.
    /// Stronger than `could_contain` (first+last only); used as a pre-gate
    /// before candidate verification so parts whose full word never occurs
    /// are rejected without walking their first quad's positions (repetitive
    /// text can hold hundreds of candidates per quad).
    pub fn could_contain_all(&self, part: &[u8]) -> bool {
        let mut w = 0;
        while w + 4 <= part.len() {
            let q = u32::from_le_bytes([part[w], part[w + 1], part[w + 2], part[w + 3]]);
            if !self.quad_present(q) {
                return false;
            }
            w += 1;
        }
        true
    }
}

/// Per-keyword info collected while building the string blob.
struct KwInfo {
    raw_off: u32,
    raw_len: u32,
    lower_off: u32,
    lower_len: u32,
    no_wildcard: bool,
    parts_off: u32,
    parts_len: u32,
    parts: Vec<(u32, u32)>, // (offset, len) into the blob for each literal part
}

/// Precompile every keyword in a set of AST blocks into a dense table.
/// All keyword strings (raw, lowered, literal parts) are concatenated into
/// ONE process-lifetime blob; every `Pattern` is then a small record of
/// (offset, len) pairs into that blob. The blob is also what the boot cache
/// mmaps, so a cache hit needs zero keyword copying. Matching never hashes
/// keyword strings (leaves resolve to table indices at boot).
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

pub fn compile_all<'a>(blocks: impl Iterator<Item = &'a Node>) -> Vec<Pattern> {
    // 1) unique keywords in first-seen order
    let mut kws: Vec<&str> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    let mut leaves = Vec::new();
    for b in blocks {
        leaves.clear();
        collect_leaves(b, &mut leaves);
        for kw in leaves.drain(..) {
            if seen.insert(kw) {
                kws.push(kw);
            }
        }
    }
    // 2) build the blob: raw bytes, lowered bytes ('?'-globs), part bytes,
    //    then the parts table (u32 (off,len) pairs).
    let mut blob: Vec<u8> = Vec::new();
    let mut infos: Vec<KwInfo> = Vec::with_capacity(kws.len());
    for kw in &kws {
        // Leading/trailing whitespace in a keyword is a data artifact
        // (SDG07 contains `TITLE-ABS(" international") ...`); Scopus ignores
        // it in phrases too.
        let kw = kw.trim();
        let raw_off = blob.len() as u32;
        blob.extend_from_slice(kw.as_bytes());
        let raw_len = kw.len() as u32;
        let lower = kw.to_ascii_lowercase();
        let has_star = lower.contains('*');
        let has_q = lower.contains('?');
        let (no_wildcard, parts_bytes): (bool, Vec<Vec<u8>>) = if !has_star && !has_q {
            (true, vec![lower.as_bytes().to_vec()])
        } else if !has_q {
            (false, lower.split('*').filter(|p| !p.is_empty()).map(|p| p.as_bytes().to_vec()).collect())
        } else {
            (false, Vec::new())
        };
        let mut parts = Vec::with_capacity(parts_bytes.len());
        for pb in parts_bytes {
            let off = blob.len() as u32;
            blob.extend_from_slice(&pb);
            parts.push((off, pb.len() as u32));
        }
        let (lower_off, lower_len) = if has_q {
            let off = blob.len() as u32;
            blob.extend_from_slice(lower.as_bytes());
            (off, lower.len() as u32)
        } else {
            (0, 0)
        };
        infos.push(KwInfo { raw_off, raw_len, lower_off, lower_len, no_wildcard, parts_off: 0, parts_len: 0, parts });
    }
    // parts table (pairs are read via `Pattern::part`)
    for info in &mut infos {
        info.parts_off = blob.len() as u32;
        for &(off, len) in &info.parts {
            blob.extend_from_slice(&off.to_le_bytes());
            blob.extend_from_slice(&len.to_le_bytes());
        }
        info.parts_len = info.parts.len() as u32;
    }
    let blob_static: &'static [u8] = Box::leak(blob.into_boxed_slice());
    set_blob(blob_static);
    let table: Vec<Pattern> = infos
        .into_iter()
        .map(|i| Pattern {
            raw_off: i.raw_off,
            raw_len: i.raw_len,
            lower_off: i.lower_off,
            lower_len: i.lower_len,
            parts_off: i.parts_off,
            parts_len: i.parts_len,
            no_wildcard: i.no_wildcard,
        })
        .collect();
    // Union every literal part's first-4-bytes so TextIndex builds record
    // positions for all of them (multiple compile_all calls stay sound).
    let mut g = FIRST_QUADS.lock().unwrap();
    let s = g.get_or_insert_with(|| Arc::new(HashSet::with_hasher(FastHasher::default())));
    let s = Arc::make_mut(s);
    for p in &table {
        for part in p.parts() {
            if part.len() >= 4 {
                s.insert(u32::from_le_bytes([part[0], part[1], part[2], part[3]]));
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
        map.insert(p.raw(), i);
    }
    // Slot ids: dense over the seen (pid, mask) pairs, assigned in firsteat 
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

/// Substring glob with Scopus `?` semantics, matching the reference Python
/// engine (`re.search` with `re.DOTALL`): `?` = any single byte, `*` = any
/// run (including newlines), and the pattern matches ANY substring of
/// `text` - not just a match anchored at the start. Runs per field segment
/// on folded buffers, so a pattern can never span fields.
///
/// Only the three corpus patterns containing `?` reach this path, and their
/// results are memoized per (pattern, mask), so the O(n*m) start-offset
/// scan is negligible.
fn glob_match(pat: &[u8], text: &[u8]) -> bool {
    for start in 0..=text.len() {
        if glob_match_at(pat, &text[start..]) {
            return true;
        }
    }
    false
}

/// Classic iterative glob with star backtracking. Succeeds as soon as the
/// pattern is fully consumed: leftover text is allowed (substring
/// semantics), unlike an anchored whole-string match.
fn glob_match_at(pat: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while p < pat.len() {
        if t < text.len() && (pat[p] == b'?' || pat[p] == text[t]) {
            p += 1;
            t += 1;
        } else if pat[p] == b'*' {
            star = Some(p);
            mark = t;
            p += 1;
        } else if let Some(sp) = star {
            // Backtrack: advance the star's end by one. Once it has consumed
            // the whole text there is nothing left to try (this bound also
            // prevents `t` from running past the text end forever when the
            // star's suffix can never match).
            if mark >= text.len() {
                return false;
            }
            p = sp + 1;
            mark += 1;
            t = mark;
        } else {
            return false;
        }
    }
    true
}

/// `text[p..]` starts with `part` (per-candidate check in the quad-position
/// index). A single unaligned u64 load+compare on both sides covers the
/// first 8 bytes: the first 4 are already known equal (they ARE the quad the
/// position was recorded for), and most candidates diverge from the pattern
/// within bytes 4-7, so this avoids a libc memcmp call per candidate. Falls
/// back to slice equality for the tail of longer parts (rare path).
#[inline]
fn starts_at(text: &[u8], p: usize, part: &[u8]) -> bool {
    let n = part.len();
    if text.len() - p < n {
        return false;
    }
    let t = &text[p..];
    if n >= 8 {
        let pm = u64::from_le_bytes(part[..8].try_into().unwrap());
        let tm = u64::from_le_bytes(t[..8].try_into().unwrap());
        if tm != pm {
            return false;
        }
        n == 8 || &t[8..n] == &part[8..]
    } else {
        &t[..n] == part
    }
}

impl Pattern {
    /// Cheap pre-filter: every literal part of the pattern must be present
    /// in the indexed text, otherwise `matches` cannot succeed. Glob ('?')
    /// patterns have no literal parts and always pass.
    pub fn could_match(&self, idx: &TextIndex) -> bool {
        if self.n_parts() == 0 {
            return true;
        }
        self.parts().all(|p| idx.could_contain(p))
    }

    pub fn matches(&self, text: &[u8]) -> bool {
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::MATCHES_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        if self.no_wildcard {
            find_boundary(text, self.part(0))
        } else if self.n_parts() == 0 {
            glob_match(self.lower_raw(), text)
        } else {
            let mut from = 0usize;
            for k in 0..self.n_parts() {
                let part = self.part(k);
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

    /// Like `matches`, but uses the index's quad positions to verify a
    /// handful of candidate starts instead of scanning the whole buffer.
    /// Semantics are identical (same word-boundary rule for plain terms,
    /// same no-space wildcard gap rule); `positions` has no false negatives,
    /// so a missing entry for the part's first quad means the quad is absent
    /// and `false` is authoritative. Falls back to `matches` when the first
    /// quad is common (more than 1024 occurrences: scattered verification
    /// costs more than one streaming SIMD pass).
    pub fn matches_indexed(&self, text: &[u8], idx: &TextIndex) -> bool {
        #[cfg(feature = "prof")]
        {
            use std::sync::atomic::Ordering;
            prof::MATCHES_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let part0 = if self.n_parts() == 0 {
            return self.matches(text); // '?' glob: no literal parts to index
        } else {
            self.part(0)
        };
        if part0.len() >= 4 {
            // Reject parts whose full word never occurs before walking
            // candidate positions (first+last-quad filter is not enough on
            // repetitive text where both quads occur independently).
            if !idx.could_contain_all(part0) {
                return false;
            }
            let q = u32::from_le_bytes([part0[0], part0[1], part0[2], part0[3]]);
            match idx.positions(q) {
                Some(ps) if ps.len() <= 1024 => {
                    // Exhaustive: verify every candidate (1024 unaligned
                    // loads is ~20-40 us, still cheaper than one streaming
                    // SIMD pass, and a miss is a hard false).
                    for &p in ps {
                        let p = p as usize;
                        if !starts_at(text, p, part0) {
                            continue;
                        }
                        if self.no_wildcard {
                            let before = p == 0 || !is_word(text[p - 1]);
                            let after_p = p + part0.len();
                            let after = after_p >= text.len() || !is_word(text[after_p]);
                            if before && after {
                                return true;
                            }
                        } else {
                            // rest_matches scans forward and misses only after
                            // reaching the end of the buffer; every later
                            // candidate starts even further forward, so it
                            // would miss too (same first-occurrence semantics
                            // as `matches`).
                            if self.rest_matches_at(text, p + part0.len()) {
                                return true;
                            }
                            return false;
                        }
                    }
                    return false;
                }
                // No positions recorded: every literal part's first four
                // bytes are in FIRST_QUADS (compile_all unions them), so the
                // quad is absent from this text: hard no-match, no scan.
                None => return false,
                // Common quad: one streaming SIMD pass is cheaper.
                Some(_) => {}
            }
        }
        self.matches(text)
    }

    /// Subsequent literal parts must follow with only non-space characters
    /// in between (Scopus `*` matches within a word only).
    fn rest_matches_at(&self, text: &[u8], mut from: usize) -> bool {
        for k in 1..self.n_parts() {
            let part = self.part(k);
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
        true
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
    joined: Vec<JoinedEntry<'a>>,
    /// O(1) mask -> joined-buffer index (i32 for a -1 "unset" sentinel).
    mask_cache: [i32; 256],
    /// Buffer-id-list signature -> joined entry, so every mask that selects
    /// the same buffers shares ONE buffer and ONE TextIndex build (masks
    /// that all fall back to the full text used to rebuild the full text
    /// copy + index per mask: measured 2.3x slower on a 1.8 MB paper).
    join_cache: FastMap<u64, usize>,
}

/// A mask's folded buffer plus its per-field segments (for `?` globs) and a
/// lazily built `TextIndex` pre-filter.
struct JoinedEntry<'a> {
    /// Borrowed when the entry is exactly the full text (no copy); owned
    /// when it is a real multi-buffer fold.
    buf: Cow<'a, [u8]>,
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
            join_cache: new_fast_map(),
        }
    }

    /// Get (building on first use) the folded buffer for `mask`. Returns its
    /// index in `self.joined`.
    fn joined_for(&mut self, mask: u8) -> usize {
        let cached = self.mask_cache[mask as usize];
        if cached >= 0 {
            return cached as usize;
        }
        // Decode the mask into deduplicated buffer ids.
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
        // Share one joined buffer across every mask selecting the same ids
        // (e.g. all masks falling back to the full text): one copy + one
        // TextIndex build per distinct selection instead of per mask.
        let mut key = n as u64;
        for &id in &ids[..n] {
            key = (key << 8) | u64::from(id);
        }
        if let Some(&j) = self.join_cache.get(&key) {
            self.mask_cache[mask as usize] = j as i32;
            return j;
        }
        let mut cap = 0usize;
        for &id in &ids[..n] {
            cap += self.bufs[id as usize].len();
        }
        let (buf, segs, nsegs);
        if n == 1 && ids[0] == self.full_id {
            // Exactly the full text: borrow it, zero copy. Its per-section
            // ranges (when full covers the sections) keep '?'-glob
            // semantics; otherwise it is one segment.
            let full = self.bufs[self.full_id as usize];
            if self.full_covers_sections {
                let mut sg = [(0usize, 0usize); 5];
                let mut ns = 0usize;
                for (s, e) in self.full_segs.iter() {
                    if s != e {
                        sg[ns] = (*s, *e);
                        ns += 1;
                    }
                }
                buf = Cow::Borrowed(full);
                segs = sg;
                nsegs = ns;
            } else {
                buf = Cow::Borrowed(full);
                segs = [(0, full.len()), (0, 0), (0, 0), (0, 0), (0, 0)];
                nsegs = 1;
            }
        } else {
            let mut v = Vec::with_capacity(cap + n);
            let mut sg = [(0usize, 0usize); 5];
            let mut ns = 0usize;
            for &id in &ids[..n] {
                let b = self.bufs[id as usize];
                if ns > 0 {
                    v.push(b'\n');
                }
                let start = v.len();
                v.extend_from_slice(b);
                sg[ns] = (start, v.len());
                ns += 1;
            }
            buf = Cow::Owned(v);
            segs = sg;
            nsegs = ns;
        }
        self.joined.push(JoinedEntry { buf, idx: None, nsegs: nsegs as u8, segs });
        let idx = self.joined.len() - 1;
        self.mask_cache[mask as usize] = idx as i32;
        self.join_cache.insert(key, idx);
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

    /// Evaluate one pre-resolved leaf occurrence (`pid`/`mask`/`slot` come
    /// straight from a `LeafDesc` record) with the usual per-request
    /// memoization. This is the report-light evaluation used by the Advanced
    /// keyword browser (`/api/keywords`), which only needs the *present*
    /// keywords of one SDG - not the per-block hits/misses/excluded lists.
    /// Skipping the boolean VM and the classification pushes is significantly
    /// cheaper than `scan_flat_into` when only presence matters.
    #[inline]
    pub fn leaf_hit(&mut self, pat: &Pattern, mask: u8, slot: u32) -> bool {
        self.term_hit(pat, mask, slot as usize)
    }

    /// Actual search for `pat` under `mask` (uncached), against the mask's
    /// single folded buffer. `?`-glob patterns (no literal parts) run per
    /// segment so a `?` can never consume the join separator; everything
    /// else uses the TextIndex pre-filter then one SIMD search.
    fn compute(&mut self, pat: &Pattern, mask: u8) -> bool {
        // A zero mask means "no field scoping" -> the default TITLE-ABS-KEY.
        let mask = if mask == 0 { field_mask(&ALL_FIELDS) } else { mask };
        let jidx = self.joined_for(mask);
        if pat.n_parts() == 0 {
            // '?' glob: run per segment so patterns cannot span fields.
            let j = &self.joined[jidx];
            if j.nsegs > 1 {
                for s in &j.segs[..j.nsegs as usize] {
                    if glob_match(pat.lower_raw(), &j.buf[s.0..s.1]) {
                        return true;
                    }
                }
                return false;
            }
            return glob_match(pat.lower_raw(), &j.buf);
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
        if filter && (prof::skip_find() || pat.matches_indexed(&entry.buf, idx)) {
            return true;
        }
        #[cfg(not(feature = "prof"))]
        if filter && pat.matches_indexed(&entry.buf, idx) {
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
    pub hits: Vec<&'static str>,
    pub misses: Vec<&'static str>,
    pub excluded_hits: Vec<&'static str>,
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
/// Groups are emitted as a single n-ary op (`OrN(k)` / `AndN(k)` pops the
/// top k stack entries and folds them at once) instead of k-1 binary folds:
/// the min-add union/concat is then O(total keywords) per group instead of
/// O(n^2) re-unioning the accumulated result at every fold.
///
/// Fixed 8-byte record (`tag` + `payload`) so the program is a dense, aligned
/// array that the hot loop walks sequentially and the boot cache can mmap
/// zero-copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Op {
    pub tag: u32,
    pub payload: u32,
}

pub const OP_PUSH: u32 = 0;
pub const OP_TRUE: u32 = 1;
pub const OP_FALSE: u32 = 2;
pub const OP_NOT: u32 = 3;
pub const OP_AND: u32 = 4;
pub const OP_OR: u32 = 5;
pub const OP_ANDN: u32 = 6;
pub const OP_ORN: u32 = 7;

impl Op {
    #[inline]
    pub fn push(i: u32) -> Op {
        Op { tag: OP_PUSH, payload: i }
    }
    #[inline]
    pub fn and_n(n: u32) -> Op {
        Op { tag: OP_ANDN, payload: n }
    }
    #[inline]
    pub fn or_n(n: u32) -> Op {
        Op { tag: OP_ORN, payload: n }
    }
}

/// One keyword occurrence inside a flattened block.
#[derive(Clone, Debug)]
/// One keyword occurrence inside a flattened block. `raw_off`/`raw_len`
/// index the process string blob, so a cached (mmap'd) block needs no
/// keyword copies. Fixed-size for direct mmap viewing.
#[repr(C)]
pub struct LeafDesc {
    pub pid: u32,
    pub slot: u32,
    pub mask: u8,
    pub excluded: bool,
    pub raw_off: u32,
    pub raw_len: u32,
}

impl LeafDesc {
    #[inline]
    pub fn raw(&self) -> &'static str {
        let b = blob();
        unsafe { std::str::from_utf8_unchecked(&b[self.raw_off as usize..(self.raw_off + self.raw_len) as usize]) }
    }
}

/// A block compiled to a postfix program + flat leaf list. Built once at
/// boot, AFTER `resolve_blocks` has stamped slots onto the AST.
/// A block compiled to a postfix program + flat leaf list. Both are
/// fixed-size records; the slices are `'static` (leaked at compile time or
/// zero-copy views of the mmap'd boot cache).
pub struct FlatBlock {
    pub prog: &'static [Op],
    pub leaves: &'static [LeafDesc],
}

/// Flatten one block (call AFTER `resolve_blocks`). Leaf order and exclusion
/// parity match the AST traversal exactly, so `scan_flat` produces the same
/// hits/misses/excluded lists (including duplicates) as the tree walk.
pub fn flatten_block(block: &Node, table: &[Pattern]) -> FlatBlock {
    fn emit(node: &Node, excluded: bool, table: &[Pattern], prog: &mut Vec<Op>, leaves: &mut Vec<LeafDesc>) {
        match node {
            Node::Leaf { pid, mask, slot, .. } => {
                let i = leaves.len() as u32;
                leaves.push(LeafDesc {
                    pid: *pid,
                    slot: *slot,
                    mask: *mask,
                    excluded,
                    raw_off: table[*pid as usize].raw_off_len().0,
                    raw_len: table[*pid as usize].raw_off_len().1,
                });
                prog.push(Op::push(i));
            }
            Node::Field { child, .. } => emit(child, excluded, table, prog, leaves),
            Node::Not { child } => {
                emit(child, !excluded, table, prog, leaves);
                prog.push(Op { tag: OP_NOT, payload: 0 });
            }
            Node::Group { op, children } => {
                // A group with k children folds as a single n-ary op
                // (Op::AndN/OrN): one pass, no intermediate binary folds, so
                // the min-add union/concat is O(total keywords) per group.
                match children.len() {
                    0 => prog.push(if op == "OR" { Op { tag: OP_FALSE, payload: 0 } } else { Op { tag: OP_TRUE, payload: 0 } }),
                    1 => emit(&children[0], excluded, table, prog, leaves),
                    _ => {
                        let n = children.len() as u32;
                        for c in children {
                            emit(c, excluded, table, prog, leaves);
                        }
                        prog.push(if op == "OR" { Op::or_n(n) } else { Op::and_n(n) });
                    }
                }
            }
        }
    }
    let mut prog: Vec<Op> = Vec::new();
    let mut leaves: Vec<LeafDesc> = Vec::new();
    emit(block, false, table, &mut prog, &mut leaves);
    // 'static slices: leaked once at boot (or zero-copy views of the mmap'd
    // boot cache) - the user-facing trade is RAM for startup speed.
    FlatBlock {
        prog: Box::leak(prog.into_boxed_slice()),
        leaves: Box::leak(leaves.into_boxed_slice()),
    }
}

/// Tiny boolean stack: 32 inline slots plus a heap fallback. 2974 of the
/// 2975 corpus blocks never exceed depth 32 (measured 2026-08), so the
/// per-block stack allocation of the previous `Vec::with_capacity(8)` is
/// gone; only pathological deep spines (SDG07 b1 reaches 611) touch the
/// heap.
struct BoolStack {
    buf: [bool; 32],
    sp: usize,
    extra: Vec<bool>,
}

impl BoolStack {
    fn new() -> BoolStack {
        BoolStack { buf: [false; 32], sp: 0, extra: Vec::new() }
    }
    #[inline]
    fn push(&mut self, v: bool) {
        if self.sp < 32 {
            self.buf[self.sp] = v;
            self.sp += 1;
        } else {
            self.extra.push(v);
        }
    }
    #[inline]
    fn pop(&mut self) -> bool {
        if let Some(v) = self.extra.pop() {
            return v;
        }
        if self.sp == 0 {
            return false;
        }
        self.sp -= 1;
        self.buf[self.sp]
    }
    /// Fold the top `k` values with `f` (left-associative), then replace
    /// them with the result. No allocation: values are read inline from the
    /// inline buffer or the heap spill.
    #[inline]
    fn fold_top<F: Fn(bool, bool) -> bool>(&mut self, k: u32, f: F) -> bool {
        let k = k as usize;
        let total = self.sp + self.extra.len();
        debug_assert!(k <= total);
        // All values that matter live in `buf[0..sp]` plus `extra`; the
        // top k values span the tail of `extra` and/or the tail of `buf`.
        // Iterate from the bottom of the top-k window upward.
        let mut acc = false;
        let mut first = true;
        let take_from_extra = self.extra.len().min(k);
        let need_from_buf = k - take_from_extra;
        for i in (self.extra.len() - take_from_extra)..self.extra.len() {
            let v = self.extra[i];
            acc = if first { v } else { f(acc, v) };
            first = false;
        }
        for i in (self.sp - need_from_buf)..self.sp {
            let v = self.buf[i];
            acc = if first { v } else { f(acc, v) };
            first = false;
        }
        // Truncate k entries.
        let drop_from_extra = self.extra.len().min(k);
        self.extra.truncate(self.extra.len() - drop_from_extra);
        self.sp -= k - drop_from_extra;
        // Push the folded result.
        self.push(acc);
        acc
    }
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
    #[cfg(feature = "prof")]
    let report = !prof::skip_report();
    #[cfg(not(feature = "prof"))]
    let report = true;
    let mut stack = BoolStack::new();
    for op in flat.prog {
        match op.tag {
            OP_PUSH => {
                let l = &flat.leaves[op.payload as usize];
                let v = memo.term_hit(&table[l.pid as usize], l.mask, l.slot as usize);
                if report {
                    #[cfg(feature = "prof")]
                    {
                        use std::sync::atomic::Ordering;
                        prof::REPORT_PUSHES.fetch_add(1, Ordering::Relaxed);
                    }
                    if l.excluded {
                        if v {
                            ex_hits.push(l.raw());
                        }
                    } else if v {
                        hits.push((l.raw(), l.mask));
                    } else {
                        misses.push((l.raw(), l.mask));
                    }
                }
                stack.push(v);
            }
            OP_TRUE => stack.push(true),
            OP_FALSE => stack.push(false),
            OP_NOT => {
                let t = stack.pop();
                stack.push(!t);
            }
            OP_AND => {
                let b = stack.pop();
                let a = stack.pop();
                stack.push(a && b);
            }
            OP_OR => {
                let b = stack.pop();
                let a = stack.pop();
                stack.push(a || b);
            }
            OP_ANDN => {
                stack.fold_top(op.payload, |a, b| a && b);
            }
            OP_ORN => {
                stack.fold_top(op.payload, |a, b| a || b);
            }
            _ => unreachable!("unknown op tag"),
        }
    }
    stack.pop()
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

// ---------------------------------------------------------------------------
// Near-miss analysis: minimum keywords to add ("missing tags") + reranking
//
// For every block that did not match we compute, with no LLM and no heuristic
// scoring, the *exact* minimum number of keywords that must be added to the
// paper text for the block to become true, plus the candidate keyword groups
// to choose from (any ONE keyword from each group qualifies). Blocks are then
// reranked by that cost (cheapest first), which surfaces e.g. an AND block
// that is a single country name away from SDG 3 instead of a huge OR block
// with fewer total keywords but no topical overlap.
//
// cost == INF_COST means the block is false and CANNOT be fixed by adding
// keywords: a NOT branch on a required path is already true (the paper
// contains an excluded term). Such blocks are reported as "disqualified" and
// are never suggested as near misses.
// ---------------------------------------------------------------------------

pub const INF_COST: usize = usize::MAX;

/// Result of the minimum-addition analysis for one (sub)tree.
#[derive(Debug, Clone)]
pub struct MinAdd {
    /// Current boolean value of the subtree against the paper.
    pub value: bool,
    /// Minimum keywords to add to make the subtree true (INF_COST if impossible).
    pub cost: usize,
    /// Candidate keywords to add: pick any ONE keyword from each group. Only
    /// meaningful when `value == false && cost < INF_COST`. Keywords borrow
    /// the process string blob ('static - zero-copy from the mmap'd cache).
    pub need: Vec<Vec<&'static str>>,
}

/// Fast path: for a plain keyword child (`Leaf` or `Field(Leaf)` - the
/// overwhelming majority of OR/AND children) return `(keyword, hit)` without
/// allocating a `MinAdd`. This avoids the per-leaf vector allocations that
/// dominated large blocks (e.g. SDG07's giant lists).
fn leaf_kw_hit(node: &Node, _mask: u8, table: &[Pattern], memo: &mut Memo) -> Option<(&'static str, bool)> {
    let (leaf, lm): (&Node, u8) = match node {
        Node::Leaf { mask: m, .. } => (node, *m),
        Node::Field { fields, child } => match child.as_ref() {
            Node::Leaf { .. } => (child, field_mask_from_strings(fields)),
            _ => return None,
        },
        _ => return None,
    };
    match leaf {
        Node::Leaf { slot, .. } => {
            let p = pat(table, leaf);
            let v = memo.term_hit(p, lm, *slot as usize);
            Some((p.raw(), v))
        }
        _ => unreachable!(),
    }
}

/// Group fingerprint: FNV-1a over (pointer, len) of each keyword. Duplicate
/// groups across TITLE-ABS / AUTHKEY / TITLE variants share the same blob
/// strings, hence the same addresses, so pointer-based dedup is exact and
/// costs O(1) per keyword.
fn group_fp(g: &[&'static str]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for kw in g {
        h ^= kw.as_ptr() as usize as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= kw.len() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn min_add(node: &Node, mask: u8, table: &[Pattern], memo: &mut Memo) -> MinAdd {
    match node {
        Node::Leaf { mask: lm, slot, .. } => {
            let p = pat(table, node);
            if memo.term_hit(p, *lm, *slot as usize) {
                MinAdd { value: true, cost: 0, need: Vec::new() }
            } else {
                MinAdd { value: false, cost: 1, need: vec![vec![p.raw()]] }
            }
        }
        Node::Field { fields, child } => {
            min_add(child, field_mask_from_strings(fields), table, memo)
        }
        Node::Not { child } => {
            let c = min_add(child, mask, table, memo);
            if c.value {
                // Child is already true; adding keywords can only make it
                // "more true", so NOT(child) can never become true.
                MinAdd { value: false, cost: INF_COST, need: Vec::new() }
            } else {
                MinAdd { value: true, cost: 0, need: Vec::new() }
            }
        }
        Node::Group { op, children } => {
            if op == "OR" {
                // Streaming pass: no intermediate Vec<MinAdd> per node. Track
                // the cheapest satisfiable branch plus the union of the
                // single-keyword groups of every equally-cheap branch.
                let mut any_true = false;
                let mut min_cost = INF_COST;
                let mut best: Option<MinAdd> = None;
                let mut all_single = true;
                let mut seen: HashSet<&'static str> = HashSet::new();
                let mut union: Vec<&'static str> = Vec::new();
                for c in children {
                    // Plain keywords (the overwhelming majority of children)
                    // are handled inline so a leaf never allocates a MinAdd;
                    // only the final union and the single `best` candidate do.
                    match leaf_kw_hit(c, mask, table, memo) {
                        Some((kw, hit)) => {
                            if hit {
                                any_true = true;
                                break;
                            }
                            // cost-1, single-keyword candidate, no allocation
                            if 1 < min_cost {
                                min_cost = 1;
                                all_single = true;
                                seen.clear();
                                union.clear();
                                seen.insert(kw);
                                union.push(kw);
                                best = Some(MinAdd { value: false, cost: 1, need: vec![vec![kw]] });
                            } else if 1 == min_cost && all_single {
                                if seen.insert(kw) {
                                    union.push(kw);
                                }
                            }
                        }
                        None => {
                            let r = min_add(c, mask, table, memo);
                            if r.value {
                                any_true = true;
                                break;
                            }
                            if r.cost == INF_COST {
                                continue;
                            }
                            if r.cost < min_cost {
                                // New cheapest branch: reset the union.
                                min_cost = r.cost;
                                let single = r.need.len() == 1;
                                seen.clear();
                                union.clear();
                                if single {
                                    for kw in &r.need[0] {
                                        seen.insert(kw);
                                        union.push(kw);
                                    }
                                }
                                all_single = single;
                                best = Some(r);
                            } else if r.cost == min_cost {
                                // Tie with the cheapest: union only counts
                                // when every cheapest branch is single-keyword.
                                if all_single {
                                    if r.need.len() != 1 {
                                        all_single = false;
                                        union.clear();
                                        seen.clear();
                                    } else {
                                        for kw in &r.need[0] {
                                            if seen.insert(kw) {
                                                union.push(kw);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if any_true {
                    return MinAdd { value: true, cost: 0, need: Vec::new() };
                }
                let b = match best {
                    None => return MinAdd { value: false, cost: INF_COST, need: Vec::new() },
                    Some(b) => b,
                };
                let need = if all_single && !union.is_empty() { vec![union] } else { b.need };
                MinAdd { value: false, cost: min_cost, need }
            } else {
                // AND, W/n, PRE/n, ... : every child must hit. Costs add up
                // across groups; any impossible child makes the block
                // impossible. Groups are deduped by a pointer fingerprint
                // (linear deep `contains` was O(checks x group chars) and
                // dominated large AND chains like SDG07's).
                let mut cost = 0usize;
                let mut need: Vec<Vec<&'static str>> = Vec::new();
                let mut seen_fp: HashMap<u64, Vec<&'static str>> = HashMap::new();
                for c in children {
                    let r = min_add(c, mask, table, memo);
                    if r.cost == INF_COST {
                        return MinAdd { value: false, cost: INF_COST, need: Vec::new() };
                    }
                    cost = cost.saturating_add(r.cost);
                    for g in r.need {
                        let fp = group_fp(&g);
                        match seen_fp.get(&fp) {
                            // Same pointer sequence => identical group: skip.
                            Some(first) if *first == g => {}
                            // New group, or a rare fp collision with different
                            // content: keep it (collision is verified).
                            _ => {
                                seen_fp.insert(fp, g.clone());
                                need.push(g);
                            }
                        }
                    }
                }
                MinAdd { value: cost == 0, cost, need }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Flat-program min-add (mechanically sympathetic)
//
// The AST walk above is correct but allocates a MinAdd (two nested Vecs) at
// every node, which dominates large blocks (SDG07 b1: 18 ms). This variant
// evaluates the SAME semantics on the block's postfix program (`FlatBlock`):
// one sequential pass over `prog` with SoA stacks (bool+u32 pairs), and the
// need groups live in a per-block arena of plain integers + Arc clones.
// Memory layout follows the access pattern (dense vectors, no per-node heap
// churn, index-based references) and we gladly trade a little RAM for the
// ~50x speedup on pathological blocks.
// ---------------------------------------------------------------------------

/// u32 sentinel for INF_COST inside the flat evaluator.
const INF_U32: u32 = u32::MAX;

// TEMP probe counters
pub static DBG_OPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static DBG_UNION_KW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static DBG_UNIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static DBG_AND_GROUPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static DBG_FP_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[derive(Clone, Copy, Default)]
struct EvalEntry {
    value: bool,
    cost: u32,
    /// Need groups slice into the groups arena: [g_start, g_start + g_len).
    g_start: u32,
    g_len: u32,
    /// Single-union keyword slice into the kw arena (only when g_len == 1).
    k_start: u32,
    k_len: u32,
}

#[derive(Clone, Copy)]
struct GroupSlice {
    start: u32,
    len: u32,
}

/// Per-block scratch arenas, reused across blocks/requests.
pub struct MinAddScratch {
    /// (value, cost) pairs for the cost-only pass.
    pairs: Vec<(bool, u32)>,
    /// Full entries for the need pass.
    eval: Vec<EvalEntry>,
    /// Keyword arena: every keyword that ended up in a need group.
    kw: Vec<&'static str>,
    /// Group arena: slices into `kw`.
    groups: Vec<GroupSlice>,
    /// Pointer fingerprints of keywords in the current union.
    kw_seen: HashSet<u64>,
    /// Pointer fingerprints of groups already in the current AND need.
    group_seen: HashMap<u64, u32>,
    /// Reusable operand buffer for n-ary group ops (avoids per-op allocs).
    kids: Vec<EvalEntry>,
}

impl Default for MinAddScratch {
    fn default() -> Self {
        MinAddScratch {
            pairs: Vec::with_capacity(64),
            eval: Vec::with_capacity(64),
            kw: Vec::new(),
            groups: Vec::new(),
            kw_seen: HashSet::new(),
            group_seen: HashMap::new(),
            kids: Vec::with_capacity(16),
        }
    }
}

#[inline]
fn str_fp(a: &str) -> u64 {
    (a.as_ptr() as usize) as u64 ^ ((a.len() as u64) << 32)
}

#[inline]
fn group_fp_slice(kw: &[&'static str], start: u32, len: u32) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for kw in &kw[start as usize..(start + len) as usize] {
        h ^= str_fp(kw);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Cost-only pass on the flat program: (value, cost) with a dense pair
/// stack. No recursion, no allocation (stacks live in the scratch).
pub fn min_add_flat_cost(
    flat: &FlatBlock,
    table: &[Pattern],
    memo: &mut Memo,
    scr: &mut MinAddScratch,
) -> (bool, u32) {
    scr.pairs.clear();
    for op in flat.prog {
        match op.tag {
            OP_PUSH => {
                let l = &flat.leaves[op.payload as usize];
                let v = memo.term_hit(&table[l.pid as usize], l.mask, l.slot as usize);
                scr.pairs.push((v, if v { 0 } else { 1 }));
            }
            OP_TRUE => scr.pairs.push((true, 0)),
            OP_FALSE => scr.pairs.push((false, INF_U32)),
            OP_NOT => {
                let (v, _) = scr.pairs.pop().unwrap();
                scr.pairs.push((!v, if v { INF_U32 } else { 0 }));
            }
            OP_AND => {
                let (bv, bc) = scr.pairs.pop().unwrap();
                let (av, ac) = scr.pairs.pop().unwrap();
                let inf = (!av && ac == INF_U32) || (!bv && bc == INF_U32);
                let cost = if inf {
                    INF_U32
                } else {
                    (if av { 0 } else { ac }).saturating_add(if bv { 0 } else { bc })
                };
                scr.pairs.push((cost == 0, cost));
            }
            OP_OR => {
                let (bv, bc) = scr.pairs.pop().unwrap();
                let (av, ac) = scr.pairs.pop().unwrap();
                if av || bv {
                    scr.pairs.push((true, 0));
                } else {
                    scr.pairs.push((false, ac.min(bc)));
                }
            }
            OP_ANDN => {
                let k = op.payload as usize;
                let n = scr.pairs.len();
                let mut inf = false;
                let mut any_false = false;
                let mut cost = 0u32;
                for &(v, c) in &scr.pairs[n - k..] {
                    if !v {
                        any_false = true;
                        if c == INF_U32 {
                            inf = true;
                            break;
                        }
                        cost = cost.saturating_add(c);
                    }
                }
                scr.pairs.truncate(n - k);
                scr.pairs.push(if inf {
                    (false, INF_U32)
                } else if any_false {
                    (false, cost)
                } else {
                    (true, 0)
                });
            }
            OP_ORN => {
                let k = op.payload as usize;
                let n = scr.pairs.len();
                let mut any_true = false;
                let mut min = INF_U32;
                for &(v, c) in &scr.pairs[n - k..] {
                    if v {
                        any_true = true;
                        break;
                    }
                    if c < min {
                        min = c;
                    }
                }
                scr.pairs.truncate(n - k);
                scr.pairs.push(if any_true { (true, 0) } else { (false, min) });
            }
            _ => unreachable!("unknown op tag"),
        }
    }
    scr.pairs.pop().unwrap_or((false, INF_U32))
}

/// Full pass: value + cost + need groups, arena-backed.
pub fn min_add_flat(
    flat: &FlatBlock,
    table: &[Pattern],
    memo: &mut Memo,
    scr: &mut MinAddScratch,
) -> MinAdd {
    scr.eval.clear();
    scr.kw.clear();
    scr.groups.clear();
    scr.group_seen.clear();
    for op in flat.prog {
        DBG_OPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match op.tag {
            OP_PUSH => {
                let l = &flat.leaves[op.payload as usize];
                let v = memo.term_hit(&table[l.pid as usize], l.mask, l.slot as usize);
                if v {
                    scr.eval.push(EvalEntry { value: true, cost: 0, ..Default::default() });
                } else {
                    let k = scr.kw.len() as u32;
                    scr.kw.push(l.raw());
                    let g = scr.groups.len() as u32;
                    scr.groups.push(GroupSlice { start: k, len: 1 });
                    scr.eval.push(EvalEntry { value: false, cost: 1, g_start: g, g_len: 1, k_start: k, k_len: 1 });
                }
            }
            OP_TRUE => scr.eval.push(EvalEntry { value: true, cost: 0, ..Default::default() }),
            OP_FALSE => scr.eval.push(EvalEntry { value: false, cost: INF_U32, ..Default::default() }),
            OP_NOT => {
                let e = scr.eval.pop().unwrap();
                scr.eval.push(if e.value {
                    EvalEntry { value: false, cost: INF_U32, ..Default::default() }
                } else {
                    EvalEntry { value: true, cost: 0, ..Default::default() }
                });
            }
            OP_AND => {
                let b = scr.eval.pop().unwrap();
                let a = scr.eval.pop().unwrap();
                let inf = (!a.value && a.cost == INF_U32) || (!b.value && b.cost == INF_U32);
                if inf {
                    scr.eval.push(EvalEntry { value: false, cost: INF_U32, ..Default::default() });
                    continue;
                }
                let cost = (if a.value { 0 } else { a.cost })
                    .saturating_add(if b.value { 0 } else { b.cost });
                if a.value && b.value {
                    scr.eval.push(EvalEntry { value: true, cost: 0, ..Default::default() });
                    continue;
                }
                // concat need groups with fp dedup
                let g0 = scr.groups.len() as u32;
                let mut ng = 0u32;
                DBG_AND_GROUPS.fetch_add((a.g_len + b.g_len) as usize, std::sync::atomic::Ordering::Relaxed);
                for e in [&a, &b] {
                    for gi in e.g_start..e.g_start + e.g_len {
                        let gs = scr.groups[gi as usize];
                        let fp = group_fp_slice(&scr.kw, gs.start, gs.len);
                        match scr.group_seen.get(&fp) {
                            Some(&first) if scr.group_slice_eq(first, gs) => { DBG_FP_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                            _ => {
                                scr.group_seen.insert(fp, g0 + ng);
                                scr.groups.push(gs);
                                ng += 1;
                            }
                        }
                    }
                }
                scr.eval.push(EvalEntry {
                    value: false,
                    cost,
                    g_start: g0,
                    g_len: ng,
                    ..Default::default()
                });
            }
            OP_OR => {
                let b = scr.eval.pop().unwrap();
                let a = scr.eval.pop().unwrap();
                if a.value || b.value {
                    scr.eval.push(EvalEntry { value: true, cost: 0, ..Default::default() });
                    continue;
                }
                if a.cost < b.cost {
                    scr.eval.push(a);
                    continue;
                }
                if b.cost < a.cost {
                    scr.eval.push(b);
                    continue;
                }
                // tie at min cost: union when both single, else first
                if a.g_len == 1 && b.g_len == 1 {
                    DBG_UNIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    DBG_UNION_KW.fetch_add((a.k_len + b.k_len) as usize, std::sync::atomic::Ordering::Relaxed);
                    let k0 = scr.kw.len() as u32;
                    scr.kw_seen.clear();
                    let mut fresh: Vec<&'static str> = Vec::new();
                    for e in [&a, &b] {
                        for kw in &scr.kw[e.k_start as usize..(e.k_start + e.k_len) as usize] {
                            if scr.kw_seen.insert(str_fp(kw)) {
                                fresh.push(*kw);
                            }
                        }
                    }
                    let nk = fresh.len() as u32;
                    scr.kw.extend(fresh);
                    let g = scr.groups.len() as u32;
                    scr.groups.push(GroupSlice { start: k0, len: nk });
                    scr.eval.push(EvalEntry {
                        value: false,
                        cost: a.cost,
                        g_start: g,
                        g_len: 1,
                        k_start: k0,
                        k_len: nk,
                    });
                } else {
                    scr.eval.push(a);
                }
            }
            OP_ANDN => {
                let k = op.payload as usize;
                let n = scr.eval.len();
                let mut kids = std::mem::take(&mut scr.kids);
                kids.clear();
                kids.extend(scr.eval.drain(n - k..));
                let mut inf = false;
                let mut cost = 0u32;
                for e in &kids {
                    if !e.value {
                        if e.cost == INF_U32 {
                            inf = true;
                            break;
                        }
                        cost = cost.saturating_add(e.cost);
                    }
                }
                if inf {
                    scr.eval.push(EvalEntry { value: false, cost: INF_U32, ..Default::default() });
                } else if cost == 0 {
                    scr.eval.push(EvalEntry { value: true, cost: 0, ..Default::default() });
                } else {
                    // concat all children's groups, one dedup pass. The fp
                    // dedup is scoped to THIS AndN op (duplicates across
                    // siblings), not across the block: a persistent set would
                    // empty later ANDs whose groups were seen earlier.
                    scr.group_seen.clear();
                    let g0 = scr.groups.len() as u32;
                    let mut ng = 0u32;
                    for e in &kids {
                        for gi in e.g_start..e.g_start + e.g_len {
                            let gs = scr.groups[gi as usize];
                            let fp = group_fp_slice(&scr.kw, gs.start, gs.len);
                            if let Some(&first) = scr.group_seen.get(&fp) {
                                if scr.group_slice_eq(first, gs) {
                                    continue;
                                }
                            }
                            scr.group_seen.insert(fp, g0 + ng);
                            scr.groups.push(gs);
                            ng += 1;
                        }
                    }
                    let (k_start, k_len) = if ng == 1 {
                        let gs = scr.groups[g0 as usize];
                        (gs.start, gs.len)
                    } else {
                        (0, 0)
                    };
                    scr.eval.push(EvalEntry {
                        value: false,
                        cost,
                        g_start: g0,
                        g_len: ng,
                        k_start,
                        k_len,
                    });
                }
                scr.kids = kids;
            }
            OP_ORN => {
                let k = op.payload as usize;
                let n = scr.eval.len();
                let mut kids = std::mem::take(&mut scr.kids);
                kids.clear();
                kids.extend(scr.eval.drain(n - k..));
                let mut any_true = false;
                let mut min_cost = INF_U32;
                let mut best = 0usize;
                for (i, e) in kids.iter().enumerate() {
                    if e.value {
                        any_true = true;
                        break;
                    }
                    if e.cost < min_cost {
                        min_cost = e.cost;
                        best = i;
                    }
                }
                // `best` must be a MIN-COST kid before the tie-break pass:
                // kids[0] may not be min-cost, and comparing group counts
                // against a non-min-cost baseline would never move `best`.
                let mut first_min = 0usize;
                for (i, e) in kids.iter().enumerate() {
                    if e.cost == min_cost {
                        first_min = i;
                        break;
                    }
                }
                best = first_min;
                if any_true {
                    scr.eval.push(EvalEntry { value: true, cost: 0, ..Default::default() });
                    scr.kids = kids;
                    continue;
                }
                if min_cost == INF_U32 {
                    scr.eval.push(EvalEntry { value: false, cost: INF_U32, ..Default::default() });
                    scr.kids = kids;
                    continue;
                }
                // Best min-cost child: fewest groups, then fewest keywords.
                for (i, e) in kids.iter().enumerate() {
                    if e.cost == min_cost && e.g_len < kids[best].g_len {
                        best = i;
                    }
                }
                let all_single = kids
                    .iter()
                    .all(|e| e.cost != min_cost || e.g_len == 1);
                if !all_single {
                    scr.eval.push(kids[best]);
                    scr.kids = kids;
                    continue;
                }
                // ONE union pass over every min-cost single group.
                let k0 = scr.kw.len() as u32;
                scr.kw_seen.clear();
                let mut fresh: Vec<&'static str> = Vec::new();
                for e in kids.iter().filter(|e| e.cost == min_cost) {
                    for kw in &scr.kw[e.k_start as usize..(e.k_start + e.k_len) as usize] {
                        if scr.kw_seen.insert(str_fp(kw)) {
                            fresh.push(*kw);
                        }
                    }
                }
                let nk = fresh.len() as u32;
                scr.kw.extend(fresh);
                let g = scr.groups.len() as u32;
                scr.groups.push(GroupSlice { start: k0, len: nk });
                scr.eval.push(EvalEntry {
                    value: false,
                    cost: min_cost,
                    g_start: g,
                    g_len: 1,
                    k_start: k0,
                    k_len: nk,
                });
                scr.kids = kids;
            }
            _ => unreachable!("unknown op tag"),
        }
    }
    let top = scr.eval.pop().unwrap_or(EvalEntry { value: false, cost: INF_U32, ..Default::default() });
    // Materialize the final need from the arena (keywords are 'static
    // blob refs - zero-copy).
    let mut need: Vec<Vec<&'static str>> = Vec::new();
    for gi in top.g_start..top.g_start + top.g_len {
        let gs = scr.groups[gi as usize];
        need.push(scr.kw[gs.start as usize..(gs.start + gs.len) as usize].to_vec());
    }
    MinAdd { value: top.value, cost: top.cost as usize, need }
}

impl MinAddScratch {
    fn group_slice_eq(&self, gi: u32, gs: GroupSlice) -> bool {
        let first = self.groups[gi as usize];
        if first.len != gs.len {
            return false;
        }
        for k in 0..first.len {
            let a = &self.kw[(first.start + k) as usize];
            let b = &self.kw[(gs.start + k) as usize];
            if !std::ptr::eq(&**a, &**b) {
                return false;
            }
        }
        true
    }
}

/// Near-miss analysis for a whole block (default TITLE-ABS-KEY scope),
/// reusing the caller's per-request `Memo` so leaf verdicts are cached.
pub fn min_add_block<'a>(
    block: &Node,
    table: &[Pattern],
    memo: &mut Memo<'a>,
) -> MinAdd {
    min_add(block, field_mask(&ALL_FIELDS), table, memo)
}

/// (value, cost) twin with NO allocations: the fast path used to rerank all
/// non-matching blocks. The `need` groups are only materialized afterwards
/// for the blocks that make the displayed list (see `min_add_block`).
fn min_add_vc(node: &Node, mask: u8, table: &[Pattern], memo: &mut Memo) -> (bool, usize) {
    match node {
        Node::Leaf { mask: lm, slot, .. } => {
            let v = memo.term_hit(pat(table, node), *lm, *slot as usize);
            (v, if v { 0 } else { 1 })
        }
        Node::Field { fields, child } => {
            min_add_vc(child, field_mask_from_strings(fields), table, memo)
        }
        Node::Not { child } => {
            let (v, _) = min_add_vc(child, mask, table, memo);
            if v {
                (false, INF_COST)
            } else {
                (true, 0)
            }
        }
        Node::Group { op, children } => {
            if op == "OR" {
                let mut min = INF_COST;
                for c in children {
                    let (v, cost) = min_add_vc(c, mask, table, memo);
                    if v {
                        return (true, 0);
                    }
                    if cost < min {
                        min = cost;
                    }
                }
                (false, min)
            } else {
                let mut cost = 0usize;
                for c in children {
                    let (v, cst) = min_add_vc(c, mask, table, memo);
                    if !v && cst == INF_COST {
                        return (false, INF_COST);
                    }
                    if !v {
                        cost = cost.saturating_add(cst);
                    }
                }
                (cost == 0, cost)
            }
        }
    }
}

/// Minimum keywords to add for a whole block - cost only, no allocation.
pub fn min_add_block_cost<'a>(
    block: &Node,
    table: &[Pattern],
    memo: &mut Memo<'a>,
) -> usize {
    min_add_vc(block, field_mask(&ALL_FIELDS), table, memo).1
}

/// Evaluate a block with every `NOT` clause REMOVED: true iff the block's
/// positive (include) side alone would match. Used to decide whether excluded
/// terms genuinely *disqualify* a block - i.e. the paper already satisfies
/// everything else and only a NOT clause stands in the way. When this is
/// false the block is simply off-topic, so its NOT-leaf hits are noise and
/// must not be reported as "can disqualify". Semantics: a removed NOT child
/// is the identity element of its group (true under AND, absent under OR).
pub fn eval_ignore_not_block<'a>(
    block: &Node,
    table: &[Pattern],
    memo: &mut Memo<'a>,
) -> bool {
    fn go(node: &Node, mask: u8, table: &[Pattern], memo: &mut Memo) -> bool {
        match node {
            Node::Leaf { mask: lm, slot, .. } => {
                memo.term_hit(pat(table, node), *lm, *slot as usize)
            }
            Node::Field { fields, child } => go(child, field_mask_from_strings(fields), table, memo),
            Node::Not { .. } => true, // clause removed: identity
            Node::Group { op, children } => {
                let kids: Vec<&Node> =
                    children.iter().filter(|c| !matches!(c, Node::Not { .. })).collect();
                if op == "OR" {
                    kids.iter().any(|c| go(c, mask, table, memo))
                } else {
                    kids.iter().all(|c| go(c, mask, table, memo))
                }
            }
        }
    }
    go(block, field_mask(&ALL_FIELDS), table, memo)
}

// ---------------------------------------------------------------------------
// Keyword suggestions ("best-fit keywords to add") - deterministic, no LLM
//
// Every SDG is an OR of keyword blocks; a paper qualifies as soon as ONE
// include keyword is present. To help pick the right keyword we rank an
// SDG's unique include keywords by word-token overlap with the paper text:
// score = (keyword tokens found in the text) / (keyword tokens). Tokens are
// lowercased and pre-tokenized ONCE at boot ("pretokenize"): requests only
// do set lookups, never re-derive tokens. `*`/`?` wildcards match text words
// by prefix / literal-substring. Keywords the paper already contains are
// flagged (advanced tab) or skipped (report suggestions), and keywords that
// are also excluded (NOT) leaves somewhere in the same SDG are flagged.
// ---------------------------------------------------------------------------

/// Keyword dictionary of one SDG (leaf level), pre-tokenized at boot so the
/// per-request suggestion scoring is allocation-free.
pub struct SdgDict {
    /// (keyword, lowercased overlap tokens) for every unique include
    /// keyword. Keywords are `Arc<str>` so scoring clones a cheap refcount
    /// instead of a String; tokens are lowercased once at boot ("pretokenize")
    /// so requests only do set lookups over them.
    pub keywords: Vec<(Arc<str>, Vec<String>)>,
    pub excluded: HashSet<String>,
}

impl SdgDict {
    /// Number of unique include keywords.
    pub fn len(&self) -> usize {
        self.keywords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keywords.is_empty()
    }
}

/// One suggested keyword for a paper.
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub keyword: Arc<str>,
    /// Fraction of the keyword's word tokens found in the paper text (0..=1).
    pub score: f32,
    /// The keyword also appears under a NOT clause somewhere in this SDG.
    pub excluded_in_sdg: bool,
}

/// One scored keyword of an SDG against a paper (advanced browser view).
#[derive(Debug, Clone)]
pub struct ScoredKw {
    pub keyword: Arc<str>,
    /// Fraction of the keyword's word tokens found in the paper text (0..=1).
    pub score: f32,
    /// The keyword is already present in the paper (it is a hit right now).
    pub present: bool,
    /// The keyword also appears under a NOT clause somewhere in this SDG.
    pub excluded_in_sdg: bool,
}

fn collect_leafs(node: &Node, excluded: bool, inc: &mut HashSet<String>, exc: &mut HashSet<String>) {
    match node {
        Node::Leaf { keyword, .. } => {
            if excluded {
                exc.insert(keyword.clone());
            } else {
                inc.insert(keyword.clone());
            }
        }
        Node::Field { child, .. } => collect_leafs(child, excluded, inc, exc),
        Node::Not { child } => collect_leafs(child, !excluded, inc, exc),
        Node::Group { children, .. } => {
            for c in children {
                collect_leafs(c, excluded, inc, exc);
            }
        }
    }
}

/// Build the keyword dictionary of an SDG from its query blocks.
pub fn collect_sdg_dict(blocks: &[Node]) -> SdgDict {
    let mut inc: HashSet<String> = HashSet::new();
    let mut exc: HashSet<String> = HashSet::new();
    for b in blocks {
        collect_leafs(b, false, &mut inc, &mut exc);
    }
    let mut include: Vec<String> = inc.into_iter().collect();
    include.sort_unstable();
    let keywords = include
        .into_iter()
        .map(|kw| {
            let all: Vec<String> = kw
                .split(' ')
                .filter(|t| !t.is_empty())
                .map(|t| t.to_lowercase())
                .collect();
            let meaningful: Vec<String> = all.iter().filter(|t| t.len() > 2).cloned().collect();
            // Overlap tokens: ignore stopword-ish tokens unless the phrase is
            // all stopwords (short phrases still score).
            let tokens = if meaningful.is_empty() { all } else { meaningful };
            (Arc::from(kw), tokens)
        })
        .collect();
    SdgDict { keywords, excluded: exc }
}

/// Paper word index: a set for exact lookups plus a sorted slice for
/// prefix (wildcard) lookups via binary search. Built ONCE per request and
/// shared by every SDG's scoring; memory layout matches the access pattern
/// (dense, sorted, cache-friendly) at the cost of a little extra RAM.
pub struct PaperWords<'a> {
    set: HashSet<&'a str>,
    sorted: Vec<&'a str>,
}

/// Is the UTF-8 char starting at byte `i` alphanumeric? ASCII takes a
/// branchless table lookup; non-ASCII decodes one char (exact parity with
/// `char::is_alphanumeric`). Callers guarantee `i < text.len()`.
#[inline]
fn alnum_at(b: &[u8], i: usize) -> bool {
    let c = b[i];
    if c < 0x80 {
        c.is_ascii_alphanumeric()
    } else {
        // SAFETY: `i` is a char boundary inside a valid UTF-8 buffer (the
        // scanner only steps by full char widths from a &str boundary).
        unsafe { std::str::from_utf8_unchecked(&b[i..]) }
            .chars()
            .next()
            .unwrap()
            .is_alphanumeric()
    }
}

/// Byte width of the UTF-8 char starting at `i` (ASCII = 1).
#[inline]
fn char_len_at(b: &[u8], i: usize) -> usize {
    let c = b[i];
    if c < 0x80 {
        1
    } else if c >> 5 == 0b110 {
        2
    } else if c >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

pub fn text_words(text: &str) -> PaperWords<'_> {
    let b = text.as_bytes();
    let n = b.len();

    // One pass: insert every alphanumeric run into a growing set. A run ==
    // one word token, identical to the reference
    // `split(|c: char| !c.is_alphanumeric())` splitter: ASCII bytes take the
    // branchless table path, non-ASCII chars are decoded so the token set is
    // byte-for-byte the same. The set grows naturally: preallocating for the
    // token *count* is wrong for repetitive text (300k tokens but ~700
    // unique words -> a multi-MB zeroed table for nothing).
    let mut set: HashSet<&str> = HashSet::new();
    let mut i = 0usize;
    while i < n {
        if alnum_at(b, i) {
            let start = i;
            while i < n && alnum_at(b, i) {
                i += char_len_at(b, i);
            }
            set.insert(&text[start..i]);
        } else {
            i += char_len_at(b, i);
        }
    }
    let mut sorted: Vec<&str> = set.iter().copied().collect();
    sorted.sort_unstable();
    PaperWords { set, sorted }
}

fn token_matches(tok: &str, words: &PaperWords) -> bool {
    // `tok` is already lowercased (pre-tokenized at boot).
    if tok.ends_with('*') {
        let p = tok.trim_end_matches('*').trim_end_matches(|c: char| !c.is_alphanumeric());
        if p.is_empty() {
            return false;
        }
        // Prefix range [p, p+) via binary search over the sorted words.
        let lo = words.sorted.partition_point(|w| w.as_bytes() < p.as_bytes());
        return words.sorted.get(lo).is_some_and(|w| w.starts_with(p));
    }
    if tok.contains('*') || tok.contains('?') {
        let literal: String = tok.chars().filter(|c| c.is_alphanumeric()).collect();
        if literal.is_empty() {
            return false;
        }
        return words.sorted.iter().any(|w| w.contains(literal.as_str()));
    }
    let stripped = tok.trim_matches(|c: char| !c.is_alphanumeric());
    !stripped.is_empty() && words.set.contains(stripped)
}

/// Score every unique include keyword of an SDG against the paper's word
/// index, best score first, at most `limit` entries. `present` = keywords
/// already hit by the paper (kept in the list, flagged). Tokens were
/// lowercased when the dict was built, so scoring is pure set lookups; the
/// keyword clone is a cheap `Arc` refcount bump (RAM for speed).
pub fn score_keywords(
    words: &PaperWords,
    dict: &SdgDict,
    present: &HashSet<&str>,
    limit: usize,
) -> Vec<ScoredKw> {
    let mut out: Vec<ScoredKw> = Vec::with_capacity(dict.keywords.len().min(limit + 64));
    for (kw, tokens) in &dict.keywords {
        let hit = tokens.iter().filter(|t| token_matches(t, words)).count();
        out.push(ScoredKw {
            keyword: kw.clone(),
            score: if tokens.is_empty() { 0.0 } else { hit as f32 / tokens.len() as f32 },
            present: present.contains(kw.as_ref()),
            excluded_in_sdg: dict.excluded.contains(kw.as_ref()),
        });
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.keyword.cmp(&b.keyword))
    });
    out.truncate(limit);
    out
}

/// Ordering for the bounded top-N heap: lower score, then higher keyword
/// (so the heap keeps the `limit` BEST suggestions and drains them best-first).
struct Cand {
    score: u32,
    keyword: Arc<str>,
    excluded: bool,
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| other.keyword.cmp(&self.keyword))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.keyword == other.keyword
    }
}
impl Eq for Cand {}

/// Best-fit keywords to add: streams every include keyword through a bounded
/// min-heap (limit entries, no full-list allocation/sort), skipping keywords
/// the paper already contains and keywords with zero token overlap. Score is
/// percent (0..=100) for integer-only heap comparisons.
pub fn suggest_keywords(
    words: &PaperWords,
    dict: &SdgDict,
    present: &HashSet<&str>,
    limit: usize,
) -> Vec<Suggestion> {
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(limit + 1);
    for (kw, tokens) in &dict.keywords {
        if present.contains(kw.as_ref()) {
            continue;
        }
        let hit = tokens.iter().filter(|t| token_matches(t, words)).count();
        if hit == 0 || tokens.is_empty() {
            continue;
        }
        let pct = ((hit as f32 / tokens.len() as f32) * 100.0).round() as u32;
        if pct == 0 {
            continue;
        }
        let cand = Cand {
            score: pct,
            keyword: kw.clone(),
            excluded: dict.excluded.contains(kw.as_ref()),
        };
        if heap.len() < limit {
            heap.push(cand);
        } else if cand < *heap.peek().unwrap() {
            heap.pop();
            heap.push(cand);
        }
    }
    heap
        .into_sorted_vec()
        .into_iter()
        .map(|c| Suggestion {
            keyword: c.keyword,
            score: c.score as f32 / 100.0,
            excluded_in_sdg: c.excluded,
        })
        .collect()
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
) -> (Vec<(&'static str, u8)>, Vec<(&'static str, u8)>, Vec<&'static str>, bool) {
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
    hits: &mut Vec<(&'static str, u8)>,
    misses: &mut Vec<(&'static str, u8)>,
    ex_hits: &mut Vec<&'static str>,
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
                        ex_hits.push(p.raw());
                    }
                }
            } else if found {
                #[cfg(feature = "prof")]
                {
                    use std::sync::atomic::Ordering;
                    prof::REPORT_PUSHES.fetch_add(1, Ordering::Relaxed);
                }
                if report {
                    hits.push((p.raw(), *mask));
                }
            } else {
                #[cfg(feature = "prof")]
                {
                    use std::sync::atomic::Ordering;
                    prof::REPORT_PUSHES.fetch_add(1, Ordering::Relaxed);
                }
                if report {
                    misses.push((p.raw(), *mask));
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
                    out.excluded_hits.push(p.raw());
                }
            } else if found {
                out.hits.push(p.raw());
            } else {
                out.misses.push(p.raw());
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


// ---------------------------------------------------------------------------
// Boot cache serialization
//
// Recompiling ~21k keyword patterns at every boot (web server start, each CLI
// invocation) costs ~70-80 ms. We persist the precomputed patterns, flattened
// blocks and pretokenized SDG dictionaries in a compact binary cache (see
// cache.rs); boot then reads the file (a few ms) instead of re-deriving
// everything. The cache is validated by the query files' mtimes, so it is
// always consistent with the sources. Memory layout is length-prefixed and
// sequential - written once, read as a linear scan.
// ---------------------------------------------------------------------------

fn write_u32<W: std::io::Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32<R: std::io::Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_bytes<W: std::io::Write>(w: &mut W, b: &[u8]) -> std::io::Result<()> {
    write_u32(w, b.len() as u32)?;
    w.write_all(b)
}

fn read_bytes<R: std::io::Read>(r: &mut R) -> std::io::Result<Vec<u8>> {
    let n = read_u32(r)? as usize;
    if n > (1 << 28) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "cache: oversized field"));
    }
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    Ok(v)
}

impl Pattern {
    /// Record layout: 6 x u32 offsets/lengths + 1 flag byte + 3 pad bytes
    /// (28 bytes total - matches `size_of::<Pattern>()` so the boot cache
    /// can be viewed as `&[Pattern]` directly from the mmap).
    pub fn serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_u32(w, self.raw_off)?;
        write_u32(w, self.raw_len)?;
        write_u32(w, self.lower_off)?;
        write_u32(w, self.lower_len)?;
        write_u32(w, self.parts_off)?;
        write_u32(w, self.parts_len)?;
        w.write_all(&[self.no_wildcard as u8, 0, 0, 0])?;
        Ok(())
    }

    pub fn deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Pattern> {
        let raw_off = read_u32(r)?;
        let raw_len = read_u32(r)?;
        let lower_off = read_u32(r)?;
        let lower_len = read_u32(r)?;
        let parts_off = read_u32(r)?;
        let parts_len = read_u32(r)?;
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;
        Ok(Pattern { raw_off, raw_len, lower_off, lower_len, parts_off, parts_len, no_wildcard: flag[0] != 0 })
    }
}

impl Op {
    fn serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_u32(w, self.tag)?;
        write_u32(w, self.payload)
    }

    fn deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Op> {
        Ok(Op { tag: read_u32(r)?, payload: read_u32(r)? })
    }
}

impl FlatBlock {
    /// Fixed-record write: op records (8 bytes) + leaf records (20 bytes).
    pub fn serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_u32(w, self.prog.len() as u32)?;
        for op in self.prog {
            op.serialize(w)?;
        }
        write_u32(w, self.leaves.len() as u32)?;
        for l in self.leaves {
            write_u32(w, l.pid)?;
            write_u32(w, l.slot)?;
            w.write_all(&[l.mask, l.excluded as u8, 0, 0])?; // 20-byte records
            write_u32(w, l.raw_off)?;
            write_u32(w, l.raw_len)?;
        }
        Ok(())
    }

    pub fn deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<FlatBlock> {
        let np = read_u32(r)? as usize;
        let mut prog = Vec::with_capacity(np.min(1 << 20));
        for _ in 0..np {
            prog.push(Op::deserialize(r)?);
        }
        let nl = read_u32(r)? as usize;
        let mut leaves = Vec::with_capacity(nl.min(1 << 20));
        for _ in 0..nl {
            let pid = read_u32(r)?;
            let slot = read_u32(r)?;
            let mut m = [0u8; 2];
            r.read_exact(&mut m)?;
            let raw_off = read_u32(r)?;
            let raw_len = read_u32(r)?;
            leaves.push(LeafDesc { pid, slot, mask: m[0], excluded: m[1] != 0, raw_off, raw_len });
        }
        Ok(FlatBlock {
            prog: Box::leak(prog.into_boxed_slice()),
            leaves: Box::leak(leaves.into_boxed_slice()),
        })
    }
}

impl SdgDict {
    pub fn serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_u32(w, self.keywords.len() as u32)?;
        for (kw, toks) in &self.keywords {
            write_bytes(w, kw.as_bytes())?;
            write_u32(w, toks.len() as u32)?;
            for t in toks {
                write_bytes(w, t.as_bytes())?;
            }
        }
        write_u32(w, self.excluded.len() as u32)?;
        let mut exc: Vec<&str> = self.excluded.iter().map(String::as_str).collect();
        exc.sort_unstable();
        for e in exc {
            write_bytes(w, e.as_bytes())?;
        }
        Ok(())
    }

    pub fn deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<SdgDict> {
        let n = read_u32(r)? as usize;
        let mut keywords = Vec::with_capacity(n.min(1 << 20));
        for _ in 0..n {
            let kw = String::from_utf8(read_bytes(r)?)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "cache: bad keyword"))?;
            let nt = read_u32(r)? as usize;
            let mut toks = Vec::with_capacity(nt.min(64));
            for _ in 0..nt {
                let t = String::from_utf8(read_bytes(r)?)
                    .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "cache: bad token"))?;
                toks.push(t);
            }
            keywords.push((Arc::from(kw), toks));
        }
        let ne = read_u32(r)? as usize;
        let mut excluded = HashSet::with_capacity(ne.min(1 << 20));
        for _ in 0..ne {
            let e = String::from_utf8(read_bytes(r)?)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "cache: bad exclude"))?;
            excluded.insert(e);
        }
        Ok(SdgDict { keywords, excluded })
    }
}

/// Rebuild the global FIRST_QUADS set from loaded patterns (normally done
/// inside `compile_all`; needed when patterns come from the boot cache).
pub fn rebuild_first_quads(patterns: &[Pattern]) {
    let mut g = FIRST_QUADS.lock().unwrap();
    let s = g.get_or_insert_with(|| Arc::new(HashSet::with_hasher(FastHasher::default())));
    let s = Arc::make_mut(s);
    for p in patterns {
        for part in p.parts() {
            if part.len() >= 4 {
                s.insert(u32::from_le_bytes([part[0], part[1], part[2], part[3]]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Shared test blob: patterns built by `compile_pattern` append their
    /// strings here (pre-reserved, never reallocates); `set_blob` repoints
    /// the global (ptr, len) at each call so all test patterns stay valid.
    static TEST_BLOB: Mutex<Option<&'static mut Vec<u8>>> = Mutex::new(None);
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Serialize the blob-dependent tests (the global blob is process-wide)
    /// and give each test a fresh blob so offsets never go stale.
    fn reset_test_blob() -> std::sync::MutexGuard<'static, ()> {
        let lock = TEST_LOCK.lock().unwrap();
        let mut g = TEST_BLOB.lock().unwrap();
        let mut v: Box<Vec<u8>> = Box::new(Vec::with_capacity(1 << 20));
        *g = Some(Box::leak(v));
        let b = g.as_ref().unwrap();
        set_blob(unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) });
        lock
    }

    fn compile_pattern(kw: &str) -> Pattern {
        let mut g = TEST_BLOB.lock().unwrap();
        if g.is_none() {
            let mut v: Box<Vec<u8>> = Box::new(Vec::with_capacity(1 << 20));
            *g = Some(Box::leak(v));
        }
        let b = g.as_mut().unwrap();
        let kw = kw.trim();
        let raw_off = b.len() as u32;
        b.extend_from_slice(kw.as_bytes());
        let raw_len = kw.len() as u32;
        let lower = kw.to_ascii_lowercase();
        let has_star = lower.contains('*');
        let has_q = lower.contains('?');
        let (no_wildcard, parts_bytes): (bool, Vec<Vec<u8>>) = if !has_star && !has_q {
            (true, vec![lower.as_bytes().to_vec()])
        } else if !has_q {
            (false, lower.split('*').filter(|p| !p.is_empty()).map(|p| p.as_bytes().to_vec()).collect())
        } else {
            (false, Vec::new())
        };
        let mut parts = Vec::with_capacity(parts_bytes.len());
        for pb in parts_bytes {
            let off = b.len() as u32;
            b.extend_from_slice(&pb);
            parts.push((off, pb.len() as u32));
        }
        let (lower_off, lower_len) = if has_q {
            let off = b.len() as u32;
            b.extend_from_slice(lower.as_bytes());
            (off, lower.len() as u32)
        } else {
            (0, 0)
        };
        let parts_off = b.len() as u32;
        for &(off, len) in &parts {
            b.extend_from_slice(&off.to_le_bytes());
            b.extend_from_slice(&len.to_le_bytes());
        }
        set_blob(unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) });
        Pattern { raw_off, raw_len, lower_off, lower_len, parts_off, parts_len: parts.len() as u32, no_wildcard }
    }

    /// The flattened-block path must produce byte-identical reports (verdict,
    /// hits, misses, excluded) to the AST tree walk, for every block of every
    /// SDG query and every paper in the repo.
    #[test]
    fn flat_matches_ast_scan() {
        let _lock = reset_test_blob();
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

                    assert_eq!(flat.3, ast.3, "verdict mismatch {p} q{qi} b{bi}");
                    assert_eq!(flat.0, ast.0, "hits mismatch {p} q{qi} b{bi}");
                    assert_eq!(flat.1, ast.1, "misses mismatch {p} q{qi} b{bi}");
                    assert_eq!(
                        flat.2.into_iter().map(String::from).collect::<Vec<_>>(),
                        ast.2.into_iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                        "excluded mismatch {p} q{qi} b{bi}"
                    );
                }
            }
        }
    }

    /// Frontmatter + body papers take the `full_covers_sections = false`
    /// path (per-field buffers, body is the full text). Flat and AST scans
    /// must still agree, and a body-only term must not leak into a
    /// TITLE-scoped search.
    #[test]
    fn flat_matches_ast_with_body_text() {
        let _lock = reset_test_blob();
        use crate::query::{load_queries, Query};
        use std::path::Path;

        let text = "---
title: \"Climate finance and energy poverty\"
abstract: |
  We study climate finance flows to developing countries.
keywords: [energy poverty]
---
This is the body. It discusses carbon markets and debt relief at length.";
        let paper = Paper::from_text(text);
        assert!(!paper.full_covers_sections, "body text must disable full-text folding");

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
        let mut memo = Memo::new(&paper, nslots);
        for (qi, q) in queries.iter().enumerate() {
            for (bi, b) in q.blocks.iter().enumerate() {
                let ast = scan_with_fields(b, &paper, &table, &mut memo);
                let flat = scan_flat(&flats[qi][bi], &table, &mut memo);
                assert_eq!(flat.3, ast.3, "verdict mismatch q{qi} b{bi}");
                assert_eq!(
                    flat.0.iter().map(|(s, m)| ((*s).to_owned(), *m)).collect::<Vec<_>>(),
                    ast.0.iter().map(|(s, m)| (s.to_string(), *m)).collect::<Vec<_>>(),
                    "hits mismatch q{qi} b{bi}"
                );
                assert_eq!(
                    flat.1.iter().map(|(s, m)| ((*s).to_owned(), *m)).collect::<Vec<_>>(),
                    ast.1.iter().map(|(s, m)| (s.to_string(), *m)).collect::<Vec<_>>(),
                    "misses mismatch q{qi} b{bi}"
                );
            }
        }

        // A TITLE-scoped term present only in the body must NOT match.
        let mut q = Query { sdg: "t".into(), blocks: Vec::new() };
        let root = crate::parser::Parser::new(crate::tokenizer::tokenize("TITLE(carbon markets)").unwrap())
            .parse()
            .unwrap();
        q.blocks = match &root {
            crate::ast::Node::Group { op, children } if op == "OR" => children.clone(),
            _ => vec![root],
        };
        let table2 = compile_all(q.blocks.iter());
        let mut nslots2 = 0u32;
        resolve_blocks(&mut q.blocks, &table2, &mut nslots2);
        let f = flatten_block(&q.blocks[0], &table2);
        let mut memo2 = Memo::new(&paper, nslots2);
        assert!(!scan_flat(&f, &table2, &mut memo2).3, "body text leaked into TITLE scope");
    }

    /// `matches_indexed` (quad positions) must agree with `matches` (full
    /// scan) across boundary and wildcard edge cases.
    #[test]
    fn matches_indexed_equiv_matches() {
        let _lock = reset_test_blob();
        let cases = [
            ("foreign aid", "foreign aid in developing countries", true),
            ("foreign aid", "xforeign aid", false),
            ("foreign aid", "foreign aidy", false),
            ("foreign aid", "foreign-aid policies", false),
            ("foreign aid", "  foreign aid.", true),
            ("foreign aid", "foreign  aid", false),
            ("aa", "xaa aa", true),
            ("aa", "aaa", false),
            ("aa", "aa_", false),
            ("aa", "aa x", true),
            ("developing* countr*", "studies in developing countries", true),
            ("developing* countr*", "developing countries", true),
            ("developing* countr*", "developing and countries", false),
            ("developing*countr*", "developing_countries", true),
            ("developing*countr*", "developing countries", false),
            ("poverty*-reducing*", "poverty-reducing policies", true),
            ("poverty*-reducing*", "povertyreducing policies", false),
            ("a* b", "a b", true),
            ("a* b", "a b c b", true),
            ("a* b* c", "a b c", true),
        ];
        for (pat, text, want) in cases {
            let p = compile_pattern(pat);
            let needed: HashSet<u32, FastHasher> = p
                .parts()
                .filter(|x| x.len() >= 4)
                .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect();
            let idx = TextIndex::build_with(text.as_bytes(), &needed);
            assert_eq!(p.matches(text.as_bytes()), want, "scan {pat:?} vs {text:?}");
            assert_eq!(p.matches_indexed(text.as_bytes(), &idx), want, "indexed {pat:?} vs {text:?}");
        }
    }

    /// First quad occurs 500+ times before the real match: candidate
    /// verification must stay exhaustive (no cap that drops the match).
    #[test]
    fn matches_indexed_common_quad_verifies_all() {
        let _lock = reset_test_blob();
        let mut txt = String::new();
        for _ in 0..600 {
            txt.push_str("forest ");
        }
        txt.push_str("foreign aid matters");
        let p = compile_pattern("foreign aid");
        let needed: HashSet<u32, FastHasher> = [u32::from_le_bytes(*b"fore")].into_iter().collect();
        let idx = TextIndex::build_with(txt.as_bytes(), &needed);
        assert!(idx.positions(u32::from_le_bytes(*b"fore")).unwrap().len() > 512);
        assert!(p.matches_indexed(txt.as_bytes(), &idx));
        // Absent quad: hard false without a scan.
        let idx = TextIndex::build_with(b"zzz zzz".as_slice(), &first_quads());
        assert!(!p.matches_indexed(b"zzz zzz", &idx));
    }

    /// Repeated text (every quad present) must not degenerate into a scan
    /// per term: the positions gate answers from the index.
    #[test]
    fn matches_indexed_repetitive_text() {
        let _lock = reset_test_blob();
        let txt = "tax evasion in developing countries. ".repeat(4000);
        let p = compile_pattern("quantum computing");
        let needed: HashSet<u32, FastHasher> = [u32::from_le_bytes(*b"quan")].into_iter().collect();
        let idx = TextIndex::build_with(txt.as_bytes(), &needed);
        assert!(idx.positions(u32::from_le_bytes(*b"quan")).is_none());
        assert!(!p.matches_indexed(txt.as_bytes(), &idx));
    }

    /// Edge papers that stress the folded-buffer logic the repo papers do
    /// not: '?'-glob matches and cross-field non-matches, '*' parts split
    /// across fields, unicode bytes, empty and whitespace-only papers, and
    /// a large body. Flat and AST scans must agree block-for-block, and
    /// cross-field matches must not happen.
    #[test]
    fn flat_matches_ast_edge_papers() {
        let _lock = reset_test_blob();
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

        let mut check = |text: &str, label: &str| {
            let paper = Paper::from_text(text);
            let mut memo = Memo::new(&paper, nslots);
            for (qi, q) in queries.iter().enumerate() {
                for (bi, b) in q.blocks.iter().enumerate() {
                    let ast = scan_with_fields(b, &paper, &table, &mut memo);
                    let flat = scan_flat(&flats[qi][bi], &table, &mut memo);
                    assert_eq!(flat.3, ast.3, "{label}: verdict mismatch q{qi} b{bi}");
                    assert_eq!(
                        flat.0.iter().map(|(s, m)| ((*s).to_owned(), *m)).collect::<Vec<_>>(),
                        ast.0.iter().map(|(s, m)| (s.to_string(), *m)).collect::<Vec<_>>(),
                        "{label}: hits mismatch q{qi} b{bi}"
                    );
                    assert_eq!(
                        flat.1.iter().map(|(s, m)| ((*s).to_owned(), *m)).collect::<Vec<_>>(),
                        ast.1.iter().map(|(s, m)| (s.to_string(), *m)).collect::<Vec<_>>(),
                        "{label}: misses mismatch q{qi} b{bi}"
                    );
                }
            }
        };

        // '?' patterns from the corpus: "small?sc*", "conditional ?-convergence",
        // "Accoya? wood" - all in one paper, plus unicode and a large body.
        let mut big = String::with_capacity(24 << 10);
        for _ in 0..120 {
            big.push_str("smallscale farming and conditional convergence in Accoya wood production. ");
        }
        check(
            &format!(
                "---
title: \"Accoya wood from smallscale farms\"
abstract: |
  Conditional -convergence is observed for smallscale farms growing Accoya wood.
  São Tomé and Curaçao report similar patterns.
keywords: [Accoya wood]
---
{big}body terms: developing countries, foreign aid, poverty reduction, climate change.",
            ),
            "glob+unicode+large",
        );
        // Cross-field non-matches: '?' and '*' parts split across fields.
        check(
            "---
title: \"small\"
abstract: |
  scale farming is discussed here at length with developing countries.
keywords: [convergence]
",
            "cross-field glob",
        );
        check(
            "---
title: \"developing\"
abstract: |
  countries need aid. Convergence is not conditional.
",
            "cross-field star",
        );
        // Empty and whitespace-only papers.
        check("", "empty");
        check("   \n\t  ", "whitespace");
        check("---\ntitle: \"x\"\n---\n", "title only");
    }

    /// '?'-globs match ANY substring (Python re.search parity), '?' matches
    /// any byte including newlines, '*' any run including newlines.
    #[test]
    fn glob_matches_substrings() {
        assert!(glob_match(b"abc", b"abc"));
        assert!(glob_match(b"abc", b"xabc"));
        assert!(glob_match(b"abc", b"xabcyy"));
        assert!(!glob_match(b"abc", b"abd"));
        assert!(!glob_match(b"abc", b"ab"));
        assert!(glob_match(b"a?c", b"xxabc"));
        assert!(!glob_match(b"a?c", b"xxac"));
        assert!(glob_match(b"a?b", b"a\nb")); // '?' = any byte (DOTALL)
        assert!(glob_match(b"a*c", b"ac"));
        assert!(glob_match(b"a*c", b"xaZZZcy"));
        assert!(!glob_match(b"a*c", b"ca"));
        assert!(glob_match(b"a*b", b"xaaybz")); // star backtracking
        assert!(glob_match(b"*", b"anything"));
        assert!(glob_match(b"", b""));
        assert!(glob_match(b"", b"x"));
        // The three corpus '?' patterns against realistic text.
        assert!(glob_match(b"conditional ?-convergence", b"herds sustain smallxscales. conditional --convergence is observed"));
        assert!(glob_match(b"small?sc*", b"herds sustain smallxscales of production"));
        assert!(glob_match(b"accoya? wood", b"accoyax wood remains a niche"));
        assert!(!glob_match(b"accoya? wood", b"accoya wood")); // needs a char before the space
        assert!(!glob_match(b"small?sc*", b"smallscale")); // needs small + X + sc
    }

    /// Seeded xorshift64 PRNG so randomized tests are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
    }

    /// Randomized papers (seeded) with random field combinations, joins and
    /// pattern-interacting tokens: flat and AST scans must agree on every
    /// block. Catches fold/segment/glob interactions the hand-crafted papers
    /// miss.
    #[test]
    fn random_papers_flat_matches_ast() {
        let _lock = reset_test_blob();
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

        let toks = [
            "small", "scale", "smallxscales", "smallscale", "accoya", "accoyax", "wood",
            "conditional", "--convergence", "convergence", "herds", "herd", "sustain",
            "sustainable", "developing", "countries", "tax", "evasion", "poverty", "poor",
            "water", "cattle", "treatment", "climate", "change", "coral", "reef", "foreign",
            "aid", "gender", "inequalit", "energy", "renewable", "ocean", "acidification",
            "growth", "gdp", "income", "rights", "a", "b", "x", "ab", "São", "Tomé", "??",
            "?", "*", "---", "a-b", "x y",
        ];
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut trials = 0usize;
        let mut body_buf = String::new();
        let mut abstract_buf = String::new();
        let mut title_buf = String::new();
        let mut kw_buf = String::new();
        for _ in 0..30 {
            let mut words = |n: usize, buf: &mut String, rng: &mut Rng| {
                buf.clear();
                for k in 0..n {
                    if k > 0 && rng.below(5) == 0 {
                        buf.push_str(if rng.below(2) == 0 { "\n" } else { "  " });
                    } else if k > 0 {
                        buf.push(' ');
                    }
                    buf.push_str(toks[rng.below(toks.len())]);
                }
            };
            words(1 + rng.below(6), &mut title_buf, &mut rng);
            let has_abs = rng.below(2) == 0;
            let has_kw = rng.below(2) == 0;
            let has_body = rng.below(3) == 0;
            if has_abs {
                words(2 + rng.below(20), &mut abstract_buf, &mut rng);
            }
            if has_kw {
                words(1 + rng.below(4), &mut kw_buf, &mut rng);
            }
            if has_body {
                body_buf.clear();
                body_buf.push_str(&format!("---\n"));
                words(5 + rng.below(60), &mut body_buf, &mut rng);
            }
            let mut paper = format!("---\ntitle: \"{}\"\n", title_buf);
            if has_abs {
                paper.push_str(&format!("abstract: |\n  {}\n", abstract_buf));
            }
            if has_kw {
                paper.push_str(&format!("keywords: [{}]\n", kw_buf));
            }
            paper.push_str("---\n");
            if has_body {
                paper.push_str(&body_buf);
            }

            let p = Paper::from_text(&paper);
            let mut memo = Memo::new(&p, nslots);
            for (qi, q) in queries.iter().enumerate() {
                for (bi, b) in q.blocks.iter().enumerate() {
                    let ast = scan_with_fields(b, &p, &table, &mut memo);
                    let flat = scan_flat(&flats[qi][bi], &table, &mut memo);
                    assert_eq!(flat.3, ast.3, "trial {trials} q{qi} b{bi}: verdict");
                    assert_eq!(
                        flat.0.iter().map(|(s, m)| ((*s).to_owned(), *m)).collect::<Vec<_>>(),
                        ast.0.iter().map(|(s, m)| (s.to_string(), *m)).collect::<Vec<_>>(),
                        "trial {trials} q{qi} b{bi}: hits"
                    );
                    assert_eq!(
                        flat.1.iter().map(|(s, m)| ((*s).to_owned(), *m)).collect::<Vec<_>>(),
                        ast.1.iter().map(|(s, m)| (s.to_string(), *m)).collect::<Vec<_>>(),
                        "trial {trials} q{qi} b{bi}: misses"
                    );
                }
            }
            trials += 1;
        }
    }

    /// The TextIndex pre-filter must never reject a part that actually
    /// occurs in the indexed text (no false negatives), across all part
    /// lengths and random texts.
    #[test]
    fn could_contain_no_false_negatives() {
        let _lock = reset_test_blob();
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        let alphabet: Vec<u8> = b"abcdefghijklmnopqrstuvwxyz 0123456789-_".to_vec();
        for _ in 0..200 {
            let n = rng.below(600);
            let mut text = Vec::with_capacity(n);
            for _ in 0..n {
                text.push(alphabet[rng.below(alphabet.len())]);
            }
            let idx = TextIndex::build(&text);
            // parts drawn from the text itself (guaranteed present)
            for _ in 0..8 {
                let len = 1 + rng.below(40);
                if text.len() >= len {
                    let start = rng.below(text.len() - len + 1);
                    let part = &text[start..start + len];
                    assert!(
                        idx.could_contain(part),
                        "false negative: {:?} in {:?}",
                        String::from_utf8_lossy(part),
                        String::from_utf8_lossy(&text)
                    );
                }
            }
        }
    }

    /// The public `eval` entry point (boolean verdict only) agrees with the
    /// full scan on a small hand-built query.
    #[test]
    fn eval_public_api() {
        let _lock = reset_test_blob();
        use crate::parser::Parser;
        use crate::tokenizer::tokenize;

        let root = Parser::new(
            tokenize("TITLE(tax evasion) AND (ABS(poverty) OR AUTHKEY(food*))").unwrap(),
        )
        .parse()
        .unwrap();
        let table = compile_all(std::iter::once(&root));
        let mut root = root;
        let mut nslots = 0u32;
        resolve_blocks(std::slice::from_mut(&mut root), &table, &mut nslots);

        let hit = Paper::from_text("---\ntitle: \"Tax evasion\"\nabstract: |\n  poverty is bad\n---");
        assert!(eval(&root, None, &hit, &table), "expected match");
        assert!(!eval(&root, None, &Paper::from_text("---\ntitle: \"tax evasion\"\n---"), &table), "abs part missing");
        assert!(!eval(&root, None, &Paper::from_text("---\ntitle: \"Other\"\nabstract: |\n  food security\n---"), &table), "title part missing");
    }

    #[test]
    fn pattern_matches_plain_term() {
        let _lock = reset_test_blob();
        let p = compile_pattern("foreign aid");
        assert!(p.matches(b"tax evasion and foreign aid in developing countries"));
        assert!(!p.matches(b"foreign-aid policies"));
    }

    #[test]
    fn pattern_matches_wildcards() {
        let _lock = reset_test_blob();
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

    /// Keyword suggestions: token-overlap ranking, present-keyword skip, and
    /// the real SDG10 dictionary against a health-systems abstract.
    #[test]
    fn suggest_ranks_by_token_overlap() {
        let mk = |kws: &[&str]| {
            SdgDict {
                keywords: kws
                    .iter()
                    .map(|k| {
                        (
                            Arc::from(*k),
                            k.split(' ').filter(|t| !t.is_empty()).map(|t| t.to_lowercase()).collect(),
                        )
                    })
                    .collect(),
                excluded: HashSet::new(),
            }
        };
        let dict = mk(&["health care access", "educational inequality", "digital government", "zebra farming"]);
        let words = text_words("the school health system improves access to care for students in indonesia");
        let present: HashSet<&str> = HashSet::new();
        let sug = suggest_keywords(&words, &dict, &present, 10);
        assert_eq!(sug[0].keyword.as_ref(), "health care access", "best token overlap first");
        assert!(sug[0].score > 0.5);
        assert!(sug.iter().all(|s| s.keyword.as_ref() != "zebra farming"), "no overlap -> skipped");

        // Keywords already present in the paper are never suggested.
        let present2: HashSet<&str> = ["health care access"].into_iter().collect();
        let sug2 = suggest_keywords(&words, &dict, &present2, 10);
        assert!(sug2.iter().all(|s| s.keyword.as_ref() != "health care access"));

        // Real SDG10 dictionary vs the SistaUKS abstract: health/access terms
        // must rank high.
        use crate::query::load_queries;
        use std::path::Path;
        let qdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/data/queries");
        let queries = load_queries(&qdir).unwrap();
        let q10 = queries.iter().find(|q| q.sdg == "10").expect("SDG 10");
        let dict10 = collect_sdg_dict(&q10.blocks);
        let abstract_text = "SistaUKS, a health information system comprising software and data, \
            facilitates the automation of the UKS stratification assessment process because of \
            lack of health-care professionals. This study analyzed teacher satisfaction regarding \
            the utilization of the SistaUKS system by 33 junior high school teachers in Boyolali \
            Regency using the System Usability Scale.";
        let words = text_words(abstract_text);
        let top = suggest_keywords(&words, &dict10, &HashSet::new(), 10);
        let names: Vec<&str> = top.iter().map(|s| s.keyword.as_ref()).collect();
        assert!(
            names.contains(&"health care access") || names.contains(&"access to health care"),
            "health-access term must surface for SDG10, got {names:?}"
        );
    }
    #[test]
    fn text_words_matches_char_splitter() {
        // Reference: the exact tokenization text_words used to implement
        // (and must keep matching): char-wise !is_alphanumeric split.
        fn reference(text: &str) -> Vec<String> {
            // The old text_words deduped tokens through a HashSet before
            // sorting - keep that contract (split can emit the same word
            // twice, e.g. snake_case kebab-case).
            let mut set: std::collections::HashSet<&str> = text
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .collect();
            let mut v: Vec<String> = set.drain().map(|w| w.to_string()).collect();
            v.sort_unstable();
            v
        }
        let cases = [
            "",
            "   ",
            "just a plain ascii sentence, with 2024 numbers & _underscores_.",
            "café naïve résumé …",
            "中文文本 + देवनागरी",
            "emoji 🚀 rocket 🌍 and #hashtag!",
            "a1 b2 c3 -d- _e_ f.g,h;i:j",
            "école Ångström",
            "hygiene (wash) - avoid* /divers/",
            "camelCase PascalCase ALLCAPS snake_case kebab-case",
            "ÅåÄäÖö",
        ];
        for text in cases {
            let pw = text_words(text);
            let mut got: Vec<String> = pw.set.iter().map(|s| s.to_string()).collect();
            got.sort_unstable();
            assert_eq!(got, reference(text), "set mismatch for {text:?}");
            assert_eq!(
                pw.sorted.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                reference(text),
                "sorted mismatch for {text:?}"
            );
        }
        // deterministic long mixed sample (ASCII + accents + CJK)
        let long: String = (0..200)
            .map(|i| format!("word{i} café 中文{i}!! end."))
            .collect();
        let pw = text_words(&long);
        let mut got: Vec<String> = pw.set.iter().map(|s| s.to_string()).collect();
        got.sort_unstable();
        assert_eq!(got, reference(&long));
    }

}
