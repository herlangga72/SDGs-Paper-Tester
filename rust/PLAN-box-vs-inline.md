# Plan: reduce per-request memory reads in the Rust matcher — "Box vs normal type"

Status: proposed (no code changed yet)
Scope: `rust/src` (lib `sdg_tools`), plus the dev benches it needs
Owner: Jcode (2026-09-05)

---

## 1. Goal

The hypothesis to test: *storing a value behind `Box<T>` instead of a "normal"
inline field/record reduces memory reads per operation and makes matching
faster.*

This document turns that hypothesis into a falsifiable experiment plan with
measured baselines, an audit of every memory pattern in `rust/src`, ranked
candidate changes, and an A/B protocol with revert criteria.

Important framing up front: the hypothesis is **wrong for the request hot
path** and **possibly right for two cold structures**. The plan is therefore
an A/B plan that will *prove* each claim with wall-clock numbers rather than
assume it, and that also lists the non-Box changes which actually cut memory
traffic per request (so the goal — fewer memory reads — is reached even where
Box is the wrong tool).

---

## 2. Measured baseline (this machine, release build, before any change)

CPU: AMD Ryzen 3 3200G (Zen+), SIMD route AVX2.
Corpus: 17 SDG query files → **2975 blocks, 21,186 patterns, 44,559 memo
slots**.

Benchmark input: synthetic 1.1 MB paper (746 KB searchable body, made from
SDG vocabulary, ~10k "sustainable development" sentences).

| Harness | Command | Result |
|---|---|---|
| server-like hot path | `cargo run --release --example hotbench -- ../.bench_big.md 100` | **avg 74.2 ms/request** (2975 blocks, 5300 matched, 13.5 req/s) |
| prof counters | `cargo run --release --features prof --example prof -- ../.bench_big.md 50` | 61.1 ms/request |

Per-request prof counters (per request, averaged):

| Counter | Value | Meaning |
|---|---|---|
| `index_builds` | 4 | TextIndex builds per request (folded buffers) |
| `index_bytes` | 2,231,401 | total text indexed per request (~3x the 746 KB body) |
| `leaf_evals` | 93,354 | pushes through the flat program |
| `term_computes` | 44,559 | memo misses that actually run a search |
| `term_cache_hits` | 48,795 | memo hits (no search) |
| `could_calls` | 45,272 | TextIndex pre-filter calls |
| `matches_calls` | 3,636 | searches that reached the SIMD scanner |

Read the counters as: per request the matcher walks **~93k leaf records**, but
only **~3.6k searches touch the paper text**, because the bit-set pre-filter
kills ~41k of 44.6k computes. Per-request time is dominated by (a) building 4
TextIndexes over ~2.2 MB of (mostly repeated) folded text, (b) walking 93k
leaf records + memo slots, (c) the 3.6k SIMD scans.

---

## 3. What `Box<T>` actually does to memory reads

Measured sizes (throwaway `size_of` probe, rustc 1.97):

| Type | `size_of` |
|---|---|
| `String` | 24 B (ptr + len + cap) |
| `Box<str>` / `Arc<str>` | 16 B (ptr + len) |
| `Box<T>` | 8 B (ptr only) |
| `Node` (AST enum) | 48 B |
| `Token` | 32 B |
| `Pattern` | 28 B |
| `Op` | 8 B |
| `LeafDesc` | 20 B |
| `TextIndex` | 8,520 B (mostly the 8 KiB bigram matrix inline) |
| `Memo` | 1,264 B |

Rules of thumb that govern every candidate below:

1. **Boxing adds a load, never removes one from a single access.** Reading
   `x.field` is one load; reading `x.boxed.field` is two (pointer, then
   field) and usually two cache lines instead of one.
2. **Boxing pays off when it shrinks a *container* that is scanned or copied
   hot, and the boxed payload is cold.** A `Vec<Big>` reads `len(Big)` bytes
   per element on a linear scan; `Vec<Box<Big>>` reads 8 B/element and only
   dereferences the entries you actually need. Shrinking the outer type from
   48 B to 40 B cuts *all* linear walks over it by 17%.
3. **Boxing is a tax when every element of a dense array is needed.** That is
   the current request hot path: `FlatBlock.prog`/`leaves`, `Pattern`,
   `TextIndex` are deliberately *dense, fixed-size, mmap-visible records*
   (see section 4). Turning any of them into pointer arrays would be a
   literal regression: +1 indirection per record, -cache-locality, and it
   would break the zero-copy mmap view (`cache.rs` views the file as
   `&[Pattern]`, `&[Op]`, `&[LeafDesc]`).
4. **This codebase already went the opposite direction from naive boxing**
   and that is why it is fast: `compile_all` replaced per-keyword `String`s
   with `(offset, len)` pairs into one leaked blob; `flatten_block` replaced
   the boxed AST recursion with dense `Op`/`LeafDesc` arrays; the AST's
   `Field { child: Box<Node> }` / `Not { child: Box<Node> }` are the *only*
   remaining boxes and are recursion plumbing, not a memory optimization.

Conclusion: to reduce memory reads, prefer (i) smaller dense records, (ii)
fewer duplicate text builds/copies, (iii) boxed *only* where the outer type
is traversed/copied hot and the payload is cold. Candidates E1–E3 below are
the honest Box cases; candidates E4–E5 are the non-Box cases that actually
move the per-request number.

---

## 4. Memory-pattern audit of `rust/src`

| Structure | File | Size | Read where | Hot per request? |
|---|---|---|---|---|
| `FlatBlock.prog: &[Op]` | matcher.rs | 8 B/op | `scan_flat_into`, `min_add_flat*` linear loop | **yes** (93k pushes/req) |
| `FlatBlock.leaves: &[LeafDesc]` | matcher.rs | 20 B/leaf | same loops, one per `OP_PUSH` | **yes** |
| `table: &[Pattern]` | matcher.rs | 28 B/pat | `term_hit(&table[pid])` per push | **yes** |
| string blob + parts table | matcher.rs (static) | — | `Pattern::part()`, `raw()` per filtered part | yes (parts) |
| `Memo.terms: Vec<u8>` | matcher.rs | 1 B/slot | memo hit/miss | **yes** |
| `Memo.mask_cache` `[i32;256]` | matcher.rs | 1 KB | `joined_for` per distinct mask | yes (cold-ish) |
| `Memo.joined: Vec<JoinedEntry>` | matcher.rs | Cow buf + `Option<TextIndex>` | folded buffers + index | **yes** (4 entries/req) |
| `TextIndex` | matcher.rs | bit sets + `pos: FastMap` | pre-filter | **yes** (4 builds/req) |
| `ast::Node` | ast.rs | **48 B** | parse/boot, CLI `scan_block_shared`, `eval_ignore_not_block` | **no** for web hot path; yes for CLI |
| `SdgDict.keywords: Vec<(Arc<str>, Vec<String>)>` | matcher.rs | 40 B/entry + tokens | suggestions/scoring (per SDG, end of request) | partially |
| `Query.blocks: Vec<Node>` | query.rs | — | boot + cache serialize | no |
| `Paper` | paper.rs | 184 B + lower buffers | per request | yes (buffers) |

Per `OP_PUSH` in the flat scan the loop currently reads ≈
`8 (op) + 20 (leaf) + 28 (pattern) + 1 (memo slot)` ≈ **57 bytes of record
traffic**, ×93k ≈ **5.3 MB/request**, before any text search. That is the
floor this plan attacks.

---

## 5. Candidate changes (ranked by expected benefit / risk)

### E1. Box the AST `Node` payloads (`keyword`/`op`: `String` → `Box<str>`)

Change:
- `Node::Leaf { keyword: Box<str>, .. }`
- `Node::Group { op: Box<str>, .. }` (must box `op`, not just `keyword`, or
  the enum stays 48 B — the `Group` variant of `String`+`Vec<Node>` sets the
  size)

Why it can help: `Node` 48 → 40 B (−17%). Every AST walk then reads 8 fewer
bytes per node: the CLI match path (`scan_block_shared` recursion over every
block, `main.rs`), `min_add`/`scan_with_fields` fallbacks, boot-time
`flatten_block`, and `cache.rs` node serialization. Parser construction cost
rises slightly (`into_boxed_str` may shrink the allocation to exact len).

Why it can fail: the web per-request hot path never walks `Node` (it walks
`FlatBlock`), so E1 will not move `hotbench`. Only the CLI and boot paths
gain.

Measure: `cargo build --release` boot time (parse + compile + cache write),
`sdg_tools match` on a mid paper, plus full `cargo test` (flat-vs-AST parity
tests must stay green).

### E2. `Vec<Box<Node>>` children (rejected by analysis, cheap to A/B)

Change: `Group.children: Vec<Node>` → `Vec<Box<Node>>`.

Why it is expected to lose: every full traversal (the common case: scans must
report every leaf) then pays the pointer load *and* the node load, on
scattered cache lines. Boxing only wins when traversal short-circuits before
touching most children (`eval`, `min_add_vc` on big OR groups whose first
children hit). Corpus terms mostly *miss*, so short-circuit is rare. Keep as
a one-line probe if evidence contradicts the analysis, not a default.

### E3. `SdgDict` token storage (`Vec<String>` → `Vec<Box<str>>` or flat)

Change: keyword tokens `Vec<String>` (24 B/header) → `Vec<Box<str>>`
(16 B/header), or a single joined token arena + offsets.

Effect: cuts suggestion/advanced-tab scoring traffic and the cache rebuild
footprint. Small: this path runs once per SDG at the tail of a request, not
93k times. Expected win « 1 ms on the 1.1 MB input; measurable only on
`/api/keywords` with `limit=2000`.

### E4. (Recommended, no Box) Shrink the hot leaf record 20 B → 12 B

`LeafDesc` today: `pid u32, slot u32, mask u8, excluded bool, raw_off u32,
raw_len u32` = 20 B (2 pad). `raw_off`/`raw_len` exist so report pushes call
`l.raw()` without indexing `table[pid]`. But every push *already* indexes
`table[l.pid]` for `term_hit`, and `Pattern::raw()` reads the same blob bytes
from its own offsets. So:

- drop `raw_off`/`raw_len` from `LeafDesc` (20 → 12 B),
- report pushes call `table[pid].raw()` (same blob read as today),
- serialize + mmap layout update, bump cache `VERSION` 3 → 4,
- per-request record traffic 57 → 49 B/push (−14%, ≈ −0.75 MB/request).

This is the highest-leverage "reduce memory read" change that keeps dense
records; it is a strict removal of redundant inline bytes, which is the
actual direction the codebase's design already points.

### E5. (Recommended, no Box) Stop re-indexing duplicated folded text

Prof shows 4 TextIndex builds over 2.23 MB per request for a 746 KB body.
`Memo.joined_for` already shares one entry per distinct buffer selection; the
remaining duplication is (a) multi-buffer folds copy the section bytes into a
new `Cow::Owned` buffer, and (b) every entry gets its own `TextIndex`.

Options, cheapest first:
1. Cache the `TextIndex` of the full text and *reuse* it for any mask whose
   selection is exactly the full text (already done for the borrow path —
   verify `full_covers_sections` papers actually hit it), then
2. skip `TextIndex` rebuilds when the buffer pointer+len is identical to an
   existing entry (dedupe by `(ptr,len)` like `push_dedup` does for buffers),
3. only build `positions` for the quads actually needed per mask (already
   gated by `FIRST_QUADS` — confirm no mask builds the full pass twice).

Measure: watch `index_builds`/`index_bytes` counters drop on a
full-covers-sections paper (web form / `from_sections`), which is the shape
of every `/match` request.

### E6. Explicitly DO NOT Box: `Op`, `LeafDesc`, `Pattern`, `TextIndex`

These are fixed-size `repr(C)` records viewed directly from the mmap
(`cache.rs`). Boxing them adds a heap pointer per record, breaks zero-copy
cache views, and turns linear cache-friendly loops into pointer chases. Any
plan step that proposes `Box` here is a guaranteed regression and is out of
scope.

---

## 6. A/B protocol

Keep measurements honest and comparable:

1. Fixed bench input (the synthetic paper above, or an equivalent committed
   fixture under `papers/` for reproducibility).
2. `cargo run --release --example hotbench -- <paper> 100` for the request
   hot path; report the median of 3 runs.
3. `cargo run --release --features prof --example prof -- <paper> 50` for
   counters (`index_builds`, `index_bytes`, `term_*`, `matches_calls`) so a
   wall-clock win/loss can be attributed to the right subsystem.
4. Full correctness gate before every measurement: `cargo test --release`
   (flat-vs-AST parity, indexed-vs-scan parity, random-paper trials). The
   parity tests are the safety net for every layout change.
5. Each candidate lands on its own commit with its before/after numbers in
   the commit message, so a change that regresses can be reverted in one
   step.

Acceptance:
- Keep: ≥ 3% median improvement on `hotbench`/`prof` with parity green.
- Revert: ≤ 0% or parity red. For E1 (boot/CLI only), the same bar applies
  to `sdg_tools match` wall time, not `hotbench`.

---

## 7. Execution order

1. Commit the bench fixture + baseline numbers (section 2) so every later
   commit is measured against the same input. (Optional but cheap.)
2. E4 (LeafDesc 20 → 12 B, cache v4): highest expected request-path win,
   touches one struct + serializer + one report-push call site.
3. E5 (index dedupe/rebuild audit): verify with prof counters; implement
   only the confirmed duplication.
4. E1 (AST `Box<str>`): only if CLI/boot time still matters after E4/E5.
5. E3 (dict token layout): only if suggestion/advanced timing shows up in a
   profile.
6. E2: run as a probe only if E1 changes the AST so much that the analysis
   needs re-checking; do not ship without a measured win.

Non-goals: touching `Op`/`Pattern`/`TextIndex` layout for boxing, changing
`?`-glob or wildcard semantics, Python-engine parity, or anything that breaks
the mmap cache format without a version bump.

---

## 8. Risks

- **Cache format coupling**: `Pattern`/`Op`/`LeafDesc` are laid out by hand in
  `cache.rs` with hard-coded record sizes (28/8/12 after E4; was 20 for
  `LeafDesc`). Any layout change must bump `VERSION` (currently 4) and
  delete/reject stale `sdg_cache.bin` files, which the mtime validation
  already handles for source changes.
- **E1 allocations**: `String::into_boxed_str()` can reallocate when
  capacity > len; parser boot time may rise a little. Measure `parse`/cache
  write time before accepting.
- **Premise risk**: if the point of the exercise is strictly "make
  `/match` faster on large papers", E4/E5 are the levers, not Box. E1–E3 will
  likely show ≤ 1% on `hotbench`, and this plan is designed to prove that
  with data instead of asserting it.

---

## 9. Outcome (2026-09-05)

Shipped on `main` as commit `a1003c3` + follow-up:

- **E4 shipped** (`LeafDesc` 20 → 12 B, cache `VERSION` 4). Report strings are
  now read from `table[pid].raw()` (the `Pattern` is already in cache from the
  `term_hit` load) instead of a second `(raw_off, raw_len)` copy in every leaf
  record. All 47 tests pass; CLI cache write + mmap read round trip is
  byte-identical.
- **E5-borrow-full implemented, then reverted.** Prof counters proved it never
  fires on this corpus: every leaf is field-wrapped (`TITLE`, `TITLE-ABS`,
  `TITLE-ABS-KEY`, `AUTHKEY`, one `SUBJAREA`), so no leaf evaluates under the
  default `0x0F` mask and the fold-to-full-text dedupe is dead code.
- **A/B measurement reality**: this Zen+ host varies ±6% between identical
  runs (thermal/frequency), so sub-5% changes are not resolvable with
  `hotbench` medians. Interleaved E4-vs-baseline rounds landed within that
  noise (round deltas −1.9, −1.3, +3.3 ms). E4 is kept for its deterministic
  effect (smaller cache file, fewer bytes read per leaf) not for a claimed
  wall-clock win.
- **The remaining, real duplication** (from prof, deterministic): a 855 KB
  sections paper indexes 2.51 MB per request. Masks `TITLE-ABS` (0x03),
  `TITLE-ABS-KEY` (0x07) and the full-text/ANY selection each copy ~850 KB of
  the same title+abstract prefix into their own joined buffer and build their
  own `TextIndex`. The follow-up worth doing (needs a quieter measurement host
  or `perf`): share one `TextIndex` per unique underlying buffer and evaluate
  a mask as a union of its buffers, which would cut `index_bytes` ~65%. This
  is a Memo/`JoinedEntry` refactor, not a Box change.
