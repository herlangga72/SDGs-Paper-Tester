# Optional io_uring Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional Linux `io_uring` connection acceptance, automatically selecting it after a successful runtime probe and retaining the current threaded backend as fallback, then validate concurrency and report the active backend like SIMD dispatch.

**Architecture:** Preserve `handle_conn` as the single synchronous HTTP implementation. Add a Linux-gated optional accept backend that submits accepts through `io_uring` and hands accepted sockets to the existing panic-contained connection worker path. Make the Docker/Render build include the optional feature, detect ring availability at startup, and fall back cleanly when unavailable.

**Tech Stack:** Rust 2021, Cargo optional target dependency, Linux `io-uring` crate, existing `std::net` HTTP handler, Docker BuildKit, ApacheBench (`ab`) and curl.

---

## File map

- `rust/Cargo.toml`, `rust/Cargo.lock`: optional Linux-only dependency and feature.
- `rust/src/bin/web_support/io_backend.rs` (new): backend identity, runtime probe, and Linux ring accept-loop boundary with non-Linux/default fallback stubs if needed.
- `rust/src/bin/web.rs`: startup backend selection, shared accepted-stream dispatch, selected-backend response header and startup diagnostics.
- `Dockerfile`: build the Render production binary with the io_uring feature enabled; the runtime probe still determines whether it is actually selected.
- `rust/src/bin/web.rs` tests or `rust/src/bin/web_support/io_backend.rs` tests: deterministic selection and response reporting.
- `benchmarks/fixtures/match-form.txt` and `benchmarks/fixtures/keywords-form.txt` (new): stable URL-encoded request bodies reused for threaded and io_uring POST benchmarks. Store only synthetic benchmark data.
- `benchmarks/concurrent-endpoints-2026-10-16.md` (new before implementation): preserve the threaded-only concurrent baseline, then append feature-enabled measurements and an explicit comparison.

## Task 1: Add optional Linux dependency and preserve portable default builds

**Files:** `rust/Cargo.toml`, `rust/Cargo.lock`

- [ ] **Step 1: Capture the existing threaded concurrent baseline and current build status**

Before code changes, build the current release server and benchmark the routes and concurrency levels planned in Task 5. Create `benchmarks/concurrent-endpoints-2026-10-16.md` with a `Threaded baseline` section and store the exact URL-encoded POST request body used for each match route in `benchmarks/fixtures/`. Include the complete shell commands and record server startup backend and SIMD headers. Also run:

```sh
cargo test --bin web
cargo build --release --bin web
```

Expected before changes: both commands succeed. Preserve the concurrent results because the existing Markdown baseline only covers serial traffic.

- [ ] **Step 2: Add optional target-specific dependency and feature**

Use a Linux-only optional dependency and an explicitly named feature. Select a currently released `io-uring` crate version during implementation, record that version in `rust/Cargo.toml`, and let Cargo update `Cargo.lock`. Do not add the crate to the default feature set.

```toml
[target.'cfg(target_os = "linux")'.dependencies]
io-uring = { version = "0.7", optional = true }

[features]
io-uring = ["dep:io-uring"]
```

- [ ] **Step 3: Verify default and enabled builds**

Run:

```sh
cargo test --bin web
cargo build --release --bin web
cargo check --features io-uring --bin web
```

Expected: default build remains independent of io_uring code; Linux feature-enabled check resolves and compiles the dependency.

- [ ] **Step 4: Commit the dependency boundary**

```sh
git add rust/Cargo.toml rust/Cargo.lock
git commit -m "feat(web): add optional Linux io_uring dependency"
```

## Task 2: Implement deterministic runtime selection and fallback

**Files:** create `rust/src/bin/web_support/io_backend.rs`; modify `rust/src/bin/web.rs`; test in the new module.

- [ ] **Step 1: Write tests for backend selection before implementation**

Define a small pure selection function that accepts compile-time availability and a probe result. Test these behaviors:

```rust
#[test]
fn disabled_feature_selects_threaded_backend() {
    assert_eq!(select_backend(false, Ok(())), BackendKind::Threaded);
}

#[test]
fn successful_runtime_probe_selects_io_uring() {
    assert_eq!(select_backend(true, Ok(())), BackendKind::IoUring);
}

#[test]
fn failed_runtime_probe_selects_threaded_backend() {
    assert_eq!(select_backend(true, Err("operation not permitted")), BackendKind::Threaded);
}
```

Keep the probe injectable so tests do not depend on host kernel or seccomp configuration.

- [ ] **Step 2: Run selection tests and confirm expected failure**

Run:

```sh
cargo test --bin web io_backend
```

Expected: FAIL because the selection module/types are not yet implemented.

- [ ] **Step 3: Implement backend kind and probe boundary**

Add `BackendKind::{IoUring, Threaded}` and a startup result carrying the kind and optional fallback reason. Under `cfg(all(target_os = "linux", feature = "io-uring"))`, initialize the ring with the minimum resources needed to submit an accept operation. Do not select based only on OS/kernel version. On feature-disabled or non-Linux builds return `Threaded` without importing Linux-only types.

- [ ] **Step 4: Add focused accept loop and shared connection dispatch**

Refactor the current `listener.incoming()` body into a shared function that accepts a `TcpStream`, spawns the existing handler closure, and retains `catch_unwind`. The io_uring loop submits accept on the listener fd, handles completion results, converts each accepted fd into `TcpStream` exactly once using the platform ownership API, then dispatches through that shared function. Keep request parsing and `handle_conn` unchanged.

On ring errors after startup, stop submitting/using ring operations, log the error, and continue with the still-owned listener through the existing threaded accept loop. Ensure no accepted fd is leaked or double-closed. If the crate's API cannot safely provide this transition with current ownership, keep startup fallback mandatory and make unrecoverable post-start ring errors terminate with an explicit error rather than silently hanging.

- [ ] **Step 5: Run tests and Linux feature check**

Run:

```sh
cargo test --bin web
cargo check --features io-uring --bin web
```

Expected: deterministic selection tests pass; both compiled paths type-check.

- [ ] **Step 6: Commit backend selection and accept loop**

```sh
git add rust/src/bin/web.rs rust/src/bin/web_support/io_backend.rs
git commit -m "feat(web): add runtime-probed io_uring accept backend"
```

## Task 3: Report active backend and enable the Render build

**Files:** `rust/src/bin/web.rs`, `Dockerfile`; tests in `rust/src/bin/web.rs`.

- [ ] **Step 1: Add failing tests for response and startup report values**

Test the small header-formatting helper for stable backend strings. Expected mappings:

```rust
assert_eq!(backend_header_value(BackendKind::IoUring), "io_uring");
assert_eq!(backend_header_value(BackendKind::Threaded), "threaded");
```

Also assert the existing `X-SIMD-Usage` header remains present and unchanged in the generated response header block.

- [ ] **Step 2: Add selected backend response header and startup diagnostics**

Report `X-IO-Backend: io_uring` or `X-IO-Backend: threaded` using the single startup-selected backend value. Startup output must state the compile target OS, selected backend, and fallback reason when probing fails. Do not add a client-provided header as a capability signal.

- [ ] **Step 3: Enable feature in Docker/Render build**

Change the Docker build command from:

```dockerfile
cargo build --release --bin web
```

to:

```dockerfile
cargo build --release --features io-uring --bin web
```

The feature is optional for ordinary/local and non-Linux builds; the production Docker build includes it. Runtime startup probing still selects threaded fallback when the container disallows ring setup.

- [ ] **Step 4: Verify production and portable build paths**

Run from `rust/`:

```sh
cargo test --bin web
cargo build --release --features io-uring --bin web
cargo build --release --bin web
```

Run a Docker build when Docker/BuildKit is available. If a non-Linux Rust target is installed, run `cargo check --target x86_64-apple-darwin --bin web`; skip this command if that target is not installed.

- [ ] **Step 5: Commit reporting and deployment configuration**

```sh
git add rust/src/bin/web.rs Dockerfile
git commit -m "feat(web): report selected IO backend and enable Render probe"
```

## Task 4: Validate both operational paths and endpoint compatibility

**Files:** `rust/src/bin/web.rs` tests; no route behavior changes expected.

- [ ] **Step 1: Add a test seam for forced probe outcomes**

Use the injected probe boundary to start the test server in threaded mode even on a host where io_uring works, and to simulate a denied setup. Keep this test-only seam out of public runtime controls.

- [ ] **Step 2: Test server header and response compatibility in threaded mode**

Start the local test server and verify `/health` returns 200, `X-IO-Backend: threaded`, and the existing `X-SIMD-Usage` header.

- [ ] **Step 3: Exercise io_uring path when host permits it**

Run the feature-enabled server. Verify startup logs say `io_uring` only after ring initialization succeeds, then check `/health` and an SDG POST route. If ring setup is denied, verify fallback log and `X-IO-Backend: threaded` instead.

- [ ] **Step 4: Run full regression tests**

Run from `rust/`:

```sh
cargo test
cargo test --features io-uring
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features io-uring -- -D warnings
```

Expected: tests pass in both feature configurations and no new warnings remain.

- [ ] **Step 5: Commit integration coverage**

```sh
git add rust/src/bin/web.rs rust/src/bin/web_support/io_backend.rs
git commit -m "test(web): cover io_uring selection and fallback"
```

## Task 5: Benchmark concurrency and document the result

**Files:** create `benchmarks/concurrent-endpoints-2026-10-16.md`.

- [ ] **Step 1: Build two comparable binaries**

Build the threaded-only binary and the feature-enabled binary from the same source and release settings. Confirm their startup logs and response header report the expected active backend before collecting benchmark data.

- [ ] **Step 2: Run repeated concurrent measurements**

For each backend, run three repeats at concurrency levels `1`, `8`, and `32`, with 300 requests per GET run and 90 requests per POST run. Benchmark `GET /health`, `POST /match`, `POST /api/match`, and `POST /api/keywords`, using the identical URL-encoded body fixtures saved during Task 1. For example, run `ab -n 90 -c 8 -p benchmarks/fixtures/match-form.txt -T application/x-www-form-urlencoded -l http://127.0.0.1:8000/match`; substitute the exact fixture and route for each POST. Use `ab -n 300 -c 8 http://127.0.0.1:8000/health` for GET. Repeat commands with each specified concurrency and `-n` value. Record full commands, successful/failed responses, throughput, mean latency, and available percentile output. If a concurrency level overwhelms the host, stop and record the resource constraint rather than hiding failures. Do not run both servers on the same port simultaneously.

- [ ] **Step 3: Compare and record outcomes**

Write environment details (OS/kernel, CPU allocation if known, tool/version, build flags, date), all measured values, and a concise comparison in the Markdown report. Explicitly state if the local runtime denies io_uring or if Render runtime access could not be tested. Do not describe a speedup unless repeated runs show a consistent measurable improvement without higher errors or regressions.

- [ ] **Step 4: Re-run representative correctness checks after load**

Verify `/health` and one representative SDG match result on both binaries; check server logs for panics, accept errors, leaked-resource symptoms, and unexpected fallback.

- [ ] **Step 5: Commit benchmark evidence**

```sh
git add benchmarks/concurrent-endpoints-2026-10-16.md
git commit -m "bench(web): compare threaded and io_uring concurrency"
```

## Final acceptance checklist

- Default build remains portable and uses threaded fallback.
- Docker/Render build compiles the optional Linux backend.
- Runtime probe, not OS guesswork, determines backend selection.
- Ring setup denial is non-fatal and visibly falls back.
- Both paths preserve the same HTTP handling and `X-SIMD-Usage` behavior.
- `X-IO-Backend` and startup logs report the actual selected backend.
- Tests cover successful probe, failed probe, and disabled feature deterministically.
- Concurrent measurements include the SDG-check endpoints and report actual observed gains or lack thereof.
