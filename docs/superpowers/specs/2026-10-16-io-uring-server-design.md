# Optional io_uring Accept Backend Design

**Date:** 2026-10-16  
**Status:** Approved architecture, awaiting spec review before implementation

## Goal

Improve the Rust web server's behavior under concurrent connections on Render by adding an optional Linux `io_uring` accept backend. Select it automatically only when the binary includes the feature and runtime ring initialization succeeds. Preserve the existing threaded TCP server as the fallback on unsupported operating systems, disabled builds, or restricted runtimes.

## Current context

`rust/src/bin/web.rs` currently binds a `std::net::TcpListener`, accepts one connection at a time, and spawns one OS thread per accepted connection. The synchronous `handle_conn` function performs request parsing, routing, and response writing. The server reports the chosen SIMD route in the `X-SIMD-Usage` response header and in startup output. A localhost serial endpoint baseline is saved at `benchmarks/endpoint-baseline-2026-10-16.md`; it is not evidence of concurrent-load capacity.

Render documents a Debian 12 native runtime, but public platform documentation does not establish whether `io_uring` setup is permitted in each service runtime. Runtime setup must therefore be probed, with fallback on failure.

## Approaches considered

1. **Focused io_uring accept loop (recommended):** use the optional backend for accepting connections while retaining `handle_conn` and the existing worker-thread handling for accepted streams. Lowest-risk initial change and preserves route semantics, though it does not eliminate blocking request handling or all per-connection threads.
2. **Full async connection lifecycle:** migrate accept, reads, writes, and request processing to asynchronous I/O. Greater potential to reduce thread overhead, but substantially broadens scope and requires careful isolation of existing blocking filesystem, matching, and external-service work.
3. **Keep threaded backend only and tune worker management:** avoids io_uring compatibility concerns but does not implement the requested backend and changes concurrency behavior less directly.

The initial implementation uses approach 1. Further async conversion is out of scope unless concurrent measurements show the accept-only approach is insufficient and a separate design is approved.

## Design

### Backend capability and selection

- Add an opt-in Cargo feature for the Linux `io_uring` dependency so non-Linux builds and builds without the feature remain supported.
- Put ring initialization and accept-loop details behind a small backend module or similarly focused boundary. Keep HTTP parsing and routing in `web.rs` unchanged.
- At server startup, attempt ring creation and required setup. A successful setup selects `io_uring`; unavailable feature, non-Linux target, or setup error selects the existing threaded listener.
- Treat setup errors as a normal fallback condition, report the reason in startup logs, and do not fail server startup solely because `io_uring` is unavailable.
- Do not infer capability from OS name or kernel version alone. The actual runtime initialization result is authoritative.
- Automatic selection is the default. An explicit override is not included in this initial design.

### Request handling and reporting

- Both accept backends hand accepted TCP connections to the same existing `handle_conn` path so application behavior, response generation, logging, and panic containment remain consistent.
- Follow the existing SIMD reporting convention: add an `X-IO-Backend` response header naming the selected server backend and include the operating-system target in startup diagnostics. Log which backend was selected and, on fallback, the concise setup error.
- Ensure the reported backend is the backend selected at process startup, not a guessed platform value. Keep report values stable and documented.

### Concurrency, lifecycle, and failure handling

- Preserve existing connection-close behavior and per-request handling semantics.
- Keep the accept loop resilient to transient accept errors. Do not allow a connection-handler panic to terminate the listener loop.
- If ring initialization fails, fall back before accepting requests. Runtime ring failure after startup must be handled without silently dropping the listener; implementation should either recover/reinitialize or transition to the threaded backend with a clear log, depending on what the selected io_uring API safely permits.
- Avoid unbounded changes to handler execution policy in this scope. Existing thread-per-connection behavior remains unless implementation review identifies a necessary bounded mechanism to safely satisfy the selected backend's API.

## Validation and success criteria

1. Build and test the default configuration without the io_uring feature, including the existing non-Linux compile path where toolchains are available.
2. Build and test the Linux io_uring-enabled configuration.
3. Unit-test backend selection for feature-disabled, unsupported-target, successful-probe, and failed-probe outcomes using an injectable or otherwise deterministic probe boundary.
4. Integration-test HTTP behavior for both selected backends, including `X-SIMD-Usage`, the new backend header, health, and representative SDG matching routes.
5. Run concurrent benchmarks for the threaded baseline and io_uring backend at multiple concurrency levels. Include lightweight routes and actual SDG checks (`POST /match`, `POST /api/match`, and `POST /api/keywords`), record errors and latency percentiles as well as throughput, and document environment and command lines.
6. Attempt an `io_uring` runtime probe in the actual Render deployment environment before claiming Render uses the backend. If unavailable, verify and report the threaded fallback.
7. Do not claim a performance win unless repeated comparable concurrent runs show a measurable improvement without correctness regressions or elevated failures. If the accept-only backend does not improve results, document that result rather than implying full async I/O was achieved.

## Out of scope

- Rewriting request parsing, routing, or all file/network operations as async.
- Changing SDG matching algorithms or SIMD dispatch.
- Removing support for non-Linux platforms or requiring `io_uring` to start.
- Making claims about production capacity from the existing serial localhost baseline.
- Enabling a specific Render service configuration or deploying changes.

## Risks

- Render may deny ring setup through runtime restrictions, in which case fallback is expected and the backend will not be active there.
- An accept-only `io_uring` loop that still delegates each connection to a new OS thread may provide little benefit. Concurrent benchmarks are a release criterion, not an optional follow-up.
- Maintaining an additional Linux-specific backend adds dependency, compilation, and test-matrix cost.
- Per-response backend reporting can be overwritten by upstream proxies only if they modify the same header; startup logs remain the operator-side source of truth.
