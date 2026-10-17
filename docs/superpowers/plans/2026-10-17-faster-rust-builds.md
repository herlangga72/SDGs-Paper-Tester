# Faster Rust Compilation Implementation Plan

**Goal:** Reduce normal release compilation time without changing application code or requiring extra tools.

**Design:** Match the existing Docker settings: thin LTO and 16 codegen units. Preserve opt-level 3 and symbol stripping. Add `release-max`, inheriting release and restoring full LTO and one codegen unit. Keep debug profiles and dependencies unchanged. Runtime performance may differ and is not guaranteed identical.

## Steps
- [x] Measure the existing release build, then force a package rebuild with `cargo rustc --release --bin web -- --cfg build_speed_probe` and measure it. Dependencies stay cached.
- [x] Change `rust/Cargo.toml` release LTO to `"thin"` and codegen units to `16`. Add `[profile.release-max]` with `inherits = "release"`, `lto = true`, `codegen-units = 1`.
- [x] Document normal builds, `cargo check`, single-binary builds, and `release-max` output paths in README.md. Remove redundant Docker profile overrides and unsupported performance estimates.
- [x] Validate manifest profile assertions (observed failing before changes), compile the new release profile, then time the same forced package rebuild. Report warm dependency rebuilds separately from initial builds.
- [x] Run `cargo test --all-features` and `cargo test --release --all-features --lib --bins`, verify `release-max` builds, review the diff, and commit.

## Acceptance
Both binaries build, existing tests pass, original optimization settings remain available, and a measured comparison is reported without claiming unmeasured runtime improvements.

## Results

Measured on Ryzen 3 3200G, Rust 1.97.1, with Bash's elapsed-time timer.
One forced web-binary rebuild per configuration, with its dependencies already
built and no concurrent validation builds:

| Settings | Web rebuild |
| --- | ---: |
| Original full LTO, 1 codegen unit | 30.215 s |
| Thin LTO, 16 codegen units | 18.417 s |

This run took 39.0% less time. This is not a clean-build benchmark or a
statistical estimate. The initial new-profile build took 39.936 s including
dependency recompilation, so switching settings has a one-time cache cost.

- Manifest assertions failed before the change and passed afterward.
- Debug all-feature tests: 59 passed, 2 ignored, zero failures.
- Release all-feature library/binary tests: 59 passed, 2 ignored, zero failures.
- Both binaries built with release/all-features and release-max/default features.
- Existing unused-assignment and unused-mut warnings remain unchanged.
- Docker profile settings are equivalent by inspection. No container build was run.
- Runtime throughput was not benchmarked. The original profile remains available.
