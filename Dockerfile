# syntax=docker/dockerfile:1
# SDG Paper Matcher — zero-cost deployment image
# Works on Render (free tier) or any Docker host.
# The web server is Rust (SIMD engine): matching a paper takes ~10 ms
# instead of ~0.9 s with the old Python server.
#
# Build speed: BuildKit cache mounts persist the cargo registry and the
# target dir across builds, so unchanged deps/artifacts are reused instead
# of rebuilt. The release profile is overridden to thin LTO + 16 codegen
# units + symbol stripping: builds several times faster with ~1% runtime
# cost, and the stripped binary shrinks the image (faster cold start).
#
# Pin the build stage to bookworm: rust:1.97-slim is trixie (glibc 2.41),
# whose binaries won't run on the bookworm runtime (glibc 2.36).
FROM rust:1.97-slim-bookworm AS build

WORKDIR /build

# Fetch dependencies first (layer-cached, and cached by the registry mount).
COPY rust/Cargo.toml rust/Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo fetch

COPY rust/src ./src

ENV CARGO_PROFILE_RELEASE_LTO=thin \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    CARGO_PROFILE_RELEASE_STRIP=symbols

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin web

FROM debian:bookworm-slim

WORKDIR /app

COPY engine/ engine/
COPY web/ web/
COPY papers/ papers/
COPY LICENSE .

# The Rust web server (queries are parsed from engine/data/queries at boot,
# which takes <10 ms with the SIMD tokenizer — no build-time DB step needed).
COPY --from=build /build/target/release/web /usr/local/bin/sdg-web

# Copy the entrypoint that honors the platform-assigned $PORT
# (Render injects $PORT; Hugging Face Spaces expects 7860 by default).
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
    CMD /usr/local/bin/sdg-web --self-check http://127.0.0.1:7860/health

CMD ["/app/entrypoint.sh"]
