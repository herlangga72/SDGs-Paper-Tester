# SDG Paper Matcher — zero-cost deployment image
# Works on Render (free tier) or any Docker host.
# The web server is Rust (SIMD engine): matching a paper takes ~80 ms
# instead of ~0.9 s with the old Python server.
FROM rust:1.97-slim AS build

WORKDIR /build
COPY rust/Cargo.toml rust/Cargo.lock ./
COPY rust/src ./src
# Fetch deps first for better layer caching, then build.
RUN cargo build --release --bin web --bin sdg_tools

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
