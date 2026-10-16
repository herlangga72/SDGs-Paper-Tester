# Concurrent endpoint benchmark: threaded baseline

**Run date:** 2026-10-16  
**Source revision:** `846c92a` (before server-code changes)  
**Environment:** Local Linux x86_64, Manjaro kernel `6.18.12-1-MANJARO`, 4 reported online CPUs  
**Tool:** ApacheBench 2.3 (`ab`)  
**Server:** existing `std::net` accept loop, one thread per connection, `127.0.0.1:18000`  
**Purpose:** Capture a concurrency baseline before the io_uring implementation. These are local measurements, not Render capacity estimates.

## Test setup

- Built with `cd rust && cargo build --release --bin web`.
- Ran the server with `./rust/target/release/web --host 127.0.0.1 --port 18000 --no-browser`.
- Verified `GET /health` returned HTTP 200 and `X-SIMD-Usage: AVX2`. The original server has no IO-backend header.
- Each route/concurrency combination was run three times. GET runs used 300 requests; POST runs used 90 requests. Concurrency was 1, 8, and 32.
- POST bodies are fixed synthetic URL-encoded data in `benchmarks/fixtures/match-form.txt` and `benchmarks/fixtures/keywords-form.txt`. `/match` and `/api/match` used the same match fixture; `/api/keywords` used the keywords fixture (`sdg=01&limit=50`).
- POST commands included `-l` for varying response lengths. All 36 runs reported 0 failed requests and 0 non-2xx responses.
- ApacheBench reports percentile times as integer milliseconds. Sub-millisecond values appear as 0 and are not precise latency measurements.

## Results

Values are ranges across three repeats. Mean latency is ApacheBench's first `Time per request` value. Throughput is requests per second. Percentile values are the ranges of reported integer-millisecond percentiles across repeats.

| Route | Concurrency | Requests/run | Mean latency range | Throughput range | p50 range | p95 range | Failed / non-2xx |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GET /health` | 1 | 300 | 0.110–0.171 ms | 5,852.74–9,122.15 req/s | 0 ms | 0 ms | 0 / 0 |
| `GET /health` | 8 | 300 | 0.371–0.724 ms | 11,042.40–21,570.32 req/s | 0–1 ms | 1–2 ms | 0 / 0 |
| `GET /health` | 32 | 300 | 1.877–2.322 ms | 13,778.53–17,045.45 req/s | 2 ms | 3 ms | 0 / 0 |
| `POST /match` | 1 | 90 | 20.020–22.131 ms | 45.19–49.95 req/s | 20–21 ms | 23–28 ms | 0 / 0 |
| `POST /match` | 8 | 90 | 53.384–65.855 ms | 121.48–149.86 req/s | 50–61 ms | 66–107 ms | 0 / 0 |
| `POST /match` | 32 | 90 | 214.291–223.189 ms | 143.38–149.33 req/s | 190–224 ms | 294–334 ms | 0 / 0 |
| `POST /api/match` | 1 | 90 | 48.572–51.996 ms | 19.23–20.59 req/s | 47–50 ms | 59–68 ms | 0 / 0 |
| `POST /api/match` | 8 | 90 | 128.444–141.687 ms | 56.46–62.28 req/s | 126–136 ms | 176–209 ms | 0 / 0 |
| `POST /api/match` | 32 | 90 | 598.772–669.941 ms | 47.77–53.44 req/s | 472–605 ms | 870–1,421 ms | 0 / 0 |
| `POST /api/keywords` | 1 | 90 | 0.634–1.451 ms | 689.02–1,577.59 req/s | 1 ms | 1–3 ms | 0 / 0 |
| `POST /api/keywords` | 8 | 90 | 1.872–2.186 ms | 3,660.32–4,274.32 req/s | 2 ms | 3 ms | 0 / 0 |
| `POST /api/keywords` | 32 | 90 | 9.876–11.769 ms | 2,719.03–3,240.09 req/s | 6–9 ms | 21–27 ms | 0 / 0 |

## Exact command forms

Run each command three times for each `C` in `1`, `8`, and `32`:

```sh
ab -n 300 -c "$C" -s 30 http://127.0.0.1:18000/health
ab -n 90 -c "$C" -s 30 -l -p benchmarks/fixtures/match-form.txt -T application/x-www-form-urlencoded http://127.0.0.1:18000/match
ab -n 90 -c "$C" -s 30 -l -p benchmarks/fixtures/match-form.txt -T application/x-www-form-urlencoded http://127.0.0.1:18000/api/match
ab -n 90 -c "$C" -s 30 -l -p benchmarks/fixtures/keywords-form.txt -T application/x-www-form-urlencoded http://127.0.0.1:18000/api/keywords
```

## Baseline verification

- `cargo test --bin web`: passed, 12 passed, 0 failed, 1 ignored. The ignored test is an existing benchmark test.
- Existing compiler warnings in `rust/src/matcher.rs` at lines 2227 and 2235 report unused assignments to `best`; they predate the io_uring work.
- All endpoint benchmark responses returned successfully. The startup/server log confirmed normal startup and no listener failure.

## io_uring comparison status

Not measured in this baseline run because the io_uring implementation has not yet been added. Results for the feature-enabled binary will be appended after implementation and runtime probing.
