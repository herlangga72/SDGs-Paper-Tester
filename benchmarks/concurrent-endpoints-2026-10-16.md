# Concurrent endpoint benchmark: threaded baseline and io_uring comparison

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

## io_uring-enabled comparison

**Run date:** 2026-10-16
**Build:** `cargo build --release --features io-uring --bin web`
**Active backend:** Confirmed `X-IO-Backend: io_uring`; startup log reported `io_uring accept backend active`.
**Workload:** Identical routes, fixtures, request counts, concurrency values, and three repeats per case as the threaded baseline. All 36 runs reported 0 failed requests.
**Raw outputs:** `/home/server/.jcode/scratch/io-uring-enabled/` (local scratch, not committed).

Ranges across the three repeats. Latency is ApacheBench's mean time per request. Baseline values are repeated here for direct comparison.

| Route | Concurrency | Threaded throughput (req/s) | io_uring throughput (req/s) | Threaded mean latency | io_uring mean latency |
|---|---:|---:|---:|---:|---:|
| `GET /health` | 1 | 5,852.74–9,122.15 | 6,678.39–10,010.68 | 0.110–0.171 ms | 0.100–0.150 ms |
| `GET /health` | 8 | 11,042.40–21,570.32 | 20,701.08–22,258.50 | 0.371–0.724 ms | 0.359–0.386 ms |
| `GET /health` | 32 | 13,778.53–17,045.45 | 12,433.17–19,612.97 | 1.877–2.322 ms | 1.632–2.574 ms |
| `POST /match` | 1 | 45.19–49.95 | 44.30–49.44 | 20.020–22.131 ms | 20.225–22.571 ms |
| `POST /match` | 8 | 121.48–149.86 | 171.35–175.15 | 53.384–65.855 ms | 45.675–46.687 ms |
| `POST /match` | 32 | 143.38–149.33 | 175.03–180.39 | 214.291–223.189 ms | 177.392–182.821 ms |
| `POST /api/match` | 1 | 19.23–20.59 | 20.13–21.71 | 48.572–51.996 ms | 46.066–49.669 ms |
| `POST /api/match` | 8 | 56.46–62.28 | 63.03–63.81 | 128.444–141.687 ms | 125.363–126.932 ms |
| `POST /api/match` | 32 | 47.77–53.44 | 62.48–65.10 | 598.772–669.941 ms | 491.558–512.171 ms |
| `POST /api/keywords` | 1 | 689.02–1,577.59 | 1,194.60–1,644.95 | 0.634–1.451 ms | 0.608–0.837 ms |
| `POST /api/keywords` | 8 | 3,660.32–4,274.32 | 3,639.01–4,455.89 | 1.872–2.186 ms | 1.795–2.198 ms |
| `POST /api/keywords` | 32 | 2,719.03–3,240.09 | 2,363.26–4,284.49 | 9.876–11.769 ms | 7.469–13.541 ms |

### Interpretation and limitations

- The io_uring run selected the real io_uring accept path on this Linux host, rather than silently exercising fallback.
- This implementation changes socket acceptance only. Each accepted connection still runs the same synchronous handler on its own OS thread, so matching compute remains unchanged.
- Results suggest better throughput under concurrency for `/match` and `/api/match` on this local run, with little or mixed change on lightweight routes. The benchmark is a single local host run with three repetitions and is not evidence of Render production capacity or a guaranteed gain.
- The two builds were not interleaved; machine load and timing variation may account for some differences. Repeat on the deployed Render instance before making a production-performance claim.

## io_uring rerun against the saved baseline

**Run date:** 2026-10-16 17:26 UTC
**Source revision:** `fccf589`
**Build:** `cargo build --release --features io-uring --bin web`
**Environment:** Same local Linux x86_64 host, Manjaro kernel `6.18.12-1-MANJARO`, 4 CPUs, ApacheBench 2.3.
**Active backend:** Confirmed on `/health`: `X-IO-Backend: io_uring`, `X-IO-Platform: linux`, `X-SIMD-Usage: AVX2`; startup log says `io_uring accept backend active`.
**Workload:** Exactly as the baseline: three repeats at concurrency 1/8/32, 300 requests per GET run and 90 per POST run, same fixtures. All 36 runs had 0 failed requests and 0 non-2xx responses.
**Raw outputs:** `/home/server/.jcode/scratch/io-uring-rerun-2026-10-16/` (local scratch, not committed).

Ranges across the three rerun repeats, compared with the original threaded ranges.

| Route | Concurrency | Threaded throughput | Rerun io_uring throughput | Threaded mean latency | Rerun io_uring mean latency |
|---|---:|---:|---:|---:|---:|
| `GET /health` | 1 | 5,852.74–9,122.15 req/s | 7,888.51–9,651.58 req/s | 0.110–0.171 ms | 0.104–0.127 ms |
| `GET /health` | 8 | 11,042.40–21,570.32 req/s | 15,046.64–21,284.14 req/s | 0.371–0.724 ms | 0.376–0.532 ms |
| `GET /health` | 32 | 13,778.53–17,045.45 req/s | 15,890.67–17,472.34 req/s | 1.877–2.322 ms | 1.831–2.014 ms |
| `POST /match` | 1 | 45.19–49.95 req/s | 52.16–52.17 req/s | 20.020–22.131 ms | 19.167–19.173 ms |
| `POST /match` | 8 | 121.48–149.86 req/s | 136.95–149.48 req/s | 53.384–65.855 ms | 53.519–58.418 ms |
| `POST /match` | 32 | 143.38–149.33 req/s | 145.58–153.61 req/s | 214.291–223.189 ms | 208.322–219.804 ms |
| `POST /api/match` | 1 | 19.23–20.59 req/s | 21.54–21.60 req/s | 48.572–51.996 ms | 46.295–46.428 ms |
| `POST /api/match` | 8 | 56.46–62.28 req/s | 55.92–56.32 req/s | 128.444–141.687 ms | 142.034–143.068 ms |
| `POST /api/match` | 32 | 47.77–53.44 req/s | 50.84–56.47 req/s | 598.772–669.941 ms | 566.720–629.487 ms |
| `POST /api/keywords` | 1 | 689.02–1,577.59 req/s | 1,423.01–1,601.59 req/s | 0.634–1.451 ms | 0.624–0.703 ms |
| `POST /api/keywords` | 8 | 3,660.32–4,274.32 req/s | 2,920.37–3,892.57 req/s | 1.872–2.186 ms | 2.055–2.739 ms |
| `POST /api/keywords` | 32 | 2,719.03–3,240.09 req/s | 3,277.85–3,829.79 req/s | 9.876–11.769 ms | 8.356–9.762 ms |

### Rerun interpretation

- This is not a consistent across-the-board speedup. `/match` at concurrency 8 and 32 substantially overlaps the threaded range. `/api/match` at concurrency 8 has slightly lower throughput and higher mean latency, while concurrency 1 and 32 are modestly better. `/api/keywords` at concurrency 8 has a wider, slower range, and concurrency 32 is faster in this rerun.
- Lightweight health results overlap substantially. Small differences can come from host noise and ApacheBench timing. The earlier feature run had more favorable match-route numbers, so repeat-to-repeat variation is material.
- Both feature runs use one local host with three repetitions; no Render deployment comparison was performed. This accept-only io_uring backend still uses one OS thread and synchronous processing per connection, so CPU-bound matching is unchanged.
- Current evidence shows no benchmark failures and modest/mixed performance differences, not a robust general performance win. A production decision should be based on repeated measurements on Render with representative traffic.
