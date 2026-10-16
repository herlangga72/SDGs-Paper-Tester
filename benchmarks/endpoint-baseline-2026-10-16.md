# Rust web endpoint benchmark baseline

**Run date:** 2026-10-16  
**Environment:** Local Linux x86_64, release build rebuilt from current source  
**Tool:** ApacheBench (`ab`)  
**Load:** Serial requests (`-c 1`), 100 requests per GET route and 25 per POST route.  
**Scope:** Localhost only. These figures are a baseline for comparison, not production capacity estimates.

## Measured routes

| Endpoint | Method | Scenario | Requests | Mean latency | Throughput | HTTP | Failed |
|---|---|---|---:|---:|---:|---:|---:|
| `/` | GET | Landing page | 100 | 0.245 ms | 4,074.98 req/s | 200 | 0 |
| `/index.html` | GET | Index alias | 100 | 0.493 ms | 2,028.15 req/s | 200 | 0 |
| `/health` | GET | Liveness | 100 | 0.115 ms | 8,662.51 req/s | 200 | 0 |
| `/api/stats` | GET | Counters and user stats | 100 | 0.296 ms | 3,382.49 req/s | 200 | 0 |
| `/api/logs?format=jsonl&limit=50` | GET | Access-log export | 100 | 0.205 ms | 4,868.31 req/s | 200 | 0 |
| `/api/matches?format=jsonl&limit=50` | GET | Match-log export | 100 | 0.186 ms | 5,367.40 req/s | 200 | 0 |
| `/samples` | GET | Sample-paper list | 100 | 0.396 ms | 2,524.55 req/s | 200 | 0 |
| `/sample?name=sample_paper.md&format=json` | GET | Sample as JSON | 100 | 0.196 ms | 5,089.32 req/s | 200 | 0 |
| `/sample?name=sample_paper.md` | GET | Sample as Markdown | 100 | 0.174 ms | 5,742.51 req/s | 200 | 0 |
| `/static/index.html` | GET | Static HTML | 100 | 0.164 ms | 6,080.14 req/s | 200 | 0 |
| `/static/app.js` | GET | Static JavaScript | 100 | 0.185 ms | 5,394.03 req/s | 200 | 0 |
| `/static/style.css` | GET | Static CSS | 100 | 0.157 ms | 6,365.78 req/s | 200 | 0 |
| `/verify` | GET | Verification page without a code | 100 | 0.142 ms | 7,035.81 req/s | 200 | 0 |
| `/match` | POST | Representative tax-evasion paper, HTML report | 25 | 23.341 ms | 42.84 req/s | 200 | 0 |
| `/api/match` | POST | Same paper, JSON report | 25 | 53.782 ms | 18.59 req/s | 200 | 0 |
| `/api/keywords` | POST | SDG 01 suggestions, limit 50 | 25 | 1.062 ms | 941.73 req/s | 200 | 0 |

The POST routes used URL-encoded form data. `/match` and `/api/match` used the same representative paper fields. `/api/keywords` used those fields plus `sdg=01&limit=50`.

## Not fully benchmarked

| Endpoint | Observation | Why no normal-path benchmark |
|---|---|---|
| `/api/verify` | 503, 0.242 ms | Turso credentials are not configured in this environment. |
| `/api/chain-status` | 503, 0.241 ms | Turso credentials are not configured in this environment. |
| `/api/validate` | 503, 0.256 ms | Turso credentials are not configured. The request did not write a validation record. |
| `/doi` | Not run | Calls the external Crossref service. |

## Notes

- ApacheBench reported integer-millisecond percentiles. Many sub-millisecond percentile values display as `0`, so they are omitted here rather than presented as precise measurements.
- ApacheBench's variable-response-length option (`-l`) was used for routes whose JSON output changes length between requests.
- Match endpoint requests append normal metadata entries to `logs/matches.jsonl`. Those entries were left intact.
- No server implementation code was changed. The release binary was rebuilt from the current Rust source before the final benchmark pass.
