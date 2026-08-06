# Zero-Cost Deployment

The app runs as a plain HTTP server with a self-contained Rust binary
(SIMD-accelerated matching) and needs no runtime package installs, which
makes it trivially deployable on free tiers. All options below cost
**$0.00** and need **no credit card**.

> **Note (2026):** Hugging Face now requires a **PRO subscription** to host
> Docker/Gradio Spaces on the free tier (static Spaces only are free), so HF
> is no longer a zero-cost option for this app. The two viable paths are
> **Render** and **PythonAnywhere** below.

| Option | Cost | Sleeps? | Setup effort |
|---|---|---|---|
| **Render** free tier | $0 (750 h/month) | after 15 min idle, cold start ~1 min | **low — recommended** |
| **PythonAnywhere** free | $0 | **never (always on)** | medium (WSGI, included) |

The Dockerfile builds the SQLite query DB into the image, so first requests
are fast. The DOI auto-fill (Crossref) works on both — outbound HTTPS is
allowed.

---

## Option 1 — Render (recommended, lowest effort)

Your repo is already on GitHub — no new files needed.

1. Sign up at <https://render.com> (free, no credit card).
2. **New → Web Service** → connect the `herlangga72/sdg-paper-matcher` repo.
3. **Runtime:** Docker. **Instance type:** Free. Region: any (e.g. Frankfurt).
4. **Health check path:** `/health` (the app exposes it).
5. Create — Render builds the image and starts the service.

Alternatively, **New → Blueprint** and select this repo: `render.yaml` is
already in the repo with everything preconfigured (free plan, Docker, health
check).

Your app: `https://sdg-paper-matcher.onrender.com`

Free-tier behavior: sleeps after 15 min of no traffic (cold start ~1 min),
~750 free hours/month (plenty for a hobby tool — it only counts while the
service is awake).

---

## Option 2 — PythonAnywhere (always on, never sleeps)

Free tier: one web app, 512 MB disk, always online. The repo ships a WSGI
adapter (`wsgi.py`) so no code changes are needed.

1. Sign up at <https://www.pythonanywhere.com> (free, no credit card).
2. Open a **Bash console** and clone + build the DB:

   ```bash
   git clone https://github.com/herlangga72/sdg-paper-matcher.git
   cd sdg-paper-matcher
   python3.11 engine/sdg2sqlite.py --quiet
   ```

3. **Web tab → Add a new web app → Manual configuration → Python 3.11**.
   - Source directory: `/home/<youruser>/sdg-paper-matcher`
   - Working directory: `/home/<youruser>/sdg-paper-matcher`
4. Edit the **WSGI configuration file** — replace its contents with:

   ```python
   import sys
   sys.path.insert(0, '/home/<youruser>/sdg-paper-matcher')
   from wsgi import application
   ```

5. Click **Reload**.

Your app: `https://<youruser>.pythonanywhere.com`

Notes: free-tier CPU is throttled (matching a paper takes ~1–2 s, fine for
occasional use); static files are served by the app itself, so no extra
static-file mapping is needed.

---

## What happened to the Hugging Face option?

`DEPLOY.md` earlier pointed at free HF Docker Spaces, and a GitHub Actions
workflow (`deploy-hf.yml`) was added to auto-deploy there. Hugging Face's API
now rejects creating Docker Spaces on the free tier:

> "hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
> subscription"

so that workflow was removed. The `HF_TOKEN` GitHub secret is harmless to
keep — it's useful the day you have PRO or want to use HF for other things.
If HF ever re-opens free Docker Spaces, the workflow can be restored from git
history (`git show <commit>:.github/workflows/deploy-hf.yml`).

---

## Cost & behavior summary

- **Render free:** $0, no card, sleeps after 15 min idle, 750 h/month.
- **PythonAnywhere free:** $0, no card, always on, throttled CPU.
- **DOI auto-fill:** works on both (outbound HTTPS to Crossref allowed).
- **No data is stored server-side** — papers are matched in memory and never
  persisted, so there's nothing to migrate or clean up.
