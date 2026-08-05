# Zero-Cost Deployment

This app is pure Python **standard library** (no pip installs) and runs as a
plain HTTP server, which makes it trivially deployable on free tiers. All
options below cost **$0.00** and need **no credit card**.

| Option | Cost | Sleeps? | Setup effort |
|---|---|---|---|
| **Hugging Face Spaces** (recommended) | $0 forever | after 48 h idle, wakes on visit | medium |
| **Render** free tier | $0 (750 h/month) | after 15 min idle, cold start ~1 min | low |
| PythonAnywhere free | $0 | never | medium (needs WSGI) |

The Dockerfile builds the SQLite query DB into the image, so first requests
are fast. The DOI auto-fill (Crossref) works on all of these — outbound HTTPS
is allowed.

---

## Option 1 — Hugging Face Spaces (recommended)

Free CPU-basic Docker Spaces, no hourly limits, auto-sleep after 48 h of no
traffic (first visit after sleep takes ~30–60 s to wake).

### 1. Create the Space

1. Sign up at <https://huggingface.co/join> (free).
2. Go to <https://huggingface.co/new-space>:
   - **Space name:** `sdg-paper-matcher`
   - **License:** MIT
   - **SDK:** `Docker`
   - **Hardware:** `CPU basic` (free) — 2 vCPU / 16 GB, enough for this app
   - **Visibility:** Public (private Spaces need a paid plan)
   - Create Space.

### 2. Deploy — pick one:

**A. Automatic (recommended): GitHub Actions** — this repo already contains
`.github/workflows/deploy-hf.yml`.

1. Create a write token at <https://huggingface.co/settings/tokens>.
2. On GitHub: repo **Settings → Secrets and variables → Actions → New
   repository secret** → name `HF_TOKEN`, paste the token.
3. (Optional) Set repo **variable** `HF_SPACE_ID` to `youruser/sdg-paper-matcher`
   if you named your Space differently. Defaults to
   `herlangga72/sdg-paper-matcher`.
4. Push to `main` — the workflow syncs the repo to the Space and prints the
   public URL. Every future push auto-deploys.

**B. Manual (no GitHub Actions):**

```bash
git clone https://huggingface.co/spaces/<youruser>/sdg-paper-matcher
cd sdg-paper-matcher
# copy the project files (NOT the Space's README.md — it has the
# required "sdk: docker" front matter that HF generates)
cp -r <this-repo>/engine <this-repo>/web <this-repo>/papers <this-repo>/Dockerfile \
      <this-repo>/entrypoint.sh <this-repo>/LICENSE .
git add -A && git commit -m "Deploy SDG Paper Matcher" && git push
```

### 3. Done

Your app: `https://<youruser>-sdg-paper-matcher.hf.space` (HF builds the
Docker image automatically on push).

---

## Option 2 — Render (free tier)

Zero code changes. Your repo is already on GitHub.

1. Sign up at <https://render.com> (free, no card).
2. **New → Web Service** → connect the `sdg-paper-matcher` repo.
3. **Runtime:** Docker. **Instance type:** Free. Region: any.
4. **Health check path:** `/health` (the app exposes it).
5. Create — Render builds the image and starts the service.

Alternatively, **New → Blueprint** and select this repo (it contains
`render.yaml` with everything preconfigured, including the free plan).

Your app: `https://sdg-paper-matcher.onrender.com`

Notes on the free tier: sleeps after 15 min of no traffic (cold start ~1 min),
and you get ~750 free hours/month (plenty for a hobby tool; it counts only
while the service is running).

---

## Option 3 — PythonAnywhere (free)

Always-on free tier (never sleeps), 1 web app, 512 MB disk. Needs a WSGI
adapter since PythonAnywhere doesn't run `http.server`:

```python
# wsgi.py — add to the repo and point PythonAnywhere at it
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from web.app import Handler  # noqa: E402  (Handler is a WSGI-compatible class)
application = Handler  # needs the do_GET/do_POST methods exposed — see below
```

`http.server` handlers are not WSGI apps, so the practical approach is:

```bash
# On PythonAnywhere: install nothing, just run the app in a "Always-on task"
# (free tier supports one) — or use the Flask-less trick:
python3 web/app.py --host 0.0.0.0 --port 8000 --no-browser
```

PythonAnywhere free "always-on tasks" are a paid feature on newer plans, so
if you want a truly always-on free host, prefer Hugging Face Spaces or Render
for this project. (Listed here for completeness — it works, but with more
friction.)

---

## Cost & behavior summary

- **Hugging Face Spaces:** $0, no card, sleeps after 48 h idle → wakes on
  visit. Best long-term home.
- **Render free:** $0, no card, sleeps after 15 min idle, 750 h/month.
- **DOI auto-fill:** works everywhere (outbound HTTPS to Crossref allowed).
- **No data is stored server-side** — papers are matched in memory and never
  persisted, so there's nothing to migrate or clean up.
