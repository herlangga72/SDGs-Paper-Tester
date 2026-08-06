#!/usr/bin/env python3
# ---
# title: sdg web app
# description: Zero-dependency webserver + browser UI for the SDG paper matcher
# purpose: Serve the paper-matching UI; everything else lives in engine/
# version: 2.0.0
# created: 2025-08-05
# project: sdg-paper-matcher
# language: python (standard library only: http.server, urllib, email, sqlite3)
# usage: python3 web/app.py [--host 127.0.0.1] [--port 8000] [--no-browser]
# input:  papers entered via the form (separate fields), pasted YAML, uploaded
#         file, or auto-filled from a DOI via the Crossref API
# output: HTML report per SDG: matched blocks, near misses with the exact
#         missing keywords (+ the field Scopus looks at), excluded terms, and
#         keyword hits highlighted in the paper text
# layout:
#   web/app.py            this server (API + result rendering)
#   web/static/           index.html, style.css, app.js (the UI shell)
#   engine/               parser, DB builder, matcher (imported as engine.*)
#   papers/               sample papers offered in the UI
# related: [engine/match_paper.py, engine/sdg2sqlite.py, engine/parse_sdg.py]
# ---
"""
web/app.py — local webserver for the SDG paper matcher.

Run from anywhere:

    python3 web/app.py

then open http://127.0.0.1:8000 . The UI offers four ways to enter a paper
(separate form fields, raw YAML paste, sample papers, file upload) plus
auto-fill from a DOI via the Crossref API. Matching uses the engine package
(engine/match_paper.py) and the query DB built by engine/sdg2sqlite.py
(engine/data/sdg_queries.sqlite3); if the DB is missing the SDG*.txt files
in engine/data/queries are parsed directly as a fallback.

Endpoints:
    GET  /                       UI page
    GET  /static/<file>          CSS / JS
    GET  /samples                JSON list of sample papers (name, title, year)
    GET  /sample?name=&format=   raw markdown or parsed JSON fields
    GET  /doi?doi=...            Crossref lookup -> JSON fields
    POST /match                  fields -> HTML report
    GET  /health                 liveness
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from email import policy
from email.parser import BytesParser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from engine.match_paper import (Paper, eval_node, load_queries_from_db,
                                load_queries_from_dir, parse_paper_text,
                                term_hit)
from engine.parse_sdg import FieldWrap, Group, Leaf, Not

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
PAPERS_DIR = ROOT / "papers"
QUERY_DIR = ROOT / "engine" / "data" / "queries"
DEFAULT_DB = QUERY_DIR / "sdg_queries.sqlite3"

# --------------------------------------------------------------------------
# SDG metadata (official UN short names + brand colors)
# --------------------------------------------------------------------------

SDGS = {
    "01": ("No Poverty", "#E5243B"), "02": ("Zero Hunger", "#DDA63A"),
    "03": ("Good Health and Well-being", "#4C9F38"), "04": ("Quality Education", "#C5192D"),
    "05": ("Gender Equality", "#FF3A21"), "06": ("Clean Water and Sanitation", "#26BDE2"),
    "07": ("Affordable and Clean Energy", "#FCC30B"), "08": ("Decent Work and Economic Growth", "#A21942"),
    "09": ("Industry, Innovation and Infrastructure", "#FD6925"), "10": ("Reduced Inequalities", "#DD1367"),
    "11": ("Sustainable Cities and Communities", "#FD9D24"), "12": ("Responsible Consumption and Production", "#BF8B2E"),
    "13": ("Climate Action", "#3F7E44"), "14": ("Life Below Water", "#0A97D9"),
    "15": ("Life on Land", "#56C02B"), "16": ("Peace, Justice and Strong Institutions", "#00689D"),
    "17": ("Partnerships for the Goals", "#19486A"),
}


def sdg_name(no: str) -> str:
    return SDGS.get(no, ("SDG " + no, "#555"))[0]


def sdg_color(no: str) -> str:
    return SDGS.get(no, ("", "#555"))[1]


# --------------------------------------------------------------------------
# Query cache (loaded once; matching itself is ~1-2 s per paper)
# --------------------------------------------------------------------------

_lock = threading.Lock()
_QUERIES: list[tuple[str, list]] | None = None


def get_queries() -> list[tuple[str, list]]:
    global _QUERIES
    if _QUERIES is None:
        with _lock:
            if _QUERIES is None:
                if DEFAULT_DB.exists():
                    _QUERIES = load_queries_from_db(str(DEFAULT_DB))
                else:
                    _QUERIES = load_queries_from_dir(str(QUERY_DIR))
    return _QUERIES


# --------------------------------------------------------------------------
# Matching (identical semantics to engine/match_paper.py)
# --------------------------------------------------------------------------

def scan_with_fields(block, paper: Paper):
    """Like match_paper.scan_block, but each entry also carries the field(s)
    the keyword is searched in ('' -> the default TITLE-ABS-KEY)."""
    hits, misses, ex_hits = [], [], []

    def rec(node, fields, excluded):
        if isinstance(node, Leaf):
            found = term_hit(node.keyword, fields, paper)
            entry = (node.keyword, ",".join(fields) if fields else "")
            if excluded:
                if found:
                    ex_hits.append(entry)
            elif found:
                hits.append(entry)
            else:
                misses.append(entry)
        elif isinstance(node, FieldWrap):
            rec(node.child, node.fields, excluded)
        elif isinstance(node, Not):
            rec(node.child, fields, not excluded)
        elif isinstance(node, Group):
            for c in node.children:
                rec(c, fields, excluded)

    rec(block, (), False)
    return hits, misses, ex_hits


def match_paper(paper: Paper, top: int, max_kw: int) -> list[dict]:
    """Full report: one dict per SDG."""
    out = []
    for sdg, blocks in get_queries():
        matched, near, ex = [], [], []
        for bno, block in enumerate(blocks):
            hits, misses, ex_hits = scan_with_fields(block, paper)
            ex.extend(ex_hits)
            if eval_node(block, (), paper):
                matched.append((bno, hits))
            else:
                near.append((bno, misses, len(hits)))
        near.sort(key=lambda t: len(t[1]))  # fewest missing keywords first
        out.append({
            "sdg": sdg,
            "matched": matched,
            "near": near[:top],
            "near_total": len(near),
            "excluded": sorted({k for k, _ in ex}),
            "max_kw": max_kw,
        })
    return out


def paper_from_fields(f: dict) -> tuple[Paper, dict]:
    """Build a Paper straight from the form inputs (no YAML round-trip),
    with the same fallback semantics as parse_paper_text: fields that are
    missing fall back to the full text."""
    sections: dict[str, str] = {}
    meta: dict[str, object] = {}
    if f.get("title"):
        sections["TITLE"] = f["title"]
        meta["title"] = f["title"]
    if f.get("abstract"):
        sections["ABS"] = f["abstract"]
        meta["abstract"] = f["abstract"]
    if f.get("keywords"):
        kws = [k.strip() for k in re.split(r"[;,]", f["keywords"]) if k.strip()]
        joined = ", ".join(kws)
        sections["KEY"] = joined
        sections["AUTHKEY"] = joined
        meta["keywords"] = kws
    if f.get("authors"):
        meta["authors"] = [a.strip() for a in re.split(r"[;,]", f["authors"]) if a.strip()]
    for k in ("year", "journal", "doi"):
        if f.get(k):
            meta[k] = f[k]
    full = "\n".join(sections.values())
    return Paper(sections, full), meta


# --------------------------------------------------------------------------
# Keyword highlighting in the paper text (span-based, HTML-safe)
# --------------------------------------------------------------------------

def _hl_pattern(term: str):
    """Like match_paper.term_pattern but for highlighting: non-greedy and
    line-local, so a wildcard term like 'developing* countr*' marks a short
    span instead of the whole text (the *presence* check still uses the
    greedy DOTALL pattern from term_pattern)."""
    out = []
    for piece in re.split(r"([*?])", term):
        if piece == "*":
            out.append("[^\\n]*?")
        elif piece == "?":
            out.append("[^\\n]")
        else:
            out.append(re.escape(piece))
    rx = rf"\b{''.join(out)}\b" if "*" not in term and "?" not in term else "".join(out)
    return re.compile(rx, re.IGNORECASE)


def highlight(text: str, terms: list[str]) -> str:
    spans: list[tuple[int, int]] = []
    for t in terms:
        for m in _hl_pattern(t).finditer(text):
            spans.append((m.start(), m.end()))
    if not spans:
        return escape(text)
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    covered = sum(e - s for s, e in merged)
    worst = max((e - s for s, e in merged), default=0)
    if worst > len(text) * 0.8:  # one degenerate catch-all term (e.g. '*') -> skip
        return escape(text)
    out, pos = [], 0
    for s, e in merged:
        out.append(escape(text[pos:s]))
        out.append(f"<mark>{escape(text[s:e])}</mark>")
        pos = e
    out.append(escape(text[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------
# DOI lookup (Crossref REST API — free, no key; requires internet)
# --------------------------------------------------------------------------

CROSSREF_URL = "https://api.crossref.org/works/{}"
CROSSREF_UA = "sdg-paper-matcher/2.0 (local paper-matching app)"
_JATS_TAG = re.compile(r"<[^>]+>")


def fetch_doi(doi: str) -> dict:
    """Fetch paper metadata from Crossref for a DOI.

    Returns a dict in the same shape as the sample endpoint (title, authors,
    year, journal, doi, abstract, keywords) — empty keys omitted. Raises
    ValueError for bad/unknown DOIs and RuntimeError for network/API errors.
    """
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/") \
        .removeprefix("https://dx.doi.org/").removeprefix("http://dx.doi.org/") \
        .removeprefix("doi:")
    if not re.fullmatch(r"10\.\d{4,9}/[^\s]+", doi):
        raise ValueError(f"not a valid DOI: {doi!r} (expected e.g. 10.1257/jep.28.4.99)")

    req = urllib.request.Request(
        CROSSREF_URL.format(urllib.parse.quote(doi, safe="")),
        headers={"User-Agent": CROSSREF_UA, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"DOI not found in Crossref: {doi}")
        raise RuntimeError(f"Crossref API error {e.code}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e.reason}")

    msg = data.get("message", {})
    out: dict = {"doi": doi}

    if msg.get("title"):
        out["title"] = html.unescape(msg["title"][0]).strip()
    authors = []
    for a in msg.get("author", []):
        name = " ".join(x for x in (a.get("given"), a.get("family")) if x).strip() \
            or a.get("name", "")
        if name:
            authors.append(html.unescape(name))
    if authors:
        out["authors"] = authors
    if msg.get("container-title"):
        out["journal"] = html.unescape(msg["container-title"][0]).strip()
    for key in ("issued", "published-print", "published-online"):
        parts = msg.get(key, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            out["year"] = str(parts[0][0])
            break
    if msg.get("abstract"):  # JATS-XML with tags -> plain text
        abstract = re.sub(r"\s+", " ", _JATS_TAG.sub(" ", html.unescape(msg["abstract"]))).strip()
        if abstract:
            out["abstract"] = abstract
    subjects = [html.unescape(s).strip() for s in msg.get("subject", []) if s.strip()]
    if subjects:  # publisher subjects are the closest thing Crossref has to keywords
        out["keywords"] = subjects
    return out


# --------------------------------------------------------------------------
# HTML rendering of the match report
# --------------------------------------------------------------------------

def _meta_str(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _kw_tags(entries, cls, max_kw: int) -> str:
    """entries: list of (keyword, fields); render keyword chips."""
    if not entries:
        return '<span class="none">none</span>'
    shown, rest = entries[:max_kw], entries[max_kw:]
    out = []
    for kw, fields in shown:
        field = f'<span class="field">[{fields or "TITLE-ABS-KEY"}]</span>' if fields else ""
        out.append(f'<span class="kw {cls}">{escape(kw)}{field}</span>')
    if rest:
        out.append(f'<span class="muted-text">… +{len(rest)} more</span>')
    return "".join(out)


def render_results(report: list[dict], paper: Paper, meta: dict, ms: float) -> str:
    meta_parts = []
    if meta.get("title"):
        meta_parts.append(f"<b>{escape(_meta_str(meta['title']))}</b>")
    for k in ("authors", "year", "journal", "doi"):
        if meta.get(k):
            meta_parts.append(escape(_meta_str(meta[k])))
    info_html = f'<div class="card paper-info">{" · ".join(meta_parts)}</div>' if meta_parts else ""

    matched_sdgs = [r for r in report if r["matched"]]
    near_sdgs = [r for r in report if not r["matched"] and r["near"]]
    ex_sdgs = [r for r in report if r["excluded"]]

    chips = []
    for r in matched_sdgs:
        color = sdg_color(r["sdg"])
        chips.append(f'<span class="chip matched" style="color:{color}">'
                     f'<span class="dot" style="background:{color}"></span>SDG {r["sdg"]} ✓</span>')
    for r in near_sdgs:
        color = sdg_color(r["sdg"])
        chips.append(f'<span class="chip near" style="color:{color}">'
                     f'<span class="dot" style="background:{color}"></span>SDG {r["sdg"]} near</span>')
    for r in ex_sdgs:
        color = sdg_color(r["sdg"])
        chips.append(f'<span class="chip" style="color:{color}">'
                     f'<span class="dot" style="background:{color}"></span>SDG {r["sdg"]} ⚠ excluded terms</span>')
    chips_html = f'<div class="chips">{"".join(chips) or "<span class=muted-text>no SDG signal found</span>"}</div>'

    stat = (f'<div class="stat"><b>{len(matched_sdgs)}</b> of <b>17</b> SDGs matched'
            f' · <b>{len(near_sdgs)}</b> near misses · <b>{len(ex_sdgs)}</b> with excluded terms found'
            f' · processed in <b>{ms:.1f}</b> ms</div>')

    cards = []
    for r in report:
        color = sdg_color(r["sdg"])
        matched, near, ex = r["matched"], r["near"], r["excluded"]
        if not (matched or near or ex):
            continue
        badges = []
        if matched:
            badges.append(f'<span class="badge ok">✓ {len(matched)} block(s) matched</span>')
        if near:
            badges.append(f'<span class="badge miss">{len(near)} near miss(es)</span>')
        if ex:
            badges.append(f'<span class="badge ex">excluded terms found</span>')

        body = []
        if matched:
            body.append('<div class="block"><h4>Matched blocks — keywords that hit</h4>')
            for bno, hits in matched:
                body.append(f'<div class="muted-text" style="margin:4px 0 2px">block {bno}</div>')
                body.append(_kw_tags(hits, "hit", r["max_kw"]))
            body.append("</div>")
        if near:
            body.append('<div class="block"><h4>Near misses — add any of these keywords to qualify</h4>')
            for bno, misses, n_hit in near:
                body.append(f'<div class="muted-text" style="margin:4px 0 2px">block {bno}: '
                            f'{n_hit} keyword(s) already hit</div>')
                body.append(_kw_tags(misses, "missing", r["max_kw"]))
            if r["near_total"] > len(near):
                body.append(f'<div class="muted-text">… {r["near_total"] - len(near)} more near-miss blocks not shown</div>')
            body.append("</div>")
        if ex:
            body.append('<div class="block"><h4>Excluded terms found in the text (can disqualify a match)</h4>')
            body.append(_kw_tags([(k, "") for k in ex], "ex", r["max_kw"]))
            body.append("</div>")

        cards.append(f'''
<details class="card sdg-card" {'open' if matched else ''}>
  <summary class="sdg-head">
    <span class="num" style="background:{color}">{r["sdg"]}</span>
    <span class="name">{escape(sdg_name(r["sdg"]))}</span>
    <span class="badges">{"".join(badges)}</span>
    <span class="chev">▼</span>
  </summary>
  <div class="sdg-body">{"".join(body)}</div>
</details>''')

    # highlight all matched keywords in the full paper text
    all_terms = sorted({kw for r in matched_sdgs for _, hits in r["matched"] for kw, _ in hits})
    hl = ""
    if all_terms:
        text = paper.full_text.strip() or "\n".join(paper.sections.values())
        hl = f'''<div class="card highlight-card">
  <h3>Matched keywords highlighted in the paper text ({len(all_terms)})</h3>
  <div class="papertext">{highlight(text, all_terms)}</div>
</div>'''

    return (f'<div id="results-inner"><h2 class="section">Results</h2>{info_html}{stat}{chips_html}'
            f'<div id="cards">{"".join(cards)}</div>{hl}</div>')


def error_box(msg: str) -> str:
    return f'<div class="error-box">{escape(msg)}</div>'


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SDGMatcher/2.0"

    # -- helpers ----------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8",
              extra_headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _read_form(self) -> tuple[dict[str, str], dict[str, bytes]]:
        """Parse x-www-form-urlencoded or multipart/form-data. Returns
        (fields, files) with decoded str values / raw file bytes."""
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        fields, files = {}, {}
        if "multipart/form-data" in ctype:
            # the parser needs the Content-Type header (with boundary) to
            # split the body into parts — prepend it
            head = f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            msg = BytesParser(policy=policy.default).parsebytes(head + raw)
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if not name:
                    continue
                filename = part.get_filename()
                data = part.get_payload(decode=True) or b""
                if filename:
                    files[name] = data
                else:
                    fields[name] = data.decode("utf-8", "replace")
        elif ctype.startswith("application/x-www-form-urlencoded"):
            fields = {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "replace")).items()}
        return fields, files

    def _sample_meta(self) -> list[dict]:
        """[{name, title, year}] for every papers/*.md, for the UI."""
        out = []
        for p in sorted(PAPERS_DIR.glob("*.md")):
            meta = parse_paper_text(p.read_text(encoding="utf-8"))[1]
            out.append({
                "name": p.name,
                "title": meta.get("title") or p.name,
                "year": meta.get("year", ""),
            })
        return out

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            page = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, page)
        elif url.path.startswith("/static/"):
            name = Path(url.path[len("/static/"):]).name  # basename only, no traversal
            path = (STATIC_DIR / name).resolve()
            if path.is_relative_to(STATIC_DIR) and path.is_file():
                self._send(200, path.read_bytes(), MIME.get(path.suffix, "application/octet-stream"))
            else:
                self._send(404, b"not found", "text/plain")
        elif url.path == "/samples":
            self._send_json(200, self._sample_meta())
        elif url.path == "/sample":
            qs = parse_qs(url.query)
            name = qs.get("name", [""])[0]
            fmt = qs.get("format", [""])[0]
            path = (PAPERS_DIR / name).resolve()
            if path.is_relative_to(PAPERS_DIR) and path.is_file() and path.suffix == ".md":
                if fmt == "json":  # parsed fields, so the UI can fill its form
                    _, meta = parse_paper_text(path.read_text(encoding="utf-8"))
                    meta["raw"] = path.read_text(encoding="utf-8")
                    self._send_json(200, meta)
                else:
                    self._send(200, path.read_bytes(), "text/markdown; charset=utf-8")
            else:
                self._send(404, b"sample not found", "text/plain")
        elif url.path == "/doi":
            doi = parse_qs(url.query).get("doi", [""])[0]
            try:
                self._send_json(200, fetch_doi(doi))
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            except RuntimeError as e:
                self._send_json(502, {"error": str(e)})
        elif url.path == "/health":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/match":
            self._send(404, b"not found", "text/plain")
            return
        try:
            fields, files = self._read_form()
            paper = meta = None
            # 1) separate form fields win if any are filled in
            form = {k: fields.get(k, "").strip() for k in
                    ("title", "abstract", "keywords", "authors", "year", "journal", "doi")}
            if any(form.values()):
                paper, meta = paper_from_fields(form)
            else:
                # 2) raw pasted text, 3) uploaded file
                text = fields.get("paper", "")
                if not text:
                    for fname in ("file", "paper"):  # accept uploads under either name
                        if fname in files:
                            text = files[fname].decode("utf-8", "replace")
                            break
                if not text.strip():
                    self._send(200, error_box("No paper entered — fill in the form (Title / Abstract / Keywords), paste raw text, or upload a file.").encode())
                    return
                paper, meta = parse_paper_text(text)
            top = min(max(int(fields.get("top", "3") or 3), 1), 20)
            max_kw = min(max(int(fields.get("maxkw", "10") or 10), 1), 50)
            t0 = time.perf_counter()
            report = match_paper(paper, top, max_kw)
            ms = (time.perf_counter() - t0) * 1000.0
            html = render_results(report, paper, meta, ms)
            self._send(200, html.encode(),
                       extra_headers={"X-Processing-Time": f"{ms:.1f} ms"})
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            self._send(200, error_box(f"Matching failed: {e}").encode())

    def log_message(self, fmt, *args):
        sys.stderr.write("[web] %s\n" % (fmt % args))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="SDG Paper Matcher web app (zero dependencies)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    # warm the query cache so the first request isn't slow
    try:
        n = len(get_queries())
        print(f"[web] loaded {n} SDG query sets", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[web] warning: could not load queries: {e}", file=sys.stderr)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[web] SDG Paper Matcher running at {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
