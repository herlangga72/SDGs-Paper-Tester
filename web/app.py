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
    GET  /api/stats              usage counters (visits, matches; cumulative)
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from email import policy
from email.parser import BytesParser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from engine.match_paper import (INF_COST, Paper, collect_sdg_dict, eval_node,
                                eval_ignore_not, load_queries_from_db,
                                load_queries_from_dir, min_add, parse_paper_text,
                                score_keywords, suggest_keywords, term_hit)
from engine.parse_sdg import FieldWrap, Group, Leaf, Not

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
PAPERS_DIR = ROOT / "papers"
QUERY_DIR = ROOT / "engine" / "data" / "queries"
DEFAULT_DB = QUERY_DIR / "sdg_queries.sqlite3"

# --------------------------------------------------------------------------
# Usage logging: durable visit/match counters + per-request access log
#
# Every request (except /health and /api/stats themselves) is appended to
# logs/access.jsonl (one JSON line each, no IPs or user agents) and counted
# in-process. Counters persist to engine/data/site_stats.json every 25
# requests and reload at boot, so totals survive restarts of the same
# instance (local dev, PythonAnywhere). Free-tier hosts that rebuild the
# container on every deploy reset the file; the UI hides the footer counter
# when /api/stats is missing or empty. Standard library only.
# --------------------------------------------------------------------------

_STATS_FILE = ROOT / "engine" / "data" / "site_stats.json"
_ACCESS_LOG = ROOT / "logs" / "access.jsonl"
_STATS_LOCK = threading.Lock()
_BOOT_TIME = time.time()
_STATS = {"total": 0, "pages": 0, "match_html": 0, "api_match": 0,
          "api_keywords": 0, "errors": 0, "not_found": 0, "booted_at": 0}
_STATS_LOADED = False


def _load_stats() -> None:
    """Resume cumulative counters saved by a previous run (zeros if absent)."""
    global _STATS_LOADED
    with _STATS_LOCK:
        try:
            data = json.loads(_STATS_FILE.read_text(encoding="utf-8"))
            for key in _STATS:
                if key in data:
                    _STATS[key] = int(data[key])
        except Exception:  # noqa: BLE001 — first run / corrupt file: start at zero
            pass
        _STATS["booted_at"] = int(_BOOT_TIME * 1000)
        _STATS_LOADED = True


def _save_stats() -> None:
    try:
        _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATS_FILE.write_text(json.dumps(_STATS, indent=2) + "\n",
                               encoding="utf-8")
    except Exception:  # noqa: BLE001 — logging must never break a request
        pass


def _append_access_line(line: str) -> None:
    """One JSON line per request; rotate at ~4 MB (keep access.jsonl.1)."""
    try:
        _ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        if _ACCESS_LOG.exists() and _ACCESS_LOG.stat().st_size > 4 << 20:
            _ACCESS_LOG.replace(_ACCESS_LOG.with_name("access.jsonl.1"))
        with _ACCESS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def _record_request(method: str, path: str, status: int, ms: float) -> None:
    """Classify + count one request; /health and /api/stats are excluded."""
    if not _STATS_LOADED:
        _load_stats()
    path = path.split("?", 1)[0]
    if path in ("/health", "/api/stats"):
        return
    with _STATS_LOCK:
        _STATS["total"] += 1
        if status >= 500:
            _STATS["errors"] += 1
        elif status == 404:
            _STATS["not_found"] += 1
        if (method, path) in (("GET", "/"), ("GET", "/index.html")):
            if status < 400:
                _STATS["pages"] += 1
        elif (method, path) == ("POST", "/match"):
            if status < 400:
                _STATS["match_html"] += 1
        elif (method, path) == ("POST", "/api/match"):
            if status < 400:
                _STATS["api_match"] += 1
        elif (method, path) == ("POST", "/api/keywords"):
            if status < 400:
                _STATS["api_keywords"] += 1
        total = _STATS["total"]
        line = json.dumps({"ts": int(time.time() * 1000), "method": method,
                           "path": path, "status": status,
                           "ms": round(ms, 1)}, ensure_ascii=True)
    _append_access_line(line)
    if total % 25 == 0:
        _save_stats()


def stats_payload() -> dict:
    """GET /api/stats — cumulative usage counters + unique users + uptime."""
    if not _STATS_LOADED:
        _load_stats()
    with _STATS_LOCK:
        base = {"total": _STATS["total"], "pages": _STATS["pages"],
                "match_html": _STATS["match_html"],
                "api_match": _STATS["api_match"],
                "api_keywords": _STATS["api_keywords"],
                "errors": _STATS["errors"],
                "not_found": _STATS["not_found"],
                "booted_at": _STATS["booted_at"],
                "uptime_s": round(time.time() - _BOOT_TIME, 1)}
    base.update(_user_stats())
    return base


# --------------------------------------------------------------------------
# Unique-user tracking (anonymous uid cookie) — Python mirror of web.rs
#
# A page load without a `uid` cookie receives one (32 hex chars, HttpOnly,
# SameSite=Lax, ~6 months) and is added to engine/data/visitors.json, which
# maps uid -> last-seen epoch ms. Distinct-user counts for any window are
# derived from last-seen, so no per-day sets are needed. No IPs, no personal
# data; only full page loads mint cookies. Standard library only.
# --------------------------------------------------------------------------

_VISITORS_FILE = ROOT / "engine" / "data" / "visitors.json"
_VISITORS_LOCK = threading.Lock()
_VISITORS: dict[str, int] = {}
_VISITORS_LOADED = False


def _load_visitors() -> None:
    """Resume the uid -> last-seen map saved by a previous run."""
    global _VISITORS_LOADED
    with _VISITORS_LOCK:
        try:
            data = json.loads(_VISITORS_FILE.read_text(encoding="utf-8"))
            for uid, last in (data.get("users") or {}).items():
                try:
                    ms = int(last)
                except (TypeError, ValueError):
                    continue
                if len(uid) == 32 and all(c in "0123456789abcdef" for c in uid.lower()):
                    _VISITORS[uid] = ms
        except Exception:  # noqa: BLE001 — first run / corrupt file: start empty
            pass
        _VISITORS_LOADED = True


def _save_visitors() -> None:
    with _VISITORS_LOCK:
        if len(_VISITORS) > 100_000:  # bound the file: prune idle > 180 days
            cutoff = int(time.time() * 1000) - 180 * 86_400_000
            for uid in [u for u, last in _VISITORS.items() if last < cutoff]:
                del _VISITORS[uid]
        data = {"users": _VISITORS}
    try:
        _VISITORS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _VISITORS_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001 — logging must never break a request
        pass


def _cookie_uid(headers) -> str | None:
    cookie = headers.get("Cookie", "") if headers else ""
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith("uid="):
            value = part[4:].strip()
            if len(value) == 32 and all(c in "0123456789abcdef" for c in value.lower()):
                return value
    return None


def _track_page_user(handler) -> str | None:
    """Mint/refresh the uid on a page load; return a Set-Cookie header or None."""
    if not _VISITORS_LOADED:
        _load_visitors()
    uid = _cookie_uid(handler.headers)
    set_cookie = None
    if uid is None:
        uid = secrets.token_hex(16)
        set_cookie = f"uid={uid}; Path=/; Max-Age=15552000; HttpOnly; SameSite=Lax"
    now = int(time.time() * 1000)
    with _VISITORS_LOCK:
        is_new = uid not in _VISITORS
        _VISITORS[uid] = now
    if is_new:
        _save_visitors()  # new users persist immediately
    elif len(_VISITORS) % 256 == 0:
        _save_visitors()  # periodic last-seen refresh
    return set_cookie


def _user_stats() -> dict:
    if not _VISITORS_LOADED:
        _load_visitors()
    today = int(time.time() * 1000) // 86_400_000
    d1 = d7 = d30 = 0
    with _VISITORS_LOCK:
        for last in _VISITORS.values():
            day = last // 86_400_000
            if day == today:
                d1 += 1
            if day + 6 >= today:
                d7 += 1
            if day + 29 >= today:
                d30 += 1
        total = len(_VISITORS)
    return {"users_total": total, "users_today": d1, "users_7d": d7, "users_30d": d30}


# --------------------------------------------------------------------------
# Error reporting (Sentry) — Python mirror of web.rs
#
# The DSN is a hard-coded default (public client key by design), so reporting
# is on out of the box; set SENTRY_DSN to override or SENTRY_DSN=0/off to
# disable. Boot events, unhandled handler exceptions and WSGI adapter
# failures are forwarded to the store over the envelope API with urllib — no
# sentry-sdk install needed, so the PythonAnywhere deploy stays
# standard-library only. Standard library only.
# --------------------------------------------------------------------------

_DEFAULT_SENTRY_DSN = ("https://b7a8b16ab31f6a94ee2944534183f03e"
                       "@o4512018920439808.ingest.us.sentry.io/4512018935447552")
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip()
if _SENTRY_DSN.lower() in ("0", "off"):
    _SENTRY_DSN = ""  # explicit opt-out
elif not _SENTRY_DSN:
    _SENTRY_DSN = _DEFAULT_SENTRY_DSN
_SENTRY_SDK = {"name": "sdg-tools-web-py", "version": "2.0.0"}


def _sentry_target() -> tuple[str, str] | None:
    """(dsn, envelope ingest url) when SENTRY_DSN looks valid, else None."""
    if not _SENTRY_DSN:
        return None
    try:
        parsed = urllib.parse.urlsplit(_SENTRY_DSN)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return None
        host = parsed.netloc.rsplit("@", 1)[-1]
        project = (parsed.path or "").strip("/").split("/")[-1]
        if not host or not project:
            return None
        return _SENTRY_DSN, f"{parsed.scheme}://{host}/api/{project}/envelope/"
    except Exception:  # noqa: BLE001
        return None


def _sentry_send(dsn: str, url: str, event: dict) -> bool:
    now = datetime.datetime.now(datetime.timezone.utc)
    env_hdr = {"event_id": event["event_id"], "dsn": dsn,
               "sent_at": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
               "sdk": _SENTRY_SDK}
    event_json = json.dumps(event, ensure_ascii=True)
    payload = (json.dumps(env_hdr) + "\n" +
               json.dumps({"type": "event", "length": len(event_json)}) + "\n" +
               event_json).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/x-sentry-envelope"})
    with urllib.request.urlopen(req, timeout=4) as resp:
        return resp.status == 200


def sentry_report(level: str, logger: str, message: str,
                  exc: BaseException | None = None,
                  tags: dict | None = None, extra: dict | None = None) -> bool:
    """Best-effort report; returns silently (False) when SENTRY_DSN is unset."""
    target = _sentry_target()
    if not target:
        return False
    dsn, url = target
    now = datetime.datetime.now(datetime.timezone.utc)
    ev = {"event_id": uuid.uuid4().hex,
          "timestamp": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
          "platform": "python", "level": level, "logger": logger,
          "message": {"formatted": message},
          "environment": os.environ.get("SENTRY_ENV", "").strip() or "production"}
    for key in ("RENDER_GIT_COMMIT", "SOURCE_VERSION", "SENTRY_RELEASE"):
        value = os.environ.get(key, "").strip()
        if value:
            ev["release"] = value
            break
    if tags:
        ev["tags"] = {k: str(v) for k, v in tags.items()}
    if extra:
        ev["extra"] = {k: (v if isinstance(v, (dict, list)) else str(v))
                        for k, v in extra.items()}
    if exc is not None:
        ev["exception"] = {"values": [
            {"type": type(exc).__name__, "value": str(exc) or repr(exc)}]}
    try:
        sent = _sentry_send(dsn, url, ev)
    except Exception as e:  # noqa: BLE001 — logging must never break a request
        sys.stderr.write(f"[sentry] {level} event send failed: {e}\n")
        return False
    sys.stderr.write(f"[sentry] {level} event sent ({logger})\n")
    return sent


_PREV_THREAD_EXCEPTHOK = threading.excepthook


def _install_thread_excepthook() -> None:
    """Report unhandled ThreadingHTTPServer handler-thread exceptions."""

    def hook(args) -> None:  # threading.ExceptHookArgs
        sentry_report("error", "web.thread",
                      f"Unhandled exception in {args.thread.name}",
                      exc=args.exc_value,
                      extra={"traceback": "".join(traceback.format_exception(
                          args.exc_type, args.exc_value, args.exc_traceback))})
        try:
            _PREV_THREAD_EXCEPTHOK(args)
        except Exception:  # noqa: BLE001 — never break the server
            sys.stderr.write("".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback)))

    threading.excepthook = hook

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


_DICT_CACHE: dict[str, tuple[set[str], set[str]]] = {}


def sdg_dict(sdg: str, blocks) -> tuple[set[str], set[str]]:
    """Cached (unique include keywords, excluded set) of one SDG."""
    d = _DICT_CACHE.get(sdg)
    if d is None:
        d = collect_sdg_dict(blocks)
        _DICT_CACHE[sdg] = d
    return d


def match_paper(paper: Paper, top: int, max_kw: int) -> list[dict]:
    """Full report: one dict per SDG."""
    out = []
    for sdg, blocks in get_queries():
        matched, near, ex = [], [], []
        present: set[str] = set()
        # Keywords that alone qualify the SDG / still need N more keyword(s).
        solo: set[str] = set()
        extra: dict[str, int] = {}
        for bno, block in enumerate(blocks):
            hits, misses, ex_hits = scan_with_fields(block, paper)
            present.update(kw for kw, _ in hits)
            if eval_node(block, (), paper):
                matched.append((bno, hits))
            else:
                # Near-miss analysis: exact minimum keywords to add ("missing
                # tags"), mirrored from engine/match_paper (no LLM).
                _val, cost, need = min_add(block, (), paper)
                if cost == INF_COST:
                    # Only report excluded terms when the positive side alone
                    # would have matched (off-topic blocks are dropped).
                    if eval_ignore_not(block, (), paper):
                        ex.extend(ex_hits)
                else:
                    if cost == 1 and len(need) == 1:
                        solo.update(need[0])
                    else:
                        need_more = cost - 1
                        for g in need:
                            for k in g:
                                if k not in extra or need_more < extra[k]:
                                    extra[k] = need_more
                    near.append((bno, hits, cost, need))
        # Rerank: fewest keywords to add first, then most keywords hit.
        near.sort(key=lambda t: (t[2], -len(t[1]), sum(len(g) for g in t[3])))
        inc, exc = sdg_dict(sdg, blocks)
        suggestions = suggest_keywords(paper.lowered(""), inc, exc, present, 10)
        out.append({
            "sdg": sdg,
            "matched": matched,
            "near": near[:top],
            "near_total": len(near),
            "excluded": sorted({k for k, _ in ex}),
            "max_kw": max_kw,
            "suggestions": suggestions,
            "solo": solo,
            "extra": extra,
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
_LANDING_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_JATS_TAG = re.compile(r"<[^>]+>")


def _split_keywords(s: str) -> list[str]:
    """Split a keywords string on commas/semicolons: trim, drop empties, dedupe."""
    out: list[str] = []
    for part in re.split(r"[;,]", s):
        t = part.strip()
        if t and t not in out:
            out.append(t)
    return out


def _landing_page_keywords(doi: str) -> list[str]:
    """Best-effort author keywords from the DOI landing page, used when
    Crossref has no `subject`. Tries the Google-Scholar `citation_keywords`
    meta tag first (Springer/Elsevier/IEEE/...), then the
    `Keywords</strong><div class="txt">` markup used by Business
    Perspectives. Never fails the DOI lookup: on any network error, timeout
    or parse miss it returns an empty list.
    """
    url = f"https://doi.org/{urllib.parse.quote(doi, safe='')}"
    req = urllib.request.Request(url, headers={"User-Agent": _LANDING_UA, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            page = r.read().decode("utf-8", "replace")
    except Exception:
        return []
    m = re.search(
        r'<meta\s+name=["\']citation_keywords["\']\s+content=["\']([^"\']*)["\']',
        page, re.I)
    if m:
        kws = _split_keywords(html.unescape(m.group(1)))
        if kws:
            return kws
    m = re.search(r"Keywords</strong>\s*<div class=\"txt\">(.*?)</div>", page, re.I | re.S)
    if m:
        return _split_keywords(html.unescape(_JATS_TAG.sub(" ", m.group(1))))
    return []


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
    else:  # no Crossref subject: fall back to the DOI landing page (best-effort)
        kws = _landing_page_keywords(doi)
        if kws:
            out["keywords"] = kws
    return out


# --------------------------------------------------------------------------
# HTML rendering of the match report
# --------------------------------------------------------------------------

def _meta_str(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _kw_tags(entries, cls, max_kw: int) -> str:
    """entries: list of (keyword, fields); render keyword chips + a
    "Show all N" toggle button revealing the full list (app.js handler)."""
    if not entries:
        return '<span class="none">none</span>'
    shown, rest = entries[:max_kw], entries[max_kw:]
    out = []
    for kw, fields in shown:
        field = f'<span class="field">[{fields or "TITLE-ABS-KEY"}]</span>' if fields else ""
        out.append(f'<span class="kw {cls}">{escape(kw)}{field}</span>')
    if rest:
        hidden = "".join(
            f'<span class="kw {cls}">{escape(kw)}{f"<span class=\"field\">[{fields}]</span>" if fields else ""}</span>'
            for kw, fields in rest
        )
        out.append(
            f'<span class="kw-more" hidden>{hidden}</span>'
            f'<button type="button" class="kw-toggle" data-all="Show all {len(entries)}" '
            f'data-few="Show fewer">Show all {len(entries)}</button>'
        )
    return "".join(out)


def _need_chain(body: list, need: list, max_kw: int) -> None:
    """Append one near-miss path as AND-joined "pick any one" boxes."""
    body.append('<div class="and-chain">')
    n_groups = len(need)
    for gi, group in enumerate(need[:3]):
        if gi > 0:
            body.append('<div class="and-op">AND</div>')
        body.append(f'<div class="and-group"><div class="and-group-label">Box {gi + 1} — pick any ONE:</div>')
        body.append(_kw_tags([(k, "") for k in group], "missing", max_kw))
        body.append('</div>')
    if n_groups > 3:
        rest = []
        for gi, group in enumerate(need[3:], start=4):
            rest.append('<div class="and-op">AND</div>')
            rest.append(f'<div class="and-group"><div class="and-group-label">Box {gi} — pick any ONE:</div>')
            rest.append(_kw_tags([(k, "") for k in group], "missing", max_kw))
            rest.append('</div>')
        body.append(
            f'<div class="kw-more kw-more-block" hidden>{"".join(rest)}</div>'
            f'<button type="button" class="kw-toggle" data-all="Show all {n_groups} boxes" '
            f'data-few="Show fewer boxes">Show all {n_groups} boxes</button>'
        )
    body.append('</div>')


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
            body.append('<div class="block"><h4>How to qualify this SDG</h4>')
            first, *rest = near
            bno, first_hits, cost, need = first
            word = "keyword" if cost == 1 else "keywords"
            body.append(f'<div class="status-line">Your text is <b>{cost} {word}</b> short of '
                        f'qualifying for <b>SDG {r["sdg"]} — {escape(sdg_name(r["sdg"]))}</b>.</div>')
            body.append('<div class="fastest">')
            body.append(f'<div class="fastest-head">⚡ Fastest — add {cost} {word} '
                        f'(pick any one from each box):</div>')
            _need_chain(body, need, r["max_kw"])
            body.append('<div class="min-hint">1 keyword from each box = the SDG qualifies. '
                        'Click a chip to add it to your Keywords field.</div>')
            body.append('<div class="have-line"><b>Already in your text:</b> ')
            if not first_hits:
                body.append('<span class="none">none yet</span>')
            else:
                body.append(_kw_tags(first_hits, "hit", r["max_kw"]))
            body.append('</div></div>')
            if rest:
                alts = []
                for ai, (_, _, c, nd) in enumerate(rest, start=2):
                    w = "keyword" if c == 1 else "keywords"
                    alts.append(f'<div class="way-head">Way {ai} — add {c} {w}:</div>')
                    _need_chain(alts, nd, r["max_kw"])
                body.append(
                    f'<div class="kw-more kw-more-block" hidden>{"".join(alts)}</div>'
                    f'<button type="button" class="kw-toggle" '
                    f'data-all="Show {len(rest)} other ways to qualify" data-few="Hide other ways">'
                    f'Show {len(rest)} other ways to qualify</button>'
                )
            if r["near_total"] > len(near):
                body.append(f'<div class="muted-text">… {r["near_total"] - len(near)} more ways not shown</div>')
            body.append("</div>")
        if ex:
            body.append('<div class="block"><h4>Excluded terms that blocked a near match — remove them from the text to qualify</h4>')
            body.append(_kw_tags([(k, "") for k in ex], "ex", r["max_kw"]))
            body.append("</div>")
        if r.get("suggestions"):
            heading = ("Best-fit keywords to add (click to copy)"
                       if not matched else
                       "Related keywords from this SDG — best fit to your text (click to copy)")
            body.append(f'<div class="block"><h4>{heading}</h4>')
            if not matched:
                body.append('<div class="sug-legend">'
                            '<span class="sug-badge solo">✓ alone</span> qualifies by itself · '
                            '<span class="sug-badge more">+N</span> still needs N more keyword(s) · '
                            '<span class="sug-badge block">⚠ blocked</span> excluded term — adding it blocks a match'
                            '</div>')
            body.append('<div class="sug-row">')
            solo, extra = r.get("solo", set()), r.get("extra", {})
            for s in r["suggestions"][: min(r["max_kw"], 10)]:
                kw = s["keyword"]
                if s["excluded"]:
                    badge = ('<span class="sug-badge block" title="also an excluded (NOT) term in this SDG — '
                             'adding it can block a match">⚠ blocked</span>')
                elif not matched and kw in solo:
                    badge = '<span class="sug-badge solo" title="this keyword alone qualifies the SDG">✓ alone</span>'
                elif not matched:
                    n = extra.get(kw)
                    if n is not None:
                        badge = (f'<span class="sug-badge more" title="still needs {n} more keyword(s) — '
                                 f'see the near-miss boxes above">+{n} more</span>')
                    else:
                        badge = ('<span class="sug-badge more" title="does not qualify by itself — '
                                 'see the near-miss boxes above">+ more</span>')
                else:
                    badge = ""
                body.append(
                    f'<button type="button" class="kw sug" data-kw="{escape(kw)}" '
                    f'title="add to Keywords field &amp; copy">{escape(kw)}'
                    f'<span class="score">{s["score"]}%</span>{badge}</button>'
                )
            body.append('</div>')
            body.append(f'<div class="muted-text">Auto-ranked by word overlap with your text — no AI. '
                        f'Open the <b>Advanced</b> tab for the full SDG {r["sdg"]} keyword list.</div>')
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

    explainer = ('<details class="card explainer"><summary>How SDG matching works (30 seconds)</summary>'
                 '<p>Each SDG is made of several <b>keyword paths</b>. A paper qualifies for an SDG as soon as '
                 '<b>one full path</b> is present in its text. For every SDG you are close to, we show the '
                 '<b>shortest missing path</b> first: pick <b>one keyword from each box</b> and the SDG qualifies. '
                 'Click any suggested keyword to add it to your Keywords field (it is copied to the clipboard too).</p></details>')
    return (f'<div id="results-inner"><h2 class="section">Results</h2>{info_html}{stat}{chips_html}{explainer}'
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
            extra = None
            set_cookie = _track_page_user(self)  # anonymous uid for unique users
            if set_cookie:
                extra = {"Set-Cookie": set_cookie}
            self._send(200, page, extra_headers=extra)
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
        elif url.path == "/api/stats":
            self._send_json(200, stats_payload())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            fields, files = self._read_form()
            # POST /api/keywords — full keyword list of one SDG scored against
            # the paper (Advanced tab; deterministic, no LLM).
            if path == "/api/keywords":
                sdg = fields.get("sdg", "10").strip()
                blocks = None
                for s, bl in get_queries():
                    if s == sdg:
                        blocks = bl
                        break
                if blocks is None:
                    self._send_json(400, {"error": f"unknown sdg {sdg}"})
                    return
                paper, _meta = self._build_paper(fields, files)
                if paper is None:
                    self._send_json(400, {"error": "No paper entered"})
                    return
                present: set[str] = set()
                for b in blocks:
                    hits, _m, _e = scan_with_fields(b, paper)
                    present.update(kw for kw, _f in hits)
                inc, exc = sdg_dict(sdg, blocks)
                limit = min(max(int(fields.get("limit", "300") or 300), 1), 2000)
                scored = score_keywords(paper.lowered(""), inc, exc, present, limit)
                self._send_json(200, {
                    "sdg": sdg,
                    "sdg_name": sdg_name(sdg),
                    "total": len(inc),
                    "present": len([k for k in scored if k["present"]]),
                    "limit": limit,
                    "keywords": scored,
                })
                return
            if path != "/match":
                self._send(404, b"not found", "text/plain")
                return
            paper, meta = self._build_paper(fields, files)
            if paper is None:
                self._send(200, error_box("No paper entered — fill in the form (Title / Abstract / Keywords), paste raw text, or upload a file.").encode())
                return
            top = min(max(int(fields.get("top", "3") or 3), 1), 30)
            max_kw = min(max(int(fields.get("maxkw", "10") or 10), 1), 50)
            t0 = time.perf_counter()
            report = match_paper(paper, top, max_kw)
            ms = (time.perf_counter() - t0) * 1000.0
            html = render_results(report, paper, meta, ms)
            self._send(200, html.encode(),
                       extra_headers={"X-Processing-Time": f"{ms:.1f} ms"})
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            sentry_report("error", "web.do_POST", f"Matching failed: {e}", exc=e,
                          extra={"traceback": traceback.format_exc()})
            self._send(200, error_box(f"Matching failed: {e}").encode())

    def _build_paper(self, fields, files):
        """(paper, meta) or (None, None) when no paper was entered."""
        form = {k: fields.get(k, "").strip() for k in
                ("title", "abstract", "keywords", "authors", "year", "journal", "doi")}
        if any(form.values()):
            return paper_from_fields(form)
        text = fields.get("paper", "")
        if not text:
            for fname in ("file", "paper"):  # accept uploads under either name
                if fname in files:
                    text = files[fname].decode("utf-8", "replace")
                    break
        if not text.strip():
            return None, None
        return parse_paper_text(text)

    def handle_one_request(self):
        """Start a per-request timer so log_message can record latency."""
        self._t0 = time.perf_counter()
        super().handle_one_request()

    def log_message(self, fmt, *args):
        msg = fmt % args
        sys.stderr.write("[web] %s\n" % msg)
        # BaseHTTPRequestHandler calls this as '"%s" %s %s' with
        # (requestline, code, size) after each response.
        try:
            parts = msg.split('"')
            rl = parts[1] if len(parts) > 1 else ""
            bits = rl.split()
            method = bits[0] if bits else "-"
            target = bits[1] if len(bits) > 1 else "-"
            status = next((int(tok) for tok in parts[-1].split() if tok.isdigit()), 0)
            ms = (time.perf_counter() - getattr(self, "_t0", time.perf_counter())) * 1000.0
            _record_request(method, target, status, ms)
        except Exception:  # noqa: BLE001 — logging must never break a request
            pass


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
    _load_stats()
    _load_visitors()
    total = _STATS["total"]
    pages = _STATS["pages"]
    matches = _STATS["match_html"] + _STATS["api_match"]
    pl = lambda n: "" if n == 1 else "s"  # noqa: E731 — tiny boot-log helper
    print(f"[web] usage so far: {total} request{pl(total)}, {pages} page view{pl(pages)}, "
          f"{matches} match{'es' if matches != 1 else ''} "
          "(cumulative; log: logs/access.jsonl)", file=sys.stderr)
    print(f"[web] {len(_VISITORS)} unique user{pl(len(_VISITORS))} on file", file=sys.stderr)
    if _SENTRY_DSN:
        _install_thread_excepthook()
    try:
        n = len(get_queries())
        print(f"[web] loaded {n} SDG query sets", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[web] warning: could not load queries: {e}", file=sys.stderr)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[web] SDG Paper Matcher running at {url}  (Ctrl-C to stop)")
    if _SENTRY_DSN:
        sentry_report("info", "web.boot", "SDG Paper Matcher started",
                      extra={"queries": n, "addr": f"{args.host}:{args.port}",
                             "users": len(_VISITORS)})
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
