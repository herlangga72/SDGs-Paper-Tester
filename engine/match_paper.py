#!/usr/bin/env python3
# ---
# title: match_paper
# description: Check a paper's title/abstract/keywords against the Scopus SDG query AST
# purpose: Tell which SDG query blocks a paper satisfies, which blocks it almost satisfies
#          (with the exact missing keywords), and which excluded (NOT) terms appear in the text
# version: 1.0.0
# created: 2025-08-05
# project: SDG paper-matching tools
# language: python (standard library only, no dependencies)
# usage: python3 match_paper.py paper.txt [--dir .] [--db sdg_queries.sqlite3] [--top 3] [--max-kw 10]
#        cat paper.txt | python3 match_paper.py - --top 5
#
# input:
#   paper : markdown/text file with YAML frontmatter (--- delimited)
#           title, abstract, keywords, author_keywords keys supported;
#           keywords may be a list (- item / [a, b]) or text; body text is
#           used as fallback for fields not given; '-' = stdin
#   queries : the SDG query ASTs come from the SQLite database built by
#           sdg2sqlite.py (auto-detected as <dir>/sdg_queries.sqlite3, or
#           given with --db); if no database is found the SDG*.txt files in
#           --dir are parsed directly as a fallback
# output: per SDG -> matched blocks, near-miss blocks with missing keywords,
#         excluded terms found in the text
# semantics: AST walk with NOT > AND/W-n > OR precedence;
#            W/n requires term presence (proximity distance ignored, conservative check)
# related: [parse_sdg.py, SDG17.txt]
# ---
"""
match_paper.py — check a paper's text against the Scopus SDG query files.

The parser (parse_sdg.py) turns each SDG*.txt query into an AST. This tool
walks that AST the same way Scopus would evaluate the boolean query, but
against *your* paper's text instead of the whole literature database.

Input (paper text):
    The paper file is a markdown/text file that may start with YAML frontmatter:

        ---
        title: "..."
        authors: ["A", "B"]
        year: 2025
        abstract: |
          ...
        keywords: [tax evasion, foreign aid, ...]
        ---

    Supported frontmatter keys: title, abstract (or summary), keywords
    (or keyword, author_keywords). keywords may be a YAML list or plain text.
    The body of the file is used as fallback text for fields not given.
    Plain files without frontmatter still work (TITLE:/ABSTRACT:/KEYWORDS:
    line markers or raw text). Pass "-" to read from stdin.

Output, per SDG present in the folder:
    - MATCHED:     top-level query blocks the paper satisfies
    - NEAR MISSES: blocks that fail by only a few missing keywords — with the
                   exact keywords to add (and the field Scopus looks at) —
                   this is the "paper should be included but isn't" case
    - EXCLUDED:    NOT-keywords found in the text that can disqualify a block

Usage:
    python3 match_paper.py paper.md [--dir .] [--top 3] [--max-kw 10]
    cat paper.md | python3 match_paper.py - --top 5
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from engine.parse_sdg import (FieldWrap, Group, Leaf, Not, ParseError,
                           Parser, tokenize, walk)

DEFAULT_DB = "sdg_queries.sqlite3"  # built by sdg2sqlite.py

# --------------------------------------------------------------------------
# Matching primitives
# --------------------------------------------------------------------------

_PAT_CACHE: dict[str, re.Pattern] = {}


def term_pattern(term: str) -> re.Pattern:
    """'*'/'?' wildcards -> regex; plain terms are matched as whole words
    (Scopus phrase semantics)."""
    pat = _PAT_CACHE.get(term)
    if pat is not None:
        return pat
    out = []
    for piece in re.split(r"([*?])", term):
        if piece == "*":
            out.append(".*")
        elif piece == "?":
            out.append(".")
        else:
            out.append(re.escape(piece))
    rx = rf"\b{''.join(out)}\b" if "*" not in term and "?" not in term else "".join(out)
    # DOTALL: the paper fields join YAML block-scalar lines with '\n';
    # a phrase like "foreign* trad*" must match across line breaks
    # (Scopus treats all whitespace alike).
    pat = re.compile(rx, re.IGNORECASE | re.DOTALL)
    _PAT_CACHE[term] = pat
    return pat


@dataclass
class Paper:
    """Field texts. Missing fields fall back to the full text."""
    sections: dict[str, str]
    full_text: str

    def text_for(self, field: str) -> str:
        if field in self.sections and self.sections[field].strip():
            return self.sections[field]
        return self.full_text


def term_hit(term: str, fields: tuple[str, ...], paper: Paper) -> bool:
    pat = term_pattern(term)
    field_set = set(fields) if fields else {"TITLE", "ABS", "KEY", "AUTHKEY"}
    for f in field_set:
        if pat.search(paper.text_for(f)):
            return True
    return False


# --------------------------------------------------------------------------
# AST evaluation (same semantics as Scopus: NOT > AND/W-n > OR)
# --------------------------------------------------------------------------

def eval_node(node, fields: tuple[str, ...], paper: Paper) -> bool:
    if isinstance(node, Leaf):
        return term_hit(node.keyword, fields, paper)
    if isinstance(node, FieldWrap):
        return eval_node(node.child, node.fields, paper)
    if isinstance(node, Not):
        return not eval_node(node.child, fields, paper)
    if isinstance(node, Group):
        if node.op == "OR":
            return any(eval_node(c, fields, paper) for c in node.children)
        # AND, W/n, PRE/n, ... : every child must hit (proximity distance is
        # ignored here — presence is required, which is the conservative check)
        return all(eval_node(c, fields, paper) for c in node.children)
    raise TypeError(f"unknown node {node!r}")


def scan_block(block, paper: Paper) -> tuple[list[str], list[str], list[str]]:
    """(include terms hit, include terms missed, excluded terms hit)."""
    hits: list[str] = []
    misses: list[str] = []
    excluded_hits: list[str] = []

    def rec(node, fields, excluded):
        if isinstance(node, Leaf):
            found = term_hit(node.keyword, fields, paper)
            if excluded:
                if found:
                    excluded_hits.append(node.keyword)
            elif found:
                hits.append(node.keyword)
            else:
                misses.append(node.keyword)
        elif isinstance(node, FieldWrap):
            rec(node.child, node.fields, excluded)
        elif isinstance(node, Not):
            rec(node.child, fields, not excluded)
        elif isinstance(node, Group):
            for c in node.children:
                rec(c, fields, excluded)

    rec(block, (), False)
    return hits, misses, excluded_hits

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _parse_scalar(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
        return v[1:-1]
    return v


def parse_simple_yaml(block: str) -> dict:
    """Minimal YAML-subset parser (no dependencies): key: value, [a, b] lists,
    '- item' block lists, and '|' block scalars."""
    meta: dict = {}
    lines = block.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key = m.group(1).lower()
        val = m.group(2).strip()
        i += 1
        if not val or val in ("|", ">", "|-", ">-"):
            j = i
            while j < n and not lines[j].strip():
                j += 1
            if j < n and re.match(r"^\s+-\s+", lines[j]):      # block list
                items = []
                while j < n:
                    lm = re.match(r"^\s+-\s+(.*)$", lines[j])
                    if not lm:
                        break
                    items.append(_parse_scalar(lm.group(1)))
                    j += 1
                meta[key] = items
                i = j
            elif j < n and (lines[j].startswith(" ") or lines[j].startswith("\t")):
                bl = []                                          # block scalar
                while j < n and (lines[j].startswith(" ") or lines[j].startswith("\t")
                                 or not lines[j].strip()):
                    if lines[j].strip():
                        bl.append(lines[j].strip())
                    j += 1
                meta[key] = "\n".join(bl)
                i = j
            else:
                meta[key] = ""
        elif val.startswith("[") and val.endswith("]"):        # inline list
            meta[key] = [_parse_scalar(x) for x in val[1:-1].split(",") if x.strip()]
        else:
            meta[key] = _parse_scalar(val)
    return meta


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """(meta, body) — YAML frontmatter between --- lines, if present."""
    if not text.lstrip().startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    return parse_simple_yaml("\n".join(lines[1:end])), "\n".join(lines[end + 1:])


def parse_paper_text(text: str) -> tuple[Paper, dict]:
    meta, body = parse_frontmatter(text)

    # body marker sections (TITLE:/ABSTRACT:/KEYWORDS: lines, legacy format)
    sections: dict[str, str] = {}
    cur_field: str | None = None
    for ln in body.splitlines():
        m = re.match(r"^\s*(TITLE|ABSTRACT|KEYWORDS|AUTHKEY)\s*:\s*(.*)$", ln, re.I)
        if m:
            field = {"TITLE": "TITLE", "ABSTRACT": "ABS",
                     "KEYWORDS": "KEY", "AUTHKEY": "AUTHKEY"}[m.group(1).upper()]
            sections[field] = m.group(2)
            cur_field = field
        elif cur_field and ln.strip():
            sections[cur_field] += " " + ln.strip()

    # frontmatter wins over body markers
    if meta.get("title"):
        sections["TITLE"] = str(meta["title"])
    absv = meta.get("abstract") or meta.get("summary")
    if absv:
        sections["ABS"] = str(absv)
    kw = meta.get("keywords") or meta.get("keyword") or meta.get("author_keywords")
    if kw:
        kw_text = ", ".join(kw) if isinstance(kw, list) else str(kw)
        sections["KEY"] = kw_text
        sections["AUTHKEY"] = kw_text

    if not sections:
        sections = {"TITLE": text, "ABS": text, "KEY": text, "AUTHKEY": text}
    full_text = body.strip() if body.strip() else "\n".join(sections.values())
    return Paper(sections, full_text), meta


# --------------------------------------------------------------------------
# Query loading: SQLite database (preferred) or SDG*.txt files (fallback)
# --------------------------------------------------------------------------

def load_queries_from_db(db_path: str | Path) -> list[tuple[str, list]]:
    """Rebuild the SDG query ASTs from the sdg2sqlite.py database.

    Returns [(sdg_no, [block_root, ...]), ...] with the same node types
    (Leaf/Group/Not/FieldWrap) the text parser produces, so the evaluation
    code below is identical for both sources.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT b.sdg_no, b.block_no, n.id, n.parent_id, n.kind, n.op,
                   n.fields, n.keyword, n.exact, n.ord
            FROM block b JOIN node n ON n.block_id = b.id
            ORDER BY b.sdg_no, b.block_no, n.id
        """).fetchall()
    finally:
        conn.close()

    nodes: dict[int, object] = {}
    children: dict[int, list[tuple[int, int]]] = defaultdict(list)  # parent -> [(ord, id)]
    roots: dict[str, list[tuple[int, int]]] = defaultdict(list)     # sdg -> [(block_no, root_id)]
    for sdg_no, block_no, nid, parent_id, kind, op, fields, keyword, exact, ord_ in rows:
        if kind == "leaf":
            nodes[nid] = Leaf(keyword, bool(exact))
        elif kind == "not":
            nodes[nid] = Not(None)
        elif kind == "field":
            nodes[nid] = FieldWrap(tuple(fields.split(",")), None)
        else:  # group
            nodes[nid] = Group(op, [])
        if parent_id is None:
            roots[sdg_no].append((block_no, nid))
        else:
            children[parent_id].append((ord_, nid))

    for pid, kids in children.items():
        kids.sort()
        child_nodes = [nodes[k[1]] for k in kids]
        parent = nodes[pid]
        if isinstance(parent, (Not, FieldWrap)):
            parent.child = child_nodes[0]
        else:
            parent.children = child_nodes

    out = []
    for sdg_no in sorted(roots, key=lambda s: int(s)):
        blocks = [nodes[nid] for _, nid in sorted(roots[sdg_no])]
        out.append((sdg_no, blocks))
    return out


def load_queries_from_dir(dirpath: str) -> list[tuple[str, list]]:
    """Parse SDG*.txt query files (fallback when no database is available)."""
    out = []
    for f in sorted(Path(dirpath).glob("SDG*.txt")):
        m = re.search(r"SDG(\d+)", f.name, re.I)
        if not m:
            continue
        try:
            root = Parser(tokenize(f.read_text(encoding="utf-8-sig"))).parse()
        except ParseError as e:
            print(f"SDG {m.group(1)}: parse error ({e})", file=sys.stderr)
            continue
        blocks = root.children if isinstance(root, Group) and root.op == "OR" else [root]
        out.append((m.group(1), blocks))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def report(queries: list[tuple[str, list]], paper: Paper, args) -> int:
    """Match a paper against query blocks and print the report."""
    any_sdg = False
    for sdg, blocks in queries:
        matched = []
        near = []
        excluded_hits: list[str] = []
        for bno, block in enumerate(blocks):
            hits, misses, ex_hits = scan_block(block, paper)
            excluded_hits += ex_hits
            if eval_node(block, (), paper):
                matched.append((bno, hits))
            else:
                near.append((bno, misses, len(hits), len(misses)))

        near.sort(key=lambda t: len(t[1]))  # fewest missing keywords first
        any_sdg = True
        print(f"=== SDG {sdg} ===")
        if matched:
            for bno, hits in matched:
                shown = hits[: args.max_kw]
                more = len(hits) - len(shown)
                print(f"  MATCHED  block {bno}: {len(hits)} keyword(s) hit: {', '.join(shown)}{' ...' if more else ''}")
        else:
            print("  MATCHED  none")
        if near:
            print(f"  NEAR MISSES (missing keywords -> add any of these to qualify):")
            for bno, misses, n_hit, n_miss in near[: args.top]:
                shown = misses[: args.max_kw]
                more = n_miss - len(shown)
                print(f"    block {bno}: {n_hit} hit, missing {n_miss} of {n_miss + n_hit}: "
                      f"{', '.join(shown)}{' ...' if more else ''}")
        else:
            print("  NEAR MISSES none")
        if excluded_hits:
            uniq = sorted(set(excluded_hits))
            print(f"  EXCLUDED terms found in text (can disqualify a match): {', '.join(uniq)}")
        print()
    return 0 if any_sdg else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Check a paper against SDG query files.")
    ap.add_argument("paper", help="paper text file, or '-' for stdin")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / "data" / "queries"),
                    help="folder with SDG*.txt files")
    ap.add_argument("--db", default=None,
                    help=f"SQLite query database (default: {DEFAULT_DB} next to the SDG files; "
                         f"falls back to parsing SDG*.txt)")
    ap.add_argument("--top", type=int, default=3, help="near-miss blocks to show per SDG")
    ap.add_argument("--max-kw", type=int, default=10, help="max missing keywords to list")
    args = ap.parse_args()

    if args.paper == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.paper).read_text(encoding="utf-8")
    paper = parse_paper_text(text)[0]
    title = parse_frontmatter(text)[0].get("title")
    print(f"Paper: {title or Path(args.paper).name}")
    print()

    db_path = Path(args.db) if args.db else Path(args.dir) / DEFAULT_DB
    if args.db or db_path.exists():
        try:
            queries = load_queries_from_db(db_path)
        except sqlite3.Error as e:
            print(f"cannot read query database {db_path}: {e}", file=sys.stderr)
            return 1
        if not queries:
            print(f"no SDG queries in {db_path}", file=sys.stderr)
            return 1
    else:
        queries = load_queries_from_dir(args.dir)
        if not queries:
            print(f"no SDG*.txt files in {args.dir} and no {DEFAULT_DB} next to them",
                  file=sys.stderr)
            return 1

    return report(queries, paper, args)


if __name__ == "__main__":
    sys.exit(main())
