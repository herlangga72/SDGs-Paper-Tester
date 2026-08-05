#!/usr/bin/env python3
# ---
# title: parse_sdg
# description: Parse Scopus SDG search-query files (SDG*.txt) into a flat keyword table
# purpose: Machine-readable index of every keyword Scopus uses to classify papers into SDGs
# version: 1.0.0
# created: 2025-08-05
# project: SDG paper-matching tools
# language: python (standard library only, no dependencies)
# usage: python3 parse_sdg.py [DIR_OR_FILE] [-o out.csv] [--dedup]
# input: folder with SDG*.txt files (Scopus TITLE-ABS-KEY boolean syntax) or a single query file
# output: CSV with one row per keyword (default sdg_keywords.csv)
# columns:
#   sdgs_no            : SDG number from file name, e.g. 17
#   include_or_exclude : include (term must be present) | exclude (term under a NOT)
#   where_to_look      : Scopus field prefix, e.g. TITLE-ABS-KEY, TITLE, AUTHKEY
#   keyword            : search term, wildcards kept, quotes/braces stripped
#   block_no           : index of the top-level OR block the keyword belongs to
#   logic              : operator chain from root to keyword, e.g. OR>AND>OR
#   exact              : 1 if written as {term} (exact match), else 0
# related: [match_paper.py, SDG17.txt]
# ---
"""
parse_sdg.py — Parse Scopus SDG search-query files (SDG*.txt) into a flat table.

For every keyword in a query file we emit one row:

    sdgs_no            : SDG number taken from the file name (e.g. "17")
    include_or_exclude : "include" (term must be present) | "exclude" (term under a NOT)
    where_to_look      : Scopus field prefix, e.g. TITLE-ABS-KEY, TITLE, AUTHKEY
    keyword            : the search term, wildcards kept, quotes/braces stripped
    block_no           : index of the top-level OR block the keyword belongs to
    logic              : operator chain from the root to the keyword (e.g. OR>AND)
    exact              : 1 if written as {...} (exact-match term), else 0

Why: once every SDG query is a machine-readable keyword table, we can build
tools that (a) tell which SDG keywords a paper's title/abstract/keywords hit,
(b) suggest keywords to add so a paper qualifies for a target SDG, and
(c) audit papers that *should* be in an SDG but are missing.

Usage:
    python parse_sdg.py [DIR_OR_FILE] [-o out.csv] [--dedup]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# The real Elsevier query trees are up to ~600 levels deep (SDG07 contains a
# huge consecutive-AND chain), which exceeds Python's default 1000-frame
# recursion limit when walk()/eval() recurse over the AST. This module is
# imported by every tool in the project, so setting it here covers all of
# them (match_paper.py, sdg2sqlite.py).
sys.setrecursionlimit(10000)

# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------

@dataclass
class Leaf:
    keyword: str
    exact: bool = False            # {...} exact term instead of "..."


@dataclass
class FieldWrap:
    fields: tuple[str, ...]        # e.g. ("TITLE", "ABS", "KEY")
    child: object


@dataclass
class Group:
    op: str                        # "OR" | "AND" | "W/n" | "PRE/n" ...
    children: list


@dataclass
class Not:
    child: object


@dataclass
class Row:
    sdgs_no: str
    include_or_exclude: str
    where_to_look: str
    keyword: str
    block_no: int
    logic: str
    exact: int


class ParseError(Exception):
    pass

# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

RE_OP = re.compile(r"(?i)\b(OR|AND|NOT)\b")
RE_PROX = re.compile(r"(?i)\b((?:W|PRE|POST|NEAR|ONEAR))/(\d+)")
RE_ID = re.compile(r"-*[A-Za-z0-9][A-Za-z0-9-]*")  # leading dashes allowed (--Avoids in SDG07)


def _scan_quoted(text: str, i: int) -> tuple[str, int]:
    """text[i] == '"' -> (content, index after closing quote)."""
    out = []
    j = i + 1
    n = len(text)
    while j < n:
        c = text[j]
        if c == "\\" and j + 1 < n:          # escaped char
            out.append(text[j + 1])
            j += 2
        elif c == '"':
            return "".join(out), j + 1
        else:
            out.append(c)
            j += 1
    raise ParseError(f"unterminated quoted string at position {i}")


def _scan_braced(text: str, i: int) -> tuple[str, int]:
    """text[i] == '{' -> (content, index after closing brace); handles nesting."""
    depth = 0
    j = i
    n = len(text)
    while j < n:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    raise ParseError(f"unterminated {{...}} term at position {i}")


def tokenize(text: str) -> list[tuple[str, str | None]]:
    toks: list[tuple[str, str | None]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "(":
            toks.append(("LPAREN", None))
            i += 1
        elif c == ")":
            toks.append(("RPAREN", None))
            i += 1
        elif c == '"':
            kw, i = _scan_quoted(text, i)
            toks.append(("STR", kw))
        elif c == "{":
            kw, i = _scan_braced(text, i)
            toks.append(("BRACE", kw))
        else:
            m = RE_OP.match(text, i)
            if m:
                toks.append(("OP", m.group(1).upper()))
                i = m.end()
                continue
            m = RE_PROX.match(text, i)
            if m:
                toks.append(("PROX", f"{m.group(1).upper()}/{m.group(2)}"))
                i = m.end()
                continue
            m = RE_ID.match(text, i)
            if m:
                name = m.group(0)
                j = m.end()
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] == "(" and name.upper() not in {"OR", "AND", "NOT"}:
                    toks.append(("FIELD", name.upper()))
                    i = j  # leave '(' for the next iteration
                else:
                    # bare (unquoted) keyword — Scopus allows e.g. TITLE-ABS(H2)
                    # and wildcards attached to it: cereal*, girl* (a trailing
                    # '.' also sticks to the word, e.g. `articles.` in SDG07)
                    k = m.end()
                    while k < n and text[k] in "*?.":
                        k += 1
                    toks.append(("STR", name + text[m.end():k]))
                    i = k
                continue
            m = re.match(r"[*?]+", text[i:])
            if m:
                # leading wildcards: *divers*, ?term*
                j = i + m.end()
                m2 = re.match(r"[A-Za-z0-9][A-Za-z0-9-]*", text[j:])
                if m2:
                    k = j + m2.end()
                    while k < n and text[k] in "*?":
                        k += 1
                    toks.append(("STR", text[i:k]))
                    i = k
                    continue
                raise ParseError(f"unexpected wildcard at position {i}")
            raise ParseError(f"unexpected character {c!r} at position {i}")
    return toks

# --------------------------------------------------------------------------
# Recursive-descent parser  (precedence: NOT > AND/W-n > OR)
# --------------------------------------------------------------------------

class Parser:
    def __init__(self, toks: list[tuple[str, str | None]]):
        self.toks = toks
        self.pos = 0

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def next(self):
        t = self.peek()
        if t is None:
            raise ParseError("unexpected end of query")
        self.pos += 1
        return t

    def accept(self, kind: str, val: str | None = None):
        t = self.peek()
        if t and t[0] == kind and (val is None or t[1] == val):
            self.pos += 1
            return t
        return None

    def expect(self, kind: str):
        t = self.next()
        if t[0] != kind:
            raise ParseError(f"expected {kind}, got {t!r}")
        return t

    def parse(self):
        root = self.parse_or()
        if self.peek():
            raise ParseError(f"unexpected trailing token {self.peek()!r}")
        return root

    def at_end(self):
        """True at a closing paren or the end of the token stream — used to
        tolerate dangling operators in real-world Elsevier data, e.g.
        `TITLE-ABS-KEY(mitigat*OR))` (keyword glued to a stray `OR`)."""
        t = self.peek()
        return t is None or t[0] == "RPAREN"

    def parse_or(self):
        parts = [self.parse_and()]
        while self.accept("OP", "OR"):
            if self.at_end():
                break  # dangling OR before ')' / EOF: drop the operator
            parts.append(self.parse_and())
        return parts[0] if len(parts) == 1 else Group("OR", parts)

    def parse_and(self):
        terms = [self.parse_not()]
        ops: list[str] = []
        while True:
            t = self.peek()
            if t and t[0] == "OP" and t[1] == "AND":
                self.pos += 1
                if self.at_end():
                    break  # dangling AND before ')' / EOF
                ops.append("AND")
                terms.append(self.parse_not())
            elif t and t[0] == "PROX":
                self.pos += 1
                if self.at_end():
                    break  # dangling W/n before ')' / EOF
                ops.append(t[1])
                terms.append(self.parse_not())
            else:
                break
        # Right-nested: A AND B W/2 C  ->  AND(A, W/2(B, C))
        node = terms[-1]
        for op, term in reversed(list(zip(ops, terms[:-1]))):
            node = Group(op, [term, node]) if op != "AND" else Group("AND", [term, node])
        return node

    def parse_not(self):
        if self.accept("OP", "NOT"):
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self):
        t = self.peek()
        if t is None:
            raise ParseError("unexpected end of query")
        if t[0] == "LPAREN":
            self.pos += 1
            inner = self.parse_or()
            self.expect("RPAREN")
            return inner
        if t[0] == "FIELD":
            self.pos += 1
            self.expect("LPAREN")
            inner = self.parse_or()
            self.expect("RPAREN")
            return FieldWrap(tuple(t[1].split("-")), inner)
        if t[0] == "STR":
            self.pos += 1
            # Merge adjacent plain (unquoted) terms into a phrase. Real
            # Elsevier data contains unquoted multi-word terms such as
            # `TITLE-ABS-KEY(pes scheme*)` (SDG02), `TITLE-ABS(ethylene
            # terephthalate)` (SDG12) and `TITLE-ABS(neogobius
            # melanostomus)` (SDG15); braced terms keep their boundaries.
            parts = [t[1]]
            while True:
                nxt = self.peek()
                if nxt and nxt[0] == "STR":
                    self.pos += 1
                    parts.append(nxt[1])
                else:
                    break
            return Leaf(" ".join(parts), exact=False)
        if t[0] == "BRACE":
            self.pos += 1
            return Leaf(t[1], exact=True)
        raise ParseError(f"unexpected token {t!r}")

# --------------------------------------------------------------------------
# Tree walk -> flat rows
# --------------------------------------------------------------------------

def walk(node, excluded: bool, fields: tuple[str, ...],
         sdgs_no: str, block_no: int, logic: tuple[str, ...]):
    if isinstance(node, FieldWrap):
        yield from walk(node.child, excluded, node.fields, sdgs_no, block_no, logic)
    elif isinstance(node, Not):
        yield from walk(node.child, not excluded, fields, sdgs_no, block_no, logic)
    elif isinstance(node, Group):
        nl = logic + (node.op,)
        for ch in node.children:
            yield from walk(ch, excluded, fields, sdgs_no, block_no, nl)
    elif isinstance(node, Leaf):
        yield Row(
            sdgs_no=sdgs_no,
            include_or_exclude="exclude" if excluded else "include",
            where_to_look="-".join(fields) if fields else "",
            keyword=node.keyword,
            block_no=block_no,
            logic=">".join(logic),
            exact=1 if node.exact else 0,
        )
    else:
        raise TypeError(f"unknown node {node!r}")


def parse_file(path: Path) -> tuple[str, list[Row]] | None:
    m = re.search(r"SDG(\d+)", path.name, re.I)
    if not m:
        return None
    text = path.read_text(encoding="utf-8-sig")
    root = Parser(tokenize(text)).parse()
    blocks = root.children if isinstance(root, Group) and root.op == "OR" else [root]
    rows: list[Row] = []
    for bno, block in enumerate(blocks):
        rows.extend(walk(block, False, (), m.group(1), bno, ("OR",)))
    return m.group(1), rows

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

CSV_COLUMNS = ["sdgs_no", "include_or_exclude", "where_to_look", "keyword",
               "block_no", "logic", "exact"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse Scopus SDG query files into a keyword table.")
    ap.add_argument("path", nargs="?", default=".",
                    help="folder containing SDG*.txt files, or a single query file")
    ap.add_argument("-o", "--output", default="sdg_keywords.csv",
                    help="output CSV path (default: sdg_keywords.csv)")
    ap.add_argument("--dedup", action="store_true",
                    help="collapse identical (sdgs_no, include_or_exclude, where_to_look, keyword) rows")
    ap.add_argument("--quiet", action="store_true", help="suppress the summary report")
    args = ap.parse_args()

    p = Path(args.path)
    files = sorted(p.glob("SDG*.txt")) if p.is_dir() else [p]
    if not files:
        print(f"no SDG*.txt files found in {p}", file=sys.stderr)
        return 1

    all_rows: list[Row] = []
    per_file: dict[str, list[Row]] = {}
    for f in files:
        res = parse_file(f)
        if res is None:
            print(f"skipping {f.name}: file name does not match SDG<number>.txt", file=sys.stderr)
            continue
        sdg, rows = res
        per_file[sdg] = rows
        all_rows.extend(rows)

    if args.dedup:
        seen = set()
        kept = []
        for r in all_rows:
            key = (r.sdgs_no, r.include_or_exclude, r.where_to_look, r.keyword)
            if key not in seen:
                seen.add(key)
                kept.append(r)
        all_rows = kept

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in all_rows:
            w.writerow([r.sdgs_no, r.include_or_exclude, r.where_to_look,
                        r.keyword, r.block_no, r.logic, r.exact])

    if not args.quiet:
        print(f"wrote {len(all_rows)} rows -> {args.output}")
        for sdg in sorted(per_file, key=lambda s: int(s)):
            rows = per_file[sdg]
            n_inc = sum(1 for r in rows if r.include_or_exclude == "include")
            n_exc = len(rows) - n_inc
            uniq = len({(r.include_or_exclude, r.where_to_look, r.keyword) for r in rows})
            print(f"  SDG {sdg:>2}: {len(rows):>6} rows "
                  f"({n_inc} include, {n_exc} exclude, {uniq} unique keyword/field combos)")
        print(f"  total: {len(all_rows)} rows")
        print(f"  where_to_look values: {sorted({r.where_to_look for r in all_rows})}")
        excl = sorted({r.keyword for r in all_rows if r.include_or_exclude == "exclude"})
        print(f"  excluded keywords ({len(excl)}): {excl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
