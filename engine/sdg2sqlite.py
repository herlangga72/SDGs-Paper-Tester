#!/usr/bin/env python3
# ---
# title: sdg2sqlite
# description: Export the parsed Scopus SDG query ASTs into a SQLite database
# purpose: Materialize the SDG query trees so match_paper.py (and any other
#          tool) can load them without re-parsing the SDG*.txt files
# version: 1.0.0
# created: 2025-08-05
# project: SDG paper-matching tools
# language: python (standard library only: sqlite3)
# usage: python3 sdg2sqlite.py [DIR] [-o out.sqlite3] [--quiet]
# input:  folder with SDG*.txt query files (same input as parse_sdg.py)
# output: SQLite database (default sdg_queries.sqlite3)
# schema:
#   sdg   (sdg_no, file, n_blocks)          one row per SDG query file
#   block (id, sdg_no, block_no)            top-level OR branches
#   node  (id, block_id, parent_id, kind, op, fields, keyword, exact, ord)
#         AST: kind = leaf | group | not | field; parent_id NULL = block root
#   flat  (VIEW)                            the parse_sdg.py keyword table,
#         columns: sdgs_no, include_or_exclude, where_to_look, keyword,
#                  block_no, logic, exact   (identical to the CSV output)
# related: [parse_sdg.py, match_paper.py]
# ---
"""
sdg2sqlite.py — write the SDG query ASTs into a SQLite database.

Parses every SDG*.txt in a folder with the same parser as parse_sdg.py
(which now tolerates the malformed constructs found in the real Elsevier
files: dangling operators, unquoted multi-word phrases, `--Avoids`,
`articles.`, ...) and stores the full AST — NOT/group/field structure
preserved — in a normalized schema. match_paper.py can then load queries
from this database instead of re-tokenizing the text files.

The `flat` VIEW reproduces the parse_sdg.py keyword-table CSV exactly:

    SELECT * FROM flat;

Usage:
    python3 sdg2sqlite.py [DIR] [-o out.sqlite3] [--quiet]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from engine.parse_sdg import FieldWrap, Group, Leaf, Not, ParseError, Parser, tokenize

SCHEMA = """
CREATE TABLE sdg (
    sdg_no   TEXT PRIMARY KEY,          -- digits from the file name, e.g. "01"
    file     TEXT NOT NULL,             -- source file name
    n_blocks INTEGER NOT NULL           -- top-level OR branches
);
CREATE TABLE block (
    id       INTEGER PRIMARY KEY,
    sdg_no   TEXT NOT NULL REFERENCES sdg(sdg_no),
    block_no INTEGER NOT NULL,
    UNIQUE (sdg_no, block_no)
);
CREATE TABLE node (
    id        INTEGER PRIMARY KEY,
    block_id  INTEGER NOT NULL REFERENCES block(id),
    parent_id INTEGER REFERENCES node(id),   -- NULL = block root
    kind      TEXT NOT NULL CHECK (kind IN ('leaf', 'group', 'not', 'field')),
    op        TEXT,                          -- group: OR/AND/W/n/PRE/n...
    fields    TEXT,                          -- field: "TITLE,ABS,KEY"
    keyword   TEXT,                          -- leaf
    exact     INTEGER,                       -- leaf: 1 = {...} exact term
    ord       INTEGER NOT NULL               -- child order within the parent
);
CREATE INDEX idx_node_block  ON node(block_id);
CREATE INDEX idx_node_parent ON node(parent_id);

-- The parse_sdg.py keyword table, as a view:
-- (same columns and semantics as the CSV: logic starts at "OR" per block
--  and includes every group's op from the block root down, field wraps
--  replace where_to_look, NOT toggles include/exclude)
CREATE VIEW flat AS
WITH RECURSIVE walk(id, sdg_no, block_no, excluded, fields, logic) AS (
    SELECT n.id, b.sdg_no, b.block_no, 0,
           CASE WHEN n.kind = 'field' THEN n.fields ELSE '' END,
           CASE WHEN n.kind = 'group' THEN 'OR>' || n.op ELSE 'OR' END
    FROM node n JOIN block b ON b.id = n.block_id
    WHERE n.parent_id IS NULL
    UNION ALL
    SELECT n.id, w.sdg_no, w.block_no,
           CASE WHEN n.kind = 'not'   THEN 1 - w.excluded ELSE w.excluded END,
           CASE WHEN n.kind = 'field' THEN n.fields      ELSE w.fields  END,
           CASE WHEN n.kind = 'group' THEN w.logic || '>' || n.op
                                      ELSE w.logic END
    FROM walk w JOIN node n ON n.parent_id = w.id
)
SELECT w.sdg_no                                     AS sdgs_no,
       CASE WHEN w.excluded THEN 'exclude' ELSE 'include' END
                                                   AS include_or_exclude,
       REPLACE(w.fields, ',', '-')                 AS where_to_look,
       n.keyword,
       w.block_no,
       w.logic,
       n.exact
FROM walk w JOIN node n ON n.id = w.id
WHERE n.kind = 'leaf'
ORDER BY w.sdg_no, w.block_no, n.id;
"""


def insert_node(conn: sqlite3.Connection, node, block_id: int, parent_id: int | None, ord_: int = 0) -> int:
    """Insert one AST node (pre-order, so parents precede children)."""
    if isinstance(node, Leaf):
        cur = conn.execute(
            "INSERT INTO node (block_id, parent_id, kind, keyword, exact, ord)"
            " VALUES (?, ?, 'leaf', ?, ?, ?)",
            (block_id, parent_id, node.keyword, 1 if node.exact else 0, ord_),
        )
        return cur.lastrowid
    if isinstance(node, Not):
        cur = conn.execute(
            "INSERT INTO node (block_id, parent_id, kind, ord) VALUES (?, ?, 'not', ?)",
            (block_id, parent_id, ord_),
        )
        insert_node(conn, node.child, block_id, cur.lastrowid)
        return cur.lastrowid
    if isinstance(node, FieldWrap):
        cur = conn.execute(
            "INSERT INTO node (block_id, parent_id, kind, fields, ord)"
            " VALUES (?, ?, 'field', ?, ?)",
            (block_id, parent_id, ",".join(node.fields), ord_),
        )
        insert_node(conn, node.child, block_id, cur.lastrowid)
        return cur.lastrowid
    if isinstance(node, Group):
        cur = conn.execute(
            "INSERT INTO node (block_id, parent_id, kind, op, ord)"
            " VALUES (?, ?, 'group', ?, ?)",
            (block_id, parent_id, node.op, ord_),
        )
        nid = cur.lastrowid
        for i, ch in enumerate(node.children):
            insert_node(conn, ch, block_id, nid, i)
        return nid
    raise TypeError(f"unknown node {node!r}")


def build_db(dirpath: Path, db_path: Path) -> dict[str, int]:
    """Parse all SDG*.txt files in dirpath and store their ASTs. Returns
    {sdg_no: leaf_count}."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        per_sdg: dict[str, int] = {}
        for f in sorted(dirpath.glob("SDG*.txt")):
            import re
            m = re.search(r"SDG(\d+)", f.name, re.I)
            if not m:
                continue
            sdg = m.group(1)
            root = Parser(tokenize(f.read_text(encoding="utf-8-sig"))).parse()
            blocks = root.children if isinstance(root, Group) and root.op == "OR" else [root]
            conn.execute("INSERT INTO sdg (sdg_no, file, n_blocks) VALUES (?, ?, ?)",
                         (sdg, f.name, len(blocks)))
            n_leaves = 0
            for bno, block in enumerate(blocks):
                cur = conn.execute("INSERT INTO block (sdg_no, block_no) VALUES (?, ?)",
                                   (sdg, bno))
                insert_node(conn, block, cur.lastrowid, None)
                n_leaves += count_leaves(block)
            per_sdg[sdg] = n_leaves
        conn.commit()
        return per_sdg
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def count_leaves(node) -> int:
    if isinstance(node, Leaf):
        return 1
    if isinstance(node, Not):
        return count_leaves(node.child)
    if isinstance(node, FieldWrap):
        return count_leaves(node.child)
    if isinstance(node, Group):
        return sum(count_leaves(c) for c in node.children)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Export SDG query ASTs to SQLite.")
    ap.add_argument("dir", nargs="?", default=str(Path(__file__).resolve().parent / "data" / "queries"),
                    help="folder with SDG*.txt files")
    ap.add_argument("-o", "--output", default=str(Path(__file__).resolve().parent / "data" / "sdg_queries.sqlite3"),
                    help="output database path (default: sdg_queries.sqlite3)")
    ap.add_argument("--quiet", action="store_true", help="suppress the summary report")
    args = ap.parse_args()

    p = Path(args.dir)
    if not p.is_dir():
        print(f"{p} is not a directory", file=sys.stderr)
        return 1
    try:
        per_sdg = build_db(p, Path(args.output))
    except ParseError as e:
        print(f"parse error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"wrote SDG query ASTs -> {args.output}")
        for sdg in sorted(per_sdg, key=lambda s: int(s)):
            print(f"  SDG {sdg:>2}: {per_sdg[sdg]:>6} keywords")
        print(f"  total: {sum(per_sdg.values())} keywords")
        print(f"  hint: SELECT * FROM flat;  reproduces the parse_sdg.py CSV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
