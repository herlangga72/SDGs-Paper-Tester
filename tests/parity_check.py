#!/usr/bin/env python3
"""Parity check: the reference Python engine and the Rust engine must agree
block-for-block on every test paper (the README promises identical matching
semantics). Exits 1 on any mismatch, 2 when the Rust binary is missing.

Usage:
    python3 tests/parity_check.py                 # all papers/*.md
    python3 tests/parity_check.py paper.md ...    # extra papers
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUST_BIN = ROOT / "rust" / "target" / "release" / "sdg_tools"
QDIR = ROOT / "engine" / "data" / "queries"


def matched_blocks(cmd: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Per SDG: [(block_no, n_hits)] for every matched block."""
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    res: dict[str, list[tuple[int, int]]] = {}
    cur: str | None = None
    for line in out.splitlines():
        m = re.match(r"=== SDG (\d+) ===", line)
        if m:
            cur = m.group(1)
            res[cur] = []
            continue
        if cur is None:
            continue
        m = re.match(r"\s+MATCHED\s+block (\d+): (\d+) keyword\(s\) hit", line)
        if m:
            res[cur].append((int(m.group(1)), int(m.group(2))))
    return res


def main() -> int:
    if not RUST_BIN.exists():
        print(f"rust binary not found at {RUST_BIN} (build it with: "
              "cd rust && cargo build --release); skipping the Rust side")
        return 2
    papers = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "papers").glob("*.md"))
    py_cmd = [sys.executable, str(ROOT / "engine" / "match_paper.py")]
    rs_cmd = [str(RUST_BIN), "match", "--dir", str(QDIR)]
    bad = 0
    for p in papers:
        py = matched_blocks([*py_cmd, str(p)])
        rs = matched_blocks([*rs_cmd, str(p)])
        diffs = [sdg for sdg in set(py) | set(rs) if py.get(sdg, []) != rs.get(sdg, [])]
        if diffs:
            bad += 1
            print(f"{p.name}: MISMATCH on SDG {', '.join(diffs)}")
            for sdg in diffs:
                print(f"    python={py.get(sdg, [])}")
                print(f"    rust  ={rs.get(sdg, [])}")
        else:
            print(f"{p.name}: identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
