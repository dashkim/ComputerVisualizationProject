#!/usr/bin/env python3
"""Rows 0..128 copied; rows 129..255 = mirror(triCase[255 - i]) with (a,b,c)->(a,c,b)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW = re.compile(r"\{([^}]*)\}")


def parse_tricase_rows(text: str) -> list[list[int]]:
    start = text.index("{")
    depth = 0
    end = start
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    rows: list[list[int]] = []
    for line in text[start + 1 : end].splitlines():
        m = ROW.search(line.strip())
        if not m:
            continue
        rows.append([int(x) for x in m.group(1).split(",") if x.strip()])
    return rows


def mirror_row(row: list[int]) -> list[int]:
    vals = [x for x in row if x != -1]
    out: list[int] = []
    for i in range(0, len(vals), 3):
        a, b, c = vals[i], vals[i + 1], vals[i + 2]
        out += [a, c, b]
    out += [-1] * (16 - len(out))
    return out


def build_mirrored_table(src: list[list[int]]) -> list[list[int]]:
    return [list(src[i]) if i <= 128 else mirror_row(src[255 - i]) for i in range(256)]


def emit_cxx(table: list[list[int]]) -> str:
    lines = ["static int triCase[256][16] = {"]
    for i in range(256):
        inner = ", ".join(str(x) for x in table[i])
        comma = "," if i < 255 else ""
        lines.append(f"    {{{inner}}}{comma}   /* {i} */")
    lines.append("};")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    src = parse_tricase_rows(args.input.read_text(encoding="utf-8"))
    args.output.write_text(emit_cxx(build_mirrored_table(src)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
