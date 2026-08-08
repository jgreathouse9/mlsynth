#!/usr/bin/env python3
"""Recover the plotted values from the figures of arXiv:2405.00953v3, Section 5.

Section 5 of Zhang, Zhang & Zhang (2024) prints no tables: its four panels are
figures, and the numbers behind them appear nowhere in the text. The arXiv
source ships those panels as vector PDFs written by R 4.2.2, so each series
survives as an explicit polyline in the page content stream, alongside the
gridlines and tick labels needed to map device coordinates back onto the axes.

This script performs that mapping. Its output is the ``_PAPER_RATIO`` and
``_PAPER_NORM`` tables hard-coded in ``benchmarks/cases/dsc_mc.py``; run it to
re-derive them from the source instead of trusting the transcription.

Usage
-----

    curl -o 2405.00953v3.tar.gz https://arxiv.org/e-print/2405.00953v3
    mkdir -p src && tar -xzf 2405.00953v3.tar.gz -C src
    python benchmarks/reference/dsc_mc/digitize_figures.py src

Accuracy: the ratio panels resolve to about 6e-5 per device unit and the
weight-error panels to about 3e-4, both far below the Monte-Carlo noise in the
plotted values themselves.
"""
from __future__ import annotations

import re
import sys
import zlib

from pathlib import Path

# figure file -> (what it plots, the two donor-pool sizes, in draw order)
FIGURES = {
    "Fig22.pdf": ("Figure 1  model-free   Rbar(w_hat) / inf Rbar", (20, 50)),
    "Fig11.pdf": ("Figure 2  model-free   ||w_hat - w_opt||", (20, 50)),
    "Figf2.pdf": ("Figure 3  factor model Rbar(w_hat) / inf Rbar", (10, 20)),
    "Figf1.pdf": ("Figure 4  factor model ||w_hat - w_opt||", (10, 20)),
}

_NUMBER = re.compile(r"-?[\d.]+")
_MOVE = re.compile(r"([\d.]+) ([\d.]+) m")
_LINE = re.compile(r"([\d.]+) ([\d.]+) l")
_LABEL = re.compile(
    r"Tf [\d.]+ 0\.00 0\.00 [\d.]+ ([\d.]+) ([\d.]+) Tm \(([^)]*)\) Tj"
)


def page_content(path: Path) -> str:
    """Inflate the first content stream of a single-page R pdf() output."""
    raw = path.read_bytes()
    stream = re.findall(rb"stream\r?\n(.*?)endstream", raw, re.S)[0]
    return zlib.decompress(stream).decode("latin1")


def _strokes(content: str):
    """Every stroked path, as ``(linewidth, points)``.

    Gridlines are two-point axis-parallel segments; the plotted series are
    polylines through four points; the plotting symbols are Bezier curves or
    filled triangles and are not stroked as paths, so they do not appear.
    """
    out, current, width = [], [], None
    for line in content.split("\n"):
        token = line.strip()
        if (w := re.fullmatch(r"([\d.]+) w", token)) is not None:
            width = float(w.group(1))
        elif (move := _MOVE.fullmatch(token)) is not None:
            current = [(float(move.group(1)), float(move.group(2)))]
        elif (seg := _LINE.fullmatch(token)) is not None and current:
            current.append((float(seg.group(1)), float(seg.group(2))))
        elif token == "S":
            if current:
                out.append((width, current))
            current = []
    return out


def data_polylines(content: str) -> list[list[tuple[float, float]]]:
    return [pts for _, pts in _strokes(content) if len(pts) > 2]


def _major_gridlines(content: str, vertical: bool) -> list[float]:
    """Positions of the labelled gridlines along one axis.

    R draws the minor grid at 0.53 pt and the labelled major grid at 1.07 pt,
    so the majors are the axis-parallel two-point strokes at the wider of the
    two gridline widths.
    """
    fixed, varies = (0, 1) if vertical else (1, 0)
    grid = [(w, pts) for w, pts in _strokes(content)
            if len(pts) == 2
            and pts[0][fixed] == pts[1][fixed]
            and pts[0][varies] != pts[1][varies]]
    if not grid:
        return []
    widest = max(w for w, _ in grid if w is not None)
    return sorted({pts[0][fixed] for w, pts in grid if w == widest})


def _axis(content: str, vertical: bool, values: list[float]):
    """Linear device-to-data map, calibrated on the labelled gridlines.

    Calibrating on the gridlines, not on the label text, sidesteps the
    text anchor: R places a label at its left edge, an offset that depends on
    the string's width, so "0" and "100" do not sit the same distance from
    their ticks.
    """
    positions = _major_gridlines(content, vertical)
    if len(positions) != len(values):
        raise ValueError(
            f"{len(positions)} major gridlines against {len(values)} labels"
        )
    (p0, v0), (p1, v1) = (positions[0], values[0]), (positions[-1], values[-1])
    scale = (v1 - v0) / (p1 - p0)
    index = 0 if vertical else 1
    return lambda pt: v0 + (pt[index] - p0) * scale


def digitize(path: Path) -> list[list[tuple[float, float]]]:
    content = page_content(path)
    labels = [(t, float(x), float(y)) for x, y, t in _LABEL.findall(content)
              if _NUMBER.fullmatch(t)]
    # x labels share the lowest baseline; y labels share the leftmost anchor.
    bottom = min(y for _, _, y in labels)
    left = min(x for _, x, _ in labels)
    x_values = sorted(float(t) for t, _, y in labels if y == bottom)
    y_values = sorted(float(t) for t, x, _ in labels if x == left)
    to_x = _axis(content, True, x_values)
    to_y = _axis(content, False, y_values)
    return [[(to_x(pt), to_y(pt)) for pt in line]
            for line in data_polylines(content)]


def main(source: Path) -> int:
    missing = [name for name in FIGURES if not (source / name).exists()]
    if missing:
        print(f"missing from {source}: {', '.join(missing)}", file=sys.stderr)
        return 1
    for name, (caption, pools) in FIGURES.items():
        print(f"{name}  {caption}")
        for J, line in zip(pools, digitize(source / name)):
            cells = "  ".join(f"M={round(m):>3}: {v:.4f}" for m, v in line)
            print(f"    J = {J:>2}   {cells}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
