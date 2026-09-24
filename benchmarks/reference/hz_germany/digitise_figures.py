"""Recover the plotted series of Hsiao & Zhou (2024) Figures 1-2 from the PDF.

    python benchmarks/reference/hz_germany/digitise_figures.py <paper.pdf>

The empirical section of the paper prints no number and the replication package
ships no code, so the figures are the only Path-A referent. Both are vector
graphics: Figure 1 carries three 44-point polylines over 1960-2003 (observed,
LP, FB) and Figure 2 six 13-point polylines over 1991-2003 (each method's effect
and its two 95% bounds).

Figure 1 is self-calibrating. Its black path is West Germany's observed log real
GDP per capita, which is in ``basedata/repgermany.dta``, so regressing PDF
y-units on the known series both fixes the axis map and confirms which polyline
is which. Figure 2 is calibrated off its labelled ticks and then checked against
Figure 1: the effects it plots must equal the observed series minus Figure 1's
counterfactuals. Both checks are asserted below, so a re-run against a different
PDF rendering fails here instead of writing a wrong referent.

Needs ``pdfminer.six``. Rewrites the two ``gold_*.csv`` files in this directory.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTLine, LTTextContainer

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

# PDF page indices (0-based) of the two figures in the published article.
FIG1_PAGE, FIG2_PAGE = 17, 18
BLACK, BLUE, RED = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
# Figure 2's y-axis: the '0' tick sits at 559.20 PDF units and '2' at 720.48.
FIG2_ZERO, FIG2_PER_UNIT = 559.20, 80.64


def _walk(obj):
    for el in obj:
        yield el
        if hasattr(el, "__iter__") and not isinstance(el, LTTextContainer):
            yield from _walk(el)


def _curves(pdf: pathlib.Path, page_index: int, min_pts: int):
    for pno, page in enumerate(extract_pages(pdf)):
        if pno != page_index:
            continue
        return [(el.stroking_color, round(el.linewidth, 2), list(el.pts))
                for el in _walk(page)
                if isinstance(el, (LTCurve, LTLine))
                and len(getattr(el, "pts", []) or []) >= min_pts]
    raise SystemExit(f"page {page_index} not found in {pdf}")


def main(pdf: pathlib.Path) -> None:
    # ---- Figure 1 ----------------------------------------------------------
    fig1 = {c: np.array(p)[:, 1] for c, _lw, p in _curves(pdf, FIG1_PAGE, 44)}
    for colour in (BLACK, BLUE, RED):
        if colour not in fig1:
            raise SystemExit(f"Figure 1: no 44-point path of colour {colour}")
    years1 = np.arange(1960, 2004)

    d = pd.read_stata(_ROOT / "basedata" / "repgermany.dta")
    wg = d[d.country == "West Germany"].sort_values("year")
    if wg.year.to_numpy(int).tolist() != years1.tolist():
        raise SystemExit("repgermany.dta does not cover 1960-2003 for West Germany")
    observed = np.log(wg.gdp.to_numpy(float))

    slope, intercept = np.polyfit(fig1[BLACK], observed, 1)
    digitised = slope * fig1[BLACK] + intercept
    resid = np.abs(observed - digitised).max()
    if resid > 1e-3:
        raise SystemExit(f"Figure 1 calibration failed: max |resid| = {resid:.5f}")
    print(f"Figure 1 calibration on the observed path: max |resid| = {resid:.2e} "
          f"log points, R^2 = {1 - (observed - digitised).var() / observed.var():.8f}")

    lp_path = slope * fig1[BLUE] + intercept
    fb_path = slope * fig1[RED] + intercept

    # ---- Figure 2 ----------------------------------------------------------
    fig2 = _curves(pdf, FIG2_PAGE, 13)
    cal = lambda pts: (np.array(pts)[:, 1] - FIG2_ZERO) / FIG2_PER_UNIT
    series = {}
    for colour, name in ((BLUE, "lp"), (RED, "fb")):
        group = [(lw, p) for c, lw, p in fig2 if c == colour]
        if len(group) != 3:
            raise SystemExit(f"Figure 2: expected 3 {name} paths, found {len(group)}")
        heavy = max(lw for lw, _ in group)          # the point estimate is the thick line
        series[f"{name}_effect"] = cal(next(p for lw, p in group if lw == heavy))
        lo, hi = sorted((cal(p) for lw, p in group if lw != heavy), key=lambda a: a[0])
        series[f"{name}_lower"], series[f"{name}_upper"] = lo, hi

    # ---- the two calibrations must agree -----------------------------------
    T0 = 31                                          # 1960-1990 pre-period
    for name, path in (("lp", lp_path), ("fb", fb_path)):
        gap = np.abs((observed[T0:] - path[T0:]) - series[f"{name}_effect"]).max()
        if gap > 1e-3:
            raise SystemExit(f"Figures 1 and 2 disagree on {name}: max |diff| = {gap:.5f}")
        print(f"Figure 2 vs Figure 1, {name}: max |diff| = {gap:.2e} log points")

    pd.DataFrame({"year": years1, "actual_observed": observed,
                  "actual_digitised": digitised, "lp_counterfactual": lp_path,
                  "fb_counterfactual": fb_path}).to_csv(
        _HERE / "gold_figure1_paths.csv", index=False, float_format="%.6f")
    pd.DataFrame({"year": np.arange(1991, 2004), **series}).to_csv(
        _HERE / "gold_figure2_effects.csv", index=False, float_format="%.6f")
    print(f"wrote {_HERE / 'gold_figure1_paths.csv'}")
    print(f"wrote {_HERE / 'gold_figure2_effects.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(pathlib.Path(sys.argv[1]))
