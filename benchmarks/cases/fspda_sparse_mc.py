"""PDA against fsPDA's sparse simulation, the second study in their simulation/.

``zhentaoshi/fsPDA``'s ``simulation/`` directory holds two Monte Carlos. The
dense one is ``fspda_dense_mc``; this is the sparse one, behind their appendix
Tables B1-B3, where the truth really is a handful of donors and the question is
whether a method finds them.

Path: cross-validation against an authoritative reference implementation. The
reference script sources their ``fs.R``, ``lasso_ic.R``, ``oracle.R`` and the
Newey-West branch of ``prd.R`` verbatim, and generates panels with the three
DGPs of ``main_function.R``. mlsynth sees only the finished panel, handed to
``PDA(...).fit()`` as a long DataFrame.

Design: their smallest grid cell, ``t = 100`` with ``t1 = 50``, ``n = 50``
donors of which ``s0 = 4`` are relevant, ten replications on each of their three
DGPs -- ``iid`` (independent normals), ``inid`` (four ARMA blocks) and ``nnd`` (a
four-factor model, the dense-factor case). ``lasso_ic`` runs at their
``ic = "WIC"``, ``cst = 2``.

Why this is a comparison of rates, not of digits
-------------------------------------------------
The sparse study does not run the same estimators as the dense one, and mlsynth
implements the dense one's. Three differences, none of them accidental:

``sparse/fs.R`` maximises the fitted sum of squares over all ``N`` donors at
every step, including ones it has already picked; it scores its stopping rule on
``var(e)``; and it discards its last pick on exit. ``nonsparse/FS.R`` minimises
``mean(e^2)`` over the donors not yet picked and keeps what it selects. mlsynth
implements the latter, which is also what the released ``est.fsPDA.R`` does.

``lasso_ic``'s WIC uses ``var(y1 - yhat)`` where ``lasso.BIC`` uses
``colMeans(e.hat^2)``, and searches ``seq(0.01, 0.5, length = 100)`` where
``lasso.BIC`` searches ``seq(0.01, 1, by = 0.01)``. mlsynth's
``lasso_criterion="mbic"`` reproduces ``lasso.BIC``.

``oracle`` regresses on the true support. No estimator can run it, so it enters
here only as the floor the others are measured against.

So the cells below are agreement rates and error ratios, and none of them is
expected to be 1. What they pin is that two rules built from the same idea land
in the same place most of the time, and that where they diverge they diverge by
a bounded amount. A port regression moves these sharply; the standing difference
between the two rules does not.

What the numbers say. Forward selection picks the identical donor set on 28 of
30 panels despite the algorithms differing, and its out-of-sample RMSE stays
within 5.4% of theirs. The LASSO agrees on 21 of 30 -- 9 of 10 on ``iid``, 10 of
10 on ``inid``, but only 2 of 10 on ``nnd``, the dense-factor DGP where the
criterion difference bites hardest and neither rule is really in its element.
Both recover the true four donors on every panel of the two sparse DGPs.

Reference bundle: ``benchmarks/reference/fspda_sparse_mc/`` (regenerate with
``Rscript benchmarks/reference/fspda_sparse_mc/reference.R``).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_CASE = "fspda_sparse_mc"
_PANELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reference", _CASE, "panels.tsv.gz")

DGPS = ("iid", "inid", "nnd")


def _panels() -> dict:
    df = pd.read_csv(_PANELS, sep="\t")
    out = {}
    for cell, blk in df.groupby("cell", sort=False):
        blk = blk.sort_values("t")
        out[str(cell)] = (blk["y"].to_numpy(dtype=float),
                          np.array([np.fromstring(r, sep=" ") for r in blk["donors"]]))
    return out


def _long(y: np.ndarray, Y: np.ndarray, T1: int) -> pd.DataFrame:
    T, N = Y.shape
    recs = [{"unit": "treated", "time": t + 1, "y": float(y[t]),
             "treat": int(t >= T1)} for t in range(T)]
    recs += [{"unit": f"d{j + 1}", "time": t + 1, "y": float(Y[t, j]), "treat": 0}
             for t in range(T) for j in range(N)]
    return pd.DataFrame(recs)


def _fit(df: pd.DataFrame, lag: int):
    from mlsynth import PDA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDA({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "methods": ["fs", "LASSO"], "fs_intercept": False,
            "lrvar_lag": lag, "lasso_criterion": "mbic", "display_graphs": False,
        }).fit()


def _selected(fit) -> list:
    return sorted(int(str(u)[1:]) for u in fit.selected_donors)


def _ref_vec(values: dict, prefix: str, n: int) -> list:
    return [values[f"{prefix}_k{i + 1}"] for i in range(n)]


def run() -> dict:
    ref = load_reference(_CASE)["values"]
    T, T1 = int(ref["T"]), int(ref["T1"])
    S0, reps = int(ref["s0"]), int(ref["n_reps"])
    # prd.R's Newey-West branch fixes the truncation at round(t2^(1/3)).
    lag = int(round((T - T1) ** (1 / 3)))
    truth = set(range(1, S0 + 1))
    panels = _panels()

    acc = {k: [] for k in ("fs_same", "las_same", "fs_true", "las_true",
                           "fs_ratio", "las_ratio", "fs_vs_orc", "las_vs_orc")}

    for dgp in DGPS:
        for r in range(1, reps + 1):
            cell = f"{dgp}_r{r}"
            y, Y = panels[cell]
            res = _fit(_long(y, Y, T1), lag)
            orc = ref[f"orc_rmse_{cell}"]
            for nm, key, tag in (("fs", "fs", "fs"), ("lasso", "las", "las")):
                fit = res.fits[nm]
                sel = _selected(fit)
                their = [int(v) for v in
                         _ref_vec(ref, f"{key}_sel_{cell}", int(ref[f"{key}_R_{cell}"]))]
                gap = np.asarray(fit.gap, dtype=float)[T1:]
                rmse = float(np.sqrt(np.mean(gap ** 2)))
                acc[f"{tag}_same"].append(float(sel == their))
                acc[f"{tag}_true"].append(float(truth <= set(sel)))
                acc[f"{tag}_ratio"].append(rmse / ref[f"{key}_rmse_{cell}"])
                acc[f"{tag}_vs_orc"].append(rmse / orc)

    m = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    return {
        "n_cells": float(len(m["fs_same"])),
        "fs_selection_match_rate": float(m["fs_same"].mean()),
        "lasso_selection_match_rate": float(m["las_same"].mean()),
        "fs_true_support_rate": float(m["fs_true"].mean()),
        "lasso_true_support_rate": float(m["las_true"].mean()),
        "fs_max_rmse_ratio_dev": float(np.abs(m["fs_ratio"] - 1).max()),
        "lasso_max_rmse_ratio_dev": float(np.abs(m["las_ratio"] - 1).max()),
        "fs_mean_rmse_vs_oracle": float(m["fs_vs_orc"].mean()),
        "lasso_mean_rmse_vs_oracle": float(m["las_vs_orc"].mean()),
        "min_rmse_vs_oracle": float(min(m["fs_vs_orc"].min(), m["las_vs_orc"].min())),
    }


# Deterministic: the panels are fixed in the bundle, forward selection is a
# greedy search followed by OLS, and the LASSO path at a fixed penalty is
# strictly convex. No seeds enter at run time.
#
# The agreement rates carry one panel of slack (1/30 = 0.034) on fs, which agrees
# on 28 of 30, and two panels (0.067) on the LASSO, which agrees on 21 of 30 and
# is the more fragile of the two because its criterion and its penalty grid both
# differ from theirs. These are rates between deliberately different rules, so
# they are pinned where they land, not at 1.0.
#
# The RMSE ratio bounds are the observed maxima rounded up: 0.054 -> 0.08 for fs
# and 0.121 -> 0.18 for the LASSO. They bound how far the two rules' predictions
# can drift apart, which is the quantity a port regression would blow through.
#
# min_rmse_vs_oracle guards the direction of the comparison: the oracle regresses
# on the true support, so no method that has to find it should beat the oracle by
# much. A value below 1 on some panel is ordinary sampling noise; a mean below 1
# would mean the oracle column had been misread.
EXPECTED = {
    "n_cells": (30.0, 0.0),
    "fs_selection_match_rate": (0.933, 0.034),        # 28/30
    "lasso_selection_match_rate": (0.700, 0.067),     # 21/30
    "fs_true_support_rate": (0.900, 0.034),           # misses only on nnd
    "lasso_true_support_rate": (1.0, 0.034),
    "fs_max_rmse_ratio_dev": (0.0, 0.08),
    "lasso_max_rmse_ratio_dev": (0.0, 0.18),
    "fs_mean_rmse_vs_oracle": (1.25, 0.15),
    "lasso_mean_rmse_vs_oracle": (1.26, 0.15),
    "min_rmse_vs_oracle": (1.0, 0.35),
}
