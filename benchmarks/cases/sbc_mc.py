"""Path B benchmark: SBC vs conventional SC on nonstationary data (Shi-Xi-Xie 2025).

Reproduces the headline of the paper's simulation study (Section 4, Table 1):
on nonstationary panels SBC -- which matches on the *cyclical* component after a
Hamilton trend projection -- has lower post-treatment MSE than a conventional
synthetic control on raw levels, except in the regime conventional SC is actually
built for (partial cointegration), where it is merely competitive. The reported
statistic is the post-MSE ratio ``MSE(SBC) / MSE(conventional SC)`` against the
treated unit's true (untreated) path; below 1 means SBC wins.

* Model 1 -- independent random walks (the spurious-regression regime): SBC wins
  big.
* Model 2 -- idiosyncratic unit roots + common stationary AR factors: SBC wins.
* Model 3 -- partial cointegration (half the units share RW factors in levels):
  conventional SC is closest, so the ratio is nearest 1.

The panels are drawn in R and estimated in Python
-------------------------------------------------
The data comes from the authors' own DGP -- ``simulation/simulation_nonnegative.R``
in their repository, the script behind Table 1's non-negative-weights columns --
run in R and captured under ``benchmarks/reference/sbc_mc/`` (regenerate with
``Rscript benchmarks/reference/sbc_mc/reference.R``; base R, no CRAN package
needed). mlsynth then estimates on those panels. Splitting the two puts the
simulator on the reference side: a ratio that disagrees with Table 1 is now the
estimator's disagreement, where a Python re-implementation of the DGP would have
confounded the two. The captured panels also make the case deterministic without
a seeded generator -- every run reads the same 180 panels.

Provenance
----------
* Panels: ``benchmarks/reference/sbc_mc/panels.tsv.gz`` -- Models 1-3 at the
  Table 1 cell T0 = 100, phi = 0.5, N + 1 = 12, h = p = Fh = 2, 60 replications
  (Model 1 is the authors' DGP 1.3, the mu ~ N(0, 1/4) column).
* Table 1, post-treatment MSE ratio, non-negative weights, T0 = 100: 0.01 for
  Model 1, 0.09 for Model 2 (phi = 0.5), 0.47 for Model 3 (phi = 0.5), from
  10,000 replications. These 60 replications land at 0.0083 / 0.0222 / 0.7670,
  so the pinned quantities are the paper's ordering claims -- every ratio below
  1, Model 3 highest -- and not agreement with those three cells. The gap to
  them is the replication count and the Monte Carlo draw, and it is reported
  here at the size it is.
"""
from __future__ import annotations

import gzip
import warnings
from pathlib import Path

import numpy as np

NU, H, P, T0 = 12, 2, 2, 100
PANELS = Path(__file__).resolve().parents[1] / "reference" / "sbc_mc" / "panels.tsv.gz"


def _panels() -> dict:
    """The R-drawn panels, as ``{model: [ (NU, T) array, ... ]}``."""
    import pandas as pd

    with gzip.open(PANELS, "rt") as fh:
        raw = pd.read_csv(fh, sep="\t")
    units = [c for c in raw.columns if c.startswith("y")]
    out: dict = {}
    for (model, _rep), block in raw.groupby(["model", "rep"], sort=True):
        panel = block.sort_values("t")[units].to_numpy(dtype=float).T
        out.setdefault(int(model), []).append(panel)
    return out


def _ratio(model: int, panels: dict) -> float:
    import cvxpy as cp
    import pandas as pd

    from mlsynth import SBC

    def conv_sc(y_pre, X_pre, X_full):
        w = cp.Variable(X_pre.shape[1])
        cp.Problem(cp.Minimize(cp.sum_squares(y_pre - X_pre @ w)),
                   [w >= 0, cp.sum(w) == 1]).solve(solver="CLARABEL")
        return X_full @ w.value

    num = den = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for Y in panels[model]:
            T = Y.shape[1]
            df = pd.DataFrame([{"unit": i, "time": t, "y": float(Y[i, t]),
                                "treat": int(i == 0 and t >= T0)}
                               for i in range(NU) for t in range(T)])
            cf = np.asarray(SBC({"df": df, "outcome": "y", "treat": "treat",
                                 "unitid": "unit", "time": "time", "h": H, "p": P,
                                 "weights_mode": "simplex",
                                 "display_graphs": False}).fit().counterfactual_full)
            y1, X = Y[0], Y[1:].T
            cf_sc = conv_sc(y1[:T0], X[:T0], X)
            po = slice(T0, T)
            num += float(np.sum((cf[po] - y1[po]) ** 2))
            den += float(np.sum((cf_sc[po] - y1[po]) ** 2))
    return num / den


def run() -> dict:
    panels = _panels()
    r = {m: _ratio(m, panels) for m in (1, 2, 3)}
    return {
        "ratio_model1": r[1],
        "ratio_model2": r[2],
        "ratio_model3": r[3],
        # 1.0 iff SBC beats conventional SC in every model (all ratios < 1).
        "sbc_beats_sc_all": float(max(r.values()) < 1.0),
        # 1.0 iff conventional SC is closest under cointegration (Model 3 highest).
        "model3_closest_to_sc": float(r[3] > r[1] and r[3] > r[2]),
        "n_panels": float(sum(len(v) for v in panels.values())),
    }


# The paper's two ordering claims are pinned exactly (tolerance 0); they are what
# Table 1 supports at 60 replications, and they are what the DGP is meant to
# show. The three ratios are pinned as values: the panels are read from disk and
# both estimators are deterministic, so a run reproduces them bit for bit. Their
# tolerance is 1e-6 relative, which covers a cvxpy solver-version shift in the
# conventional-SC leg. A shift larger than that is a finding to record, not noise
# to widen the band for.
EXPECTED = {
    "ratio_model1": (0.008299959616345843, 8.3e-9),
    "ratio_model2": (0.022152692010237052, 2.2e-8),
    "ratio_model3": (0.7669984836621694, 7.7e-7),
    "sbc_beats_sc_all": (1.0, 0.0),
    "model3_closest_to_sc": (1.0, 0.0),
    "n_panels": (180.0, 0.0),
}
