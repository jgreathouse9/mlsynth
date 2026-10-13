r"""Wan, Xie & Hsiao (2018) Table 2, Design 6a: PDA against SCM.

Path B (scenario 3 -- full replication archive). Reproduces the Design 6a column
of Table 2 in Wan, Xie & Hsiao, "Panel data approach vs synthetic control
method", Economics Letters 164 (2018), 121-123: the post-treatment mean squared
prediction error of the panel data approach and of the synthetic control method,
over five ``(J, T0)`` cells and under both of the paper's two aggregation rules.

Design 6a's data-generating process needs no external data, and it is the design
the paper singles out as the one where SCM beats PDA: "PDA significantly
dominates SCM except in a few cases where J = T0 = 5 or J/T0 is large, or for
Design 6a where the best predictor is the simple average of the control units."
Its outcome is a common stochastic trend with unit loadings,

.. math::

   \lambda_t = \eta + \lambda_{t-1} + v_t, \quad v_t \sim N(0, 0.5^2),
   \quad \eta \sim \chi^2_1, \quad \lambda_1 = 0,

   y_{i,t} = \lambda_t + \varepsilon_{i,t},
   \quad \varepsilon_{i,t} \sim N(0, 0.25^2),

with :math:`\eta` drawn once per replication and shared by every unit. Every
unit therefore loads on the trend identically, which makes the simple average of
the controls the best predictor of the treated unit -- the paper's own reading of
this design. That average lies inside the simplex SCM searches, so SCM finds it;
PDA, whose coefficients are unrestricted, spends degrees of freedom estimating a
combination it does not need.

The theoretical floor follows from that reading. A convex combination
:math:`w` of the :math:`J` controls predicts with error
:math:`\varepsilon_{1,t} - \sum_i w_i \varepsilon_{i,t}`, minimised at equal
weights, giving :math:`\sigma^2 (1 + 1/J) = 0.0625\,(1 + 1/J)`: 0.075 at
:math:`J = 5`, 0.069 at :math:`J = 10`, 0.066 at :math:`J = 20`. The paper's SCM
row sits just above it in every cell, which is what a benchmark on this design
can pin most sharply.

Provenance
----------

* Paper: Wan, Xie & Hsiao (2018), Economics Letters 164, 121-123, Table 2,
  Design 6a column (both the "1000-rule" and "MAE-rule" halves).
* Archive: the paper's supplementary code, ``sim6a_c3ii.R`` (the ``(5,40)``
  cell) and ``sim6a_c9.R`` (the ``(20,20)`` cell). The data-generating block
  here is a transcription of theirs; the estimation settings follow the same
  scripts -- ``pampe(select = "AICc", nvmax = t0 - 4)`` for PDA, which is
  mlsynth's ``PDA(method="hcw", hcw_nvmax=t0-4)``, and an SCM with each
  pre-period outcome as its own predictor, which is ``VanillaSC``.
* The companion case ``wan_pda_vs_scm_ref`` cross-validates both estimators
  against ``pampe`` and ``Synth`` themselves, on all six design variants the
  archive ships. This case is the paper-facing half and runs without R.

The two rules
-------------

The 1000-rule averages over every replication. The MAE-rule is Gardeazabal &
Vega-Bayo's good-match filter, which keeps a replication when
:math:`\mathrm{MAE}_{0,\mathrm{SCM}} < 0.2\,|\bar y^0_1|`, with
:math:`\bar y^0_1` the pre-treatment mean of the treated unit. Section 3 of the
paper objects that this rule keeps many more replications for PDA than for SCM,
so the two columns would average over different experiments, and fixes it by
keeping an equal number: if the rule selects :math:`K` good SCM matches, the
:math:`K` best PDA matches are kept by adjusting the critical value :math:`c` in
:math:`\mathrm{MAE}_{0,\mathrm{PDA}} < c\,|\bar y^0_1|`. Ranking on
:math:`\mathrm{MAE}_{0,\mathrm{PDA}} / |\bar y^0_1|` and cutting at the
:math:`K`-th is that adjustment.

One cell that does not reproduce
--------------------------------

Nineteen of the twenty cells land on the paper's printed values. The exception
is the ``(5,5)`` MAE-rule PDA cell: the paper prints 0.77 and this case measures
0.99. The 1000-rule cell beside it reproduces (0.903 against 0.91), so the fit
agrees and the filter is what diverges.

The gap is not a property of this port. The authors' script defines six
subsetting rules (``s1`` through ``s6``) and reports both methods under each;
Section 3 of the paper describes a seventh, the equal-K adjustment above. Their
script was run unmodified at ``(5,5)`` -- their DGP, their seed, their
``pampe``, their ``Synth`` -- for 400 replications, and every rule gives the
same answer:

============== ========= =========  =====
rule            PDA MSE   SCM MSE       n
============== ========= =========  =====
unfiltered         0.861     0.094    400
``s1``             0.864     0.093    241
``s2``             0.874     0.094    384
``s3``             0.847     0.092    258
``s4``             0.853     0.092    257
``s5``             0.865     0.093    277
``s6``             0.864     0.094    397
equal-K            0.871     0.093    241
============== ========= =========  =====

The PDA column spans 0.847 to 0.874 across all seven filters, against an
unfiltered 0.861 and the paper's printed 0.77. The SCM column reproduces under
every one of them (0.092 to 0.094 against the paper's 0.10), so the panels, the
estimators and the metric are all behaving.

What accounts for the PDA column refusing to move: the correlation between
:math:`\mathrm{MAE}_{0,\mathrm{PDA}} / |\bar y^0_1|` and the post-period MSE
is -0.030 at ``(5,5)``. Across the five cells the largest such correlation in
absolute value is 0.081, at ``(20,20)``, where the replication count is 60 and
the sampling error on a correlation is itself about 0.13. The pre-period fit
carries no information about post-period accuracy in this design, so no filter
built on it can move the PDA column, whichever form it takes. That is
consistent with the paper's other four cells, where its MAE-rule and 1000-rule
values differ by at most 0.01, and inconsistent with its ``(5,5)`` pair alone.

``nvmax`` is not the cause either: the AICc choice at :math:`T_0 = 5` selects
one control whether the cap is 1, 2, 3 or absent, and the cell is identical
under all four.

The cell is reported here at the measured value with the paper's printed value
alongside. The reference run that produced the table above is
``benchmarks/R/wan_pda_vs_scm.R``'s Design 6a block under the archive's own
subsetting rules; it is a negative result, and recording it is what keeps the
next reader from spending the same afternoon on it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# (J, T0, T1, replications). The (20,20) cell runs a best-subset search over 20
# donors at 16 admissible sizes and costs about a second a replication, two
# orders of magnitude more than the others, so it gets fewer draws.
CELLS = {
    "j5t5":   (5, 5, 10, 400),
    "j5t20":  (5, 20, 10, 400),
    "j10t20": (10, 20, 10, 400),
    "j20t20": (20, 20, 10, 60),
    "j5t40":  (5, 40, 20, 400),
}

SEED = 20180101
_SIGMA2 = 0.25 ** 2          # idiosyncratic variance in the authors' DGP


def _panel(rng, J, T0, T1):
    """One replication of Design 6a, transcribed from ``sim6a_c9.R``."""
    T = T0 + T1
    eta = rng.chisquare(1)
    lam = np.zeros(T)
    for s in range(1, T):
        lam[s] = eta + lam[s - 1] + rng.normal(0.0, 0.5)
    return lam[:, None] + rng.normal(0.0, 0.25, size=(T, J + 1))


def _frame(y, T0):
    T, n = y.shape
    unit = np.repeat(np.arange(1, n + 1), T)
    period = np.tile(np.arange(1, T + 1), n)
    return pd.DataFrame({"unit": unit, "time": period, "y": y.T.reshape(-1),
                         "treat": ((unit == 1) & (period > T0)).astype(int)})


def _cell(tag):
    """Both rules for one ``(J, T0)`` cell: MSE of each method, plus diagnostics."""
    from mlsynth.estimators.pda import PDA
    from mlsynth.estimators.vanillasc import VanillaSC

    J, T0, T1, M = CELLS[tag]
    nvmax = max(1, T0 - 4)                # pampe(..., nvmax = t0 - 4)
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(M):
        y = _panel(rng, J, T0, T1)
        base = dict(df=_frame(y, T0), outcome="y", treat="treat", unitid="unit",
                    time="time", display_graphs=False)
        rec = {"ybar": abs(float(y[:T0, 0].mean()))}
        ok = True
        for key, make in (
            ("pda", lambda: PDA({**base, "method": "hcw", "hcw_nvmax": nvmax})),
            ("scm", lambda: VanillaSC(base)),
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cf = np.asarray(
                        make().fit().time_series.counterfactual_outcome, float)
                if not np.isfinite(cf).all():
                    raise ValueError("non-finite counterfactual")
                rec[f"mse_{key}"] = float(np.mean((y[T0:, 0] - cf[T0:]) ** 2))
                rec[f"mae_{key}"] = float(np.mean(np.abs(y[:T0, 0] - cf[:T0])))
            except Exception:
                ok = False
        if ok:
            rows.append(rec)

    d = pd.DataFrame(rows)
    keep = d.mae_scm < 0.2 * d.ybar                     # the good-match filter
    K = int(keep.sum())
    best = (d.mae_pda / d.ybar).sort_values().index[:K]   # the equal-K adjustment
    return {
        "pda_1000": float(d.mse_pda.mean()),
        "scm_1000": float(d.mse_scm.mean()),
        "pda_mae": float(d.loc[best, "mse_pda"].mean()) if K else float("nan"),
        "scm_mae": float(d.loc[keep, "mse_scm"].mean()) if K else float("nan"),
        "floor": _SIGMA2 * (1.0 + 1.0 / J),
        "corr": float(np.corrcoef(d.mae_pda / d.ybar, d.mse_pda)[0, 1]),
        "n": len(d),
    }


def run() -> dict:
    cells = {tag: _cell(tag) for tag in CELLS}
    out = {}
    for tag, r in cells.items():
        out[f"pda_1000_{tag}"] = round(r["pda_1000"], 4)
        out[f"scm_1000_{tag}"] = round(r["scm_1000"], 4)
        out[f"pda_mae_{tag}"] = round(r["pda_mae"], 4)
        out[f"scm_mae_{tag}"] = round(r["scm_mae"], 4)
    # The paper's reading of Design 6a: SCM is the better predictor in every
    # cell, because the equal-weight average of the controls is optimal here and
    # lies inside the simplex.
    out["scm_beats_pda_all_cells"] = float(
        all(r["scm_1000"] < r["pda_1000"] for r in cells.values()))
    # and it is close to the floor a convex combination cannot beat, while PDA
    # is not. The worst cell of each is reported.
    out["scm_worst_floor_ratio"] = round(
        max(r["scm_1000"] / r["floor"] for r in cells.values()), 3)
    out["pda_worst_floor_ratio"] = round(
        max(r["pda_1000"] / r["floor"] for r in cells.values()), 3)
    # The pre-period fit carries no information about post-period accuracy, which
    # is why the two rules agree in four cells of five.
    out["max_abs_mae_mse_corr"] = round(
        max(abs(r["corr"]) for r in cells.values()), 3)
    return out


# Deterministic: one seeded RNG stream per cell, so re-running returns identical
# numbers. Tolerances are Monte Carlo noise at these replication counts and
# nothing else -- the MSE distribution on this design is right-skewed, so the
# (5,5) cells, where the skew is worst, carry the widest bands.
#
# The paper's printed Design 6a column is quoted against each cell. Nineteen of
# the twenty agree; ``pda_mae_j5t5`` is the exception documented in the module
# docstring, pinned at the measured value with the paper's 0.77 recorded beside
# it so the gap stays visible instead of being absorbed into a tolerance.
EXPECTED = {
    # --- 1000-rule: the mean over every replication ---------------------------
    "pda_1000_j5t5":   (0.9030, 0.30),    # paper 0.91
    "scm_1000_j5t5":   (0.0980, 0.012),   # paper 0.10
    "pda_1000_j5t20":  (0.1410, 0.020),   # paper 0.14
    "scm_1000_j5t20":  (0.0860, 0.010),   # paper 0.09
    "pda_1000_j10t20": (0.1520, 0.025),   # paper 0.15
    "scm_1000_j10t20": (0.0780, 0.010),   # paper 0.08
    "pda_1000_j20t20": (0.2360, 0.090),   # paper 0.23; M = 60, so the band is wide
    "scm_1000_j20t20": (0.0740, 0.020),   # paper 0.08
    "pda_1000_j5t40":  (0.1080, 0.012),   # paper 0.11
    "scm_1000_j5t40":  (0.0820, 0.008),   # paper 0.08
    # --- MAE-rule: the equal-K good-match subsets ----------------------------
    "pda_mae_j5t5":    (0.9920, 0.30),    # paper 0.77 -- see the docstring
    "scm_mae_j5t5":    (0.0950, 0.012),   # paper 0.10
    "pda_mae_j5t20":   (0.1390, 0.020),   # paper 0.14
    "scm_mae_j5t20":   (0.0850, 0.010),   # paper 0.09
    "pda_mae_j10t20":  (0.1530, 0.025),   # paper 0.15
    "scm_mae_j10t20":  (0.0780, 0.010),   # paper 0.08
    "pda_mae_j20t20":  (0.2456, 0.090),   # paper 0.24
    "scm_mae_j20t20":  (0.0730, 0.020),   # paper 0.08
    "pda_mae_j5t40":   (0.1060, 0.012),   # paper 0.11
    "scm_mae_j5t40":   (0.0820, 0.008),   # paper 0.08
    # --- what the design is for ---------------------------------------------
    "scm_beats_pda_all_cells": (1.0, 0.0),
    "scm_worst_floor_ratio": (1.31, 0.20),   # SCM stays near sigma^2 (1 + 1/J)
    "pda_worst_floor_ratio": (12.0, 6.0),    # PDA does not
    "max_abs_mae_mse_corr": (0.081, 0.050),  # pre-period fit predicts nothing
}
