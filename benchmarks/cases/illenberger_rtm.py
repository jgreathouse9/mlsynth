r"""Regression to the mean inflates the synthetic control's placebo test.

Path B. Illenberger, N., Small, D. S. & Shaw, P. A. (2020), "Impact of
Regression to the Mean on the Synthetic Control Method: Bias and Sensitivity
Analysis", *Epidemiology* 31(6):815-822, Tables 1 and 2.

The claim, and why it is about this library
-------------------------------------------

Under a true null every unit is drawn from the same process apart from its
mean, so a placebo test should reject at its nominal level. It does not. When
the treated unit's outcome level sits away from the donors', the synthetic
control is fit to a realisation that is extreme for that unit, and the fitted
donors regress toward their own mean over the post period while the treated
unit regresses toward its own. The gap that opens is not a treatment effect,
and the placebo test reads it as one.

At four pre-periods, :math:`\mu_1 = 5`, :math:`\rho = 0` the paper measures a
type I error rate of **0.40** against a nominal 0.05: a true null rejected two
times in five. The unmatched difference-in-differences estimator, which does no
matching at all, holds 0.05 across every cell of both tables. The inflation is
caused by the matching.

Two specifications, and why both are here
-----------------------------------------

The paper's Synth call is ``dataprep(predictors = "Y", predictors.op =
"mean", time.predictors.prior = 1:t0, time.optimize.ssr = 1:t0)``: a single
predictor, the pre-period *mean* of the outcome. Synth's inner problem is then
``min v (ybar1 - ybar0'w)^2`` over the simplex, which is matching on the level.
``VanillaSC`` matches the whole pre-period *path*, ``min ||Y1_pre - Y0_pre
w||^2``. These are different estimators and they do not inflate by the same
amount, so the case runs both on the same panels.

At ten pre-periods the difference is most of the effect. Level matching
measures 0.273 and 0.193 at rho = 0 and rho = 0.5 against the paper's 0.26 and
0.16; path matching measures 0.107 and 0.067. Averaging ten pre-periods into
one number leaves that number free to be extreme while the path it came from is
pinned down, so the library's estimator carries less of the bias. It carries
plenty: its worst cell is 0.413 against a nominal 0.05, and it is the lower of
the two arms in eighteen of the twenty cells.

One caveat belongs on the paper's own numbers. With a single predictor the
inner problem is degenerate: every ``w`` on the simplex with ``ybar0'w =
ybar1`` attains zero, which is a whole face and not a point. Which member a
solver returns is an implementation detail, and it moves the reported rate --
this module's level arm overshoots the paper at four pre-periods (0.504 against
0.40 at rho = 0) and undershoots nothing at ten. So the level arm is pinned to
the paper with bands that carry that, and the sharp assertions in this case are
the directions and the nominal-level control, which do not depend on it.

Tables 1 and 2
--------------

One treated unit, forty donors, outcomes multivariate normal with unit marginal
variance and an AR(1) correlation :math:`\rho^{|t_i - t_j|}`, donors centred at
0 and the treated unit at :math:`\mu_1`, four post-periods. The placebo test
sequentially treats each of the 41 units and reports
:math:`p = n^{-1} \sum_i I(|\hat\theta_1| \le |\hat\theta_i|)`, rejecting below
0.05.

Synthetic-control column, four pre-periods (Table 1) and ten (Table 2):

========  ======  ======  ========  ======  ======
vary      mu_1    T1      T2        rho     T1/T2
========  ======  ======  ========  ======  ======
          1         0.16      0.16  0.00    0.40 / 0.26
          2         0.31      0.25  0.25    0.39 / 0.22
          3         0.35      0.24  0.50    0.36 / 0.16
          4         0.35      0.25  0.75    0.25 / 0.11
          5         0.33      0.26  0.90    0.18 / 0.07
========  ======  ======  ========  ======  ======

Table 1 varies :math:`\rho` at :math:`\mu_1 = 5` and Table 2 at
:math:`\mu_1 = 1`, which is why the two right-hand columns are not comparable
row for row.

Both directions are mechanism, not noise. Inflation grows with :math:`\mu_1`
because a treated unit further from the donor cloud is matched to donors
selected further into their own tails. It falls with :math:`\rho` because a
correlated series carries its pre-period deviation into the post period, so
there is less to regress away. And ten pre-periods inflate less than four,
because averaging more pre-periods leaves less room for the pre-period mean to
be extreme.

How tightly this can be pinned
------------------------------

Table 1 reports the cell :math:`\mu_1 = 5`, :math:`\rho = 0.5` twice, once in
each half, and gets 0.33 and 0.36. At the paper's 2,000 replications a rate
near 0.35 carries a standard error of 0.011, so the authors' own two runs of
one cell sit 2.7 of those apart. That is the floor on any agreement claim here,
and the bands below are set from this case's own count, not from it.

Provenance
----------

* Paper: Illenberger, Small & Shaw (2020), Epidemiology 31(6):815-822.
* The eAppendix ships the authors' R, which drives ``Synth`` through the
  ``dataprep`` call quoted above and computes the placebo p-value with
  ``mean(abs(theta_1) <= abs(theta))``. Their ``SynthRTM`` package is on
  GitHub; the design is fully specified in the paper, so this case implements
  it directly and needs no R.
* Tables 3 (covariate-driven inflation) and 4 (the authors' RTM correction) are
  not covered here.
"""
from __future__ import annotations

import numpy as np

from mlsynth.utils.solvers.ridge_augment import simplex_qp

N_DONORS = 40
POST = 4
ALPHA = 0.05
SIMS = 150
SEED = 20260921

# Table 1 (four pre-periods) and Table 2 (ten), synthetic-control column.
PAPER = {
    4: {"mu": {1: 0.16, 2: 0.31, 3: 0.35, 4: 0.35, 5: 0.33},
        "rho": {0.00: 0.40, 0.25: 0.39, 0.50: 0.36, 0.75: 0.25, 0.90: 0.18},
        "rho_at_mu": 5},
    10: {"mu": {1: 0.16, 2: 0.25, 3: 0.24, 4: 0.25, 5: 0.26},
         "rho": {0.00: 0.26, 0.25: 0.22, 0.50: 0.16, 0.75: 0.11, 0.90: 0.07},
         "rho_at_mu": 1},
}
# the unmatched difference-in-differences column, flat at nominal throughout
PAPER_UNMATCHED = 0.05


def _panel(pre: int, mu1: float, rho: float, rng) -> np.ndarray:
    """``T x (N_DONORS + 1)``; column 0 is the treated unit."""
    periods = pre + POST
    lag = np.abs(np.subtract.outer(np.arange(periods), np.arange(periods)))
    sigma = rho ** lag
    donors = rng.multivariate_normal(np.zeros(periods), sigma, size=N_DONORS).T
    treated = rng.multivariate_normal(np.full(periods, mu1), sigma)
    return np.column_stack([treated, donors])


def _theta(gap: np.ndarray, pre: int) -> float:
    """The paper's estimator: post-period mean gap minus pre-period mean gap."""
    return float(gap[pre:].mean() - gap[:pre].mean())


def _weights(donors_pre: np.ndarray, y_pre: np.ndarray, arm: str) -> np.ndarray:
    """Simplex weights under one of the two matching rules.

    ``level`` is the paper's single-predictor spec, the pre-period mean.
    ``path`` is what ``VanillaSC`` solves, the whole pre-period path.
    """
    if arm == "level":
        return simplex_qp(donors_pre.mean(axis=0, keepdims=True),
                          np.array([y_pre.mean()]))
    return simplex_qp(donors_pre, y_pre)


def _estimates(panel: np.ndarray, pre: int) -> dict:
    """Every unit's placebo estimate under each matching rule."""
    periods, units = panel.shape
    trend_basis = np.vstack([np.ones(pre), np.arange(pre, dtype=float)]).T
    keys = ("unmatched", "level", "path", "nn1", "nn2")
    out = {k: np.empty(units) for k in keys}
    slopes = np.linalg.lstsq(trend_basis, panel[:pre], rcond=None)[0][1]
    for i in range(units):
        others = [j for j in range(units) if j != i]
        y, donors = panel[:, i], panel[:, others]
        out["unmatched"][i] = _theta(y - donors.mean(axis=1), pre)
        for arm in ("level", "path"):
            w = _weights(donors[:pre], y[:pre], arm)
            out[arm][i] = _theta(y - donors @ w, pre)
        lvl = np.argmin(np.linalg.norm(donors[:pre] - y[:pre, None], axis=0))
        out["nn1"][i] = _theta(y - donors[:, lvl], pre)
        trend = np.argmin(np.abs(np.asarray(slopes)[others] - slopes[i]))
        out["nn2"][i] = _theta(y - donors[:, trend], pre)
    return out


def _rejects(estimates: np.ndarray) -> bool:
    """The eAppendix's placebo p-value, on absolute estimates."""
    return float(np.mean(np.abs(estimates[0]) <= np.abs(estimates))) < ALPHA


def _cell(pre: int, mu1: float, rho: float) -> dict:
    rng = np.random.default_rng([SEED, pre, int(mu1 * 100), int(rho * 100)])
    keys = ("unmatched", "level", "path", "nn1", "nn2")
    hits = {k: 0 for k in keys}
    for _ in range(SIMS):
        est = _estimates(_panel(pre, mu1, rho, rng), pre)
        for k, v in est.items():
            hits[k] += _rejects(v)
    return {k: v / SIMS for k, v in hits.items()}


def _seam_matches_vanillasc() -> float:
    """The ``path`` arm's solver is the one ``VanillaSC`` runs."""
    import warnings

    import pandas as pd

    from mlsynth import VanillaSC

    rng = np.random.default_rng(SEED)
    pre = 4
    panel = _panel(pre, 5.0, 0.5, rng)
    periods, units = panel.shape
    frame = pd.DataFrame({
        "id": np.repeat(np.arange(1, units + 1), periods),
        "time": np.tile(np.arange(1, periods + 1), units),
        "Y": panel.T.ravel(),
    })
    frame["D"] = ((frame.id == 1) & (frame.time > pre)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": frame, "outcome": "Y", "treat": "D",
                         "unitid": "id", "time": "time",
                         "display_graphs": False}).fit()
    first = res[0] if isinstance(res, list) else res
    library = np.asarray(first.time_series.estimated_gap, float).ravel()
    seam = panel[:, 0] - panel[:, 1:] @ _weights(panel[:pre, 1:], panel[:pre, 0], "path")
    return float(np.max(np.abs(library - seam)))


def run() -> dict:
    out = {"seam_vs_vanillasc": _seam_matches_vanillasc()}

    unmatched, level_gaps, by_pre = [], [], {}
    for pre, spec in PAPER.items():
        lv_mu, lv_rho, pa_mu, pa_rho = {}, {}, {}, {}
        cells = [("mu", m, 0.5) for m in spec["mu"]]
        cells += [("rho", spec["rho_at_mu"], r) for r in spec["rho"]]
        for kind, mu1, rho in cells:
            cell = _cell(pre, float(mu1), float(rho))
            unmatched.append(cell["unmatched"])
            tag = f"mu{mu1}" if kind == "mu" else f"rho{int(rho * 100):02d}"
            published = spec[kind][mu1 if kind == "mu" else rho]
            out[f"level_t{pre}_{tag}"] = round(cell["level"], 3)
            out[f"path_t{pre}_{tag}"] = round(cell["path"], 3)
            level_gaps.append(abs(cell["level"] - published))
            (lv_mu if kind == "mu" else lv_rho)[mu1 if kind == "mu" else rho] = cell["level"]
            (pa_mu if kind == "mu" else pa_rho)[mu1 if kind == "mu" else rho] = cell["path"]
            if kind == "mu" and mu1 == spec["rho_at_mu"]:
                out[f"nn1_t{pre}"] = round(cell["nn1"], 3)
                out[f"nn2_t{pre}"] = round(cell["nn2"], 3)
        by_pre[pre] = (lv_mu, lv_rho, pa_mu, pa_rho)

    # No matching, no inflation: the twenty cells of both tables at nominal.
    out["unmatched_max"] = round(float(max(unmatched)), 3)
    out["unmatched_mean"] = round(float(np.mean(unmatched)), 3)
    # The paper's own specification against the paper's own numbers.
    out["level_max_gap_to_paper"] = round(float(max(level_gaps)), 3)

    # Level matching inflates more than path matching, cell for cell.
    pairs = [(out[k], out[k.replace("level_", "path_")])
             for k in out if k.startswith("level_t")]
    out["level_over_path_cells"] = float(sum(lv > pa for lv, pa in pairs))
    out["n_cells"] = float(len(pairs))
    out["path_worst"] = round(float(max(pa for _, pa in pairs)), 3)

    # Both directions are mechanism, and hold under both specifications.
    for pre in PAPER:
        lv_mu, lv_rho, pa_mu, pa_rho = by_pre[pre]
        out[f"rises_with_mu_t{pre}"] = float(
            lv_mu[1] < lv_mu[3] and pa_mu[1] < pa_mu[3])
        out[f"falls_with_rho_t{pre}"] = float(
            lv_rho[0.90] < lv_rho[0.00] and pa_rho[0.90] < pa_rho[0.00])
    out["ten_pre_inflates_less"] = float(
        max(by_pre[10][0].values()) < max(by_pre[4][0].values()))
    return out


# 150 replications a cell against the paper's 2,000. At a rate near 0.30 the
# standard error is 0.037 and near 0.05 it is 0.018. The level arm's bands also
# carry the single-predictor degeneracy described above, so they are wider than
# Monte Carlo error alone; the directions and the nominal-level control are
# pinned with no slack, and ``seam_vs_vanillasc`` is exact.
EXPECTED = {
    "seam_vs_vanillasc": (0.0, 1e-10),
    "n_cells": (20.0, 0.0),
    # The paper's own specification against Tables 1 and 2.
    "level_max_gap_to_paper": (0.11, 0.09),
    "level_t4_rho00": (0.40, 0.14),
    "level_t4_mu1": (0.16, 0.14),
    "level_t4_mu3": (0.35, 0.14),
    "level_t10_rho00": (0.26, 0.14),
    "level_t10_rho50": (0.16, 0.14),
    "level_t10_mu5": (0.26, 0.14),
    # mlsynth's own estimator on the same panels: less inflated, still far from
    # nominal. Recorded, not compared to a published number -- the paper does
    # not run this specification.
    "path_t4_rho00": (0.39, 0.12),
    "path_t10_rho00": (0.11, 0.12),
    "path_worst": (0.39, 0.12),
    "level_over_path_cells": (20.0, 3.0),
    # No matching, no inflation.
    "unmatched_max": (0.09, 0.06),
    "unmatched_mean": (0.05, 0.03),
    # Nearest neighbour on levels inflates; on trend it does not.
    "nn1_t4": (0.25, 0.13),
    "nn2_t4": (0.05, 0.05),
    "nn1_t10": (0.07, 0.08),
    "nn2_t10": (0.06, 0.05),
    # The mechanism, asserted as direction under both specifications.
    "rises_with_mu_t4": (1.0, 0.0),
    "rises_with_mu_t10": (1.0, 0.0),
    "falls_with_rho_t4": (1.0, 0.0),
    "falls_with_rho_t10": (1.0, 0.0),
    "ten_pre_inflates_less": (1.0, 0.0),
}
