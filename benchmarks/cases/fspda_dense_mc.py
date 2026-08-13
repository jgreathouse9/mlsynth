"""PDA cross-validation: forward selection and modified-BIC LASSO against fsPDA.

Shi & Huang (2023) release the code behind their dense ("nonsparse") Monte Carlo
as ``zhentaoshi/fsPDA``. This case runs mlsynth's ``PDA`` estimator, through its
public API, on the panels their own generator produces, and compares the answer
to what their ``FS()`` and ``lasso.BIC()`` return on those same panels.

Path: cross-validation against an authoritative reference implementation. What
makes it a cross-validation and not a re-derivation is that nothing about the
data is ported. The reference script sources their ``FS.R`` and ``lasso.BIC.R``
verbatim, generates the panels with the DGP block of their
``FS.simulation.dense.R``, and reads the factor loadings from the
``loading.RData`` committed in their repository. mlsynth never sees the DGP --
only the finished panel, handed to ``PDA(...).fit()`` as a long DataFrame.

The design is their smallest grid cell: ``T1 = T2 = 50``, one treated unit, 100
donors, four factors. Their ``Run.simulation.dense.R`` pairs that cell with
``h = 1`` for the long-run variance, ``H = 1`` for forward selection's modified
BIC and ``H = 2`` for the LASSO's, so the fits below are configured to match
(``fs_intercept=False``, ``lrvar_lag=1``, ``lasso_criterion="mbic"``).

What is pinned, and what glmnet's default tolerance costs
---------------------------------------------------------
Forward selection matches outright. On all eight panels mlsynth selects the same
donors, and the coefficients, the ATE, the pre-period R-squared and the
t-statistic all agree to about ``1e-13``. The t-statistic is the part that took a
fix: mlsynth's long-run variance divided the lag-1 autocovariance by ``n - 1``
where ``acf(type = "covariance")`` divides by ``n``, which was wrong in the third
decimal of Z until it was corrected.

The LASSO needs the comparison split in two, and the split is the finding.

Against their published call, mlsynth agrees on the selected donors on five of
the eight panels and on the penalty on four. Against the same function with
glmnet converged, it agrees on both on all eight, with coefficients to ``2e-05``
and the ATE to ``9e-07``.

The gap is glmnet's default tolerance, not the port. ``lasso.BIC`` calls glmnet
at ``thresh = 1e-7``, and this design is ``p = 100`` against ``n = 50``, where
that default stops short: on the first dynamic panel glmnet's coefficients sit
``1.2e-02`` from the converged solution, and mlsynth's attain a *lower* value of
the LASSO objective than glmnet's do (2.103448099730 against 2.103482823094).
Since the criterion scores those coefficients, an under-converged fit can hand a
different grid point the minimum, which is what moves the penalty on the panels
where it moves.

Both runs are in the reference bundle so the two claims stay separable: a future
change that breaks the port would move the converged cells, while the published
cells stay where they are because they measure something about glmnet.

Their third column, the synthetic control, is covered too. ``scm.R`` solves
``min ||y1 - Y1 b||^2`` over the simplex with CVXR's ``ECOS_BB``; CVXR does not
install in this environment (it now needs ``scs``, ``osqp``, ``clarabel`` -- a
Rust package -- and ``S7``, none in apt and none buildable from the CRAN mirror
here), so the reference reproduces the program exactly and solves it with
``quadprog``. mlsynth's counterpart is ``VanillaSC``, not a PDA variant, and it
attains the same optimum: the objective agrees to ``1e-11`` and the two pick the
same three donors with weights summing to one. The objective is what is
compared, since it is the quantity the program determines; it is stable to
``1e-9`` across ridges from ``1e-6`` to ``1e-10``, so the solver substitution
does not decide the answer.

The eight panels are enough to cross-validate the estimators cell by cell, which
is what this case is for; the aggregate size and power geometry over hundreds of
replications is ``pda_table1``. The replication count is bounded by storage --
each panel is a 100 x 101 matrix of doubles carried in the bundle so the
comparison needs no R at run time.

Reference bundle: ``benchmarks/reference/fspda_dense_mc/`` (regenerate with
``Rscript benchmarks/reference/fspda_dense_mc/reference.R``).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_CASE = "fspda_dense_mc"
_PANELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reference", _CASE, "panels.tsv")

CELLS = [f"{dgp}_s{s}" for dgp in ("iid", "nnd") for s in range(1, 5)]


def _panels() -> dict:
    """``{cell: (y, Y)}`` from the bundle's panel file."""
    out: dict = {}
    df = pd.read_csv(_PANELS, sep="\t")
    for cell, block in df.groupby("cell", sort=False):
        block = block.sort_values("t")
        y = block["y"].to_numpy(dtype=float)
        Y = np.array([np.fromstring(r, sep=" ") for r in block["donors"]])
        out[str(cell)] = (y, Y)
    return out


def _long(y: np.ndarray, Y: np.ndarray, T1: int) -> pd.DataFrame:
    """The panel as mlsynth's public API takes it: one long DataFrame."""
    T, N = Y.shape
    recs = [{"unit": "treated", "time": t + 1, "y": float(y[t]),
             "treat": int(t >= T1)} for t in range(T)]
    recs += [{"unit": f"d{j + 1}", "time": t + 1, "y": float(Y[t, j]), "treat": 0}
             for t in range(T) for j in range(N)]
    return pd.DataFrame(recs)


def _fit_scm(df: pd.DataFrame):
    """Their third column: the plain simplex-constrained synthetic control."""
    from mlsynth import VanillaSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "display_graphs": False,
        }).fit()


def _fit(df: pd.DataFrame, lrvar_lag: int):
    from mlsynth import PDA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDA({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "methods": ["fs", "LASSO"],
            "fs_intercept": False, "lrvar_lag": lrvar_lag,
            "lasso_criterion": "mbic", "display_graphs": False,
        }).fit()


def _selected(fit) -> list:
    """1-based donor indices, matching R's column numbering."""
    return sorted(int(str(u)[1:]) for u in fit.selected_donors)


def _beta_by_donor(fit, labels) -> dict:
    """``{1-based donor index: coefficient}``.

    ``dataprep`` orders donors by their label as a string, so the panel's column
    ``j`` is not position ``j - 1`` of ``beta`` -- with a hundred donors named
    ``d1..d100`` the string order opens ``d1, d10, d100, d11``. The coefficients
    have to be looked up through the label the estimator reports, not by
    position, which is why this map exists instead of a slice.
    """
    return {int(str(lab)[1:]): float(b) for lab, b in zip(labels, fit.beta)}


def _ref_vec(values: dict, prefix: str, n: int) -> list:
    return [values[f"{prefix}_k{i + 1}"] for i in range(n)]


def run() -> dict:
    ref = load_reference(_CASE)["values"]
    T1 = int(ref["T1"])
    lag = int(ref["lrvar_lag"])
    panels = _panels()

    fs_set_ok = fs_ate_err = fs_z_err = fs_coef_err = fs_r2_err = 0.0
    las_set_ok = las_lam_ok = las_coef_err = las_ate_err = las_z_err = 0.0
    las_phi_ok = 0.0
    las_cv_set_ok = las_cv_lam_ok = 0.0
    scm_obj_rel = scm_set_ok = scm_wsum_err = 0.0
    n = len(CELLS)

    for cell in CELLS:
        y, Y = panels[cell]
        long_df = _long(y, Y, T1)
        r = _fit(long_df, lag)
        fs, las = r.fits["fs"], r.fits["lasso"]
        labels = r.inputs.donor_labels

        # --- forward selection: their FS(), verbatim ------------------------
        r_sel = [int(v) for v in _ref_vec(ref, f"fs_sel_{cell}", int(ref[f"fs_R_{cell}"]))]
        fs_set_ok += float(_selected(fs) == r_sel)
        fs_ate_err = max(fs_ate_err, abs(fs.att - ref[f"fs_ATE_{cell}"]))
        fs_z_err = max(fs_z_err, abs(fs.att / fs.att_se - ref[f"fs_Z_{cell}"]))
        fs_r2_err = max(fs_r2_err, abs(
            float(np.var(fs.counterfactual[:T1]) / np.var(y[:T1]))
            - ref[f"fs_RSquared_{cell}"]))
        fs_beta = _beta_by_donor(fs, labels)
        r_coef = _ref_vec(ref, f"fs_coef_{cell}", len(r_sel))
        fs_coef_err = max(fs_coef_err, max(
            abs(fs_beta[j] - c) for j, c in zip(r_sel, r_coef)))

        # --- LASSO: support and penalty vs their published call ------------
        n_las = int(ref[f"las_R_{cell}"])
        r_sel_l = [int(v) for v in _ref_vec(ref, f"las_sel_{cell}", n_las)]
        las_set_ok += float(_selected(las) == r_sel_l)
        las_lam_ok += float(abs(las.metadata["alpha"] - ref[f"las_lambda_{cell}"]) < 1e-12)

        # --- LASSO coefficients vs their call with glmnet converged --------
        # Only where the converged run agrees on the support, since a
        # coefficient comparison across different models is not a comparison.
        n_cv = int(ref[f"lascv_R_{cell}"])
        cv_sel = [int(v) for v in _ref_vec(ref, f"lascv_sel_{cell}", n_cv)]
        las_cv_set_ok += float(_selected(las) == cv_sel)
        las_cv_lam_ok += float(
            abs(las.metadata["alpha"] - ref[f"lascv_lambda_{cell}"]) < 1e-12)
        if _selected(las) == cv_sel:
            las_beta = _beta_by_donor(las, labels)
            cv_coef = _ref_vec(ref, f"lascv_coef_{cell}", n_cv)
            las_coef_err = max(las_coef_err, max(
                abs(las_beta[j] - c) for j, c in zip(cv_sel, cv_coef)))
            las_ate_err = max(las_ate_err, abs(las.att - ref[f"lascv_ATE_{cell}"]))
            # The t-statistic, which is the ATE and the variance convention
            # together. It is what the pinned quantities above cannot see: a
            # standard error built from a different formula leaves the support,
            # the penalty, the coefficients and the ATE all matching.
            las_z_err = max(las_z_err,
                            abs(las.att / las.att_se - ref[f"lascv_Z_{cell}"]))
            las_phi_ok += float(int(las.p_value < 0.05)
                                == int(ref[f"lascv_phi_{cell}"]))

        # --- synthetic control: their scm.R program -------------------------
        sc = _fit_scm(long_df)
        w = np.zeros(Y.shape[1])
        for lab, v in sc.weights.donor_weights.items():
            w[int(str(lab)[1:]) - 1] = float(v)
        obj = float(np.sum((y[:T1] - Y[:T1] @ w) ** 2))
        scm_obj_rel = max(scm_obj_rel, abs(obj / ref[f"scm_obj_{cell}"] - 1.0))
        scm_wsum_err = max(scm_wsum_err, abs(float(w.sum()) - 1.0))
        their_sel = [int(v) for v in
                     _ref_vec(ref, f"scm_sel_{cell}", int(ref[f"scm_R_{cell}"]))]
        mine = sorted(int(j) + 1 for j in np.flatnonzero(w > 1e-6))
        scm_set_ok += float(mine == their_sel)

    return {
        "n_cells": float(n),
        "fs_selection_match_rate": fs_set_ok / n,
        "fs_max_abs_coef_err": fs_coef_err,
        "fs_max_abs_ate_err": fs_ate_err,
        "fs_max_abs_z_err": fs_z_err,
        "fs_max_abs_r2_err": fs_r2_err,
        "lasso_selection_match_rate_vs_published": las_set_ok / n,
        "lasso_lambda_match_rate_vs_published": las_lam_ok / n,
        "lasso_selection_match_rate_vs_converged": las_cv_set_ok / n,
        "lasso_lambda_match_rate_vs_converged": las_cv_lam_ok / n,
        "lasso_max_abs_coef_err_vs_converged": las_coef_err,
        "lasso_max_abs_ate_err_vs_converged": las_ate_err,
        "lasso_max_abs_z_err_vs_converged": las_z_err,
        "lasso_rejection_match_rate_vs_converged": las_phi_ok / n,
        "scm_max_rel_objective_err": scm_obj_rel,
        "scm_selection_match_rate": scm_set_ok / n,
        "scm_max_abs_weight_sum_err": scm_wsum_err,
    }


# Forward selection is a deterministic greedy search followed by OLS, and the
# LASSO path at a fixed penalty is a strictly convex problem, so every cell below
# is an exact re-run -- no seeds, no solver randomness.
#
# The fs tolerances are the tight ones because nothing separates the two
# implementations: same stopping rule, same no-intercept OLS, same fixed-lag
# Bartlett variance. Observed error is ~1e-13; 1e-8 leaves room for BLAS
# reassociation and nothing else.
#
# The converged-LASSO coefficient tolerance is 1e-3 against an observed 1.7e-05.
# Two different stopping criteria are being compared -- glmnet's thresh on the
# deviance against scikit-learn's tol on the coordinate update -- so where the
# two descents halt is the one quantity here that can drift with a different
# BLAS or scikit-learn build.
#
# The match rates carry one cell of slack (0.125 = 1/8). The penalty grid is
# spaced 0.01 apart and the criterion is genuinely close between neighbouring
# grid points on some panels -- that closeness is what lets glmnet's default
# tolerance move the answer at all -- so a single cell flipping on another
# platform is not a regression. A port regression would take the rate to the
# floor, not shave one cell off it.
#
# The published-call rates are pinned at what they are, not at 1.0. They measure
# how far glmnet's default tolerance moves its own answer on this design, and
# asserting 1.0 there would assert something untrue about glmnet.
EXPECTED = {
    "n_cells": (8.0, 0.0),
    "fs_selection_match_rate": (1.0, 0.0),                 # same donors, all 8
    "fs_max_abs_coef_err": (0.0, 1e-8),
    "fs_max_abs_ate_err": (0.0, 1e-8),
    "fs_max_abs_z_err": (0.0, 1e-8),                       # the acf normalisation
    "fs_max_abs_r2_err": (0.0, 1e-8),
    "lasso_selection_match_rate_vs_published": (0.625, 0.125),   # 5/8
    "lasso_lambda_match_rate_vs_published": (0.5, 0.125),        # 4/8
    "lasso_selection_match_rate_vs_converged": (1.0, 0.125),     # 8/8
    "lasso_lambda_match_rate_vs_converged": (1.0, 0.125),        # 8/8
    "lasso_max_abs_coef_err_vs_converged": (0.0, 1e-3),
    "lasso_max_abs_ate_err_vs_converged": (0.0, 1e-4),
    # The t-statistic carries the ATE and the variance convention together, and
    # it is the cell that separates their studentization from Li & Bell's: with
    # the first-stage term in the sum and the lag chosen automatically, every
    # other LASSO cell above still matched and this one sat 4.5e-01 out.
    # Observed 7.4e-06 against their converged call; 1e-3 is the ATE tolerance
    # divided by the smallest standard error on these panels.
    "lasso_max_abs_z_err_vs_converged": (0.0, 1e-3),
    "lasso_rejection_match_rate_vs_converged": (1.0, 0.0),
    # The SCM objective is the quantity their program determines, and two
    # different convex solvers agree on it to 1e-11 here; 1e-6 covers the ridge
    # quadprog needs plus OSQP-vs-quadprog slack. The weights sum to one by
    # constraint, so 1e-8 there is a feasibility check, not a fit check.
    "scm_max_rel_objective_err": (0.0, 1e-6),
    "scm_selection_match_rate": (1.0, 0.125),
    "scm_max_abs_weight_sum_err": (0.0, 1e-8),
}
