"""PDA against every cell of Shi & Huang's (2023) Table 1.

Table 1 is the paper's Monte Carlo: forward selection and the modified-BIC LASSO
on a four-factor panel of one treated unit and 100 donors, at ``T1 = T2 = 50``,
``100`` and ``200``, under two factor structures, reporting the number of
selected donors, the out-of-sample RMSPE, and the rejection rate of the
post-selection t-test under seven treatment processes ``D1``-``D7``. Twelve rows
of nine columns: 108 cells, and this case compares all of them.

The published paper carries the two factor blocks as one table. The i.i.d. block
and the dynamic block are separate studies with separate truncation lags, so
they are reported separately below and each is complete on its own.

Path B (scenario 3: the authors' full repository). The DGP is not inferred from
the paper's prose -- it is traced line by line to
``simulation/nonsparse/FS.simulation.dense.R`` in ``zhentaoshi/fsPDA`` at
``5f4542c`` and lives in
:func:`mlsynth.utils.pda_helpers.simulation.simulate_pda_panel`. The factor
loadings are their committed ``loading.RData``, carried in the bundle, so the
design matrix is theirs. The cells are compared against two references: the
table as printed (Shi & Huang 2023, J. Econometrics 234(2), p. 521) and the same
table as their own ``FS.R`` and ``lasso.BIC.R`` produce it, recorded in the
bundle at 2000 replications.

Where the prose and the code disagree
--------------------------------------
Three parameters of the DGP read differently in Section 4.1 and in the code the
table was run on. The code is what produced Table 1, so the code is what is
followed, and the differences are recorded because a reader working from the
paper alone would land somewhere else:

* Section 4.1 prints the i.i.d. factors as ``N(0, I^2)``. The code draws
  ``rnorm(Tn, sd = k * sigma.u)``, so factor ``k`` has standard deviation ``k``
  and the fourth carries sixteen times the variance of the first. The ``I`` is
  the index ``l`` in an upright font.
* The idiosyncratic shocks are ``N(0, 0.5^2)``: ``sigma.eta = 0.5`` is a
  standard deviation.
* Section 4.1 gives the irrelevant loadings as ``U(-0.1, 0.1)``. The committed
  ``loading.RData`` has them ``U(-0.5, 0.5)``, five times wider. This one is the
  paper's own subject: with loadings that wide the irrelevant donors carry real
  signal, the regression is dense, and no method can select its way to a sparse
  truth. At ``U(-0.1, 0.1)`` the design would be nearly sparse and the table
  would not be the table.

One fit per replication
-----------------------
The seven ``D`` columns of a row come from one fit. Both estimators select and
estimate on pre-treatment data only, so the post-period prediction error does
not depend on the treatment effect, and their ``FS.simulation.dense`` forms the
effect for DGP ``j`` as ``Delta_j + d`` and tests that. This case does the same:
one ``PDA(...).fit()`` on the untreated panel, then mlsynth's own
``fs_ate_inference`` / ``lasso_ate_inference`` on each of the seven treated
paths, at the truncation lag ``Run.simulation.dense.R`` pairs with that column.

``shortcut_matches_full_api`` is the check that this is allowed: on a sample of
replications the whole panel is rebuilt with ``Delta_j`` added to the treated
unit and put through ``PDA(...).fit()`` end to end, and the ATE, standard error
and rejection must come back identical.

What the cells say
------------------
The selection counts match on eleven of the twelve rows: the median number of
donors is the published integer for the LASSO everywhere (9/11/14 and 6/8/13)
and for forward selection everywhere except i.i.d. factors at ``T1 = 200``,
where it comes out at 8 against their 9. That row is a coin flip -- 55% of its
draws select 8 donors or fewer and the counts run from 5 to 13 -- so the median
sits on the boundary and lands either side of it. Nothing was tuned to get the
other eleven; the counts fall out of the two stopping rules on the reconstructed
design.

RMSPE matches to 0.005 for forward selection in all six rows and to 0.007 for
the LASSO in five of six. The sixth is the LASSO at ``T1 = 50`` under i.i.d.
factors, 0.930 against their 0.968, and the direction is the one
``fspda_dense_mc`` documents: ``lasso.BIC`` calls glmnet at its default
``thresh = 1e-7``, this design is ``p = 100`` against ``n = 50``, and mlsynth
attains the lower LASSO objective. A better fit is a smaller prediction error.

The rejection rates reproduce the paper's four claims: the LASSO takes in more
donors than forward selection, forward selection predicts better out of sample,
the LASSO's size inflates under dynamic factors while forward selection stays
the better-sized of the two at every length, and both are near-fully powered by
``T1 = 200`` against every alternative.

Runtime. 400 replications on each of six cells, two estimators, spread over up
to four processes -- about fifteen minutes on four cores and three quarters of
an hour on one. The paper's script runs 5000 and its text says 1000. The
replication count is set by the selection-count cells, which are the tight ones:
halving it to 200 costs three of the LASSO's six exact matches, because the
median of a discrete count needs the draws to settle.

Reference bundle: ``benchmarks/reference/fspda_table1/`` (regenerate with
``Rscript benchmarks/reference/fspda_table1/reference.R``).
"""
from __future__ import annotations

import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference
from mlsynth.utils.pda_helpers.simulation import (
    simulate_pda_panel,
    simulate_pda_shocks,
)

_CASE = "fspda_table1"
_LOADINGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "reference", _CASE, "loadings.tsv")

M = 400                      # replications per cell; the paper's script runs 5000
N_WORKERS = min(4, os.cpu_count() or 1)
N_DONORS = 100
T_GRID = (50, 100, 200)
BASE_SEED = 20250812

#: Truncation lags per D column, from ``Run.simulation.dense.R``'s h.iid/h.nnd.
H = {
    ("iid", 50): (1, 1, 2, 1, 1, 2, 2),
    ("iid", 100): (2, 1, 3, 1, 1, 3, 3),
    ("iid", 200): (2, 1, 4, 1, 1, 4, 4),
    ("nnd", 50): (1, 1, 3, 1, 1, 3, 3),
    ("nnd", 100): (2, 1, 4, 1, 1, 4, 4),
    ("nnd", 200): (4, 1, 7, 1, 1, 7, 7),
}

#: Shi & Huang (2023), J. Econometrics 234(2), Table 1, p. 521:
#: (median donors selected, RMSPE, rejection rate under D1..D7).
PAPER = {
    ("fs", "iid", 50): (6, 0.813, (.066, .066, .099, .785, 1.000, .664, .992)),
    ("fs", "iid", 100): (7, 0.710, (.059, .057, .084, .983, 1.000, .908, 1.000)),
    ("fs", "iid", 200): (9, 0.656, (.059, .055, .077, 1.000, 1.000, .995, 1.000)),
    ("fs", "nnd", 50): (6, 0.815, (.115, .087, .112, .756, .998, .636, .986)),
    ("fs", "nnd", 100): (7, 0.710, (.088, .070, .091, .975, 1.000, .892, 1.000)),
    ("fs", "nnd", 200): (8, 0.657, (.069, .059, .079, 1.000, 1.000, .994, 1.000)),
    ("lasso", "iid", 50): (9, 0.968, (.063, .067, .096, .724, .998, .622, .985)),
    ("lasso", "iid", 100): (11, 0.842, (.058, .057, .081, .964, 1.000, .881, 1.000)),
    ("lasso", "iid", 200): (14, 0.739, (.056, .055, .078, 1.000, 1.000, .993, 1.000)),
    ("lasso", "nnd", 50): (6, 1.046, (.244, .191, .180, .618, .940, .513, .899)),
    ("lasso", "nnd", 100): (8, 0.902, (.184, .146, .126, .870, .998, .775, .994)),
    ("lasso", "nnd", 200): (13, 0.746, (.116, .095, .089, 1.000, 1.000, .978, 1.000)),
}

_NULL_COLS = (0, 1, 2)       # D1-D3: the null holds, so these are size
_POWER_COLS = (3, 4, 5, 6)   # D4-D7: the null is false, so these are power


def _loadings() -> np.ndarray:
    """Their committed ``loading.RData``, 101 x 4, row 0 the treated unit."""
    return np.loadtxt(_LOADINGS)


def _shifted_df(sample, delta: np.ndarray) -> pd.DataFrame:
    """The sample's own frame with ``delta`` added to the treated post-periods."""
    df = sample.df.copy()
    post = (df["unit"] == "treated") & (df["treat"] == 1)
    df.loc[post, "y"] = df.loc[post, "y"].to_numpy() + delta
    return df


def _fit(df: pd.DataFrame, lag: int):
    from mlsynth import PDA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDA({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "methods": ["fs", "LASSO"], "fs_intercept": False,
            "lasso_criterion": "mbic", "lrvar_lag": lag, "display_graphs": False,
        }).fit()


def _test(method: str, y: np.ndarray, X: np.ndarray, fit, T1: int, lag: int):
    """mlsynth's own post-selection t-test on a treated path, at a fixed lag.

    Each estimator's own inference entry point, so that what this case measures
    is what the estimator reports and not a re-derivation of it.
    """
    from mlsynth.utils.pda_helpers.fs.inference import fs_ate_inference
    from mlsynth.utils.pda_helpers.lasso.inference import lasso_ate_inference
    cf = np.asarray(fit.counterfactual, dtype=float)
    if method == "fs":
        return fs_ate_inference(y, cf, T1, lrvar_lag=lag)
    support = np.abs(np.asarray(fit.beta, dtype=float)) > 1e-10
    return lasso_ate_inference(y, X, cf, support, T1, lrvar_lag=lag)


def _replication(dgp: str, T1: int, seed: int, loadings: np.ndarray):
    """One draw: the untreated panel, one fit, and seven tests per estimator."""
    rng = np.random.default_rng(seed)
    sample = simulate_pda_panel(N=N_DONORS, T1=T1, T2=T1,
                                dynamic_factors=(dgp == "nnd"), effect=0.0,
                                loadings=loadings, rng=rng)
    shocks = simulate_pda_shocks(T1, rng=rng)                 # (T2, 7)
    lags = H[(dgp, T1)]
    res = _fit(sample.df, lags[0])
    X = sample.Y_controls.T                                   # (T, N), label order
    out = {}
    for method in ("fs", "lasso"):
        fit = res.fits[method]
        gap = np.asarray(fit.gap, dtype=float)[T1:]
        phi = []
        for j in range(7):
            y_j = sample.Y_treated.copy()
            y_j[T1:] += shocks[:, j]
            _, _, _, p = _test(method, y_j, X, fit, T1, lags[j])
            phi.append(int(p < 0.05))
        out[method] = (len(fit.selected_donors),
                       float(np.sqrt(np.mean(gap ** 2))), phi)
    return out, sample, shocks, res, lags


def _audit(sample, shocks, res, lags, T1: int, column: int) -> float:
    """1.0 when the whole public API on the shifted panel agrees with the
    shortcut. The estimators use pre-period data only, so adding ``Delta`` to
    the treated unit must leave the fit alone and shift the gap by exactly
    ``Delta`` -- this is the assumption the seven-columns-from-one-fit design
    rests on, checked instead of assumed."""
    y_j = sample.Y_treated.copy()
    y_j[T1:] += shocks[:, column]
    shifted = _fit(_shifted_df(sample, shocks[:, column]), lags[column])
    ok = 1.0
    for method in ("fs", "lasso"):
        att, se, _, p = _test(method, y_j, sample.Y_controls.T,
                              res.fits[method], T1, lags[column])
        full = shifted.fits[method]
        ok = min(ok, float(np.isclose(full.att, att, rtol=1e-10, atol=1e-12)
                           and np.isclose(full.att_se, se, rtol=1e-10, atol=1e-12)
                           and (full.p_value < 0.05) == (p < 0.05)))
    return ok


def _job(task):
    """One replication, addressed by cell and seed. Runs in a worker process."""
    dgp, T1, seed = task
    out, *_ = _replication(dgp, T1, seed, _job.loadings)
    return dgp, T1, out


def _init(loadings: np.ndarray) -> None:
    _job.loadings = loadings


def _sweep(tasks, loadings):
    """Every replication, over a process pool when there is more than one core.

    Each replication is addressed by its own seed and touches nothing shared, so
    the results do not depend on the order they complete in or on the number of
    workers. A single-core machine takes the serial path and gets identical
    numbers.
    """
    if N_WORKERS <= 1:
        _init(loadings)
        return [_job(t) for t in tasks]
    with Pool(N_WORKERS, initializer=_init, initargs=(loadings,)) as pool:
        return list(pool.imap_unordered(_job, tasks, chunksize=4))


def run() -> dict:
    ref = load_reference(_CASE)["values"]
    loadings = _loadings()
    got: dict = {}

    tasks = [(dgp, T1, BASE_SEED + i)
             for dgp in ("iid", "nnd") for T1 in T_GRID for i in range(M)]
    acc: dict = {}
    for dgp, T1, out in _sweep(tasks, loadings):
        for m, (n_sel, rmspe, phi) in out.items():
            cell = acc.setdefault((m, dgp, T1), {"n": [], "rmspe": [], "phi": []})
            cell["n"].append(n_sel)
            cell["rmspe"].append(rmspe)
            cell["phi"].append(phi)
    for key, cell in acc.items():
        got[key] = (float(np.median(cell["n"])),
                    float(np.mean(cell["rmspe"])),
                    np.asarray(cell["phi"], dtype=float).mean(0))

    # The audits re-run a couple of replications in this process, cycling the D
    # column so all seven are covered across the six cells.
    audits: list = []
    for dgp in ("iid", "nnd"):
        for T1 in T_GRID:
            for i in range(2):
                _, sample, shocks, res, lags = _replication(
                    dgp, T1, BASE_SEED + i, loadings)
                audits.append(_audit(sample, shocks, res, lags, T1,
                                     len(audits) % 7))

    def _dev(method: str, which: str, cols=None) -> list:
        out = []
        for (m, dgp, T1), (n_sel, rmspe, phi) in got.items():
            if m != method:
                continue
            p_n, p_r, p_phi = PAPER[(m, dgp, T1)]
            if which == "n":
                out.append(n_sel - p_n)
            elif which == "rmspe":
                out.append(rmspe - p_r)
            else:
                out += [phi[c] - p_phi[c] for c in cols]
        return out

    result = {
        "n_cells": float(len(got) * 9),
        "shortcut_matches_full_api": float(np.mean(audits)),
    }
    for m in ("fs", "lasso"):
        result[f"{m}_nsel_exact_match_rate"] = float(
            np.mean([d == 0 for d in _dev(m, "n")]))
        result[f"{m}_max_abs_nsel_dev"] = float(max(abs(d) for d in _dev(m, "n")))
        result[f"{m}_max_abs_rmspe_dev"] = float(max(abs(d) for d in _dev(m, "rmspe")))
        result[f"{m}_max_abs_size_dev"] = float(
            max(abs(d) for d in _dev(m, "phi", _NULL_COLS)))
        result[f"{m}_max_abs_power_dev"] = float(
            max(abs(d) for d in _dev(m, "phi", _POWER_COLS)))
        result[f"{m}_rms_rejection_dev"] = float(np.sqrt(np.mean(
            np.asarray(_dev(m, "phi", _NULL_COLS + _POWER_COLS)) ** 2)))

    # ... and against their own code at 2000 replications on the same design.
    theirs = []
    for (m, dgp, T1), (n_sel, rmspe, phi) in got.items():
        tag = f"{m}_{dgp}_{T1}"
        theirs.append(("nsel", n_sel - ref[f"{tag}_nsel"]))
        theirs.append(("rmspe", rmspe - ref[f"{tag}_rmspe"]))
        theirs += [("phi", phi[j] - ref[f"{tag}_D{j + 1}"]) for j in range(7)]
    result["vs_reference_nsel_exact_match_rate"] = float(
        np.mean([d == 0 for k, d in theirs if k == "nsel"]))
    result["vs_reference_max_abs_nsel_dev"] = float(
        max(abs(d) for k, d in theirs if k == "nsel"))
    result["vs_reference_max_abs_rmspe_dev"] = float(
        max(abs(d) for k, d in theirs if k == "rmspe"))
    result["vs_reference_rms_rejection_dev"] = float(np.sqrt(np.mean(
        [d ** 2 for k, d in theirs if k == "phi"])))

    # The paper's four qualitative claims, as indicators (1.0 == holds).
    result["lasso_over_selects"] = float(all(
        got[("lasso", d, t)][0] >= got[("fs", d, t)][0]
        for d in ("iid", "nnd") for t in T_GRID))
    result["fs_rmspe_below_lasso"] = float(all(
        got[("fs", d, t)][1] < got[("lasso", d, t)][1]
        for d in ("iid", "nnd") for t in T_GRID))
    result["fs_better_sized_than_lasso_under_dynamic_factors"] = float(all(
        got[("fs", "nnd", t)][2][0] < got[("lasso", "nnd", t)][2][0]
        for t in T_GRID))
    result["lasso_size_inflates_under_dynamic_factors"] = float(all(
        got[("lasso", "nnd", t)][2][0] > got[("lasso", "iid", t)][2][0]
        for t in T_GRID))
    result["both_fully_powered_at_the_longest_panel"] = float(all(
        got[(m, d, 200)][2][c] >= 0.95
        for m in ("fs", "lasso") for d in ("iid", "nnd") for c in _POWER_COLS))
    return result


# Deterministic: every replication is seeded from BASE_SEED, forward selection is
# a greedy search followed by OLS, and the LASSO path at a fixed penalty is
# strictly convex. Re-running returns identical numbers, and so does running on a
# different number of cores -- each replication is addressed by its own seed.
#
# Tolerances. The paper's script runs 5000 replications and its text says 1000;
# this runs 400, so every rejection cell carries binomial Monte Carlo noise of
# sqrt(p(1-p)/400) -- 0.012 at p = 0.06, 0.025 at p = 0.5, 0 at p = 1. The
# published cells carry their own, an order smaller. Each bound below is about
# twice the observed deviation, which is where three of those standard errors
# land for the worst cell in its group.
#
# The selection counts are the tight cells. Eleven of the twelve medians are the
# published integer; the twelfth is forward selection under i.i.d. factors at
# T1 = 200, which comes out at 8 against their 9. That row is a coin flip and not
# a disagreement: 55% of its 400 draws select 8 donors or fewer, and the counts
# run from 5 to 13. So what gets pinned is the deviation in donors -- no median
# may sit more than one donor from the published integer -- with the exact-match
# rates carried alongside, centred where they land and given one row of slack.
#
# RMSPE: forward selection is within 0.005 of the published column in all six
# rows. The LASSO's bound is set by one row, i.i.d. at T1 = 50, 0.930 against
# 0.968, where glmnet's default tolerance leaves their fit short of the optimum
# and mlsynth's prediction error is the smaller of the two. fspda_dense_mc
# measures that effect directly at the panel level.
#
# The RMS rejection deviations are the aggregate cells and they concentrate: 42
# cells each with an independent binomial error puts the RMS near 0.01 even
# though individual maxima reach 0.03.
#
# vs_reference_* compare against their own FS.R and lasso.BIC.R at 2000
# replications on the same design, recorded in the bundle. Those bounds are
# tighter than the published ones because only one side then carries appreciable
# Monte Carlo noise.
#
# shortcut_matches_full_api is exact by construction, so it carries no tolerance:
# either the estimator ignores the post-period outcome or the whole
# seven-columns-from-one-fit design is invalid. The five indicators are the
# paper's qualitative claims and are pinned the same way.
EXPECTED = {
    "n_cells": (108.0, 0.0),
    "shortcut_matches_full_api": (1.0, 0.0),

    "fs_nsel_exact_match_rate": (0.833, 0.167),       # 5/6
    "fs_max_abs_nsel_dev": (0.0, 1.0),                # one donor, on one row
    "fs_max_abs_rmspe_dev": (0.0, 0.02),              # observed 0.005
    "fs_max_abs_size_dev": (0.0, 0.05),               # observed 0.026
    "fs_max_abs_power_dev": (0.0, 0.06),              # observed 0.029
    "fs_rms_rejection_dev": (0.0, 0.03),              # observed 0.011

    "lasso_nsel_exact_match_rate": (1.0, 0.167),      # 6/6
    "lasso_max_abs_nsel_dev": (0.0, 1.0),             # observed 0
    "lasso_max_abs_rmspe_dev": (0.0, 0.06),           # observed 0.038, glmnet
    "lasso_max_abs_size_dev": (0.0, 0.05),            # observed 0.024
    "lasso_max_abs_power_dev": (0.0, 0.06),           # observed 0.033
    "lasso_rms_rejection_dev": (0.0, 0.03),           # observed 0.013

    "vs_reference_nsel_exact_match_rate": (0.917, 0.167),   # 11/12
    "vs_reference_max_abs_nsel_dev": (0.0, 1.0),
    "vs_reference_max_abs_rmspe_dev": (0.0, 0.05),    # observed 0.029
    "vs_reference_rms_rejection_dev": (0.0, 0.03),    # observed 0.015

    "lasso_over_selects": (1.0, 0.0),
    "fs_rmspe_below_lasso": (1.0, 0.0),
    "fs_better_sized_than_lasso_under_dynamic_factors": (1.0, 0.0),
    "lasso_size_inflates_under_dynamic_factors": (1.0, 0.0),
    "both_fully_powered_at_the_longest_panel": (1.0, 0.0),
}
