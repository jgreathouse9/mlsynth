"""Feng's Kansas application: ingestion, both arms, and the tuning sensitivity.

The paper's Section 6.1 compares local PCA against a simplex synthetic control
on the 2012 Kansas tax cut. This module rebuilds both arms from
``basedata/kansas_taxcut.csv`` and exposes the pieces the spike measures:
the two counterfactual paths, the v1 and 2024 variants of the SC arm, and how
the answer moves with the two tuning constants the paper fixes by fiat.
"""

from __future__ import annotations

import os
import sys

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpca_oracle import knn_index, lpca_row  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "basedata", "kansas_taxcut.csv")

# Feng_LPCA_app.R, lines 36-38: K = round(n^(2/3)), p1 = 40, nlam = 3.
N_MATCH_PERIODS = 40
MAX_COMPONENTS = 3


def kansas_matrices():
    """Ingest the panel and reproduce the R script's preprocessing.

    Returns
    -------
    dict
        ``x`` is the demeaned, first-differenced panel with the treated unit's
        post-treatment cells zeroed -- the matrix both arms consume. ``col_mean``
        is the per-period mean that ``x`` was centred by, which the counterfactual
        has to be put back on before it can be compared to ``observed``.
    """
    from mlsynth.utils.datautils import dataprep

    prepped = dataprep(
        pd.read_csv(os.path.abspath(_DATA)),
        unit_id_column_name="state",
        time_period_column_name="year_qtr",
        outcome_column_name="lngdpcapita",
        treatment_indicator_column_name="treated",
    )
    wide = prepped["Ywide"]
    units = list(wide.columns)
    treated = units.index(prepped["treated_unit_name"])
    n_pre = prepped["pre_periods"]

    levels = np.array(wide.to_numpy(dtype=float).T, copy=True)   # (50, 105)
    observed = np.diff(levels[treated]) * 100.0                  # raw treated growth

    # R: x[kansas$treated == 1, 2] <- NA before the first difference, so every
    # difference that touches a treated quarter goes missing.
    masked = levels.copy()
    masked[treated, n_pre:] = np.nan
    x = np.diff(masked, axis=1) * 100.0
    col_mean = np.nanmean(x, axis=0)
    x = x - col_mean

    post_start = n_pre - 1            # first differenced period touched by treatment
    x[treated, post_start:] = 0.0     # R: x[id, T0:p] <- 0

    return {
        "x": x, "col_mean": col_mean, "observed": observed,
        "treated": treated, "post_start": post_start,
        "unit_names": units, "levels": levels,
        "time": np.asarray(wide.index, dtype=float)[1:],
    }


def lpca_arm(m, n_neighbours=None, n_match=N_MATCH_PERIODS,
             n_components=MAX_COMPONENTS):
    """Local PCA counterfactual on the outcome's scale, plus its diagnostics."""
    x, treated = m["x"], m["treated"]
    n = x.shape[0]
    K = int(round(n ** (2 / 3))) if n_neighbours is None else int(n_neighbours)

    keep = knn_index(x[:, :n_match], K)
    fitted, rank = lpca_row(x[:, n_match:], keep[:, treated], treated, K,
                            n_components=n_components)
    return {
        "path": fitted + m["col_mean"][n_match:],   # re-centred; see README
        "offset": n_match, "K": K, "rank": rank,
        "neighbours": [m["unit_names"][i] for i in np.flatnonzero(keep[:, treated])],
    }


def sc_arm(m, n_match=N_MATCH_PERIODS):
    """Simplex synthetic control, returned in both the v1 and 2024 conventions.

    R reference: ``sc.pred``. ``recentred`` adds ``col.mean`` back, as the
    2024 script does; ``demeaned`` omits it, as the November 2023 script did.
    """
    x, treated, post_start = m["x"], m["treated"], m["post_start"]
    donors = np.delete(x, treated, axis=0)
    w = cp.Variable(donors.shape[0])
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x[treated, :post_start]
                                   - donors[:, :post_start].T @ w)),
        [cp.sum(w) == 1, w >= 0],
    )
    problem.solve(solver=cp.CLARABEL)
    demeaned = donors.T @ w.value
    return {
        "demeaned": demeaned[n_match:],
        "recentred": (demeaned + m["col_mean"])[n_match:],
        "offset": n_match, "weights": w.value,
        "pre_sse": float(problem.value),
        "unit_names": [u for i, u in enumerate(m["unit_names"]) if i != treated],
    }


def post_mean(path, offset, m):
    """Mean of a counterfactual path over the 16 post-treatment quarters."""
    lo = m["post_start"] - offset
    return float(np.mean(path[lo:lo + (len(m["observed"]) - m["post_start"])]))


def gap(path, offset, m):
    """Observed minus counterfactual, averaged over the post-treatment periods.

    The paper's sign convention: a negative value means the counterfactual sits
    above the observed series, i.e. the policy is estimated to have cost growth.
    """
    return float(np.mean(m["observed"][m["post_start"]:])) - post_mean(path, offset, m)
