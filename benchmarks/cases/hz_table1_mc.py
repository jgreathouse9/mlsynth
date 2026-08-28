"""Hsiao & Zhou (2024) Table 1 -- DGP1, Case 1, and whether their ordering survives.

Hsiao, C. and Zhou, Q. (2024), *Panel treatment effects measurement: Factor or
linear projection modelling?*, Journal of Applied Econometrics 39(7):1332-1358,
`10.1002/jae.3081 <https://doi.org/10.1002/jae.3081>`_.

Path B. The paper's Propositions 1-3 rank a linear projection (LP) below a
principal-component factor predictor (FB) on mean square prediction error, and
Table 1 reports the ranking holding in all twelve cells of its first design.
The replication package ships no code, so this rebuilds the design from
Section 6: DGP1 (eq. 43) draws ``alpha_i ~ U(0,2)`` and ``lambda ~ N(0,1)``,
Case 1 draws the factors ``chi2(1)`` and the errors ``chi2(1) - 1``, and
``r = floor(N^(1/3))``, with ``T`` in {10,30,60}, ``m = 5`` post-periods, and
control counts in {10,30,50,100}. FB is given the true ``r``, which is what the
concluding section means by the factor dimension being known a priori. Note 9's
LASSO screen caps the LP at ``T/2`` donors, note 11's Box-Jenkins baseline is an
AR(1) through the origin, and eqs. (35)-(37)'s prediction averaging runs only
where the paper's own dashes say it does, ``N - 1 > T``.

What the run establishes
------------------------
Their LP column reproduces and their FB column does not. Across the twelve cells
the LP MSE lands within roughly a tenth of the printed value at the median,
while the FB MSE is about half the printed value -- and it is low in every cell,
not scattered, which is a specification gap and not Monte Carlo noise. The same
one-sided gap appears in the empirical section: see ``hz_germany``, where the
FB path plotted in their Figure 1 misses the observed series in sample by 54
times what a correct principal-component fit misses by.

The consequence is that the ordering reverses. Given a factor predictor that is
not handicapped, LP wins a small minority of the twelve cells, not all twelve,
and prediction averaging beats FB in a minority of the six cells where it is
defined, against Proposition 3. Two of the paper's claims do
survive and are asserted as such: both predictors beat the univariate baseline
everywhere, and prediction averaging beats the plain LP in five of six cells,
reproducing even the sign of the exception their own table shows at
``N - 1 = 100, T = 30``.

``R`` is 300 here against the paper's 1000, which is enough for the orderings
and for a median relative error read to a tenth; the tolerances are set for that
budget. A 1000-replication run gave LP median 9.5% and FB median 49.0%, with LP
ahead of FB in 3 of 12 cells and prediction averaging ahead of FB in 2 of 6.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LassoLarsIC

R = 300
M = 5
CELLS = [(nd, T) for nd in (10, 30, 50, 100) for T in (10, 30, 60)]

# Table 1, page 1343: MSE per method. ``None`` is a printed dash.
PAPER_MSE = {
    (10, 10):  {"LP": 6.979, "LP_ave": None,  "FB": 10.165, "B-J": 67.239},
    (10, 30):  {"LP": 4.005, "LP_ave": None,  "FB": 6.269,  "B-J": 6.061},
    (10, 60):  {"LP": 3.213, "LP_ave": None,  "FB": 4.697,  "B-J": 6.624},
    (30, 10):  {"LP": 7.005, "LP_ave": 5.223, "FB": 8.754,  "B-J": 27.625},
    (30, 30):  {"LP": 4.154, "LP_ave": None,  "FB": 5.317,  "B-J": 9.065},
    (30, 60):  {"LP": 2.997, "LP_ave": None,  "FB": 4.036,  "B-J": 8.128},
    (50, 10):  {"LP": 7.825, "LP_ave": 5.643, "FB": 11.714, "B-J": 12.296},
    (50, 30):  {"LP": 4.600, "LP_ave": 3.266, "FB": 8.232,  "B-J": 11.629},
    (50, 60):  {"LP": 3.363, "LP_ave": None,  "FB": 5.752,  "B-J": 10.042},
    (100, 10): {"LP": 8.562, "LP_ave": 6.654, "FB": 15.065, "B-J": 74.890},
    (100, 30): {"LP": 4.228, "LP_ave": 4.662, "FB": 9.151,  "B-J": 11.574},
    (100, 60): {"LP": 3.399, "LP_ave": 2.445, "FB": 7.977,  "B-J": 12.613},
}

EXPECTED = {
    # their LP column reproduces
    "lp_median_rel_err": (0.10, 0.09),
    # their FB column does not, and the gap is one-sided in every cell
    "fb_median_rel_err": (0.49, 0.15),
    "fb_cells_where_ours_is_better": (12.0, 0.0),
    # the ordering Propositions 1-2 assert, against a competent FB
    "lp_beats_fb_cells": (3.0, 2.0),          # the paper reports 12 of 12
    # Proposition 3, on the six cells where prediction averaging is defined
    "lp_ave_beats_fb_cells": (2.0, 2.0),      # the paper reports 6 of 6
    # what does survive
    "both_beat_bj_cells": (12.0, 0.0),
    "lp_ave_beats_lp_cells": (5.0, 1.0),      # the paper's own table also gives 5 of 6
}


def _ols(X, y, Xp):
    A = np.column_stack([np.ones(len(X)), X])
    B = np.column_stack([np.ones(len(Xp)), Xp])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return B @ c


def _lp(Ypre, Ypost, T):
    """Eq. (16)/(32) behind note 9's LASSO screen, capped at ``T/2`` donors."""
    X, Xp, y = Ypre[:, 1:], Ypost[:, 1:], Ypre[:, 0]
    cap = max(1, T // 2)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    kw = {"noise_variance": float(np.var(y)) / 2} if Xs.shape[1] >= len(y) - 1 else {}
    m = LassoLarsIC("bic", max_iter=20000, **kw).fit(Xs, y)
    if not np.flatnonzero(m.coef_).size:
        return np.full(len(Xp), y.mean())
    return ((Xp - mu) / sd) @ m.coef_ + m.intercept_


def _fb(Ypre, Ypost, r):
    """Eqs. (30)-(31); invariant to which of the paper's normalisations is used."""
    _, _, Vt = np.linalg.svd(Ypre, full_matrices=False)
    r = min(r, Vt.shape[0], Ypre.shape[1] - 1)
    Lam = Vt[:r].T
    lam1, Lc = Lam[0], Lam[1:]
    return Ypost[:, 1:] @ Lc @ np.linalg.inv(Lc.T @ Lc).T @ lam1


def _lp_ave(Ypre, Ypost, G, rng):
    """Eqs. (35)-(37): a random partition into G subgroups, forecasts averaged."""
    nd = Ypre.shape[1] - 1
    return np.mean([_ols(Ypre[:, g], Ypre[:, 0], Ypost[:, g])
                    for g in np.array_split(rng.permutation(nd) + 1, G) if g.size],
                   axis=0)


def _bj(Ypre, m):
    """Note 11: ``yhat_{T+h} = phi_hat^h y_T``."""
    y = Ypre[:, 0]
    phi = float(y[:-1] @ y[1:] / (y[:-1] @ y[:-1]))
    return phi ** np.arange(1, m + 1) * y[-1]


def _cell(nd, T, seed):
    N = nd + 1
    r = int(np.floor(N ** (1 / 3)))
    G = 20 if T == 10 else 5
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("LP", "LP_ave", "FB", "B-J")}
    for _ in range(R):
        alpha = rng.uniform(0, 2, N)
        lam = rng.standard_normal((N, r))
        f = rng.chisquare(1, (T + M, r))
        u = rng.chisquare(1, (T + M, N)) - 1.0
        Y = alpha + f @ lam.T + u
        Ypre, Ypost = Y[:T], Y[T:]
        truth = Ypost[:, 0]
        acc["LP"].append(truth - _lp(Ypre, Ypost, T))
        acc["FB"].append(truth - _fb(Ypre, Ypost, r))
        acc["B-J"].append(truth - _bj(Ypre, M))
        if nd > T:
            acc["LP_ave"].append(truth - _lp_ave(Ypre, Ypost, G, rng))
    return {k: (float(np.mean(np.concatenate(v) ** 2)) if v else None)
            for k, v in acc.items()}


def run() -> dict:
    warnings.simplefilter("ignore")
    got = {c: _cell(c[0], c[1], seed=2000 + i) for i, c in enumerate(CELLS)}

    rel = lambda m: [abs(got[c][m] - PAPER_MSE[c][m]) / PAPER_MSE[c][m]
                     for c in CELLS if got[c][m] and PAPER_MSE[c][m]]
    ave_cells = [c for c in CELLS if got[c]["LP_ave"] is not None]

    return {
        "lp_median_rel_err": float(np.median(rel("LP"))),
        "fb_median_rel_err": float(np.median(rel("FB"))),
        "fb_cells_where_ours_is_better":
            float(sum(got[c]["FB"] < PAPER_MSE[c]["FB"] for c in CELLS)),
        "lp_beats_fb_cells":
            float(sum(got[c]["LP"] < got[c]["FB"] for c in CELLS)),
        "lp_ave_beats_fb_cells":
            float(sum(got[c]["LP_ave"] < got[c]["FB"] for c in ave_cells)),
        "both_beat_bj_cells":
            float(sum(got[c]["LP"] < got[c]["B-J"] and got[c]["FB"] < got[c]["B-J"]
                      for c in CELLS)),
        "lp_ave_beats_lp_cells":
            float(sum(got[c]["LP_ave"] < got[c]["LP"] for c in ave_cells)),
    }
