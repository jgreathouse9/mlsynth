"""Example 1 of Lv & Li (2026) and the arms of the ablation.

Lv, J. and Li, J. (2026), "Rank-based Model Average Estimator for Average
Treatment Effects", arXiv:2610.15553v1, Section 5, Example 1 (p. 24).

Three latent factors with ``a_i = 2``::

    f1_t = 0.8 f1_{t-1} + v1_t
    f2_t = -0.6 f2_{t-1} + v2_t + 0.8 v2_{t-1}
    f3_t = v3_t + 0.9 v3_{t-1} + 0.4 v3_{t-2},     v_jt ~ N(0, 0.5^2)

Loadings: 2 for unit 1 (treated) and units 11-21; N(0, 1) for units 2-10;
N(0, 0.5^2) for units 22-N, redrawn every replication. Errors: case a is
N(0, 1) throughout; case b is t_3 before treatment and N(0, 1) after; case c is
the mixture 0.9 N(0, 1) + 0.1 N(0, 25) before and N(0, 1) after.

The treatment effect never enters an arm's error. Every arm estimates
``Delta_1 = mean(Y_1t - mu_1t)`` over the post-period and the true ATE is the
realised ``mean(Delta_1t)``, so the error is ``mean(Y_1t(0) - mu_1t)``,
the same number at ``alpha_0 = 0`` and at ``alpha_0 = 1``. Table 1's two
``alpha_0`` columns are two independent Monte Carlo runs of one quantity, and
the study draws only ``Y(0)``.
"""
from __future__ import annotations

import time
import warnings
from typing import Dict

import numpy as np

from mlsynth.utils.pda_helpers.fs.estimation import forward_select
from mlsynth.utils.pda_helpers.lasso.estimation import fit_lasso

from . import oracle as O

CASES = ("a", "b", "c")
BURN = 100
#: Section 4.1, "We take M_min = 3 and M_max = 10 in simulations".
M_RANGE = range(3, 11)
#: The arms, in the order the tables print them. The 2 x 2 core is
#: {least squares, rank} x {one model on the screened donors, averaged}.
ARMS = ("RRMA", "RRMA_level", "EW", "RANK_one", "RANK_one_u", "LS_avg", "OLS_one",
        "MSCM", "PDA_fs", "PDA_lasso")


def seed_for(case: str, T1: int, N: int, rep: int) -> int:
    """One stream per (design, replication); every arm sees the same panel."""
    return 20261011 + 1_000_000 * CASES.index(case) + 10_000 * (T1 // 10) + 100 * N + rep * 7919


def draw(case: str, T1: int, T2: int, N: int, rng: np.random.Generator):
    """One panel of untreated outcomes: ``(y0, X)``, ``T x 1`` and ``T x (N-1)``."""
    T = T1 + T2
    L = T + BURN
    v = rng.normal(0.0, 0.5, size=(L + 2, 3))
    f = np.zeros((L + 2, 3))
    for t in range(2, L + 2):
        f[t, 0] = 0.8 * f[t - 1, 0] + v[t, 0]
        f[t, 1] = -0.6 * f[t - 1, 1] + v[t, 1] + 0.8 * v[t - 1, 1]
        f[t, 2] = v[t, 2] + 0.9 * v[t - 1, 2] + 0.4 * v[t - 2, 2]
    F = f[-T:]
    B = np.empty((N, 3))
    B[0] = 2.0
    B[1:10] = rng.normal(0.0, 1.0, size=(9, 3))
    B[10:21] = 2.0
    B[21:] = rng.normal(0.0, 0.5, size=(N - 21, 3))
    u = rng.normal(0.0, 1.0, size=(T, N))
    if case == "b":
        u[:T1] = rng.standard_t(3, size=(T1, N))
    elif case == "c":
        wide = rng.random((T1, N)) < 0.1
        u[:T1] = np.where(wide, rng.normal(0.0, 5.0, size=(T1, N)), rng.normal(0.0, 1.0, size=(T1, N)))
    Y = 2.0 + F @ B.T + u
    return Y[:, 0], Y[:, 1:]


def _err(y0, cf, T1) -> float:
    return float(np.mean(y0[T1:] - cf[T1:]))


def one_rep(case: str, T1: int, T2: int, N: int, rep: int) -> Dict:
    """Every arm on one panel. Returns errors, timings and Algorithm 1's diagnostics."""
    warnings.filterwarnings("ignore")
    lp_before = dict(O.LP_FALLBACKS)
    rng = np.random.default_rng(seed_for(case, T1, N, rep))
    y0, X = draw(case, T1, T2, N, rng)
    err, sec = {}, {}

    t0 = time.perf_counter()
    order, pvals, ties = O.screen(y0[:T1], X[:T1])
    rank_paths = O.candidate_paths(y0, X, T1, M_RANGE, O.rank_fit, True, rng, order)
    rr = O.choose(rank_paths, y0, T1, O.weigh_pairwise, "abs", rng)
    err["RRMA"] = _err(y0, rr.counterfactual, T1)
    sec["RRMA"] = time.perf_counter() - t0

    lv = O.choose(rank_paths, y0, T1, O.weigh_level, "abs", rng)
    err["RRMA_level"] = _err(y0, lv.counterfactual, T1)
    ew = rank_paths[rr.m].M.mean(axis=1)
    err["EW"] = _err(y0, ew, T1)

    pool = order if len(order) else np.arange(X.shape[1])
    h1, h1_ok = O.mallows_or_ones(X[:T1][:, pool], rng)
    a, b = O.rank_fit(y0[:T1], X[:T1][:, pool], h1)
    err["RANK_one"] = _err(y0, a + X[:, pool] @ b, T1)
    a, b = O.rank_fit(y0[:T1], X[:T1][:, pool], None)
    err["RANK_one_u"] = _err(y0, a + X[:, pool] @ b, T1)

    t0 = time.perf_counter()
    ls_paths = O.candidate_paths(y0, X, T1, M_RANGE, O.ols_fit, False, rng, order)
    ls = O.choose(ls_paths, y0, T1, O.weigh_ls, "sq", rng)
    err["LS_avg"] = _err(y0, ls.counterfactual, T1)
    sec["LS_avg"] = time.perf_counter() - t0
    a, b = O.ols_fit(y0[:T1], X[:T1][:, pool])
    err["OLS_one"] = _err(y0, a + X[:, pool] @ b, T1)

    a, b = O.mscm_fit(y0[:T1], X[:T1])
    err["MSCM"] = _err(y0, a + X @ b, T1)
    _, _, _, cf = forward_select(y0, X, T1, intercept=True)
    err["PDA_fs"] = _err(y0, cf, T1)
    _, _, cf, _ = fit_lasso(y0, X, T1)
    err["PDA_lasso"] = _err(y0, cf, T1)

    diag = dict(d=int(len(order)), ties=ties, m_rrma=rr.m, S_rrma=len(rr.groups),
                m_level=lv.m, m_ls_avg=ls.m, w_rrma=rr.w.tolist(),
                b_fallbacks=sum(p.mallows_fallbacks for p in rank_paths.values()),
                tau_fallbacks=rr.tau_fallbacks, rank_one_mallows=bool(h1_ok),
                strong_in_screen=int(np.isin(np.arange(9, 20), order).sum()),
                lp_retry=O.LP_FALLBACKS["retry"] - lp_before["retry"],
                lp_pairfree=O.LP_FALLBACKS["pairfree"] - lp_before["pairfree"])
    return dict(case=case, T1=T1, T2=T2, N=N, rep=rep, err=err, sec=sec, diag=diag)
