"""Monte-Carlo DGP for the SCM-relaxation benchmark.

The latent-group factor model of Liao, Shi & Zheng (2026, "A Relaxation Approach
to Synthetic Control", arXiv 2508.01793, Section 5.1), kept here (not inline in
the benchmark case) per the benchmarking definition-of-done.

The donor pool has a latent **group** structure: control loadings are equal
within a group, so the oracle synthetic control spreads weight equally across the
members of each group. This is the regime SCM-relaxation targets -- a dense,
group-diversified weight vector -- where it beats the sparse classical SCM in
out-of-sample counterfactual prediction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def simulate_relaxation_groups(
    rng: np.random.Generator,
    J: int,
    T0: int,
    *,
    T1: int = 50,
    K: int | None = None,
    r: int | None = None,
    ar: float = 0.5,
    approximate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Liao-Shi-Zheng (2026) Section-5 latent-group factor DGP.

    ``y^N_{jt} = lambda_j' f_t + u_{jt}``, ``u ~ N(0, 1)``; ``r`` independent
    AR(1) factors ``f_{l,t} = ar f_{l,t-1} + N(0,1)``; core loadings
    ``Lambda^co`` (K x r) iid ``N(0, 3/r)``; control loadings
    ``Lambda = Z Lambda^co`` (equal within group); treated loading
    ``lambda_0 = Lambda^co' w*_G + eps`` with ``eps ~ U(-0.1/sqrt(r),
    0.1/sqrt(r))`` and ``w*_G`` on the simplex with a zero first entry (Dirichlet
    on the rest). With ``approximate=True`` the loadings are perturbed by
    ``U(-0.2/sqrt(r), 0.2/sqrt(r))`` so the group structure only holds
    approximately (Section 5.2).

    Parameters
    ----------
    rng : np.random.Generator
    J : int
        Number of control units.
    T0, T1 : int
        Pre- and post-treatment period counts (paper: T1 = 50).
    K : int, optional
        Number of latent groups (default ``r``).
    r : int, optional
        Number of factors (default ``floor(log T0)``).
    ar : float, default 0.5
        AR(1) coefficient of the factors.
    approximate : bool, default False
        Exact (False) vs approximate (True) group structure.

    Returns
    -------
    (Yc, y0, oracle_cf, T0)
        ``Yc`` -- ``(J, T)`` control outcomes; ``y0`` -- ``(T,)`` treated
        untreated-potential outcome; ``oracle_cf`` -- ``(T,)`` oracle synthetic
        control (group-equal weights applied to the realized controls);
        ``T0`` echoed back. ``T = T0 + T1``.
    """
    if r is None:
        r = max(1, int(np.floor(np.log(T0))))
    if K is None:
        K = r
    T = T0 + T1

    F = np.zeros((T, r))
    for el in range(r):
        e = rng.normal(size=T)
        F[0, el] = e[0]
        for t in range(1, T):
            F[t, el] = ar * F[t - 1, el] + e[t]

    Lam_co = rng.normal(0.0, np.sqrt(3.0 / r), size=(K, r))
    groups = np.arange(J) % K                      # roughly equal group sizes
    Lam = Lam_co[groups]                           # (J, r), equal within group
    if approximate:
        Lam = Lam + rng.uniform(-0.2 / np.sqrt(r), 0.2 / np.sqrt(r), size=(J, r))

    wG = np.zeros(K)
    if K > 1:
        wG[1:] = rng.dirichlet(np.ones(K - 1))
    else:
        wG[0] = 1.0
    eps = rng.uniform(-0.1 / np.sqrt(r), 0.1 / np.sqrt(r), size=r)
    lam0 = Lam_co.T @ wG + eps

    # Oracle control weights: spread each group's weight equally over its members.
    w_star = np.zeros(J)
    for k in range(K):
        members = np.where(groups == k)[0]
        if members.size:
            w_star[members] = wG[k] / members.size

    Yc = Lam @ F.T + rng.normal(size=(J, T))       # (J, T)
    y0 = lam0 @ F.T + rng.normal(size=T)           # (T,)
    oracle_cf = w_star @ Yc                         # (T,)
    return Yc, y0, oracle_cf, T0


def to_panel(Yc: np.ndarray, y0: np.ndarray, T0: int) -> pd.DataFrame:
    """Reshape ``(Yc, y0)`` into the long panel RESCM consumes: ``unit``,
    ``time``, ``y`` (outcome), ``treat`` (the treated unit 0 after ``T0``).
    There is no additive treatment effect -- the post-period is marked treated
    only so RESCM emits a counterfactual to compare against the oracle.
    """
    J, T = Yc.shape
    rows = []
    for t in range(T):
        rows.append({"unit": "treated", "time": t, "y": float(y0[t]),
                     "treat": int(t >= T0)})
    for j in range(J):
        for t in range(T):
            rows.append({"unit": f"c{j:03d}", "time": t, "y": float(Yc[j, t]),
                         "treat": 0})
    return pd.DataFrame(rows)
