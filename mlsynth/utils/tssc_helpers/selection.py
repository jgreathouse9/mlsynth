"""Step 1 of TSSC: select the SC-class method by subsampling tests.

Implements Section 3.2 of Li & Shankar (2023). The benchmark MSC(c)
coefficient ``beta_hat_{MSC,T1}`` is computed on the full pre-treatment
sample, then a subsampling-with-replacement procedure approximates the
null distribution of the restriction test statistics:

    Step i.   For ``t = 1, ..., m`` draw ``(x_t*, y_{1t}*)`` with
              replacement from the ``T1`` pre-treatment observations.
    Step ii.  Refit MSC(c) on the subsample -> ``beta_hat*_{MSC,m,b}``.
    Step iii. Repeat ``B`` times.

The full-sample statistics are (Eqs. 3.6/3.9, 3.13)

    joint:  S_hat_{T1}   = T1 * d_hat'  V_hat^{-1} d_hat,    d_hat = R beta - q
    single: S_hat_{T1,s} = (sqrt(T1) d_hat_s)^2,             V replaced by 1

with ``V_hat = R Var*(sqrt(T1) beta_hat) R'`` (Eqs. 3.7-3.8). The
subsampling distribution ``{S*_{m,b}}`` (built from ``sqrt(m) R(beta* -
beta_hat)``) gives the ``(1 - alpha)`` acceptance region
``[S*_{(alpha B/2)}, S*_{((1 - alpha/2) B)}]`` (Proposition 3.2). H0 is
rejected when the full-sample statistic falls outside it.

Decision tree (Figure 1): joint not rejected -> SC; else sum-to-one not
rejected -> MSCa; else zero-intercept not rejected -> MSCb; else MSCc.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from .estimation import fit_mscc_beta
from .structures import (
    MSCA,
    MSCB,
    MSCC,
    SC,
    TSSCInputs,
    TSSCRestrictionTest,
    TSSCSelection,
)


def _restriction_matrices(p: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build ``(R, q)`` for the joint and two single restrictions.

    ``p = N`` is the MSC(c) coefficient count (intercept + ``N-1`` donors).
    The intercept is coordinate 0; donor slopes are coordinates ``1..p-1``.
    """
    R_a = np.zeros((1, p)); R_a[0, 1:] = 1.0          # sum_{j>=2} beta_j
    q_a = np.array([1.0])                             # = 1
    R_b = np.zeros((1, p)); R_b[0, 0] = 1.0           # beta_1
    q_b = np.array([0.0])                             # = 0
    R_joint = np.vstack([R_a, R_b])                   # both
    q_joint = np.array([1.0, 0.0])
    return {
        "joint": (R_joint, q_joint),
        "sum_to_one": (R_a, q_a),
        "zero_intercept": (R_b, q_b),
    }


def _subsample_betas(
    inputs: TSSCInputs, m: int, B: int, rng: np.random.Generator
) -> np.ndarray:
    """Refit MSC(c) on ``B`` size-``m`` with-replacement subsamples."""
    T0 = inputs.T0
    n = inputs.n_donors
    donor_pre = inputs.donor_matrix[:T0]
    y_pre = inputs.y[:T0]
    betas: List[np.ndarray] = []
    for _ in range(B):
        idx = rng.integers(0, T0, size=m)
        beta_b = fit_mscc_beta(donor_pre[idx], y_pre[idx], m, n)
        if beta_b is not None and np.all(np.isfinite(beta_b)):
            betas.append(beta_b)
    if len(betas) < 2:
        raise MlsynthEstimationError(
            "TSSC: subsampling produced too few feasible MSC(c) refits "
            "to estimate the test distribution."
        )
    return np.asarray(betas)


def _run_test(
    name: str,
    R: np.ndarray,
    q: np.ndarray,
    beta: np.ndarray,
    diff: np.ndarray,
    T0: int,
    m: int,
    alpha: float,
    studentize: bool,
) -> TSSCRestrictionTest:
    """Compute one restriction test and its subsampling acceptance region.

    Parameters
    ----------
    diff : np.ndarray
        ``(B, p)`` matrix of subsample deviations ``beta* - beta_hat``.
    studentize : bool
        Joint test uses ``V_hat^{-1}`` (Eq. 3.6); single restrictions
        replace ``V_hat`` by one (Eq. 3.13).
    """
    d_hat = R @ beta - q                       # (k,)
    Rdiff = diff @ R.T                          # (B, k)

    if studentize:
        # V_hat = R Var*(sqrt(T1) beta_hat) R', Var* = (m/B) sum diff diff'.
        B_count = diff.shape[0]
        var_star = (m / B_count) * (diff.T @ diff)   # (p, p)
        V = R @ var_star @ R.T                        # (k, k)
        V_inv = np.linalg.pinv(V)
        stat = float(T0 * d_hat @ V_inv @ d_hat)
        sub_stats = m * np.einsum("bi,ij,bj->b", Rdiff, V_inv, Rdiff)
    else:
        # Single restriction (k = 1): V replaced by 1.
        stat = float(T0 * d_hat[0] ** 2)
        sub_stats = m * Rdiff[:, 0] ** 2

    lower = float(np.quantile(sub_stats, alpha / 2))
    upper = float(np.quantile(sub_stats, 1 - alpha / 2))
    rejected = not (lower <= stat <= upper)
    return TSSCRestrictionTest(
        name=name, statistic=stat, ci_lower=lower, ci_upper=upper,
        rejected=rejected,
    )


def select_method(
    inputs: TSSCInputs,
    alpha: float = 0.05,
    subsample_size: Optional[int] = None,
    n_subsamples: int = 500,
    seed: Optional[int] = None,
) -> TSSCSelection:
    """Run the Step-1 decision tree and return the recommended method.

    Parameters
    ----------
    inputs : TSSCInputs
    alpha : float
        Two-sided significance level for each restriction test.
    subsample_size : int, optional
        Subsample size ``m``. Defaults to ``T1`` (the bootstrap special
        case the paper's simulations validate; choose ``m`` smaller --
        rule of thumb ``T1/2`` to ``T1`` -- for genuine subsampling).
    n_subsamples : int
        Number of subsampling replications ``B``.
    seed : int, optional
        RNG seed for reproducible subsampling.
    """
    T0 = inputs.T0
    m = int(subsample_size) if subsample_size is not None else T0
    m = max(2, min(m, T0))
    rng = np.random.default_rng(seed)

    beta = fit_mscc_beta(
        inputs.donor_matrix[:T0], inputs.y[:T0], T0, inputs.n_donors
    )
    if beta is None:
        raise MlsynthEstimationError(
            "TSSC: benchmark MSC(c) fit failed; cannot run Step-1 selection."
        )

    p = beta.shape[0]                       # N = intercept + (N-1) donors
    R = _restriction_matrices(p)
    beta_star = _subsample_betas(inputs, m, n_subsamples, rng)
    diff = beta_star - beta                  # (B, p)
    B_eff = diff.shape[0]

    tests: Dict[str, TSSCRestrictionTest] = {}
    path: List[str] = []

    joint = _run_test("joint", *R["joint"], beta, diff, T0, m, alpha, True)
    tests["joint"] = joint
    if not joint.rejected:
        path.append("joint H0 not rejected -> SC")
        recommended = SC
    else:
        path.append("joint H0 rejected -> test sum-to-one")
        h0a = _run_test(
            "sum_to_one", *R["sum_to_one"], beta, diff, T0, m, alpha, False
        )
        tests["sum_to_one"] = h0a
        if not h0a.rejected:
            path.append("sum-to-one not rejected -> MSCa")
            recommended = MSCA
        else:
            path.append("sum-to-one rejected -> test zero-intercept")
            h0b = _run_test(
                "zero_intercept", *R["zero_intercept"], beta, diff, T0, m,
                alpha, False,
            )
            tests["zero_intercept"] = h0b
            if not h0b.rejected:
                path.append("zero-intercept not rejected -> MSCb")
                recommended = MSCB
            else:
                path.append("zero-intercept rejected -> MSCc")
                recommended = MSCC

    return TSSCSelection(
        recommended=recommended,
        tests=tests,
        alpha=alpha,
        subsample_size=m,
        n_subsamples=B_eff,
        mscc_beta=beta,
        decision_path=tuple(path),
    )
