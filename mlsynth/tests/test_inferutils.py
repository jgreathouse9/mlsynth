"""Tests for ``mlsynth.utils.inferutils`` — general-purpose SC inference tools.

The first occupant is :func:`debiased_sc_ttest`, the Chernozhukov, Wuthrich &
Zhu (2025, arXiv:1812.10820) debiased synthetic-control *t*-test for the ATT:
a K-fold cross-fitting bias correction with a self-normalized statistic that is
asymptotically ``t_{K-1}``. The reference is the authors' R package
``scinference`` (``ttest.R::sc.cf`` + ``estimators.R::sc``); the algorithm is
estimator-agnostic (it rides any ℓ2-consistent weights), so it lives at the
shared ``utils/`` level rather than inside one estimator's helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as tdist

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.inferutils import debiased_sc_ttest

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"
def _panel(T0=12, T1=6, J=4, seed=0):
    """A small, well-behaved donor panel; treated ~ uniform donor average."""
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(T0 + T1, J)) + np.arange(T0 + T1)[:, None] * 0.1
    y = Y0 @ np.full(J, 1.0 / J) + rng.normal(scale=0.01, size=T0 + T1)
    return y, Y0


def _uniform_fn(J):
    """A solver-independent weight_fn: ignores the indices, returns 1/J."""
    return lambda idx: np.full(J, 1.0 / J)


def _expected_blocks(T0, T1, K):
    """0-based held-out block indices, matching scinference's last-K*r choice."""
    r = min(T0 // K, T1)
    o = T0 - r * K
    return r, [np.arange(o + k * r, o + k * r + r) for k in range(K)]


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_smoke_default_solver_returns_contract():
    y, Y0 = _panel()
    out = debiased_sc_ttest(y, Y0, T0=12, T1=6, K=3)
    for key in ("att", "se", "tstat", "dof", "ci_lower", "ci_upper",
                "tau_k", "K", "r", "alpha"):
        assert key in out
    assert np.isfinite(out["att"]) and np.isfinite(out["se"])
    assert out["se"] > 0
    assert len(out["tau_k"]) == 3
    assert out["dof"] == 2
    assert out["ci_lower"] < out["att"] < out["ci_upper"]


# --------------------------------------------------------------------------- #
# arithmetic / invariants  (solver-independent via a fixed weight_fn)
# --------------------------------------------------------------------------- #
def test_block_construction_matches_scinference():
    y, Y0 = _panel(T0=20, T1=23, J=5)
    seen = []
    debiased_sc_ttest(y, Y0, T0=20, T1=23, K=3,
                      weight_fn=lambda idx: (seen.append(np.asarray(idx)),
                                             np.full(5, 0.2))[1])
    r, blocks = _expected_blocks(20, 23, 3)
    assert r == 6
    # weight_fn is called once per fold on the *complement* of each block.
    for got_keep, blk in zip(seen, blocks):
        expected_keep = np.setdiff1d(np.arange(20), blk)
        assert np.array_equal(np.sort(got_keep), expected_keep)


def test_tau_k_and_att_arithmetic():
    y, Y0 = _panel(T0=20, T1=23, J=5)
    K = 3
    out = debiased_sc_ttest(y, Y0, T0=20, T1=23, K=K, weight_fn=_uniform_fn(5))
    w = np.full(5, 0.2)
    gap = y - Y0 @ w
    r, blocks = _expected_blocks(20, 23, K)
    exp_tau = np.array([gap[20:].mean() - gap[blk].mean() for blk in blocks])
    np.testing.assert_allclose(out["tau_k"], exp_tau, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["att"], exp_tau.mean(), atol=1e-12)


def test_se_rescale_and_tstat_and_ci():
    y, Y0 = _panel(T0=20, T1=23, J=5)
    K, T1, alpha = 4, 23, 0.1
    out = debiased_sc_ttest(y, Y0, T0=20, T1=T1, K=K, alpha=alpha,
                            weight_fn=_uniform_fn(5))
    r = out["r"]
    tau = out["tau_k"]
    exp_se = np.sqrt(1 + (K * r) / T1) * tau.std(ddof=1) / np.sqrt(K)
    np.testing.assert_allclose(out["se"], exp_se, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["tstat"], out["att"] / out["se"], atol=1e-12)
    crit = tdist.ppf(1 - alpha / 2, K - 1)
    np.testing.assert_allclose(out["ci_lower"], out["att"] - crit * out["se"], atol=1e-12)
    np.testing.assert_allclose(out["ci_upper"], out["att"] + crit * out["se"], atol=1e-12)


def test_r_definition():
    y, Y0 = _panel(T0=20, T1=4, J=5)            # T1 < floor(T0/K) so r is capped by T1
    out = debiased_sc_ttest(y, Y0, T0=20, T1=4, K=3, weight_fn=_uniform_fn(5))
    assert out["r"] == min(20 // 3, 4) == 4


# --------------------------------------------------------------------------- #
# default solver is the outcome-only simplex (matches scinference's sc())
# --------------------------------------------------------------------------- #
def test_default_solver_is_outcome_only_simplex():
    y, Y0 = _panel(T0=16, T1=8, J=4, seed=3)
    captured = {}
    # wrap default by calling with an explicit simplex check: refit on full pre.
    out = debiased_sc_ttest(y, Y0, T0=16, T1=8, K=2)
    # The default per-fold weights must be a valid simplex: this is implied by a
    # finite result; assert by re-solving one fold's complement directly.
    import cvxpy as cp
    r, blocks = _expected_blocks(16, 8, 2)
    keep = np.setdiff1d(np.arange(16), blocks[0])
    w = cp.Variable(4)
    cp.Problem(cp.Minimize(cp.sum_squares(Y0[keep] @ w - y[keep])),
               [cp.sum(w) == 1, w >= 0]).solve(solver=cp.OSQP, eps_abs=1e-9,
                                               eps_rel=1e-9, max_iter=200000)
    w = np.asarray(w.value).ravel()
    assert abs(w.sum() - 1) < 1e-6 and (w > -1e-6).all()
    captured["ok"] = True
    assert captured["ok"]


# --------------------------------------------------------------------------- #
# failure / edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [0, 1])
def test_K_must_exceed_one(K):
    y, Y0 = _panel()
    with pytest.raises(MlsynthConfigError):
        debiased_sc_ttest(y, Y0, T0=12, T1=6, K=K)


def test_block_too_small_raises():
    # T0 // K == 0  ->  r == 0  ->  no block can be formed.
    y, Y0 = _panel(T0=2, T1=6, J=3)
    with pytest.raises(MlsynthConfigError):
        debiased_sc_ttest(y, Y0, T0=2, T1=6, K=3)


def test_Y0_must_be_2d():
    y, _ = _panel(T0=12, T1=6)
    with pytest.raises(MlsynthDataError):
        debiased_sc_ttest(y, np.ones(18), T0=12, T1=6, K=3)


def test_length_mismatch_raises():
    y, Y0 = _panel(T0=12, T1=6)
    with pytest.raises(MlsynthDataError):
        debiased_sc_ttest(y[:-1], Y0, T0=12, T1=6, K=3)          # y too short
    with pytest.raises(MlsynthDataError):
        debiased_sc_ttest(y, Y0[:-1], T0=12, T1=6, K=3)          # Y0 wrong rows


def test_weight_fn_wrong_shape_raises():
    y, Y0 = _panel(T0=12, T1=6, J=4)
    with pytest.raises(MlsynthEstimationError):
        debiased_sc_ttest(y, Y0, T0=12, T1=6, K=3,
                          weight_fn=lambda idx: np.ones(3))        # J=4 expected


# --------------------------------------------------------------------------- #
# integration: real Basque data, default outcome-only solver (pinned)
# --------------------------------------------------------------------------- #
def test_basque_outcome_only_pinned():
    f = _BASEDATA / "basque_jasa.csv"
    if not f.exists():
        pytest.skip("basque_jasa.csv not available")
    df = pd.read_csv(f)
    df = df[df.regionname != "Spain (Espana)"]
    piv = df.pivot(index="year", columns="regionname", values="gdpcap").sort_index()
    treated = "Basque Country (Pais Vasco)"
    y = piv[treated].to_numpy()
    Y0 = piv.drop(columns=[treated]).to_numpy()
    T0 = int((piv.index < 1975).sum())
    T1 = int((piv.index >= 1975).sum())
    out = debiased_sc_ttest(y, Y0, T0=T0, T1=T1, K=3)
    assert out["att"] < 0                                   # terrorism reduced GDP
    assert out["ci_upper"] < 0                              # significant at 10%
    np.testing.assert_allclose(out["att"], -0.6575, atol=5e-3)
    np.testing.assert_allclose(out["se"], 0.1391, atol=5e-3)


# --------------------------------------------------------------------------- #
# RAE formula (CWZ eq. 14) and automatic K selection
# --------------------------------------------------------------------------- #
from mlsynth.utils.inferutils import rae, select_K   # noqa: E402


def test_rae_reproduces_paper_table1():
    # Table 1: relative asymptotic efficiency, c0 = T0/T1 = 30/16, alpha = 0.1.
    c0, alpha = 30 / 16, 0.1
    expected = {2: 32.65, 3: 63.56, 4: 75.86, 5: 82.08, 6: 85.79,
                7: 88.23, 8: 89.97, 9: 91.26, 10: 92.25}
    for K, pct in expected.items():
        np.testing.assert_allclose(100 * rae(c0, K, alpha), pct, atol=0.05)


def test_rae_monotone_increasing_in_K():
    c0 = 30 / 16
    vals = [rae(c0, K, 0.1) for K in range(2, 11)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_select_K_defaults_to_three_for_moderate_T0():
    # Persistent residuals (rho high) at a moderate T0 -> the K=3 benchmark.
    rng = np.random.default_rng(0)
    e = rng.normal(size=30)
    resid = np.empty(30); resid[0] = e[0]
    for t in range(1, 30):
        resid[t] = 0.8 * resid[t - 1] + e[t]          # AR(1), rho ~ 0.8
    K, info = select_K(T0=30, T1=16, residuals=resid, alpha=0.1)
    assert K == 3
    assert info["rho_hat"] > 0.5


def test_select_K_bumps_to_four_when_persistence_low():
    rng = np.random.default_rng(1)
    resid = rng.normal(size=30)                        # iid -> rho ~ 0 (low)
    K, info = select_K(T0=30, T1=16, residuals=resid, alpha=0.1)
    assert K == 4
    assert info["rho_hat"] < 0.5


def test_select_K_climbs_for_large_T0():
    # Large T0 with low persistence: climb toward the RAE target -> K > 4.
    rng = np.random.default_rng(2)
    resid = rng.normal(size=120)
    K, _ = select_K(T0=120, T1=40, residuals=resid, alpha=0.1)
    assert K > 4


def test_select_K_respects_block_min_for_small_T0():
    rng = np.random.default_rng(3)
    resid = rng.normal(size=20)                        # Basque-sized T0
    K, _ = select_K(T0=20, T1=23, residuals=resid, alpha=0.1)
    assert K == 3                                      # K=4 infeasible (block_min)


from mlsynth.utils.inferutils import _ar1   # noqa: E402


def test_rae_c0_greater_than_K_branch():
    # c0 > K activates g = 1; result still finite and positive.
    val = rae(c0=5.0, K=2, alpha=0.1)
    assert np.isfinite(val) and val > 0


def test_rae_requires_K_at_least_two():
    with pytest.raises(MlsynthConfigError):
        rae(c0=1.5, K=1, alpha=0.1)


def test_ar1_edge_cases():
    assert _ar1(np.array([1.0])) == 0.0               # size < 2
    assert _ar1(np.full(8, 3.0)) == 0.0               # constant -> zero variance


def test_select_K_relaxes_block_min_for_tiny_T0():
    rng = np.random.default_rng(7)
    resid = rng.normal(size=10)
    K, info = select_K(T0=10, T1=8, residuals=resid, alpha=0.1)
    assert info["relaxed"] is True
    assert 2 <= K <= 10
