"""Abadie-L'Hour penalization without covariates (outcome-lag matching).

The pairwise penalty is defined on the *matching variables*. In the canonical
outcome-only specification those are the pre-treatment outcome lags, so
``backend="penalized"`` is well defined with no covariates at all -- and that is
exactly the configuration where it matters most. When the donor pool is large
relative to the pre-period, the treated unit lies inside the donors' convex hull
and the unpenalized program has a *continuum* of solutions that all reproduce
the pre-treatment path perfectly; which one a solver returns is an artifact of
its path, not a feature of the data. Abadie & L'Hour (2021) Theorem 1 says any
``lambda > 0`` restores a unique solution with at most ``K + 1`` non-zero
weights, selecting the one with the least interpolation bias.

Written test-first: before the fix, ``backend="penalized"`` silently fell
through to the unpenalized outcome-only simplex whenever covariates were absent
and discarded ``penalized_lambda`` without warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _degenerate_panel(J: int = 12, T_pre: int = 4, T_post: int = 4, seed: int = 0):
    """Panel where the unpenalized solution is *not* unique.

    ``J`` donors against ``T_pre`` pre-periods with the treated unit built as an
    exact convex combination of donors, so it sits strictly inside the hull and
    the interpolation polytope has dimension ``J - T_pre - 1 > 0``.
    """
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    Y0 = rng.uniform(1.0, 5.0, size=(T, J))
    w = np.zeros(J)
    w[[0, 1, 2]] = [0.5, 0.3, 0.2]          # treated is inside the hull
    y1 = Y0 @ w
    y1[T_pre:] += 3.0                        # a post-treatment effect
    rows = []
    for t in range(T):
        rows.append({"unit": "treated", "time": t, "y": float(y1[t]),
                     "D": int(t >= T_pre)})
        for j in range(J):
            rows.append({"unit": f"d{j:02d}", "time": t, "y": float(Y0[t, j]),
                         "D": 0})
    return pd.DataFrame(rows)


def _covariate_panel(J: int = 8, T_pre: int = 15, T_post: int = 6, seed: int = 0):
    """Ordinary panel carrying a unit-level covariate (non-regression guard)."""
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, 2))
    lam = rng.standard_normal((J + 1, 2))
    Y = F @ lam.T + 0.2 * rng.standard_normal((T, J + 1))
    x = lam[:, 0] * 3.0 + rng.standard_normal(J + 1) * 0.1
    return pd.DataFrame([
        {"unit": j, "time": t, "y": float(Y[t, j]), "x": float(x[j]),
         "D": int(j == 0 and t >= T_pre)}
        for j in range(J + 1) for t in range(T)
    ])


@pytest.fixture
def deg():
    return _degenerate_panel()


@pytest.fixture
def cov_df():
    return _covariate_panel()


def _fit(df, **over):
    cfg = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
               inference=False, display_graphs=False)
    cfg.update(over)
    return VanillaSC(cfg).fit()


def _w(res, keys=None) -> np.ndarray:
    """Weight vector aligned to ``keys`` (``donor_weights`` drops zero entries,
    so two fits with different support are not directly comparable)."""
    d = res.weights.donor_weights or {}
    if keys is None:
        keys = sorted(d)
    return np.array([float(d.get(k, 0.0)) for k in keys], dtype=float)


def _nnz(res, tol: float = 1e-6) -> int:
    return int((_w(res) > tol).sum())


# --------------------------------------------------------------------------- #
# the penalty is actually applied
# --------------------------------------------------------------------------- #
def test_penalized_without_covariates_differs_from_unpenalized(deg):
    """The whole point: penalizing must change the selected weights."""
    pen = _fit(deg, backend="penalized", penalized_lambda=1e-3)
    plain = _fit(deg, backend="outcome-only")
    keys = sorted(set(pen.weights.donor_weights) | set(plain.weights.donor_weights))
    assert not np.allclose(_w(pen, keys), _w(plain, keys), atol=1e-8)


def test_explicit_lambda_is_honoured_and_reported(deg):
    res = _fit(deg, backend="penalized", penalized_lambda=1e-3)
    assert res.method_details.parameters_used["penalized_lambda"] == pytest.approx(1e-3)


def test_reports_the_penalized_backend(deg):
    res = _fit(deg, backend="penalized", penalized_lambda=1e-3)
    assert "penalized" in res.method_details.method_name.lower()
    assert res.method_details.parameters_used["backend"] == "penalized"


def test_cv_selects_a_lambda_without_covariates(deg):
    """With no fixed lambda the CV path must still run and surface its choice."""
    res = _fit(deg, backend="penalized")
    lam = res.method_details.parameters_used["penalized_lambda"]
    assert lam is not None and lam >= 0.0


# --------------------------------------------------------------------------- #
# Abadie-L'Hour Theorem 1: uniqueness and sparsity
# --------------------------------------------------------------------------- #
def test_unpenalized_solution_is_degenerate_on_this_panel(deg):
    """Guard on the fixture: without a penalty the fit is perfect and dense,
    which is the signature of a non-unique optimum."""
    plain = _fit(deg, backend="outcome-only")
    assert plain.fit_diagnostics.rmse_pre < 1e-6
    assert _nnz(plain) > 5          # weight smeared across the polytope


def test_positive_lambda_gives_at_most_K_plus_one_nonzero_weights(deg):
    """Theorem 1 sparsity bound, with the pre-period lags as matching variables.

    Covers the small-lambda end too. That regime used to be excluded here and
    pinned separately as a solver limitation: the inner QP was a first-order
    method that under-converged when the penalty was small relative to the fit
    term. It is now solved exactly, so the bound holds at every positive lambda
    as the theorem states.
    """
    T_pre = 4
    for lam in (1e-4, 1e-2, 1e-1, 1.0):
        assert _nnz(_fit(deg, backend="penalized", penalized_lambda=lam)) <= T_pre + 1


def test_penalized_fit_is_deterministic(deg):
    """Uniqueness implies repeated fits agree exactly."""
    a = _w(_fit(deg, backend="penalized", penalized_lambda=1e-3))
    b = _w(_fit(deg, backend="penalized", penalized_lambda=1e-3))
    assert np.allclose(a, b, atol=1e-12)


def test_larger_lambda_is_weakly_sparser(deg):
    counts = [_nnz(_fit(deg, backend="penalized", penalized_lambda=lam))
              for lam in (1e-4, 1e-2, 1.0)]
    assert counts[-1] <= counts[0]


def test_small_lambda_keeps_the_pre_treatment_fit(deg):
    """lambda -> 0 selects *among* the best synthetic controls, so a tiny
    penalty must not meaningfully degrade the pre-period fit."""
    plain = _fit(deg, backend="outcome-only")
    pen = _fit(deg, backend="penalized", penalized_lambda=1e-8)
    assert pen.fit_diagnostics.rmse_pre <= plain.fit_diagnostics.rmse_pre + 1e-4


def test_weights_remain_a_valid_simplex(deg):
    w = _w(_fit(deg, backend="penalized", penalized_lambda=1e-2))
    assert (w >= -1e-12).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #
def test_negative_lambda_rejected_without_covariates(deg):
    with pytest.raises(MlsynthConfigError):
        VanillaSC(dict(df=deg, outcome="y", treat="D", unitid="unit",
                       time="time", backend="penalized", penalized_lambda=-1.0))


# --------------------------------------------------------------------------- #
# non-regression
# --------------------------------------------------------------------------- #
def test_outcome_only_backend_is_unchanged(deg):
    """The fix must not perturb the plain outcome-only path."""
    res = _fit(deg, backend="outcome-only")
    assert res.method_details.parameters_used["backend"] == "outcome-only"
    assert res.fit_diagnostics.rmse_pre < 1e-6


def test_penalized_with_covariates_still_works(cov_df):
    res = _fit(cov_df, covariates=["x"], backend="penalized", penalized_lambda=0.2)
    assert res.method_details.parameters_used["penalized_lambda"] == pytest.approx(0.2)
    w = _w(res)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_auto_backend_without_covariates_stays_outcome_only(cov_df):
    """``auto`` must keep routing to outcome-only, not to penalized."""
    res = _fit(cov_df, backend="auto")
    assert res.method_details.parameters_used["backend"] == "outcome-only"
