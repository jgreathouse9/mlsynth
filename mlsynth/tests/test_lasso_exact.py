"""Tests for the exact lasso backend (Langlois & Darbon 2025 homotopy).

Validates the differential-inclusion solver against scikit-learn's coordinate
descent (the trusted oracle) across wide / tall / square / degenerate designs,
and checks the warm-started path equals per-penalty exact solves. The mapping is
``t = n_samples * alpha``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Lasso, lasso_path

from mlsynth.utils.lasso.exact import lasso_exact, lasso_exact_path


def _obj(A, b, t, z):
    return np.sum(np.abs(z)) + (1.0 / (2 * t)) * np.sum((A @ z - b) ** 2)


@pytest.mark.parametrize("shape", [(40, 90), (70, 25), (50, 50)],
                         ids=["wide", "tall", "square"])
@pytest.mark.parametrize("t", [2.0, 6.0])
def test_matches_sklearn_to_machine_precision(shape, t):
    rng = np.random.default_rng(0)
    m, n = shape
    A = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    x, _, _ = lasso_exact(A, b, t)
    sk = Lasso(alpha=t / m, fit_intercept=False, tol=1e-13, max_iter=400000).fit(A, b).coef_
    assert np.max(np.abs(x - sk)) < 1e-8
    assert _obj(A, b, t, x) <= _obj(A, b, t, sk) + 1e-9      # exact: no worse than CD


def test_degenerate_donors_finite():
    """Exact duplicates, a collinear combination, and a zero column stay finite."""
    rng = np.random.default_rng(1)
    T0 = 40
    base = rng.normal(size=(T0, 6)).cumsum(0)
    A = np.hstack([base, base[:, [0, 3]], (base[:, 1] + base[:, 4])[:, None],
                   np.zeros((T0, 1)), rng.normal(size=(T0, 4))])
    b = rng.normal(size=T0)
    x, _, _ = lasso_exact(A, b, 3.0)
    assert np.all(np.isfinite(x))


def test_warm_started_path_matches_persolve():
    rng = np.random.default_rng(2)
    m, n = 18, 76
    A = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    lam_max = np.max(np.abs(A.T @ (b - b.mean()))) / m
    alphas = np.exp(np.linspace(np.log(lam_max), np.log(lam_max * 0.01), 60))  # descending
    coefs = lasso_exact_path(A, b, m * alphas)
    # sklearn path (tight) is the oracle for the unique solutions (no intercept).
    _, sk, _ = lasso_path(A, b, alphas=alphas, tol=1e-12, max_iter=2_000_000)
    assert coefs.shape == sk.shape
    assert np.max(np.abs(coefs - sk)) < 1e-6
