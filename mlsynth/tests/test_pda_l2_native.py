"""Native (OSQP) L2-relaxation solver for PDA: parity + independence tests.

The L2-relaxation primal (Shi & Wang 2024, Eq. 3)

    min_beta  (1/2) ||beta||_2^2   s.t.   || eta_hat - Sigma_hat beta ||_inf <= tau

has a strictly convex objective, so its solution is *unique* (Lemma 2). OSQP and
cvxpy/CLARABEL therefore converge to the same beta -- only solver tolerance
differs. That makes routing the single-treated solve through the same OSQP path
the multiple-treated batch already uses (``l2/batch.py``) safe: it removes cvxpy
from the L2 default path and lets the tau cross-validation reuse one KKT
factorization across the whole grid (shared Sigma).

These tests pin, TDD-first:

1. parity of ``l2_relax`` (OSQP) with a cvxpy/CLARABEL oracle (beta + intercept),
2. constraint feasibility of the returned solution,
3. parity across a tau grid (the CV inputs),
4. cross-validation returning a finite grid member matching a cvxpy CV oracle,
5. independence from cvxpy on the default path,
6. ``fit_l2`` end-to-end without cvxpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.pda_helpers.l2.estimation import (
    _standardize,
    cross_validate_tau,
    fit_l2,
    l2_relax,
)


# ---------------------------------------------------------------------------
# Fixtures / oracle
# ---------------------------------------------------------------------------

def _data(seed: int = 0, T: int = 40, N: int = 8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N)) * 2 + 5
    # treated series = sparse combo of controls + noise (well-conditioned)
    beta = np.zeros(N)
    beta[:3] = [0.5, 0.3, 0.2]
    y = X @ beta + rng.standard_normal(T) * 0.3 + 1.0
    return y, X


def _l2_relax_cvxpy(y_pre, X_pre, tau, standardize=True):
    """CLARABEL oracle mirroring the original l2_relax (unique optimum)."""
    import cvxpy as cp

    T1 = X_pre.shape[0]
    mu_y, Mu_X, sd_y, Sd_X = _standardize(y_pre, X_pre, standardize)
    yt = (y_pre - mu_y) / sd_y
    Xt = (X_pre - Mu_X) / Sd_X
    Sigma = (Xt.T @ Xt) / T1
    eta = (Xt.T @ yt) / T1
    beta = cp.Variable(X_pre.shape[1])
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(beta)),
               [cp.norm(eta - Sigma @ beta, "inf") <= tau]).solve(solver=cp.CLARABEL)
    beta_tilde = np.asarray(beta.value).ravel()
    beta_hat = sd_y * (beta_tilde / Sd_X)
    intercept = mu_y - float(Mu_X @ beta_hat)
    return beta_hat, intercept


# ---------------------------------------------------------------------------
# Parity (unique optimum)
# ---------------------------------------------------------------------------

class TestParity:
    @pytest.mark.parametrize("tau", [0.02, 0.1, 0.3])
    def test_matches_cvxpy(self, tau):
        y, X = _data(0)
        b, a = l2_relax(y[:30], X[:30], tau)
        b_ref, a_ref = _l2_relax_cvxpy(y[:30], X[:30], tau)
        np.testing.assert_allclose(b, b_ref, atol=1e-4)
        # intercept = mu_y - Mu_X . beta amplifies the ~2e-5 beta agreement by the
        # donor means, so it agrees a little looser -- still solver-precision tight.
        assert a == pytest.approx(a_ref, abs=2e-3)

    def test_matches_cvxpy_unstandardized(self):
        y, X = _data(1)
        b, a = l2_relax(y[:30], X[:30], 0.05, standardize=False)
        b_ref, a_ref = _l2_relax_cvxpy(y[:30], X[:30], 0.05, standardize=False)
        np.testing.assert_allclose(b, b_ref, atol=1e-4)
        # intercept = mu_y - Mu_X . beta amplifies the ~2e-5 beta agreement by the
        # donor means, so it agrees a little looser -- still solver-precision tight.
        assert a == pytest.approx(a_ref, abs=2e-3)


class TestFeasibility:
    def test_constraint_satisfied(self):
        y, X = _data(2)
        T0 = 30
        tau = 0.1
        mu_y, Mu_X, sd_y, Sd_X = _standardize(y[:T0], X[:T0], True)
        Xt = (X[:T0] - Mu_X) / Sd_X
        yt = (y[:T0] - mu_y) / sd_y
        Sigma = (Xt.T @ Xt) / T0
        eta = (Xt.T @ yt) / T0
        b, _ = l2_relax(y[:T0], X[:T0], tau)
        beta_tilde = b * Sd_X / sd_y          # back to standardized scale
        infnorm = float(np.max(np.abs(eta - Sigma @ beta_tilde)))
        assert infnorm <= tau + 1e-4


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_returns_finite(self):
        y, X = _data(0)
        tau = cross_validate_tau(y[:30], X[:30])
        assert np.isfinite(tau) and tau > 0

    def test_explicit_grid_matches_cvxpy_choice(self):
        y, X = _data(3)
        yp, Xp = y[:30], X[:30]
        grid = np.logspace(-3, -0.5, 12)
        tau = cross_validate_tau(yp, Xp, tau_grid=grid)

        # cvxpy reference CV on the same split / grid.
        n_val = max(2, int(round(0.2 * len(yp))))
        yt, Xt = yp[:-n_val], Xp[:-n_val]
        yv, Xv = yp[-n_val:], Xp[-n_val:]
        mses = []
        for t in grid:
            b, a = _l2_relax_cvxpy(yt, Xt, t)
            mses.append(float(np.mean((yv - (Xv @ b + a)) ** 2)))
        tau_ref = float(grid[int(np.argmin(mses))])
        # same grid member, or one neighbouring step (flat CV curve).
        i = int(np.argmin(np.abs(grid - tau)))
        i_ref = int(np.argmin(np.abs(grid - tau_ref)))
        assert abs(i - i_ref) <= 1


# ---------------------------------------------------------------------------
# Independence from cvxpy
# ---------------------------------------------------------------------------

class TestNoCvxpy:
    def test_l2_relax_without_cvxpy(self, monkeypatch):
        import cvxpy

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("native l2_relax must not call cvxpy")

        monkeypatch.setattr(cvxpy, "Problem", boom)
        y, X = _data(0)
        b, a = l2_relax(y[:30], X[:30], 0.1)
        assert b.shape == (X.shape[1],) and np.isfinite(a)

    def test_fit_l2_without_cvxpy(self, monkeypatch):
        import cvxpy

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("native fit_l2 must not call cvxpy")

        monkeypatch.setattr(cvxpy, "Problem", boom)
        y, X = _data(0)
        beta, intercept, cf, tau = fit_l2(y, X, T0=30)
        assert cf.shape == (X.shape[0],)
        assert np.isfinite(tau)

    def test_module_does_not_import_cvxpy(self):
        import mlsynth.utils.pda_helpers.l2.estimation as est

        assert not hasattr(est, "cp")
