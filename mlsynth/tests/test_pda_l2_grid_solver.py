"""What the L2-relaxation tau grid is for, and what it is allowed to return.

The grid exists to rank candidate ``tau`` by out-of-sample validation error.
Its winner is then refit on the full pre-period, and that refit is the estimate
the benchmarks pin value-for-value. So the two solves have different jobs: the
final one has to be right to the digit, the grid only has to order 80 numbers.

Two things follow, and these tests pin both.

The grid does not need the final solve's tolerance. Solved to ``1e-9`` the grid
is where an ``l2`` fit spends essentially all of its time -- the small-``tau``
end drives OSQP to its iteration cap -- and the ordering it produces is the same
one a looser solve produces.

A solve that failed is not a fit. OSQP answers with a status, and on an
ill-conditioned pre-period it can return an infeasibility certificate whose
``x`` is finite, enormous, and not a solution to the problem at all. An
infeasibility certificate must never reach the validation ranking; an imprecise
solution is a different matter and stays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pda_helpers.l2 import batch as l2_batch
from mlsynth.utils.pda_helpers.l2.batch import (
    _GRID_EPS,
    _SINGLE_EPS,
    _usable_status,
    l2_relax_batch,
    l2_relax_grid,
    l2_relax_solve,
)
from mlsynth.utils.pda_helpers.l2.estimation import (
    _auto_tau_grid,
    _destandardize,
    _standardize,
    cross_validate_tau,
    fit_l2,
    l2_relax,
)

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"


# ======================================================================
# Panels: the Hong Kong donors, truncated to reach the regime where the
# donor pool is as large as the pre-period and Sigma goes singular.
# ======================================================================
def _hong_kong():
    d = pd.read_csv(_BASEDATA / "HongKong.csv")
    times = np.sort(pd.unique(d["Time"]))
    wide = d.pivot(index="Time", columns="Country", values="GDP").reindex(times)
    donors = [u for u in wide.columns if u != "Hong Kong"]
    y = wide["Hong Kong"].to_numpy(float)
    X = wide[donors].to_numpy(float)
    T0 = int(np.sum(d[d["Country"] == "Hong Kong"]["Integration"].to_numpy() == 0))
    return y, X, T0


def _sim(T0, N, seed, n_factor=4):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T0 + 10, n_factor))
    X = F @ rng.normal(size=(n_factor, N)) + rng.normal(size=(T0 + 10, N))
    y = X @ (rng.normal(size=N) / N) * 3 + 0.4 * rng.normal(size=T0 + 10)
    return y[:T0], X[:T0]


def _moments(y_pre, X_pre, standardize=True):
    T1 = X_pre.shape[0]
    mu_y, Mu_X, sd_y, Sd_X = _standardize(y_pre, X_pre, standardize)
    Xt = (X_pre - Mu_X) / Sd_X
    Sigma = Xt.T @ Xt / T1
    eta = Xt.T @ ((y_pre - mu_y) / sd_y) / T1
    return Sigma, eta


@pytest.fixture(scope="module")
def hk():
    return _hong_kong()


# The window where OSQP returns a false infeasibility certificate: 24 Hong Kong
# donors on 24 pre-periods, so Sigma has rank 23 and the small-tau end of the
# grid is badly conditioned. The certificate is false -- the smallest reachable
# ||eta - Sigma beta||_inf is ~1e-12, far below the tau it fires at.
_ILL_T1 = 24


# ======================================================================
# Smoke
# ======================================================================
class TestSmoke:

    def test_cross_validation_returns_a_usable_tau(self, hk):
        y, X, T0 = hk
        tau = cross_validate_tau(y[:T0], X[:T0])
        assert np.isfinite(tau) and tau > 0

    def test_fit_runs_end_to_end(self, hk):
        y, X, T0 = hk
        beta, intercept, cf, tau = fit_l2(y, X, T0)
        assert beta.shape == (X.shape[1],)
        assert cf.shape == (X.shape[0],) and np.all(np.isfinite(cf))
        assert np.isfinite(intercept) and tau > 0


# ======================================================================
# The two tolerances have two jobs
# ======================================================================
class TestTolerances:

    def test_the_grid_is_looser_than_the_final_solve(self):
        assert _GRID_EPS > _SINGLE_EPS

    def test_the_single_solve_keeps_its_tolerance(self, hk):
        """The estimate the benchmarks pin is still solved tight.

        ``l2_relax`` is the fit that gets reported, so it goes on solving the
        primal to ``_SINGLE_EPS``; only the ranking sweep was loosened.
        """
        y, X, T0 = hk
        Sigma, eta = _moments(y[:T0], X[:T0])
        tau = 0.1 * float(np.abs(eta).max())
        beta = l2_relax_solve(Sigma, eta, tau)
        viol = float(np.max(np.abs(eta - Sigma @ beta)) - tau)
        assert viol <= 1e-7, f"single solve left the constraint violated by {viol:.2e}"

    @pytest.mark.parametrize("T1", [44, 36, 30, 26])
    def test_the_grid_coefficients_track_a_tight_grid(self, hk, T1):
        """Typical agreement is ~1e-5; the hardest tau on a thin window is not.

        At ``T1 = 26`` the donor pool is nearly as large as the pre-period, and
        one interior tau -- where the tight sweep itself needs 15,850 iterations
        to converge -- differs by ~1e-2 on a coefficient of ~0.9. That is a real
        tolerance effect, not a truncated solve, and it is the reason this test
        bounds the bulk of the sweep and leaves the ranking to the test below.
        """
        y, X, _ = hk
        Sigma, eta = _moments(y[:T1], X[:T1])
        grid = _auto_tau_grid(y[:T1], X[:T1], True, 12)
        loose = l2_relax_batch(Sigma, eta[:, None], grid, eps=_GRID_EPS)[0]
        tight = l2_relax_batch(Sigma, eta[:, None], grid, eps=_SINGLE_EPS)[0]
        ok = np.all(np.isfinite(loose), axis=1) & np.all(np.isfinite(tight), axis=1)
        assert ok.any()
        per_tau = np.abs(loose[ok] - tight[ok]).max(axis=1)
        assert np.median(per_tau) < 1e-4
        assert np.quantile(per_tau, 0.75) < 1e-3

    @pytest.mark.parametrize("T1", [44, 36, 30, 26])
    def test_the_grid_orders_taus_like_a_tight_grid(self, hk, T1):
        """The ordering is the grid's entire output, and it is identical.

        Coefficients at a given tau may differ in the last digits the sweep
        resolves; the validation errors they produce still sort the candidates
        the same way, which is the only thing the caller reads.
        """
        y, X, _ = hk
        y_pre, X_pre = y[:T1], X[:T1]
        n_val = max(2, int(round(0.2 * T1)))
        yt, Xt, yv, Xv = y_pre[:-n_val], X_pre[:-n_val], y_pre[-n_val:], X_pre[-n_val:]
        Sigma, eta = _moments(yt, Xt)
        mu_y, Mu_X, sd_y, Sd_X = _standardize(yt, Xt, True)
        grid = _auto_tau_grid(yt, Xt, True, 20)

        def mses(eps):
            out = np.full(len(grid), np.inf)
            for i, bt in enumerate(l2_relax_batch(Sigma, eta[:, None], grid, eps=eps)[0]):
                if np.all(np.isfinite(bt)):
                    b, a = _destandardize(bt, mu_y, Mu_X, sd_y, Sd_X)
                    out[i] = float(np.mean((yv - (Xv @ b + a)) ** 2))
            return out

        loose, tight = mses(_GRID_EPS), mses(_SINGLE_EPS)
        np.testing.assert_array_equal(np.argsort(loose), np.argsort(tight))

    @pytest.mark.parametrize("T1", [44, 36, 30])
    def test_the_selected_tau_is_the_one_a_tight_grid_selects(self, hk, T1, monkeypatch):
        """The ranking is what the grid contributes, and it does not move."""
        y, X, _ = hk
        loose = cross_validate_tau(y[:T1], X[:T1])
        monkeypatch.setattr(l2_batch, "_GRID_EPS", _SINGLE_EPS)
        tight = cross_validate_tau(y[:T1], X[:T1])
        assert loose == pytest.approx(tight, rel=1e-12)

    @pytest.mark.parametrize("T0,N,seed", [(40, 30, 1), (40, 60, 2), (30, 80, 3)])
    def test_the_selected_tau_holds_on_simulated_panels(self, T0, N, seed, monkeypatch):
        y_pre, X_pre = _sim(T0, N, seed)
        loose = cross_validate_tau(y_pre, X_pre)
        monkeypatch.setattr(l2_batch, "_GRID_EPS", _SINGLE_EPS)
        tight = cross_validate_tau(y_pre, X_pre)
        assert loose == pytest.approx(tight, rel=1e-12)


# ======================================================================
# A failed solve is not a fit
# ======================================================================
class TestSolverStatus:

    @pytest.mark.parametrize("status", ["solved", "solved inaccurate",
                                        "maximum iterations reached",
                                        "run time limit reached"])
    def test_a_solution_is_kept_however_imprecise(self, status):
        """Imprecision is not failure: these iterates solve the right problem."""
        assert _usable_status(status) is True

    @pytest.mark.parametrize("status", ["primal infeasible",
                                        "primal infeasible inaccurate",
                                        "dual infeasible",
                                        "dual infeasible inaccurate",
                                        "problem non convex",
                                        "interrupted"])
    def test_a_certificate_is_not_a_solution(self, status):
        """An infeasibility certificate answers a different question.

        Its ``x`` is finite, so a finiteness check admits it; on the Hong Kong
        donors it arrives with entries of order 1e9.
        """
        assert _usable_status(status) is False

    def test_the_false_certificate_never_reaches_the_ranking(self, hk):
        """The panel that produced a beta of 2.1e+09 now yields no beta at all."""
        y, X, _ = hk
        Sigma, eta = _moments(y[:_ILL_T1], X[:_ILL_T1])
        grid = _auto_tau_grid(y[:_ILL_T1], X[:_ILL_T1], True, 40)
        B = l2_relax_grid(Sigma, eta, grid)
        finite = np.all(np.isfinite(B), axis=1)
        assert not finite.all(), "expected the certificate tau to be rejected"
        assert np.abs(B[finite]).max() < 1e3

    def test_a_rejected_solve_is_reported_not_returned_as_zero(self, hk):
        """The refusal is a NaN row, which the callers translate into an error.

        Returning zeros would be worse than the huge beta it replaces: an
        all-zero beta is a legitimate-looking fit that the validation ranking
        would happily score.
        """
        y, X, _ = hk
        Sigma, eta = _moments(y[:_ILL_T1], X[:_ILL_T1])
        grid = _auto_tau_grid(y[:_ILL_T1], X[:_ILL_T1], True, 40)
        B = l2_relax_grid(Sigma, eta, grid)
        bad = ~np.all(np.isfinite(B), axis=1)
        assert bad.any()
        assert np.isnan(B[bad]).all()

    def test_a_rejected_single_solve_raises(self, hk, monkeypatch):
        """``l2_relax`` must fail loud, not hand back a rejected solve."""
        monkeypatch.setattr(l2_batch, "_usable_status", lambda s: False)
        y, X, T0 = hk
        with pytest.raises(MlsynthEstimationError, match="did not converge"):
            l2_relax(y[:T0], X[:T0], tau=0.05)

    def test_cross_validation_survives_a_rejected_tau(self, hk):
        """A rejected tau drops out of the ranking; the rest still competes."""
        y, X, _ = hk
        tau = cross_validate_tau(y[:_ILL_T1], X[:_ILL_T1])
        assert np.isfinite(tau) and tau > 0

    def test_cross_validation_raises_when_the_whole_grid_is_rejected(
            self, hk, monkeypatch):
        """No usable fit anywhere is a failure, not a silent pick of grid[0].

        Every candidate scores ``inf``, and ``argmin`` over all-``inf`` returns
        the first index -- a tau chosen by position.
        """
        monkeypatch.setattr(l2_batch, "_usable_status", lambda s: False)
        y, X, T0 = hk
        with pytest.raises(MlsynthEstimationError, match="no usable fit"):
            cross_validate_tau(y[:T0], X[:T0])


# ======================================================================
# Invariants that do not depend on how the grid is solved
# ======================================================================
class TestInvariants:

    @pytest.mark.parametrize("T1", [44, 30])
    def test_returned_coefficients_satisfy_the_constraint(self, hk, T1):
        y, X, _ = hk
        Sigma, eta = _moments(y[:T1], X[:T1])
        grid = _auto_tau_grid(y[:T1], X[:T1], True, 10)
        for tau, beta in zip(grid, l2_relax_grid(Sigma, eta, grid)):
            if np.all(np.isfinite(beta)):
                assert float(np.max(np.abs(eta - Sigma @ beta))) <= tau * (1 + 1e-3) + 1e-6

    def test_a_tau_above_the_moments_gives_the_zero_solution(self, hk):
        """``beta = 0`` is feasible once ``tau >= max|eta|``, so it is optimal."""
        y, X, T0 = hk
        Sigma, eta = _moments(y[:T0], X[:T0])
        beta = l2_relax_solve(Sigma, eta, float(np.abs(eta).max()) * 1.5)
        assert np.abs(beta).max() < 1e-6

    @pytest.mark.parametrize("T1", [44, 30])
    def test_the_norm_falls_as_tau_rises(self, hk, T1):
        """A looser constraint admits the tighter ball, so ||beta|| is monotone."""
        y, X, _ = hk
        Sigma, eta = _moments(y[:T1], X[:T1])
        grid = np.sort(_auto_tau_grid(y[:T1], X[:T1], True, 10))
        norms = [float(b @ b) for b in l2_relax_grid(Sigma, eta, grid)
                 if np.all(np.isfinite(b))]
        assert all(b <= a + 1e-6 for a, b in zip(norms, norms[1:]))

    def test_a_supplied_grid_is_the_grid_searched(self, hk):
        y, X, T0 = hk
        grid = np.array([0.02, 0.05, 0.2])
        tau = cross_validate_tau(y[:T0], X[:T0], tau_grid=grid)
        assert tau in set(grid.tolist())

    def test_more_donors_than_pre_periods(self):
        """The paper's regime: Sigma is singular and the fit still returns."""
        y_pre, X_pre = _sim(20, 60, 11)
        beta, intercept, cf, tau = fit_l2(
            np.concatenate([y_pre, y_pre[:5]]),
            np.vstack([X_pre, X_pre[:5]]), 20)
        assert np.all(np.isfinite(beta)) and np.all(np.isfinite(cf))

    def test_a_donor_with_no_variation_does_not_break_the_fit(self, hk):
        y, X, T0 = hk
        X = X.copy()
        X[:, 0] = 3.0                        # constant column: sd 0
        beta, intercept, cf, tau = fit_l2(y, X, T0)
        assert np.all(np.isfinite(beta)) and np.all(np.isfinite(cf))
