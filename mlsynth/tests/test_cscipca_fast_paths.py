"""Tests for the CSC-IPCA fast paths: speed only, answers unchanged.

Three sites in CSC-IPCA spent more time on per-call overhead than on the
arithmetic they wrapped. Each is replaced by a form that computes the same
quantity in one vectorized step:

1. ``_moving_block_pvalue`` built its ``T`` circular blocks with a ``np.roll``
   per shift. The blocks are the windows of the doubled residual vector, so one
   strided view replaces the loop. Bit-identical, and pinned as such below.
2. ``solve_factors`` solved its ``T`` per-period ``K x K`` systems in a Python
   loop of ``np.linalg.lstsq`` calls. One batched LU replaces the loop, with the
   per-period minimum-norm loop kept as the fallback a rank-deficient period
   needs.
3. ``solve_gamma`` and ``solve_factors`` passed ``optimize=True`` to every
   ``einsum``, which re-runs the contraction-path search on each call. The path
   depends only on the subscripts and the operand shapes, so it is cached.

The oracles in this file are the previous implementations. Sites 1 and 3 are
asserted bit-identical to them (``array_equal``, not ``allclose``): both are
the same arithmetic in the same order, so anything looser would hide a
reordering. Site 2 is a different factorization -- LU where the loop ran an SVD
-- so it is asserted to machine precision and its fallback is asserted to
reproduce the loop exactly.

The guard on ``block`` is new behaviour, not a reorganization. The roll form
read ``u[-block:]`` which silently returns the whole vector for ``block <= 0``
and for ``block > T``, so a caller asking for a window the series cannot hold
got a p-value of 1.0 -- the most conservative answer available, from a question
nobody asked. It raises now, per the fail-early rule in CLAUDE.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.cscipca_helpers import als
from mlsynth.utils.cscipca_helpers.als import solve_factors, solve_gamma
from mlsynth.utils.cscipca_helpers.inference import _moving_block_pvalue


# --------------------------------------------------------------------------- #
# reference oracles -- the implementations these fast paths replace
# --------------------------------------------------------------------------- #
def _roll_stats(u: np.ndarray, block: int) -> np.ndarray:
    """Previous statistic: one ``np.roll`` per circular shift."""
    return np.array([np.roll(u, s)[-block:].mean() for s in range(u.shape[0])])


def _roll_pvalue(resid: np.ndarray, block: int) -> float:
    """Previous ``_moving_block_pvalue``, verbatim."""
    u = np.abs(np.asarray(resid, dtype=float))
    stats = _roll_stats(u, block)
    return float(np.mean(stats >= stats[0]))


def _loop_factors(Y: np.ndarray, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Previous ``solve_factors`` inner loop: ``T`` minimum-norm lstsq solves."""
    T = X.shape[1]
    XG = np.einsum("itl,lk->itk", X, gamma, optimize=True)
    denom = np.einsum("itk,itj->tkj", XG, XG, optimize=True)
    numer = np.einsum("itk,it->tk", XG, Y, optimize=True)
    F = np.empty((T, XG.shape[2]))
    for t in range(T):
        F[t], *_ = np.linalg.lstsq(denom[t], numer[t], rcond=None)
    return F.T


def _panel(seed=0, N=8, T=15, L=3, K=2, collinear=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, T, L))
    if collinear:                       # what log GDP and log GDP-per-capita do
        X[:, :, 1] = X[:, :, 0]
    gamma = rng.uniform(-0.5, 0.5, (L, K))
    F = rng.standard_normal((K, T))
    Y = np.einsum("itl,lk,kt->it", X, gamma, F)
    return Y, X, gamma, F, K


# ======================================================================
# 1. the moving-block p-value
# ======================================================================
class TestMovingBlockPvalue:
    @pytest.mark.parametrize("T", [1, 2, 5, 12, 22, 41])
    @pytest.mark.parametrize("scale", [1e-8, 1.0, 1e7])
    def test_bit_identical_to_the_roll_oracle(self, T, scale):
        rng = np.random.default_rng(T * 17 + int(np.log10(scale)))
        for _ in range(12):
            resid = rng.standard_normal(T) * scale
            for block in {1, 2, 3, max(1, T // 2), T}:
                if not 1 <= block <= T:
                    continue
                assert _moving_block_pvalue(resid, block) == _roll_pvalue(resid, block)

    def test_block_one_is_the_reversed_absolute_residual(self):
        # np.roll(u, s)[-1] == u[T-1-s], so the statistic sequence is |u| reversed
        # and the p-value is the share of periods at least as extreme as the last.
        rng = np.random.default_rng(3)
        u = np.abs(rng.standard_normal(20))
        expected = float(np.mean(u >= u[-1]))
        assert _moving_block_pvalue(u, 1) == expected

    def test_a_full_length_block_scores_every_shift_at_the_series_mean(self):
        # Every circular block of length T is the whole series, so in exact
        # arithmetic all T statistics equal mean(|u|). In floating point they are
        # summed from different offsets, so the p-value can fall below 1 on a
        # wide-dynamic-range series -- a property of the statistic, which the
        # roll form shares. The claim that holds is agreement with that mean.
        rng = np.random.default_rng(4)
        u = np.abs(rng.standard_normal(9))
        assert _moving_block_pvalue(u, 9) == pytest.approx(1.0)
        wide = np.array([1.8685236347373575, 48576.86852363474, 999999.530581009])
        assert _moving_block_pvalue(wide, 3) == _roll_pvalue(wide, 3)
        assert np.allclose(_roll_stats(np.abs(wide), 3), np.abs(wide).mean(),
                           rtol=0, atol=1e-9)

    def test_the_post_block_always_counts_itself(self):
        rng = np.random.default_rng(5)
        for T in (3, 8, 17):
            for _ in range(20):
                u = rng.standard_normal(T)
                for block in (1, 2, min(3, T)):
                    p = _moving_block_pvalue(u, block)
                    assert 1.0 / T - 1e-12 <= p <= 1.0

    def test_sign_and_positive_scale_leave_the_p_value_alone(self):
        rng = np.random.default_rng(6)
        u = rng.standard_normal(14)
        for block in (1, 3, 7):
            base = _moving_block_pvalue(u, block)
            assert _moving_block_pvalue(-u, block) == base
            assert _moving_block_pvalue(4.25 * u, block) == base

    def test_a_single_period_is_certain(self):
        assert _moving_block_pvalue(np.array([0.0]), 1) == 1.0
        assert _moving_block_pvalue(np.array([-3.5]), 1) == 1.0

    def test_zero_residuals_give_a_p_value_of_one(self):
        assert _moving_block_pvalue(np.zeros(7), 1) == 1.0
        assert _moving_block_pvalue(np.zeros(7), 3) == 1.0

    # ---- failure: a window the series cannot hold is reported, not clamped ----
    @pytest.mark.parametrize("block", [0, -1, -4])
    def test_a_non_positive_block_is_reported(self, block):
        with pytest.raises(MlsynthEstimationError, match="block"):
            _moving_block_pvalue(np.arange(6.0), block)

    @pytest.mark.parametrize("block", [7, 8, 40])
    def test_a_block_longer_than_the_series_is_reported(self, block):
        with pytest.raises(MlsynthEstimationError, match="block"):
            _moving_block_pvalue(np.arange(6.0), block)

    def test_an_empty_residual_is_reported(self):
        with pytest.raises(MlsynthEstimationError, match="block"):
            _moving_block_pvalue(np.array([]), 1)

    def test_the_rejected_block_used_to_return_a_p_value(self):
        # The regression this guard closes: the roll form answered 1.0 instead
        # of refusing, so an out-of-range block looked like a null that could
        # not be rejected. Pinned so the guard is not "simplified" away.
        assert _roll_pvalue(np.arange(6.0), 99) == 1.0


# ======================================================================
# 2. the batched F-step
# ======================================================================
class TestSolveFactors:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_the_per_period_loop(self, seed):
        Y, X, gamma, _F, _K = _panel(seed)
        got, want = solve_factors(Y, X, gamma), _loop_factors(Y, X, gamma)
        assert got.shape == want.shape
        assert np.allclose(got, want, rtol=0, atol=1e-12)

    def test_satisfies_the_per_period_normal_equations(self):
        Y, X, gamma, _F, K = _panel(7)
        F = solve_factors(Y, X, gamma)
        XG = np.einsum("itl,lk->itk", X, gamma, optimize=True)
        for t in range(X.shape[1]):
            A = XG[:, t, :].T @ XG[:, t, :]
            b = XG[:, t, :].T @ Y[:, t]
            assert np.allclose(A @ F[:, t], b, atol=1e-9)

    def test_a_rank_deficient_period_falls_back_instead_of_raising(self):
        # Zero the covariates at one period: that period's K x K system is
        # exactly singular, so the batched LU cannot be used for the batch.
        Y, X, gamma, _F, K = _panel(8)
        X = X.copy()
        X[:, 4, :] = 0.0
        F = solve_factors(Y, X, gamma)
        assert F.shape == (K, X.shape[1])
        assert np.all(np.isfinite(F))
        # The fallback is the loop, exactly -- including the zero it returns for
        # the degenerate period (minimum-norm solution of 0 @ f = 0).
        assert np.array_equal(F, _loop_factors(Y, X, gamma))
        assert np.allclose(F[:, 4], 0.0)

    def test_a_near_singular_period_is_not_passed_through_amplified(self):
        """The defect the conditioning gate exists for.

        This panel's second period is near-singular but not singular, so
        ``np.linalg.solve`` neither raises nor truncates: it returns 7.2e227
        where the minimum-norm solution is 0.33. A ``try/except LinAlgError``
        around the batched solve therefore does not catch it -- the amplified
        factors flow into the counterfactual and the ATT with nothing reported.
        """
        Y = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        X = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                       [1.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
        gamma = np.array([[2.57296425e-245, 1.0], [0.0, 0.0]])
        # What the unguarded batched solve would have answered.
        XG = np.einsum("itl,lk->itk", X, gamma, optimize=True)
        denom = np.einsum("itk,itj->tkj", XG, XG, optimize=True)
        numer = np.einsum("itk,it->tk", XG, Y, optimize=True)
        unguarded = np.linalg.solve(denom, numer[..., None])[..., 0].T
        assert np.abs(unguarded).max() > 1e200, "the hazard this test pins is gone"

        got = solve_factors(Y, X, gamma)
        assert np.abs(got).max() < 1.0
        assert np.array_equal(got, _loop_factors(Y, X, gamma))

    def test_collinear_covariates_still_match_the_loop(self):
        # L > K, so duplicating a covariate leaves each period's K x K system
        # full rank; the Gamma-step system is the one that goes rank-deficient.
        Y, X, gamma, _F, _K = _panel(9, collinear=True)
        assert np.allclose(solve_factors(Y, X, gamma),
                           _loop_factors(Y, X, gamma), atol=1e-12)

    def test_an_all_zero_covariate_cube_gives_zero_factors(self):
        Y, X, gamma, _F, K = _panel(10)
        F = solve_factors(Y, np.zeros_like(X), gamma)
        assert F.shape == (K, X.shape[1])
        assert np.allclose(F, 0.0)

    def test_a_single_period_panel(self):
        Y, X, gamma, _F, K = _panel(11, T=1)
        F = solve_factors(Y, X, gamma)
        assert F.shape == (K, 1) and np.all(np.isfinite(F))


# ======================================================================
# 2b. the work the fast paths remove
# ======================================================================
class TestBoundedWork:
    """The equivalence tests above hold for the loop too, so they cannot show
    the loop is gone. These count the calls instead.

    A call count is the assertion a timing threshold only approximates: it is
    the quantity that scales with the panel, it does not move with the machine,
    and it fails for a stated reason. ``O(T)`` LAPACK calls per F-step and
    ``O(T)`` rolls per p-value are what made these sites the hot ones.
    """

    def test_the_f_step_makes_one_lapack_call_not_one_per_period(self, monkeypatch):
        Y, X, gamma, _F, _K = _panel(19, T=17)
        calls = []
        real = np.linalg.lstsq
        monkeypatch.setattr(np.linalg, "lstsq",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        F = solve_factors(Y, X, gamma)
        assert F.shape[1] == 17
        assert len(calls) == 0, (
            f"a full-rank panel should need no lstsq at all; made {len(calls)}")

    def test_the_f_step_falls_back_once_for_the_whole_batch(self, monkeypatch):
        # A degenerate period costs the loop, but only after the batched attempt
        # fails -- and the fallback is one pass, not one retry per period.
        Y, X, gamma, _F, _K = _panel(20, T=11)
        X = X.copy()
        X[:, 3, :] = 0.0
        calls = []
        real = np.linalg.lstsq
        monkeypatch.setattr(np.linalg, "lstsq",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        solve_factors(Y, X, gamma)
        assert len(calls) == 11, f"expected one pass of T=11 solves, got {len(calls)}"

    def test_the_p_value_does_not_roll_once_per_shift(self, monkeypatch):
        rng = np.random.default_rng(21)
        u = rng.standard_normal(40)
        calls = []
        real = np.roll
        monkeypatch.setattr(np, "roll",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        _moving_block_pvalue(u, 3)
        assert len(calls) == 0, f"the window form should not roll; rolled {len(calls)}x"


# ======================================================================
# 3. the cached einsum contraction paths
# ======================================================================
class TestEinsumPathCache:
    def test_the_cached_path_is_the_greedy_path(self):
        # optimize=True is numpy's 'greedy', so caching it changes the planning
        # and not the contraction order.
        Y, X, _g, F, K = _panel(12)
        G = np.einsum("itl,itm->tlm", X, X, optimize=True)
        for sub, ops in (
            ("it,itl,kt->lk", (Y, X, F)),
            ("itl,itm->tlm", (X, X)),
            ("tlm,kt,jt->lkmj", (G, F, F)),
        ):
            cached = np.einsum(sub, *ops, optimize=als._einsum_path(sub, *ops))
            assert np.array_equal(cached, np.einsum(sub, *ops, optimize=True))

    def test_solve_gamma_is_bit_identical_across_a_warm_cache(self):
        Y, X, _g, F, K = _panel(13)
        als._EINSUM_PATHS.clear()
        cold = solve_gamma(Y, X, F, K)
        assert als._EINSUM_PATHS, "the first call should populate the cache"
        warm = solve_gamma(Y, X, F, K)
        assert np.array_equal(cold, warm)

    def test_the_cache_is_keyed_on_the_operand_shapes(self):
        # A shape-blind key would reuse one panel's path on another's operands.
        # Interleave two shapes and demand both answers stay right.
        als._EINSUM_PATHS.clear()
        a = _panel(14, N=6, T=11, L=3, K=2)
        b = _panel(15, N=9, T=20, L=5, K=3)
        want_a = np.einsum("it,itl,kt->lk", a[0], a[1], a[3], optimize=True)
        want_b = np.einsum("it,itl,kt->lk", b[0], b[1], b[3], optimize=True)
        for _ in range(3):
            for panel, want in ((a, want_a), (b, want_b)):
                Y, X, _g, F, _K = panel
                got = np.einsum("it,itl,kt->lk", Y, X, F,
                                optimize=als._einsum_path("it,itl,kt->lk", Y, X, F))
                assert np.array_equal(got, want)
        assert len(als._EINSUM_PATHS) == 2

    def test_solve_gamma_matches_the_uncached_contraction(self):
        for seed in (16, 17):
            Y, X, _g, F, K = _panel(seed)
            N, T, L = X.shape
            numer = np.einsum("it,itl,kt->lk", Y, X, F, optimize=True).reshape(L * K)
            G = np.einsum("itl,itm->tlm", X, X, optimize=True)
            denom = np.einsum("tlm,kt,jt->lkmj", G, F, F,
                              optimize=True).reshape(L * K, L * K)
            want, *_ = np.linalg.lstsq(denom, numer, rcond=None)
            assert np.array_equal(solve_gamma(Y, X, F, K), want.reshape(L, K))


# ======================================================================
# 4. end to end -- the fast paths move no reported number
# ======================================================================
class TestFitUnchanged:
    def test_als_still_recovers_a_noiseless_fit(self):
        from mlsynth.utils.cscipca_helpers.als import als_estimate, counterfactual
        Y, X, _g, _F, K = _panel(18)
        F_hat, gamma_hat, n_iter, converged = als_estimate(Y, X, K, 200, 1e-10)
        assert converged and n_iter <= 200
        assert np.max(np.abs(counterfactual(X, gamma_hat, F_hat) - Y)) < 1e-6

    def test_the_brexit_path_is_unchanged(self):
        import pathlib
        import pandas as pd
        from mlsynth import CSCIPCA

        path = (pathlib.Path(__file__).resolve().parents[2]
                / "basedata" / "fdi_oecd_brexit.csv")
        if not path.exists():               # pragma: no cover - data ships with the repo
            pytest.skip("Brexit panel not present")
        covs = ["log_gdp", "log_gdp_percap", "import_to_gdp", "export_to_gdp",
                "inflation_gdp_deflator", "gross_capital_forma_gdp",
                "unemployment", "employment_15", "log_population"]
        res = CSCIPCA({
            "df": pd.read_csv(path), "outcome": "fdi", "treat": "treated",
            "unitid": "country", "time": "year", "covariates": covs,
            "n_factors": 2, "inference": False, "display_graphs": False,
        }).fit()
        years = np.asarray(res.time_series.time_periods)
        gap = np.asarray(res.time_series.estimated_gap, dtype=float)
        got = {int(y): float(gap[years == y][0]) for y in (2017, 2018, 2019)}
        # Wang (2024) Table: -7.8 / -12.9 / -18.3. The benchmark's tolerance is
        # 0.3; these are tighter, so a fast path that drifts is caught here
        # before the benchmark sees it.
        assert got[2017] == pytest.approx(-7.7633, abs=1e-3)
        assert got[2018] == pytest.approx(-12.9045, abs=1e-3)
        assert got[2019] == pytest.approx(-18.3413, abs=1e-3)
        assert res.metadata["converged"] is True
        assert res.metadata["n_iter"] == 49
