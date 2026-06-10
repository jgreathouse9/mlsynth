"""Tests for the multiple-treated-units L2-relaxation PDA + its batched solver."""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.pda_helpers.l2.batch import l2_relax_batch
from mlsynth.utils.pda_helpers.multitreat import run_pda_multitreat


class TestBatchedSolver:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 12)); Sigma = X.T @ X / 40
        Eta = np.column_stack([X.T @ rng.normal(size=40) / 40 for _ in range(3)])
        B = l2_relax_batch(Sigma, Eta, np.array([0.5, 0.1, 0.02]))
        assert B.shape == (3, 3, 12)

    def test_matches_cvxpy(self):
        cp = pytest.importorskip("cvxpy")
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 15)); Sigma = X.T @ X / 50
        Eta = np.column_stack([X.T @ rng.normal(size=50) / 50 for _ in range(2)])
        taus = np.array([0.4, 0.08])
        B = l2_relax_batch(Sigma, Eta, taus, eps=1e-7, max_iter=20000)
        for j in range(2):
            for k, tau in enumerate(taus):
                b = cp.Variable(15)
                cp.Problem(cp.Minimize(0.5 * cp.sum_squares(b)),
                           [cp.norm(Eta[:, j] - Sigma @ b, "inf") <= tau]).solve(solver="CLARABEL")
                assert np.abs(B[j, k] - np.asarray(b.value).ravel()).max() < 2e-3


class TestMultiTreat:
    def _panel(self, seed=1, T=60, T0=40, N=20, J=5, shift=2.0):
        rng = np.random.default_rng(seed)
        f = rng.normal(size=(T, 2))
        Xc = f @ rng.normal(size=(2, N)) + rng.normal(0, 0.3, (T, N))
        Yt = f @ rng.normal(size=(2, J)) + rng.normal(0, 0.3, (T, J))
        Yt[T0:] += shift                      # known post-period treatment effect
        return Yt, Xc, T0

    def test_structure_and_recovers_shift(self):
        Yt, Xc, T0 = self._panel(shift=2.0)
        grid = np.exp(np.linspace(np.log(1e-2), np.log(1.0), 6))
        r = run_pda_multitreat(Yt, Xc, T0, grid)
        assert r.ate.shape == (60 - T0,) and r.tau.shape == (5,)
        assert r.se > 0 and np.isfinite(r.se)
        assert r.ate_mean > 1.0               # recovers the +2 cross-sectional shift
        assert (r.pvalue[0] < 0.05)           # the effect is detected

    def test_null_panel_not_significant_on_average(self):
        Yt, Xc, T0 = self._panel(shift=0.0)
        grid = np.exp(np.linspace(np.log(1e-2), np.log(1.0), 6))
        r = run_pda_multitreat(Yt, Xc, T0, grid)
        assert abs(r.ate_mean) < 1.0          # no injected effect -> small ATE

    def test_no_standardize_runs(self):
        Yt, Xc, T0 = self._panel(shift=1.0)
        grid = np.array([0.05, 0.2, 0.5])
        r = run_pda_multitreat(Yt, Xc, T0, grid, standardize=False)
        assert r.ate.shape == (20,) and r.se > 0
