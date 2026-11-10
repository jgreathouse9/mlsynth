"""SparseSC's robust lambda sweep returns, at each grid point, the best V it computed (#1037).

The forward pass, the backward continuation and the outer restarts each try a few
starts per grid point and keep the best place L-BFGS-B ends. Two things were missing.
A start was never itself a candidate, so a run that ended above its start (scipy can
return ``x`` from another iterate after a failed line search) discarded the better
point. And no grid point was compared against the V's found at the others. On the
restart tests' factor panels (9-point grid, training window), 17 of 40 panels
returned a grid point whose V was beaten at its own lambda by another grid point's V,
by up to 60 percent of the outer objective (seed 5). On the Proposition 99 benchmark
specification it was 16 of 50 grid points, and the selected lambda, and with it the
number of predictors kept, followed whichever path the search happened to take.

The sweep now scores every start as a candidate and, after its passes, settles the
path: wherever another grid point's V is lower at a grid point's lambda, the outer
problem is solved again from it, until no grid point is beaten.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

import mlsynth.utils.sparse_sc_helpers.optimization as optimization
from mlsynth.utils.sparse_sc_helpers.objective import outer_loss
from mlsynth.utils.sparse_sc_helpers.optimization import sweep_lambda

GRID = np.concatenate([[0.0], np.logspace(-4, 0, 8)])
T0_TRAIN = 10


def _arrays(seed, N=8, P=10, T=20, T0=14):
    """The restart tests' factor panel, as ``sweep_lambda`` takes it."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 3))
    L = rng.normal(size=(N + 1, 3))
    Y = F @ L.T + 0.35 * rng.normal(size=(T, N + 1))
    X = rng.normal(size=(P, N + 1))
    X[0] = Y[:T0].mean(axis=0)
    X[1] = Y[: T0 // 2].mean(axis=0)
    sd = X.std(axis=1, ddof=1)
    sd[sd == 0] = 1.0
    X = X / sd[:, None]
    return Y[:, 0], Y[:, 1:], X[:, 0], X[:, 1:], T0


def _sweep(seed, grid=GRID, **kw):
    Y1, Y0, X1, X0, T0 = _arrays(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sweep_lambda(X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0, T0_train=T0_TRAIN,
                           lambda_grid=grid, outer_loss_window="training", outcome_scale=1.0,
                           use_analytical_grad=True, **kw)
    return out, (X1, X0, Y1[:T0_TRAIN], Y0[:T0_TRAIN])


def _beaten(out, data):
    """Grid points whose reported V is beaten, at their lambda, by another grid point's V."""
    _, _, grid, outer, _, vp = out
    X1, X0, Z1, Z0 = data
    found = []
    for i, lam in enumerate(grid):
        f_i = outer_loss(vp[i][1:], X1, X0, Z1, Z0, float(lam))
        best = min(outer_loss(vp[j][1:], X1, X0, Z1, Z0, float(lam)) for j in range(len(grid)))
        if best < f_i * (1.0 - 1e-9):
            found.append((i, f_i, best))
    return found


# --------------------------------------------------------------------------- smoke
def test_a_robust_sweep_returns_a_settled_path():
    out, data = _sweep(0)
    assert np.isfinite(out[3]).all() and not _beaten(out, data)


# --------------------------------------------------------------------------- unit
def test_the_worst_panel_is_settled():
    """Seed 5: six grid points beaten by up to 60 percent before the settle pass."""
    out, data = _sweep(5)
    assert not _beaten(out, data)


def test_the_outer_curve_is_the_objective_at_the_reported_v():
    out, (X1, X0, Z1, Z0) = _sweep(5)
    _, _, grid, outer, _, vp = out
    for i, lam in enumerate(grid):
        assert outer[i] == pytest.approx(outer_loss(vp[i][1:], X1, X0, Z1, Z0, float(lam)), rel=1e-12)


def test_the_validation_curve_is_read_at_the_settled_v():
    """Selection reads val_curve, so a settled V must carry its own validation MSE."""
    from mlsynth.utils.sparse_sc_helpers.objective import selection_mse
    Y1, Y0, X1, X0, T0 = _arrays(5)
    out, _ = _sweep(5)
    _, lam, grid, _, val, vp = out
    for i in range(len(grid)):
        assert val[i] == pytest.approx(
            selection_mse(vp[i][1:], X1, X0, Y1[T0_TRAIN:T0], Y0[T0_TRAIN:T0]), rel=1e-12)
    assert lam == grid[int(np.nanargmin(val))]


def test_a_start_is_a_candidate(monkeypatch):
    """A run that ends above its start keeps the start."""
    real = optimization._minimize_outer

    def worse(**kw):
        res = real(**kw)
        res.x = np.asarray(kw["x0"], dtype=float) + 1e3   # the L1 penalty alone puts it far above
        return res
    monkeypatch.setattr(optimization, "_minimize_outer", worse)
    out, data = _sweep(1, grid=np.array([0.01]), robust=False)
    X1, X0, Z1, Z0 = data
    v_cold = optimization.default_v20(X0)
    assert out[3][0] <= outer_loss(v_cold, X1, X0, Z1, Z0, 0.01) * (1.0 + 1e-12)


def test_the_settle_pass_never_raises_a_grid_point():
    """Settling only replaces a V by a lower one: compare against the unsettled sweep."""
    out, data = _sweep(14)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(optimization, "_SETTLE_ROUNDS", 0)
        raw, _ = _sweep(14)
    assert np.all(out[3] <= raw[3] * (1.0 + 1e-12))


# --------------------------------------------------------------------------- edge
def test_a_single_grid_point_needs_no_settling():
    out, data = _sweep(2, grid=np.array([0.01]))
    assert not _beaten(out, data)


def test_the_fast_sweep_is_not_settled_by_default():
    """robust=False keeps the single pass, as documented."""
    with pytest.MonkeyPatch.context() as mp:
        calls = []
        real = optimization._settle
        mp.setattr(optimization, "_settle", lambda *a, **k: calls.append(1) or real(*a, **k))
        _sweep(3, robust=False)
    assert not calls


def test_a_grid_point_that_never_solved_is_left_to_the_sweeps_report():
    """_settle skips a non-finite objective: that lambda failed, and the sweep says so."""
    grid = np.array([0.0, 0.1])
    v_path = np.array([[1.0, 2.0], [1.0, 0.5]])
    outer = np.array([np.inf, 1.0])
    calls = []
    changed = optimization._settle(grid, v_path, outer, lambda v, lam: float(v[0]),
                                   lambda i, v: calls.append(i) or (v, float(v[0])), rounds=3)
    assert 0 not in changed and 0 not in calls and outer[0] == np.inf


def test_a_start_whose_score_raises_is_skipped_not_fatal(monkeypatch):
    """The first outer_loss call scores the start; it raising leaves the run's own point."""
    real = optimization.outer_loss
    n = [0]

    def flaky(*a, **k):
        n[0] += 1
        if n[0] == 1:
            raise ValueError("start refused")
        return real(*a, **k)
    monkeypatch.setattr(optimization, "outer_loss", flaky)
    out, _ = _sweep(1, grid=np.array([0.01]), robust=False)
    assert np.isfinite(out[3][0])


# --------------------------------------------------------------------------- failure
def test_a_settle_pass_that_hits_its_cap_says_so(monkeypatch):
    monkeypatch.setattr(optimization, "_SETTLE_ROUNDS", 1)
    Y1, Y0, X1, X0, T0 = _arrays(5)
    with pytest.warns(RuntimeWarning, match="settle"):
        sweep_lambda(X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0, T0_train=T0_TRAIN, lambda_grid=GRID,
                     outer_loss_window="training", outcome_scale=1.0, use_analytical_grad=True)


# --------------------------------------------------------------------------- property
@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10_000))
@example(seed=5)
def test_property_no_grid_point_is_beaten_by_another(seed):
    out, data = _sweep(seed)
    assert not _beaten(out, data)
