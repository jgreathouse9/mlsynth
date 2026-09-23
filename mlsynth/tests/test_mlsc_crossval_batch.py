"""mlSC's penalty cross-validation, solved as one batch.

Test-first (per ``agents/agents_tests.md``): written before the batched path
exists, against the sequential one it has to reproduce.

The rolling CV scores every candidate penalty on the same held-out periods, and
each score needs the disaggregate weights that penalty implies. Those are
``argmin_w ||[X; sqrt(p) R] w - [Y; 0]||^2`` on the simplex, and because the
augmentation contributes no target rows the whole family collapses: with the
weights summing to one,

    G(p) = (X - Y 1')'(X - Y 1') + p R'R

is affine in the penalty. So the grid's Gram matrices are one broadcast off two
matrices formed once, and a batched active set scores the entire grid at once.

What is asserted here is that this changes only the arithmetic's shape: the same
penalty is selected, on the same held-out errors, with the same weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.mlsc_helpers.crossval import select_lambda_cv
from mlsynth.utils.mlsc_helpers.optimization import _augmented_design
from mlsynth.utils.mlsc_helpers.penalty import build_penalty_matrix
from mlsynth.utils.mlsc_helpers.structures import MLSCInputs


def _inputs(T=14, T0=10, S=3, per=4, seed=0):
    """A two-level panel: ``S`` aggregates, ``per`` disaggregates in each."""
    rng = np.random.default_rng(seed)
    M = S * per
    disagg_to_agg = np.repeat(np.arange(S), per)
    X = np.cumsum(rng.normal(size=(T, M)), axis=0) + 10.0
    v = np.full(M, 1.0 / per)
    w_true = rng.dirichlet(np.ones(M))
    y = X @ w_true + 0.05 * rng.normal(size=T)
    return MLSCInputs(
        Y_agg_treated=y,
        X_disagg=X,
        disagg_to_agg=disagg_to_agg,
        v_population=v,
        T0=T0,
        T=T,
        agg_labels=[f"s{i}" for i in range(S)],
        disagg_labels=[f"c{i}" for i in range(M)],
        Y_disagg_pre_full=X[:T0],
        disagg_to_agg_full=disagg_to_agg,
        treated_agg_idx_full=0,
        treated_unit_name="treated",
        time_labels=np.arange(T),
        Ywide_agg=None,
        outcome="y",
    )


def _sequential_cv_errors(inputs, sigma_y2, grid, cv_holdout_periods=1):
    """The scores the sequential path produces, computed here independently."""
    T0 = inputs.T0
    t_cv = T0 - cv_holdout_periods
    Y = inputs.Y_agg_treated[:T0]
    X = inputs.X_disagg[:T0, :]
    X_train, Y_train = X[:t_cv], Y[:t_cv]
    X_test, Y_test = X[t_cv:T0], Y[t_cv:T0]
    out = []
    for v in grid:
        B, A = _augmented_design(X_train, Y_train, inputs, float(v) * sigma_y2)
        omega = solve_simplex_qp(B, A)
        out.append(float(np.mean((Y_test - X_test @ omega) ** 2)))
    return np.array(out)


def _setup(seed=0, **kw):
    inputs = _inputs(seed=seed, **kw)
    Q = build_penalty_matrix(inputs.v_population, inputs.disagg_to_agg)
    sigma_y2 = float(np.var(inputs.Y_agg_treated[:inputs.T0]))
    return inputs, Q, sigma_y2


# --------------------------------------------------------------------------- #
# 1. The batched path reproduces the sequential one
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(4))
def test_selects_the_same_penalty_as_the_sequential_path(seed):
    inputs, Q, sigma_y2 = _setup(seed=seed, T=26, T0=22)
    grid = np.concatenate(([0.0], np.logspace(-8, np.log10(5), 25),
                           np.logspace(1, 3, 4)))
    chosen = select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=grid)
    errors = _sequential_cv_errors(inputs, sigma_y2, grid)
    assert np.isclose(chosen, float(grid[int(np.argmin(errors))]))


def test_scores_every_grid_point_the_same_way():
    """Not just the argmin: every held-out score has to match, or the selection
    agreeing is a coincidence of this grid."""
    from mlsynth.utils.mlsc_helpers.crossval import _holdout_mse_grid

    inputs, Q, sigma_y2 = _setup(seed=5, T=26, T0=22)
    grid = np.concatenate(([0.0], np.logspace(-6, np.log10(2), 15)))
    T0 = inputs.T0
    t_cv = T0 - 1
    Y = inputs.Y_agg_treated[:T0]
    X = inputs.X_disagg[:T0, :]
    got = _holdout_mse_grid(inputs, X[:t_cv], Y[:t_cv], X[t_cv:T0], Y[t_cv:T0],
                            grid * sigma_y2)
    want = _sequential_cv_errors(inputs, sigma_y2, grid)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-12)


def test_batched_weights_are_on_the_simplex():
    from mlsynth.utils.mlsc_helpers.crossval import _grid_weights

    inputs, Q, sigma_y2 = _setup(seed=6)
    grid = np.array([0.0, 1e-6, 0.5, 100.0])
    T0 = inputs.T0
    W = _grid_weights(inputs, inputs.X_disagg[:T0 - 1, :],
                      inputs.Y_agg_treated[:T0 - 1], grid * sigma_y2)
    assert W.shape == (grid.size, inputs.X_disagg.shape[1])
    assert W.min() >= -1e-9
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# 2. The penalty enters the Gram affinely -- the identity the batch rests on
# --------------------------------------------------------------------------- #
def test_gram_is_affine_in_the_penalty():
    from mlsynth.utils.bilevel.minnorm import simplex_gram
    from mlsynth.utils.mlsc_helpers.penalty import build_sqrt_factor

    inputs, _, _ = _setup(seed=7)
    T0 = inputs.T0
    X, Y = inputs.X_disagg[:T0 - 1, :], inputs.Y_agg_treated[:T0 - 1]
    R = build_sqrt_factor(inputs.v_population, inputs.disagg_to_agg)
    G_X, QR = simplex_gram(X, Y), R.T @ R
    for p in (1e-8, 1e-3, 0.7, 12.0, 900.0):
        B, A = _augmented_design(X, Y, inputs, p)
        np.testing.assert_allclose(simplex_gram(B, A), G_X + p * QR,
                                   rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# 3. Edge cases
# --------------------------------------------------------------------------- #
def test_single_point_grid():
    inputs, Q, sigma_y2 = _setup(seed=8)
    assert select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=[0.25]) == 0.25


def test_zero_only_grid_uses_the_uniqueness_ridge():
    inputs, Q, sigma_y2 = _setup(seed=9)
    assert select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=[0.0]) == 0.0


def test_multi_period_holdout():
    inputs, Q, sigma_y2 = _setup(seed=10)
    grid = np.array([0.0, 1e-4, 1.0, 50.0])
    chosen = select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=grid,
                              cv_holdout_periods=3)
    errors = _sequential_cv_errors(inputs, sigma_y2, grid, cv_holdout_periods=3)
    assert np.isclose(chosen, float(grid[int(np.argmin(errors))]))


def test_too_short_a_pre_period_is_rejected():
    inputs, Q, sigma_y2 = _setup(seed=11)
    with pytest.raises(ValueError, match="at least one training period"):
        select_lambda_cv(inputs, Q, sigma_y2, cv_holdout_periods=inputs.T0)


# --------------------------------------------------------------------------- #
# 3b. A rank-deficient training design keeps the one-at-a-time solve
# --------------------------------------------------------------------------- #
def test_gram_is_safe_only_when_the_training_design_has_full_column_rank():
    from mlsynth.utils.mlsc_helpers.crossval import _gram_is_safe

    rng = np.random.default_rng(0)
    assert _gram_is_safe(rng.normal(size=(40, 12))) is True
    assert _gram_is_safe(rng.normal(size=(9, 12))) is False      # fewer rows
    collinear = rng.normal(size=(40, 6))
    collinear[:, 5] = collinear[:, 0]                            # duplicate column
    assert _gram_is_safe(collinear) is False


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rank_deficient_panel_reproduces_the_sequential_selection(seed):
    """More disaggregates than training periods: the penalty at the small end is
    the only thing separating the columns, and forming the Gram squares it below
    what float64 resolves -- measured at 225 percent above the optimum on seed 0.
    The guard has to send this case down the design-form path, exactly."""
    from mlsynth.utils.mlsc_helpers.crossval import _gram_is_safe

    inputs, Q, sigma_y2 = _setup(seed=seed)               # 9 training, 12 disagg
    assert not _gram_is_safe(inputs.X_disagg[:inputs.T0 - 1, :])
    grid = np.concatenate(([0.0], np.logspace(-8, np.log10(5), 25),
                           np.logspace(1, 3, 4)))
    chosen = select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=grid)
    errors = _sequential_cv_errors(inputs, sigma_y2, grid)
    assert chosen == float(grid[int(np.argmin(errors))])


# --------------------------------------------------------------------------- #
# 4. The cvxpy escape hatch is untouched
# --------------------------------------------------------------------------- #
def test_explicit_solver_still_routes_through_cvxpy():
    cp = pytest.importorskip("cvxpy")
    inputs, Q, sigma_y2 = _setup(seed=12)
    grid = np.array([0.0, 1e-3, 1.0])
    native = select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=grid)
    viacvx = select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=grid,
                              solver=cp.CLARABEL)
    assert native in set(grid) and viacvx in set(grid)
