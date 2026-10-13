"""What SpSyDiD's WLS rank check should and should not warn about.

The final WLS design is ``[intercept, alpha_2..alpha_N, beta_2..beta_T, D, WD]``
with row weights ``omega_i * lambda_t``, scaled as ``Xw = X * sqrt(w)``. A
zero-weight period therefore empties its own time dummy, and a zero-weighted
*reference* period leaves the intercept equal to the sum of the surviving
dummies on the rows that remain. Both drop the rank without touching ``D`` or
``WD``.

That distinction is the whole content of the check. ``tau`` and ``tau_s`` are
read off the last two columns, so a deficiency confined to the nuisance block
leaves both uniquely determined and the reported effects sound, while a
deficiency that reaches either effect column means ``lstsq`` picked one of many
answers for a number the caller is about to use.

So the predicate is not ``rank < n_cols``. It is whether the last two columns
each still add a dimension:

    rank(Xw) - rank(Xw[:, :-2]) == 2

These tests state both halves on designs built here, so they hold whatever the
weight solver is.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.spsydid_helpers.pipeline import effect_columns_are_identified


def _two_way_design(N: int, T: int, omega: np.ndarray, lam: np.ndarray, rng):
    """The pipeline's design, weighted the way the pipeline weights it."""
    n_cols = 1 + (N - 1) + (T - 1) + 2
    X = np.zeros((N * T, n_cols))
    X[:, 0] = 1.0
    for i in range(1, N):
        X[i * T : (i + 1) * T, i] = 1.0
    base = 1 + (N - 1)
    for t in range(1, T):
        X[np.arange(N) * T + t, base + t - 1] = 1.0
    D = np.zeros((N, T))
    D[0, T // 2 :] = 1.0                       # one treated unit, back half
    WD = np.zeros((N, T))
    WD[1, T // 2 :] = 0.5                      # a neighbour, partially exposed
    X[:, -2] = D.flatten()
    X[:, -1] = WD.flatten()
    w = np.outer(omega, lam).flatten()
    return X * np.sqrt(w)[:, None]


# ------------------------------------------------ a full-rank design is fine
def test_a_full_rank_design_is_identified():
    rng = np.random.default_rng(0)
    N, T = 6, 8
    Xw = _two_way_design(N, T, np.full(N, 1 / N), np.full(T, 1 / T), rng)
    assert np.linalg.matrix_rank(Xw) == Xw.shape[1]
    assert effect_columns_are_identified(Xw)


# ------------------------------- a nuisance-only deficiency is not a problem
def test_an_emptied_time_dummy_does_not_threaten_the_effects():
    """A zero weight on a non-reference period empties that period's dummy."""
    rng = np.random.default_rng(1)
    N, T = 6, 8
    lam = np.full(T, 1 / T)
    lam[3] = 0.0                                # not the reference period
    Xw = _two_way_design(N, T, np.full(N, 1 / N), lam, rng)
    assert np.linalg.matrix_rank(Xw) < Xw.shape[1]      # deficient
    assert effect_columns_are_identified(Xw)            # but not where it counts


def test_a_zero_weighted_reference_period_does_not_threaten_the_effects():
    """With t=0 unweighted the intercept is the sum of surviving dummies."""
    rng = np.random.default_rng(2)
    N, T = 6, 8
    lam = np.full(T, 1 / T)
    lam[0] = 0.0                                # the reference period
    Xw = _two_way_design(N, T, np.full(N, 1 / N), lam, rng)
    assert np.linalg.matrix_rank(Xw) < Xw.shape[1]
    assert effect_columns_are_identified(Xw)


def test_an_emptied_unit_dummy_does_not_threaten_the_effects():
    rng = np.random.default_rng(3)
    N, T = 6, 8
    omega = np.full(N, 1 / N)
    omega[4] = 0.0                              # a donor carrying no weight
    Xw = _two_way_design(N, T, omega, np.full(T, 1 / T), rng)
    assert np.linalg.matrix_rank(Xw) < Xw.shape[1]
    assert effect_columns_are_identified(Xw)


def test_several_nuisance_deficiencies_at_once_are_still_fine():
    rng = np.random.default_rng(4)
    N, T = 8, 10
    omega, lam = np.full(N, 1 / N), np.full(T, 1 / T)
    omega[5] = 0.0
    lam[0] = lam[2] = lam[3] = 0.0              # the measured spsydid pattern
    Xw = _two_way_design(N, T, omega, lam, rng)
    assert np.linalg.matrix_rank(Xw) <= Xw.shape[1] - 4
    assert effect_columns_are_identified(Xw)


# --------------------------------- a deficiency on the effects is a problem
def test_a_duplicated_effect_column_is_not_identified():
    """``WD`` equal to ``D`` is the collinearity the check exists for."""
    rng = np.random.default_rng(5)
    N, T = 6, 8
    Xw = _two_way_design(N, T, np.full(N, 1 / N), np.full(T, 1 / T), rng)
    Xw[:, -1] = Xw[:, -2]
    assert not effect_columns_are_identified(Xw)


def test_an_effect_column_of_zeros_is_not_identified():
    """No exposure anywhere leaves ``tau_s`` with nothing behind it."""
    rng = np.random.default_rng(6)
    N, T = 6, 8
    Xw = _two_way_design(N, T, np.full(N, 1 / N), np.full(T, 1 / T), rng)
    Xw[:, -1] = 0.0
    assert not effect_columns_are_identified(Xw)


def test_an_effect_column_inside_the_nuisance_span_is_not_identified():
    """``D`` equal to a time dummy cannot be told from a period effect."""
    rng = np.random.default_rng(7)
    N, T = 6, 8
    Xw = _two_way_design(N, T, np.full(N, 1 / N), np.full(T, 1 / T), rng)
    Xw[:, -2] = Xw[:, 1 + (N - 1)]              # the first time dummy
    assert not effect_columns_are_identified(Xw)


def test_both_effect_columns_zero_is_not_identified():
    rng = np.random.default_rng(8)
    N, T = 6, 8
    Xw = _two_way_design(N, T, np.full(N, 1 / N), np.full(T, 1 / T), rng)
    Xw[:, -2:] = 0.0
    assert not effect_columns_are_identified(Xw)


# ------------------------------------------------------------------ shapes
def test_a_design_with_two_columns_is_all_effects():
    """Degenerate but well posed: both columns are the effects."""
    Xw = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert effect_columns_are_identified(Xw)


def test_a_design_narrower_than_two_columns_is_refused():
    with pytest.raises(ValueError):
        effect_columns_are_identified(np.ones((4, 1)))
