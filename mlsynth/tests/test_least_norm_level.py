"""``solve_simplex_qp_least_norm`` does not depend on the data's level (#942).

The tie-breaking ridge was sized by the design's mean column energy, level
included, so on uncentred data the ridge grew with the level squared and
outweighed the variation it was meant to break ties within. With donors
``N(0, 1)`` plus a level and a treated unit inside their hull plus noise, the
loss returned exceeded the optimum by up to 110 percent at levels 1e2 to 1e4
and 800 percent at 1e4 to 1e6, with no warning (agents_solvers.md, rule 5).

On the simplex ``1'w = 1``, so ``A - B w = (A - c) - (B - c 1') w`` for any
per-row level ``c``; removing ``c = A`` (the difference form ``B - A 1'``
against a zero target) changes the problem's conditioning and not its answer.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.solvers.active_set import solve_simplex_qp, solve_simplex_qp_least_norm
from mlsynth.utils.solvers.minnorm import simplex_optimum_is_unique


def _problem(seed, level, T=20, J=6, noise=0.3):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, J)) + level
    A = B @ rng.dirichlet(np.ones(J)) + noise * rng.normal(size=T)
    return B, A


def _loss(B, A, w):
    """Scored on data centred by ``A``: the same loss, without the level's rounding."""
    return float(np.sum(((B - A[:, None]) @ w) ** 2))


# --------------------------------------------------------------------------- smoke
def test_a_high_level_returns_the_optimum():
    B, A = _problem(0, 1e5)
    w = solve_simplex_qp_least_norm(B, A)
    best = _loss(B, A, solve_simplex_qp(B, A))
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)
    assert _loss(B, A, w) <= best * (1.0 + 1e-9) + 1e-12


# --------------------------------------------------------------------------- unit
@pytest.mark.parametrize("level", [1e2, 1e4, 1e6])
def test_the_regression_table(level):
    """The issue's measurement, per level band: the loss is the plain solver's."""
    worst = 0.0
    for seed in range(20):
        B, A = _problem(seed, level)
        best = _loss(B, A, solve_simplex_qp(B, A))
        worst = max(worst, (_loss(B, A, solve_simplex_qp_least_norm(B, A)) - best) / best)
    assert worst <= 1e-9


def test_a_per_row_level_is_removed_too():
    B, A = _problem(3, 0.0)
    c = np.linspace(-1e5, 1e5, B.shape[0])
    np.testing.assert_allclose(solve_simplex_qp_least_norm(B + c[:, None], A + c),
                               solve_simplex_qp_least_norm(B, A), atol=1e-8)


def test_the_tie_break_is_unchanged_on_a_face():
    """A wide design: still the least-norm member, now at any level."""
    rng = np.random.default_rng(4)
    B, A = rng.normal(size=(3, 9)), rng.normal(size=3)
    w0 = solve_simplex_qp_least_norm(B, A)
    np.testing.assert_allclose(solve_simplex_qp_least_norm(B + 1e6, A + 1e6), w0, atol=1e-6)


# --------------------------------------------------------------------------- edge
def test_a_target_equal_to_one_donor_selects_it():
    B, _ = _problem(5, 1e4)
    w = solve_simplex_qp_least_norm(B, B[:, 2].copy())
    assert w[2] == pytest.approx(1.0, abs=1e-6)            # the ridge (1e-8) biases it by about 2e-8


def test_every_donor_equal_to_the_target_keeps_the_fallback():
    """Nothing to fit: every point is a minimiser, and the plain solver's is returned."""
    B = np.full((4, 3), 7.0)
    w = solve_simplex_qp_least_norm(B, np.full(4, 7.0))
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- failure
@pytest.mark.parametrize("bad", ["B", "A"])
def test_a_non_finite_input_is_refused(bad):
    B, A = _problem(6, 1.0)
    (B if bad == "B" else A)[1] = np.nan
    with pytest.raises(MlsynthDataError, match="finite"):
        solve_simplex_qp_least_norm(B, A)


# --------------------------------------------------------------------------- property
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(0, 8), J=st.integers(2, 12))
@example(seed=0, k=6, J=6)
def test_property_the_answer_does_not_depend_on_the_level(seed, k, J):
    B, A = _problem(seed, 0.0, J=J)
    c = 10.0 ** k
    w0 = solve_simplex_qp_least_norm(B, A)
    w = solve_simplex_qp_least_norm(B + c, A + c)
    # Storing B + c rounds each entry by eps c / 2 (rule 11); the loss on the stored
    # data may exceed the optimum by that error's effect on the fit.
    floor = (np.finfo(float).eps * c) ** 2 * B.shape[0] * 16
    best = _loss(B + c, A + c, solve_simplex_qp(B + c, A + c))
    assert _loss(B + c, A + c, w) <= best * (1.0 + 1e-9) + floor
    if simplex_optimum_is_unique(B, A, w0):
        np.testing.assert_allclose(w, w0, atol=1e-6 + 1e-12 * c)
