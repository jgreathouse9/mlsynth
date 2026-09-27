"""The priced seed for the simplex active set, and what it may and may not move.

A cold active set starting from the uniform point sheds one donor per pivot, so
its work scales with the pool and not with the support it ends on: 153 pivots on
a 10x160 design, 32 on Prop 99's 19x38. ``fista_warm_start`` cuts that to 2
pivots and costs more than it saves on a wide, short design -- 8.6 ms of a 9.0 ms
solve at 10x160, because the support never settles for ``support_patience`` and
the loop runs to its 400-iteration cap.

The priced seed names the same support for the cost of one gradient. At the
uniform point the reduced gradient is ``B'(B w0 - A)``; the columns whose entry is
most negative are the ones that most improve the fit. ``SEED_KEEP`` of them are
kept, capped at ``m + 1``, the largest support an identified optimum can have.
Seeding there is an ``argpartition``, microseconds against the milliseconds either
alternative costs.

What it cannot do is change the answer. The active set certifies KKT optimality
over every column, so a seed changes which pivots happen and nothing else. These
tests hold it to that: bit-identical weights wherever the optimum is identified,
an objective never worse anywhere, and the certificate intact.

Where the optimum is *not* identified the returned point moves, and that is true
of any change to pivot order, the existing FISTA seed included. Measured over 400
fuzzed designs: the objective agrees to 7.22e-15 relative, the weights differ in
144, and ``simplex_optimum_is_unique`` rejects all 144 -- none of the differences
lands on an identified optimum. Over 500 tall and 600 wide designs whose optimum
*is* identified, the seeded and cold solves agree bit for bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.solvers.accelerate import SEED_KEEP, priced_seed
from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.solvers.minnorm import simplex_optimum_is_unique


def _rand(rng, m, J, scale=1.0):
    B = rng.normal(size=(m, J)) * scale + 5.0
    A = rng.normal(size=m) + B.mean(axis=1)
    return B, A


def _obj(B, A, w):
    r = A - B @ w
    return float(r @ r)


# --------------------------------------------------------------------------- #
# The seed itself
# --------------------------------------------------------------------------- #
def test_the_seed_is_a_point_on_the_simplex():
    rng = np.random.default_rng(0)
    B, A = _rand(rng, 10, 160)
    w = priced_seed(B, A)
    assert w.shape == (160,)
    assert w.min() >= 0.0
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)


def test_the_seed_keeps_its_budget_capped_at_the_support_bound():
    """``m + 1`` is the largest support an identified optimum can have, so the
    budget never exceeds it however large ``SEED_KEEP`` is."""
    rng = np.random.default_rng(1)
    for m, J in ((100, 300), (30, 90), (10, 160), (5, 40)):
        B, A = _rand(rng, m, J)
        assert int((priced_seed(B, A) > 0).sum()) == min(SEED_KEEP, m + 1)


def test_the_budget_is_an_argument():
    rng = np.random.default_rng(11)
    B, A = _rand(rng, 40, 200)
    assert int((priced_seed(B, A, keep=7) > 0).sum()) == 7
    assert int((priced_seed(B, A, keep=300) > 0).sum()) == 41   # the m + 1 cap


def test_the_seed_names_the_most_negative_reduced_gradient():
    """Which columns, not merely how many."""
    rng = np.random.default_rng(2)
    m, J = 8, 50
    B, A = _rand(rng, m, J)
    g = B.T @ (B @ np.full(J, 1.0 / J) - A)
    k = min(SEED_KEEP, m + 1)
    expected = set(np.argsort(g)[:k].tolist())
    assert set(np.flatnonzero(priced_seed(B, A)).tolist()) == expected


def test_the_seed_declines_a_pool_it_cannot_prune():
    """With nothing to price away the solve starts from the uniform point.

    The gate is the budget and not the panel's orientation: a tall design with
    more donors than the budget is still pruned, and measurably faster for it --
    1.7x at 40x20, 2.4x at 100x30, 11x at 100x60.
    """
    rng = np.random.default_rng(3)
    for m, J in ((40, 10), (100, 16), (10, 11), (10, 5), (40, 12)):
        B, A = _rand(rng, m, J)
        assert priced_seed(B, A) is None


def test_the_seed_is_deterministic():
    rng = np.random.default_rng(4)
    B, A = _rand(rng, 12, 90)
    assert np.array_equal(priced_seed(B, A), priced_seed(B, A))


def test_a_degenerate_design_still_yields_a_feasible_seed():
    B = np.ones((6, 30))
    A = np.ones(6)
    w = priced_seed(B, A)
    assert w.min() >= 0.0 and float(w.sum()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# What the solver does with it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,J", [(10, 80), (10, 160), (19, 38), (30, 100)])
def test_the_seeded_solve_is_bit_identical_on_an_identified_optimum(m, J):
    rng = np.random.default_rng(100 + m + J)
    found = 0
    for _ in range(40):
        B, A = _rand(rng, m, J)
        ref = solve_simplex_qp(B, A, accelerate=False)
        if not simplex_optimum_is_unique(B, A, ref):
            continue
        found += 1
        np.testing.assert_array_equal(solve_simplex_qp(B, A), ref)
    assert found >= 1, "no identified optimum drawn; the test had no power"


def test_the_seeded_solve_never_worsens_the_objective():
    """Over every shape, identified or not, the value is the value."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(120):
        m = int(rng.integers(3, 50)); J = int(rng.integers(2, 200))
        B, A = _rand(rng, m, J, scale=float(rng.choice([1.0, 100.0])))
        cold = solve_simplex_qp(B, A, accelerate=False)
        seeded = solve_simplex_qp(B, A)
        o_cold, o_seed = _obj(B, A, cold), _obj(B, A, seeded)
        worst = max(worst, (o_seed - o_cold) / max(o_cold, 1e-12))
    assert worst < 1e-10, worst


def test_the_seeded_solve_keeps_the_kkt_certificate():
    rng = np.random.default_rng(6)
    for _ in range(40):
        m = int(rng.integers(4, 40)); J = int(rng.integers(m + 2, 180))
        B, A = _rand(rng, m, J)
        w = solve_simplex_qp(B, A)
        g = B.T @ (B @ w - A)
        support = w > 1e-9
        nu = float(g[support].mean())
        scale = 1.0 + float(np.max(np.abs(g)))
        assert np.all(g[~support] >= nu - 1e-7 * scale)
        assert np.abs(g[support] - nu).max() <= 1e-6 * scale


def test_a_pool_below_the_budget_is_the_same_solve_it_always_was():
    """A declined seed leaves accelerate on and off identical."""
    rng = np.random.default_rng(7)
    for m, J in ((40, 10), (100, 16), (60, 12)):
        B, A = _rand(rng, m, J)
        np.testing.assert_array_equal(
            solve_simplex_qp(B, A), solve_simplex_qp(B, A, accelerate=False))


def test_the_seed_cuts_the_pivot_count_on_a_wide_design():
    rng = np.random.default_rng(8)
    B, A = _rand(rng, 10, 160)
    _, cold = solve_simplex_qp(B, A, accelerate=False, return_info=True)
    _, seeded = solve_simplex_qp(B, A, return_info=True)
    assert cold["converged"] and seeded["converged"]
    assert seeded["pivots"] * 5 < cold["pivots"], (seeded["pivots"], cold["pivots"])
