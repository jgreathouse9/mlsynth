"""Chaining the conformal grid sweep's refits, and proving it changes only cost.

Inverting a conformal test means solving the same simplex program once per
candidate effect, on a design that barely moves between neighbours: only the
post block of the treated series shifts, by one grid step. Solved cold, each
candidate rebuilds its active set from scratch. Seeded with the previous
candidate's solution it starts from the right support and certifies in a
handful of pivots.

``conformal_refit_gaps`` has taken a ``warm_start`` and offered ``return_base``
from the beginning, and ``conformal_att_interval``'s docstring already says a
grid sweep can chain them, but nothing between them exposed the seam:
``conformal_pvalue`` neither accepted a seed nor handed one back, so every
caller sweeping a grid solved cold. These tests pin the seam and, more
importantly, pin that using it is free of consequence.

The test this rests on is not the speed one. An optimisation that changes an
answer is not an optimisation, and a warm-started active set could in principle
certify at a different optimum where the program is degenerate -- equally
optimal weights, different residuals, a different p-value. So the sweep is
required to reproduce the unchained p-values exactly, not nearly.

Work is asserted in pivots, not seconds. ``solve_simplex_qp`` returns them for
this purpose and they are machine independent, where wall-clock on a shared
runner is not.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.bilevel.ridge_inference import (conformal_pvalue,
                                                   conformal_pvalue_sweep)


def _panel(J=20, pre=40, horizon=8, effect=0.3, seed=0):
    """A donor pool the treated unit sits inside, with a constant post effect."""
    rng = np.random.default_rng(seed)
    T = pre + horizon
    f = np.column_stack([np.cumsum(rng.normal(0, 1, T)) for _ in range(2)])
    lam = rng.dirichlet(np.ones(J - 1), size=2).T * 2.0
    Y0 = f @ lam.T + 0.5 * rng.standard_normal((T, J - 1))
    y = Y0 @ rng.dirichlet(np.ones(J - 1)) + 0.5 * rng.standard_normal(T)
    y[pre:] += effect
    return y, Y0, pre


def _unchained(y, Y0, pre, grid, **kw):
    """One cold call per candidate: what the sweep has to reproduce."""
    out = np.empty(len(grid))
    for i, tau in enumerate(grid):
        shifted = y.copy()
        shifted[pre:] -= tau
        out[i] = conformal_pvalue(shifted, Y0, pre, **kw)
    return out


_KW = dict(refit="sc", conformal_type="block", q=1.0)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_returns_one_p_value_per_candidate():
    y, Y0, pre = _panel()
    grid = np.linspace(-1.0, 1.5, 11)
    p = conformal_pvalue_sweep(y, Y0, pre, grid, **_KW)
    assert p.shape == grid.shape
    assert np.isfinite(p).all()
    assert ((p > 0.0) & (p <= 1.0)).all()


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_sweep_reproduces_the_unchained_p_values_exactly(seed):
    """The test the optimisation lives or dies by. Not ``allclose``: a conformal
    p-value is a rank among a finite reference set, so a discrepancy of any size
    means a different rank, and the interval that inverts it moves."""
    y, Y0, pre = _panel(seed=seed)
    grid = np.linspace(-1.5, 2.0, 25)
    assert np.array_equal(conformal_pvalue_sweep(y, Y0, pre, grid, **_KW),
                          _unchained(y, Y0, pre, grid, **_KW))


def test_an_unsorted_grid_returns_values_aligned_to_the_input():
    """The chain only saves anything if it walks the grid in order, but the
    caller's ordering is what the result must line up with."""
    y, Y0, pre = _panel()
    grid = np.array([0.6, -0.9, 1.4, 0.0, -0.3])
    swept = conformal_pvalue_sweep(y, Y0, pre, grid, **_KW)
    assert np.array_equal(swept, _unchained(y, Y0, pre, grid, **_KW))


def test_a_repeated_candidate_gets_the_same_p_value_twice():
    y, Y0, pre = _panel()
    grid = np.array([0.3, -0.2, 0.3])
    p = conformal_pvalue_sweep(y, Y0, pre, grid, **_KW)
    assert p[0] == p[2]


def test_seeding_a_single_call_does_not_move_its_p_value():
    """``warm_start`` on ``conformal_pvalue`` is a seed for the solver, not a
    parameter of the estimand."""
    y, Y0, pre = _panel()
    cold = conformal_pvalue(y, Y0, pre, **_KW)
    seed_w = np.full(Y0.shape[1], 1.0 / Y0.shape[1])
    assert conformal_pvalue(y, Y0, pre, warm_start=seed_w, **_KW) == cold


def test_asking_for_the_base_returns_simplex_weights_and_the_same_p_value():
    y, Y0, pre = _panel()
    plain = conformal_pvalue(y, Y0, pre, **_KW)
    p, base = conformal_pvalue(y, Y0, pre, return_base=True, **_KW)
    assert p == plain
    assert base.shape == (Y0.shape[1],)
    assert (base >= -1e-9).all()
    assert base.sum() == pytest.approx(1.0)


def _pivots_of(call):
    """Total active-set pivots spent inside ``call``.

    ``simplex_qp`` imports ``solve_simplex_qp`` from :mod:`active_set` at call
    time, so replacing the module attribute is seen by the code under test. The
    replacement forwards to the real solver and only records its work, which is
    what makes this a measurement of the path actually taken, not of a
    reimplementation of it.
    """
    from mlsynth.utils.bilevel import active_set

    real = active_set.solve_simplex_qp
    spent = []

    def counting(B, A, **kw):
        want_info = kw.pop("return_info", False)
        w, info = real(B, A, return_info=True, **kw)
        spent.append(info["pivots"])
        return (w, info) if want_info else w

    active_set.solve_simplex_qp = counting
    try:
        call()
    finally:
        active_set.solve_simplex_qp = real
    return sum(spent)


def test_the_sweep_itself_spends_far_fewer_pivots_than_cold_calls():
    """Work, in the unit the solver reports it, measured through the sweep.

    Asserting on the solver directly would pass even if the sweep never chained,
    which is exactly what two catalogued mutants do: walking the caller's order
    instead of the grid's, and passing no seed at all. Both leave every p-value
    correct and only the saving disappears, so nothing but a work assertion
    through this entry point can see them.

    The bound is loose on purpose -- the measured saving is about 95 percent and
    this asserts only that most of it survives a slower machine or a different
    BLAS, neither of which changes a pivot count.
    """
    y, Y0, pre = _panel(J=30, seed=4)
    grid = np.linspace(-1.5, 2.0, 41)
    chained = _pivots_of(
        lambda: conformal_pvalue_sweep(y, Y0, pre, grid, **_KW))
    cold = _pivots_of(lambda: _unchained(y, Y0, pre, grid, **_KW))
    assert chained < cold / 3, f"chained {chained} pivots against cold {cold}"


def test_the_sweep_walks_the_grid_in_order():
    """The saving comes from consecutive solves being neighbouring problems, so
    a shuffled grid must cost no more than a sorted one -- which it can only
    manage by sorting internally."""
    y, Y0, pre = _panel(J=30, seed=4)
    grid = np.linspace(-1.5, 2.0, 41)
    rng = np.random.default_rng(0)
    shuffled = grid.copy()
    rng.shuffle(shuffled)
    sorted_cost = _pivots_of(
        lambda: conformal_pvalue_sweep(y, Y0, pre, grid, **_KW))
    shuffled_cost = _pivots_of(
        lambda: conformal_pvalue_sweep(y, Y0, pre, shuffled, **_KW))
    assert shuffled_cost <= sorted_cost * 1.25


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_single_candidate_sweep_matches_a_single_call():
    y, Y0, pre = _panel()
    swept = conformal_pvalue_sweep(y, Y0, pre, np.array([0.25]), **_KW)
    shifted = y.copy()
    shifted[pre:] -= 0.25
    assert swept[0] == conformal_pvalue(shifted, Y0, pre, **_KW)


def test_an_infeasible_seed_is_ignored_rather_than_used():
    """A seed off the simplex is not a feasible starting point. The solver drops
    it, so the answer is the cold one and no exception is owed to the caller."""
    y, Y0, pre = _panel()
    cold = conformal_pvalue(y, Y0, pre, **_KW)
    bad = np.full(Y0.shape[1], -3.0)
    assert conformal_pvalue(y, Y0, pre, warm_start=bad, **_KW) == cold


def test_the_iid_scheme_sweeps_too_and_still_matches():
    """Chaining is a property of the refit, not of the permutation set."""
    y, Y0, pre = _panel()
    grid = np.linspace(-0.5, 1.0, 9)
    kw = dict(refit="sc", conformal_type="iid", q=1.0, ns=200, seed=3)
    assert np.array_equal(conformal_pvalue_sweep(y, Y0, pre, grid, **kw),
                          _unchained(y, Y0, pre, grid, **kw))


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [[], "0,1", 0.5])
def test_a_bad_grid_is_refused(bad):
    y, Y0, pre = _panel()
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="grid"):
        conformal_pvalue_sweep(y, Y0, pre, bad, **_KW)


def test_a_non_finite_grid_is_refused():
    y, Y0, pre = _panel()
    with pytest.raises(MlsynthDataError, match="finite"):
        conformal_pvalue_sweep(y, Y0, pre, [0.0, np.inf], **_KW)


def test_a_seed_of_the_wrong_length_is_refused():
    """A seed sized for a different donor pool is a caller bug, and silently
    ignoring it would hide the mix-up behind a correct-looking answer."""
    y, Y0, pre = _panel()
    with pytest.raises(MlsynthDataError, match="warm_start"):
        conformal_pvalue(y, Y0, pre, warm_start=np.ones(3) / 3.0, **_KW)


def test_an_unknown_refit_rule_is_still_refused_by_the_sweep():
    y, Y0, pre = _panel()
    with pytest.raises(MlsynthConfigError, match="refit"):
        conformal_pvalue_sweep(y, Y0, pre, np.array([0.0, 0.5]), refit="lasso")
