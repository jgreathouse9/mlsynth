"""Generative property tests for the SYNDES Gram reduction.

Reference: Doudchenko, Khosravi, Imbens et al. (2021), the two-way global
program. Naming the treated set ``S`` removes the ``q = w D`` coupling and
leaves, with ``G = Y'Y / T`` and ``sigma`` the +/-1 indicator of ``S``,

    f(S) = min { u'(G + lam I)u : 1'u = 0, sigma'u = 2, sigma * u >= 0 }.

``mlsynth.utils.syndes_helpers.gram`` drops the sign conditions and solves the
remaining two-equality program in closed form. The claims below hold for every
panel and every penalty, so they are asserted over the domain instead of at a
fixture.

Why this layer. Property tests generate inside the specification, which is where
a fault of omission lives -- see the instrument contract in
``agents/agents_tests.md``. These are its first-choice target: pure linear
algebra, no solver, no DGP, microseconds per call.

Two of the properties carry the module's meaning and are the reason the rest of
the pipeline may trust it.

``R 1 = 0`` is what makes the Rayleigh bound valid at all. It says the cut form
cannot see the component of ``sigma`` along the constant vector, which is what
lets the orthogonal component's fixed norm cap the form. Lose it and the global
bound stops being a bound.

Exactness is self-proving and needs no reference solver. When the closed-form
minimiser satisfies the dropped sign conditions it is feasible for the original
program, and it attains the optimum of a relaxation of that program, so it is
the optimum. Asserting feasibility and objective agreement over the domain
establishes ``lb(S) = f(S)`` on those rows outright.

Solver tolerance does not enter here, but conditioning does: ``H`` inverts
``G + lam I``, so the generators keep ``T >= N`` and ``lam`` away from zero, and
``assume`` discards any draw the module itself declines as ill-conditioned.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.syndes_helpers.gram import (
    BoundEngine,
    TwoWayProblem,
    maxcut_candidates,
    signs_from_subsets,
)

_TOL = 1e-8

_entry = st.floats(min_value=-5.0, max_value=5.0,
                   allow_nan=False, allow_infinity=False)
_penalty = st.floats(min_value=1e-3, max_value=10.0,
                     allow_nan=False, allow_infinity=False)


@st.composite
def panels(draw, min_units: int = 2, max_units: int = 7) -> np.ndarray:
    """A ``(T, N)`` pre-period matrix with ``T >= N``, so ``G`` is not degenerate."""
    N = draw(st.integers(min_value=min_units, max_value=max_units))
    T = draw(st.integers(min_value=N, max_value=N + 5))
    flat = draw(st.lists(_entry, min_size=T * N, max_size=T * N))
    Y = np.asarray(flat, dtype=float).reshape(T, N)
    # a column of zeros makes the unit useless but not the problem ill-posed;
    # an all-zero panel does, so exclude only that
    assume(np.abs(Y).max() > 1e-6)
    return Y


@st.composite
def instances(draw):
    """A panel, its penalty, and a treated-set size in range."""
    Y = draw(panels())
    lam = draw(_penalty)
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    engine = BoundEngine.from_problem(problem)
    assume(engine.reliable)
    K = draw(st.integers(min_value=1, max_value=problem.N - 1))
    return problem, engine, K


def _all_signs(N, K):
    return signs_from_subsets(N, np.array(list(combinations(range(N), K)),
                                          dtype=np.int64))


# ----------------------------------------------------------------------
# structure -- what makes the Rayleigh bound valid
# ----------------------------------------------------------------------


@settings(max_examples=150)
@given(instances())
def test_cut_matrix_annihilates_the_constant_vector(inst):
    """``R 1 = 0``; without it the Rayleigh bound is not a bound."""
    _, engine, _ = inst
    residual = engine.R @ np.ones(engine.N)
    assert np.abs(residual).max() <= _TOL * max(1.0, np.abs(engine.R).max())


@settings(max_examples=150)
@given(instances())
def test_cut_matrix_is_psd(inst):
    _, engine, _ = inst
    smallest = float(np.linalg.eigvalsh(engine.R)[0])
    assert smallest >= -_TOL * max(1.0, np.abs(engine.R).max())


@settings(max_examples=150)
@given(instances())
def test_alpha_positive_and_p_solves_the_system(inst):
    problem, engine, _ = inst
    assert engine.alpha > 0.0
    residual = problem.M @ engine.p - np.ones(engine.N)
    assert np.abs(residual).max() <= _TOL * max(1.0, np.abs(engine.p).max())


# ----------------------------------------------------------------------
# the closed form solves the program it claims to
# ----------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(instances())
def test_contrast_satisfies_both_normalizations(inst):
    """The two weight normalizations are exactly ``1'u = 0`` and ``sigma'u = 2``."""
    problem, engine, K = inst
    signs = _all_signs(problem.N, K)
    out = engine.bound(signs)
    scale = max(1.0, float(np.abs(out.contrast).max()))
    assert np.abs(out.contrast.sum(axis=1)).max() <= _TOL * scale
    assert np.abs(np.einsum("bi,bi->b", signs, out.contrast) - 2.0).max() <= _TOL * scale


@settings(max_examples=150, deadline=None)
@given(instances())
def test_exact_rows_are_feasible_and_attain_the_reported_value(inst):
    """Exactness proves itself: a feasible point attaining a relaxation optimum.

    On a row flagged exact the minimiser respects the dropped sign conditions, so
    it is feasible for the original program; it already attains the optimum of a
    relaxation of that program, so it is the optimum. No reference solver needed.
    """
    problem, engine, K = inst
    signs = _all_signs(problem.N, K)
    out = engine.bound(signs)
    for idx in np.flatnonzero(out.exact):
        u = out.contrast[idx]
        a, b = np.maximum(u, 0.0), np.maximum(-u, 0.0)
        treated = signs[idx] > 0
        # feasible for the original program
        assert np.abs(a.sum() - 1.0) <= 1e-7
        assert np.abs(b.sum() - 1.0) <= 1e-7
        assert np.abs(a[~treated]).max() <= 1e-9
        assert np.abs(b[treated]).max() <= 1e-9
        # and attains the value the bound reports
        assert abs(problem.evaluate(a, b) - out.lower[idx]) <= 1e-7 * max(
            1.0, abs(out.lower[idx]))


@settings(max_examples=150, deadline=None)
@given(instances())
def test_bound_is_positive(inst):
    problem, engine, K = inst
    out = engine.bound(_all_signs(problem.N, K))
    assert np.all(out.lower > 0.0)
    assert np.all(np.isfinite(out.lower))


# ----------------------------------------------------------------------
# the Rayleigh step, checked against the bound it caps
# ----------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(instances())
def test_rayleigh_bound_sits_below_every_closed_form_bound(inst):
    """``global_lower_bound`` caps the cut form, so it must not exceed any ``lb(S)``."""
    problem, engine, K = inst
    lower = engine.bound(_all_signs(problem.N, K)).lower
    rayleigh = engine.global_lower_bound(K)
    assert rayleigh is not None
    assert rayleigh <= lower.min() * (1 + 1e-7)


# ----------------------------------------------------------------------
# metamorphic relations
# ----------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(instances(), st.floats(min_value=0.2, max_value=5.0,
                              allow_nan=False, allow_infinity=False))
def test_scaling_the_panel_scales_the_bound_quadratically(inst, c):
    """``Y -> cY`` with ``lam -> c^2 lam`` scales the objective by ``c^2``."""
    problem, engine, K = inst
    Y_scaled_gram = TwoWayProblem(
        G=problem.G * c * c, M=problem.M * c * c,
        lam=problem.lam * c * c, N=problem.N)
    scaled = BoundEngine.from_problem(Y_scaled_gram)
    assume(scaled.reliable)

    signs = _all_signs(problem.N, K)
    base = engine.bound(signs).lower
    after = scaled.bound(signs).lower
    np.testing.assert_allclose(after, base * c * c, rtol=1e-6)


@settings(max_examples=100, deadline=None)
@given(instances(), st.integers(min_value=0, max_value=10_000))
def test_relabelling_units_permutes_the_bound_identically(inst, key):
    """Unit labels are arbitrary, so the bound must move with them."""
    problem, engine, K = inst
    order = np.random.default_rng(key).permutation(problem.N)
    permuted = BoundEngine.from_problem(TwoWayProblem(
        G=problem.G[np.ix_(order, order)],
        M=problem.M[np.ix_(order, order)],
        lam=problem.lam, N=problem.N))
    assume(permuted.reliable)

    subsets = np.array(list(combinations(range(problem.N), K)), dtype=np.int64)
    base = engine.bound(signs_from_subsets(problem.N, subsets)).lower

    inverse = np.argsort(order)
    relabelled = np.sort(inverse[subsets], axis=1)
    after = permuted.bound(signs_from_subsets(problem.N, relabelled)).lower
    np.testing.assert_allclose(after, base, rtol=1e-6)


@settings(max_examples=100, deadline=None)
@given(panels(), _penalty, st.floats(min_value=-3.0, max_value=3.0,
                                     allow_nan=False, allow_infinity=False))
def test_a_period_constant_leaves_the_bound_alone(Y, lam, shift):
    """Adding a per-period constant to every unit cannot change the design.

    The feasible set lies in ``1'u = 0``, and ``(Y + c 1')u = Yu`` there, so the
    fit term -- and with it the whole objective -- is invariant.
    """
    per_period = shift * np.ones((Y.shape[0], 1))
    base = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))
    shifted = BoundEngine.from_problem(
        TwoWayProblem.from_panel(Y + per_period, lam=lam))
    assume(base.reliable and shifted.reliable)

    K = max(1, base.N // 2)
    signs = _all_signs(base.N, K)
    np.testing.assert_allclose(
        shifted.bound(signs).lower, base.bound(signs).lower, rtol=1e-5)


# ----------------------------------------------------------------------
# the candidate search agrees with the objective it optimizes
# ----------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(instances())
def test_maxcut_candidates_are_swap_local_optima(inst):
    """Every returned design survives its whole one-swap neighbourhood."""
    problem, engine, K = inst
    for treated in maxcut_candidates(engine, K, n_restarts=3, seed=0):
        sigma = -np.ones(problem.N)
        sigma[treated] = 1.0
        base = float(sigma @ engine.R @ sigma)
        for i in np.flatnonzero(sigma > 0):
            for j in np.flatnonzero(sigma < 0):
                probe = sigma.copy()
                probe[i], probe[j] = -1.0, 1.0
                assert float(probe @ engine.R @ probe) <= base + 1e-7 * max(1.0, abs(base))


@settings(max_examples=60, deadline=None)
@given(instances())
def test_maxcut_candidates_are_valid_designs(inst):
    problem, engine, K = inst
    cands = maxcut_candidates(engine, K, n_restarts=3, seed=0)
    assert cands.shape[1] == K
    for row in cands:
        assert len(set(row.tolist())) == K
        assert 0 <= row.min() and row.max() < problem.N
