"""The least-norm tie-break returns the least-norm point, at any design scale.

``solve_simplex_qp_least_norm`` exists because ``min ||A - Bw||^2`` over the
simplex can have a face of minimisers instead of a point. On such a program the
plain solver returns whichever member its pivots reach, so relabelling the
donors moves the weights. The tie-break is supposed to remove that dependence by
selecting the shortest member of the face.

It did not. The augmentation adds ``lambda ||w||^2`` with ``lambda`` set
relative to the design's mean column energy, and the pivot loop releases a
pinned donor when its reduced gradient falls below ``tol * (1 + max|g|)``. On a
face the fit is exact, so ``max|g|`` vanishes and that threshold floors at
``tol`` in absolute terms while the ridge's signal scales with the design. The
shipped pair, ``lambda`` relative at 1e-10 against ``tol = 1e-9``, put the
signal below the threshold: the selection never bound, and the answer came from
pivot order after all. It read as working because on a narrow face the pivots
happen to stop at the shortest member anyway.

Measured on three shapes -- the 5-dimensional face of one equation in six
donors, the 38-dimensional face the single-predictor level spec produces over
forty donors, and a rank-deficient 8-by-39 design -- against a two-stage cvxpy
reference that minimises the fit and then the norm among the minimisers:

    formulation                     err vs reference   scale invariant
    relative lambda, 1e-10/1e-9         2.5e-01            no
    normalised, 1e-8/1e-9               5.6e-02            yes
    normalised, 1e-8/1e-10              2.2e-05            yes

Scaling ``(B, A)`` together leaves the argmin untouched, so normalising the
design to unit mean column energy costs nothing and puts the ridge and the
release threshold in the same units. What decides whether the selection
resolves is then the ratio of the two constants, and 100 is enough on all three
shapes across four decades of design scale while 10 is not.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.solvers import active_set as AS
from mlsynth.utils.solvers.active_set import (
    _LEAST_NORM_RIDGE,
    _LEAST_NORM_TOL,
    solve_simplex_qp,
    solve_simplex_qp_least_norm,
)

cp = pytest.importorskip("cvxpy")

SCALES = (1e-4, 1e-2, 1.0, 1e2, 1e4)


def _designs():
    """Three programs whose minimisers are a face, widest face last.

    Every target sits off the uniform mix. The active set starts from the
    uniform weights, so a face the uniform point already lies on is one the
    plain solver reaches without pivoting: its answer is then uniform and
    permutation-invariant by accident, and a check run on such a design cannot
    fail. The level spec built from ``B.mean()`` is exactly that design, which
    is why its target is a skewed mix instead.
    """
    rng = np.random.default_rng(3)
    level = rng.normal(size=(1, 40)) * 3.0 + 10.0
    rankdef = np.cumsum(rng.normal(size=(8, 39)), axis=0) + 10.0
    skewed = rng.dirichlet(np.full(40, 0.3))
    return {
        "one equation, six donors": (np.arange(1.0, 7.0).reshape(1, 6),
                                     np.array([3.0])),
        "level spec, forty donors": (level, level @ skewed),
        "rank deficient 8x39": (rankdef, rankdef @ rng.dirichlet(np.ones(39))),
    }


def test_every_design_has_a_target_off_the_uniform_mix():
    """The power the checks below need, asserted once so it cannot rot away."""
    for name, (B, A) in _designs().items():
        uniform = B @ np.full(B.shape[1], 1.0 / B.shape[1])
        assert np.abs(A - uniform).max() > 1e-3, name


def _reference(B, A):
    """Least fit, then least norm among the minimisers, by two cvxpy solves."""
    J = B.shape[1]
    w = cp.Variable(J)
    cp.Problem(cp.Minimize(cp.sum_squares(A - B @ w)),
               [w >= 0, cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    fstar = float(np.sum((A - B @ w.value) ** 2))
    short = cp.Variable(J)
    cp.Problem(cp.Minimize(cp.sum_squares(short)),
               [short >= 0, cp.sum(short) == 1,
                cp.sum_squares(A - B @ short) <= fstar + 1e-10]
               ).solve(solver=cp.CLARABEL)
    return np.asarray(short.value).ravel(), fstar


@pytest.mark.parametrize("name", list(_designs()))
def test_it_returns_the_least_norm_point_of_the_face(name):
    """The rule's whole claim, against a reference that does not share its code."""
    B, A = _designs()[name]
    ref, fstar = _reference(B, A)
    got = solve_simplex_qp_least_norm(B, A)

    assert got.min() >= -1e-9 and abs(got.sum() - 1.0) < 1e-8
    # It is a minimiser: the fit it gives up is far below the data's own scale.
    assert float(np.sum((A - B @ got) ** 2)) <= fstar + 1e-6 * float(A @ A) + 1e-9
    # And it is the shortest one.
    # The reference is allowed 1e-10 of fit slack, so it can sit fractionally
    # off the face and be fractionally shorter for it; compare relatively.
    assert float(got @ got) <= float(ref @ ref) * (1.0 + 1e-3) + 1e-9
    assert np.abs(got - ref).max() < 1e-3, np.abs(got - ref).max()


@pytest.mark.parametrize("name", list(_designs()))
def test_the_selection_does_not_move_with_the_design_scale(name):
    """Scaling ``(B, A)`` together cannot move the argmin, so it must not move.

    The assertion is against the reference and not against the answer at unit
    scale: two wrong answers that agree with each other would pass that.
    """
    B, A = _designs()[name]
    ref, _ = _reference(B, A)
    for c in SCALES:
        got = solve_simplex_qp_least_norm(c * B, c * A)
        assert np.abs(got - ref).max() < 1e-3, (c, np.abs(got - ref).max())


@pytest.mark.parametrize("name", list(_designs()))
def test_relabelling_the_donors_does_not_move_the_selection(name):
    """The reason the rule exists: the answer is the data's, not the pivots'."""
    B, A = _designs()[name]
    base = solve_simplex_qp_least_norm(B, A)
    cold = solve_simplex_qp(B, A, accelerate=False)
    rng = np.random.default_rng(11)
    moved_rule = moved_cold = 0.0
    for _ in range(12):
        q = rng.permutation(B.shape[1])
        back = np.empty(B.shape[1])
        back[q] = solve_simplex_qp_least_norm(B[:, q], A)
        moved_rule = max(moved_rule, float(np.abs(back - base).max()))
        back[q] = solve_simplex_qp(B[:, q], A, accelerate=False)
        moved_cold = max(moved_cold, float(np.abs(back - cold).max()))
    assert moved_rule < 1e-6, moved_rule
    # The check has power only where an untied solve actually moves, and the
    # foil has to be the cold path. A seeding rule that prices the columns is
    # itself permutation-equivariant, so under one the plain solve does not move
    # either and comparing against it would assert nothing.
    assert moved_cold > 1e-3, moved_cold


def test_the_ridge_clears_the_pivot_release_threshold():
    """The invariant nobody asserted, and the one the shipped constants broke.

    A ridge at or below ``tol`` asks the pivot loop to resolve a gradient it
    treats as zero, so the selection does not happen. The signal is ``ridge``
    times the distance to the face's shortest point, which runs to 1e-2, so the
    ratio has to cover that distance as well: at 1e2 the widest face measured
    here lands 2.8e-3 off the reference, and 1e4 brings it to 2.2e-5. The margin
    is taken out of ``tol``, since the ridge cannot be raised to get it -- past
    roughly 1e-7 it moves an already-identified answer, which
    ``test_an_identified_problem_is_left_alone`` in the SCMO reference module
    holds it to. The shipped pair was 1e-10 against 1e-9, a ratio of 0.1.
    """
    assert _LEAST_NORM_RIDGE >= 1e4 * _LEAST_NORM_TOL, (
        f"ridge {_LEAST_NORM_RIDGE:.0e} against tol {_LEAST_NORM_TOL:.0e} "
        "leaves the selection below the release threshold")


@pytest.mark.parametrize("name", list(_designs()))
def test_a_ridge_under_the_threshold_leaves_the_answer_to_the_pivots(name):
    """The mechanism, demonstrated instead of asserted in a comment.

    The augmentation is rebuilt here so the starting point can be set, because
    that is what the defect is about: under the threshold the ridge cannot move
    the pivots off whichever face point they reach, and two different starts
    therefore disagree. The old pair also happens to land on the least-norm
    point under some starting rules, which is why it read as working -- so this
    drives the start explicitly instead of relying on any one seed.
    """
    B, A = _designs()[name]
    J = B.shape[1]
    ref, _ = _reference(B, A)
    root = np.sqrt(float(np.mean(np.sum(B * B, axis=0))))

    def solve(ridge, tol, warm):
        return solve_simplex_qp(
            np.vstack([B / root, np.sqrt(ridge) * np.eye(J)]),
            np.concatenate([A / root, np.zeros(J)]),
            tol=tol, warm_start=warm)

    first, last = np.zeros(J), np.zeros(J)
    first[0] = last[-1] = 1.0

    under = [solve(1e-10, 1e-9, w) for w in (first, last)]
    assert min(float(np.abs(u - ref).max()) for u in under) > 1e-2

    now = [solve(_LEAST_NORM_RIDGE, _LEAST_NORM_TOL, w) for w in (first, last)]
    for got in now:
        assert float(np.abs(got - ref).max()) < 1e-2, float(np.abs(got - ref).max())
    assert float(np.abs(now[0] - now[1]).max()) < 1e-2


@pytest.mark.parametrize("name", list(_designs()))
def test_it_is_a_selection_and_not_a_penalty(name):
    """Every ridge across four decades picks the same point out of the face."""
    B, A = _designs()[name]
    base = solve_simplex_qp_least_norm(B, A)
    for ridge in (1e-9, 1e-8, 1e-7, 3e-7):
        # The ratio is held at the shipped one, so this varies the ridge's size
        # alone; at a ratio of 1e2 the widest face here is 2.8e-3 off and the
        # sweep would be measuring the margin instead of the ridge.
        got = solve_simplex_qp_least_norm(
            B, A, ridge=ridge, tol=ridge / 1e4)
        assert np.abs(got - base).max() < 1e-3, (ridge, np.abs(got - base).max())


def test_a_program_with_a_unique_minimiser_is_left_where_it_was():
    """Off a face the rule has nothing to select, and must not bend the answer."""
    rng = np.random.default_rng(5)
    B = rng.normal(size=(12, 4))
    A = B @ np.array([0.1, 0.2, 0.3, 0.4]) + 0.01 * rng.normal(size=12)
    assert np.abs(solve_simplex_qp_least_norm(B, A) - solve_simplex_qp(B, A)).max() < 1e-6


def test_a_design_with_no_scale_falls_back_to_the_plain_solve():
    """An all-zero design has no energy to set a ridge from."""
    B = np.zeros((3, 4))
    A = np.zeros(3)
    got = solve_simplex_qp_least_norm(B, A)
    assert got.min() >= -1e-9 and abs(got.sum() - 1.0) < 1e-8


def test_a_single_donor_is_the_answer():
    got = solve_simplex_qp_least_norm(np.array([[2.0]]), np.array([5.0]))
    assert np.allclose(got, [1.0])


# --------------------------------------------------------------------------- #
# the two invariants over drawn designs, not three chosen ones
# --------------------------------------------------------------------------- #
_SETTINGS = settings(derandomize=True, deadline=None, max_examples=60,
                     suppress_health_check=[HealthCheck.too_slow])

_FINITE = dict(allow_nan=False, allow_infinity=False)


@st.composite
def _drawn_face(draw):
    """A design with more donors than rows, and a target inside the hull.

    Fewer rows than donors leaves a null space, and a target that is an exact
    mix puts the optimum on a face of it, which is the regime the tie-break is
    for. The mix is drawn away from uniform so the cold start is not already
    sitting on the answer.
    """
    m = draw(st.integers(1, 4))
    J = draw(st.integers(m + 2, 10))
    flat = draw(st.lists(st.floats(-20.0, 20.0, **_FINITE),
                         min_size=m * J, max_size=m * J))
    B = np.asarray(flat, dtype=float).reshape(m, J)
    raw = draw(st.lists(st.floats(0.0, 1.0, **_FINITE), min_size=J, max_size=J))
    mix = np.asarray(raw, dtype=float)
    assume(mix.sum() > 1e-6)
    mix = mix / mix.sum()
    # A ridge set from the design needs the design to have some energy.
    assume(float(np.mean(np.sum(B * B, axis=0))) > 1e-6)
    return B, B @ mix


@given(design=_drawn_face(), seed=st.integers(0, 2 ** 32 - 1))
@_SETTINGS
def test_relabelling_is_a_symmetry_of_the_selection(design, seed):
    """Donor order carries no information, so the answer cannot depend on it."""
    B, A = design
    perm = np.random.default_rng(seed).permutation(B.shape[1])
    base = solve_simplex_qp_least_norm(B, A)
    back = np.empty(B.shape[1])
    back[perm] = solve_simplex_qp_least_norm(B[:, perm], A)
    assert np.abs(back - base).max() < 1e-6, np.abs(back - base).max()


@given(design=_drawn_face(),
       factor=st.floats(1e-3, 1e3, **_FINITE).filter(lambda c: c > 0))
@_SETTINGS
def test_joint_rescaling_is_a_symmetry_of_the_selection(design, factor):
    """``(B, A) -> (cB, cA)`` scales the objective and fixes its argmin."""
    B, A = design
    base = solve_simplex_qp_least_norm(B, A)
    got = solve_simplex_qp_least_norm(factor * B, factor * A)
    assert np.abs(got - base).max() < 1e-6, (factor, np.abs(got - base).max())


@given(design=_drawn_face())
@_SETTINGS
def test_the_answer_is_always_feasible_and_a_minimiser(design):
    """Feasibility and optimality hold whether or not the face is wide."""
    B, A = design
    w = solve_simplex_qp_least_norm(B, A)
    assert w.min() >= -1e-8
    assert abs(w.sum() - 1.0) < 1e-7
    # The target is an exact mix, so the attainable fit is zero up to rounding.
    scale = max(1.0, float(A @ A))
    assert float(np.sum((A - B @ w) ** 2)) < 1e-6 * scale
