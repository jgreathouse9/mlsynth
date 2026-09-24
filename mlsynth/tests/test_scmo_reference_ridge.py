"""SCMO's weights are the ones the authors' program returns, tie included.

Tian, Lee & Panchenko's ``fn_W`` (``benchmarks/reference/scmo_germany/reference.R``
and the identical copy under ``scmo_covid_sweden/``) builds

    Dmat <- ZJ %*% V %*% t(ZJ) + (10^-7) * diag(J)
    dvec <- ZJ %*% V %*% Zi

and hands it to ``quadprog::solve.QP`` under ``sum(w) == 1``, ``w >= 0``.
Doubling that objective and dropping the constant, with ``V = (1/p) I``, the
program is

    (1/p)||Zi - ZJ'w||^2 + 1e-7 ||w||^2   ==   ||Zi - ZJ'w||^2 + p*1e-7 ||w||^2

so the reference carries a ridge. mlsynth's port solved the first term alone.

On a problem whose minimiser is a point the term is worth nothing -- 4e-3 in the
German ATT, 1.5e-7 in the concatenated weights. On a problem whose minimiser is
a face it is the only thing that picks a point, and SCMO has such a problem: the
averaged scheme on a single-period multiple-outcome spec averages every outcome
into one column, so the German panel gives one equation in sixteen donors and a
14-dimensional face of exact minimisers. Without the ridge the answer came from
the solver's pivot order, and relabelling the donors moved the weights by 0.419.

What is implemented is the rule, not the constant. The authors' 1e-7 is absolute
and assumes the sd-scaled columns their script builds; rescaling the matching
columns by 1e-3 moves its answer by 2.4e-2. A ridge scaled by the design is
invariant to that (3.2e-16) and lands on the same point where their scaling does
hold (2.8e-8 on the averaged problem, 4.3e-6 worst of the three). The limit is
reached long before either value: lambda from 1e-14 to 9e-7 all give the same
weights, which is what makes this a selection rule and not a penalty.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.bilevel.active_set import (
    solve_simplex_qp,
    solve_simplex_qp_least_norm,
)
from mlsynth.utils.scmo_helpers import solvers


def _authors_program(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``fn_W`` exactly: the absolute ``p * 1e-7`` ridge, by augmentation."""
    p, J = B.shape
    lam = p * 1e-7
    return solve_simplex_qp(
        np.vstack([B, np.sqrt(lam) * np.eye(J)]), np.concatenate([A, np.zeros(J)])
    )


def _face_problem(J: int = 6):
    """One equation in ``J`` donors: every feasible ``w`` hitting it is optimal.

    This is the shape SCMO's averaged scheme produces on a single-period spec,
    reduced to the smallest panel that still has a face of minimisers.

    The target is deliberately not ``B.mean()``. The active set starts from the
    uniform weights, so a face that the uniform point already lies on is one the
    plain solver reaches without pivoting -- and its answer is then uniform, and
    permutation-invariant by accident. Measured on this design: at the uniform
    target both programs agree to 0.0 under relabelling, and the check has no
    power at all; at ``A = 3.0`` the plain program moves by 1.5e-1.
    """
    B = np.arange(1.0, J + 1.0).reshape(1, J)        # (1, J), distinct entries
    A = np.array([3.0])                               # off the uniform mix (3.5)
    return B, A


def _point_problem():
    """A well-posed matching problem: the minimiser is a single point."""
    rng = np.random.default_rng(11)
    B = rng.normal(size=(12, 4))
    w = np.array([0.5, 0.3, 0.2, 0.0])
    return B, B @ w


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_simplex_weights_returns_a_point_on_the_simplex():
    Z_donors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Z_treated = np.array([0.5, 0.5, 0.0])
    w = solvers.simplex_weights(Z_treated, Z_donors)
    assert w.shape == (3,)
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_the_least_norm_helper_returns_a_point_on_the_simplex():
    B, A = _face_problem()
    w = solve_simplex_qp_least_norm(B, A)
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# the invariant the ridge restores
# --------------------------------------------------------------------------- #
def test_a_face_of_minimisers_is_resolved_the_same_way_whatever_the_donor_order():
    """Relabelling the donors is not information. Without the ridge it moved
    the German averaged weights by 0.419."""
    B, A = _face_problem(J=6)
    w0 = solve_simplex_qp_least_norm(B, A)
    rng = np.random.default_rng(3)
    for _ in range(25):
        perm = rng.permutation(B.shape[1])
        wp = solve_simplex_qp_least_norm(B[:, perm], A)
        back = np.empty_like(wp)
        back[perm] = wp
        assert np.abs(back - w0).max() < 1e-10


def test_the_plain_program_does_not_have_that_invariant():
    """The instrument has power: the same sweep on the unridged program finds a
    different answer. A test that passed on both could not separate them."""
    B, A = _face_problem(J=6)
    w0 = solve_simplex_qp(B, A)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(25):
        perm = rng.permutation(B.shape[1])
        wp = solve_simplex_qp(B[:, perm], A)
        back = np.empty_like(wp)
        back[perm] = wp
        worst = max(worst, float(np.abs(back - w0).max()))
    assert worst > 1e-3, f"the face problem is not exercising a face: {worst}"


def test_the_selected_point_has_the_smallest_norm_among_the_minimisers():
    B, A = _face_problem(J=6)
    w = solve_simplex_qp_least_norm(B, A)
    f_star = float(np.sum((A - B @ w) ** 2))
    rng = np.random.default_rng(5)
    for _ in range(400):
        cand = rng.dirichlet(np.ones(B.shape[1]))
        if float(np.sum((A - B @ cand) ** 2)) <= f_star + 1e-12:
            assert np.linalg.norm(w) <= np.linalg.norm(cand) + 1e-9


def test_a_face_centred_on_the_uniform_mix_cannot_separate_the_two_programs():
    """Recorded because it is the trap this file fell into first. The plain
    program looks order-invariant on such a design, for a reason that has
    nothing to do with the ridge."""
    B = np.arange(1.0, 7.0).reshape(1, 6)
    A = np.array([B.mean()])
    w0 = solve_simplex_qp(B, A)
    assert np.allclose(w0, 1.0 / 6.0, atol=1e-9)
    rng = np.random.default_rng(3)
    for _ in range(10):
        perm = rng.permutation(6)
        wp = solve_simplex_qp(B[:, perm], A)
        back = np.empty_like(wp)
        back[perm] = wp
        assert np.abs(back - w0).max() == pytest.approx(0.0, abs=1e-12)


def test_two_identical_donors_are_split_evenly():
    """Any split of their shared mass is optimal; least norm splits it in two.
    This is the tie the Abadie-L'Hour penalty cannot break, since a linear
    penalty on the non-negative orthant selects a face and not a point."""
    col = np.array([[1.0], [2.0], [0.5]])
    B = np.hstack([col, col])
    w = solve_simplex_qp_least_norm(B, (col * 1.0).ravel())
    assert w[0] == pytest.approx(w[1], abs=1e-9)


# --------------------------------------------------------------------------- #
# the ridge must vanish where the answer is already determined
# --------------------------------------------------------------------------- #
def test_an_identified_problem_is_left_alone():
    B, A = _point_problem()
    assert np.abs(solve_simplex_qp_least_norm(B, A) - solve_simplex_qp(B, A)).max() < 1e-7


def test_it_lands_where_the_authors_absolute_ridge_lands():
    """The rule, checked against the constant it was read off."""
    for B, A in (_face_problem(J=6), _point_problem()):
        got = solve_simplex_qp_least_norm(B, A)
        assert np.abs(got - _authors_program(B, A)).max() < 1e-5


def test_the_answer_does_not_move_when_the_columns_are_rescaled():
    """What the absolute constant cannot do: at c=1e-3 it moves by 2.4e-2."""
    B, A = _face_problem(J=6)
    w0 = solve_simplex_qp_least_norm(B, A)
    for c in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        assert np.abs(solve_simplex_qp_least_norm(c * B, c * A) - w0).max() < 1e-9


def test_the_ridge_is_small_enough_to_be_a_tie_break_and_not_a_penalty():
    """On an identified problem the objective it reaches is the unridged
    optimum, to the last digits that matter."""
    B, A = _point_problem()
    f_plain = float(np.sum((A - B @ solve_simplex_qp(B, A)) ** 2))
    f_ridge = float(np.sum((A - B @ solve_simplex_qp_least_norm(B, A)) ** 2))
    assert f_ridge >= f_plain - 1e-12
    assert f_ridge - f_plain < 1e-9 * max(1.0, f_plain)


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
def test_a_single_donor_takes_all_the_weight():
    B = np.array([[2.0], [3.0]])
    w = solve_simplex_qp_least_norm(B, np.array([2.0, 3.0]))
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0, abs=1e-12)


def test_an_all_zero_design_still_returns_a_simplex_point():
    """No scale to set a ridge from; the fall-back must still be feasible."""
    B = np.zeros((3, 4))
    w = solve_simplex_qp_least_norm(B, np.zeros(3))
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_collinear_and_duplicated_donors_still_give_a_simplex_point():
    Z_donors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    Z_treated = np.array([0.5, 0.5, 0.0])
    w = solvers.simplex_weights(Z_treated, Z_donors)
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# failure is reported, not swallowed
# --------------------------------------------------------------------------- #
def test_a_failed_solve_reaches_the_caller_as_a_translated_error():
    from unittest import mock

    Z_donors = np.eye(3)
    Z_treated = np.array([0.5, 0.5, 0.0])
    with mock.patch(
        "mlsynth.utils.scmo_helpers.solvers.solve_simplex_qp_least_norm",
        side_effect=RuntimeError("singular design"),
    ):
        with pytest.raises(MlsynthEstimationError, match="degenerate or ill-conditioned"):
            solvers.simplex_weights(Z_treated, Z_donors)
