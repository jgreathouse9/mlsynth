"""Reducing the centred design to its column space before the screen runs.

Write ``Xt = Q R`` with ``Q`` having orthonormal columns. Then
``(QR)' (QR) = R' Q' Q R = R' R``, so every inner product between columns is
unchanged, the hull's geometry is unchanged, and ``Q' 0 = 0`` keeps the origin where
it was. Sunniness is a statement about the origin's position relative to that hull,
so it is identical on ``R``'s columns -- while the row count falls from ``m`` to
``min(m, J)``, which is what Eq (9)'s ``m+1`` equality rows are charged for.

The reduction is taken only when ``m > J``; a wide design already has fewer rows than
columns and there is nothing to remove. No rank tolerance is involved, deliberately:
truncating to the numerical rank could drop a direction that carries real geometry and
turn a shady donor sunny, so the reduction stops at ``J`` rows where it is exact.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.solvers.sunny import (
    _alpha,
    _centred,
    _program,
    _reduced,
    sunny_alphas,
    sunny_donors,
)


def _alphas_unreduced(B, A):
    """Eq (9) on the full-height design, bypassing both the gate and the reduction."""
    Xt = _centred(B, A)
    A_eq, b_eq, c = _program(Xt)
    return np.array([_alpha(Xt, j, A_eq, b_eq, c) for j in range(Xt.shape[1])])


# --------------------------------------------------------------------------- #
# the algebraic invariant the reduction rests on
# --------------------------------------------------------------------------- #
def test_the_reduction_preserves_every_inner_product():
    rng = np.random.default_rng(3)
    for m, J in ((40, 6), (2000, 12), (200, 33), (9, 9)):
        Xt = rng.normal(size=(m, J)) * 2.0
        R = _reduced(Xt)
        np.testing.assert_allclose(R.T @ R, Xt.T @ Xt, rtol=1e-10, atol=1e-9)


def test_the_reduction_takes_a_tall_design_down_to_its_column_count():
    rng = np.random.default_rng(5)
    for m, J in ((2000, 33), (200, 33), (40, 6)):
        assert _reduced(rng.normal(size=(m, J))).shape == (J, J)


def test_the_reduction_stops_at_the_column_count_even_when_rank_is_lower():
    """Pins the contract rather than the consequence. Truncating to the numerical rank
    instead would be harmless on measurement -- 150 designs with a direction placed at
    the rank tolerance showed no change in classification or in ``alpha*`` -- but it
    trades a threshold for no gain, so the shape is fixed at ``J`` rows whatever the
    rank. Without this the choice is only a comment, since on a full-rank design the two
    agree."""
    rng = np.random.default_rng(37)
    Xt = rng.normal(size=(400, 11))
    Xt[:, 4] = Xt[:, 9]                       # rank 10 < J = 11
    assert np.linalg.matrix_rank(Xt) < Xt.shape[1]
    assert _reduced(Xt).shape == (11, 11)


def test_a_wide_design_is_left_alone():
    """With ``m <= J`` there are already no more rows than columns."""
    rng = np.random.default_rng(7)
    for m, J in ((5, 74), (8, 40), (12, 12)):
        Xt = rng.normal(size=(m, J))
        out = _reduced(Xt)
        assert out.shape == Xt.shape
        np.testing.assert_array_equal(out, Xt)


# --------------------------------------------------------------------------- #
# the classification and the values must not move
# --------------------------------------------------------------------------- #
def test_the_classification_is_unchanged_across_shapes_and_ranks():
    """Both sides of ``m > J``, and both full and deficient rank, with the deficiency
    forced by duplicating a column so the vacuity gate cannot mask the comparison."""
    rng = np.random.default_rng(11)
    seen_tall = seen_deficient = False
    for m, J in ((40, 6), (60, 10), (30, 12), (9, 9), (8, 20), (5, 30)):
        for dup in (False, True):
            B = rng.normal(size=(m, J)) * 2.0
            if dup:
                B[:, -1] = B[:, 0]
            A = rng.normal(size=m) * 2.0
            got = sunny_alphas(B, A)
            ref = _alphas_unreduced(B, A)
            np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)
            assert ((got >= 1 - 1e-7) == (ref >= 1 - 1e-7)).all()
            seen_tall |= m > J
            seen_deficient |= np.linalg.matrix_rank(B - A[:, None]) < J
    assert seen_tall and seen_deficient, "the sweep missed a side it was meant to cover"


# --------------------------------------------------------------------------- #
# the rank-deficient cases, where the reduction actually bites
# --------------------------------------------------------------------------- #
def test_a_duplicated_donor_on_a_tall_design_classifies_as_before():
    rng = np.random.default_rng(13)
    B = rng.normal(size=(300, 8)) * 2.0
    B[:, 3] = B[:, 6]
    A = rng.normal(size=300) * 2.0
    got, ref = sunny_donors(B, A), _alphas_unreduced(B, A) >= 1 - 1e-7
    assert got.tolist() == ref.tolist()
    assert got[3] == got[6], "identical columns must classify alike"


def test_an_exact_convex_combination_stays_shady_after_the_reduction():
    """The case the reduction must not break: a donor built as twice the midpoint of
    two others has ``alpha* <= 1/2`` by construction, on a design tall enough that the
    reduction is taken."""
    rng = np.random.default_rng(17)
    A = rng.normal(size=400) * 2.0
    x1, x2 = rng.normal(size=400), rng.normal(size=400)
    Xt = np.column_stack([x1, x2, x1 + x2])
    B = A[:, None] + Xt
    al = sunny_alphas(B, A)
    assert al[2] <= 0.5 + 1e-7, al
    assert sunny_donors(B, A).tolist() == [True, True, False]


def test_a_zero_column_stays_shady_after_the_reduction():
    """A donor matching the treated path exactly centres to zero, so ``alpha* = 0``
    and an exact fit exists -- no donor is sunny and nothing may be pruned."""
    rng = np.random.default_rng(19)
    A = rng.normal(size=250) * 2.0
    B = A[:, None] + np.column_stack([rng.normal(size=250), np.zeros(250)])
    assert not sunny_donors(B, A).any()
    assert sunny_alphas(B, A)[1] == pytest.approx(0.0, abs=1e-9)


def test_the_reduction_does_not_let_the_gate_fire_on_a_deficient_design():
    """``rank(Xt) = J`` is the gate's condition and the reduction must not manufacture
    it: duplicating a column leaves the rank below ``J`` in either height."""
    from mlsynth.utils.solvers.sunny import sunny_screen_is_vacuous

    rng = np.random.default_rng(23)
    B = rng.normal(size=(500, 9)) * 2.0
    B[:, 2] = B[:, 5]
    A = rng.normal(size=500) * 2.0
    assert not sunny_screen_is_vacuous(B, A)
    assert np.linalg.matrix_rank(_reduced(B - A[:, None])) < B.shape[1]


# --------------------------------------------------------------------------- #
# and it has to be cheaper, which is the whole point
# --------------------------------------------------------------------------- #
def test_the_tall_design_solves_a_program_sized_by_columns_not_rows():
    """Eq (9) carries ``rows + 1`` equality rows, so the reduction is only worth
    anything if the program the solver sees actually shrinks."""
    rng = np.random.default_rng(29)
    Xt = rng.normal(size=(2000, 33))
    A_eq_full, _, _ = _program(Xt)
    A_eq_red, _, _ = _program(_reduced(Xt))
    assert A_eq_full.shape[0] == 2001
    assert A_eq_red.shape[0] == 34
