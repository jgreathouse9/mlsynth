"""Becker and Klossner's sunny/shady donor screen, on an outcome-only design.

The screen answers one question per donor: is there a shorter multiple of this
donor's centred column that still lies in the hull of all of them? Write
``x_j = B[:, j] - A`` and ``H = conv(x_1, ..., x_J)``. Then

    alpha*(j) = min { alpha >= 0 : alpha x_j in H }

and the donor is sunny when ``alpha*(j) == 1``, shady when it is strictly less.
Because ``x_j`` generates ``H``, ``alpha* <= 1`` always, so the test is one-sided
by construction.

Two propositions carry the weight, and both are asserted here against designs
whose answer is set by construction, not read off the implementation:

* Proposition 1 -- no donor is sunny exactly when ``0`` is in ``H``, which is
  exactly when the simplex program admits an exact fit.
* Proposition 2 -- a shady donor takes zero weight at every optimum, provided no
  exact fit exists. The simplex stationarity conditions give
  ``<x_k, u*> = min_i <x_i, u*> = ||u*||^2`` for every ``k`` in the support, and the
  move ``w* - t e_j + t lambda`` sends ``u*`` to ``u* - t (1 - alpha) x_j``, so
  ``alpha ||u*||^2 >= ||u*||^2``; with ``alpha < 1`` that forces ``u* = 0``. The
  hypothesis is not decoration: with an exact fit available a shady donor can take
  all the weight, which is why the cascade branches on "no sunny donor" before it
  prunes. Both halves are asserted below.

Reference: Becker and Klossner (2017), MSCMT, ``isSunny`` in ``R/Helpers.r`` and
the donor loop in ``R/multiOpt.r``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.solvers.sunny import (
    certified_sunny,
    sunny_alphas,
    sunny_donors,
    sunny_support,
)


# --------------------------------------------------------------------------- #
# designs whose classification is known before the solver runs
# --------------------------------------------------------------------------- #
def _with_a_shady_donor():
    """``x_3 = x_1 + x_2`` makes donor 3 shady: ``0.5 x_3`` is the midpoint of
    ``x_1`` and ``x_2``, so ``alpha*(3) <= 0.5``."""
    A = np.array([10.0, 20.0, 30.0])
    x1 = np.array([1.0, 0.0, 2.0])
    x2 = np.array([0.0, 3.0, -1.0])
    Xt = np.column_stack([x1, x2, x1 + x2])
    return A[:, None] + Xt, A


def _with_the_origin_inside():
    """``0 = (x_1 + x_2) / 2`` puts the origin in the hull, so by Proposition 1
    no donor is sunny and an exact fit exists at ``w = (1/2, 1/2)``."""
    A = np.array([5.0, -2.0])
    x1 = np.array([1.0, 4.0])
    Xt = np.column_stack([x1, -x1])
    return A[:, None] + Xt, A


def _all_sunny():
    """The standard basis. Its hull is the unit simplex, whose points have
    coordinates summing to 1, so ``alpha x_j`` lies in it only at ``alpha = 1`` and
    every donor is sunny."""
    return np.eye(3), np.zeros(3)


def _with_a_certifiable_donor():
    """``x_1 = (1, 0)`` minimises ``x_1' x`` over the three generators at
    ``||x_1||^2 = 1 > 0``, so the Gram test certifies it without a linear program.
    ``x_2 = (2, 0)`` is shady -- ``0.5 x_2 = x_1`` -- and ``x_3`` is sunny, so the
    design also pins that the certificate does not overreach."""
    A = np.zeros(2)
    Xt = np.column_stack([[1.0, 0.0], [2.0, 0.0], [1.0, 5.0]])
    return A[:, None] + Xt, A


def _certified_by_a_cross_direction():
    """Three columns sharing a first coordinate of 1. The hull lies in the plane
    ``z_1 = 1``, so ``alpha x_j`` meets it only at ``alpha = 1`` and every donor is
    sunny -- and ``c = x_1`` certifies all three at once."""
    A = np.zeros(2)
    Xt = np.column_stack([[1.0, 0.0], [1.0, 2.0], [1.0, -2.0]])
    return A[:, None] + Xt, A


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_returns_one_finite_alpha_and_one_flag_per_donor():
    B, A = _with_a_shady_donor()
    al = sunny_alphas(B, A)
    flags = sunny_donors(B, A)
    assert al.shape == (B.shape[1],)
    assert np.isfinite(al).all()
    assert flags.shape == (B.shape[1],)
    assert flags.dtype == bool


# --------------------------------------------------------------------------- #
# the geometry the screen rests on
# --------------------------------------------------------------------------- #
def test_alpha_never_exceeds_one():
    """``x_j`` generates the hull, so ``alpha = 1`` is always feasible."""
    rng = np.random.default_rng(0)
    for _ in range(12):
        m, J = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        B = rng.normal(size=(m, J)) * 3.0
        A = rng.normal(size=m)
        assert sunny_alphas(B, A).max() <= 1.0 + 1e-9


def test_a_lone_donor_is_sunny():
    B = np.array([[2.0], [5.0]])
    A = np.array([0.0, 1.0])
    assert sunny_donors(B, A).tolist() == [True]


def test_the_constructed_shady_donor_is_found_and_the_others_are_not():
    B, A = _with_a_shady_donor()
    al = sunny_alphas(B, A)
    flags = sunny_donors(B, A)
    assert al[2] <= 0.5 + 1e-7, al
    assert flags.tolist() == [True, True, False], (al, flags)


def test_every_donor_is_sunny_when_none_is_a_shorter_multiple():
    B, A = _all_sunny()
    assert sunny_donors(B, A).all(), sunny_alphas(B, A)


# --------------------------------------------------------------------------- #
# Proposition 1: no sunny donor iff an exact fit exists
# --------------------------------------------------------------------------- #
def test_no_sunny_donor_when_the_origin_is_in_the_hull():
    B, A = _with_the_origin_inside()
    assert not sunny_donors(B, A).any(), sunny_alphas(B, A)


def test_no_sunny_donor_iff_an_exact_fit_exists():
    """The two sides are computed independently: the screen from the LPs, the
    exact fit from the solver's own achieved objective."""
    rng = np.random.default_rng(7)
    seen_both = set()
    for _ in range(24):
        m, J = int(rng.integers(1, 4)), int(rng.integers(2, 8))
        B = rng.normal(size=(m, J)) * 2.0
        A = (B @ rng.dirichlet(np.ones(J)) if rng.random() < 0.5
             else rng.normal(size=m) * 2.0)
        any_sunny = bool(sunny_donors(B, A).any())
        w = solve_simplex_qp(B, A)
        scale = max(1.0, float(A @ A))
        exact = float(np.sum((A - B @ w) ** 2)) <= 1e-9 * scale
        assert exact == (not any_sunny), (any_sunny, exact, m, J)
        seen_both.add(exact)
    assert seen_both == {True, False}, "the design never exercised both sides"


# --------------------------------------------------------------------------- #
# Proposition 2: a shady donor takes no weight -- what pruning rests on
# --------------------------------------------------------------------------- #
def test_a_shady_donor_carries_no_weight_when_no_exact_fit_exists():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(40):
        m, J = int(rng.integers(2, 6)), int(rng.integers(3, 12))
        B = rng.normal(size=(m, J)) * 2.0
        A = rng.normal(size=m) * 2.0
        sunny = sunny_donors(B, A)
        if not (~sunny).any() or not sunny.any():
            continue        # the claim's hypothesis is that some donor is sunny
        w = solve_simplex_qp(B, A)
        assert np.abs(w[~sunny]).max() <= 1e-7, (w[~sunny], m, J)
        checked += 1
    assert checked > 0, "no shady donor arose, so the claim was never tested"


def test_with_an_exact_fit_a_shady_donor_can_take_all_the_weight():
    """``x_3 = 0``, so donor 3 alone reproduces the treated path while
    ``alpha*(3) = 0`` marks it shady. Pruning here would drop an exact fit, so the
    hypothesis on Proposition 2 is doing work and the cascade has to branch first."""
    A = np.array([5.0, -2.0])
    Xt = np.column_stack([[1.0, 4.0], [-1.0, -4.0], [0.0, 0.0]])
    B = A[:, None] + Xt
    flags = sunny_donors(B, A)
    assert not flags.any(), flags
    assert not flags[2]
    assert np.allclose(B @ np.array([0.0, 0.0, 1.0]), A)
    assert sunny_support(B, A).tolist() == [0, 1, 2]


def test_dropping_the_shady_donors_leaves_the_objective_unchanged():
    rng = np.random.default_rng(13)
    checked = 0
    for _ in range(40):
        m, J = int(rng.integers(2, 6)), int(rng.integers(3, 12))
        B = rng.normal(size=(m, J)) * 2.0
        A = rng.normal(size=m) * 2.0
        keep = sunny_donors(B, A)
        if keep.all() or not keep.any():
            continue
        full = float(np.sum((A - B @ solve_simplex_qp(B, A)) ** 2))
        cut = float(np.sum((A - B[:, keep] @ solve_simplex_qp(B[:, keep], A)) ** 2))
        assert abs(cut - full) <= 1e-8 * max(1.0, full), (full, cut)
        checked += 1
    assert checked > 0, "pruning never happened, so the claim was never tested"


# --------------------------------------------------------------------------- #
# invariants of the classification
# --------------------------------------------------------------------------- #
def test_relabelling_permutes_the_classification():
    B, A = _with_a_shady_donor()
    rng = np.random.default_rng(3)
    for _ in range(6):
        q = rng.permutation(B.shape[1])
        back = np.empty(B.shape[1], dtype=bool)
        back[q] = sunny_donors(B[:, q], A)
        assert back.tolist() == sunny_donors(B, A).tolist()


def test_joint_rescaling_leaves_the_classification():
    """The LP is homogeneous in ``(B, A)``, so a common factor cannot move it."""
    B, A = _with_a_shady_donor()
    base = sunny_donors(B, A)
    for c in (1e-3, 0.5, 7.0, 1e3):
        assert sunny_donors(c * B, c * A).tolist() == base.tolist(), c


def test_duplicated_donors_are_classified_alike():
    B, A = _with_a_shady_donor()
    Bd = np.column_stack([B, B[:, 0]])
    flags = sunny_donors(Bd, A)
    assert flags[0] == flags[-1], flags


# --------------------------------------------------------------------------- #
# the Gram certificate: one-sided, and it has to fire
# --------------------------------------------------------------------------- #
def test_the_certificate_never_claims_a_shady_donor_is_sunny():
    rng = np.random.default_rng(17)
    for _ in range(30):
        m, J = int(rng.integers(1, 6)), int(rng.integers(2, 12))
        B = rng.normal(size=(m, J)) * 2.0
        A = rng.normal(size=m) * 2.0
        cert = certified_sunny(B, A)
        truth = sunny_alphas(B, A) >= 1.0 - 1e-7
        assert not (cert & ~truth).any(), (cert, truth, m, J)


def test_the_certificate_shortcut_agrees_with_the_pure_lp_classification():
    rng = np.random.default_rng(19)
    for _ in range(24):
        m, J = int(rng.integers(1, 6)), int(rng.integers(2, 10))
        B = rng.normal(size=(m, J)) * 2.0
        A = (B @ rng.dirichlet(np.ones(J)) if rng.random() < 0.4
             else rng.normal(size=m) * 2.0)
        fast = sunny_donors(B, A, certify=True)
        ref = sunny_donors(B, A, certify=False)
        assert fast.tolist() == ref.tolist(), (fast, ref, m, J)


def test_the_certificate_uses_directions_other_than_the_donors_own_column():
    """Every column here has first coordinate 1, so ``c = x_1`` gives
    ``c'x_j = 1`` for all three: the hyperplane supports the hull and separates the
    origin, and all three donors are sunny. Donors 2 and 3 fail the test taken with
    their own column (``||x_2||^2 = 5`` against a row minimum of ``-3``), so this
    design fails unless the direction is allowed to range over the other columns."""
    B, A = _certified_by_a_cross_direction()
    assert sunny_donors(B, A).all(), sunny_alphas(B, A)
    assert certified_sunny(B, A).all(), certified_sunny(B, A)

    Xt = B - A[:, None]
    G = Xt.T @ Xt
    own = np.diag(G) <= G.min(axis=1)
    assert own.tolist() == [True, False, False], own


def test_the_certificate_fires_where_the_geometry_says_it_should():
    """A certificate that never fires is decoration, so one design has to show it
    deciding a donor, and showing it stop at the donor it cannot decide."""
    B, A = _with_a_certifiable_donor()
    cert = certified_sunny(B, A)
    assert cert[0]
    assert sunny_donors(B, A).tolist() == [True, False, True], sunny_alphas(B, A)
    assert not (cert & ~sunny_donors(B, A)).any(), cert


def test_the_support_drops_exactly_the_shady_donors_when_some_donor_is_sunny():
    B, A = _with_a_shady_donor()
    assert sunny_support(B, A).tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# edge and failure
# --------------------------------------------------------------------------- #
def test_one_pre_period_is_allowed():
    B = np.array([[1.0, 3.0, 2.0]])
    A = np.array([2.0])
    flags = sunny_donors(B, A)
    assert flags.shape == (3,)


@pytest.mark.parametrize("bad, msg", [
    (np.ones(3), "2-D"),
    (np.ones((2, 0)), "no columns"),
])
def test_rejects_a_malformed_design(bad, msg):
    with pytest.raises(ValueError, match=msg):
        sunny_donors(bad, np.ones(2))


def test_rejects_a_target_of_the_wrong_length():
    with pytest.raises(ValueError, match="rows"):
        sunny_donors(np.ones((3, 4)), np.ones(2))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_a_non_finite_entry(bad):
    B = np.ones((2, 3)); B[0, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        sunny_donors(B, np.ones(2))


def test_rejects_a_two_dimensional_target():
    with pytest.raises(ValueError, match="1-D"):
        sunny_donors(np.ones((2, 3)), np.ones((2, 1)))


def test_no_donor_is_certified_when_every_donor_matches_the_treated_path():
    """All columns centre to zero, so the Gram has nothing positive to work with,
    every donor is shady with ``alpha* = 0``, and the exact fit is what the screen
    has found."""
    A = np.array([3.0, -1.0])
    B = np.tile(A[:, None], (1, 4))
    assert not certified_sunny(B, A).any()
    assert not sunny_donors(B, A).any()
    assert np.allclose(sunny_alphas(B, A), 0.0)
    assert sunny_support(B, A).tolist() == [0, 1, 2, 3]


def test_reports_a_linear_program_failure_instead_of_swallowing_it(monkeypatch):
    """A failed solve has to reach the caller, and the donor has to survive it: a
    wrong "shady" drops a column that may carry weight, a wrong "sunny" only keeps
    one. The call counter is there so a patch that misses its target cannot pass."""
    import mlsynth.utils.solvers.sunny as mod

    calls = []

    class _Failed:
        success = False
        message = " solver gave up "

    def _fail(*args, **kwargs):
        calls.append(1)
        return _Failed()

    monkeypatch.setattr(mod, "linprog", _fail)
    B, A = _with_a_shady_donor()
    with pytest.warns(RuntimeWarning, match="did not solve"):
        flags = mod.sunny_donors(B, A, certify=False)
    assert len(calls) == B.shape[1], calls
    assert flags.all(), flags


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_a_non_finite_target(bad):
    A = np.ones(2); A[1] = bad
    with pytest.raises(ValueError, match="finite"):
        sunny_donors(np.ones((2, 3)), A)
