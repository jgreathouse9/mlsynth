"""A linear term in the simplex QP: ``min ||A - Bw||^2 + l'w`` over the simplex.

Abadie and L'Hour's penalised SCM is this shape, with ``l = lam * d2`` the
pairwise donor discrepancies. On the non-negative orthant an L1 penalty is
linear, so this is the weighted non-negative lasso, and the active-set family
is its instrument (Cobb et al. 2025 for the pure case).

The term cannot be folded into the target in general. Folding needs
``l`` in ``range(B')``, which Zou and Hastie (2005, Lemma 1) identify as a rank
condition -- their augmented design has rank ``p`` and that is what makes their
transformation exact. On Proposition 99 the donor block is 19 by 38 with rank
19, and 40 percent of the centred penalty lies outside that row space.

It folds on the *free set*, which is where the active set actually solves,
for the part of the term in the range of the free block. That set is small,
but it need not be of full column rank: it can be wider than the design has
rows, or hold two near-copies of one market. The part of the term along
directions the block cannot resolve is a ray the loop follows to a bound, and
the rest folds into the residual, so the inner solve stays in residual form.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.solvers.active_set import solve_simplex_qp


@pytest.fixture(scope="module")
def wide():
    """The regime penalised SCM is for: more donors than pre-periods."""
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "P99data.csv")
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    p = dataprep(df, "state", "year", "cigsale", "treat")
    T0 = int(p["pre_periods"])
    X0 = np.asarray(p["donor_matrix"], float)[:T0]
    X1 = np.asarray(p["y"], float).ravel()[:T0]
    s = float(max(np.abs(X0).max(), np.abs(X1).max()))
    X1, X0 = X1 / s, X0 / s
    return X0, X1, np.sum((X1[:, None] - X0) ** 2, axis=0)      # the ADH penalty


def _cvxpy(B, A, lin):
    import cvxpy as cp
    w = cp.Variable(B.shape[1], nonneg=True)
    obj = cp.Minimize(cp.sum_squares(A - B @ w) + lin @ w)
    cp.Problem(obj, [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    return np.asarray(w.value, float).ravel()


def _obj(B, A, lin, w):
    return float(np.sum((A - B @ w) ** 2) + lin @ w)


# --------------------------------------------------------------------------
# It is not foldable, which is why the solver has to carry it
# --------------------------------------------------------------------------
def test_the_penalty_does_not_lie_in_the_row_space_of_a_wide_donor_block(wide):
    B, _, d2 = wide
    c = d2 - d2.mean()                       # constants are free on the simplex
    resid = c - B.T @ np.linalg.lstsq(B.T, c, rcond=None)[0]
    assert np.linalg.matrix_rank(B) < B.shape[1]
    assert np.linalg.norm(resid) / np.linalg.norm(c) > 0.3


# --------------------------------------------------------------------------
# Default behaviour is untouched
# --------------------------------------------------------------------------
def test_omitting_the_linear_term_is_the_old_program(wide):
    B, A, _ = wide
    assert solve_simplex_qp(B, A, linear=None) == pytest.approx(
        solve_simplex_qp(B, A), abs=0.0
    )


def test_a_zero_linear_term_changes_nothing(wide):
    B, A, _ = wide
    got = solve_simplex_qp(B, A, linear=np.zeros(B.shape[1]))
    assert got == pytest.approx(solve_simplex_qp(B, A), abs=1e-12)


# --------------------------------------------------------------------------
# It solves the program it claims to
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [1e-6, 1e-3, 0.1, 1.0, 10.0])
def test_it_matches_cvxpy_across_the_penalty_grid(wide, lam):
    B, A, d2 = wide
    lin = lam * d2
    got = solve_simplex_qp(B, A, linear=lin)
    ref = _cvxpy(B, A, lin)
    assert got.min() >= 0.0
    assert got.sum() == pytest.approx(1.0, abs=1e-9)
    assert _obj(B, A, lin, got) <= _obj(B, A, lin, ref) + 1e-9


@pytest.mark.parametrize("lam", [1e-3, 0.1, 1.0])
def test_the_certificate_holds_with_the_linear_term(wide, lam):
    """Stationarity on the support, dual feasibility off it, with the gradient
    the linear term shifts: ``g = 2B'(Bw - A) + l``."""
    B, A, d2 = wide
    lin = lam * d2
    w = solve_simplex_qp(B, A, linear=lin)
    g = 2.0 * (B.T @ (B @ w - A)) + lin
    on = w > 1e-9
    nu = float(g[on].mean())
    scale = max(float(np.abs(g).max()), 1e-12)
    assert np.abs(g[on] - nu).max() / scale < 1e-8          # stationarity
    assert (g[~on] - nu).min() / scale > -1e-8              # dual feasibility


def test_a_heavier_penalty_gives_a_sparser_fit(wide):
    """The penalty prices each donor by its distance from the treated unit, so
    raising it drives weight onto the nearest donors."""
    B, A, d2 = wide
    light = solve_simplex_qp(B, A, linear=1e-6 * d2)
    heavy = solve_simplex_qp(B, A, linear=50.0 * d2)
    assert int((heavy > 1e-9).sum()) <= int((light > 1e-9).sum())
    assert float(d2 @ heavy) < float(d2 @ light)


def test_a_huge_penalty_selects_the_single_nearest_donor(wide):
    B, A, d2 = wide
    w = solve_simplex_qp(B, A, linear=1e6 * d2)
    assert int(np.argmax(w)) == int(np.argmin(d2))
    assert w.max() > 0.99


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------
def test_a_constant_shift_of_the_penalty_is_free_on_the_simplex(wide):
    B, A, d2 = wide
    base = solve_simplex_qp(B, A, linear=0.5 * d2)
    shifted = solve_simplex_qp(B, A, linear=0.5 * d2 + 7.0)
    assert shifted == pytest.approx(base, abs=1e-8)


def test_the_solution_is_equivariant_to_a_common_rescaling(wide):
    """Scaling B and A by f scales the quadratic by f^2, so the linear term
    has to scale with it for the same program."""
    B, A, d2 = wide
    f = 1e3
    base = solve_simplex_qp(B, A, linear=0.3 * d2)
    got = solve_simplex_qp(B * f, A * f, linear=0.3 * d2 * f ** 2)
    assert got == pytest.approx(base, abs=1e-7)


def test_a_warm_start_does_not_change_the_answer(wide):
    B, A, d2 = wide
    lin = 0.2 * d2
    cold = solve_simplex_qp(B, A, linear=lin)
    warm = solve_simplex_qp(B, A, linear=lin, warm_start=np.full(B.shape[1], 1.0 / B.shape[1]))
    assert warm == pytest.approx(cold, abs=1e-9)


def test_a_misshaped_linear_term_raises(wide):
    B, A, _ = wide
    with pytest.raises(ValueError, match="linear"):
        solve_simplex_qp(B, A, linear=np.zeros(3))


# --------------------------------------------------------------------------
# Through the shared layer
# --------------------------------------------------------------------------
def test_the_objective_carries_a_linear_term(wide):
    from mlsynth.utils.weights import WeightConstraint, WeightObjective, solve_weights
    B, A, d2 = wide
    lin = 0.4 * d2
    sol = solve_weights(B, A, WeightConstraint(), WeightObjective(linear=lin))
    assert sol.status == "optimal" and sol.kkt_residual < 1e-8
    assert sol.weights == pytest.approx(solve_simplex_qp(B, A, linear=lin), abs=1e-12)
    assert sol.objective == pytest.approx(_obj(B, A, lin, np.array(sol.weights)), rel=1e-12)


def test_the_certificate_rejects_the_unpenalised_point(wide):
    """The residual must see the linear term, or it certifies the wrong program."""
    from mlsynth.utils.weights import (
        WeightConstraint, WeightObjective, kkt_residual, solve_weights,
    )
    B, A, d2 = wide
    lin = 0.4 * d2
    plain = np.array(solve_weights(B, A).weights)
    obj = WeightObjective(linear=lin)
    assert kkt_residual(B, A, plain, 0.0, WeightConstraint(), obj) > 1e-4


def test_a_linear_term_on_the_cone_is_refused_by_name(wide):
    """The cone with a linear term is the weighted non-negative lasso. It is a
    real program and a different algorithm -- an active-set NNLS pricing on the
    shifted gradient -- so it is refused here instead of approximated."""
    from mlsynth.exceptions import MlsynthConfigError
    from mlsynth.utils.weights import WeightConstraint, WeightObjective, solve_weights
    B, A, d2 = wide
    with pytest.raises(MlsynthConfigError, match="non-negative lasso"):
        solve_weights(B, A, WeightConstraint(sum_to_one=False),
                      WeightObjective(linear=0.4 * d2))


def test_a_misshaped_linear_term_is_refused_by_the_objective():
    from mlsynth.exceptions import MlsynthConfigError
    from mlsynth.utils.weights import WeightObjective
    with pytest.raises(MlsynthConfigError, match="linear"):
        WeightObjective(linear=np.array([1.0, np.nan]))


def test_a_linear_term_of_the_wrong_length_is_refused_by_the_layer(wide):
    from mlsynth.exceptions import MlsynthConfigError
    from mlsynth.utils.weights import WeightConstraint, WeightObjective, solve_weights
    B, A, _ = wide
    with pytest.raises(MlsynthConfigError, match="entries but there are"):
        solve_weights(B, A, WeightConstraint(), WeightObjective(linear=np.zeros(3)))


def test_a_non_finite_linear_term_raises_in_the_solver(wide):
    B, A, d2 = wide
    bad = d2.copy(); bad[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        solve_simplex_qp(B, A, linear=bad)


# --------------------------------------------------------------------------
# Validation runs before any shortcut. A single donor is forced to weight 1
# whatever the linear term says, so a malformed one changes no number -- and
# is still refused, because accepting it silently is the leniency this library
# declines on purpose.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.zeros(5), np.array([np.nan]), np.array([np.inf])])
def test_a_malformed_linear_term_is_refused_even_for_a_single_donor(bad):
    B = np.array([[1.0], [2.0], [3.0]])
    A = np.array([1.0, 2.0, 3.5])
    with pytest.raises(ValueError, match="linear"):
        solve_simplex_qp(B, A, linear=bad)


def test_a_well_formed_linear_term_still_forces_the_lone_donor():
    B = np.array([[1.0], [2.0], [3.0]])
    A = np.array([1.0, 2.0, 3.5])
    assert solve_simplex_qp(B, A, linear=np.array([7.0])) == pytest.approx([1.0])


# --------------------------------------------------------------------------
# The fold is exact only when the free set fits inside the design's row space.
# It does not hold for a design with fewer rows than the free set needs, and
# `bilevel/penalized.py` builds exactly one of those: `R` is the rank-K factor
# of a wide Gram, so it can have far fewer rows than donors. It need not hold
# on a panel either: two near-copies of a market leave the free block flat to
# working precision along their difference (the section on #782 below).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m,J", [(8, 12), (12, 12), (6, 6), (20, 12)])
def test_it_is_optimal_when_the_design_admits_the_free_set(m, J):
    import cvxpy as cp
    rng = np.random.default_rng(0)
    B = rng.normal(size=(m, J)); A = np.zeros(m); lin = np.abs(rng.normal(size=J))
    got = solve_simplex_qp(B, A, linear=lin)
    v = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(A - B @ v) + lin @ v),
               [cp.sum(v) == 1]).solve(solver=cp.CLARABEL)
    obj = lambda w: float(np.sum((A - B @ w) ** 2) + lin @ w)
    assert got.min() >= 0.0 and got.sum() == pytest.approx(1.0, abs=1e-9)
    assert obj(got) <= obj(np.asarray(v.value, float).ravel()) + 1e-8


@pytest.mark.parametrize("m,J", [
    (1, 8), (2, 6), (2, 20), (3, 12), (4, 12), (5, 40),
    (6, 6), (8, 12), (12, 12), (20, 12),
])
def test_it_reaches_the_minimiser_whatever_the_design_shape(m, J):
    """A free set wider than the design has rows makes the subproblem unbounded
    along a null direction, which the loop answers by stepping the ray to its
    blocking bound. Before that branch existed, 3 by 12 came back 54 percent
    above the optimum reporting convergence."""
    import cvxpy as cp
    rng = np.random.default_rng(0)
    B = rng.normal(size=(m, J)); A = np.zeros(m); lin = np.abs(rng.normal(size=J))

    got = solve_simplex_qp(B, A, linear=lin)
    v = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(A - B @ v) + lin @ v),
               [cp.sum(v) == 1]).solve(solver=cp.CLARABEL)

    obj = lambda x: float(np.sum((A - B @ x) ** 2) + lin @ x)
    assert got.min() >= 0.0
    assert got.sum() == pytest.approx(1.0, abs=1e-9)
    assert obj(got) <= obj(np.asarray(v.value, float).ravel()) + 1e-8


@pytest.mark.parametrize("seed", range(12))
def test_it_reaches_the_minimiser_on_random_wide_designs(seed):
    """The regime the branch exists for, swept: far more donors than rows."""
    import cvxpy as cp
    rng = np.random.default_rng(seed)
    m, J = int(rng.integers(1, 6)), int(rng.integers(8, 30))
    B = rng.normal(size=(m, J)); A = rng.normal(size=m)
    lin = np.abs(rng.normal(size=J)) * float(rng.choice([1e-3, 1.0, 10.0]))

    got = solve_simplex_qp(B, A, linear=lin)
    v = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(A - B @ v) + lin @ v),
               [cp.sum(v) == 1]).solve(solver=cp.CLARABEL)

    obj = lambda x: float(np.sum((A - B @ x) ** 2) + lin @ x)
    assert obj(got) <= obj(np.asarray(v.value, float).ravel()) + 1e-8
    assert got.min() >= 0.0 and got.sum() == pytest.approx(1.0, abs=1e-9)


def test_the_ray_step_keeps_the_weights_on_the_hyperplane():
    """The ray is mapped through the difference basis, so it sums to zero and
    the step cannot leave the sum-to-one constraint."""
    rng = np.random.default_rng(3)
    B = rng.normal(size=(2, 15)); A = np.zeros(2); lin = np.abs(rng.normal(size=15))
    w = solve_simplex_qp(B, A, linear=lin)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert w.min() >= 0.0



# --------------------------------------------------------------------------
# The guard itself. The ray branch is what stops a non-stationary point being
# produced. It was recorded as never firing in practice until #782, where a
# ray read from rounding noise made the loop cycle on a 12-by-3 panel and the
# guard turned a point 17.5 percent above the optimum into a raise. With the
# null part a projection it no longer fires on the near-collinear designs
# swept below, and its mutant is again equivalent; it is tested here directly,
# so that its behaviour is measured and not assumed.
# --------------------------------------------------------------------------
def test_the_guard_accepts_a_stationary_point(wide):
    from mlsynth.utils.solvers.active_set import _assert_optimal_with_linear
    B, A, d2 = wide
    lin = 0.3 * d2
    w = solve_simplex_qp(B, A, linear=lin)
    _assert_optimal_with_linear(B, A, w, lin, 1e-9)      # does not raise


def test_the_guard_rejects_a_point_that_is_not_stationary(wide):
    from mlsynth.utils.solvers.active_set import _assert_optimal_with_linear
    B, A, d2 = wide
    lin = 0.3 * d2
    w = np.array(solve_simplex_qp(B, A, linear=lin))
    moved = w.copy()
    lead = int(np.argmax(moved))
    other = int(np.argmin(np.where(moved > 0, moved, np.inf)))
    shift = min(0.25, moved[lead])
    moved[lead] -= shift
    moved[other] += shift
    with pytest.raises(ValueError, match="did not reach the minimiser"):
        _assert_optimal_with_linear(B, A, moved, lin, 1e-9)


# --------------------------------------------------------------------------
# Near-collinear columns (#782). Two markets that agree to 1e-12 give a free
# set whose difference block M has one singular value 2.5e-11 of the largest:
# full rank on paper, flat to working precision. The shipped ray test read the
# null part of the term as the residual q - M'u of a solve that kept that
# direction, so it was rounding noise of size eps |M| |u|, and the ray followed
# its sign. On the reproduction below the ray pinned the cheaper copy, the dual
# test released it, and the loop cycled through all 150 pivots, returning a
# point 17.5 percent above the optimum for the guard to catch. Over 400
# near-collinear designs 30 raised and 8 more came back silently above the
# optimum or unconverged.
#
# The ladder, rung by rung: the reproduction (0 and 1), the pivot count that
# shows the cycle (2), the property over near-collinear designs (3), and the
# error contract when the guard fires (4). The mutants are in
# tools/mutation/targets.toml, target active-set-linear-ray (5).
# --------------------------------------------------------------------------
def _near_copies(seed=0, gap=1e-12):
    """A 12-by-3 design whose second and third columns are near-copies."""
    rng = np.random.default_rng(seed)
    n, T = 8, 12
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = 1e3 * (20 + f @ rng.uniform(0.5, 1.5, size=(2, n))
               + rng.normal(size=(T, n)) * 0.3)
    B[:, 5] = B[:, 4] * (1 + gap * rng.normal(size=T))
    A = B.mean(axis=1)
    a = np.abs(np.random.default_rng(seed).normal(size=n)) * 1e6
    cols = [1, 4, 5]
    return B[:, cols], A, a[cols]


def _reference(B, A, lin):
    """Clarabel at tight tolerances: the minimiser, renormalised onto the simplex."""
    import cvxpy as cp
    w = cp.Variable(B.shape[1])
    cp.Problem(cp.Minimize(cp.sum_squares(A - B @ w) + lin @ w),
               [w >= 0, cp.sum(w) == 1]).solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-12, tol_feas=1e-12)
    w = np.maximum(np.asarray(w.value, float).ravel(), 0.0)
    return w / w.sum()


def test_near_copies_with_different_penalties_reach_the_optimum():
    """Rungs 0 and 1: the answer, and the convergence flag that moved with it."""
    B, A, lin = _near_copies()
    w, info = solve_simplex_qp(B, A, linear=lin, return_info=True)
    ref = _reference(B, A, lin)
    np.testing.assert_allclose(w, [0.089881, 0.0, 0.910119], atol=1e-6)
    assert _obj(B, A, lin, w) <= _obj(B, A, lin, ref) * (1 + 1e-12)
    assert info["converged"]


def test_a_near_null_direction_does_not_make_the_loop_cycle():
    """Rung 2: the shipped loop alternated a ray step and a release 75 times."""
    B, A, lin = _near_copies()
    _, info = solve_simplex_qp(B, A, linear=lin, return_info=True)
    assert info["pivots"] <= B.shape[1]


@pytest.mark.parametrize("gap", [0.0, 1e-15, 1e-12, 1e-9, 1e-6])
def test_the_cheaper_copy_carries_the_weight_at_every_gap(gap):
    """Exact copies, rounding-level gaps and resolvable ones alike.

    The quadratic barely tells the copies apart, so the penalty decides: the
    optimum puts the pair's weight on the cheaper one.
    """
    B, A, lin = _near_copies(gap=gap)
    w = solve_simplex_qp(B, A, linear=lin)
    cheap, dear = (2, 1) if lin[2] < lin[1] else (1, 2)
    assert w[dear] <= 1e-9
    assert _obj(B, A, lin, w) <= _obj(B, A, lin, _reference(B, A, lin)) * (1 + 1e-10)


def test_nearly_equal_penalties_on_near_copies_still_reach_the_optimum():
    """The term then has almost nothing along the near-null direction."""
    B, A, lin = _near_copies()
    lin = lin.copy()
    lin[2] = lin[1] * (1 + 1e-9)
    w, info = solve_simplex_qp(B, A, linear=lin, return_info=True)
    assert info["converged"]
    assert _obj(B, A, lin, w) <= _obj(B, A, lin, _reference(B, A, lin)) * (1 + 1e-10)


def test_a_point_the_guard_rejects_is_an_estimation_error():
    """Rung 4: the guard's failure is an estimation failure, and still a ValueError.

    `bilevel/penalized.py` catches ValueError from the solver and re-raises it
    as MlsynthEstimationError, so the translated type keeps both.
    """
    from mlsynth.exceptions import MlsynthEstimationError
    from mlsynth.utils.solvers.active_set import _assert_optimal_with_linear
    B, A, lin = _near_copies()
    w = np.array([0.100211, 0.899789, 0.0])             # the face the cycle ended on
    with pytest.raises(MlsynthEstimationError, match="did not reach the minimiser") as exc:
        _assert_optimal_with_linear(B, A, w, lin, 1e-9)
    assert isinstance(exc.value, ValueError)
    assert "A panel does not meet that condition" not in str(exc.value)


from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st         # noqa: E402


@given(seed=st.integers(0, 10_000), J=st.integers(3, 6), T=st.integers(6, 20),
       gap_exp=st.sampled_from([-16, -15, -14, -13, -12, -11, -10, -8, -6, -3]),
       scale_exp=st.floats(3, 7), equal=st.booleans())
@example(seed=3, J=3, T=6, gap_exp=-14, scale_exp=3.0, equal=False)
@settings(max_examples=80, deadline=None)
def test_it_reaches_the_minimiser_on_near_collinear_designs(
        seed, J, T, gap_exp, scale_exp, equal):
    """Rung 3: over near-copies at every gap, with the term along the pair or not."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = 1e3 * (20 + f @ rng.uniform(0.5, 1.5, size=(2, J))
               + rng.normal(size=(T, J)) * 0.3)
    i = int(rng.integers(0, J - 1))
    B[:, i + 1] = B[:, i] * (1 + 10.0 ** gap_exp * rng.normal(size=T))
    A = B[:, rng.permutation(J)[:2]].mean(axis=1) + rng.normal(size=T) * 200 * rng.random()
    lin = np.abs(rng.normal(size=J)) * 10.0 ** scale_exp
    if equal:
        lin[i + 1] = lin[i] * (1 + 1e-6 * rng.normal())
    w, info = solve_simplex_qp(B, A, linear=lin, return_info=True)
    ref = _reference(B, A, lin)
    assert info["converged"]
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0, abs=1e-9)
    assert _obj(B, A, lin, w) <= _obj(B, A, lin, ref) + 1e-8 * abs(_obj(B, A, lin, ref))
