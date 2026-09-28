"""MSCMT's sunny-donor cascade: the two branches taken before the outer search.

Becker and Klossner (2018), Section 3.1 and Figure 2, branch three ways on the
number of sunny donors.

* None sunny. By Proposition 1 that happens exactly when an exact predictor fit
  exists, and then the inner optimum does not depend on the predictor weights ``V``
  at all -- every ``w`` with ``X1 - X0 w = 0`` ties. Their Eq (10) states the
  selection: among those weights, take the one with the best outcome fit. Solving
  ``min w' Z' Z w`` subject to ``X w = 0``, ``w >= 0``, ``1'w = 1`` settles the fit
  and the outer search has nothing to do.
* Exactly one sunny. That donor takes all the weight for every ``V``, so there is
  again no outer problem and ``V`` is unidentified.
* Two or more. Prune the shady donors and run the nested search.

The tests below pin the first two. Each asserts the design actually reaches the
branch under test, so a construction that stopped triggering it fails here instead
of passing on the fall-through path.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel import BilevelProblem, solve_bilevel
from mlsynth.utils.bilevel.mscmt import _sunny_mask


# --------------------------------------------------------------------------- #
# designs whose branch is fixed by construction
# --------------------------------------------------------------------------- #
def _exact_fit_design():
    """Four donors, two predictors, and ``X1`` inside the donor hull, so an exact
    predictor fit exists and no donor is sunny. With ``J = 4`` against ``K = 2``
    plus the simplex row, the exact-fit set is a segment, so Eq (10) has a genuine
    choice to make and the test can check which end it takes.

    The outcome data is built so the unconstrained outcome optimum is far from the
    exact-fit segment, which keeps the earlier ``mscmt-feasible`` exit from firing.
    """
    X0 = np.array([[0.0, 2.0, 1.0, 3.0],
                   [0.0, 0.0, 2.0, 2.0]])
    X1 = X0 @ np.full(4, 0.25)                      # = [1.5, 1.0]
    # The exact-fit set is the segment between (0, .5, .5, 0) and (.5, 0, 0, .5).
    # Unweighted NNLS on the equality block alone returns the second of those, so the
    # outcome target is aimed at the first: a branch that ignored the outcome rows
    # would return the wrong end and the optimality test would catch it. The target
    # sits just outside the segment (weight .6/.4, not .5/.5) so the global outcome
    # optimum stays predictor-infeasible and the earlier mscmt-feasible exit does not
    # swallow the design.
    rng = np.random.default_rng(5)
    T = 14
    Y0 = rng.normal(size=(T, 4))
    y1 = Y0 @ np.array([0.0, 0.6, 0.4, 0.0]) + 0.02 * rng.normal(size=T)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def _single_sunny_design():
    """All three predictor discrepancies lie on one ray, ``r_j = c_j d`` with
    ``c = (1, 2, 3)``. The hull is the segment from ``d`` to ``3 d``, so ``alpha r_1``
    meets it only at ``alpha = 1`` while ``r_2`` and ``r_3`` are reached at 1/2 and
    1/3. Donor 1 is the only sunny one, and it takes all the weight for every ``V``
    because minimising ``|c' w|`` over the simplex puts everything on the smallest
    ``c``."""
    d = np.array([1.0, 2.0])
    c = np.array([1.0, 2.0, 3.0])
    X1 = np.array([10.0, 20.0])
    X0 = X1[:, None] - d[:, None] * c[None, :]
    rng = np.random.default_rng(11)
    T = 12
    Y0 = rng.normal(size=(T, 3))
    y1 = Y0 @ np.array([0.0, 0.3, 0.7]) + 0.05 * rng.normal(size=T)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def _eq10_reference(prob):
    """Eq (10) solved directly, as an independent check on the branch."""
    import cvxpy as cp

    J = prob.n_donors
    w = cp.Variable(J, nonneg=True)
    pr = cp.Problem(
        cp.Minimize(cp.sum_squares(prob.y1_pre - prob.Y0_pre @ w)),
        [cp.sum(w) == 1, prob.X0 @ w == prob.X1],
    )
    pr.solve(solver=cp.CLARABEL)
    assert pr.status == "optimal", pr.status
    return np.asarray(w.value, dtype=float)


# --------------------------------------------------------------------------- #
# the screen's raw verdict
# --------------------------------------------------------------------------- #
def test_the_mask_reports_no_sunny_donor_when_an_exact_fit_exists():
    """The count has to survive to the caller: a mask forced to all-True when
    everything is shady cannot drive the branch that case is supposed to take."""
    prob = _exact_fit_design()
    assert int(_sunny_mask(prob.X1, prob.X0).sum()) == 0


def test_the_mask_reports_exactly_one_sunny_donor_on_the_ray_design():
    prob = _single_sunny_design()
    mask = _sunny_mask(prob.X1, prob.X0)
    assert mask.tolist() == [True, False, False], mask


# --------------------------------------------------------------------------- #
# branch 1: no sunny donors -> Eq (10)
# --------------------------------------------------------------------------- #
def test_no_sunny_donor_takes_the_exact_fit_branch():
    prob = _exact_fit_design()
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
    assert sol.stage == "mscmt-exact-fit", sol.stage
    assert sol.iterations == 0


def test_the_exact_fit_branch_returns_a_perfect_predictor_fit():
    prob = _exact_fit_design()
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
    np.testing.assert_allclose(prob.X0 @ sol.W, prob.X1, atol=1e-7)
    assert sol.W.min() >= -1e-9
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-7)


def test_the_exact_fit_branch_picks_the_outcome_best_perfect_fit():
    """Eq (10) is a selection rule, not just any feasible point. Solved
    independently with cvxpy, the two must agree on the objective."""
    prob = _exact_fit_design()
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
    w_ref = _eq10_reference(prob)
    ours = float(np.sum((prob.y1_pre - prob.Y0_pre @ sol.W) ** 2))
    ref = float(np.sum((prob.y1_pre - prob.Y0_pre @ w_ref) ** 2))
    # Equality, not "no worse": a one-sided check passes for any point that happens
    # to beat the reference, which would hide a branch solving a different program.
    assert ours == pytest.approx(ref, rel=1e-6, abs=1e-9), (ours, ref)
    # The set has to discriminate, or the test passes for any feasible point. The
    # affine dimension is only an upper bound on the polytope's, so check the
    # polytope: two distinct vertices, and the outcome objective differing across
    # them by far more than the tolerance above.
    import itertools
    from scipy.optimize import linprog

    J = prob.n_donors
    A_eq = np.vstack([prob.X0, np.ones(J)]); b_eq = np.r_[prob.X1, 1.0]
    seen, objs = set(), []
    for c in itertools.product((-1.0, 1.0), repeat=J):
        r = linprog(np.array(c), A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                    method="highs-ds")
        if r.success:
            seen.add(tuple(np.round(r.x, 9)))
            objs.append(float(np.sum((prob.y1_pre - prob.Y0_pre @ r.x) ** 2)))
    assert len(seen) >= 2, seen
    assert max(objs) - min(objs) > 1.0, (min(objs), max(objs))
    # and the winner is not the one the equality block alone would hand back
    assert not np.allclose(sol.W, [0.5, 0.0, 0.0, 0.5], atol=1e-3), sol.W


def _count_de(monkeypatch):
    """Count calls to the outer search.

    ``solve_mscmt`` does ``from scipy.optimize import differential_evolution``
    inside the function body, so the name it calls is looked up on scipy at call
    time and patching the mscmt module attribute intercepts nothing. Patch scipy.
    ``test_the_de_counter_actually_intercepts`` is the control that keeps this
    honest -- without it, a counter that never fires reads the same as a branch
    that skipped the search.
    """
    import scipy.optimize as sciopt

    calls = []
    real = sciopt.differential_evolution

    def _counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sciopt, "differential_evolution", _counted)
    return calls


def test_the_de_counter_actually_intercepts(monkeypatch):
    """Control for the two tests below: on the ordinary path the search does run,
    so the counter is wired to the right binding."""
    calls = _count_de(monkeypatch)
    rng = np.random.default_rng(7)
    J, K, T = 12, 3, 16
    prob = BilevelProblem(
        y1_pre=rng.normal(size=T), Y0_pre=rng.normal(size=(T, J)),
        X1=rng.normal(size=K) * 3.0, X0=rng.normal(size=(K, J)),
    )
    solve_bilevel(prob, method="mscmt", seed=0, maxiter=60)
    assert len(calls) >= 1, "the patch never intercepted; the other counters are void"


def test_the_exact_fit_branch_runs_no_outer_search(monkeypatch):
    calls = _count_de(monkeypatch)
    prob = _exact_fit_design()
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
    assert calls == [], f"{len(calls)} outer searches ran on the exact-fit branch"
    assert sol.stage == "mscmt-exact-fit"


# --------------------------------------------------------------------------- #
# branch 2: exactly one sunny donor
# --------------------------------------------------------------------------- #
def test_one_sunny_donor_takes_all_the_weight():
    prob = _single_sunny_design()
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
    assert sol.stage == "mscmt-single-sunny", sol.stage
    np.testing.assert_allclose(sol.W, [1.0, 0.0, 0.0], atol=1e-9)
    assert sol.iterations == 0


def test_the_single_sunny_answer_does_not_depend_on_the_search(monkeypatch):
    calls = _count_de(monkeypatch)
    prob = _single_sunny_design()
    a = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
    b = solve_bilevel(prob, method="mscmt", seed=99, maxiter=400)
    assert calls == [], f"{len(calls)} outer searches ran on the single-sunny branch"
    np.testing.assert_allclose(a.W, b.W, atol=1e-12)


def test_an_exact_fit_that_does_not_converge_is_reported_and_falls_back(monkeypatch):
    """Proposition 1 says the fit exists whenever no donor is sunny, so a large
    equality residual means the screen and the fit disagree numerically. That has to
    reach the caller and must not be returned as an exact fit."""
    import mlsynth.utils.bilevel.mscmt as mod

    monkeypatch.setattr(
        mod, "_exact_predictor_fit_weights",
        lambda prob, mu=1e6: (np.full(prob.n_donors, 1.0 / prob.n_donors), 1.0),
    )
    prob = _exact_fit_design()
    with pytest.warns(RuntimeWarning, match="equality residual"):
        sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
    assert sol.stage != "mscmt-exact-fit", sol.stage


def test_pruning_off_leaves_the_cascade_out_of_the_way():
    """``prune_shady=False`` says do not use the screen, so neither branch may fire
    even on a design that would take one."""
    prob = _single_sunny_design()
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40, prune_shady=False)
    assert sol.stage not in ("mscmt-exact-fit", "mscmt-single-sunny"), sol.stage


def test_pruning_off_with_a_lone_donor_still_reports_no_screen_branch():
    """The one shape where the guard on the cascade is observable. With pruning off
    the mask is all-True, so ``n_sunny`` equals ``J`` and the count alone cannot
    distinguish "one sunny donor" from "one donor" -- at ``J = 1`` the two coincide.
    A screen-derived branch must not be reported when the screen is switched off,
    even though the weights are ``[1.0]`` either way."""
    rng = np.random.default_rng(3)
    prob = BilevelProblem(
        y1_pre=rng.normal(size=8), Y0_pre=rng.normal(size=(8, 1)),
        X1=rng.normal(size=2) * 3.0, X0=rng.normal(size=(2, 1)),
    )
    off = solve_bilevel(prob, method="mscmt", seed=0, maxiter=30, prune_shady=False)
    assert off.stage != "mscmt-single-sunny", off.stage
    assert off.metadata.get("mscmt_branch") != "single-sunny"

    on = solve_bilevel(prob, method="mscmt", seed=0, maxiter=30, prune_shady=True)
    assert on.stage == "mscmt-single-sunny", on.stage
    np.testing.assert_allclose(off.W, on.W, atol=1e-9)


def test_the_special_cases_report_that_v_is_unidentified():
    """In both branches the weights are the same for every ``V``, so the returned
    ``V`` carries no information and saying so is part of the answer."""
    for prob in (_exact_fit_design(), _single_sunny_design()):
        sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
        assert sol.metadata["v_identified"] is False


# --------------------------------------------------------------------------- #
# branch 3 is unchanged
# --------------------------------------------------------------------------- #
def test_two_or_more_sunny_donors_still_run_the_outer_search():
    """Regression pin: the ordinary case must keep its existing stage and its
    reported counts, so the cascade is additive."""
    rng = np.random.default_rng(7)
    J, K, T = 12, 3, 16
    prob = BilevelProblem(
        y1_pre=rng.normal(size=T), Y0_pre=rng.normal(size=(T, J)),
        X1=rng.normal(size=K) * 3.0, X0=rng.normal(size=(K, J)),
    )
    assert int(_sunny_mask(prob.X1, prob.X0).sum()) >= 2
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=60)
    assert sol.stage in ("mscmt", "mscmt-feasible"), sol.stage
    if sol.stage == "mscmt":
        assert sol.metadata["n_sunny"] + sol.metadata["n_shady_pruned"] == J


def test_the_branch_is_reported_on_every_path():
    """A caller has to be able to tell which rule produced the weights."""
    for prob, expected in ((_exact_fit_design(), "exact-fit"),
                           (_single_sunny_design(), "single-sunny")):
        sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=40)
        assert sol.metadata["mscmt_branch"] == expected
        assert sol.metadata["n_sunny"] == (0 if expected == "exact-fit" else 1)
