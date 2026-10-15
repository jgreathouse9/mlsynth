"""MAREX's MIQP reports the design its selection defines.

SCIP accepts a binary within its integrality tolerance of an integer. A
selection variable left at about 1e-6 then rounds to "not selected" while still
admitting a treated weight of the same size, and a market can carry weight on
both sides of the design. On one panel (``_panel(J=5, seed=5950)``, three
treated) SCIP's point used market u4 as a 1e-6 treated unit and a 0.146 control
unit at once. MAREX classified u4 as treated by its weight, zeroed its control
weight, and reported four treated units under ``m_eq=3``, control weights
summing to 0.854, and an objective of 6.7399 whose weights in fact score 117.4.

The fix fixes the selection at its rounding and refits the weights by a convex
QP with the selection held constant. The tests pin the invariant that broke --
the objective reported is the objective of the weights reported, and those
weights form a feasible design -- through SCIP on the panel that exposed it and
over random panels, and test the refit itself against the exact backend's
oracle without SCIP.

Levels: smoke, unit, edge, failure, property.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.marex_helpers.exact import side_fit
from mlsynth.utils.marex_helpers.formulation import (
    build_membership_mask,
    compute_cluster_means_members,
    precompute_distances,
    prepare_clusters,
    prepare_fit_slices,
)
from mlsynth.utils.marex_helpers.optimization import _augment_fit, refit_selection
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions
from mlsynth.tests.test_marex_exact_backend import _fit, _panel, _treated


# ------------------------------------------------------------------ helpers
def _pieces(J=8, seed=0, T0=10):
    """What solve_design builds before the solve, for one cluster."""
    df, T0 = _panel(J=J, seed=seed, T0=T0)
    Y = df.pivot(index="unit", columns="time", values="y").to_numpy()
    Y_np, clusters, N, labels, K, label_to_k = prepare_clusters(Y, np.zeros(J, int))
    Y_fit, _, _ = prepare_fit_slices(Y_np, T0, 0)
    X = _augment_fit(Y_fit, None, 1.0, False)
    M = build_membership_mask(clusters, label_to_k, N, K)
    Xbar, members = compute_cluster_means_members(X, M, labels)
    D1, D2 = precompute_distances(X, Xbar, members)
    return dict(X=X, Xbar=Xbar, members=members, labels=labels, M=M, D1=D1, D2=D2,
                N=N, K=K)


def _selection(N, treated):
    z = np.zeros((N, 1))
    z[list(treated), 0] = 1.0
    return z


def _refit(p, treated, m_eq, **design):
    return refit_selection(p["X"], p["Xbar"], p["members"], p["labels"], p["M"],
                           _selection(p["N"], treated), m_eq=m_eq,
                           D1=p["D1"], D2_list=p["D2"], **design)


def _recomputed(Y_fit, w, v, design="standard", beta=1e-6):
    A = Y_fit.mean(axis=0)
    u, c = Y_fit.T @ w, Y_fit.T @ v
    if design == "weakly_targeted":
        return float((A - u) @ (A - u) + beta * (u - c) @ (u - c))
    return float((A - u) @ (A - u) + (A - c) @ (A - c))


def _assert_feasible_design(res, m_eq):
    g = res.globres
    w = np.asarray(g.treated_weights_agg, float)
    v = np.asarray(g.control_weights_agg, float)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert v.sum() == pytest.approx(1.0, abs=1e-6)
    assert not np.any((w > 1e-8) & (v > 1e-8)), "a market on both sides"
    assert int((w > 1e-8).sum()) <= m_eq
    assert len(_treated(res)) <= m_eq
    Y_fit = np.asarray(g.Y_fit, float)
    assert g.objective == pytest.approx(_recomputed(Y_fit, w, v), rel=1e-6, abs=1e-9)


# -------------------------------------------------------------------- smoke
def test_the_panel_that_exposed_it_now_reports_a_feasible_design():
    df, T0 = _panel(J=5, seed=5950, n_good=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mip = _fit(df, T0, "mip", m_eq=3)
        exact = _fit(df, T0, "exact", m_eq=3)
    _assert_feasible_design(mip, 3)
    om, oe = mip.globres.objective, exact.globres.objective
    assert oe <= om + 1e-6 * (1.0 + abs(om))
    assert _treated(mip) == _treated(exact)


# --------------------------------------------------------------------- unit
@pytest.mark.parametrize("treated", [(1, 2, 4), (0, 3), (5,)])
def test_the_refit_is_the_exact_oracle_for_the_standard_design(treated):
    p = _pieces()
    w, v, obj = _refit(p, treated, m_eq=len(treated), design="standard")
    B, A = p["X"].T, p["Xbar"][0]
    control = [j for j in range(p["N"]) if j not in treated]
    oracle = side_fit(B, A, list(treated))[0] + side_fit(B, A, control)[0]
    assert obj == pytest.approx(oracle, rel=1e-6, abs=1e-9)
    # exactly zero, not zero to the QP solver's tolerance: the orchestrator
    # classifies markets by their weights
    assert (w[control, 0] == 0.0).all()
    assert (v[list(treated), 0] == 0.0).all()
    assert w[:, 0].sum() == pytest.approx(1.0) and v[:, 0].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("design, extra", [
    ("weakly_targeted", {"beta": 1.0}),
    ("penalized", {"lambda1": 0.05, "lambda2": 0.02}),
])
def test_the_refit_attains_its_objective_for_the_other_designs(design, extra):
    p = _pieces(seed=3)
    treated = (0, 2, 5)
    w, v, obj = _refit(p, treated, m_eq=3, design=design, **extra)
    control = [j for j in range(p["N"]) if j not in treated]
    assert (w[control, 0] == 0.0).all()
    assert (v[list(treated), 0] == 0.0).all()
    A = p["Xbar"][0]
    u, c = p["X"].T @ w[:, 0], p["X"].T @ v[:, 0]
    if design == "weakly_targeted":
        expected = (A - u) @ (A - u) + extra["beta"] * (u - c) @ (u - c)
    else:
        expected = ((A - u) @ (A - u) + (A - c) @ (A - c)
                    + extra["lambda1"] * w[:, 0] @ p["D1"][:, 0]
                    + extra["lambda2"] * v[:, 0] @ p["D1"][:, 0])
    assert obj == pytest.approx(float(expected), rel=1e-6, abs=1e-9)


def _joint_weakly(B, A, S, C, beta):
    """The weakly-targeted fit of a fixed selection, by a conic solver."""
    import cvxpy as cp
    w = cp.Variable(len(S), nonneg=True)
    v = cp.Variable(len(C), nonneg=True)
    obj = (cp.sum_squares(A - B[:, S] @ w)
           + beta * cp.sum_squares(B[:, S] @ w - B[:, C] @ v))
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1, cp.sum(v) == 1])
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value)


def test_the_weakly_targeted_refit_solves_the_fixed_selection():
    """The coupled design goes through the QP path; its value must be the
    minimum over the fixed selection, not a relaxation of it."""
    p = _pieces(seed=3)
    treated = [0, 2, 5]
    control = [j for j in range(p["N"]) if j not in treated]
    _, _, obj = _refit(p, tuple(treated), m_eq=3, design="weakly_targeted",
                       beta=1.0)
    ref = _joint_weakly(p["X"].T, p["Xbar"][0], treated, control, 1.0)
    assert obj == pytest.approx(ref, rel=1e-5)


def test_donor_exclusions_shrink_the_control_pool():
    p = _pieces()
    rules = DesignRestrictions(donor_exclusion=[(0, 3)])
    w, v, obj = _refit(p, (0, 1, 2), m_eq=3, design="standard",
                       restrictions=rules)
    assert v[3, 0] == 0.0
    B, A = p["X"].T, p["Xbar"][0]
    control = [4, 5, 6, 7]
    oracle = side_fit(B, A, [0, 1, 2])[0] + side_fit(B, A, control)[0]
    assert obj == pytest.approx(oracle, rel=1e-9)


# --------------------------------------------------------------------- edge
def test_a_single_control_market_takes_all_the_control_weight():
    p = _pieces(J=5, seed=4)
    w, v, _ = _refit(p, (0, 1, 2, 3), m_eq=4, design="standard")
    assert v[4, 0] == pytest.approx(1.0, abs=1e-8)


# ------------------------------------------------------------------ failure
def test_a_selection_the_constraints_reject_is_reported():
    p = _pieces()
    with pytest.raises(MlsynthEstimationError, match="refit"):
        _refit(p, (0, 1), m_eq=3, design="standard")


def test_a_qp_refit_the_constraints_reject_is_reported():
    """The coupled design refits through the solver; a selection its
    cardinality rejects must surface as the refit's error."""
    p = _pieces()
    with pytest.raises(MlsynthEstimationError, match="cannot refit"):
        _refit(p, (0, 1), m_eq=3, design="weakly_targeted", beta=1.0)


def test_a_selection_that_breaks_a_restriction_is_reported():
    p = _pieces()
    rules = DesignRestrictions(forced_in=[5])
    with pytest.raises(MlsynthEstimationError, match="restriction"):
        _refit(p, (0, 1, 2), m_eq=3, design="standard", restrictions=rules)


def test_a_cluster_left_with_no_control_is_reported():
    p = _pieces(J=4)
    rules = DesignRestrictions(donor_exclusion=[(0, 3)])
    with pytest.raises(MlsynthEstimationError, match="no control"):
        _refit(p, (0, 1, 2), m_eq=3, design="standard", restrictions=rules)


# ----------------------------------------------------------------- property
@given(seed=st.integers(0, 10_000), J=st.integers(3, 9), data=st.data())
@settings(max_examples=40, deadline=None)
def test_the_refit_matches_the_oracle_over_the_domain(seed, J, data):
    p = _pieces(J=J, seed=seed)
    m = data.draw(st.integers(1, J - 1))
    treated = tuple(sorted(data.draw(st.sets(st.integers(0, J - 1), min_size=m, max_size=m))))
    _, _, obj = _refit(p, treated, m_eq=m, design="standard")
    B, A = p["X"].T, p["Xbar"][0]
    control = [j for j in range(J) if j not in treated]
    oracle = side_fit(B, A, list(treated))[0] + side_fit(B, A, control)[0]
    assert obj == pytest.approx(oracle, rel=1e-6, abs=1e-9)


@given(seed=st.integers(0, 10_000), J=st.integers(5, 8), m=st.integers(1, 3))
@settings(max_examples=12, deadline=None)
def test_the_mip_reports_a_feasible_design_over_random_panels(seed, J, m):
    df, T0 = _panel(J=J, seed=seed, n_good=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _fit(df, T0, "mip", m_eq=m)
    _assert_feasible_design(res, m)
