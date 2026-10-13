r"""The Unit-level design has to produce a program a solver will accept.

Incident: ``MAREX(design="unit_penalized", xi=2.0, m_eq=2).fit()`` raises
``MlsynthEstimationError: ... Problem does not follow DCP rules`` for every
``xi > 0``. At ``xi = 0`` it runs, which is to say the one parameter that makes
it the Unit-level design is the one that breaks it.

The term was written as ``xi * w[j] * sum_squares(x_j - Y @ v)``: a decision
variable multiplying a convex function of another decision variable. The
Hessian of ``w v^2`` is ``[[0, 2v], [2v, 2w]]`` with determinant ``-4 v^2``, so
it is genuinely non-convex and not merely outside DCP's sufficient conditions.

Abadie and Zhao's equation (10) does not ask for that. It is
``xi * sum_j w_j * min_v ||x_j - sum_i v_ij x_i||^2`` -- ``w_j`` multiplies a
*minimum*, and the inner weights appear in one term scaled by a non-negative
``w_j``, so a positive scalar cannot move their argmin. The minimum is a
constant in ``w`` and the term is linear.

Two more terms share the shape and have no such linearisation. ``lambda2_unit``
pairs ``w`` against ``v`` through a distance matrix, so it is bilinear in two
decision variables; ``zeta`` minimises ``z(1 - z)``, which is concave. Both are
refused with a reason naming the alternative.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.marex_helpers.formulation import (
    build_objective,
    init_cvxpy_variables,
    per_unit_reproducibility,
    precompute_distances,
)


@pytest.fixture
def fit_panel():
    rng = np.random.default_rng(3)
    Y = np.abs(rng.normal(5.0, 1.0, (6, 12)))
    members = [np.arange(6)]
    return Y, members, [Y.mean(axis=0)]


# ---------------------------------------------- rung 2: the failing term
def test_the_unit_level_objective_is_a_program_a_solver_accepts(fit_panel):
    """The incident, as a unit test on the term that produced it."""
    Y, members, Xbar = fit_panel
    w, v, z = init_cvxpy_variables(N=6, K=1, boolean=False)
    obj = build_objective(Y, Xbar, members, w, v, z, design="unit_penalized", xi=0.5)
    assert obj.is_dcp(), "the Unit-level penalty must leave the program convex"


def test_the_penalty_is_linear_in_the_treated_weights(fit_panel):
    """Equation (10) multiplies w_j by a minimum over v, which is a constant.

    Asserted numerically: cvxpy cannot symbolically cancel the shared quadratic
    terms, so the difference of two convex objectives reports UNKNOWN curvature
    even when the part that differs is affine. What the penalty must satisfy is
    that it depends on w alone, and linearly.
    """
    Y, members, Xbar = fit_panel
    w, v, z = init_cvxpy_variables(N=6, K=1, boolean=False)
    base = build_objective(Y, Xbar, members, w, v, z, design="unit_penalized", xi=0.0)
    with_xi = build_objective(Y, Xbar, members, w, v, z, design="unit_penalized", xi=1.0)
    rng = np.random.default_rng(0)

    def penalty(wv, vv):
        w.value, v.value, z.value = wv, vv, wv
        return float(with_xi.args[0].value) - float(base.args[0].value)

    w1 = rng.dirichlet(np.ones(6))[:, None]
    w2 = rng.dirichlet(np.ones(6))[:, None]
    vv = rng.dirichlet(np.ones(6))[:, None]
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        mixed = alpha * w1 + (1 - alpha) * w2
        assert penalty(mixed, vv) == pytest.approx(
            alpha * penalty(w1, vv) + (1 - alpha) * penalty(w2, vv), rel=1e-9, abs=1e-9)

    # and it must not move with the control weights at all
    other_v = rng.dirichlet(np.ones(6))[:, None]
    assert penalty(w1, vv) == pytest.approx(penalty(w1, other_v), rel=1e-12, abs=1e-12)


def test_the_penalty_scores_an_unreachable_unit_higher(fit_panel):
    """It must still measure reproducibility, not merely be convex."""
    import cvxpy as cp
    rng = np.random.default_rng(7)
    Y = np.abs(rng.normal(5.0, 1.0, (6, 12)))
    Y[0, :] = Y[1:, :].max(axis=0) * 8.0        # unit 0 outside the others' hull
    members = [np.arange(6)]; Xbar = [Y.mean(axis=0)]
    w, v, z = init_cvxpy_variables(N=6, K=1, boolean=False)
    base = build_objective(Y, Xbar, members, w, v, z, design="unit_penalized", xi=0.0)
    with_xi = build_objective(Y, Xbar, members, w, v, z, design="unit_penalized", xi=1.0)
    added = with_xi.args[0] - base.args[0]
    # read the coefficient on each w_j by evaluating at the vertices
    coefs = []
    for j in range(6):
        wv = np.zeros((6, 1)); wv[j, 0] = 1.0
        w.value = wv; v.value = np.full((6, 1), 1 / 6); z.value = wv
        coefs.append(float(added.value))
    assert np.argmax(coefs) == 0, "the unreachable unit should carry the largest penalty"


def test_the_charge_is_a_convex_hull_distance_not_a_regression(fit_panel):
    """The inner program is a synthetic control, so it cannot extrapolate.

    With two donors the simplex is the segment between them, and the charge
    has a closed form: project the target onto the line and clip the
    coefficient to [0, 1]. Non-negative least squares without the sum-to-one
    constraint is a different number -- it may scale a single donor up to
    reach a target well outside the hull -- so this pins which of the two the
    penalty charges for.
    """
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(3, 7)) * 3.0 + 10.0
    d = per_unit_reproducibility(Y, [0, 1, 2])

    for j in range(3):
        a, b = [i for i in range(3) if i != j]
        xa, xb, xj = Y[a], Y[b], Y[j]
        diff = xa - xb
        t = float((xj - xb) @ diff / (diff @ diff))
        t = min(max(t, 0.0), 1.0)                       # the simplex clip
        expected = float(np.sum((xj - (t * xa + (1 - t) * xb)) ** 2))
        assert d[j] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_a_scaled_copy_of_a_donor_is_not_reproducible():
    """A unit twice another is reached by scaling, never by averaging.

    Non-negative least squares reproduces it exactly at zero cost; on the
    simplex it sits outside the hull and the charge stays positive. The two
    answers differ in kind, not by a tolerance.
    """
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Y = np.vstack([2.0 * base, base, 1.05 * base])
    d = per_unit_reproducibility(Y, [0, 1, 2])
    assert d[0] > 1.0, "the scaled unit must carry a positive charge"


# ------------------------------------------- rung 4: the contract not enforced
@pytest.mark.parametrize("kw", [{"lambda2_unit": 0.1}, {"zeta": 0.1}])
def test_a_term_with_no_convex_form_is_refused_with_a_reason(fit_panel, kw):
    """A wall of DCP text names the expression, not the decision that caused it.

    Neither has a convex form -- ``lambda2_unit`` pairs w
    against v through a distance matrix, ``zeta`` minimises a concave z(1-z) --
    so there is nothing to linearise, and the honest answer is to say so before
    a solver is reached.
    """
    Y, members, Xbar = fit_panel
    _, D2 = precompute_distances(Y, Xbar, members)
    w, v, z = init_cvxpy_variables(N=6, K=1, boolean=False)
    with pytest.raises(MlsynthConfigError, match="convex|non-convex|not supported"):
        build_objective(Y, Xbar, members, w, v, z, design="unit_penalized",
                        D2_list=D2, **kw)


def test_lambda2_unit_is_refused_at_config_construction(fit_panel):
    """The build-time guard is the backstop; the config is where it belongs.

    A parameter that can only ever be zero should fail when the user sets it,
    not after a panel has been ingested and a program built.
    """
    import pandas as pd
    from mlsynth.config_models import MAREXConfig

    df = pd.DataFrame({
        "unit": np.repeat(np.arange(6), 8),
        "time": np.tile(np.arange(8), 6),
        "y": np.arange(48, dtype=float),
        "post": np.tile([0] * 6 + [1] * 2, 6),
    })
    with pytest.raises(MlsynthConfigError, match="lambda2_unit must be 0"):
        MAREXConfig(df=df, unitid="unit", time="time", outcome="y",
                    post_col="post", design="unit_penalized", lambda2_unit=0.1)

    # and the same config with the supported unit-level penalties is accepted
    cfg = MAREXConfig(df=df, unitid="unit", time="time", outcome="y",
                      post_col="post", design="unit_penalized",
                      xi=2.0, lambda1_unit=0.5)
    assert cfg.xi == 2.0 and cfg.lambda2_unit == 0.0


# -------------------------------------------- rung 3: the invariant, generatively
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


@given(st.integers(0, 2 ** 31 - 1), st.floats(0.0, 20.0), st.integers(4, 9))
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_every_accepted_penalty_leaves_a_convex_program(seed, xi, n):
    """Whatever the panel and whatever xi, a design the config accepts must
    build a program a solver will take. This is the invariant nobody asserted:
    the suite covered the branch by calling build_objective and checking the
    return type, which a non-convex objective satisfies perfectly."""
    rng = np.random.default_rng(seed)
    Y = np.abs(rng.normal(5.0, 1.0, (n, 10))) + 0.5
    members = [np.arange(n)]
    w, v, z = init_cvxpy_variables(N=n, K=1, boolean=False)
    obj = build_objective(Y, [Y.mean(axis=0)], members, w, v, z,
                          design="unit_penalized", xi=xi)
    assert obj.is_dcp()


# ---------------------------------------------------- rung 0/1: end to end
def test_the_design_solves_end_to_end_with_a_positive_penalty():
    import pandas as pd
    from mlsynth import MAREX
    rng = np.random.default_rng(11)
    J, T, TP = 8, 40, 5
    size = np.sort(np.clip(np.exp(rng.normal(np.log(40), 0.6, J)), 10, 200))
    f = np.cumsum(rng.normal(0, 0.3, T))
    Y = size * (1.0 + 0.3 * (f[:, None] * rng.normal(1.0, 0.2, J)
                             + rng.normal(0, 1, (T, J)) * 0.4))
    df = pd.DataFrame([{"market": f"M{j}", "week": t, "sales": Y[t, j],
                        "post": int(t >= T - TP)}
                       for j in range(J) for t in range(T)])
    res = MAREX(dict(df=df, outcome="sales", unitid="market", time="week",
                     post_col="post", design="unit_penalized", xi=1.0,
                     m_eq=2, program_type="MIQP", verbose=False)).fit()
    assert res is not None
