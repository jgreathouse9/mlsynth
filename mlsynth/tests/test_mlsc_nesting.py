"""Nesting and recovery claims of the mlSC estimator, asserted by value.

Reference: Bottmer, L., "Synthetic Control with Disaggregated Data" (JMP,
Stanford). mlsynth implements the paper's preferred specification, Equation
5.2 -- the ``lambda_2 = infinity`` case in which the treated unit stays
aggregated and the weight matrix collapses to the vector ``omega``:

    argmin_omega  sum_{t<=T0} (Y_0t - sum_s sum_c omega_sc Y_sct)^2
                  + lambda * sigma_y^2 * sum_s sum_c (omega_sc - v_sc w_s)^2
    s.t.          sum omega = 1,  omega >= 0,  w_s = sum_c omega_sc

Section 5.1 states that this nests two named estimators as limiting cases,
and Proposition 3 states a recovery result. `test_mlsc.py` asserts the first
limit structurally and the second not at all, so the claims are pinned here
by value against estimators mlsynth already has -- differential testing in
the sense of ``agents/agents_tests.md``, which is the strongest instrument
available where no oracle exists.

The three claims:

* C1 -- as ``lambda -> infinity`` the estimator reduces to the classical SC
  (dGSC-AA) estimator computed on the aggregated panel.
* C2 -- at ``lambda = 0`` it recovers dGSC-AD, which Proposition 1 shows is
  the classical SC problem over the disaggregated donor pool, "differing only
  in the selection of units included in the donor pool".
* C3 -- Proposition 3: under Assumption 3 (perfect pre-treatment fit for
  classical SC) the mlSC and classical SC problems have identical solutions,
  at every penalty.

C3 is the one case in this file with a genuine oracle: Assumption 3 is
satisfiable by construction, so the fixture is built to satisfy it and the
expected weights are known before the estimator runs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from mlsynth import MLSC
from mlsynth.utils.inferutils import _outcome_only_simplex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _panels(
    Y_sct: np.ndarray,
    Y_st: np.ndarray,
    T0: int,
    treated: int = 0,
    pops: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Long-form aggregate and disaggregate frames from explicit arrays."""
    S, C, T = Y_sct.shape
    states = [f"state_{s}" for s in range(S)]
    agg = pd.DataFrame([
        {"state": states[s], "year": 2000 + t, "y": float(Y_st[s, t]),
         "treated": int(s == treated and t >= T0)}
        for s in range(S) for t in range(T)
    ])
    disagg = pd.DataFrame([
        {"county": f"county_{s}_{c}", "state": states[s], "year": 2000 + t,
         "y": float(Y_sct[s, c, t]),
         "treated": int(s == treated and t >= T0),
         "pop": 1.0 if pops is None else float(pops[s, c])}
        for s in range(S) for c in range(C) for t in range(T)
    ])
    return agg, disagg


def _factor_panel(seed=0, S=5, C=4, T=24, T0=18):
    """A panel whose two nesting limits give genuinely different answers.

    Deliberately not the rank-2 hierarchical factor panel `test_mlsc.py` uses.
    Under that design the aggregate donor series are close to collinear, and
    the aggregate weights come out the same at every penalty -- so an
    assertion about the large-penalty limit passes on a panel where it could
    not have failed. Independent county draws with a state-level offset keep
    the donors separated, and `_assert_limits_differ` below checks that on
    every fixture before the nesting claim is asserted.

    The aggregate series is the population-weighted mean of its counties, so
    aggregating the disaggregate panel with ``v`` reproduces it exactly and
    C1's comparator is the panel the estimator actually sees.
    """
    rng = np.random.default_rng(seed)
    Y_sct = rng.standard_normal((S, C, T)) + rng.standard_normal((S, 1, 1)) * 2.0
    pops = rng.uniform(0.5, 2.0, size=(S, C))
    v = pops / pops.sum(axis=1, keepdims=True)
    Y_st = np.einsum("sc,sct->st", v, Y_sct)
    return _panels(Y_sct, Y_st, T0, pops=pops), T0


def _config(df_agg, df_disagg, **overrides):
    cfg = dict(
        df_agg=df_agg, df_disagg=df_disagg, outcome="y", time="year",
        treat="treated", unitid_agg="state", unitid_disagg="county",
        agg_id="state", weight_col="pop", lambda_est="heuristic",
        display_graphs=False,
    )
    cfg.update(overrides)
    return cfg


def _aggregate_donor_matrix(res) -> np.ndarray:
    """Donor series at the aggregate level, in block order, shape ``(T, S)``."""
    v, idx = res.inputs.v_population, res.inputs.disagg_to_agg
    blocks = [
        res.inputs.X_disagg[:, idx == s] @ v[idx == s]
        for s in range(len(res.inputs.agg_labels))
    ]
    return np.column_stack(blocks)


def _assert_limits_differ(df_agg, df_disagg, attr: str, margin: float = 1e-2):
    """Guard: the two limits must give different answers on this fixture.

    A nesting claim asserted on a panel where every penalty returns the same
    thing is vacuous -- it passes whatever the estimator does. This ran green
    on a rank-2 factor panel whose aggregate weights were penalty-invariant,
    which is why it is now checked, not assumed.
    """
    lo = MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=0.0)).fit()
    hi = MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=1e10)).fit()
    spread = np.max(np.abs(getattr(lo.design, attr) - getattr(hi.design, attr)))
    assert spread > margin, (
        f"fixture is degenerate for {attr}: the two limits agree to {spread:.2e}, "
        "so the nesting assertion below cannot fail"
    )
    return lo, hi


# ---------------------------------------------------------------------------
# C2 -- lambda = 0 recovers the disaggregated-donor SC (dGSC-AD)
# ---------------------------------------------------------------------------

def test_lambda_zero_is_classical_sc_over_the_disaggregated_donor_pool():
    """Section 5.1 plus Proposition 1.

    With the aggregation pull switched off, Equation 5.2 is the classical SC
    program whose donor pool happens to be the disaggregated units. Agreement
    is limited by the 1e-8 uniqueness ridge, not by solver tolerance.
    """
    (df_agg, df_disagg), T0 = _factor_panel()
    res = MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=0.0)).fit()

    reference = _outcome_only_simplex(
        res.inputs.Y_agg_treated[:T0], res.inputs.X_disagg[:T0, :]
    )
    np.testing.assert_allclose(res.design.omega, reference, atol=1e-5)


def test_lambda_zero_matches_the_disaggregated_sc_counterfactual():
    """The weights are what nests; the counterfactual is what a user reads."""
    (df_agg, df_disagg), T0 = _factor_panel(seed=3)
    res = MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=0.0)).fit()

    reference = _outcome_only_simplex(
        res.inputs.Y_agg_treated[:T0], res.inputs.X_disagg[:T0, :]
    )
    np.testing.assert_allclose(
        res.paths.counterfactual, res.inputs.X_disagg @ reference, atol=1e-5
    )


# ---------------------------------------------------------------------------
# C1 -- lambda -> infinity recovers classical SC on the aggregated panel
# ---------------------------------------------------------------------------

def test_high_lambda_aggregate_weights_equal_classical_sc():
    """Section 5.1: the large-penalty limit is the dGSC-AA (classical SC)
    estimator on aggregated data.

    `test_mlsc.py` asserts only that omega/v is flat within each block, which
    would hold even if the aggregate weights themselves were wrong. This pins
    the value.
    """
    (df_agg, df_disagg), T0 = _factor_panel()
    lo, res = _assert_limits_differ(df_agg, df_disagg, "aggregate_weights")

    agg_donors = _aggregate_donor_matrix(res)
    reference = _outcome_only_simplex(res.inputs.Y_agg_treated[:T0], agg_donors[:T0, :])
    np.testing.assert_allclose(res.design.aggregate_weights, reference, atol=1e-4)
    # ... and the small-penalty end is genuinely somewhere else, so the
    # assertion above is about the limit and not about the estimator's only
    # possible answer.
    assert np.max(np.abs(lo.design.aggregate_weights - reference)) > 1e-2


def test_high_lambda_omega_is_the_classical_solution_spread_by_population():
    """The disaggregate weights in the limit are ``omega_sc = v_sc * w_s``
    with ``w`` the classical SC solution -- structure and value together."""
    (df_agg, df_disagg), T0 = _factor_panel()
    res = MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=1e10)).fit()

    agg_donors = _aggregate_donor_matrix(res)
    w = _outcome_only_simplex(res.inputs.Y_agg_treated[:T0], agg_donors[:T0, :])
    expected = res.inputs.v_population * w[res.inputs.disagg_to_agg]
    np.testing.assert_allclose(res.design.omega, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# C3 -- Proposition 3, recovery of classical SC under perfect pre-treatment fit
# ---------------------------------------------------------------------------

def _perfect_fit_panel(seed=0, S=5, C=4, T=24, T0=18):
    """A panel satisfying Assumption 3 by construction.

    The treated aggregate's pre-treatment series is an exact convex
    combination of the donor aggregates, so the classical SC problem attains
    zero pre-treatment loss and its solution is the constructed weight vector.
    With T0 = 18 periods and 4 donors the representation is generically
    unique, so the comparison is well posed.
    """
    rng = np.random.default_rng(seed)
    Y_sct = rng.standard_normal((S, C, T)) * 0.8 + rng.standard_normal((S, 1, 1)) * 2.0
    pops = rng.uniform(0.5, 2.0, size=(S, C))
    v = pops / pops.sum(axis=1, keepdims=True)
    Y_st = np.einsum("sc,sct->st", v, Y_sct)

    w = np.array([0.5, 0.2, 0.2, 0.1])          # on the simplex, S - 1 donors
    Y_st[0, :] = w @ Y_st[1:, :]                # exact at every t, pre and post
    Y_sct[0, :, :] = Y_st[0, :]                 # treated counties consistent

    panels = _panels(Y_sct, Y_st, T0, pops=pops)
    return panels, T0, w


@pytest.mark.parametrize("lambda_val", [0.0, 1e-4, 0.1, 1.0, 25.0, 1e4, 1e10])
def test_proposition_3_recovers_classical_sc_at_every_penalty(lambda_val):
    """Proposition 3: under Assumption 3 the mlSC and classical SC problems
    have identical solutions, so the penalty cannot move the answer."""
    (df_agg, df_disagg), T0, w_true = _perfect_fit_panel()
    res = MLSC(_config(df_agg, df_disagg,
                       lambda_est="fixed", lambda_val=lambda_val)).fit()

    np.testing.assert_allclose(res.design.aggregate_weights, w_true, atol=1e-4)


def test_proposition_3_att_matches_classical_sc():
    """The estimand form of Proposition 3: tau_mlSC = tau_SC."""
    (df_agg, df_disagg), T0, w_true = _perfect_fit_panel()
    atts = [
        MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=lam)).fit().att
        for lam in (0.0, 1.0, 1e10)
    ]
    assert np.allclose(atts, atts[0], atol=1e-6)


def test_proposition_3_fit_is_exact():
    """Assumption 3 holds by construction, so the pre-period fit is zero."""
    (df_agg, df_disagg), T0, _ = _perfect_fit_panel()
    res = MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=1.0)).fit()
    assert res.pre_rmse < 1e-6


def test_proposition_3_is_a_property_of_assumption_3_not_of_the_estimator():
    """The contrast that gives Proposition 3 its content.

    Penalty-invariance of the solution is what Proposition 3 claims, so the
    claim only says something if the solution is penalty-*dependent* when
    Assumption 3 fails. It is: on a panel without perfect pre-treatment fit
    the aggregate weights and the ATT both move with the penalty.
    """
    (perfect_agg, perfect_disagg), _, _ = _perfect_fit_panel()
    (generic_agg, generic_disagg), _ = _factor_panel()

    def spread(df_agg, df_disagg):
        fits = [
            MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=lam)).fit()
            for lam in (0.0, 1e10)
        ]
        return (
            np.max(np.abs(fits[0].design.aggregate_weights
                          - fits[1].design.aggregate_weights)),
            abs(fits[0].att - fits[1].att),
        )

    w_perfect, att_perfect = spread(perfect_agg, perfect_disagg)
    w_generic, att_generic = spread(generic_agg, generic_disagg)

    assert w_perfect < 1e-6 and att_perfect < 1e-6
    assert w_generic > 1e-2 and att_generic > 1e-4


# ---------------------------------------------------------------------------
# The interior of the trade-off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lambda_val", [0.05, 1.0, 20.0])
def test_interior_penalty_solves_equation_5_2_as_written(lambda_val):
    """The endpoints do not pin how ``lambda`` enters the objective.

    Halving the penalty's square-root factor only reparametrizes the penalty
    (``lambda -> lambda / 4``), which leaves both limits and the monotonicity
    of the fit untouched -- a mutation of exactly that form survives every
    other test in this file. This transcribes Equation 5.2 directly, with the
    estimator's own reported ``lambda_used`` and ``sigma_y2``, and solves it
    independently, so the penalty's scale is pinned at interior penalties
    where the estimator actually operates.

        penalty(omega) = sum_s sum_c (omega_sc - v_sc * w_s)^2 = ||(I - B) omega||^2

    with ``B[i, j] = v_i`` when columns ``i`` and ``j`` share an aggregate.
    """
    cp = pytest.importorskip("cvxpy")
    (df_agg, df_disagg), T0 = _factor_panel()
    res = MLSC(_config(df_agg, df_disagg,
                       lambda_est="fixed", lambda_val=lambda_val)).fit()

    X = res.inputs.X_disagg[:T0, :]
    y = res.inputs.Y_agg_treated[:T0]
    v, idx = res.inputs.v_population, res.inputs.disagg_to_agg
    M = X.shape[1]

    B = np.where(idx[:, None] == idx[None, :], v[:, None], 0.0)
    R = np.eye(M) - B

    omega = cp.Variable(M)
    cp.Problem(
        cp.Minimize(cp.sum_squares(X @ omega - y)
                    + res.design.lambda_used * res.design.sigma_y2
                    * cp.sum_squares(R @ omega)),
        [cp.sum(omega) == 1, omega >= 0],
    ).solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, max_iter=400_000)

    np.testing.assert_allclose(res.design.omega, omega.value, atol=1e-4)


# ---------------------------------------------------------------------------
# The trade-off between the two limits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pre_period_fit_is_monotone_in_the_penalty(seed):
    """The penalty only ever constrains the feasible set, so the attained
    pre-treatment loss cannot improve as it grows. Asserted across the grid,
    not at two points."""
    (df_agg, df_disagg), _ = _factor_panel(seed=seed)
    grid = [0.0, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6]
    rmses = [
        MLSC(_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=lam)).fit().pre_rmse
        for lam in grid
    ]
    for lo, hi in zip(rmses, rmses[1:]):
        assert hi >= lo - 1e-8, f"fit improved as the penalty grew: {rmses}"
