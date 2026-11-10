"""MAREX's exact backend on the weakly-targeted design.

The weakly-targeted objective ties the synthetic control to the synthetic
treated, ``||A - B w||^2 + beta ||B w - B v||^2``, so naming the treated set
does not split it into two independent fits the way the standard design splits.
It leaves a convex problem over a product of two simplexes. The backend solves
it by Newton steps over the faces of the control simplex, each one an exact
simplex fit, and stops on a Frank-Wolfe certificate: the value it returns is
within a stated tolerance of the joint optimum, not merely a point where
progress slowed. Given the incumbent's value as a cutoff, the fit stops as soon
as its certified lower bound clears it, and reports the candidate as discarded.

The search around that oracle is unchanged. The coupling term is
non-negative, so the treated-side fit and its closed-form bound are still lower
bounds on the objective and prune as before.

The MIQP is the reference for whole designs and a general-purpose conic solve
is the reference for the oracle. Levels: smoke, differential, unit, edge,
failure, property.
"""
from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.exact import (
    coupled_fit,
    search_cluster,
    side_fit,
    treated_fit_bound,
)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.tests.test_marex_exact_backend import (
    _assert_same_design,
    _both,
    _cfg,
    _fit,
    _no_scip,
    _panel,
    _problem,
    _treated,
)

BETAS = (0.01, 1.0, 100.0)


def _joint(B, A, S, C, beta):
    """The oracle's problem handed to a conic solver over the stacked weights."""
    Bs, Bc = B[:, list(S)], B[:, list(C)]
    w = cp.Variable(len(S), nonneg=True)
    v = cp.Variable(len(C), nonneg=True)
    obj = cp.sum_squares(A - Bs @ w) + beta * cp.sum_squares(Bs @ w - Bc @ v)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1, cp.sum(v) == 1])
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value)


def _objective(B, A, S, C, w, v, beta):
    u, c = B[:, list(S)] @ w, B[:, list(C)] @ v
    return float((A - u) @ (A - u) + beta * ((u - c) @ (u - c)))


def _brute_force(B, A, m, beta, filt=SubsetFilter()):
    """Every admissible treated set of size ``m``, the coupled oracle each. A
    set whose coupled treated weights leave a market at zero is not a design
    of its size (jgreathouse9/mlsynth#950)."""
    n = B.shape[1]
    best = (np.inf, None)
    for S in itertools.combinations(range(n), m):
        if not filt.accepts(S):
            continue
        C = [j for j in range(n) if j not in S]
        value, w = coupled_fit(B, A, list(S), C, beta)[:2]
        if np.any(w <= 1e-8):
            continue
        if value < best[0]:
            best = (value, tuple(S))
    return best


# --------------------------------------------------------------------- smoke
def test_the_exact_backend_solves_the_weakly_targeted_design():
    df, T0 = _panel()
    res = _fit(df, T0, "exact", m_eq=2, design="weakly_targeted", beta=1.0)
    assert type(res).__name__ == "MAREXResults"
    assert res.study.backend == "exact"
    assert len(_treated(res)) == 2
    assert res.globres.treated_weights_agg.sum() == pytest.approx(1.0)
    assert res.globres.control_weights_agg.sum() == pytest.approx(1.0)
    assert np.isfinite(res.globres.objective)


# -------------------------------------------------------------- differential
@pytest.mark.parametrize("beta", (1e-6,) + BETAS)
def test_exact_matches_the_mip_across_the_papers_range_of_beta(beta):
    """Abadie and Zhao's simulations sweep beta from 0.01 to 100."""
    df, T0 = _panel()
    _assert_same_design(*_both(df, T0, m_eq=2, design="weakly_targeted",
                               beta=beta))


def test_exact_matches_the_mip_over_a_range_of_treated_counts():
    df, T0 = _panel(seed=4)
    _assert_same_design(*_both(df, T0, m_min=1, m_max=3,
                               design="weakly_targeted", beta=1.0))


def test_exact_matches_the_mip_with_two_clusters():
    df, T0 = _panel(J=10, seed=5, region_sizes=(5, 5))
    _assert_same_design(*_both(df, T0, m_eq=2, cluster="region",
                               design="weakly_targeted", beta=1.0))


def test_exact_matches_the_mip_with_forced_and_forbidden_units():
    df, T0 = _panel(seed=6)
    _assert_same_design(*_both(df, T0, m_eq=2, design="weakly_targeted",
                               beta=1.0, to_be_treated=["u5"],
                               not_to_be_treated=["u0"]))


def test_the_exact_pool_matches_the_mip_pool():
    df, T0 = _panel(seed=7)
    exact, mip = _both(df, T0, m_eq=2, design="weakly_targeted", beta=1.0,
                       top_K=3)
    assert exact.pool and mip.pool
    assert ([frozenset(map(str, p["markets"])) for p in exact.pool]
            == [frozenset(map(str, p["markets"])) for p in mip.pool])
    np.testing.assert_allclose([p["mip_objective"] for p in exact.pool],
                               [p["mip_objective"] for p in mip.pool],
                               rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------- unit
@pytest.mark.parametrize("beta", (0.0, 1e-6) + BETAS)
def test_the_oracle_reaches_the_joint_optimum(beta):
    B, A = _problem(2)
    n = B.shape[1]
    for S in [(0, 3), (1, 2, 5), (4,)]:
        C = [j for j in range(n) if j not in S]
        value, w, v = coupled_fit(B, A, list(S), C, beta)
        ref = _joint(B, A, S, C, beta)
        assert value <= ref + 1e-6 * (1.0 + abs(ref))
        assert value == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_the_oracle_returns_weights_that_attain_its_value():
    B, A = _problem(3)
    S, C = [1, 4], [0, 2, 3, 5, 6, 7]
    value, w, v = coupled_fit(B, A, S, C, 1.0)
    assert w.shape == (2,) and v.shape == (6,)
    assert (w >= 0).all() and (v >= 0).all()
    assert w.sum() == pytest.approx(1.0) and v.sum() == pytest.approx(1.0)
    assert value == pytest.approx(_objective(B, A, S, C, w, v, 1.0), rel=1e-12)


def test_the_oracle_stops_on_its_certificate_not_on_slow_progress():
    """At large beta the blocks chase each other and each sweep gains little.

    A rule that stopped on a small decrease would stop there, above the optimum;
    the certificate keeps going. The tolerance is tightened so a stop on the
    first small step would miss by more than it allows.
    """
    B, A = _problem(5, n=9)
    S, C = [0, 1], list(range(2, 9))
    value = coupled_fit(B, A, S, C, 100.0, tol=1e-12)[0]
    ref = _joint(B, A, tuple(S), C, 100.0)
    assert value == pytest.approx(ref, rel=1e-7)


def test_the_treated_fit_bounds_the_coupled_objective():
    """What lets the existing search prune: lb(S) <= g(S) <= f(S)."""
    B, A = _problem(4)
    n = B.shape[1]
    subsets = np.array(list(itertools.combinations(range(n), 2)))
    lb = treated_fit_bound(B, A, subsets)
    for S, bound in zip(subsets, lb):
        C = [j for j in range(n) if j not in S]
        g = side_fit(B, A, list(S))[0]
        f = coupled_fit(B, A, list(S), C, 10.0)[0]
        assert bound <= g * (1 + 1e-9) + 1e-12
        assert g <= f * (1 + 1e-9) + 1e-12


@pytest.mark.parametrize("beta", BETAS)
def test_the_search_matches_plain_enumeration(beta):
    B, A = _problem(0)
    designs = search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=beta)
    value, S = _brute_force(B, A, 3, beta)
    assert designs[0].treated == S
    assert designs[0].value == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_the_search_reports_the_coupled_weights():
    """The design carries the weights that attain its value, not the
    treated-only fit the search started from."""
    B, A = _problem(1)
    d = search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=100.0)[0]
    assert d.value == pytest.approx(
        _objective(B, A, d.treated, d.control, d.w, d.v, 100.0), rel=1e-10)
    g, w_alone = side_fit(B, A, list(d.treated))
    assert d.value > g * (1 + 1e-6)          # the coupling is active
    assert not np.allclose(d.w, w_alone, atol=1e-4)


# ---------------------------------------------------------------------- edge
def test_a_single_control_market():
    B, A = _problem(6, n=4)
    value, S = _brute_force(B, A, 3, 1.0)
    if S is None:                       # every triple leaves a market unweighted
        with pytest.raises(MlsynthEstimationError, match="weight"):
            search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=1.0)
        return
    d = search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=1.0)[0]
    assert len(d.control) == 1
    assert d.v == pytest.approx([1.0])
    assert d.treated == S and d.value == pytest.approx(value, rel=1e-9)


def test_beta_zero_is_the_treated_fit():
    B, A = _problem(7)
    S, C = [2, 5], [0, 1, 3, 4, 6, 7]
    assert coupled_fit(B, A, S, C, 0.0)[0] == pytest.approx(
        side_fit(B, A, S)[0], rel=1e-12, abs=1e-12)


def test_a_treated_synthetic_inside_the_control_hull_costs_no_coupling():
    """With the treated market duplicated in the control pool the coupling
    term can reach zero, so the objective is the treated fit alone."""
    B, A = _problem(8, n=6)
    B = np.column_stack([B, B[:, 0]])
    value = coupled_fit(B, A, [0], [1, 2, 3, 4, 5, 6], 100.0)[0]
    assert value == pytest.approx(side_fit(B, A, [0])[0], rel=1e-9)


def test_duplicate_markets_reach_the_same_objective_as_the_mip():
    df, T0 = _panel(seed=9)
    dup = df[df["unit"] == "u1"].assign(unit="u1_copy")
    df2 = __import__("pandas").concat([df, dup], ignore_index=True)
    exact = _fit(df2, T0, "exact", m_eq=2, design="weakly_targeted", beta=1.0)
    mip = _fit(df2, T0, "mip", m_eq=2, design="weakly_targeted", beta=1.0)
    assert exact.globres.objective == pytest.approx(
        mip.globres.objective, rel=1e-5, abs=1e-8)


# ------------------------------------------------------------------- failure
def test_an_uncertified_oracle_raises_instead_of_returning_its_value(monkeypatch):
    """Running out of sweeps leaves a value above the optimum by an unknown
    amount; ranking candidates on it could discard the optimal design."""
    B, A = _problem(5, n=9)
    with pytest.raises(MlsynthEstimationError, match="certif"):
        coupled_fit(B, A, [0, 1], list(range(2, 9)), 100.0, max_iter=0)


def test_an_uncertified_design_is_reported_through_the_fit(monkeypatch):
    monkeypatch.setattr(exact_mod, "COUPLED_MAX_ITER", 0)
    df, T0 = _panel()
    with pytest.raises(MlsynthEstimationError, match="backend='mip'") as info:
        _fit(df, T0, "exact", m_eq=2, design="weakly_targeted", beta=100.0)
    # A fit that could not be certified is not an infeasible design, and the
    # restrictions-and-cardinality advice that comes with one would mislead.
    assert "certified" in str(info.value)
    assert "infeasible" not in str(info.value)


def test_a_negative_beta_is_refused_by_the_exact_backend():
    """beta < 0 makes the objective concave in the coupling and the
    alternation has no optimum to certify."""
    df, T0 = _panel()
    with pytest.raises(MlsynthConfigError, match="beta"):
        MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, design="weakly_targeted",
                           beta=-0.5))


def test_a_direct_call_refuses_a_negative_beta():
    df, T0 = _panel()
    Y = df.pivot(index="unit", columns="time", values="y").to_numpy()
    with pytest.raises(MlsynthConfigError, match="beta"):
        exact_mod.solve_design_exact(Y, T0, np.zeros(9, dtype=int), m_eq=2,
                                     design="weakly_targeted", beta=-0.5)


def test_a_direct_search_refuses_a_negative_beta():
    B, A = _problem(0)
    with pytest.raises(MlsynthConfigError, match="beta must be >= 0"):
        search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=-1.0)


def test_a_search_given_beta_and_a_penalty_is_refused():
    """The weakly-targeted design has no linear penalty; a caller combining
    them is asking for an objective MAREX does not state."""
    B, A = _problem(0)
    with pytest.raises(MlsynthConfigError, match="beta"):
        search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=1.0,
                       linear_treated=np.ones(B.shape[1]))


@pytest.mark.parametrize("over, fragment", [
    ({"relaxed": True}, "relaxed"),
    ({"costs": [1.0] * 9, "budget": 3}, "budget"),
    ({"time_limit": 5.0}, "time_limit"),
])
def test_what_the_exact_backend_cannot_solve_is_still_refused(over, fragment):
    df, T0 = _panel()
    with pytest.raises(MlsynthConfigError, match=fragment):
        MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, design="weakly_targeted",
                           beta=1.0, **over))


def test_the_weakly_targeted_search_never_reaches_scip(monkeypatch):
    df, T0 = _panel()
    _no_scip(monkeypatch)
    res = _fit(df, T0, "exact", m_eq=2, design="weakly_targeted", beta=1.0)
    assert res.study.backend == "exact"


# ------------------------------------------- the Newton fit and the cutoff
def _fallback_problem(seed=12):
    """A problem on which the Newton step cannot improve and a sweep must
    close the gap: the fit's start is already all but optimal at beta = 1e-6."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 10))
    B = rng.normal(size=(8, n)) + rng.normal(size=(8, 1)) * 2.0
    m = int(rng.integers(1, n))
    return B, B.mean(axis=1), list(range(m)), list(range(m, n))


@pytest.mark.parametrize("beta", (1e-6, 0.01, 1.0, 100.0))
def test_the_fit_certifies_in_a_few_steps_at_every_beta(beta):
    """The alternation this replaced converges at rate beta / (1 + beta), a
    median of 1,750 sweeps per candidate at beta = 100 on GeoLift; each Newton
    step solves the current face exactly, so the count does not grow with beta."""
    B, A = _problem(5, n=9)
    S, C = [0, 1], list(range(2, 9))
    value, w, v, info = coupled_fit(B, A, S, C, beta, return_info=True)
    assert info["iterations"] <= 5
    assert value == pytest.approx(_joint(B, A, tuple(S), C, beta), rel=1e-7)


def test_the_fallback_sweep_certifies_where_a_newton_step_cannot_improve():
    B, A, S, C = _fallback_problem()
    value, w, v, info = coupled_fit(B, A, S, C, 1e-6, return_info=True)
    assert info["sweeps"] >= 1
    assert value == pytest.approx(_joint(B, A, tuple(S), C, 1e-6), rel=1e-7)


def test_a_cutoff_above_the_optimum_never_discards():
    B, A = _problem(6, n=8)
    S, C = [0, 3], [1, 2, 4, 5, 6, 7]
    fstar = coupled_fit(B, A, S, C, 100.0)[0]
    fit = coupled_fit(B, A, S, C, 100.0, cutoff=fstar * (1 + 1e-6) + 1e-12)
    assert fit is not None
    assert fit[0] == pytest.approx(fstar, rel=1e-12)


def test_a_cutoff_below_the_optimum_discards():
    B, A = _problem(6, n=8)
    S, C = [0, 3], [1, 2, 4, 5, 6, 7]
    fstar = coupled_fit(B, A, S, C, 100.0)[0]
    assert coupled_fit(B, A, S, C, 100.0, cutoff=fstar * (1 - 1e-3)) is None


def test_the_search_hands_the_fit_its_incumbent(monkeypatch):
    """The cutoff is what discards losing candidates early; a search that never
    passed it would return the same designs at the old cost."""
    cutoffs = []
    real = exact_mod.coupled_fit

    def spy(*args, **kwargs):
        cutoffs.append(kwargs.get("cutoff"))
        return real(*args, **kwargs)

    monkeypatch.setattr(exact_mod, "coupled_fit", spy)
    B, A = _problem(7, n=10)
    search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=100.0)
    finite = [c for c in cutoffs if c is not None and np.isfinite(c)]
    assert len(cutoffs) > 1 and len(finite) == len(cutoffs) - 1


def test_a_large_beta_search_matches_plain_enumeration():
    B, A = _problem(8, n=12)
    designs = search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=100.0)
    value, S = _brute_force(B, A, 3, 100.0)
    assert designs[0].treated == S
    assert designs[0].value == pytest.approx(value, rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------ property
@given(seed=st.integers(0, 10_000), n=st.integers(3, 7), m=st.integers(1, 3),
       beta=st.sampled_from([0.0, 1e-6, 0.01, 1.0, 100.0]))
@settings(max_examples=40, deadline=None)
def test_the_search_is_exact_over_the_domain(seed, n, m, beta):
    if m >= n:
        m = n - 1
    B, A = _problem(seed, n=n, T=8)
    value, S = _brute_force(B, A, m, beta)
    if S is None:                        # no set weights all its markets (#950)
        with pytest.raises(MlsynthEstimationError, match="weight"):
            search_cluster(B, A, sizes=[m], filt=SubsetFilter(), beta=beta)
        return
    designs = search_cluster(B, A, sizes=[m], filt=SubsetFilter(), beta=beta)
    assert designs[0].value == pytest.approx(value, rel=1e-8, abs=1e-10)


@given(seed=st.integers(0, 10_000), n=st.integers(3, 8),
       betas=st.lists(st.sampled_from([0.0, 1e-3, 0.1, 1.0, 10.0, 100.0]),
                      min_size=2, max_size=4, unique=True))
@settings(max_examples=40, deadline=None)
def test_the_coupled_objective_is_bounded_below_and_monotone_in_beta(
        seed, n, betas):
    """f(S; beta) >= g(S) for every beta, and f is non-decreasing in beta:
    each beta multiplies a non-negative term, so the minimum can only rise."""
    B, A = _problem(seed, n=n, T=8)
    S, C = [0], list(range(1, n))
    g = side_fit(B, A, S)[0]
    values = [coupled_fit(B, A, S, C, b)[0] for b in sorted(betas)]
    for f in values:
        assert f >= g * (1 - 1e-9) - 1e-10
    for lo, hi in zip(values, values[1:]):
        assert hi >= lo * (1 - 1e-8) - 1e-10


@given(seed=st.integers(0, 10_000), n=st.integers(3, 7),
       beta=st.sampled_from([0.01, 1.0, 100.0]))
@settings(max_examples=25, deadline=None)
def test_the_oracle_matches_a_conic_solve_over_the_domain(seed, n, beta):
    B, A = _problem(seed, n=n, T=8)
    S, C = [0, 1] if n > 2 else [0], list(range(2, n)) if n > 2 else [1]
    value = coupled_fit(B, A, S, C, beta)[0]
    ref = _joint(B, A, tuple(S), C, beta)
    assert value <= ref + 1e-6 * (1.0 + abs(ref))
    assert value == pytest.approx(ref, rel=1e-5, abs=1e-8)


@given(seed=st.integers(0, 10_000), n=st.integers(3, 9),
       beta=st.sampled_from([0.0, 1e-6, 0.01, 1.0, 100.0]))
@settings(max_examples=60, deadline=None)
def test_the_fit_needs_few_steps_over_the_domain(seed, n, beta):
    B, A = _problem(seed, n=n, T=8)
    m = 1 + seed % (n - 1)
    info = coupled_fit(B, A, list(range(m)), list(range(m, n)), beta,
                       return_info=True)[3]
    assert info["iterations"] <= 10


@given(seed=st.integers(0, 10_000), n=st.integers(3, 8),
       beta=st.sampled_from([0.01, 1.0, 100.0]),
       ratio=st.floats(0.5, 1.5))
@settings(max_examples=60, deadline=None)
def test_a_discarded_fit_could_not_have_beaten_its_cutoff(seed, n, beta, ratio):
    """None is a certificate: the joint optimum is at least the cutoff. A
    returned fit is the optimum itself."""
    B, A = _problem(seed, n=n, T=8)
    S, C = [0], list(range(1, n))
    fstar = coupled_fit(B, A, S, C, beta)[0]
    cutoff = ratio * fstar
    fit = coupled_fit(B, A, S, C, beta, cutoff=cutoff)
    if fit is None:
        assert fstar >= cutoff * (1 - 1e-9) - 1e-12
    else:
        assert fit[0] == pytest.approx(fstar, rel=1e-9, abs=1e-12)
