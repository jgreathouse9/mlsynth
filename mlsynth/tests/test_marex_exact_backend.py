"""MAREX's exact backend: the design as a bounded search over treated sets.

Naming the treated set splits MAREX's standard, penalized and unit-penalized
objectives into two independent simplex fits -- the treated side and the
control side share nothing once the set is fixed -- so the design is a search
over treated sets with an exact oracle, not a mixed-integer program. The
backend prices every candidate by a lower bound on its treated-side fit,
solves exactly only the candidates the bound cannot rule out, and stops with
the optimum.

The MIQP is the reference. Every configuration the backend accepts is solved
both ways here and the designs compared; every configuration it refuses is
checked to raise before anything is solved. The comparison is independent only
where the two paths are: since #756 the MIQP reports the design its selection
defines by refitting it with this backend's own ``_penalties`` and ``side_fit``,
so a defect in those moves both backends together and survives here (#783).
``test_marex_exact_oracle.py`` checks them against an oracle that shares no
code with the library. Below those sit unit tests of the
search core against plain enumeration and of the bound against the exact fit,
with ``hypothesis`` properties for both.

Levels: smoke, differential, unit, edge, failure, property.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.exact import (
    search_cluster,
    side_fit,
    treated_fit_bound,
)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

ATOL_W = 1e-2


# ------------------------------------------------------------------ fixtures
def _panel(J=9, T=14, T0=10, seed=0, n_good=4, hi=3.0, region_sizes=None,
           covariate=False):
    """A factor panel on which the design has a unique, concentrated optimum.

    ``n_good`` markets track the mean loading with almost no idiosyncratic
    noise and the rest are noisy, so the optimizer has a reason to prefer a
    few markets and the optimum is well separated from the runner-up.
    """
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    load = rng.uniform(0.4, 1.6, size=(2, J))
    load[:, :n_good] = load.mean(axis=1, keepdims=True)
    scale = np.full(J, hi)
    scale[:n_good] = 0.05
    Y = f @ load + rng.normal(size=(T, J)) * scale + 20.0
    labels = [f"u{j}" for j in range(J)]
    df = pd.DataFrame({"unit": np.repeat(labels, T),
                       "time": np.tile(np.arange(T), J),
                       "y": Y.T.reshape(-1)})
    if region_sizes is not None:
        region = np.repeat(np.arange(len(region_sizes)), region_sizes)
        df["region"] = np.repeat(region, T)
    if covariate:
        df["x"] = np.repeat(rng.normal(size=J), T)
    return df, T0


def _cfg(df, T0, backend, **over):
    base = dict(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                display_graph=False, inference=False, backend=backend)
    base.update(over)
    return base


def _fit(df, T0, backend, **over):
    return MAREX(MAREXConfig(**_cfg(df, T0, backend, **over))).fit()


def _treated(res):
    return frozenset(map(str, res.assignment["treated"]))


def _assert_same_design(exact, mip):
    """Same treated markets, and an optimum no worse than SCIP's.

    The weights are compared loosely on purpose. Both backends fit the weights
    of the selected set, but where the objective is flat in some direction the
    optimal weights are not unique and two correct fits can differ. What the
    two backends must agree on is the optimum value, and the search may only
    ever be lower.
    """
    oe, om = exact.globres.objective, mip.globres.objective
    assert oe <= om + 1e-6 * (1.0 + abs(om))
    assert oe == pytest.approx(om, rel=1e-5, abs=1e-8)
    assert _treated(exact) == _treated(mip)
    np.testing.assert_allclose(exact.globres.treated_weights_agg,
                               mip.globres.treated_weights_agg, atol=ATOL_W)
    np.testing.assert_allclose(exact.globres.control_weights_agg,
                               mip.globres.control_weights_agg, atol=ATOL_W)


def _both(df, T0, **over):
    exact = _fit(df, T0, "exact", **over)
    mip_over = dict(over)
    mip = _fit(df, T0, "mip", **mip_over)
    return exact, mip


# --------------------------------------------------------------------- smoke
def test_the_exact_backend_returns_a_marex_result():
    df, T0 = _panel()
    res = _fit(df, T0, "exact", m_eq=2)
    assert type(res).__name__ == "MAREXResults"
    assert len(_treated(res)) == 2
    assert res.globres.control_weights_agg.sum() == pytest.approx(1.0)
    assert res.globres.treated_weights_agg.sum() == pytest.approx(1.0)


def test_the_backend_that_ran_is_recorded():
    df, T0 = _panel()
    assert _fit(df, T0, "exact", m_eq=2).study.backend == "exact"
    assert _fit(df, T0, "mip", m_eq=2).study.backend == "mip"


def test_mip_stays_the_default():
    df, T0 = _panel()
    cfg = MAREXConfig(**{k: v for k, v in _cfg(df, T0, "mip", m_eq=2).items()
                         if k != "backend"})
    assert cfg.backend == "mip"


# -------------------------------------------- differential: exact == MIQP
@pytest.mark.parametrize("design, extra", [
    ("standard", {}),
    ("penalized", {"lambda1": 0.05, "lambda2": 0.02}),
    ("unit_penalized", {"xi": 0.5, "lambda1_unit": 0.05}),
])
def test_exact_matches_the_mip_for_each_separable_design(design, extra):
    df, T0 = _panel(seed=1)
    exact, mip = _both(df, T0, m_eq=3, design=design, **extra)
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_over_a_range_of_treated_counts():
    df, T0 = _panel(seed=2)
    exact, mip = _both(df, T0, m_min=1, m_max=3)
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_with_two_clusters():
    df, T0 = _panel(J=12, seed=3, region_sizes=(6, 6))
    exact, mip = _both(df, T0, m_eq=2, cluster="region")
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_with_a_covariate():
    df, T0 = _panel(seed=4, covariate=True)
    exact, mip = _both(df, T0, m_eq=2, covariates=["x"])
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_with_blank_periods():
    df, T0 = _panel(seed=5)
    exact, mip = _both(df, T0, m_eq=2, blank_periods=3)
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_with_forced_and_forbidden_units():
    df, T0 = _panel(seed=6)
    exact, mip = _both(df, T0, m_eq=3, to_be_treated=["u5"],
                       not_to_be_treated=["u0", "u1"])
    assert "u5" in _treated(exact)
    assert _treated(exact).isdisjoint({"u0", "u1"})
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_under_a_stratum_quota():
    df, T0 = _panel(seed=7)
    df["tier"] = df["unit"].map(lambda u: "a" if int(u[1:]) < 5 else "b")
    exact, mip = _both(df, T0, m_eq=3, stratum_col="tier",
                       min_per_stratum=1, max_per_stratum=2)
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_under_border_conflicts():
    df, T0 = _panel(seed=8)
    labels = [f"u{j}" for j in range(9)]
    A = pd.DataFrame(0.0, index=labels, columns=labels)
    for a, b in [("u0", "u1"), ("u2", "u3"), ("u1", "u2")]:
        A.loc[a, b] = A.loc[b, a] = 1.0
    exact, mip = _both(df, T0, m_eq=3, adjacency=A, spillover_threshold=0.5)
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_when_bordering_donors_are_excluded():
    """A treated market's neighbours leave its control pool, per candidate."""
    df, T0 = _panel(seed=9)
    labels = [f"u{j}" for j in range(9)]
    A = pd.DataFrame(0.0, index=labels, columns=labels)
    for a, b in [("u0", "u4"), ("u1", "u5"), ("u2", "u6")]:
        A.loc[a, b] = A.loc[b, a] = 1.0
    exact, mip = _both(df, T0, m_eq=2, adjacency=A, spillover_threshold=0.5,
                       exclude_bordering_donors=True)
    _assert_same_design(exact, mip)


def test_the_exact_pool_matches_the_mip_pool():
    """``top_K`` designs, in order, with the same raw objective."""
    df, T0 = _panel(seed=10)
    exact, mip = _both(df, T0, m_eq=2, top_K=3)
    assert exact.pool and mip.pool
    assert [frozenset(map(str, p["markets"])) for p in exact.pool] == \
           [frozenset(map(str, p["markets"])) for p in mip.pool]
    np.testing.assert_allclose([p["mip_objective"] for p in exact.pool],
                               [p["mip_objective"] for p in mip.pool],
                               rtol=1e-4, atol=1e-6)


# ------------------------------------------------- unit: the search core
def _brute_force(B, A, m, filt, lin_t=None, lin_c=None, donor_exclusion=()):
    """Every admissible treated set of size ``m``, both fits each. A set whose
    treated fit leaves a market unweighted is not a design of its size."""
    n = B.shape[1]
    best = (np.inf, None)
    for S in itertools.combinations(range(n), m):
        if not filt.accepts(S):
            continue
        excluded = {j for i, j in donor_exclusion if i in S}
        C = [j for j in range(n) if j not in S and j not in excluded]
        if not C:
            continue
        g, w = side_fit(B, A, list(S), None if lin_t is None else lin_t[list(S)])
        if np.any(w <= 1e-8):
            continue                 # not a design of its size (jgreathouse9/mlsynth#950)
        h = side_fit(B, A, C, None if lin_c is None else lin_c[C])[0]
        if g + h < best[0]:
            best = (g + h, tuple(S))
    return best


def _problem(seed, n=8, T=10):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, n)) + rng.normal(size=(T, 1)) * 2.0
    A = B.mean(axis=1)
    return B, A


def test_the_search_matches_plain_enumeration():
    B, A = _problem(0)
    filt = SubsetFilter()
    designs = search_cluster(B, A, sizes=[3], filt=filt)
    value, S = _brute_force(B, A, 3, filt)
    assert designs[0].treated == S
    assert designs[0].value == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_the_search_returns_the_k_best_in_order():
    B, A = _problem(1)
    filt = SubsetFilter()
    designs = search_cluster(B, A, sizes=[2], filt=filt, top_k=4)
    n = B.shape[1]
    values = sorted(
        side_fit(B, A, list(S))[0]
        + side_fit(B, A, [j for j in range(n) if j not in S])[0]
        for S in itertools.combinations(range(n), 2))
    np.testing.assert_allclose([d.value for d in designs], values[:4],
                               rtol=1e-9, atol=1e-12)
    assert len({d.treated for d in designs}) == 4


def test_the_bound_never_exceeds_the_exact_treated_fit():
    B, A = _problem(2)
    subs = np.array(list(itertools.combinations(range(B.shape[1]), 3)))
    lin = np.abs(np.random.default_rng(2).normal(size=B.shape[1]))
    for linear in (None, lin):
        lb = treated_fit_bound(B, A, subs, linear=linear)
        exact = np.array([side_fit(B, A, list(S),
                                   None if linear is None else linear[list(S)])[0]
                          for S in subs])
        assert np.all(lb <= exact + 1e-9 * (1 + exact))


def test_the_bound_is_exact_when_the_unconstrained_weights_are_nonnegative():
    """Dropping only ``w >= 0`` costs nothing where the minimiser is interior."""
    B, A = _problem(3)
    subs = np.array(list(itertools.combinations(range(B.shape[1]), 2)))
    lb = treated_fit_bound(B, A, subs)
    exact = np.array([side_fit(B, A, list(S))[0] for S in subs])
    settled = np.isclose(lb, exact, rtol=1e-8, atol=1e-10)
    assert settled.any()


# ----------------------------------------------------------- edge cases
def test_a_single_control_market():
    df, T0 = _panel(J=6, seed=11)
    exact, mip = _both(df, T0, m_eq=5)
    _assert_same_design(exact, mip)


def test_forced_units_that_fill_the_design_leave_one_candidate():
    df, T0 = _panel(seed=12)
    res = _fit(df, T0, "exact", m_eq=2, to_be_treated=["u3", "u7"])
    assert _treated(res) == {"u3", "u7"}


def test_forbidding_all_but_m_units_leaves_one_candidate():
    df, T0 = _panel(J=6, seed=13)
    res = _fit(df, T0, "exact", m_eq=2,
               not_to_be_treated=["u0", "u1", "u2", "u3"])
    assert _treated(res) == {"u4", "u5"}


def test_duplicate_markets_reach_the_same_objective_as_the_mip():
    """Ties: the two backends may pick different twins, not different values."""
    df, T0 = _panel(J=8, seed=14)
    twin = df[df["unit"] == "u0"].copy()
    twin["unit"] = "u8"
    df = pd.concat([df, twin], ignore_index=True)
    exact = _fit(df, T0, "exact", m_eq=2)
    mip = _fit(df, T0, "mip", m_eq=2)
    assert exact.globres.objective == pytest.approx(
        mip.globres.objective, rel=1e-5, abs=1e-8)


# -------------------------------------------------------------- failures
@pytest.mark.parametrize("over, fragment", [
    ({"relaxed": True}, "relaxed"),
    ({"costs": [1.0] * 9, "budget": 3, "design": "weakly_targeted", "beta": 1.0},
     "weakly-targeted"),
    ({"costs": [1.0] * 9, "budget": 3, "certify": True}, "certify"),
    ({"warm_start": ["u0"]}, "warm_start"),
    ({"time_limit": 5.0}, "time_limit"),
    ({"solver": "SCIP"}, "solver"),
])
def test_configurations_the_exact_backend_cannot_solve_are_refused(over, fragment):
    df, T0 = _panel()
    with pytest.raises(MlsynthConfigError, match=fragment):
        MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, **over))


def test_the_same_options_stay_valid_for_the_mip():
    df, T0 = _panel()
    MAREXConfig(**_cfg(df, T0, "mip", m_eq=2, relaxed=True))
    MAREXConfig(**_cfg(df, T0, "mip", m_eq=2, design="weakly_targeted",
                       beta=1.0))


def test_a_conflict_across_clusters_is_refused():
    """Two clusters searched apart cannot honour a rule that spans them."""
    df, T0 = _panel(J=12, seed=15, region_sizes=(6, 6))
    labels = [f"u{j}" for j in range(12)]
    A = pd.DataFrame(0.0, index=labels, columns=labels)
    A.loc["u0", "u6"] = A.loc["u6", "u0"] = 1.0
    with pytest.raises(MlsynthConfigError, match="span"):
        _fit(df, T0, "exact", m_eq=2, cluster="region", adjacency=A,
             spillover_threshold=0.5)


def test_too_many_candidates_is_refused_with_the_way_out(monkeypatch):
    # Above the limit the separable designs go to the branch and bound
    # (test_marex_branch.py); the coupled design has no bound there.
    monkeypatch.setattr(exact_mod, "CANDIDATE_LIMIT", 10)
    df, T0 = _panel()
    with pytest.raises(MlsynthConfigError, match="backend='mip'"):
        _fit(df, T0, "exact", m_eq=3, design="weakly_targeted", beta=1.0)


def test_unsatisfiable_restrictions_raise_like_the_mip():
    df, T0 = _panel()
    over = dict(m_eq=1, to_be_treated=["u0", "u1"])
    with pytest.raises(MlsynthEstimationError):
        _fit(df, T0, "mip", **over)
    with pytest.raises(MlsynthEstimationError):
        _fit(df, T0, "exact", **over)


# ------------------------------------------------------------ properties
@given(seed=st.integers(0, 10_000), n=st.integers(3, 8),
       m=st.integers(1, 4), linear=st.booleans(),
       donor=st.booleans(), conflict=st.booleans())
@settings(max_examples=60, deadline=None)
def test_the_search_is_exact_over_the_domain(seed, n, m, linear, donor, conflict):
    m = min(m, n - 1)
    B, A = _problem(seed, n=n, T=7)
    rng = np.random.default_rng(seed + 1)
    lin_t = np.abs(rng.normal(size=n)) if linear else None
    lin_c = np.abs(rng.normal(size=n)) if linear else None
    pairs = [(0, 1)] if conflict and n > 2 else []
    filt = SubsetFilter(conflict_pairs=tuple(pairs))
    donors = ((0, n - 1), (1, n - 2)) if donor and n > 3 else ()
    want = _brute_force(B, A, m, filt, lin_t, lin_c, donors)
    if want[1] is None:
        with pytest.raises(MlsynthEstimationError):
            search_cluster(B, A, sizes=[m], filt=filt, linear_treated=lin_t,
                           linear_control=lin_c, donor_exclusion=donors)
        return
    got = search_cluster(B, A, sizes=[m], filt=filt, linear_treated=lin_t,
                         linear_control=lin_c, donor_exclusion=donors)[0]
    assert got.value == pytest.approx(want[0], rel=1e-9, abs=1e-12)


@given(seed=st.integers(0, 10_000), n=st.integers(2, 9), m=st.integers(1, 5),
       linear=st.booleans())
@settings(max_examples=80, deadline=None)
def test_the_bound_is_a_lower_bound_over_the_domain(seed, n, m, linear):
    m = min(m, n)
    B, A = _problem(seed, n=n, T=6)
    subs = np.array(list(itertools.combinations(range(n), m)))
    lin = (np.abs(np.random.default_rng(seed).normal(size=n))
           if linear else None)
    lb = treated_fit_bound(B, A, subs, linear=lin)
    exact = np.array([side_fit(B, A, list(S),
                               None if lin is None else lin[list(S)])[0]
                      for S in subs])
    assert np.all(lb <= exact + 1e-8 * (1 + np.abs(exact)))


@given(seed=st.integers(0, 10_000), J=st.integers(5, 8), m=st.integers(1, 3))
@settings(max_examples=12, deadline=None)
def test_exact_matches_the_mip_over_random_panels(seed, J, m):
    df, T0 = _panel(J=J, seed=seed, n_good=2)
    exact = _fit(df, T0, "exact", m_eq=m)
    mip = _fit(df, T0, "mip", m_eq=m)
    oe, om = exact.globres.objective, mip.globres.objective
    assert oe <= om + 1e-6 * (1.0 + abs(om))
    assert oe == pytest.approx(om, rel=1e-5, abs=1e-8)


# --------------------------------------------- paths default sizes never reach
@pytest.mark.parametrize("seed, top_k", [(20, 1), (21, 3), (22, 1), (23, 5)])
def test_the_second_pass_finds_what_the_buffer_missed(monkeypatch, seed, top_k):
    """With a two-candidate buffer the first pass cannot certify the result.

    The second pass is what makes the search exact whenever more candidates
    sit under the incumbent than the buffer held, and at the default buffer
    of 512 no test panel is large enough to need it, so it is forced here.
    """
    monkeypatch.setattr(exact_mod, "BUFFER", 2)
    B, A = _problem(seed)
    n = B.shape[1]
    filt = SubsetFilter()
    designs = search_cluster(B, A, sizes=[3], filt=filt, top_k=top_k)
    values = sorted(
        side_fit(B, A, list(S))[0]
        + side_fit(B, A, [j for j in range(n) if j not in S])[0]
        for S in itertools.combinations(range(n), 3))
    np.testing.assert_allclose([d.value for d in designs], values[:top_k],
                               rtol=1e-9, atol=1e-12)


def test_a_restricted_walk_streams_in_chunks(monkeypatch):
    monkeypatch.setattr(exact_mod, "CHUNK", 7)
    B, A = _problem(24)
    filt = SubsetFilter(conflict_pairs=((0, 1), (2, 3)))
    got = search_cluster(B, A, sizes=[3], filt=filt)[0]
    want = _brute_force(B, A, 3, filt)
    assert got.treated == want[1]
    assert got.value == pytest.approx(want[0], rel=1e-9, abs=1e-12)


def test_every_candidate_leaving_no_control_is_reported():
    """Donor rules can empty every pool; that is no design, not a crash."""
    B, A = _problem(25, n=3)
    donors = tuple((i, j) for i in range(3) for j in range(3) if i != j)
    with pytest.raises(MlsynthEstimationError, match="no control market"):
        search_cluster(B, A, sizes=[1], filt=SubsetFilter(),
                       donor_exclusion=donors)


def test_a_stratum_quota_across_clusters_is_refused():
    df, T0 = _panel(J=12, seed=26, region_sizes=(6, 6))
    df["tier"] = df["unit"].map(lambda u: "a" if int(u[1:]) % 2 else "b")
    with pytest.raises(MlsynthConfigError, match="span"):
        _fit(df, T0, "exact", m_eq=2, cluster="region", stratum_col="tier",
             min_per_stratum=1, max_per_stratum=2)


def test_lambda2_unit_is_refused():
    """The config refuses it for every backend, before the search could."""
    df, T0 = _panel()
    with pytest.raises(MlsynthConfigError, match="lambda2_unit"):
        _fit(df, T0, "exact", m_eq=2, design="unit_penalized",
             lambda1_unit=0.05, lambda2_unit=0.1)


@pytest.mark.parametrize("over, fragment", [
    ({"design": "no_such_design"}, "no_such_design"),
    ({"costs": np.ones(9), "budget": 3, "design": "weakly_targeted"}, "weakly-targeted"),
    ({"costs": np.ones(9), "budget": 3, "certify": True}, "certif"),
    ({"design": "unit_penalized", "lambda2_unit": 0.1}, "lambda2_unit"),
])
def test_a_direct_call_refuses_what_the_config_refuses(over, fragment):
    """``solve_design_exact`` is importable, so it checks for itself."""
    df, T0 = _panel()
    Y = df.pivot(index="unit", columns="time", values="y").to_numpy()
    with pytest.raises(MlsynthConfigError, match=fragment):
        exact_mod.solve_design_exact(Y, T0, np.zeros(9, dtype=int), m_eq=2,
                                     **over)


def test_a_treated_count_that_leaves_no_control_raises_like_the_mip():
    df, T0 = _panel(J=5, seed=27)
    with pytest.raises(MlsynthEstimationError):
        _fit(df, T0, "mip", m_eq=5)
    with pytest.raises(MlsynthEstimationError):
        _fit(df, T0, "exact", m_eq=5)


def _no_scip(monkeypatch):
    """Make any SCIP solve through cvxpy raise, so a fit proves it ran without one."""
    import cvxpy as cp

    original = cp.Problem.solve

    def guarded(self, *args, **kwargs):
        solver = kwargs.get("solver", args[0] if args else None)
        if solver is cp.SCIP or str(solver).upper() == "SCIP":
            raise AssertionError("the exact backend reached SCIP")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", guarded)


def test_the_exact_backend_never_reaches_scip(monkeypatch):
    """The backend recorded has to be the backend that ran.

    SCIP finds the same optimum, so a fit routed back to the MIQP would pass
    every comparison in this file while ``study.backend`` still read "exact".
    With SCIP made unreachable, an exact fit can only succeed by not calling it.
    """
    df, T0 = _panel()
    _no_scip(monkeypatch)
    res = _fit(df, T0, "exact", m_eq=2)
    assert res.study.backend == "exact"


def test_the_exact_pool_never_reaches_scip(monkeypatch):
    df, T0 = _panel()
    _no_scip(monkeypatch)
    res = _fit(df, T0, "exact", m_eq=2, top_K=3)
    assert len(res.pool) == 3


def test_a_range_of_one_treated_count_matches_m_eq():
    df, T0 = _panel(seed=28)
    a = _fit(df, T0, "exact", m_min=2, m_max=2)
    b = _fit(df, T0, "exact", m_eq=2)
    assert _treated(a) == _treated(b)
    assert a.globres.objective == pytest.approx(b.globres.objective)


_WITHOUT_PYSCIPOPT = r"""
import sys
sys.modules["pyscipopt"] = None          # import pyscipopt -> ImportError
import cvxpy as cp
assert "SCIP" not in cp.installed_solvers()
sys.path.insert(0, {tests!r})
from test_marex_exact_backend import _panel, _fit
df, T0 = _panel()
res = _fit(df, T0, "exact", m_eq=2)
assert res.study.backend == "exact"
print("treated", sorted(map(str, res.assignment["treated"])))
"""


def test_the_exact_backend_runs_without_pyscipopt():
    """SCIP is an optional extra; the search has to work on a base install.

    The monkeypatch above stops a solve that names SCIP. This goes further and
    removes ``pyscipopt`` before cvxpy loads, which is what a user without the
    ``design`` extra has.

    The child's traceback goes to captured output, not into the failure
    message. ``conftest.py`` turns a failure whose message names a missing
    solver into a skip, and a missing SCIP is the failure this test exists to
    report: with the traceback in the message, a fit routed to the MIQP was
    reported as skipped.
    """
    import os
    import subprocess
    import sys

    code = _WITHOUT_PYSCIPOPT.format(tests=os.path.dirname(os.path.abspath(__file__)))
    run = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    if run.returncode != 0 or "treated" not in run.stdout:
        print(run.stdout[-2000:], run.stderr[-4000:], sep="\n")
        pytest.fail("the exact backend did not finish in a process without "
                    "pyscipopt; its traceback is in the captured output",
                    pytrace=False)
