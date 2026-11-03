"""The batched scorer and the local search after the greedy (#962).

Two units, tested in the order they depend on each other.

``design/batched.py`` scores many whole designs in one call from cached Grams
and window sums. Every term of the reference objective -- the four gates, the
rounded correlation, the inverse required impact -- is a quadratic form in the
membership indicators, so the claim it has to meet is exact agreement with
:func:`~mlsynth.utils.tbr_helpers.design.objective.score_split`, design for
design. A design whose regression is degenerate on either window is scored by
``score_split`` itself, so the claim holds there by construction and the tests
check that the fallback is taken.

``design/local.py`` starts from the greedy's design at each treatment size, adds
seeded random restarts, and at every step scores the whole neighbourhood --
control add, drop and swap, and treatment swaps -- in one batched call, taking
the best move only after ``score_split`` confirms it improves. What it promises:
never worse than the design it starts from, Au's eligibility respected, a local
optimum under its own moves, the same answer for the same seed.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.tbr_helpers.design.batched import design_cache, score_designs
from mlsynth.utils.tbr_helpers.design.local import local_search, neighbourhood
from mlsynth.utils.tbr_helpers.design.objective import score_split
from mlsynth.utils.tbr_helpers.design.search import (
    CONTROL, ROLES, TREATMENT, UNASSIGNED, greedy_search)

ALL = frozenset(ROLES)
_GEOLIFT = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                        "geolift_test_data.csv")


def _panel(n_geos=10, n_periods=40, seed=0):
    """Geos of differing size sharing a common factor, periods by geos."""
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    common = 1.0 + 0.6 * rng.normal(size=(n_periods, 1))
    idio = 1.0 + 0.12 * rng.normal(size=(n_periods, n_geos))
    return size * (0.5 * common + 0.5 * idio)


def _geolift_window(n_last=90):
    df = pd.read_csv(_GEOLIFT).rename(columns={"location": "geo"})
    dates = sorted(df["date"].unique())[-n_last:]
    wide = df[df["date"].isin(dates)].pivot(index="date", columns="geo",
                                            values="Y").sort_index()
    return wide.to_numpy(float), list(wide.columns)


def _direct(Y, T, C, n_test):
    return score_split(Y[:, sorted(T)].sum(1), Y[:, sorted(C)].sum(1),
                       objective="reference", n_test=n_test).key


def _rows(N, designs):
    UT = np.zeros((len(designs), N))
    UC = np.zeros((len(designs), N))
    for r, (T, C) in enumerate(designs):
        UT[r, list(T)] = 1.0
        UC[r, list(C)] = 1.0
    return UT, UC


def _same(batched_key, direct_key):
    """Gates and the rounded correlation exactly; the impact to rounding."""
    assert tuple(int(v) for v in batched_key[:4]) == tuple(direct_key[:4])
    assert float(batched_key[4]) == direct_key[4]
    assert batched_key[5] == pytest.approx(direct_key[5], rel=1e-10)


def _random_designs(rng, N, m):
    out = []
    for _ in range(m):
        k = int(rng.integers(1, min(4, N - 1) + 1))
        T = rng.choice(N, size=k, replace=False).tolist()
        rest = [g for g in range(N) if g not in T]
        C = rng.choice(rest, size=int(rng.integers(1, len(rest) + 1)),
                       replace=False).tolist()
        out.append((T, C))
    return out


# --------------------------------------------------------------------------- #
# the batched scorer
# --------------------------------------------------------------------------- #
def test_batched_scores_equal_score_split_on_the_geolift_window():
    """Smoke and the claim itself, on the panel the benchmark uses."""
    Y, _ = _geolift_window()
    cache = design_cache(Y, n_test=14)
    designs = _random_designs(np.random.default_rng(0), Y.shape[1], 120)
    keys = score_designs(cache, *_rows(Y.shape[1], designs))
    assert keys.shape == (120, 6) and np.all(np.isfinite(keys[:, 4]))
    for kb, (T, C) in zip(keys, designs):
        _same(kb, _direct(Y, T, C, 14))


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), n_geos=st.integers(3, 12),
       n_periods=st.integers(12, 60), n_test=st.integers(1, 8))
def test_batched_scores_equal_score_split_over_the_domain(seed, n_geos, n_periods,
                                                          n_test):
    Y = _panel(n_geos, n_periods, seed)
    cache = design_cache(Y, n_test=n_test)
    designs = _random_designs(np.random.default_rng(seed), n_geos, 15)
    keys = score_designs(cache, *_rows(n_geos, designs))
    for kb, (T, C) in zip(keys, designs):
        _same(kb, _direct(Y, T, C, n_test))


def test_a_constant_control_series_is_scored_by_the_direct_fit():
    """Edge: the control aggregate does not vary on the window, so the slope is
    not identified and the fallback answers. It must still agree."""
    Y = _panel(6, 30, 1)
    Y[:, 5] = 7.0
    cache = design_cache(Y, n_test=5)
    keys = score_designs(cache, *_rows(6, [([0], [5]), ([0, 1], [5])]))
    _same(keys[0], _direct(Y, [0], [5], 5))
    _same(keys[1], _direct(Y, [0, 1], [5], 5))


def test_a_control_constant_only_on_the_fit_window_is_scored_directly():
    """Edge: varying over the window, constant over the A/A fit periods."""
    Y = _panel(5, 30, 2)
    Y[:25, 4] = 3.0
    cache = design_cache(Y, n_test=5)
    keys = score_designs(cache, *_rows(5, [([0], [4])]))
    _same(keys[0], _direct(Y, [0], [4], 5))


def test_an_exact_fit_is_scored_by_the_direct_fit():
    """Edge: the treatment aggregate is an exact affine function of the control
    aggregate, so the residual is zero and every gate divides by nothing."""
    Y = _panel(4, 30, 3)
    Y[:, 0] = 2.0 * Y[:, 1] + 5.0
    cache = design_cache(Y, n_test=5)
    keys = score_designs(cache, *_rows(4, [([0], [1])]))
    _same(keys[0], _direct(Y, [0], [1], 5))


def test_a_single_control_geo_is_scored():
    Y = _panel(5, 30, 4)
    cache = design_cache(Y, n_test=5)
    keys = score_designs(cache, *_rows(5, [([2], [3])]))
    _same(keys[0], _direct(Y, [2], [3], 5))


def test_a_design_with_no_control_geo_is_refused():
    Y = _panel(4, 20, 5)
    cache = design_cache(Y, n_test=3)
    with pytest.raises(MlsynthDataError, match="control"):
        score_designs(cache, *_rows(4, [([0], [])]))


def test_a_geo_in_both_groups_is_refused():
    """A geo counted in both aggregates shares its variance with both series,
    which is the double counting the greedy's step rules out."""
    Y = _panel(4, 20, 6)
    cache = design_cache(Y, n_test=3)
    with pytest.raises(MlsynthDataError, match="both"):
        score_designs(cache, *_rows(4, [([0, 1], [1, 2])]))


def test_rows_that_do_not_match_the_panel_are_refused():
    Y = _panel(4, 20, 7)
    cache = design_cache(Y, n_test=3)
    with pytest.raises(MlsynthDataError, match="geo"):
        score_designs(cache, np.zeros((2, 3)), np.zeros((2, 3)))


def test_a_window_too_short_for_the_aa_fit_is_refused():
    """The A/A test refits on the periods before the last ``n_test``; with
    fewer than three of them it has no residual scale, as ``score_split`` says."""
    with pytest.raises(MlsynthDataError, match="A/A"):
        design_cache(_panel(4, 12, 8), n_test=10)


# --------------------------------------------------------------------------- #
# the neighbourhood and its eligibility
# --------------------------------------------------------------------------- #
def _check_roles(T, C, roles, k):
    assert len(T) == k and C and not set(T) & set(C)
    for j, r in enumerate(roles):
        if j in T:
            assert TREATMENT in r
        elif j in C:
            assert CONTROL in r
        else:
            assert UNASSIGNED in r
    forced = [j for j, r in enumerate(roles) if r == frozenset({TREATMENT})]
    assert set(forced) <= set(T)


def test_every_neighbour_respects_eligibility():
    roles = [ALL] * 8
    roles[0] = frozenset({TREATMENT})                 # forced into treatment
    roles[1] = frozenset({CONTROL})                   # pinned into control
    roles[2] = frozenset({TREATMENT, UNASSIGNED})     # never a control
    roles[3] = frozenset({CONTROL, UNASSIGNED})       # never treated
    T, C = [0, 4], {1, 5}
    nbrs = neighbourhood(T, C, roles)
    assert nbrs
    for T2, C2 in nbrs:
        _check_roles(T2, C2, roles, 2)


def test_the_neighbourhood_holds_every_move_type():
    """Unit: adds, drops, control swaps and treatment swaps are all present."""
    roles = [ALL] * 6
    T, C = [0], {1, 2}
    nbrs = {(tuple(sorted(a)), frozenset(b)) for a, b in neighbourhood(T, C, roles)}
    assert ((0,), frozenset({1, 2, 3})) in nbrs          # add
    assert ((0,), frozenset({1})) in nbrs                # drop
    assert ((0,), frozenset({1, 3})) in nbrs             # control swap
    assert ((4,), frozenset({1, 2})) in nbrs             # treatment swap
    assert ((1,), frozenset({2})) in nbrs                # a control geo is treated
    assert ((4,), frozenset({0, 1, 2})) in nbrs          # the old treated geo controls


def test_no_neighbour_empties_the_control_group():
    roles = [ALL] * 3
    for T2, C2 in neighbourhood([0], {1}, roles):
        assert C2


# --------------------------------------------------------------------------- #
# the local search
# --------------------------------------------------------------------------- #
def _greedy(Y, roles, K, n_test):
    return greedy_search(Y, roles, max_treatment_size=K, n_test=n_test,
                         objective="reference")


def test_the_local_search_is_never_worse_than_the_greedy_on_geolift():
    """Smoke on the real panel, and the measurement #962 reported at k = 1."""
    Y, _ = _geolift_window()
    roles = [ALL] * Y.shape[1]
    greedy = _greedy(Y, roles, 2, 14)
    for g in greedy:
        out = local_search(Y, roles, g, n_test=14, restarts=10, seed=0)
        assert out.score.key >= g.score.key
        assert out.k == g.k and out.converged
    k1 = local_search(Y, roles, greedy[0], n_test=14, restarts=10, seed=0)
    assert k1.score.key[:4] == (1, 1, 1, 1)
    assert k1.score.key[5] >= 1.2 * greedy[0].score.key[5]


def test_the_result_is_a_local_optimum_under_its_own_moves():
    """Unit: no neighbour of the returned design scores higher under the direct
    fit, which is the invariant the stopping rule claims."""
    Y = _panel(9, 40, 11)
    roles = [ALL] * 9
    g = _greedy(Y, roles, 2, 6)[-1]
    out = local_search(Y, roles, g, n_test=6, restarts=3, seed=1)
    for T2, C2 in neighbourhood(out.treatment, set(out.control), roles):
        assert _direct(Y, T2, C2, 6) <= out.score.key


def test_the_trace_climbs_and_ends_at_the_returned_score():
    Y = _panel(9, 40, 12)
    roles = [ALL] * 9
    g = _greedy(Y, roles, 2, 6)[-1]
    out = local_search(Y, roles, g, n_test=6, restarts=0, seed=0)
    assert out.trace[0] == g.score.key
    assert all(b > a for a, b in zip(out.trace, out.trace[1:]))
    assert out.trace[-1] == out.score.key
    assert out.evaluations > 0


def test_the_same_seed_gives_the_same_design():
    Y = _panel(10, 40, 13)
    roles = [ALL] * 10
    g = _greedy(Y, roles, 2, 6)[-1]
    a = local_search(Y, roles, g, n_test=6, restarts=5, seed=3)
    b = local_search(Y, roles, g, n_test=6, restarts=5, seed=3)
    assert (a.treatment, a.control, a.score.key) == (b.treatment, b.control, b.score.key)


def test_restarts_reach_designs_the_greedy_start_does_not():
    """Unit: the restarts are used. Over these panels some restart beats the
    descent from the greedy's design alone."""
    improved = 0
    for seed in range(6):
        Y = _panel(10, 40, 100 + seed)
        roles = [ALL] * 10
        g = _greedy(Y, roles, 2, 6)[-1]
        alone = local_search(Y, roles, g, n_test=6, restarts=0, seed=seed)
        more = local_search(Y, roles, g, n_test=6, restarts=20, seed=seed)
        assert more.score.key >= alone.score.key
        improved += more.score.key > alone.score.key
    assert improved >= 1


def _exhaustive(Y, k, n_test):
    """Every design with a treatment group of k, scored in batch."""
    N = Y.shape[1]
    cache = design_cache(Y, n_test=n_test)
    designs = []
    for T in itertools.combinations(range(N), k):
        rest = [g for g in range(N) if g not in T]
        for labels in itertools.product((0, 1), repeat=len(rest)):
            C = [g for g, lab in zip(rest, labels) if lab]
            if C:
                designs.append((list(T), C))
    keys = score_designs(cache, *_rows(N, designs))
    best = max(range(len(designs)), key=lambda i: tuple(keys[i]))
    T, C = designs[best]
    return _direct(Y, T, C, n_test)


@pytest.mark.parametrize("k", [1, 2])
def test_on_a_small_panel_the_search_reaches_the_exhaustive_optimum(k):
    """Seven geos of the GeoLift window, where every design can be scored: the
    local search with restarts finds the best one, and the greedy does not at
    both sizes."""
    Y, _ = _geolift_window()
    Y = Y[:, [0, 5, 10, 15, 20, 25, 30]]
    roles = [ALL] * 7
    g = _greedy(Y, roles, k, 14)[-1]
    out = local_search(Y, roles, g, n_test=14, restarts=30, seed=0)
    assert out.score.key == pytest.approx(_exhaustive(Y, k, 14))


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), n_geos=st.integers(4, 9),
       pattern=st.lists(st.sampled_from(["all", "forced", "pinned", "no_control",
                                         "no_treat"]), min_size=9, max_size=9))
def test_eligibility_holds_and_the_start_is_never_lost(seed, n_geos, pattern):
    """Property: over random panels and random eligibility, the result keeps
    every geo in a role it allows, keeps every forced geo treated, and scores at
    least as well as the design it started from."""
    table = {"all": ALL, "forced": frozenset({TREATMENT}),
             "pinned": frozenset({CONTROL}),
             "no_control": frozenset({TREATMENT, UNASSIGNED}),
             "no_treat": frozenset({CONTROL, UNASSIGNED})}
    roles = [table[p] for p in pattern[:n_geos]]
    n_forced = sum(r == table["forced"] for r in roles)
    n_treat = sum(TREATMENT in r for r in roles)
    if not any(CONTROL in r for r in roles) or n_forced > 2:
        return
    K = max(n_forced, 1)
    spare = {j for j, r in enumerate(roles) if CONTROL in r} - \
        {j for j, r in enumerate(roles) if TREATMENT in r}
    if K > (n_treat if spare else n_treat - 1):
        return
    Y = _panel(n_geos, 30, seed)
    try:
        g = _greedy(Y, roles, K, 5)[-1]
    except MlsynthDataError:
        return
    out = local_search(Y, roles, g, n_test=5, restarts=3, seed=seed)
    _check_roles(out.treatment, out.control, roles, K)
    assert out.score.key >= g.score.key


def test_a_negative_restart_count_is_refused():
    Y = _panel(6, 30, 14)
    roles = [ALL] * 6
    g = _greedy(Y, roles, 1, 5)[0]
    with pytest.raises(MlsynthDataError, match="restarts"):
        local_search(Y, roles, g, n_test=5, restarts=-1, seed=0)


# --------------------------------------------------------------------------- #
# the estimator option
# --------------------------------------------------------------------------- #
def _long(Y):
    T, N = Y.shape
    return pd.DataFrame([dict(geo=f"g{j:02d}", date=t, Y=float(Y[t, j]))
                         for j in range(N) for t in range(T)])


def _cfg(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=2, n_test=6)
    kw.update(over)
    return TBRConfig(**kw)


def test_the_estimator_runs_the_local_search_when_asked():
    """Smoke through TBR: every design is marked local, keeps the greedy's key
    beside its own, and is never below it."""
    df = _long(_panel(10, 40, 15))
    res = TBR(_cfg(df, search="local", local_restarts=3, local_seed=0)).fit()
    greedy = TBR(_cfg(df)).fit()
    assert [d.search for d in res.designs] == ["local", "local"]
    for d, g in zip(res.designs, greedy.designs):
        assert d.greedy_key == tuple(g.matching_trace[-1])
        assert tuple(d.matching_trace[0]) == d.greedy_key
        assert all(b >= a for a, b in zip(d.matching_trace, d.matching_trace[1:]))
        assert tuple(d.matching_trace[-1]) >= tuple(d.greedy_key)
    assert res.recommended in res.designs


def test_a_restart_that_wins_reports_a_climb_from_the_greedy_key():
    """Unit: when a random start beats the greedy's descent, the trace still
    opens at the greedy's key and never falls."""
    for seed in range(6):
        Y = _panel(10, 40, 100 + seed)
        roles = [ALL] * 10
        g = _greedy(Y, roles, 2, 6)[-1]
        alone = local_search(Y, roles, g, n_test=6, restarts=0, seed=seed)
        more = local_search(Y, roles, g, n_test=6, restarts=20, seed=seed)
        if more.score.key > alone.score.key:
            assert more.trace[0] == g.score.key
            assert all(b > a for a, b in zip(more.trace, more.trace[1:]))
            assert more.trace[-1] == more.score.key
            return
    pytest.fail("no restart beat the greedy's descent on these panels")


def test_the_default_search_is_the_greedy_and_is_unchanged():
    df = _long(_panel(10, 40, 16))
    greedy = TBR(_cfg(df)).fit()
    assert _cfg(df).search == "greedy"
    assert all(d.search == "greedy" and d.greedy_key is None for d in greedy.designs)


def test_the_local_search_needs_the_reference_objective():
    """The batched scorer computes the reference score; the paper's p-values
    are not quadratic forms, so the combination is refused."""
    df = _long(_panel(6, 30, 17))
    with pytest.raises(MlsynthConfigError, match="reference"):
        _cfg(df, search="local", objective="paper")


def test_local_settings_without_the_local_search_are_refused():
    df = _long(_panel(6, 30, 18))
    with pytest.raises(MlsynthConfigError, match="search"):
        _cfg(df, local_restarts=5)
    with pytest.raises(MlsynthConfigError, match="search"):
        _cfg(df, local_seed=2)


def test_a_negative_restart_count_is_refused_by_the_config():
    df = _long(_panel(6, 30, 19))
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, search="local", local_restarts=-1)


def test_the_search_options_are_the_only_ones_accepted():
    df = _long(_panel(6, 30, 20))
    for s in ("greedy", "local"):
        assert _cfg(df, search=s).search == s
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, search="exhaustive")


def test_a_named_split_has_nothing_to_search_locally():
    df = _long(_panel(6, 30, 21))
    df["is_t"] = (df.geo == "g00").astype(int)
    df["is_c"] = (df.geo != "g00").astype(int)
    df["post"] = (df.date >= 25).astype(int)
    with pytest.raises(MlsynthConfigError, match="search"):
        TBRConfig(df=df, unitid="geo", time="date", outcome="Y",
                  treatment_col="is_t", control_col="is_c", post_col="post",
                  search="local")


def test_a_window_that_is_not_periods_by_geos_is_refused():
    with pytest.raises(MlsynthDataError, match="periods by geos"):
        design_cache(np.arange(20.0), n_test=3)


def test_a_design_with_no_treatment_geo_is_refused():
    Y = _panel(4, 20, 22)
    cache = design_cache(Y, n_test=3)
    with pytest.raises(MlsynthDataError, match="treatment"):
        score_designs(cache, *_rows(4, [([], [1])]))


def test_a_treated_geo_that_may_only_control_returns_to_the_control_group():
    """Edge: a geo eligible for treatment and control but not for exclusion
    leaves the treatment group into the control group, never out of the
    experiment."""
    roles = [ALL] * 5
    roles[0] = frozenset({TREATMENT, CONTROL})
    swaps = [(T2, C2) for T2, C2 in neighbourhood([0], {1}, roles) if 0 not in T2]
    assert swaps and all(0 in C2 for _, C2 in swaps)


def test_a_restart_with_no_room_for_a_control_group_is_skipped():
    """Edge: when every control-eligible geo is also treatable, a random
    treatment group can take them all; that draw is skipped, not scored."""
    Y = _panel(3, 30, 23)
    roles = [ALL, ALL, frozenset({TREATMENT, UNASSIGNED})]
    g = _greedy(Y, roles, 2, 5)[-1]
    out = local_search(Y, roles, g, n_test=5, restarts=25, seed=0)
    _check_roles(out.treatment, out.control, roles, 2)
