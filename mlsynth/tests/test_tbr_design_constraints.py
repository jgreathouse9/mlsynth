"""Business constraints in TBR's design search (#1046).

TBR's searched mode took Au's ``A_i`` and a maximum treatment size. SYNDES,
MAREX and LEXSCM offer a shared vocabulary of rules on the design, and these
tests pin that TBR now takes the same fields with the same meaning:

* ``to_be_treated`` / ``not_to_be_treated`` and the ``size_col`` band, folded
  into the role sets: a forced geo is eligible for treatment alone, a forbidden
  or out-of-band geo loses the treatment role and stays a control candidate;
* ``cluster_col`` / ``adjacency`` / ``spillover_threshold`` -- no two treated
  geos conflict;
* ``stratum_col`` with ``min_per_stratum`` / ``max_per_stratum`` -- quotas on the
  treatment group;
* ``market_cost_col`` with ``max_total_cost`` -- a budget on the treated geos;
* ``exclude_bordering_donors`` -- a treated geo's neighbours are not controls;
* ``min_controls`` / ``max_controls`` -- a band on the control group's size.

What the search promises: every design it returns, greedy or local, satisfies
every rule in force, and a problem it cannot satisfy raises
:class:`~mlsynth.exceptions.MlsynthDataError`. With no rule set it is the search
it was, bit for bit, which the golden outcomes below pin against the values
``main`` produced before this change.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.tests.test_tbr_design_rules import (
    ALL, C_ONLY, C_U, T_C, T_ONLY, oracle_design_ok, problems, _rules)
from mlsynth.tests.test_tbrmm import market_panel
from mlsynth.utils.tbr_helpers.design import local as local_mod
from mlsynth.utils.tbr_helpers.design import search as search_mod
from mlsynth.utils.tbr_helpers.design.local import local_search, neighbourhood
from mlsynth.utils.tbr_helpers.design.objective import score_split
from mlsynth.utils.tbr_helpers.design.rules import DesignRules
from mlsynth.utils.tbr_helpers.design.search import SearchOutcome, greedy_search
from mlsynth.utils.tbr_helpers.design.structures import TBRDesignConstraints

_GEOLIFT = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                        "geolift_test_data.csv")


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _matrix(n_geos=8, n_periods=30, seed=0):
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
    return wide.to_numpy(float)


def geo_panel(n_units=12, T=40, seed=0):
    """A market panel with the per-geo attributes the rules read.

    Geo ``m{j}`` sits in cluster ``j // 2`` (pairs), stratum ``j % 3``, has size
    ``10 (j + 1)`` and market cost ``1 + j % 4``.
    """
    df = market_panel(n_units=n_units, T=T, seed=seed)
    j = df.geo.str[1:].astype(int)
    df["dma"] = "d" + (j // 2).astype(str)
    df["region"] = "r" + (j % 3).astype(str)
    df["pop"] = 10.0 * (j + 1)
    df["test_cost"] = 1.0 + (j % 4)
    return df


def cfg(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=3, n_test=10)
    kw.update(over)
    return TBRConfig(**kw)


def fit(df, **over):
    return TBR(cfg(df, **over)).fit()


def _neighbours(df, kw):
    """Conflict neighbours by label, read off the panel and the adjacency."""
    attr = df.groupby("geo").first()
    out = {g: set() for g in attr.index}
    if kw.get("cluster_col"):
        for a in attr.index:
            for b in attr.index:
                if a != b and attr.loc[a, kw["cluster_col"]] == attr.loc[b, kw["cluster_col"]]:
                    out[a].add(b)
    adj = kw.get("adjacency")
    if adj is not None:
        thr = kw.get("spillover_threshold", 0.0)
        for a in adj.index:
            for b in adj.columns:
                if a != b and (adj.loc[a, b] > thr or adj.loc[b, a] > thr):
                    out[a].add(b)
    return out


def assert_rules_hold(res, df, kw):
    """Every rule in ``kw``, checked from the panel and not from rules.py."""
    attr = df.groupby("geo").first()
    near = _neighbours(df, kw)
    forbidden = set(kw.get("not_to_be_treated") or [])
    if kw.get("size_col"):
        s = attr[kw["size_col"]]
        lo, hi = kw.get("min_size", -np.inf), kw.get("max_size", np.inf)
        forbidden |= set(s.index[(s < lo) | (s > hi)])
    assert res.designs, "a feasible problem returns at least one design"
    for d in res.designs:
        T, C = set(d.treatment_units), set(d.control_units)
        assert len(T) == d.k <= kw.get("max_treatment_size", 3)
        assert set(kw.get("to_be_treated") or []) <= T
        assert not (T & forbidden)
        for t in T:
            assert not (near[t] & T), f"{t} conflicts with a treated geo"
        if kw.get("stratum_col"):
            strata = attr[kw["stratum_col"]]
            for s in strata.unique():
                members = set(strata.index[strata == s])
                n = len(T & members)
                if kw.get("max_per_stratum") is not None:
                    assert n <= kw["max_per_stratum"]
                if kw.get("min_per_stratum") is not None and members - forbidden:
                    assert n >= kw["min_per_stratum"]
        if kw.get("market_cost_col"):
            assert attr.loc[sorted(T), kw["market_cost_col"]].sum() \
                <= kw["max_total_cost"] + 1e-9
        if kw.get("exclude_bordering_donors"):
            for t in T:
                assert not (near[t] & C), f"a neighbour of {t} is a control"
        if kw.get("min_controls") is not None:
            assert len(C) >= kw["min_controls"]
        if kw.get("max_controls") is not None:
            assert len(C) <= kw["max_controls"]


def _adjacency(df, pairs):
    geos = sorted(df.geo.unique())
    adj = pd.DataFrame(0.0, index=geos, columns=geos)
    for a, b in pairs:
        adj.loc[a, b] = adj.loc[b, a] = 1.0
    return adj


# --------------------------------------------------------------------------- #
# smoke: each field fits and returns designs that keep it
# --------------------------------------------------------------------------- #
FIELD_CASES = {
    "to_be_treated": dict(to_be_treated=["m03"]),
    "not_to_be_treated": dict(not_to_be_treated=["m00", "m01", "m02"]),
    "size_band": dict(size_col="pop", min_size=30.0, max_size=90.0),
    "cluster": dict(cluster_col="dma"),
    "adjacency": dict(spillover_threshold=0.5),            # adjacency added below
    "strata": dict(stratum_col="region", min_per_stratum=1, max_per_stratum=1),
    "budget": dict(market_cost_col="test_cost", max_total_cost=4.0),
    "bordering": dict(cluster_col="dma", exclude_bordering_donors=True),
    "controls": dict(min_controls=4, max_controls=6),
}


@pytest.mark.parametrize("name", sorted(FIELD_CASES))
def test_each_rule_fits_and_every_design_keeps_it(name):
    df = geo_panel()
    kw = dict(FIELD_CASES[name])
    if name == "adjacency":
        kw["adjacency"] = _adjacency(df, [("m00", "m05"), ("m05", "m09"),
                                          ("m02", "m07")])
    res = fit(df, **kw)
    assert_rules_hold(res, df, kw)


def test_every_rule_together_with_the_local_search():
    df = geo_panel()
    kw = dict(to_be_treated=["m04"], not_to_be_treated=["m00"],
              size_col="pop", max_size=110.0, cluster_col="dma",
              stratum_col="region", max_per_stratum=1,
              market_cost_col="test_cost", max_total_cost=8.0,
              exclude_bordering_donors=True, min_controls=3, max_controls=6)
    res = fit(df, search="local", local_restarts=3, **kw)
    assert_rules_hold(res, df, kw)
    assert all(d.search == "local" for d in res.designs)


# --------------------------------------------------------------------------- #
# the result records the rules in force
# --------------------------------------------------------------------------- #
def test_the_result_records_the_constraints_in_force():
    df = geo_panel()
    res = fit(df, to_be_treated=["m04"], not_to_be_treated=["m00"],
              size_col="pop", max_size=110.0, cluster_col="dma",
              stratum_col="region", min_per_stratum=1, max_per_stratum=2,
              market_cost_col="test_cost", max_total_cost=9.0,
              exclude_bordering_donors=True, min_controls=2, max_controls=8)
    rec = res.design_constraints
    assert isinstance(rec, TBRDesignConstraints)
    assert rec.to_be_treated == ["m04"]
    assert rec.not_to_be_treated == ["m00"]
    assert rec.size_ineligible_units == ["m11"]
    assert ("m04", "m05") in rec.conflict_pairs and len(rec.conflict_pairs) == 6
    assert sorted(q.stratum for q in rec.strata) == ["r0", "r1", "r2"]
    assert all(q.min_treated == 1 and q.max_treated == 2 for q in rec.strata)
    r1 = next(q for q in rec.strata if q.stratum == "r1")
    assert r1.members == ["m01", "m04", "m07", "m10"]
    assert rec.market_costs["m03"] == 4.0 and rec.max_total_cost == 9.0
    assert rec.exclude_bordering_donors is True
    assert (rec.min_controls, rec.max_controls) == (2, 8)
    assert set(rec.rules) == {"cluster_col/adjacency", "min_per_stratum",
                              "max_per_stratum", "max_total_cost",
                              "exclude_bordering_donors", "min_controls",
                              "max_controls"}
    assert "m04" in res.forced_treatment_units
    assert "m00" not in res.eligible_treatment_units
    assert "m00" in res.eligible_control_units


def test_no_constraint_records_nothing():
    assert fit(geo_panel()).design_constraints is None


def test_forced_and_forbidden_alone_record_without_rules():
    res = fit(geo_panel(), to_be_treated=["m02"])
    assert res.design_constraints.rules == []
    assert res.design_constraints.to_be_treated == ["m02"]


# --------------------------------------------------------------------------- #
# regression: no rule leaves the search as it was on main
# --------------------------------------------------------------------------- #
# Captured from origin/main at 38d64b1b, before the rules existed:
# (k, treatment in augmentation order, control, evaluations, trace length).
GOLDEN = {
    "geolift-carried": [
        (1, [9], [2, 4, 5, 7, 10, 13, 16, 18, 19, 28, 29, 32, 34, 37, 39], 968, 25),
        (2, [9, 10], [2, 3, 4, 5, 7, 13, 16, 18, 19, 28, 32, 37, 39], 186, 4),
        (3, [9, 10, 6], [0, 2, 3, 5, 7, 13, 15, 16, 18, 19, 22, 23, 25, 28, 29, 32,
                         36, 37, 39], 356, 9),
        (4, [9, 10, 6, 5], [0, 2, 3, 7, 13, 16, 19, 22, 23, 25, 31, 32, 36, 39], 346, 9)],
    "geolift-pool": [
        (1, [9], [2, 4, 5, 7, 10, 13, 16, 18, 19, 28, 29, 32, 34, 37, 39], 968, 25),
        (2, [9, 6], [0, 1, 2, 3, 5, 7, 10, 13, 15, 19, 20, 22, 23, 25, 29, 31, 32,
                     33, 36, 39], 726, 19),
        (3, [9, 6, 39], [0, 2, 3, 4, 5, 7, 10, 12, 13, 15, 16, 18, 19, 20, 22, 23,
                         24, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38], 391, 10),
        (4, [9, 6, 39, 25], [0, 1, 2, 3, 5, 7, 8, 10, 13, 15, 16, 18, 19, 20, 22,
                             23, 24, 29, 32, 34, 36, 38], 550, 15)],
    "geolift-best": [
        (1, [9], [2, 4, 5, 7, 10, 13, 16, 18, 19, 28, 29, 32, 34, 37, 39], 1936, 25),
        (2, [9, 6], [0, 1, 2, 3, 5, 7, 10, 13, 15, 19, 20, 22, 23, 25, 29, 31, 32,
                     33, 36, 39], 912, 19),
        (3, [9, 6, 39], [0, 2, 3, 4, 5, 7, 10, 12, 13, 15, 16, 18, 19, 20, 22, 23,
                         24, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38], 747, 10),
        (4, [9, 6, 39, 25], [0, 1, 2, 3, 5, 7, 8, 10, 13, 15, 16, 18, 19, 20, 22,
                             23, 24, 29, 32, 34, 36, 38], 896, 15)],
    "sim-carried": [(1, [8], [1, 2, 4, 6, 9], 48, 5), (2, [8, 9], [1, 2, 4, 5, 6], 24, 2),
                    (3, [8, 9, 4], [0, 1, 2, 5, 6], 21, 2)],
    "sim-pool": [(1, [8], [1, 2, 4, 6, 9], 48, 5), (2, [8, 9], [1, 2, 4, 5, 6], 36, 4),
                 (3, [8, 9, 4], [0, 1, 2, 5, 6], 26, 3)],
    "sim-best": [(1, [8], [1, 2, 4, 6, 9], 96, 5), (2, [8, 9], [1, 2, 4, 5, 6], 60, 2),
                 (3, [8, 9, 4], [0, 1, 2, 5, 6], 47, 2)],
}
GOLDEN_LOCAL = {"geolift": ([9], [2, 4, 5, 10, 16, 18, 19, 28, 32, 37, 39], 30472),
                "sim": ([8], [1, 2, 4, 6, 9], 732)}


def _sim():
    return _matrix(n_geos=10, n_periods=40, seed=0)


@pytest.mark.parametrize("panel", ["geolift", "sim"])
@pytest.mark.parametrize("start", ["carried", "pool", "best"])
def test_no_rules_reproduce_main_bit_for_bit(panel, start):
    Y, K, n_test = ((_geolift_window(), 4, 14) if panel == "geolift"
                    else (_sim(), 3, 10))
    out = greedy_search(Y, [ALL] * Y.shape[1], max_treatment_size=K, n_test=n_test,
                        objective="reference", control_start=start)
    got = [(o.k, o.treatment, o.control, o.evaluations, len(o.trace)) for o in out]
    assert got == [tuple(g) for g in GOLDEN[f"{panel}-{start}"]]


@pytest.mark.parametrize("panel", ["geolift", "sim"])
def test_no_rules_reproduce_main_local_search(panel):
    Y, K, n_test = ((_geolift_window(), 4, 14) if panel == "geolift"
                    else (_sim(), 3, 10))
    roles = [ALL] * Y.shape[1]
    start = greedy_search(Y, roles, max_treatment_size=K, n_test=n_test,
                          objective="reference")[0]
    out = local_search(Y, roles, start, n_test=n_test, restarts=3, seed=1)
    assert (out.treatment, out.control, out.evaluations) == GOLDEN_LOCAL[panel]
    assert out.abandoned_restarts == 0


def _same_outcomes(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.k, x.treatment, x.control, x.unassigned) == \
            (y.k, y.treatment, y.control, y.unassigned)
        assert x.score.key == y.score.key
        assert x.trace == y.trace
        assert x.evaluations == y.evaluations


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), n_geos=st.integers(4, 9),
       start=st.sampled_from(["carried", "pool", "best"]),
       pattern=st.lists(st.sampled_from([ALL, ALL, ALL, C_U, C_ONLY, T_ONLY]),
                        min_size=9, max_size=9))
def test_an_empty_rule_set_searches_as_no_rules(seed, n_geos, start, pattern):
    """Property: rules that bind nothing leave the greedy's walk unchanged."""
    roles = pattern[:n_geos]
    Y = _matrix(n_geos, 20, seed)
    K = max(1, sum(r == T_ONLY for r in roles))
    run = dict(max_treatment_size=K, n_test=4, objective="reference",
               control_start=start)
    try:
        plain = greedy_search(Y, roles, **run)
    except MlsynthDataError:
        with pytest.raises(MlsynthDataError):
            greedy_search(Y, roles, rules=DesignRules.build(roles), **run)
        return
    _same_outcomes(plain, greedy_search(Y, roles, rules=DesignRules.build(roles), **run))


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000), n_geos=st.integers(4, 8))
def test_an_empty_rule_set_descends_as_no_rules(seed, n_geos):
    Y = _matrix(n_geos, 20, seed)
    roles = [ALL] * n_geos
    g = greedy_search(Y, roles, max_treatment_size=2, n_test=4,
                      objective="reference")[-1]
    a = local_search(Y, roles, g, n_test=4, restarts=3, seed=seed)
    b = local_search(Y, roles, g, n_test=4, restarts=3, seed=seed,
                     rules=DesignRules.build(roles))
    assert (a.treatment, a.control, a.score.key, a.trace, a.evaluations) == \
        (b.treatment, b.control, b.score.key, b.trace, b.evaluations)


# --------------------------------------------------------------------------- #
# unit: the greedy under rules
# --------------------------------------------------------------------------- #
def _greedy(Y, roles, K, rules, **kw):
    return greedy_search(Y, roles, max_treatment_size=K, n_test=5,
                         objective="reference", rules=rules, **kw)


def test_a_size_whose_group_breaks_a_quota_is_not_recorded():
    """Three strata at one each: only k = 3 admits a design."""
    Y = _matrix(9, 30, 1)
    roles = [ALL] * 9
    spec = {"strata": [((0, 1, 2), 1, None), ((3, 4, 5), 1, None),
                       ((6, 7, 8), 1, None)]}
    out = _greedy(Y, roles, 4, _rules(roles, spec))
    assert [o.k for o in out] == [3, 4]
    for o in out:
        assert oracle_design_ok(o.treatment, o.control, roles, spec)


def test_augmentation_prefers_geos_that_complete_the_current_size():
    """Two strata at one each and K = 3: size 2 is admissible and recorded."""
    Y = _matrix(6, 30, 2)
    roles = [ALL] * 6
    spec = {"strata": [((0, 1, 2), 1, None), ((3, 4, 5), 1, None)]}
    out = _greedy(Y, roles, 3, _rules(roles, spec))
    assert [o.k for o in out] == [2, 3]


def test_the_greedy_raises_when_no_size_admits_a_design():
    """Every member of one stratum conflicts with every member of the other."""
    Y = _matrix(6, 30, 3)
    roles = [ALL] * 6
    spec = {"strata": [((0, 1), 1, None), ((2, 3), 1, None)],
            "pairs": [(0, 2), (0, 3), (1, 2), (1, 3)]}
    with pytest.raises(MlsynthDataError, match="no treatment size"):
        _greedy(Y, roles, 2, _rules(roles, spec))


def test_the_greedy_stops_growing_when_no_geo_can_join():
    """A clique of conflicts caps the group at one geo, whatever K allows."""
    Y = _matrix(6, 30, 4)
    roles = [ALL] * 6
    spec = {"pairs": [(i, j) for i in range(5) for j in range(i + 1, 5)]}
    out = _greedy(Y, roles, 3, _rules(roles, spec))
    assert [o.k for o in out] == [1, 2]
    assert 5 in out[1].treatment


def test_bordering_controls_never_enter_the_carried_or_pooled_group():
    Y = _matrix(8, 30, 5)
    roles = [ALL] * 8
    spec = {"pairs": [(0, 1), (0, 2), (3, 4), (5, 6)], "bordering": True}
    for start in ("carried", "pool", "best"):
        for o in _greedy(Y, roles, 3, _rules(roles, spec), control_start=start):
            assert oracle_design_ok(o.treatment, o.control, roles, spec)


def test_a_starting_control_group_above_the_band_is_brought_inside():
    Y = _matrix(9, 30, 6)
    roles = [ALL] * 9
    spec = {"max_controls": 2}
    for o in _greedy(Y, roles, 2, _rules(roles, spec)):
        assert len(o.control) <= 2
        assert o.trace[-1] == o.score.key
        assert all(b >= a for a, b in zip(o.trace, o.trace[1:]))


def test_a_starting_control_group_below_the_band_is_brought_inside():
    Y = _matrix(9, 30, 7)
    roles = [ALL] * 9
    spec = {"min_controls": 7}
    for o in _greedy(Y, roles, 2, _rules(roles, spec), control_start="carried"):
        assert len(o.control) >= 7


def test_a_band_the_must_controls_overflow_skips_the_size():
    """Three geos must be controls unless treated and max_controls is 1: no
    design at k = 1, one at k = 2."""
    Y = _matrix(5, 30, 8)
    roles = [ALL, T_C, T_C, T_C, ALL]
    spec = {"max_controls": 1}
    out = _greedy(Y, roles, 2, _rules(roles, spec))
    assert [o.k for o in out] == [2]
    assert oracle_design_ok(out[0].treatment, out[0].control, roles, spec)


def test_augmentation_sheds_must_controls_to_meet_the_band():
    """Two must-controls and max_controls 1: k = 1 treats one of them."""
    Y = _matrix(5, 30, 8)
    roles = [ALL, T_C, T_C, ALL, ALL]
    spec = {"max_controls": 1}
    out = _greedy(Y, roles, 2, _rules(roles, spec))
    assert [o.k for o in out] == [1, 2]
    assert set(out[0].treatment) & {1, 2}
    for o in out:
        assert oracle_design_ok(o.treatment, o.control, roles, spec)


def test_a_control_minimum_the_pool_just_meets():
    """At k = 2 two controls are left and min_controls is 2."""
    Y = _matrix(4, 30, 9)
    roles = [ALL, ALL, ALL, ALL]
    rules = _rules(roles, {"min_controls": 2})
    out = _greedy(Y, roles, 2, rules)
    assert [o.k for o in out] == [1, 2]
    rules3 = _rules(roles, {"min_controls": 3})
    assert [o.k for o in _greedy(Y, roles, 1, rules3)] == [1]


def test_matching_brings_a_small_control_group_up_to_the_band():
    Y = _matrix(7, 30, 16)
    scorer = search_mod._Scorer(Y, "reference", 5)
    control, trace, converged = search_mod._match(
        scorer, [0], {1}, [1, 2, 3, 4, 5, 6], band=(4, 5))
    assert 4 <= len(control) <= 5 and converged
    assert trace[0] <= trace[-1] == scorer([0], control).key
    assert all(b > a for a, b in zip(trace, trace[1:]))


def test_matching_brings_a_large_control_group_down_to_the_band():
    Y = _matrix(7, 30, 17)
    scorer = search_mod._Scorer(Y, "reference", 5)
    control, _, _ = search_mod._match(
        scorer, [0], {1, 2, 3, 4, 5, 6}, [1, 2, 3, 4, 5, 6], band=(1, 2))
    assert 1 <= len(control) <= 2


def test_matching_reports_a_band_the_pinned_controls_overflow():
    Y = _matrix(5, 30, 18)
    scorer = search_mod._Scorer(Y, "reference", 5)
    assert search_mod._match(scorer, [0], {1, 2, 3}, [3], band=(1, 1)) is None


def test_rules_over_the_wrong_number_of_geos_are_refused():
    Y = _matrix(5, 30, 10)
    with pytest.raises(MlsynthDataError, match="rules cover"):
        _greedy(Y, [ALL] * 5, 2, DesignRules.build([ALL] * 4))


def test_the_audit_runs_before_the_climb():
    Y = _matrix(5, 30, 11)
    roles = [ALL] * 5
    with pytest.raises(MlsynthDataError, match="every treatment-eligible geo costs"):
        _greedy(Y, roles, 2, _rules(roles, {"costs": [5] * 5, "budget": 1}))


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_forced_count_equal_to_k_returns_that_group():
    res = fit(geo_panel(), to_be_treated=["m01", "m04", "m08"], cluster_col="dma")
    assert [d.k for d in res.designs] == [3]
    assert sorted(res.designs[0].treatment_units) == ["m01", "m04", "m08"]


def test_a_single_feasible_size():
    df = geo_panel()
    kw = dict(stratum_col="region", min_per_stratum=1, max_per_stratum=1)
    res = fit(df, **kw)
    assert [d.k for d in res.designs] == [3]
    assert_rules_hold(res, df, kw)


def test_a_budget_exactly_equal_to_a_groups_cost():
    df = geo_panel()
    df["test_cost"] = 1.0
    kw = dict(market_cost_col="test_cost", max_total_cost=2.0)
    res = fit(df, **kw)
    assert [d.k for d in res.designs] == [1, 2]
    assert_rules_hold(res, df, kw)


def test_a_stratum_with_one_eligible_member_always_treats_it():
    df = geo_panel()
    kw = dict(stratum_col="region", min_per_stratum=1,
              not_to_be_treated=["m00", "m03", "m06"])      # r0 keeps m09 alone
    res = fit(df, **kw)
    assert res.designs
    for d in res.designs:
        assert "m09" in d.treatment_units
    assert_rules_hold(res, df, kw)


def test_every_control_banned_but_one():
    """Geo 0 is forced and borders 1 and 2, so the control group is {3}."""
    Y = _matrix(4, 30, 12)
    roles = [T_ONLY, C_U, C_U, C_U]
    spec = {"pairs": [(0, 1), (0, 2)], "bordering": True}
    out = _greedy(Y, roles, 1, _rules(roles, spec))
    assert [(o.treatment, o.control) for o in out] == [([0], [3])]


# --------------------------------------------------------------------------- #
# the local search under rules
# --------------------------------------------------------------------------- #
def test_every_neighbour_under_rules_is_admissible_and_one_move_away():
    roles = [ALL] * 7
    spec = {"pairs": [(0, 1), (2, 3)], "bordering": True,
            "strata": [((0, 1, 2), None, 1)], "costs": [1, 1, 1, 2, 2, 3, 3],
            "budget": 3, "min_controls": 2, "max_controls": 4}
    rules = _rules(roles, spec)
    T, C = [0, 4], {5, 6, 3}
    free = neighbourhood(T, C, roles)
    kept = neighbourhood(T, C, roles, rules=rules)
    assert kept and len(kept) < len(free)
    assert all(m in free for m in kept)
    for T2, C2 in kept:
        assert oracle_design_ok(T2, C2, roles, spec)


def test_random_restarts_draw_only_admissible_designs():
    roles = [ALL] * 8
    spec = {"pairs": [(0, 1), (2, 3), (4, 5)], "bordering": True,
            "costs": [1, 2, 1, 2, 1, 2, 1, 2], "budget": 4,
            "min_controls": 2, "max_controls": 3}
    rules = _rules(roles, spec)
    rng = np.random.default_rng(0)
    draws = [local_mod._random_design(rng, roles, 2, rules) for _ in range(40)]
    assert any(d is not None for d in draws)
    for d in draws:
        if d is not None:
            assert oracle_design_ok(d[0], d[1], roles, spec)


def test_a_draw_that_would_ban_a_pinned_control_is_refused():
    """Treating geo 0 bans geo 1, whose only role is control."""
    roles = [ALL, C_ONLY, C_U, ALL]
    spec = {"pairs": [(0, 1)], "bordering": True}
    rules = _rules(roles, spec)
    rng = np.random.default_rng(1)
    draws = [local_mod._admissible_draw(rng, roles, 1, rules) for _ in range(30)]
    assert any(d is None for d in draws)
    assert any(d is not None for d in draws)
    for d in draws:
        if d is not None:
            assert d[0] != [0]
            assert oracle_design_ok(d[0], d[1], roles, spec)


def test_a_random_draw_gives_up_after_a_bounded_number_of_tries():
    roles = [ALL] * 4
    rules = _rules(roles, {"pairs": [(i, j) for i in range(4) for j in range(i + 1, 4)]})
    assert local_mod._random_design(np.random.default_rng(0), roles, 2, rules) is None
    assert local_mod.RESTART_DRAWS >= 10


def _outcome(Y, T, C, k, n_test=4):
    score = score_split(Y[:, sorted(T)].sum(1), Y[:, sorted(C)].sum(1),
                        objective="reference", n_test=n_test)
    n = Y.shape[1]
    return SearchOutcome(k=k, treatment=sorted(T), control=sorted(C),
                         unassigned=sorted(set(range(n)) - set(T) - set(C)),
                         score=score, trace=[score.key], converged=True,
                         evaluations=1)


def test_abandoned_restarts_are_counted_and_never_returned():
    """One admissible pair among 120: most draws fail and say so."""
    n = 16
    Y = _matrix(n, 20, 13)
    roles = [ALL] * n
    spec = {"pairs": [(i, j) for i in range(n) for j in range(i + 1, n)
                      if (i, j) != (0, 1)]}
    rules = _rules(roles, spec)
    start = _outcome(Y, [0, 1], set(range(2, n)), 2)
    out = local_search(Y, roles, start, n_test=4, restarts=6, seed=0, rules=rules)
    assert out.abandoned_restarts > 0
    assert oracle_design_ok(out.treatment, out.control, roles, spec)


def test_a_design_with_no_admissible_neighbour_is_its_own_optimum():
    Y = _matrix(4, 20, 14)
    roles = [ALL] * 4
    spec = {"costs": [0, 5, 5, 5], "budget": 1, "min_controls": 3,
            "max_controls": 3}
    rules = _rules(roles, spec)
    start = _outcome(Y, [0], {1, 2, 3}, 1)
    assert neighbourhood([0], {1, 2, 3}, roles, rules=rules) == []
    out = local_search(Y, roles, start, n_test=4, restarts=2, seed=0, rules=rules)
    assert (out.treatment, out.control) == ([0], [1, 2, 3])


def test_the_local_search_refuses_an_inadmissible_start():
    Y = _matrix(5, 20, 15)
    roles = [ALL] * 5
    rules = _rules(roles, {"pairs": [(0, 1)]})
    with pytest.raises(MlsynthDataError, match="start"):
        local_search(Y, roles, _outcome(Y, [0, 1], {2, 3}, 2), n_test=4,
                     restarts=0, seed=0, rules=rules)


def test_the_design_result_reports_abandoned_restarts():
    df = geo_panel()
    res = fit(df, search="local", local_restarts=2, cluster_col="dma")
    assert all(d.abandoned_restarts >= 0 for d in res.designs)
    assert fit(df, search="local", local_restarts=2).designs[0].abandoned_restarts == 0


# --------------------------------------------------------------------------- #
# the property: never an inadmissible design
# --------------------------------------------------------------------------- #
@settings(max_examples=60, deadline=None)
@given(problem=problems(), seed=st.integers(0, 10_000),
       start=st.sampled_from(["carried", "pool", "best"]))
def test_every_returned_design_satisfies_every_rule(problem, seed, start):
    """Property: over random panels and rule sets, greedy and local designs keep
    every rule in force, or the search raises MlsynthDataError."""
    roles, spec, K = problem
    Y = _matrix(len(roles), 16, seed)
    rules = _rules(roles, spec)
    try:
        outs = greedy_search(Y, roles, max_treatment_size=K, n_test=4,
                             objective="reference", control_start=start,
                             rules=rules)
    except MlsynthDataError:
        return
    assert outs
    for o in outs:
        assert len(o.treatment) == o.k <= K
        assert oracle_design_ok(o.treatment, o.control, roles, spec)
    loc = local_search(Y, roles, outs[-1], n_test=4, restarts=2, seed=seed,
                       rules=rules)
    assert oracle_design_ok(loc.treatment, loc.control, roles, spec)
    assert loc.k == outs[-1].k
    assert loc.score.key >= outs[-1].score.key


# --------------------------------------------------------------------------- #
# pipeline: folding and failures
# --------------------------------------------------------------------------- #
def test_a_forced_geo_the_eligibility_column_excludes_is_refused():
    df = geo_panel()
    df["can_treat"] = (df.geo != "m02").astype(int)
    with pytest.raises(MlsynthDataError, match="m02"):
        fit(df, to_be_treated=["m02"], treatment_eligible_col="can_treat")


def test_a_forbidden_geo_with_no_other_role_is_refused():
    df = geo_panel()
    df["can_control"] = (df.geo != "m02").astype(int)
    df["can_exclude"] = (df.geo != "m02").astype(int)
    with pytest.raises(MlsynthDataError, match="eligible for no group"):
        fit(df, not_to_be_treated=["m02"], control_eligible_col="can_control",
            unassigned_eligible_col="can_exclude")


def test_an_infeasible_problem_through_the_estimator_names_the_rule():
    df = geo_panel()
    with pytest.raises(MlsynthDataError, match="costs more than max_total_cost"):
        fit(df, market_cost_col="test_cost", max_total_cost=0.5)
    with pytest.raises(MlsynthDataError, match="min_per_stratum quotas need"):
        fit(df, stratum_col="region", min_per_stratum=2, max_treatment_size=5)
    with pytest.raises(MlsynthDataError, match=r"\(min_controls\)"):
        fit(df, min_controls=12)


def test_an_adjacency_missing_a_geo_is_refused():
    df = geo_panel()
    adj = _adjacency(df, [("m00", "m01")]).drop(index="m11", columns="m11")
    with pytest.raises(MlsynthDataError, match="[Aa]djacency"):
        fit(df, adjacency=adj)


# --------------------------------------------------------------------------- #
# config: named mode refuses every field
# --------------------------------------------------------------------------- #
def _named(df, **over):
    df = df.copy()
    df["post"] = (df.date >= 30).astype(int)
    df["tr"] = (df.geo == "m00").astype(int)
    df["ct"] = (df.geo != "m00").astype(int)
    kw = dict(df=df, unitid="geo", time="date", outcome="Y", post_col="post",
              treatment_col="tr", control_col="ct")
    kw.update(over)
    return TBRConfig(**kw)


NAMED_FIELDS = {
    "to_be_treated": dict(to_be_treated=["m01"]),
    "not_to_be_treated": dict(not_to_be_treated=["m01"]),
    "size_col": dict(size_col="pop", min_size=1.0),
    "cluster_col": dict(cluster_col="dma"),
    "adjacency": dict(adjacency=pd.DataFrame()),
    "spillover_threshold": dict(spillover_threshold=0.3),
    "stratum_col": dict(stratum_col="region", min_per_stratum=1),
    "market_cost_col": dict(market_cost_col="test_cost", max_total_cost=3.0),
    "exclude_bordering_donors": dict(exclude_bordering_donors=True),
    "min_controls": dict(min_controls=2),
    "max_controls": dict(max_controls=2),
}


@pytest.mark.parametrize("name", sorted(NAMED_FIELDS))
def test_named_mode_refuses_every_constraint(name):
    with pytest.raises(MlsynthConfigError, match="named split"):
        _named(geo_panel(), **NAMED_FIELDS[name])


def test_named_mode_accepts_the_defaults_spelled_out():
    _named(geo_panel(), exclude_bordering_donors=False)


# --------------------------------------------------------------------------- #
# config: contradictions refused early
# --------------------------------------------------------------------------- #
CONFIG_FAILURES = [
    (dict(to_be_treated=["zz"]), "not in the panel"),
    (dict(not_to_be_treated=["zz"]), "not in the panel"),
    (dict(to_be_treated=["m01"], not_to_be_treated=["m01"]), "both"),
    (dict(to_be_treated=["m01", "m02", "m03", "m04"]), "max_treatment_size"),
    (dict(min_size=1.0), "size_col"),
    (dict(size_col="pop"), "min_size"),
    (dict(size_col="pop", min_size=50.0, max_size=40.0), "min_size"),
    (dict(size_col="pop", min_size=50.0, to_be_treated=["m00"]), "size band"),
    (dict(size_col="pop", max_size=50.0, to_be_treated=["m09"]), "size band"),
    (dict(min_per_stratum=1), "stratum_col"),
    (dict(stratum_col="region"), "min_per_stratum"),
    (dict(stratum_col="region", min_per_stratum=0), "at least 1"),
    (dict(stratum_col="region", max_per_stratum=0), "at least 1"),
    (dict(stratum_col="region", min_per_stratum=2, max_per_stratum=1), "above"),
    (dict(stratum_col="region", min_per_stratum=2,
          not_to_be_treated=["m00", "m03", "m06"]), "r0"),
    (dict(stratum_col="region", min_per_stratum=2, size_col="pop",
          min_size=100.0), "min_per_stratum"),
    (dict(spillover_threshold=0.5), "adjacency"),
    (dict(exclude_bordering_donors=True), "cluster_col"),
    (dict(market_cost_col="test_cost"), "max_total_cost"),
    (dict(max_total_cost=3.0), "market_cost_col"),
    (dict(market_cost_col="test_cost", max_total_cost=0.0), "positive"),
    (dict(market_cost_col="test_cost", max_total_cost=float("inf")), "positive"),
    (dict(min_controls=0), "at least 1"),
    (dict(max_controls=0), "at least 1"),
    (dict(min_controls=4, max_controls=3), "above"),
]


@pytest.mark.parametrize("over, needle", CONFIG_FAILURES)
def test_contradictory_constraints_are_refused(over, needle):
    with pytest.raises(MlsynthConfigError, match=needle):
        cfg(geo_panel(), **over)


def test_a_stratum_minimum_counts_only_members_left_eligible():
    """Two of r0's four members are forbidden; two remain for a minimum of two."""
    cfg(geo_panel(), stratum_col="region", min_per_stratum=2,
        not_to_be_treated=["m00", "m03"], max_treatment_size=6)


@pytest.mark.parametrize("col", ["test_cost", "pop", "dma", "region"])
def test_an_attribute_that_varies_within_a_geo_is_refused(col):
    df = geo_panel()
    df.loc[(df.geo == "m01") & (df.date == 3), col] = (
        999.0 if df[col].dtype.kind == "f" else "other")
    over = {"test_cost": dict(market_cost_col="test_cost", max_total_cost=5.0),
            "pop": dict(size_col="pop", min_size=1.0),
            "dma": dict(cluster_col="dma"),
            "region": dict(stratum_col="region", max_per_stratum=1)}[col]
    with pytest.raises(MlsynthDataError, match="constant"):
        TBR(cfg(df, **over)).fit()


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_a_market_cost_that_is_not_a_finite_non_negative_number_is_refused(bad):
    df = geo_panel()
    df.loc[df.geo == "m01", "test_cost"] = bad
    with pytest.raises(MlsynthDataError, match="non-negative"):
        cfg(df, market_cost_col="test_cost", max_total_cost=5.0)


def test_a_missing_attribute_column_is_refused():
    with pytest.raises(MlsynthDataError, match="Missing required columns"):
        cfg(geo_panel(), cluster_col="nope")
