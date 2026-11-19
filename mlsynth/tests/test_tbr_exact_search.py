"""TBR's certified design search, ``search="exact"`` (#965, v1).

``certify_treatment`` (``design/exact/``) returns, for one treatment size, the
design the relative objective ranks first among every design the eligibility and
the business rules allow, with a certificate that no allowed design ranks higher.
``search="exact"`` runs it at every size from the forced geos' count to
``max_treatment_size``, each under its own time limit, and recommends the size
whose design ranks first, as the greedy and the local search do.

The claims are checked against enumeration: every treatment group of the size and
every non-empty control group the rules allow, scored by the library's own
``score_split``. Calibrated inference is the default under the exact search, and
then the search scores the pre-period minus the blank periods the calibration
holds back.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.tbr_helpers.calibration import design_end, resolve_blank_periods
from mlsynth.utils.tbr_helpers.design import certified
from mlsynth.utils.tbr_helpers.design.objective import score_split
from mlsynth.utils.tbr_helpers.design.search import greedy_search, ROLES
from mlsynth.utils.tbr_helpers.design.tolerance import population_distance

ALL = frozenset(ROLES)
SETTINGS = settings(max_examples=12, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def _panel(n_geos=6, n_periods=30, seed=0):
    """Geos of differing size sharing a common factor, periods by geos."""
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    common = 1.0 + 0.6 * rng.normal(size=(n_periods, 1))
    idio = 1.0 + 0.12 * rng.normal(size=(n_periods, n_geos))
    return size * (0.5 * common + 0.5 * idio)


def _long(Y):
    T, N = Y.shape
    return pd.DataFrame([dict(geo=f"g{j:02d}", date=t, Y=float(Y[t, j]))
                         for j in range(N) for t in range(T)])


def _cfg(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=2, n_test=5, objective="relative",
              search="exact")
    kw.update(over)
    return TBRConfig(**kw)


def _labels(Y):
    return [f"g{j:02d}" for j in range(Y.shape[1])]


def _optimum(Y, k, n_test, *, treatable=None, controllable=None, forced=(),
             admits=None):
    """The relative objective's best design with k treated geos, by enumeration:
    (value, treatment, control), or (None, None, None) when no design exists."""
    p = Y.shape[1]
    treatable = set(range(p)) if treatable is None else set(treatable)
    controllable = set(range(p)) if controllable is None else set(controllable)
    best = (None, None, None)
    for T in combinations(sorted(treatable), k):
        if not set(forced) <= set(T):
            continue
        rest = sorted(controllable - set(T))
        for r in range(1, len(rest) + 1):
            for C in combinations(rest, r):
                if admits is not None and not admits(T, C):
                    continue
                s = score_split(Y[:, list(T)].sum(1), Y[:, list(C)].sum(1),
                                objective="relative", n_test=n_test)
                if best[0] is None or s.key[0] > best[0]:
                    best = (s.key[0], set(T), set(C))
    return best


def _idx(units):
    return {int(u[1:]) for u in units}


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_smoke_every_size_is_certified_and_the_best_is_recommended():
    res = TBR(_cfg(_long(_panel(6, 30, 0)), inference="posterior")).fit()
    assert [d.k for d in res.designs] == [1, 2]
    for d in res.designs:
        assert d.search == "exact" and d.certified is True and d.timed_out is False
        assert d.proven_gap == 0.0
        assert np.isfinite(d.objective_value) and d.objective_value > 0
    assert res.recommended.objective_value == max(d.objective_value for d in res.designs)


# --------------------------------------------------------------------------- #
# unit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_size_is_the_enumerated_optimum(seed):
    Y = _panel(6, 30, seed)
    res = TBR(_cfg(_long(Y), inference="posterior")).fit()
    for d in res.designs:
        value, T, C = _optimum(Y, d.k, 5)
        assert (_idx(d.treatment_units), _idx(d.control_units)) == (T, C)
        assert d.objective_value == pytest.approx(value, rel=1e-9)


def test_the_value_is_the_librarys_relative_score_of_the_design():
    Y = _panel(6, 30, 3)
    res = TBR(_cfg(_long(Y), inference="posterior")).fit()
    for d in res.designs:
        T, C = sorted(_idx(d.treatment_units)), sorted(_idx(d.control_units))
        s = score_split(Y[:, T].sum(1), Y[:, C].sum(1), objective="relative", n_test=5)
        assert d.objective_value == s.value and d.detail == s.detail
        assert d.matching_trace == [tuple(float(v) for v in s.key)]
        assert sorted(_idx(d.unassigned_units)) == sorted(set(range(6)) - set(T) - set(C))


def test_calibrated_inference_is_the_default_under_the_exact_search():
    df = _long(_panel(6, 40, 4))
    assert _cfg(df).inference == "calibrated"
    assert _cfg(df, search="greedy").inference == "posterior"
    assert _cfg(df, inference="posterior").inference == "posterior"
    assert _cfg(df, variance="hac").inference == "posterior"


def test_under_calibrated_inference_the_search_scores_the_window_before_the_blanks():
    Y = _panel(6, 40, 5)
    res = TBR(_cfg(_long(Y))).fit()
    n = design_end(Y.shape[0], resolve_blank_periods(None, 5))
    assert res.n_periods_scored == n
    for d in res.designs:
        value, T, C = _optimum(Y[:n], d.k, 5)
        assert (_idx(d.treatment_units), _idx(d.control_units)) == (T, C)
    assert res.calibration is not None


def test_the_exact_design_is_never_ranked_below_the_greedy():
    Y = _panel(8, 30, 6)
    exact = TBR(_cfg(_long(Y), inference="posterior")).fit()
    greedy = TBR(_cfg(_long(Y), inference="posterior", search="greedy")).fit()
    for e, g in zip(exact.designs, greedy.designs):
        assert e.k == g.k and e.objective_value >= g.objective_value


def test_the_rules_bind_the_certified_design():
    """A geo kept out of treatment and a ceiling on the controls: the design is the
    enumerated optimum over the designs both allow."""
    Y = _panel(6, 30, 7)
    free = TBR(_cfg(_long(Y), inference="posterior")).fit().designs[-1]
    out = sorted(_idx(free.treatment_units))[0]
    cap = max(len(free.control_units) - 1, 1)
    res = TBR(_cfg(_long(Y), inference="posterior", not_to_be_treated=[f"g{out:02d}"],
                   max_controls=cap)).fit()
    for d in res.designs:
        value, T, C = _optimum(Y, d.k, 5, treatable=set(range(6)) - {out},
                               admits=lambda T, C: len(C) <= cap)
        assert (_idx(d.treatment_units), _idx(d.control_units)) == (T, C)
        assert out not in _idx(d.treatment_units) and len(d.control_units) <= cap


def test_the_population_tolerance_binds_the_certified_design():
    """At population_tolerance=1 only the treatment group closest to the
    population is admissible at each size, so the certified design treats that
    group, with the best control group for it."""
    Y = _panel(6, 30, 19)
    res = TBR(_cfg(_long(Y), inference="posterior", population_tolerance=1.0)).fit()
    for d in res.designs:
        closest = min(combinations(range(6), d.k), key=lambda T: population_distance(Y, T))
        value, T, C = _optimum(Y, d.k, 5, admits=lambda T, C: set(T) == set(closest))
        assert (_idx(d.treatment_units), _idx(d.control_units)) == (T, C)


def test_a_gap_holds_the_certificate_within_it():
    Y = _panel(8, 30, 8)
    res = TBR(_cfg(_long(Y), inference="posterior", exact_gap=0.5)).fit()
    for d in res.designs:
        value, _, _ = _optimum(Y, d.k, 5)
        assert d.certified and d.proven_gap <= 0.5
        assert value <= (1 + d.proven_gap) * d.objective_value * (1 + 1e-12)


def test_processes_do_not_change_the_designs():
    Y = _panel(8, 30, 9)
    one = TBR(_cfg(_long(Y), inference="posterior", n_jobs=1)).fit()
    two = TBR(_cfg(_long(Y), inference="posterior", n_jobs=2)).fit()
    assert [(d.treatment_units, d.control_units) for d in one.designs] == \
        [(d.treatment_units, d.control_units) for d in two.designs]


@pytest.mark.parametrize("over, want", [
    (dict(), dict(gap=0.0, time_limit=120.0, n_jobs=os.cpu_count() or 1, blank_periods=0)),
    (dict(exact_gap=0.25, exact_time_limit=None, n_jobs=2),
     dict(gap=0.25, time_limit=None, n_jobs=2, blank_periods=0)),
])
def test_the_settings_reach_the_certified_search(monkeypatch, over, want):
    """Each size's search gets the gap, the time limit and the processes the
    config names, every core by default, and no blank periods of its own: the
    pipeline holds them back before the search."""
    seen = []
    real = certified.certify_treatment

    def spy(M, k, **kw):
        seen.append(kw)
        return real(M, k, **kw)

    monkeypatch.setattr(certified, "certify_treatment", spy)
    TBR(_cfg(_long(_panel(6, 30, 21)), inference="posterior", **over)).fit()
    assert len(seen) == 2
    for kw in seen:
        assert {key: kw[key] for key in want} == want


def test_a_time_limit_that_runs_out_reports_an_uncertified_design():
    Y = _panel(10, 30, 10)
    res = TBR(_cfg(_long(Y), inference="posterior", exact_time_limit=1e-9)).fit()
    for d in res.designs:
        assert d.certified is False and d.timed_out is True
        assert len(d.treatment_units) == d.k and d.control_units


@given(seed=st.integers(0, 10_000), k=st.integers(1, 2))
@SETTINGS
def test_property_each_size_is_the_enumerated_optimum(seed, k):
    Y = _panel(6, 30, seed)
    res = TBR(_cfg(_long(Y), inference="posterior", max_treatment_size=k)).fit()
    d = res.designs[-1]
    value, T, C = _optimum(Y, k, 5)
    assert d.k == k and d.certified
    assert (_idx(d.treatment_units), _idx(d.control_units)) == (T, C)
    g = greedy_search(Y, [ALL] * 6, max_treatment_size=k, n_test=5,
                      objective="relative")[-1]
    assert d.objective_value >= g.score.value


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_sizes_below_the_forced_geos_are_not_searched():
    Y = _panel(6, 30, 11)
    res = TBR(_cfg(_long(Y), inference="posterior", to_be_treated=["g00", "g01"])).fit()
    assert [d.k for d in res.designs] == [2]
    value, T, C = _optimum(Y, 2, 5, forced=(0, 1))
    assert (_idx(res.designs[0].treatment_units), _idx(res.designs[0].control_units)) == (T, C)


def test_a_size_below_a_stratum_minimum_is_not_searched():
    """A treated geo from each of two strata: size 1 cannot meet both minimums and
    is skipped, as the greedy skips it; each reported size is the enumerated
    optimum over the groups that meet them."""
    Y = _panel(6, 30, 20)
    df = _long(Y)
    df["st"] = df.geo.map(lambda g: "a" if g in ("g00", "g01", "g02") else "b")
    res = TBR(_cfg(df, inference="posterior", stratum_col="st", min_per_stratum=1,
                   max_treatment_size=3)).fit()
    assert [d.k for d in res.designs] == [2, 3]
    for d in res.designs:
        value, T, C = _optimum(Y, d.k, 5, admits=lambda T, C: bool(set(T) & {0, 1, 2})
                               and bool(set(T) & {3, 4, 5}))
        assert (_idx(d.treatment_units), _idx(d.control_units)) == (T, C)


def test_a_size_the_rules_leave_empty_is_not_reported():
    """Two clusters of two geos with bordering donors excluded: one treated geo
    leaves the other cluster as controls, but two treated geos must come from both
    clusters and ban every control. The exact search reports k=1 alone, as the
    greedy does."""
    Y = _panel(4, 30, 30)
    df = _long(Y)
    df["cl"] = df.geo.map(lambda g: "a" if g in ("g00", "g01") else "b")
    kw = dict(inference="posterior", cluster_col="cl", exclude_bordering_donors=True)
    exact = TBR(_cfg(df, **kw)).fit()
    greedy = TBR(_cfg(df, search="greedy", **kw)).fit()
    assert [d.k for d in exact.designs] == [d.k for d in greedy.designs] == [1]
    neighbour = {0: 1, 1: 0, 2: 3, 3: 2}
    value, T, C = _optimum(Y, 1, 5, admits=lambda T, C: neighbour[T[0]] not in C)
    assert (_idx(exact.designs[0].treatment_units), _idx(exact.designs[0].control_units)) == (T, C)


def test_three_geos_one_forced():
    """Three geos with one forced into treatment: at k=1 the treatment group is the
    forced geo and the control group one or both of the other two."""
    Y = _panel(3, 30, 12)
    res = TBR(_cfg(_long(Y), inference="posterior", max_treatment_size=1,
                   to_be_treated=["g00"])).fit()
    d = res.designs[0]
    value, T, C = _optimum(Y, 1, 5, forced=(0,))
    assert (_idx(d.treatment_units), _idx(d.control_units)) == (T, C)


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("objective", ["reference", "paper"])
def test_the_exact_search_needs_the_relative_objective(objective):
    with pytest.raises(MlsynthConfigError, match="relative"):
        _cfg(_long(_panel(6, 30, 13)), objective=objective)


@pytest.mark.parametrize("setting", [dict(exact_gap=0.1), dict(exact_time_limit=60.0),
                                     dict(n_jobs=2)])
def test_exact_settings_without_the_exact_search_are_refused(setting):
    with pytest.raises(MlsynthConfigError, match="search='exact'"):
        _cfg(_long(_panel(6, 30, 14)), search="greedy", **setting)


@pytest.mark.parametrize("setting", [dict(exact_gap=-0.1), dict(exact_gap=float("inf")),
                                     dict(exact_time_limit=0.0), dict(exact_time_limit=-1.0),
                                     dict(n_jobs=0)])
def test_an_exact_setting_out_of_range_is_refused(setting):
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(_long(_panel(6, 30, 15)), **setting)


def test_a_control_start_has_nothing_to_tune_under_the_exact_search():
    with pytest.raises(MlsynthConfigError, match="control_start"):
        _cfg(_long(_panel(6, 30, 16)), control_start="pool")


def test_a_named_split_has_nothing_to_certify():
    df = _long(_panel(6, 30, 17))
    df["is_t"] = (df.geo == "g00").astype(int)
    df["is_c"] = (df.geo != "g00").astype(int)
    df["post"] = (df.date >= 25).astype(int)
    with pytest.raises(MlsynthConfigError, match="search"):
        TBRConfig(df=df, unitid="geo", time="date", outcome="Y",
                  treatment_col="is_t", control_col="is_c", post_col="post",
                  objective="relative", search="exact")


def test_a_panel_on_which_every_design_fits_exactly_is_reported_as_such():
    """Geos that are multiples of one series: every design fits exactly, has no
    residual scale and so no detectable impact to rank. The search ends and says
    that, not that the eligibility admits nothing."""
    base = 100 + np.random.default_rng(0).normal(size=30).cumsum()
    Y = np.outer(base, [1.0, 2.0, 3.0, 0.5])
    with pytest.raises(MlsynthDataError, match=r"size\(s\) \[1\].*fits the treated aggregate exactly"):
        TBR(_cfg(_long(Y), inference="posterior", max_treatment_size=1,
                 exact_time_limit=30.0)).fit()


def test_rules_no_design_meets_are_reported():
    """Every geo but one kept out of treatment and that one forbidden as well:
    nothing can be treated, and the fit says so instead of returning a design."""
    Y = _panel(4, 30, 18)
    with pytest.raises((MlsynthConfigError, MlsynthDataError)):
        TBR(_cfg(_long(Y), inference="posterior", max_treatment_size=1,
                 not_to_be_treated=_labels(Y))).fit()
