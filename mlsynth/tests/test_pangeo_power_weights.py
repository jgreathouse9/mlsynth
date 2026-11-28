"""PANGEO's power analysis weights pairs as the ATT does.

The program and arm effects (``effects._pair_records``) weight each supergeo pair by the total ``weight_col`` value
of its treated half, or by its treated unit count without one (docs/pangeo.rst, "Power and the minimum detectable
effect": ``w_p = sum_{i in A_p} omega_i / sum_q sum_{i in A_q} omega_i``). The power analysis has to weight the same
way, or its MDE is the MDE of a different estimator from the one reported. Before this fix it always weighted by the
treated unit count.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth import PANGEO
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel
from mlsynth.utils.pangeo_helpers.power import _variance_inflation, compute_pangeo_power


@functools.lru_cache(maxsize=None)
def _fit(weighted: bool = True, Q: int = 2, units_per_arm: int = 6):
    df = make_seasonal_sales_panel(units_per_arm=units_per_arm, arms=("A", "B"), T=60, seed=0, covariates=True)
    cfg = dict(df=df, outcome="sales", arm="arm", unitid="unit", time="time", max_supergeo_size=Q,
               display_graphs=False)
    if weighted:
        cfg["weight_col"] = "population"
    pop = df.groupby("unit")["population"].first().to_dict()
    return PANGEO(cfg).fit(), pop


def _pairs(res):
    return [p for d in res.arm_designs.values() for p in d.pairs]


def _expected_se(pairs, pop, rho, X):
    """The documented program variance, sum_p w_p^2 sigma_p^2 [f(X, rho) + f(T0, rho)], with population w_p."""
    w = np.array([sum(pop[u] for u in p.treatment) for p in pairs], dtype=float)
    w = w / w.sum()
    var = 0.0
    for wp, p in zip(w, pairs):
        r = np.asarray(p.holdout_resid, dtype=float)
        sigma2 = float(r @ r) / max(r.size - 1, 1)
        T0 = int(np.asarray(p.treatment_mean).size)
        var += wp ** 2 * sigma2 * (_variance_inflation(X, rho) + _variance_inflation(T0, rho))
    return math.sqrt(var)


# --- smoke -----------------------------------------------------------------------------------------------------------

def test_smoke_a_population_weighted_fit_reports_finite_power():
    res, _ = _fit()
    pts = res.power.program.points
    assert pts and all(np.isfinite(pt.mde_absolute) and pt.mde_absolute > 0 for pt in pts)


# --- unit: the weights are the ATT's ---------------------------------------------------------------------------------

def test_the_program_se_is_the_documented_population_weighted_variance():
    res, pop = _fit()
    rho = res.power.serial_correlation
    for pt in res.power.program.points:
        assert pt.se == pytest.approx(_expected_se(_pairs(res), pop, rho, pt.post_periods), rel=1e-12)


def test_each_arm_se_is_its_own_population_weighted_variance():
    res, pop = _fit()
    rho = res.power.serial_correlation
    for arm, design in res.arm_designs.items():
        for pt in res.power.arms[arm].points:
            assert pt.se == pytest.approx(_expected_se(design.pairs, pop, rho, pt.post_periods), rel=1e-12)


def test_the_fixture_has_power_to_tell_the_two_weightings_apart():
    """Count weights and population weights give different SEs here, so the tests above can fail."""
    res, pop = _fit()
    pairs = _pairs(res)
    counts = {u: 1.0 for p in pairs for u in p.treatment + p.control}
    rho = res.power.serial_correlation
    X = res.power.program.points[0].post_periods
    assert abs(_expected_se(pairs, pop, rho, X) - _expected_se(pairs, counts, rho, X)) > 1e-6


def test_the_pipeline_passes_the_weight_column():
    res, pop = _fit()
    direct = compute_pangeo_power(res.arm_designs, unit_weights=pop)
    assert [pt.se for pt in direct.program.points] == [pt.se for pt in res.power.program.points]


def test_without_a_weight_column_pairs_are_weighted_by_treated_count():
    res, _ = _fit(weighted=False)
    counts = {u: 1.0 for p in _pairs(res) for u in p.treatment + p.control}
    rho = res.power.serial_correlation
    for pt in res.power.program.points:
        assert pt.se == pytest.approx(_expected_se(_pairs(res), counts, rho, pt.post_periods), rel=1e-12)


def test_n_treated_stays_the_count_of_treated_units():
    res, _ = _fit()
    assert res.power.program.n_treated == sum(len(p.treatment) for p in _pairs(res))


def test_a_dominant_pair_carries_the_program_se():
    res, pop = _fit()
    pairs = _pairs(res)
    big = pairs[0]
    heavy = {u: (v * 1e12 if u in big.treatment else v) for u, v in pop.items()}
    out = compute_pangeo_power(res.arm_designs, unit_weights=heavy)
    rho = out.serial_correlation
    for pt in out.program.points:
        assert pt.se == pytest.approx(_expected_se([big], heavy, rho, pt.post_periods), rel=1e-6)


# --- property: scale and constant weights ----------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scale=st.floats(min_value=1e-6, max_value=1e6), seed=st.integers(0, 10_000))
def test_property_power_is_invariant_to_rescaling_the_weights(scale, seed):
    res, pop = _fit()
    rng = np.random.default_rng(seed)
    w = {u: float(rng.uniform(0.1, 10.0)) for u in pop}
    a = compute_pangeo_power(res.arm_designs, unit_weights=w)
    b = compute_pangeo_power(res.arm_designs, unit_weights={u: v * scale for u, v in w.items()})
    for pa, pb in zip(a.program.points, b.program.points):
        assert pb.se == pytest.approx(pa.se, rel=1e-9)


@settings(max_examples=40, deadline=None)
@given(c=st.floats(min_value=1e-6, max_value=1e6))
def test_property_constant_weights_are_the_unweighted_power(c):
    res, pop = _fit()
    a = compute_pangeo_power(res.arm_designs)
    b = compute_pangeo_power(res.arm_designs, unit_weights={u: c for u in pop})
    for pa, pb in zip(a.program.points, b.program.points):
        assert pb.se == pytest.approx(pa.se, rel=1e-9)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000))
def test_property_the_se_is_the_documented_variance_for_any_positive_weights(seed):
    res, pop = _fit()
    rng = np.random.default_rng(seed)
    w = {u: float(rng.lognormal(0.0, 2.0)) for u in pop}
    out = compute_pangeo_power(res.arm_designs, unit_weights=w)
    for pt in out.program.points:
        assert pt.se == pytest.approx(_expected_se(_pairs(res), w, out.serial_correlation, pt.post_periods),
                                      rel=1e-9)


# --- edge ------------------------------------------------------------------------------------------------------------

def test_pairs_q1_with_weights():
    res, pop = _fit(Q=1)
    rho = res.power.serial_correlation
    for pt in res.power.program.points:
        assert pt.se == pytest.approx(_expected_se(_pairs(res), pop, rho, pt.post_periods), rel=1e-12)


def test_an_arm_of_one_pair_has_that_pairs_se():
    res, pop = _fit(Q=1, units_per_arm=2)
    for arm, design in res.arm_designs.items():
        assert len(design.pairs) == 1
        for pt in res.power.arms[arm].points:
            assert pt.se == pytest.approx(
                _expected_se(design.pairs, pop, res.power.serial_correlation, pt.post_periods), rel=1e-12)


# --- failure ---------------------------------------------------------------------------------------------------------

def test_a_treated_unit_without_a_weight_is_refused_by_name():
    res, pop = _fit()
    missing = _pairs(res)[0].treatment[0]
    w = {u: v for u, v in pop.items() if u != missing}
    with pytest.raises(MlsynthDataError, match=str(missing)):
        compute_pangeo_power(res.arm_designs, unit_weights=w)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_weight_is_refused(bad):
    res, pop = _fit()
    u0 = _pairs(res)[0].treatment[0]
    w = dict(pop)
    w[u0] = bad
    with pytest.raises(MlsynthDataError, match="weight"):
        compute_pangeo_power(res.arm_designs, unit_weights=w)
