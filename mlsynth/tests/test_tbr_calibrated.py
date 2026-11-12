"""TBR's calibrated inference for the cumulative effect (#1043).

TBR's interval is the conjugate posterior of the static regression
``y_t = a + b x_t``. It prices independent errors on a design fixed in advance,
and on autocorrelated panels after a design search it covers 25-50% at a
nominal 80%. ``inference="calibrated"`` keeps the regression's point
prediction and replaces only the interval: a Dirichlet-process predictive
(Florens and Rolin 2024, eq. 5.13) that blends the posterior with the
regression's standardised errors on honest blank windows, pooled over the real
design and placebo designs, with a linear trend in the squared error against
the distance from the design period.

What these tests pin:

* the default ``inference="posterior"`` is unchanged, against golden values
  ``origin/main`` produced on the GeoLift panel before this change;
* the calibrated band is centred on the same estimate, is a proper interval,
  nests in ``level``, and collapses to the posterior as the prior weight grows;
* the design search runs on the pre-period minus its last B periods, so those
  periods cannot change the chosen design;
* placebo treatment groups have the real group's size, satisfy the design
  rules and exclude the real group;
* every validator raises the translated error;
* on simulated AR(1) panels with a known zero effect the calibrated band covers
  more often than the posterior and near the nominal rate.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.tests.test_tbr_design_constraints import geo_panel
from mlsynth.tests.test_tbrmm import market_panel
from mlsynth.utils.tbr_helpers import calibration as cal
from mlsynth.utils.tbr_helpers.design.pipeline import (build_inputs as design_inputs,
                                                       design_rules)
from mlsynth.utils.tbr_helpers.design.search import (CONTROL, TREATMENT,
                                                     UNASSIGNED)
from mlsynth.utils.tbr_helpers.posterior import (cumulative_posterior,
                                                 fit_pretest, interval)
from mlsynth.utils.tbr_helpers.structures import TBRCalibration

_GEOLIFT = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                        "geolift_test_data.csv")
ALL = frozenset({TREATMENT, CONTROL, UNASSIGNED})

GEOLIFT_TREATED = ["austin", "dallas", "minneapolis"]
GEOLIFT_CONTROLS = ["baltimore", "boston", "chicago", "cincinnati", "cleveland",
                    "columbus", "denver", "honolulu", "jacksonville", "memphis",
                    "new orleans", "portland", "washington"]

# origin/main (aa79c4fc), before calibrated inference existed: the named
# 3-market GeoLift design, post window the last 15 days, level 0.8.
GOLDEN_NAMED = {
    "est": [2.248525402067571, 272.2985163038484, 500.9824679318199,
            743.9990142474708, 944.6073443964542, 1188.6881868327127,
            1367.9958460700354, 1161.8797400943004, 1451.0774868531544,
            1591.8485635107936, 2176.5702456032504, 3060.4247612672334,
            2979.3225498835754, 3105.82939111214, 3096.539994404543],
    "lo": [-232.9233540340338, -63.654025185198805, 87.19569263340958,
           267.7066164060215, 409.8385106533234, 599.0742272910683,
           726.0162987955093, 469.65141000906067, 711.2799575034996,
           809.5834833227791, 1356.5804899807545, 2201.054753162697,
           2080.5482245950216, 2167.8279426159115, 2119.9220225762947],
    "hi": [237.42040483816893, 608.2510577928956, 914.7692432302301,
           1220.29141208892, 1479.3761781395851, 1778.302146374357,
           2009.9753933445616, 1854.10807017954, 2190.875016202809,
           2374.113643698808, 2996.5600012257464, 3919.7947693717697,
           3878.0968751721293, 4043.8308396083685, 4073.1579662327913],
    "ci": (141.32813483841966, 271.5438644155194),
}
# origin/main, the searched GeoLift run: max_treatment_size=2, n_test=15, the
# same post window, level 0.8.
GOLDEN_SEARCHED = {
    "treated": ["milwaukee"],
    "controls": ["austin", "boston", "detroit", "honolulu", "indianapolis",
                 "jacksonville", "kansas city", "las vegas", "new orleans",
                 "new york", "salt lake city", "tucson"],
    "required_impact": 1722.011756142282,
    "n_periods_scored": 90,
    "final": (1204.6023285590672, 169.53599244867746, 2239.668664669457),
}


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _geolift():
    d = pd.read_csv(_GEOLIFT)
    dates = sorted(d.date.unique())
    d["post"] = (d.date >= dates[90]).astype(int)
    d["tr"] = d.location.isin(GEOLIFT_TREATED).astype(int)
    d["ct"] = d.location.isin(GEOLIFT_CONTROLS).astype(int)
    return d


def _geolift_named(**over):
    kw = dict(df=_geolift(), unitid="location", time="date", outcome="Y",
              treatment_col="tr", control_col="ct", post_col="post", level=0.8,
              display_graphs=False)
    kw.update(over)
    return TBRConfig(**kw)


def _geolift_searched(**over):
    kw = dict(df=_geolift()[["location", "date", "Y", "post"]],
              unitid="location", time="date", outcome="Y", max_treatment_size=2,
              n_test=15, post_col="post", level=0.8, display_graphs=False)
    kw.update(over)
    return TBRConfig(**kw)


def _fit(config, *, quiet=True):
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        return TBR(config).fit()


def ar_matrix(seed, n_geos=10, T=72, rho=0.8, noise=3.0):
    """Geos on one common factor with AR(1) idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    size = rng.uniform(0.5, 2.0, n_geos)
    f = 50.0 + rng.normal(size=T).cumsum()
    e = np.zeros((T, n_geos))
    e[0] = rng.normal(size=n_geos) / math.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        e[t] = rho * e[t - 1] + rng.normal(size=n_geos)
    return size * f[:, None] + noise * e


def long_panel(Y, *, n_post=0, treated=(), controls=()):
    """Periods-by-geos matrix to the long frame TBR reads."""
    T, n = Y.shape
    rows = []
    for j in range(n):
        geo = f"g{j:02d}"
        for t in range(T):
            rows.append(dict(geo=geo, date=t, Y=float(Y[t, j]),
                             post=int(t >= T - n_post) if n_post else 0,
                             tr=int(j in treated), ct=int(j in controls)))
    return pd.DataFrame(rows)


def _named_sim(Y, n_post, treated, controls, **over):
    kw = dict(df=long_panel(Y, n_post=n_post, treated=treated,
                            controls=controls),
              unitid="geo", time="date", outcome="Y", treatment_col="tr",
              control_col="ct", post_col="post", level=0.8,
              display_graphs=False, inference="calibrated")
    kw.update(over)
    return TBRConfig(**kw)


def _searched_sim(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=2, n_test=5, display_graphs=False,
              inference="calibrated", n_placebos=8)
    kw.update(over)
    return TBRConfig(**kw)


# --------------------------------------------------------------------------- #
# the default is unchanged
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("explicit", [False, True])
def test_the_default_named_posterior_is_bit_for_bit_main(explicit):
    over = {"inference": "posterior"} if explicit else {}
    res = _fit(_geolift_named(**over))
    c = res.report.cumulative
    np.testing.assert_allclose(c.estimate, GOLDEN_NAMED["est"], rtol=1e-12)
    np.testing.assert_allclose(c.lower, GOLDEN_NAMED["lo"], rtol=1e-12)
    np.testing.assert_allclose(c.upper, GOLDEN_NAMED["hi"], rtol=1e-12)
    inf = res.report.inference
    assert (inf.ci_lower, inf.ci_upper) == pytest.approx(GOLDEN_NAMED["ci"],
                                                         rel=1e-12)
    assert inf.method == "tbr_posterior"
    assert res.calibration is None and res.report.calibration is None


def test_the_default_searched_posterior_is_bit_for_bit_main():
    res = _fit(_geolift_searched())
    assert res.recommended.treatment_units == GOLDEN_SEARCHED["treated"]
    assert res.recommended.control_units == GOLDEN_SEARCHED["controls"]
    assert res.recommended.detail["required_impact"] == pytest.approx(
        GOLDEN_SEARCHED["required_impact"], rel=1e-12)
    assert res.n_periods_scored == GOLDEN_SEARCHED["n_periods_scored"]
    c = res.report.cumulative
    assert (c.estimate[-1], c.lower[-1], c.upper[-1]) == pytest.approx(
        GOLDEN_SEARCHED["final"], rel=1e-12)
    assert "calibrated_required_impact" not in res.power.columns
    assert res.recommended.calibrated_required_impact is None
    assert res.calibration is None


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def geolift_named_pair():
    post = _fit(_geolift_named())
    with pytest.warns(UserWarning, match="named"):
        calib = TBR(_geolift_named(inference="calibrated")).fit()
    return post, calib


def test_named_calibrated_smoke(geolift_named_pair):
    post, calib = geolift_named_pair
    c = calib.report.cumulative
    assert isinstance(calib.calibration, TBRCalibration)
    assert calib.report.calibration == calib.calibration
    assert len(c.lower) == len(c.upper) == 15
    assert np.all(np.isfinite(c.lower)) and np.all(np.isfinite(c.upper))
    k = calib.calibration
    assert k.blank_periods == 45 and k.design_periods == 45 and k.horizon == 15
    assert k.n_placebos == 24 == k.n_placebos_requested
    assert len(k.placebo_treatment_units) == 24
    assert k.honest_split is False
    assert k.calibrated_required_impact is None
    assert len(k.k) == len(k.trend_intercept) == len(k.trend_slope) == 15
    assert calib.report.inference.method == "tbr_calibrated"
    assert calib.report.inference.ci_lower == pytest.approx(c.lower[-1] / 15)
    assert calib.report.inference.ci_upper == pytest.approx(c.upper[-1] / 15)


def test_the_calibrated_band_keeps_the_posterior_centre(geolift_named_pair):
    post, calib = geolift_named_pair
    np.testing.assert_array_equal(calib.report.cumulative.estimate,
                                  post.report.cumulative.estimate)
    np.testing.assert_array_equal(calib.report.cumulative.scale,
                                  post.report.cumulative.scale)
    np.testing.assert_array_equal(calib.report.counterfactual,
                                  post.report.counterfactual)
    # the posterior band is kept beside the calibrated one
    np.testing.assert_allclose(calib.calibration.posterior_lower,
                               post.report.cumulative.lower, rtol=1e-12)
    np.testing.assert_allclose(calib.calibration.posterior_upper,
                               post.report.cumulative.upper, rtol=1e-12)


def test_the_calibrated_band_is_a_proper_interval_at_every_horizon(
        geolift_named_pair):
    _, calib = geolift_named_pair
    c = calib.report.cumulative
    assert np.all(np.asarray(c.lower) < np.asarray(c.upper))
    # on this panel the calibration widens the final band (study: [-200, +5181])
    assert c.lower[-1] == pytest.approx(-200, abs=400)
    assert c.upper[-1] == pytest.approx(5181, abs=400)
    assert c.upper[-1] - c.lower[-1] > (calib.calibration.posterior_upper[-1]
                                        - calib.calibration.posterior_lower[-1])


def test_searched_calibrated_analysis_smoke():
    res = _fit(_geolift_searched(inference="calibrated", n_placebos=8))
    k = res.calibration
    assert k.honest_split is True and k.blank_periods == 45
    assert res.n_periods_scored == 90 - 45
    assert res.report.calibration == k
    assert k.n_placebos == 8
    # every placebo has the recommended group's size and is not it
    for g in k.placebo_treatment_units:
        assert len(g) == len(res.recommended.treatment_units)
        assert sorted(g) != sorted(res.recommended.treatment_units)
    # the calibrated MDE is k at the planned test length times the reference
    k_test = k.k[15 - 1]
    for d in res.designs:
        assert d.calibrated_required_impact == pytest.approx(
            k_test * d.detail["required_impact"], rel=1e-12)
    assert k.calibrated_required_impact == \
        res.recommended.calibrated_required_impact
    np.testing.assert_allclose(res.power["calibrated_required_impact"],
                               [d.calibrated_required_impact for d in res.designs])
    # the ranking does not change: the recommendation is the argmax of the key
    best = max(res.designs, key=lambda d: tuple(d.matching_trace[-1]))
    assert best.treatment_units == res.recommended.treatment_units


def test_searched_calibrated_design_only_smoke():
    df = market_panel(n_units=10, T=60, seed=3)
    res = _fit(_searched_sim(df))
    assert res.report is None
    k = res.calibration
    assert isinstance(k, TBRCalibration)
    assert k.blank_periods == 15 and k.design_periods == 45 and k.horizon == 5
    assert res.n_periods_scored == 45
    assert k.posterior_lower == [] and k.posterior_upper == []
    assert len(k.k) == 5 and all(v > 0 for v in k.k)
    for d in res.designs:
        assert d.calibrated_required_impact == pytest.approx(
            k.k[-1] * d.detail["required_impact"], rel=1e-12)


def test_the_calibration_is_reproducible_and_follows_the_seed():
    df = market_panel(n_units=10, T=60, seed=3)
    a = _fit(_searched_sim(df, calibration_seed=4)).calibration
    b = _fit(_searched_sim(df, calibration_seed=4)).calibration
    c = _fit(_searched_sim(df, calibration_seed=5)).calibration
    assert a == b
    assert a.placebo_treatment_units != c.placebo_treatment_units


def test_named_mode_warns_once_that_the_split_is_not_honest():
    Y = ar_matrix(0)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        res = TBR(_named_sim(Y, 8, (0, 1), tuple(range(2, 10)),
                             n_placebos=8)).fit()
    honest = [w for w in seen if "named" in str(w.message)
              and "blank" in str(w.message)]
    assert len(honest) == 1
    assert res.calibration.honest_split is False


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #
def test_a_heavy_prior_returns_the_posterior_band():
    Y = ar_matrix(1)
    cfg = dict(n_placebos=8)
    post = _fit(_named_sim(Y, 8, (0, 1), tuple(range(2, 10)),
                           inference="posterior"))
    heavy = _fit(_named_sim(Y, 8, (0, 1), tuple(range(2, 10)),
                            calibration_prior_weight=1e12, **cfg))
    np.testing.assert_allclose(heavy.report.cumulative.lower,
                               post.report.cumulative.lower, rtol=1e-7)
    np.testing.assert_allclose(heavy.report.cumulative.upper,
                               post.report.cumulative.upper, rtol=1e-7)


def test_the_band_nests_in_level():
    Y = ar_matrix(2)
    bands = []
    for level in (0.5, 0.8, 0.95):
        res = _fit(_named_sim(Y, 8, (0, 1), tuple(range(2, 10)), level=level,
                              n_placebos=8))
        bands.append((np.asarray(res.report.cumulative.lower),
                      np.asarray(res.report.cumulative.upper)))
    for (lo_a, hi_a), (lo_b, hi_b) in zip(bands, bands[1:]):
        assert np.all(lo_b <= lo_a) and np.all(hi_a <= hi_b)
        assert np.all(lo_b < hi_b)


def test_the_adid_forecast_at_the_experiment_is_tbrs_posterior():
    res = _fit(_geolift_named())
    d = _geolift()
    wide = d.pivot(index="date", columns="location", values="Y").sort_index()
    y = wide[GEOLIFT_TREATED].sum(axis=1).to_numpy(float)
    x = wide[GEOLIFT_CONTROLS].sum(axis=1).to_numpy(float)
    f = cal.adid_forecast(y, x, 90, 15)
    observed = np.cumsum(y[90:105])
    # the effect is a small difference of two sums near 1e5, so the tolerance
    # is relative to the sums
    np.testing.assert_allclose(observed - f.centre,
                               res.report.cumulative.estimate, rtol=0,
                               atol=1e-12 * observed.max())
    np.testing.assert_allclose(f.scale, res.report.cumulative.scale, rtol=1e-9)
    assert f.df == res.report.cumulative.df == 88


def test_a_flat_trend_leaves_the_blank_errors_raw():
    z = np.array([-1.5, 0.2, 0.7, 2.0])
    d = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        cal.rescale_to_experiment(z, d, 1.3, 0.0, 10.0), z)


def test_with_no_prior_weight_the_band_is_the_raw_blank_errors():
    z = np.array([-2.0, -1.0, -0.5, 0.0, 0.3, 0.9, 1.4, 2.2, 3.0, -0.1])
    lo, hi = cal.calibrated_interval(100.0, 10.0, 30, z, 0.0, 10, 0.8)
    pts = np.sort(100.0 + 10.0 * z)
    # the 10% and 90% quantiles of an equal-weight ECDF of ten points
    assert lo == pytest.approx(pts[0], abs=1e-6)
    assert hi == pytest.approx(pts[8], abs=1e-6)


def test_the_trend_is_the_ols_line_with_its_slope_clamped():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    up = np.array([1.0, 1.1, 1.3, 1.2, 1.6])
    alpha, beta = cal.distance_trend(np.repeat(d, 1), np.sqrt(up))
    b_ols, a_ols = np.polyfit(d, up, 1)
    assert (alpha, beta) == pytest.approx((a_ols, b_ols))
    down = up[::-1]
    alpha, beta = cal.distance_trend(d, np.sqrt(down))
    assert beta == 0.0
    assert alpha == pytest.approx(np.polyfit(d, down, 1)[1])


def test_the_honest_split_keeps_the_held_back_periods_out_of_the_search():
    df = market_panel(n_units=10, T=60, seed=3)
    base = _fit(_searched_sim(df))
    shaken = df.copy()
    rng = np.random.default_rng(9)
    late = shaken.date >= 45                    # the last B = 15 periods
    shaken.loc[late, "Y"] = shaken.loc[late, "Y"] * rng.uniform(
        0.2, 3.0, int(late.sum()))
    moved = _fit(_searched_sim(shaken))
    assert [d.treatment_units for d in moved.designs] == \
        [d.treatment_units for d in base.designs]
    assert [d.control_units for d in moved.designs] == \
        [d.control_units for d in base.designs]
    assert [d.detail for d in moved.designs] == [d.detail for d in base.designs]
    # without the split the same perturbation reaches the search
    def plain(frame):
        return _fit(TBRConfig(df=frame, unitid="geo", time="date", outcome="Y",
                              max_treatment_size=2, n_test=5,
                              display_graphs=False))
    a, b = plain(df), plain(shaken)
    assert [d.detail for d in a.designs] != [d.detail for d in b.designs]


def test_placebo_groups_satisfy_the_design_rules():
    df = geo_panel(n_units=12, T=60, seed=2)
    cfg = _searched_sim(df, max_treatment_size=3, cluster_col="dma",
                        stratum_col="region", max_per_stratum=1,
                        exclude_bordering_donors=True, n_test=4)
    res = _fit(cfg)
    Y, units, eligibility = design_inputs(cfg)
    eligibility, rules, _ = design_rules(cfg, units, eligibility)
    real = sorted(res.recommended.treatment_units)
    groups = res.calibration.placebo_treatment_units
    assert len(groups) == res.calibration.n_placebos >= 1
    assert len({tuple(sorted(g)) for g in groups}) == len(groups)
    for g in groups:
        idx = [units.index(u) for u in g]
        assert rules.admits(idx)
        assert len(g) == len(real) and sorted(g) != real
        assert all(TREATMENT in eligibility[j] for j in idx)


def test_k_is_the_rms_of_the_errors_trended_from_the_design_period():
    W = ar_matrix(11)
    blank = cal.calibrate(W[:64], [0, 1], list(range(2, 10)), blank_periods=24,
                          horizon=8, eligibility=[ALL] * 10, rules=None,
                          objective="reference", n_test=8, n_placebos=8, seed=0)
    n = len(blank.designs)
    assert n == 9
    for h in range(1, 9):
        z = blank.z[h - 1]
        dist = np.tile(np.arange(24 - h + 1, dtype=float), n)
        assert z.size == dist.size
        a, b = cal.distance_trend(dist, z)
        assert (blank.alpha[h - 1], blank.beta[h - 1]) == pytest.approx((a, b))
        zt = cal.rescale_to_experiment(z, dist, a, b, 24.0)
        np.testing.assert_allclose(blank.z_tilde[h - 1], zt)
        assert blank.k[h - 1] == pytest.approx(np.sqrt(np.mean(zt ** 2)))
        assert blank.n_eff[h - 1] == n * (24 // h)
    # the real design's errors come first, at distances 0..B-h in order
    T, C = blank.designs[0]
    y, x = W[:64, T].sum(axis=1), W[:64, C].sum(axis=1)
    f = cal.adid_forecast(y, x, 40, 1)
    assert blank.z[0][0] == pytest.approx((y[40] - f.centre[0]) / f.scale[0])
    assert any(b > 0 for b in blank.beta)


def test_placebo_control_groups_obey_the_control_rules():
    df = geo_panel(n_units=12, T=60, seed=2)
    cfg = _searched_sim(df, max_treatment_size=2, cluster_col="dma",
                        exclude_bordering_donors=True, max_controls=6, n_test=4)
    Y, units, eligibility = design_inputs(cfg)
    eligibility, rules, _ = design_rules(cfg, units, eligibility)
    blank = cal.calibrate(Y, [0, 2], [4, 5, 6], blank_periods=12, horizon=4,
                          eligibility=eligibility, rules=rules,
                          objective="reference", n_test=4, n_placebos=12,
                          seed=0)
    assert len(blank.placebos) == 12
    for T, C in blank.placebos:
        assert rules.admits_design(T, C)
        assert not set(C) & rules.banned_controls(T)


def test_placebo_draws_hold_the_forced_geos():
    roles = [frozenset({TREATMENT})] + [ALL] * 7
    groups, n_adm = cal.placebo_groups([0, 3], roles, None, 24, seed=0)
    assert n_adm == 6 and len(groups) == 6
    assert all(0 in g and len(g) == 2 for g in groups)
    assert [0, 3] not in [sorted(g) for g in groups]


def test_placebo_draws_sample_when_enumeration_is_too_large(monkeypatch):
    monkeypatch.setattr(cal, "ENUMERATION_LIMIT", 10)
    roles = [ALL] * 12
    groups, n_adm = cal.placebo_groups([0, 1, 2], roles, None, 20, seed=1)
    assert n_adm is None
    assert len(groups) == 20
    assert len({tuple(g) for g in groups}) == 20
    assert all(len(g) == 3 and g != [0, 1, 2] for g in groups)


def test_sampled_placebo_draws_stop_when_the_rules_admit_too_few(monkeypatch):
    monkeypatch.setattr(cal, "ENUMERATION_LIMIT", 1)
    roles = [ALL] * 6
    rules = cal.DesignRules.build(
        roles, conflict_pairs=[(i, j) for i in range(5) for j in range(i + 1, 5)])
    # only groups using geo 5 with one of 0..4 are admissible: five of them
    groups, n_adm = cal.placebo_groups([0, 5], roles, rules, 24, seed=0)
    assert n_adm is None
    assert sorted(map(tuple, groups)) == [(1, 5), (2, 5), (3, 5), (4, 5)]


def test_the_local_search_also_runs_on_the_design_window():
    df = market_panel(n_units=10, T=60, seed=3)
    base = _fit(_searched_sim(df, search="local", local_restarts=3))
    assert base.n_periods_scored == 45
    shaken = df.copy()
    late = shaken.date >= 45
    shaken.loc[late, "Y"] = shaken.loc[late, "Y"] * 2.5
    moved = _fit(_searched_sim(shaken, search="local", local_restarts=3))
    assert [d.detail for d in moved.designs] == [d.detail for d in base.designs]
    assert all(d.search == "local" for d in moved.designs)


def test_the_calibrated_mde_under_the_paper_objective_is_the_reference_formula():
    from mlsynth.utils.tbr_helpers.design.objective import score_split
    df = market_panel(n_units=10, T=60, seed=3)
    res = _fit(_searched_sim(df, objective="paper"))
    Y, units, _ = design_inputs(_searched_sim(df, objective="paper"))
    W = Y[:45]
    for d in res.designs:
        T = [units.index(u) for u in d.treatment_units]
        C = [units.index(u) for u in d.control_units]
        ref = score_split(W[:, T].sum(axis=1), W[:, C].sum(axis=1),
                          objective="reference", n_test=5)
        assert d.calibrated_required_impact == pytest.approx(
            res.calibration.k[4] * ref.detail["required_impact"], rel=1e-12)
        assert "required_impact" not in d.detail


def test_forced_geos_are_in_every_searched_placebo():
    df = market_panel(n_units=10, T=60, seed=3)
    res = _fit(_searched_sim(df, to_be_treated=["m04"]))
    groups = res.calibration.placebo_treatment_units
    assert groups and all("m04" in g for g in groups)


def test_a_real_group_the_eligibility_cannot_form_has_no_placebos():
    roles = [frozenset({TREATMENT, UNASSIGNED})] + [frozenset({CONTROL})] * 3
    assert cal.placebo_groups([0, 1], roles, None, 8, seed=0) == ([], 0)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_blank_periods_exactly_twice_the_test_length():
    Y = ar_matrix(3)
    res = _fit(_named_sim(Y, 8, (0, 1), tuple(range(2, 10)), blank_periods=16,
                          n_placebos=8))
    assert res.calibration.blank_periods == 16
    assert res.calibration.design_periods == 64 - 16


def test_one_admissible_placebo_is_used_and_warned_about():
    rng = np.random.default_rng(0)
    f = 50 + rng.normal(size=40).cumsum()
    Y = np.column_stack([f + rng.normal(size=40), 2 * f + rng.normal(size=40)])
    with pytest.warns(UserWarning, match="placebo"):
        res = TBR(_named_sim(Y, 4, (0,), (1,), n_placebos=8)).fit()
    assert res.calibration.n_placebos == 1
    assert res.calibration.placebo_treatment_units == [["g01"]]


def test_a_one_period_test():
    Y = ar_matrix(4)
    res = _fit(_named_sim(Y, 1, (0, 1), tuple(range(2, 10)), n_placebos=8))
    c = res.report.cumulative
    assert len(c.lower) == 1 and c.lower[0] < c.upper[0]
    assert res.calibration.blank_periods == 3 and res.calibration.horizon == 1


def test_no_placebo_at_all_calibrates_on_the_real_design_alone():
    W = ar_matrix(5, n_geos=4, T=40)
    roles = [frozenset({TREATMENT, UNASSIGNED})] + [frozenset({CONTROL})] * 3
    with pytest.warns(UserWarning, match="placebo"):
        blank = cal.calibrate(W, [0], [1, 2, 3], blank_periods=12, horizon=4,
                              eligibility=roles, rules=None,
                              objective="reference", n_test=4, n_placebos=8,
                              seed=0)
    assert blank.placebos == [] and len(blank.designs) == 1
    assert blank.n_eff[0] == 12 and blank.n_eff[3] == 3


def test_a_placebo_whose_blank_errors_are_undefined_is_dropped():
    rng = np.random.default_rng(0)
    f = 50 + rng.normal(size=40).cumsum()
    g1 = 2 * f + rng.normal(size=40)
    W = np.column_stack([f + rng.normal(size=40), g1, 3.0 * g1 + 5.0])
    roles = [frozenset({TREATMENT, UNASSIGNED}), frozenset({CONTROL}),
             frozenset({TREATMENT, UNASSIGNED})]
    with pytest.warns(UserWarning, match="placebo"):
        blank = cal.calibrate(W, [0], [1], blank_periods=12, horizon=4,
                              eligibility=roles, rules=None,
                              objective="reference", n_test=4, n_placebos=8,
                              seed=0)
    assert blank.placebos == [] and blank.dropped == 1


def test_a_placebo_with_no_control_group_is_skipped():
    W = ar_matrix(6, n_geos=3, T=40)
    # geo 2 may be treated but no geo other than 1 may be a control, and the
    # placebo {1} leaves none
    roles = [frozenset({TREATMENT, UNASSIGNED}), frozenset({TREATMENT, CONTROL,
                                                           UNASSIGNED}),
             frozenset({TREATMENT, UNASSIGNED})]
    with pytest.warns(UserWarning, match="placebo"):
        blank = cal.calibrate(W, [0], [1], blank_periods=12, horizon=4,
                              eligibility=roles, rules=None,
                              objective="reference", n_test=4, n_placebos=8,
                              seed=0)
    assert [g for g, _ in blank.placebos] == [[2]]


def test_a_placebo_the_control_band_cannot_fit_is_skipped():
    W = ar_matrix(7, n_geos=5, T=40)
    roles = [ALL, frozenset({TREATMENT, UNASSIGNED}), frozenset({CONTROL}),
             frozenset({CONTROL}), frozenset({CONTROL})]
    rules = cal.DesignRules.build(roles, max_controls=2)
    # the placebo {0} leaves three geos that must be controls, above the band
    with pytest.warns(UserWarning, match="placebo"):
        blank = cal.calibrate(W, [1], [2, 3], blank_periods=12, horizon=4,
                              eligibility=roles, rules=rules,
                              objective="reference", n_test=4, n_placebos=8,
                              seed=0)
    assert blank.placebos == []


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,value", [
    ("blank_periods", 30), ("n_placebos", 12),
    ("calibration_prior_weight", 3.0), ("calibration_seed", 2)])
def test_calibration_settings_need_calibrated_inference(field, value):
    with pytest.raises(MlsynthConfigError, match="calibrated"):
        _geolift_named(**{field: value})


def test_an_unknown_inference_is_refused():
    with pytest.raises((ValidationError, MlsynthConfigError)):
        _geolift_named(inference="bootstrap")


@pytest.mark.parametrize("value", [7, 0, -3])
def test_too_few_placebos_are_refused(value):
    with pytest.raises(MlsynthConfigError, match="n_placebos"):
        _geolift_named(inference="calibrated", n_placebos=value)


@pytest.mark.parametrize("value", [-0.5, float("inf"), float("nan")])
def test_a_bad_prior_weight_is_refused(value):
    with pytest.raises(MlsynthConfigError, match="calibration_prior_weight"):
        _geolift_named(inference="calibrated", calibration_prior_weight=value)


def test_a_negative_seed_is_refused():
    with pytest.raises(MlsynthConfigError, match="calibration_seed"):
        _geolift_named(inference="calibrated", calibration_seed=-1)


@pytest.mark.parametrize("value", [1, 0, -4])
def test_fewer_than_two_blank_periods_are_refused(value):
    with pytest.raises(MlsynthConfigError, match="blank_periods"):
        _geolift_named(inference="calibrated", blank_periods=value)


def test_blank_periods_under_twice_n_test_are_refused_at_config():
    df = market_panel(n_units=10, T=60, seed=3)
    with pytest.raises(MlsynthConfigError, match="2 x"):
        _searched_sim(df, blank_periods=9)


def test_blank_periods_under_twice_the_post_window_are_refused_at_fit():
    with pytest.raises(MlsynthDataError, match="2 x"):
        _fit(_geolift_named(inference="calibrated", blank_periods=29))
    with pytest.raises(MlsynthDataError, match="2 x"):
        _fit(_geolift_searched(inference="calibrated", blank_periods=29))


def test_hac_and_calibrated_are_refused_together():
    with pytest.raises(MlsynthConfigError, match="hac"):
        _geolift_named(inference="calibrated", variance="hac")


def test_a_short_pre_period_is_refused():
    Y = ar_matrix(8, T=30)
    with pytest.raises(MlsynthDataError, match="pre-period"):
        _fit(_named_sim(Y, 8, (0, 1), tuple(range(2, 10)), n_placebos=8))


def test_a_real_design_with_undefined_blank_errors_is_refused():
    rng = np.random.default_rng(0)
    x = 50 + rng.normal(size=40).cumsum()
    W = np.column_stack([2.0 * x + 1.0, x, x + rng.normal(size=40)])
    with pytest.raises(MlsynthDataError, match="exactly"):
        cal.calibrate(W, [0], [1], blank_periods=12, horizon=4,
                      eligibility=[ALL] * 3, rules=None, objective="reference",
                      n_test=4, n_placebos=8, seed=0)


def test_a_horizon_with_one_blank_origin_is_refused():
    W = ar_matrix(9, n_geos=4, T=40)
    with pytest.raises(MlsynthDataError, match="origin"):
        cal.calibrate(W, [0], [1, 2], blank_periods=4, horizon=4,
                      eligibility=[ALL] * 4, rules=None, objective="reference",
                      n_test=4, n_placebos=8, seed=0)


def test_a_planned_test_longer_than_the_blank_window_is_refused():
    # the realised post window is 4 periods, so B = 12 passes the 2 x H rule,
    # but the MDE is wanted at n_test = 15 and the blank window holds no
    # 15-period origin
    df = market_panel(n_units=10, T=60, seed=3)
    df["post"] = (df.date >= 56).astype(int)
    with pytest.raises(MlsynthDataError, match="origin"):
        _fit(_searched_sim(df, n_test=15, post_col="post", blank_periods=12))


def test_the_bisection_tolerance_is_a_stated_constant():
    assert 0.0 < cal.QUANTILE_TOL <= 1e-6
    assert cal.MIN_PLACEBOS == 8


# --------------------------------------------------------------------------- #
# known truth
# --------------------------------------------------------------------------- #
def test_calibrated_coverage_beats_the_posterior_on_ar_panels():
    """Forty seeded AR(1) rho = 0.8 panels, no effect, nominal 80%.

    Each panel is ten geos on one random-walk factor with AR(1) noise; the
    named design treats two geos against the other eight over the last eight
    periods, so the true effect is zero and the band covers when it holds 0.
    """
    hits_post, hits_cal = [], []
    for seed in range(40):
        Y = ar_matrix(100 + seed)
        res = _fit(_named_sim(Y, 8, (0, 1), tuple(range(2, 10)), n_placebos=8))
        c = res.report.cumulative
        hits_cal.append(c.lower[-1] <= 0.0 <= c.upper[-1])
        k = res.calibration
        hits_post.append(k.posterior_lower[-1] <= 0.0 <= k.posterior_upper[-1])
    cov_post, cov_cal = np.mean(hits_post), np.mean(hits_cal)
    assert cov_cal > cov_post
    assert 0.6 <= cov_cal <= 0.97
