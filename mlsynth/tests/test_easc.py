"""Tests for EASC, the exposure-augmented synthetic control (Spruk 2026).

Layer 3 and 4 of the suite: ingestion, the fit, the result contract, the edges
and the failures. The kernels' invariants are asserted generatively in
``test_easc_properties.py``.

The main fixture is the paper's model with an exact answer. Untreated outcomes
are ``a_j + d_t + theta_j x_t + mu_j f_t`` with ``theta_j = gamma s_j`` and
``mu_j = xi_j - gamma rho s_j``, and the shock ``x_t`` equals ``rho f_t``
before treatment. Pre-treatment outcomes therefore carry no trace of exposure
(the silent factor of Corollary 1): synthetic control matches the treated
unit's composite loading exactly and misses its exposure. After treatment the
shock leaves its span by ``r_t``, synthetic control is off by
``gamma EI(w) r_t``, and EASC -- whose regression recovers ``phi_t = gamma r_t``
exactly when the growth controls span the loadings -- returns the effect.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import EASC
from mlsynth.config_models import BaseEstimatorResults, EffectResult
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.easc_helpers import EASCConfig, EASCResults, plot_easc
from mlsynth.utils.easc_helpers import frontier as frontier_module

ROOT = Path(__file__).resolve().parents[2]
IRAN = ROOT / "basedata" / "iran_sanctions_oil.csv"

GAMMA, RHO = 0.01, 1.3


def _panel(*, seed=0, J=14, T=26, T0=15, effect=-0.1, s1=22.0, in_span=False, noise=0.0,
           donor_exposure=None):
    """The silent-factor model above, in long form, with unit 0 treated."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    f = (t - (T0 - 1) / 2) / 10.0
    r = np.where(t >= T0, -0.6 * (t - T0 + 1), 0.0) if not in_span else np.zeros(T)
    x = RHO * f + r
    d = 0.02 * t
    s = np.linspace(0.0, 40.0, J) if donor_exposure is None else np.asarray(donor_exposure, float)
    a = rng.normal(0, 0.3, J)
    xi = rng.normal(0, 0.08, J)
    # The treated unit is a convex combination of the low-exposure donors in
    # (a, xi), so synthetic control can fit it exactly while missing its exposure.
    low = np.argsort(s)[: max(3, J // 3)]
    w_star = np.zeros(J)
    w_star[low] = rng.dirichlet(np.ones(low.size))
    a1, xi1 = float(a @ w_star), float(xi @ w_star)
    rows = []
    for j in range(J + 1):
        if j == 0:
            aj, xij, sj, name = a1, xi1, s1, "T"
        else:
            aj, xij, sj, name = a[j - 1], xi[j - 1], s[j - 1], f"D{j:02d}"
        theta, mu = GAMMA * sj, xij - GAMMA * RHO * sj
        y = aj + d + theta * x + mu * f + noise * rng.standard_normal(T)
        if j == 0:
            y = y + np.where(t >= T0, effect, 0.0)
        for tt in range(T):
            rows.append({"unit": name, "year": 2000 + tt, "y": float(y[tt]),
                         "treated": int(j == 0 and tt >= T0), "exposure": float(sj),
                         "shock": float(x[tt])})
    return pd.DataFrame(rows), dict(gamma=GAMMA, r=r, T0=T0, effect=effect)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treated", unitid="unit", time="year",
                exposure="exposure", shock="shock", display_graphs=False)
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def exact():
    df, truth = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = EASC(_cfg(df)).fit()
    return df, truth, res


@pytest.fixture(scope="module")
def noisy():
    df, truth = _panel(seed=3, noise=0.01)
    return df, truth, EASC(_cfg(df)).fit()


# ---------------------------------------------------------------------------
# smoke and contract
# ---------------------------------------------------------------------------

def test_fit_returns_an_effect_result_with_the_shared_submodels(noisy):
    _, _, res = noisy
    assert isinstance(res, EASCResults)
    assert isinstance(res, BaseEstimatorResults) and isinstance(res, EffectResult)
    assert np.isfinite(res.att)
    assert res.method_details.method_name == "EASC"
    assert res.time_series.counterfactual_outcome.shape == res.time_series.observed_outcome.shape
    w = res.weights.donor_weights
    assert set(w) == set(res.inputs.donor_names)
    assert sum(w.values()) == pytest.approx(1.0)
    assert min(w.values()) >= 0.0
    assert isinstance(res.pre_rmse, float)
    lo, hi = res.att_ci
    assert lo <= res.att <= hi
    assert 0.0 < res.inference.p_value <= 1.0
    assert res.inference.confidence_level == pytest.approx(0.95)


def test_result_is_frozen(noisy):
    _, _, res = noisy
    with pytest.raises(ValidationError):
        res.method = "sc"


def test_serialises_through_pydantic(noisy):
    _, _, res = noisy
    dumped = res.model_dump(include={"effects", "fit_diagnostics", "inference"})
    assert dumped["effects"]["att"] == pytest.approx(res.att)


def test_four_fits_share_one_panel_and_the_headline_is_easc(noisy):
    _, _, res = noisy
    assert set(res.fits) == {"sc", "easc", "ebsc_min", "easc_tuned"}
    for fit in res.fits.values():
        assert fit.weights.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(fit.gap, res.inputs.y - fit.counterfactual)
    assert res.att == pytest.approx(res.fits["easc"].att)
    np.testing.assert_allclose(res.fits["easc"].weights, res.fits["sc"].weights)
    # EASC is SC minus phi_t EI(w) (equation 9).
    phi = res.diagnostics.exposure_effect
    ei = res.fits["sc"].exposure_imbalance
    np.testing.assert_allclose(res.fits["easc"].gap, res.fits["sc"].gap - phi * ei, atol=1e-12)


def test_ebsc_min_attains_the_smallest_imbalance(noisy):
    _, _, res = noisy
    assert abs(res.fits["ebsc_min"].exposure_imbalance) == pytest.approx(res.diagnostics.d_min, abs=1e-5)
    assert not res.fits["ebsc_min"].corrected and not res.fits["sc"].corrected
    assert res.fits["easc"].corrected and res.fits["easc_tuned"].corrected


def test_tuned_fit_sits_at_the_selected_frontier_point(noisy):
    _, _, res = noisy
    fr = res.frontier
    k = fr.selected
    assert fr.criterion[k] == pytest.approx(fr.criterion.min())
    np.testing.assert_allclose(res.fits["easc_tuned"].weights, fr.weights[k])
    np.testing.assert_allclose(fr.weights[-1], res.fits["sc"].weights)


@pytest.mark.parametrize("method", ["ebsc_min", "easc_tuned"])
def test_method_chooses_the_headline_fit(method):
    df, _ = _panel(seed=3, noise=0.01)
    res = EASC(_cfg(df, method=method, inference=False)).fit()
    assert res.att == pytest.approx(res.fits[method].att)
    assert res.weights.donor_weights == pytest.approx(
        dict(zip(map(str, res.inputs.donor_names), res.fits[method].weights)))
    assert res.inference is None
    assert res.method_details.parameters_used["method"] == method


# ---------------------------------------------------------------------------
# recovery: the exact model
# ---------------------------------------------------------------------------

def test_easc_recovers_the_effect_that_synthetic_control_misses(exact):
    _, truth, res = exact
    post = np.arange(res.inputs.T) >= truth["T0"]
    sc, ea = res.fits["sc"], res.fits["easc"]
    np.testing.assert_allclose(ea.gap[post], truth["effect"], atol=1e-5)
    # Synthetic control fits the pre-period and misses gamma EI(w) r_t afterwards.
    assert sc.pre_rmse < 1e-5
    assert abs(sc.exposure_imbalance) > 1.0
    np.testing.assert_allclose(sc.gap[post] - truth["effect"],
                               truth["gamma"] * sc.exposure_imbalance * truth["r"][post], atol=1e-5)


def test_exposure_regression_recovers_the_out_of_span_component(exact):
    """Proposition 4(a): phi_t = gamma (r_t - r_b), and r_b = 0."""
    _, truth, res = exact
    np.testing.assert_allclose(res.diagnostics.exposure_effect, truth["gamma"] * truth["r"], atol=1e-9)


def test_noiseless_shock_in_the_factor_span_leaves_the_partial_leverage_undefined():
    df, _ = _panel()
    with pytest.warns(UserWarning, match="exact linear function"):
        res = EASC(_cfg(df, inference=False)).fit()
    assert np.isnan(res.diagnostics.partial_leverage).all()


def test_diagnostic_flags_the_silent_factor(noisy):
    """Corollary 1: the shock is in the latent span before treatment (R^2 near 1)
    and far outside it after (|ell_t| well above 2). The noiseless panel puts the
    shock exactly in the span (S_r = 0), where ell_t is undefined; with noise in
    the outcomes the estimated factors leave it a small pre-period residual."""
    _, _, res = noisy
    dg = res.diagnostics
    assert dg.shock_r2 > 0.9
    assert dg.max_partial_leverage_post > 2 * max(dg.max_partial_leverage_pre, 1.0)
    np.testing.assert_allclose(dg.leverage ** 2, dg.latent_leverage ** 2 + dg.partial_leverage ** 2,
                               rtol=1e-6)


def test_in_span_shock_leaves_nothing_to_correct():
    df, _ = _panel(in_span=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = EASC(_cfg(df, inference=False)).fit()
    np.testing.assert_allclose(res.diagnostics.exposure_effect, 0.0, atol=1e-9)
    np.testing.assert_allclose(res.fits["easc"].gap, res.fits["sc"].gap, atol=1e-9)


def test_treated_inside_the_hull_balances_exactly():
    df, _ = _panel(seed=1, noise=0.01)
    res = EASC(_cfg(df, inference=False)).fit()
    assert res.diagnostics.inside_hull
    assert res.diagnostics.d_min == 0.0
    assert abs(res.fits["ebsc_min"].exposure_imbalance) < 1e-5


def test_treated_outside_the_hull_is_reported_and_warned():
    df, _ = _panel(seed=2, noise=0.01, s1=60.0)
    with pytest.warns(UserWarning, match="outside the range of the donors' exposure"):
        res = EASC(_cfg(df, inference=False)).fit()
    dg = res.diagnostics
    assert not dg.inside_hull
    assert dg.d_min == pytest.approx(60.0 - 40.0)
    # Proposition 3(a): EBSC-min puts all its weight on the most exposed donor.
    top = int(np.argmax(res.inputs.exposure_donors))
    assert res.fits["ebsc_min"].weights[top] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def test_inference_rows_cover_both_estimators_and_every_phase(noisy):
    df, _, res = noisy
    years = sorted(df.year.unique())
    phases = {"early": (years[15], years[19]), "late": (years[20], years[-1])}
    res = EASC(_cfg(df, phases=phases, m_grid=[0.0, 0.25, 1.0])).fit()
    ba = res.bias_aware
    assert {(r.estimator, r.phase) for r in ba.rows} == {
        (e, p) for e in ("SC", "EASC") for p in ("post", "early", "late")}
    for row in ba.rows:
        assert set(row.intervals) == {0.0, 0.25, 1.0}
        widths = [row.intervals[M][1] - row.intervals[M][0] for M in (0.0, 0.25, 1.0)]
        assert widths == sorted(widths)
        assert row.breakdown >= 0.0 and row.breakdown_robust >= 0.0
        assert 1 / (res.inputs.n_donors + 1) - 1e-12 <= row.placebo_p_value <= 1.0
        assert row.noise_sd > 0
    post = ba.row("EASC", "post")
    assert res.inference.ci_lower == pytest.approx(post.intervals[0.0][0])
    assert res.inference.p_value == pytest.approx(post.placebo_p_value)
    with pytest.raises(KeyError):
        ba.row("EASC", "nope")


def test_correction_variance_choice(noisy):
    df, _, res = noisy
    other = EASC(_cfg(df, correction_se="phase")).fit()
    a, b = res.bias_aware.row("EASC", "post"), other.bias_aware.row("EASC", "post")
    assert a.estimate == pytest.approx(b.estimate)
    # The mean of per-period standard errors is at least the standard error of the mean.
    assert a.noise_sd >= b.noise_sd - 1e-12
    assert res.bias_aware.row("SC", "post").noise_sd == pytest.approx(other.bias_aware.row("SC", "post").noise_sd)


def test_headline_interval_uses_the_requested_sensitivity(noisy):
    df, _, res = noisy
    wide = EASC(_cfg(df, headline_m=1.0)).fit()
    assert wide.inference.ci_upper - wide.inference.ci_lower > res.inference.ci_upper - res.inference.ci_lower


def test_inference_can_be_switched_off(noisy):
    df, _, _ = noisy
    res = EASC(_cfg(df, inference=False)).fit()
    assert res.bias_aware is None and res.inference is None


def test_phase_leverage_is_reported_without_inference(noisy):
    df, _, _ = noisy
    res = EASC(_cfg(df, inference=False)).fit()
    lev = res.diagnostics.phase_leverage
    assert set(lev) == {"post"}
    sc, ea = lev["post"]
    assert sc > 0 and ea > 0


# ---------------------------------------------------------------------------
# reproduction: Spruk (2026), Section 6 (the durable case is benchmarks/cases)
# ---------------------------------------------------------------------------

NORMALISERS = ["KAZ", "NGA", "ECU", "MEX", "COL", "JOR", "ROU", "BGR", "VNM", "ZAF", "PHL", "THA", "KEN",
               "BRA", "CHL", "PER", "CRI", "DOM", "PAN"]
PHASES = {"shock": (2012, 2015), "relief": (2016, 2017), "siege": (2018, 2024)}


def _iran(pool):
    df = pd.read_csv(IRAN)
    df = df[df.iso.isin(["IRN", *pool])].copy()
    # The Dominican Republic and Panama report no oil rents; the paper sets them to zero.
    df.loc[df.iso.isin(["DOM", "PAN"]), "oilrent"] = 0.0
    df["log_brent"] = np.log(df["brent"])
    cfg = dict(df=df, outcome="lgdppc", treat="sanctions", unitid="iso", time="year",
               exposure="oilrent", exposure_window=(1996, 2011), shock="log_brent",
               phases=PHASES, display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EASC(cfg).fit()


def _phase_means(res, name):
    years = np.asarray(res.inputs.time_labels)
    return [round(float(res.fits[name].gap[(years >= a) & (years <= b)].mean()), 3) for a, b in PHASES.values()]


@pytest.mark.skipif(not IRAN.exists(), reason="basedata/iran_sanctions_oil.csv not present")
def test_reproduces_tables_4_and_5():
    base = _iran(NORMALISERS)
    ext = _iran(NORMALISERS + ["AZE", "AGO"])
    assert _phase_means(base, "sc") == [-0.127, -0.087, -0.089]
    assert _phase_means(base, "easc") == [-0.152, -0.033, 0.086]
    assert _phase_means(ext, "easc") == [-0.122, -0.011, 0.109]
    assert _phase_means(ext, "ebsc_min") == [-0.109, -0.022, 0.054]
    assert _phase_means(ext, "easc_tuned") == [-0.122, -0.011, 0.109]
    assert round(base.diagnostics.d_min, 2) == 6.31
    assert round(base.fits["sc"].exposure_imbalance, 2) == 15.53
    assert round(base.diagnostics.shock_r2, 3) == 0.921
    assert round(base.diagnostics.max_partial_leverage_post, 2) == 10.76
    assert round(base.diagnostics.phase_leverage["siege"][0], 2) == 26.72
    assert round(base.diagnostics.phase_leverage["siege"][1], 2) == 25.29
    sc = base.bias_aware.row("SC", "shock")
    assert (round(sc.placebo_p_value, 3), round(sc.noise_sd, 3), round(sc.breakdown, 2)) == (0.05, 0.049, 0.41)
    assert [round(v, 2) for v in sc.intervals[0.0]] == [-0.22, -0.03]
    assert round(ext.bias_aware.row("EASC", "shock").breakdown, 2) == 0.21
    assert round(ext.bias_aware.row("EASC", "shock").breakdown_robust, 2) == 0.27


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------

def test_duplicated_donor_is_harmless():
    df, _ = _panel(seed=4, noise=0.01)
    dup = df[df.unit == "D05"].assign(unit="D05b")
    res = EASC(_cfg(pd.concat([df, dup]), inference=False)).fit()
    assert res.fits["sc"].weights.sum() == pytest.approx(1.0)
    assert np.isfinite(res.att)


def test_no_growth_controls():
    df, _ = _panel(seed=5, noise=0.01)
    res = EASC(_cfg(df, n_subperiods=0)).fit()
    assert np.isfinite(res.att)
    assert res.diagnostics.hat_values.sum() == pytest.approx(2.0)


def test_base_period_can_be_moved_earlier():
    df, _ = _panel(seed=5, noise=0.01)
    years = sorted(df.year.unique())
    res = EASC(_cfg(df, base_period=years[12], inference=False)).fit()
    assert res.inputs.base_index == 12


def test_exposure_window_averages_the_index(noisy):
    df, _, _ = noisy
    years = sorted(df.year.unique())
    df = df.copy()
    df.loc[(df.unit == "T") & (df.year == years[0]), "exposure"] += 10.0
    res = EASC(_cfg(df, exposure_window=(years[1], years[5]), inference=False)).fit()
    assert res.inputs.exposure_treated == pytest.approx(22.0)
    res2 = EASC(_cfg(df, exposure_window=(years[0], years[4]), inference=False)).fit()
    assert res2.inputs.exposure_treated == pytest.approx(22.0 + 10.0 / 5)


def test_shock_that_is_an_exact_function_of_the_factors_warns_and_still_fits():
    df, _ = _panel(seed=6, noise=0.01)
    df = df.copy()
    df["shock"] = 1.0  # constant: in the span of the factors' constant
    with pytest.warns(UserWarning, match="exact linear function"):
        res = EASC(_cfg(df)).fit()
    assert np.isnan(res.diagnostics.partial_leverage).all()
    assert np.isfinite(res.bias_aware.row("EASC", "post").leverage)


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"n_factors": 0}, {"n_subperiods": -1}, {"frontier_points": 1}, {"alpha": 0.0}, {"alpha": 1.0},
    {"m_grid": []}, {"m_grid": [0.0, -1.0]}, {"headline_m": -0.5},
    {"phases": {}}, {"phases": {"post": (2015, 2020)}}, {"phases": {"a": (2020, 2015)}},
    {"exposure_window": (2010, 2005)},
])
def test_invalid_settings_raise_config_errors(bad):
    df, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        EASCConfig(**_cfg(df, **bad))


def test_labels_that_cannot_be_ordered_are_left_to_the_panel_check():
    """A phase like ("2015", 2020) cannot be ordered, so the config accepts it and
    the fit names the bound that is not a period of the panel."""
    df, _ = _panel()
    cfg = EASCConfig(**_cfg(df, phases={"x": ("2015", 2020)}))
    with pytest.raises(MlsynthDataError, match="not a period"):
        EASC(cfg).fit()


def test_unknown_setting_is_refused():
    df, _ = _panel()
    with pytest.raises(ValidationError):
        EASCConfig(**_cfg(df, exposure_index="exposure"))


@pytest.mark.parametrize("col", ["exposure", "shock"])
def test_missing_column_is_a_data_error(col):
    df, _ = _panel()
    with pytest.raises(MlsynthDataError, match=col):
        EASC(_cfg(df.drop(columns=[col])))


def test_shock_varying_across_units_is_refused():
    df, _ = _panel()
    df = df.copy()
    df.loc[df.unit == "D03", "shock"] += 1.0
    with pytest.raises(MlsynthDataError, match="same for every unit"):
        EASC(_cfg(df)).fit()


def test_missing_shock_is_refused():
    df, _ = _panel()
    df = df.copy()
    df.loc[df.year == df.year.max(), "shock"] = np.nan
    with pytest.raises(MlsynthDataError, match="shock"):
        EASC(_cfg(df)).fit()


def test_unit_without_exposure_is_named():
    df, _ = _panel()
    df = df.copy()
    df.loc[df.unit == "D07", "exposure"] = np.nan
    with pytest.raises(MlsynthDataError, match="D07"):
        EASC(_cfg(df)).fit()


def test_exposure_window_after_treatment_is_refused():
    df, _ = _panel()
    years = sorted(df.year.unique())
    with pytest.raises(MlsynthDataError, match="pre-determined"):
        EASC(_cfg(df, exposure_window=(years[0], years[-1]))).fit()


@pytest.mark.parametrize("key", ["exposure_window", "phases", "base_period"])
def test_labels_not_in_the_panel_are_refused(key):
    df, _ = _panel()
    value = {"exposure_window": (1800, 2003), "phases": {"x": (2020, 2099)}, "base_period": 1801}[key]
    with pytest.raises(MlsynthDataError, match="not a period"):
        EASC(_cfg(df, **{key: value})).fit()


def test_phase_in_the_pre_period_is_refused():
    df, _ = _panel()
    with pytest.raises(MlsynthDataError, match="post-treatment"):
        EASC(_cfg(df, phases={"x": (2001, 2003)})).fit()


def test_base_period_after_treatment_is_refused():
    df, _ = _panel()
    with pytest.raises(MlsynthDataError, match="pre-treatment"):
        EASC(_cfg(df, base_period=2020)).fit()


def test_exposure_without_variation_is_refused():
    df, _ = _panel(donor_exposure=np.full(14, 5.0))
    with pytest.raises(MlsynthDataError, match="exposure"):
        EASC(_cfg(df)).fit()


@pytest.mark.parametrize("J, inference, match", [
    (1, False, "donors"),          # single donor: no cross-donor regression
    (5, False, "donors"),          # 5 donors < 2 + 3 + 1
    (6, True, "inference=False"),  # enough to estimate, too few for placebos
])
def test_too_few_donors(J, inference, match):
    df, _ = _panel(J=J, donor_exposure=np.linspace(0, 40, J))
    with pytest.raises(MlsynthDataError, match=match):
        EASC(_cfg(df, inference=inference, n_factors=1)).fit()


def test_too_few_pre_periods_for_the_growth_controls():
    df, _ = _panel(T0=6, T=12)
    with pytest.raises(MlsynthDataError, match="sub-period"):
        EASC(_cfg(df)).fit()


def test_too_many_factors():
    df, _ = _panel()
    with pytest.raises(MlsynthDataError, match="n_factors"):
        EASC(_cfg(df, n_factors=20)).fit()


def test_several_treated_units_are_refused():
    df, _ = _panel()
    df = df.copy()
    df.loc[(df.unit == "D02") & (df.year >= 2015), "treated"] = 1
    with pytest.raises(MlsynthDataError, match="one treated unit"):
        EASC(_cfg(df)).fit()


def test_treatment_in_the_first_period_is_refused():
    df, _ = _panel()
    df = df.copy()
    df.loc[df.unit == "T", "treated"] = 1
    with pytest.raises(MlsynthDataError):
        EASC(_cfg(df)).fit()


def test_solver_failure_is_translated(monkeypatch):
    df, _ = _panel(seed=3, noise=0.01)

    def boom(self, d):
        raise frontier_module.cp.error.SolverError("CLARABEL failed")

    monkeypatch.setattr(frontier_module.BalancedWeights, "_solve", boom)
    with pytest.raises(MlsynthEstimationError, match="frontier"):
        EASC(_cfg(df, inference=False)).fit()


def test_unexpected_numerical_failure_is_translated(monkeypatch):
    df, _ = _panel(seed=3, noise=0.01)
    import mlsynth.utils.easc_helpers.pipeline as pipeline

    def boom(*a, **k):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(pipeline.D, "donor_factors", boom)
    with pytest.raises(MlsynthEstimationError, match="SVD"):
        EASC(_cfg(df, inference=False)).fit()


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def test_plot_returns_its_figure(noisy):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    _, _, res = noisy
    fig = plot_easc(res)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "Partial leverage" in titles and "frontier" in titles


def test_display_and_save_are_applied_by_the_estimator(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    df, _ = _panel(seed=3, noise=0.01)
    out = tmp_path / "easc.png"
    EASC(_cfg(df, inference=False, display_graphs=True, save=str(out))).fit()
    assert shown == [True]
    assert out.exists()


def test_plotting_failure_is_translated(monkeypatch):
    import mlsynth.estimators.easc as est
    from mlsynth.exceptions import MlsynthPlottingError

    monkeypatch.setattr(est, "plot_easc", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no backend")))
    df, _ = _panel(seed=3, noise=0.01)
    with pytest.raises(MlsynthPlottingError, match="no backend"):
        EASC(_cfg(df, inference=False, display_graphs=True)).fit()
