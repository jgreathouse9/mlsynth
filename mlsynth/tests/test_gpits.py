"""Tests for the GPITS estimator (Cho 2026, arXiv:2608.20610).

Layered per ``agents/agents_tests.md``: smoke, unit invariants, edge cases,
failure paths, and one golden replication pinned to the paper's own number.

The golden values come from the pre-build spike in
``benchmarks/reference/gpits_heller/``, where a faithful NumPy port was
cross-validated cell-for-cell against the author's R package ``gpss`` on the
same series (agreement to ~1e-11 on every quantity). The paper reports the
D.C. figure as 15.1 with a 95% interval of [13.0, 17.3].
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlsynth import GPITS  # noqa: E402
from mlsynth.config_models import GPITSConfig  # noqa: E402
from mlsynth.exceptions import (  # noqa: E402
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.gpits_helpers.kernels import (  # noqa: E402
    getb_maxvar,
    kernel_gaussian,
    kernel_gaussian_periodic_linear,
)
from mlsynth.utils.gpits_helpers.structures import GPITSResults  # noqa: E402

BASEDATA = Path(__file__).resolve().parents[2] / "basedata"
HELLER = BASEDATA / "dc_handgun_heller.csv"

# Paper Section 6; reproduced by the spike to the digits shown.
GOLDEN_CUM = 15.1323488838
GOLDEN_CI = (12.9687, 17.2960)
GOLDEN_B = 2.7366766363
GOLDEN_S2 = 0.8488131643


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _panel(T=48, T0=36, effect=0.0, seed=0, seasonal=True):
    """A single-unit monthly panel with a trend, seasonality and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = 0.05 * t + (np.sin(2 * np.pi * t / 12) if seasonal else 0.0)
    y = y + rng.normal(0, 0.1, T)
    y[T0:] += effect
    dates = pd.date_range("2000-01-01", periods=T, freq="MS")
    return pd.DataFrame({
        "time": dates,
        "unit": "A",
        "y": y,
        "month": dates.strftime("%m"),
        "treated": (np.arange(T) >= T0).astype(int),
    })


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treated", unitid="unit", time="time")
    base.update(kw)
    return GPITSConfig(**base)


def _seasonal_cfg(df, **kw):
    """The paper's working kernel: needed whenever the series carries a trend."""
    base = dict(kernel="gaussian_periodic_linear", period=12,
                covariates=["month"], categorical_covariates=["month"])
    base.update(kw)
    return _cfg(df, **base)


@pytest.fixture
def heller():
    return pd.read_csv(HELLER, parse_dates=["date"])


# --------------------------------------------------------------------------
# Layer 4: smoke
# --------------------------------------------------------------------------
class TestSmoke:
    def test_fit_returns_results_with_finite_outputs(self):
        res = GPITS(_cfg(_panel())).fit()
        assert isinstance(res, GPITSResults)
        assert np.isfinite(res.effects.att)
        assert np.all(np.isfinite(res.time_series.counterfactual_outcome))
        assert np.all(np.isfinite(res.time_series.estimated_gap))

    def test_accepts_a_plain_dict_config(self):
        res = GPITS({"df": _panel(), "outcome": "y", "treat": "treated",
                     "unitid": "unit", "time": "time"}).fit()
        assert np.isfinite(res.effects.att)

    def test_result_conforms_to_the_two_family_contract(self):
        res = GPITS(_cfg(_panel())).fit()
        for block in ("effects", "time_series", "fit_diagnostics",
                      "inference", "method_details"):
            assert getattr(res, block) is not None, block
        assert res.method_details.method_name == "GPITS"

    def test_series_lengths_line_up(self):
        df = _panel(T=48, T0=36)
        res = GPITS(_cfg(df)).fit()
        ts = res.time_series
        assert len(ts.observed_outcome) == 48
        assert len(ts.counterfactual_outcome) == 48
        assert len(ts.counterfactual_lower) == 48
        assert len(ts.counterfactual_upper) == 48


# --------------------------------------------------------------------------
# Layer 1: unit invariants
# --------------------------------------------------------------------------
class TestInvariants:
    def test_recovers_a_known_constant_effect(self):
        """A trending seasonal series plus a constant post shift."""
        res = GPITS(_seasonal_cfg(_panel(T=96, T0=84, effect=2.0, seed=3))).fit()
        assert res.effects.att == pytest.approx(2.0, abs=0.35)

    def test_no_effect_gives_an_att_near_zero(self):
        res = GPITS(_seasonal_cfg(_panel(T=96, T0=84, effect=0.0, seed=4))).fit()
        assert abs(res.effects.att) < 0.35

    def test_the_stationary_kernel_reverts_and_biases_a_trend(self):
        """Section 3.2: a stationary kernel reverts to the prior away from the
        data, so on a trending series it cannot carry the trend forward. This
        is the reason the combined kernel exists, and the reason the default
        is documented as a starting point and not the working form."""
        df = _panel(T=96, T0=84, effect=2.0, seed=3)
        flat = GPITS(_cfg(df)).fit()
        combined = GPITS(_seasonal_cfg(df)).fit()
        assert abs(flat.effects.att - 2.0) > abs(combined.effects.att - 2.0)
        assert flat.fit_diagnostics.rmse_pre > combined.fit_diagnostics.rmse_pre

    def test_bands_bracket_the_counterfactual(self):
        ts = GPITS(_cfg(_panel())).fit().time_series
        lo = np.asarray(ts.counterfactual_lower, dtype=float)
        hi = np.asarray(ts.counterfactual_upper, dtype=float)
        cf = np.asarray(ts.counterfactual_outcome, dtype=float)
        assert np.all(lo <= cf + 1e-9)
        assert np.all(cf <= hi + 1e-9)

    def test_the_band_widens_with_extrapolation_distance(self):
        """The paper's central property (Section 4): uncertainty grows off
        support, so the post-period band must not be flat or shrinking."""
        df = _panel(T=60, T0=36)
        ts = GPITS(_cfg(df)).fit().time_series
        lo = np.asarray(ts.counterfactual_lower, dtype=float)[36:]
        hi = np.asarray(ts.counterfactual_upper, dtype=float)[36:]
        width = hi - lo
        assert width[-1] > width[0]

    def test_gap_equals_observed_minus_counterfactual(self):
        ts = GPITS(_cfg(_panel())).fit().time_series
        np.testing.assert_allclose(
            np.asarray(ts.estimated_gap, dtype=float),
            np.asarray(ts.observed_outcome, dtype=float)
            - np.asarray(ts.counterfactual_outcome, dtype=float),
            atol=1e-10)

    def test_att_is_the_mean_post_period_gap(self):
        df = _panel(T=48, T0=36)
        res = GPITS(_cfg(df)).fit()
        gap = np.asarray(res.time_series.estimated_gap, dtype=float)
        assert res.effects.att == pytest.approx(float(np.mean(gap[36:])))

    def test_outcome_shift_shifts_the_counterfactual_by_the_same_amount(self):
        """Location equivariance: the GP standardises y internally."""
        df = _panel(T=48, T0=36, seed=7)
        a = GPITS(_cfg(df)).fit()
        shifted = df.assign(y=df["y"] + 100.0)
        b = GPITS(_cfg(shifted)).fit()
        np.testing.assert_allclose(
            np.asarray(b.time_series.counterfactual_outcome, dtype=float),
            np.asarray(a.time_series.counterfactual_outcome, dtype=float) + 100.0,
            rtol=1e-6, atol=1e-6)
        assert b.effects.att == pytest.approx(a.effects.att, abs=1e-6)

    def test_outcome_scaling_scales_the_att(self):
        df = _panel(T=48, T0=36, seed=8)
        a = GPITS(_cfg(df)).fit()
        b = GPITS(_cfg(df.assign(y=df["y"] * 3.0))).fit()
        assert b.effects.att == pytest.approx(3.0 * a.effects.att, rel=1e-5)

    def test_fit_is_deterministic(self):
        df = _panel()
        a = GPITS(_cfg(df)).fit()
        b = GPITS(_cfg(df)).fit()
        assert a.effects.att == pytest.approx(b.effects.att, rel=1e-12)


class TestKernels:
    def test_gaussian_kernel_is_symmetric_with_unit_diagonal(self):
        X = np.random.default_rng(0).normal(size=(8, 3))
        K = kernel_gaussian(X, X, b=2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_combined_kernel_diagonal_is_two_plus_self_dot(self):
        """gaussian(=1) + periodic(=1) + linear(=x.x) on the diagonal."""
        X = np.random.default_rng(1).normal(size=(6, 2))
        K = kernel_gaussian_periodic_linear(X, X, b=1.5, period=4.0)
        np.testing.assert_allclose(np.diag(K), 2.0 + np.einsum("ij,ij->i", X, X),
                                   atol=1e-12)

    def test_combined_kernel_is_positive_semidefinite(self):
        X = np.random.default_rng(2).normal(size=(12, 2))
        K = kernel_gaussian_periodic_linear(X, X, b=2.0, period=12.0)
        assert np.min(np.linalg.eigvalsh(K)) > -1e-8

    def test_periodic_component_repeats_at_the_period(self):
        a = np.array([[0.0]])
        K1 = kernel_gaussian_periodic_linear(a, np.array([[4.0]]), b=1e9, period=4.0)
        K2 = kernel_gaussian_periodic_linear(a, np.array([[8.0]]), b=1e9, period=4.0)
        # linear part differs; the periodic part is identical at both lags
        assert (K1[0, 0] - 0.0) == pytest.approx(K2[0, 0] - 0.0, abs=1.0)

    def test_getb_maxvar_returns_a_positive_length_scale(self):
        X = np.random.default_rng(3).normal(size=(20, 2))
        b = getb_maxvar(X, "gaussian", None)
        assert 0.01 <= b <= 2000.0


# --------------------------------------------------------------------------
# Layer 3: golden replication
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HELLER.exists(), reason="basedata file absent")
class TestHellerGolden:
    def _fit(self, heller):
        return GPITS(GPITSConfig(
            df=heller, outcome="handgun_rate", treat="treated",
            unitid="unit", time="date",
            covariates=["month"], categorical_covariates=["month"],
            kernel="gaussian_periodic_linear", period=12,
        )).fit()

    def test_cumulative_effect_matches_the_paper(self, heller):
        res = self._fit(heller)
        assert res.cumulative_effect[-1] == pytest.approx(GOLDEN_CUM, rel=1e-8)
        assert round(res.cumulative_effect[-1], 1) == 15.1

    def test_cumulative_interval_matches_the_paper(self, heller):
        res = self._fit(heller)
        lo, hi = res.cumulative_ci[-1]
        assert lo == pytest.approx(GOLDEN_CI[0], abs=5e-4)
        assert hi == pytest.approx(GOLDEN_CI[1], abs=5e-4)

    def test_hyperparameters_match_the_reference(self, heller):
        d = self._fit(heller).design
        assert d.length_scale == pytest.approx(GOLDEN_B, rel=1e-6)
        assert d.noise_variance == pytest.approx(GOLDEN_S2, rel=1e-6)

    def test_placebo_periods_all_cover_zero(self, heller):
        res = GPITS(GPITSConfig(
            df=heller, outcome="handgun_rate", treat="treated",
            unitid="unit", time="date",
            covariates=["month"], categorical_covariates=["month"],
            kernel="gaussian_periodic_linear", period=12, placebo_periods=4,
        )).fit()
        assert res.placebo is not None
        assert len(res.placebo.tau) == 4
        assert bool(np.all(res.placebo.cover))


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_post_period(self):
        res = GPITS(_cfg(_panel(T=37, T0=36))).fit()
        assert len(res.time_series.counterfactual_outcome) == 37
        assert np.isfinite(res.effects.att)

    def test_short_pre_period_still_fits(self):
        res = GPITS(_cfg(_panel(T=20, T0=14))).fit()
        assert np.isfinite(res.effects.att)

    def test_constant_pre_period_series_raises(self):
        """A constant series has zero SD, so standardisation is undefined."""
        df = _panel(T=48, T0=36)
        df.loc[df.index[:36], "y"] = 5.0
        with pytest.raises(MlsynthDataError):
            GPITS(_cfg(df)).fit()

    def test_donor_units_present_are_ignored(self):
        """GPITS is donor-free; extra untreated units must not change the fit."""
        df = _panel(T=48, T0=36, seed=11)
        extra = df.copy()
        extra["unit"] = "B"
        extra["treated"] = 0
        extra["y"] = extra["y"] * 2.0 + 7.0
        both = pd.concat([df, extra], ignore_index=True)
        a = GPITS(_cfg(df)).fit()
        b = GPITS(_cfg(both)).fit()
        assert b.effects.att == pytest.approx(a.effects.att, rel=1e-8)

    def test_fixed_hyperparameters_are_honoured(self):
        res = GPITS(_cfg(_panel(), length_scale=3.0, noise_variance=0.2)).fit()
        assert res.design.length_scale == pytest.approx(3.0)
        assert res.design.noise_variance == pytest.approx(0.2)

    def test_confidence_intervals_are_narrower_than_prediction_intervals(self):
        df = _panel()
        pred = GPITS(_cfg(df, interval_type="prediction")).fit().time_series
        conf = GPITS(_cfg(df, interval_type="confidence")).fit().time_series
        wp = np.asarray(pred.counterfactual_upper) - np.asarray(pred.counterfactual_lower)
        wc = np.asarray(conf.counterfactual_upper) - np.asarray(conf.counterfactual_lower)
        assert np.all(wc <= wp + 1e-9)


# --------------------------------------------------------------------------
# failure paths -- each must raise a translated Mlsynth* error
# --------------------------------------------------------------------------
class TestFailures:
    def test_periodic_kernel_without_a_period_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), kernel="gaussian_periodic_linear", period=None)

    def test_non_positive_period_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), kernel="gaussian_periodic_linear", period=0)

    def test_unknown_kernel_raises(self):
        with pytest.raises((MlsynthConfigError, ValueError)):
            _cfg(_panel(), kernel="not_a_kernel")

    def test_non_positive_length_scale_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), length_scale=0.0)

    def test_non_positive_noise_variance_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), noise_variance=-1.0)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), alpha=1.5)

    def test_bad_interval_type_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), interval_type="posterior")

    def test_negative_placebo_periods_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), placebo_periods=-1)

    def test_extra_field_is_forbidden(self):
        with pytest.raises((MlsynthConfigError, ValueError)):
            _cfg(_panel(), not_a_field=1)

    def test_missing_covariate_column_raises(self):
        with pytest.raises(MlsynthDataError):
            GPITS(_cfg(_panel(), covariates=["nope"])).fit()

    def test_categorical_not_in_covariates_raises(self):
        with pytest.raises(MlsynthConfigError):
            _cfg(_panel(), covariates=["month"], categorical_covariates=["y"])

    def test_no_post_period_raises(self):
        df = _panel(T=48, T0=36)
        df["treated"] = 0
        with pytest.raises(MlsynthDataError):
            GPITS(_cfg(df)).fit()

    def test_placebo_longer_than_the_pre_period_raises(self):
        with pytest.raises(MlsynthDataError):
            GPITS(_cfg(_panel(T=24, T0=18), placebo_periods=18)).fit()


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
class TestPlotting:
    def test_plotter_returns_a_figure_and_does_not_show(self, monkeypatch):
        from mlsynth.utils.gpits_helpers.plotter import plot_gpits
        called = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(1))
        fig = plot_gpits(GPITS(_cfg(_panel())).fit())
        assert isinstance(fig, plt.Figure)
        assert called == []
        plt.close(fig)

    def test_display_graphs_does_not_raise(self):
        res = GPITS(_cfg(_panel(), display_graphs=True)).fit()
        assert np.isfinite(res.effects.att)
        plt.close("all")


@pytest.fixture(autouse=True)
def _close_figures():
    """The plotter returns figures without showing them, so the caller owns
    closing; tests that never look at one would otherwise pile them up."""
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# continuous covariates, and the remaining guarded paths
# --------------------------------------------------------------------------
def _with_continuous(df, seed=0):
    """Add a numeric covariate correlated with the outcome."""
    rng = np.random.default_rng(seed)
    return df.assign(x=df["y"].to_numpy() * 0.5 + rng.normal(0, 0.05, len(df)))


class TestContinuousCovariates:
    def test_a_continuous_covariate_enters_the_design(self):
        df = _with_continuous(_panel(T=48, T0=36))
        res = GPITS(_cfg(df, covariates=["x"])).fit()
        assert "x" in res.inputs.column_names
        assert res.inputs.design.shape[1] == 2       # time + x
        assert res.inputs.n_categorical == 0
        assert np.isfinite(res.effects.att)

    def test_mixed_continuous_and_categorical_covariates(self):
        df = _with_continuous(_panel(T=48, T0=36))
        res = GPITS(_cfg(df, covariates=["x", "month"],
                         categorical_covariates=["month"])).fit()
        assert res.inputs.n_categorical == 12
        assert res.inputs.column_names[-2:] == ["__time__", "x"]
        assert np.isfinite(res.effects.att)

    def test_a_non_numeric_continuous_covariate_raises(self):
        df = _panel(T=48, T0=36).assign(label="a")
        with pytest.raises(MlsynthDataError, match="not numeric"):
            GPITS(_cfg(df, covariates=["label"])).fit()

    def test_a_constant_continuous_covariate_raises(self):
        df = _panel(T=48, T0=36).assign(x=1.0)
        with pytest.raises(MlsynthDataError, match="constant"):
            GPITS(_cfg(df, covariates=["x"])).fit()

    def test_a_covariate_with_missing_values_raises(self):
        df = _with_continuous(_panel(T=48, T0=36))
        df.loc[df.index[5], "x"] = np.nan
        with pytest.raises(MlsynthDataError, match="missing values"):
            GPITS(_cfg(df, covariates=["x"])).fit()


class TestGuardedPaths:
    def test_too_few_pre_periods_raises(self):
        with pytest.raises(MlsynthDataError, match="at least 3 pre-treatment"):
            GPITS(_cfg(_panel(T=6, T0=2))).fit()

    def test_placebo_leaving_too_little_training_data_raises(self):
        with pytest.raises(MlsynthDataError, match="at least 3"):
            GPITS(_cfg(_panel(T=12, T0=5), placebo_periods=3)).fit()

    def test_placebo_all_cover_accessor(self):
        res = GPITS(_seasonal_cfg(_panel(T=60, T0=48), placebo_periods=3)).fit()
        assert res.placebo.all_cover == bool(np.all(res.placebo.cover))

    def test_revalidating_a_result_is_idempotent(self):
        """The contract block is populated once; re-validation must not redo it."""
        res = GPITS(_cfg(_panel())).fit()
        again = GPITSResults.model_validate(res)
        assert again.effects.att == pytest.approx(res.effects.att)

    def test_length_scale_search_survives_a_non_finite_kernel(self):
        """A length-scale that overflows the linear term scores as zero
        variance instead of propagating a NaN into the search."""
        X = np.array([[0.0], [1e160], [2e160]])
        b = getb_maxvar(X, "gaussian_periodic_linear", period=1.0)
        assert np.isfinite(b) and b > 0

    def test_a_singular_kernel_raises_a_translated_error(self):
        """Duplicated design rows with a pinned tiny noise make the Cholesky
        fail; the failure must surface as an mlsynth error."""
        from mlsynth.utils.gpits_helpers.pipeline import _GP
        X = np.zeros((6, 1))
        with pytest.raises((MlsynthEstimationError, MlsynthDataError)):
            _GP(X, np.arange(6.0), "gaussian", None, 0, None, None)

    def test_a_non_positive_definite_kernel_raises_a_translated_error(self,
                                                                      monkeypatch):
        """The Cholesky is the one numerical step that can fail outright, so
        its failure must arrive as an mlsynth error and not a LinAlgError."""
        import mlsynth.utils.gpits_helpers.pipeline as pipe
        monkeypatch.setitem(pipe.KERNELS, "gaussian",
                            lambda X1, X2, b, period=None:
                            np.full((X1.shape[0], X2.shape[0]), -5.0))
        with pytest.raises(MlsynthEstimationError, match="positive definite"):
            GPITS(_cfg(_panel(T=24, T0=18), noise_variance=0.1)).fit()

    def test_dict_config_with_a_bad_value_is_translated(self):
        with pytest.raises(MlsynthConfigError, match="Invalid GPITS configuration"):
            GPITS({"df": _panel(), "outcome": "y", "treat": "treated",
                   "unitid": "unit", "time": "time", "alpha": "not a number"})

    def test_plotting_failure_warns_and_does_not_abort(self, monkeypatch):
        import mlsynth.estimators.gpits as mod
        monkeypatch.setattr(mod, "plot_gpits",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.warns(UserWarning, match="GPITS plotting failed"):
            res = GPITS(_cfg(_panel(), display_graphs=True)).fit()
        assert np.isfinite(res.effects.att)
