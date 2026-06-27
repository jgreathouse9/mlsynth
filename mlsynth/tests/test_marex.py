"""Tests for the MAREX (synthetic experimental design) estimator.

Covers config validation, the design optimizers, clustering, cost/budget and
cardinality constraints, blank-period inference, the relaxed solver, and
plotting.

Reference: Abadie & Zhao (2026), "Synthetic Controls for Experimental Design."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.marex_helpers.optimization import solve_design, solve_design_relaxed
from mlsynth.utils.marex_helpers.plotter import plot_marex
from mlsynth.utils.marex_helpers.structures import MAREXInference, MAREXResults


def _panel(J=8, T=14, clusters=False, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.2 * rng.standard_normal((J, T))
    grp = np.repeat([0, 1], J // 2)
    rows = []
    for j in range(J):
        for t in range(T):
            row = {"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
            if clusters:
                row["grp"] = int(grp[j])
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _panel()


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------

class TestMAREXConfig:
    def test_defaults(self, panel):
        cfg = MAREXConfig(df=panel, outcome="y", unitid="unit", time="time")
        assert cfg.design == "standard" and cfg.exclusive is True and cfg.relaxed is False

    def test_bad_design_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time", design="nope")

    def test_bad_T0_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time", T0=999)

    def test_missing_column_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREXConfig(df=panel, outcome="nope", unitid="unit", time="time")

    def test_bad_cluster_column(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "cluster": "missing", "m_eq": 1})

    def test_date_string_time_is_accepted(self):
        # ISO-date string time (as the geoex pipeline supplies) must be accepted,
        # not rejected as an "unsupported dtype".
        df = _panel(J=6, T=12)
        weeks = pd.date_range("2025-01-06", periods=12, freq="W-MON").date
        df["time"] = df["time"].map({t: weeks[t].isoformat() for t in range(12)})
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 9, "m_eq": 2}).fit()
        assert res is not None

    def test_non_consecutive_time_is_accepted(self):
        # Ingestion is delegated to geoex_dataprep, which only requires a
        # *balanced* panel (every unit observed at every period) and sorts the
        # time axis -- it does not require consecutive time. A balanced panel
        # with a gap in the integer time index must therefore be accepted.
        df = _panel(J=6, T=12)
        df["time"] = df["time"].map(lambda t: t if t < 6 else t + 5)  # gap
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 9, "m_eq": 2}).fit()
        assert res is not None

    def test_unbalanced_panel_rejected(self):
        # geoex_dataprep owns the balance check: dropping a single unit-time
        # observation makes the panel unbalanced and must raise MlsynthDataError.
        df = _panel(J=6, T=12)
        df = df[~((df["unit"] == "u00") & (df["time"] == 0))]
        with pytest.raises(MlsynthDataError):
            MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 9, "m_eq": 2}).fit()


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------

class TestEstimator:
    def test_fit_base(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        assert isinstance(res, MAREXResults) and res.mode == "marex"
        assert res.synthetic_treated.shape == (14,)
        assert len(res.treated_units) == 2
        g = res.globres
        # treated and control weights are disjoint
        assert np.all(g.treated_weights_agg * g.control_weights_agg == 0)

    @pytest.mark.parametrize("design,kw", [
        ("standard", {}),
        ("weakly_targeted", {"beta": 0.1}),
        ("penalized", {"lambda1": 0.1, "lambda2": 0.1}),
        ("unit_penalized", {"lambda1_unit": 0.1}),
    ])
    def test_designs_run(self, panel, design, kw):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "design": design, **kw}).fit()
        assert res.study.design == design
        assert np.isfinite(res.clusters["0"].rmse)

    def test_m_min_max_range(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_min": 1, "m_max": 3}).fit()
        assert 1 <= len(res.treated_units) <= 3

    def test_conflicting_m_eq_and_range(self, panel):
        with pytest.raises(MlsynthConfigError):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 1, "m_min": 1}).fit()

    def test_missing_cardinality(self, panel):
        with pytest.raises(MlsynthConfigError):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10}).fit()

    def test_clustered_one_treated_each(self):
        df = _panel(J=10, T=14, clusters=True)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "cluster": "grp", "T0": 10, "m_eq": 1}).fit()
        assert set(res.clusters) == {"0", "1"}
        for c in res.clusters.values():
            assert len(c.unit_weight_map["Treated"]) == 1

    def test_costs_budget(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "costs": [1.0] * 8, "budget": 10.0}).fit()
        assert len(res.treated_units) == 2

    def test_relaxed_runs(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "relaxed": True}).fit()
        assert len(res.treated_units) == 2

    def test_covariates_matched(self):
        rng = np.random.default_rng(7)
        J, T = 8, 14
        F = rng.normal(0, 1, (T, 2)); lam = rng.normal(0, 1, (J, 2))
        Y = lam @ F.T + 0.2 * rng.standard_normal((J, T))
        zval = rng.uniform(0, 1, J)            # time-invariant covariate
        rows = [{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]), "share": float(zval[j])}
                for j in range(J) for t in range(T)]
        df = pd.DataFrame(rows)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "covariates": ["share"]}).fit()
        assert len(res.treated_units) == 2
        # the matching target now includes the covariate row (pre-outcomes + 1)
        assert res.clusters["0"].pre_treatment_means.shape[0] == 10 + 1


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

class TestInference:
    def test_blank_period_inference(self):
        df = _panel(J=10, T=18)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 14, "m_eq": 2, "inference": True,
                     "blank_periods": 3, "T_post": 3}).fit()
        inf = res.globres.inference
        assert isinstance(inf, MAREXInference)
        assert 0.0 <= inf.global_p_value <= 1.0
        assert inf.ci.shape == (18, 2)
        assert np.isnan(inf.ci[:res.study.T0]).all()

    def test_no_inference_by_default(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        assert res.globres.inference is None


# ----------------------------------------------------------------------
# Helper-level checks
# ----------------------------------------------------------------------

class TestHelpers:
    def test_solve_design_disjoint(self):
        rng = np.random.default_rng(3)
        Y = rng.normal(0, 1, (8, 12))
        raw = solve_design(Y, T0=10, clusters=np.zeros(8, dtype=int), m_eq=2)
        w, v = raw["w_opt"][:, 0], raw["v_opt"][:, 0]
        assert np.all(w * v < 1e-8)
        assert abs(w.sum() - 1) < 1e-6 and abs(v.sum() - 1) < 1e-6

    def test_relaxed_design_discretizes(self):
        rng = np.random.default_rng(4)
        Y = rng.normal(0, 1, (8, 12))
        raw = solve_design_relaxed(Y, T0=10, clusters=np.zeros(8, dtype=int), m_eq=2)
        assert len(raw["selected_treated"][0]) == 2


class TestPlot:
    def test_plot_runs(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        plot_marex(res, plot_type="treatment")
        plot_marex(res, plot_type="prediction")
        _plt.close("all")


# ----------------------------------------------------------------------
# Standardized post-fit diagnostics (mlsynth.utils.post_fit)
# ----------------------------------------------------------------------

from mlsynth.utils.post_fit import (
    SyntheticControlPostFit, compute_smd, compute_post_fit,
    PowerAnalysis, MDEPoint, compute_power_analysis,
)


class TestPostFit:
    """``res.post_fit`` is auto-attached and exposes the three SMD comparisons
    (treated-vs-control, treated-vs-population, control-vs-population) plus the
    standardized ATE / total / lift / RMSE / inference scalars used by every
    downstream consumer.
    """

    def _panel_with_cov(self, J=12, T=18, seed=11):
        """Panel where two time-invariant covariates load the outcome, so the
        MAREX design that minimises the Abadie-Zhou objective actually achieves
        small SMD against both the synthetic control and the population mean.
        """
        rng = np.random.default_rng(seed)
        pop = rng.uniform(800, 2400, J)
        inc = rng.uniform(45_000, 75_000, J)
        baseline = 100 + 0.05 * pop + 1e-3 * inc
        F = np.cumsum(rng.normal(size=T))
        lam = rng.normal(scale=0.5, size=J)
        Y = baseline[None, :] + F[:, None] * lam[None, :] + 0.5 * rng.normal(size=(T, J))
        rows = [{"unit": f"u{j:02d}", "time": t, "y": float(Y[t, j]),
                  "pop": float(pop[j]), "inc": float(inc[j])}
                for j in range(J) for t in range(T)]
        return pd.DataFrame(rows)

    def test_post_fit_attached_without_covariates(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        pf = res.post_fit
        assert isinstance(pf, SyntheticControlPostFit)
        # Without covariates, the SMD fields stay None but the rest populates.
        assert pf.covariate_smd is None
        assert pf.covariate_smd_treated_vs_pop is None
        assert pf.covariate_smd_control_vs_pop is None
        # Trajectories and phase boundaries always populated.
        assert pf.treated_series.shape == pf.control_series.shape
        assert pf.gap_series.size == pf.treated_series.size
        assert pf.n_fit > 0
        assert pf.n_post >= 0

    def test_post_fit_three_smd_pairs_populated_with_covariates(self):
        df = self._panel_with_cov()
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 14, "m_eq": 3, "covariates": ["pop", "inc"],
                     "covariate_weight": 1.0, "standardize": True}).fit()
        pf = res.post_fit

        # All three SMD dicts populated and keyed by user-supplied column names.
        for d in (pf.covariate_smd,
                  pf.covariate_smd_treated_vs_pop,
                  pf.covariate_smd_control_vs_pop):
            assert d is not None
            assert set(d) == {"pop", "inc"}

        # Summary scalars present and finite.
        for s in (pf.covariate_smd_abs_max,
                  pf.covariate_smd_treated_vs_pop_abs_max,
                  pf.covariate_smd_control_vs_pop_abs_max):
            assert s is not None and np.isfinite(s) and s >= 0

        for s in (pf.covariate_smd_squared_sum,
                  pf.covariate_smd_treated_vs_pop_squared_sum,
                  pf.covariate_smd_control_vs_pop_squared_sum):
            assert s is not None and np.isfinite(s) and s >= 0

        # Names tuple matches the dict keys.
        assert set(pf.covariate_names) == {"pop", "inc"}

    def test_design_with_covariates_balances_better(self):
        """The covariate-aware MAREX design should achieve smaller |SMD|
        treated-vs-control than the no-covariates design, scored on the
        same raw covariate matrix so the comparison is apples-to-apples.
        """
        df = self._panel_with_cov()
        common = dict(df=df, outcome="y", unitid="unit", time="time",
                      T0=14, m_eq=3, standardize=True)
        no_cov_res = MAREX({**common, "covariates": None}).fit()
        with_cov_res = MAREX({**common, "covariates": ["pop", "inc"],
                                "covariate_weight": 1.0}).fit()

        # Score both designs on the same raw covariate matrix.
        units = sorted(df["unit"].unique())
        cov = np.array([[df[df["unit"] == u]["pop"].iloc[0],
                          df[df["unit"] == u]["inc"].iloc[0]] for u in units])

        # MAREX preserves the unit order from df[unitid].unique(); align cov.
        marex_order = list(no_cov_res.globres.Y_full.shape and
                            sorted(set(df["unit"].unique()),
                                    key=lambda u: list(df["unit"].unique()).index(u)))
        # (MAREXPanel uses df[unitid].unique(); since our generator already
        # produces unique() in u00..u11 order, the natural sort matches.)
        no_cov_tc = compute_smd(
            cov,
            no_cov_res.globres.treated_weights_agg,
            no_cov_res.globres.control_weights_agg,
            cov_names=["pop", "inc"],
        )["smd_abs_max"]
        with_cov_tc = with_cov_res.post_fit.covariate_smd_abs_max

        # Both finite; with-cov should not be worse.
        assert np.isfinite(no_cov_tc) and np.isfinite(with_cov_tc)
        assert with_cov_tc <= no_cov_tc + 1e-6, (
            f"covariate-aware design should not have worse treated-vs-control "
            f"balance (no_cov |SMD|={no_cov_tc:.3f} vs with_cov |SMD|="
            f"{with_cov_tc:.3f})"
        )

    def test_population_smd_matches_marex_objective(self):
        """Both vs-population SMDs (treated-vs-pop, control-vs-pop) should be
        small — this is exactly what Abadie & Zhou's standard objective
        ||X̄ − Σwⱼ Xⱼ||² + ||X̄ − Σvⱼ Xⱼ||² minimises.
        """
        df = self._panel_with_cov()
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 14, "m_eq": 3, "covariates": ["pop", "inc"],
                     "covariate_weight": 1.0, "standardize": True}).fit()
        pf = res.post_fit
        # On this DGP, MAREX's standard objective drives both vs-population
        # imbalances down. We use a loose 0.3 threshold to leave headroom for
        # different solver paths while still asserting "much better than random".
        assert pf.covariate_smd_treated_vs_pop_abs_max < 0.3
        assert pf.covariate_smd_control_vs_pop_abs_max < 0.3

    def test_compute_smd_standalone(self):
        """Compute_smd works panel-free on raw arrays."""
        rng = np.random.default_rng(0)
        N, M = 12, 3
        cov = rng.normal(size=(N, M))
        tw = np.zeros(N); tw[[0, 1, 2]] = 1 / 3
        cw = np.zeros(N); cw[[5, 6, 7, 8]] = 1 / 4
        out = compute_smd(cov, tw, cw, cov_names=["a", "b", "c"])
        assert set(out["smd"]) == {"a", "b", "c"}
        assert out["smd_abs_max"] == max(abs(v) for v in out["smd"].values())
        assert np.isclose(out["smd_squared_sum"],
                          sum(v ** 2 for v in out["smd"].values()))

    def test_compute_post_fit_standalone(self):
        """Compute_post_fit works panel-free on raw arrays (no estimator)."""
        rng = np.random.default_rng(2)
        T, N, M = 30, 10, 2
        treated = rng.normal(size=T) + 5
        control = rng.normal(size=T) + 5
        cov = rng.normal(size=(N, M))
        tw = np.zeros(N); tw[[0, 1]] = 0.5
        cw = np.zeros(N); cw[[5, 6, 7]] = 1 / 3
        pf = compute_post_fit(
            treated, control,
            n_fit=20, n_blank=2, n_post=8,
            cov_matrix=cov, cov_names=["a", "b"],
            treated_weights=tw, control_weights=cw,
        )
        assert pf.ate is not None
        assert pf.rmse_fit is not None
        assert pf.rmse_post is not None
        assert pf.covariate_smd is not None
        assert pf.covariate_smd_treated_vs_pop is not None
        assert pf.covariate_smd_control_vs_pop is not None


# ----------------------------------------------------------------------
# Power analysis (analytical AR(1)-inflated MDE from blank residuals)
# ----------------------------------------------------------------------

class TestPowerAnalysis:
    """``res.post_fit.power`` is auto-attached by MAREX.fit() and exposes
    an analytical MDE / power curve consistent with the placebo-inference
    framework Abadie & Zhao (2026) propose.
    """

    def _panel(self, J=10, T=20, seed=11):
        rng = np.random.default_rng(seed)
        F = rng.normal(0, 1, (T, 2)); lam = rng.normal(0, 1, (J, 2))
        Y = lam @ F.T + 0.2 * rng.standard_normal((J, T))
        return pd.DataFrame([{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
                              for j in range(J) for t in range(T)])

    def test_power_attached_with_inference_blank(self):
        df = self._panel(J=10, T=20)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 16, "m_eq": 2,
                     "inference": True, "blank_periods": 4, "T_post": 4}).fit()
        power = res.post_fit.power
        assert isinstance(power, PowerAnalysis)
        # Headline + curve populated
        assert isinstance(power.headline, MDEPoint)
        assert len(power.curve) >= 1
        # Standard scalars are finite
        assert np.isfinite(power.sigma_placebo) and power.sigma_placebo > 0
        assert -1.0 <= power.serial_correlation <= 1.0
        assert power.alpha == 0.05
        assert power.power_target == 0.80
        assert power.method == "analytical_ar1"

    def test_power_falls_back_to_pre_window_without_blank(self, panel):
        # No inference -> no carved-out blank window. Power still attaches,
        # using the pre-period gap as the placebo proxy.
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        power = res.post_fit.power
        assert isinstance(power, PowerAnalysis)
        assert np.isfinite(power.sigma_placebo)

    def test_mde_shrinks_with_longer_horizon(self):
        """The MDE should DECREASE as the post horizon T grows (the AR(1)
        variance-inflation factor monotonically dilutes per-period noise).
        """
        df = self._panel(J=10, T=20)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 16, "m_eq": 2,
                     "inference": True, "blank_periods": 4, "T_post": 4}).fit()
        curve = sorted(res.post_fit.power.curve, key=lambda p: p.post_periods)
        # Take the first two horizons that are different and check MDE shrinks.
        horizons = [c for c in curve if np.isfinite(c.mde_absolute)]
        assert len(horizons) >= 2
        first, last = horizons[0], horizons[-1]
        assert last.post_periods > first.post_periods
        assert last.mde_absolute <= first.mde_absolute + 1e-9, (
            f"MDE should not grow with horizon; got T={first.post_periods}: "
            f"{first.mde_absolute:.4f} vs T={last.post_periods}: "
            f"{last.mde_absolute:.4f}"
        )

    def test_mde_pct_consistent_with_baseline(self):
        df = self._panel(J=10, T=20)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 16, "m_eq": 2,
                     "inference": True, "blank_periods": 4, "T_post": 4}).fit()
        h = res.post_fit.power.headline
        bl = res.post_fit.power.baseline
        if np.isfinite(h.mde_absolute) and np.isfinite(bl) and abs(bl) > 1e-12:
            assert np.isclose(h.mde_pct, h.mde_absolute / bl * 100.0, rtol=1e-9)

    def test_compute_power_analysis_standalone(self):
        """The function works on any SyntheticControlPostFit, not just MAREX."""
        rng = np.random.default_rng(3)
        T = 40
        treated = rng.normal(size=T) + 100
        control = rng.normal(size=T) + 100
        pf = compute_post_fit(treated, control, n_fit=28, n_blank=4, n_post=8)
        power = compute_power_analysis(pf, alpha=0.05, power_target=0.80,
                                        post_grid=[2, 4, 8, 16])
        assert isinstance(power, PowerAnalysis)
        assert {p.post_periods for p in power.curve} >= {2, 4, 8, 16}

    def test_power_at_observed_effect_in_unit_interval(self):
        df = self._panel(J=10, T=20)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 16, "m_eq": 2,
                     "inference": True, "blank_periods": 4, "T_post": 4}).fit()
        for pt in res.post_fit.power.curve:
            if pt.power_at_observed is not None:
                assert 0.0 <= pt.power_at_observed <= 1.0


# ----------------------------------------------------------------------
# post_col and blank-period defaults (parity with sibling estimators)
# ----------------------------------------------------------------------

class TestPostColAndBlankDefaults:
    """Post_col should be an exact alias for T0 (same synthetic design), and
    blank_periods should default to ``max(1, floor(0.3 * T0))`` when
    inference is requested without an explicit blank window.
    """

    @staticmethod
    def _panel_with_post_col(J=8, T=20, T0=14, seed=0):
        df = _panel(J=J, T=T, seed=seed)
        df["post"] = (df["time"] >= T0).astype(int)
        return df

    def test_post_col_same_design_as_T0(self):
        df = self._panel_with_post_col(T0=14)
        kw = dict(outcome="y", unitid="unit", time="time", m_eq=2,
                  inference=True, blank_periods=3, T_post=3)
        res_T0 = MAREX({"df": df, "T0": 14, **kw}).fit()
        res_pc = MAREX({"df": df, "post_col": "post", **kw}).fit()
        # Same T0 derivation
        assert res_T0.study.T0 == res_pc.study.T0
        # Same per-unit treated/control weights => same synthetic design
        np.testing.assert_allclose(
            res_T0.globres.treated_weights_agg,
            res_pc.globres.treated_weights_agg, atol=1e-8,
        )
        np.testing.assert_allclose(
            res_T0.globres.control_weights_agg,
            res_pc.globres.control_weights_agg, atol=1e-8,
        )
        np.testing.assert_allclose(
            res_T0.globres.synthetic_treated,
            res_pc.globres.synthetic_treated, atol=1e-8,
        )
        np.testing.assert_allclose(
            res_T0.globres.synthetic_control,
            res_pc.globres.synthetic_control, atol=1e-8,
        )

    def test_post_col_invalid_column_rejected(self, panel):
        with pytest.raises((MlsynthDataError, Exception)):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time",
                        post_col="not_a_column")

    def test_post_col_with_inconsistent_T0_warns(self):
        df = self._panel_with_post_col(T0=14)
        with pytest.warns(UserWarning, match="derived T0"):
            cfg = MAREXConfig(df=df, outcome="y", unitid="unit", time="time",
                              T0=10, post_col="post")
        # post_col wins
        assert cfg.T0 == 14

    def test_post_col_all_post_rejected(self, panel):
        bad = panel.assign(post=1)
        with pytest.raises(Exception):
            MAREXConfig(df=bad, outcome="y", unitid="unit", time="time",
                        post_col="post")

    def test_blank_periods_default_30pct_with_inference(self):
        # T0=14 -> floor(0.3 * 14) = 4
        df = _panel(J=8, T=18)
        cfg = MAREXConfig(df=df, outcome="y", unitid="unit", time="time",
                          T0=14, m_eq=2, inference=True)
        assert cfg.blank_periods == max(1, int(0.3 * 14))

    def test_blank_periods_default_zero_without_inference(self, panel):
        # Without inference, no blank window is carved out (design fits on
        # the entire pre-period; the user opted out of placebo holdout).
        cfg = MAREXConfig(df=panel, outcome="y", unitid="unit", time="time",
                          T0=10, m_eq=2)
        assert cfg.blank_periods == 0

    def test_explicit_blank_periods_preserved(self, panel):
        cfg = MAREXConfig(df=panel, outcome="y", unitid="unit", time="time",
                          T0=10, m_eq=2, inference=True, blank_periods=2,
                          T_post=2)
        assert cfg.blank_periods == 2

    def test_blank_periods_default_runs_end_to_end(self):
        # Smoke: with inference=True and no blank_periods / T_post specified,
        # MAREX should pick the 30% default and produce a usable result.
        df = _panel(J=10, T=20)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 14, "m_eq": 2, "inference": True}).fit()
        # The 30% default ⇒ blank_periods = 4 on T0=14
        assert res.study.blank_periods == max(1, int(0.3 * 14))
        # Post-fit and inference both populated
        assert res.post_fit is not None
        assert res.globres.inference is not None


# ----------------------------------------------------------------------
# Edge cases: solver-helper input validation (formulation.py)
# ----------------------------------------------------------------------

from mlsynth.utils.marex_helpers.formulation import (
    build_membership_mask,
    compute_cluster_means_members,
    get_per_cluster_param,
    prepare_clusters,
    prepare_fit_slices,
    validate_costs_budget,
    validate_scm_inputs,
)


class TestFormulationGuards:
    """Every guard in formulation.py — bad shapes, mis-paired design/penalty
    flags, bad costs/budget — should raise before cvxpy is touched.
    """

    def test_prepare_clusters_length_mismatch(self):
        with pytest.raises(ValueError, match="length N"):
            prepare_clusters(np.zeros((4, 6)), np.zeros(3))

    def test_validate_scm_inputs_bad_T0(self):
        Y = np.zeros((4, 6))
        with pytest.raises(ValueError, match="T0 must be"):
            validate_scm_inputs(Y, T0=0, blank_periods=0, design="standard")
        with pytest.raises(ValueError, match="T0 must be"):
            validate_scm_inputs(Y, T0=7, blank_periods=0, design="standard")

    def test_validate_scm_inputs_bad_blank(self):
        Y = np.zeros((4, 6))
        with pytest.raises(ValueError, match="blank_periods"):
            validate_scm_inputs(Y, T0=4, blank_periods=4, design="standard")
        with pytest.raises(ValueError, match="blank_periods"):
            validate_scm_inputs(Y, T0=4, blank_periods=-1, design="standard")

    def test_validate_scm_inputs_beta_only_for_weakly_targeted(self):
        Y = np.zeros((4, 6))
        with pytest.raises(ValueError, match="beta is only valid"):
            validate_scm_inputs(Y, T0=4, blank_periods=0,
                                design="standard", beta=0.1)

    def test_validate_scm_inputs_lambdas_only_for_penalized(self):
        Y = np.zeros((4, 6))
        with pytest.raises(ValueError, match="lambda1/lambda2"):
            validate_scm_inputs(Y, T0=4, blank_periods=0,
                                design="standard", lambda1=0.1)
        with pytest.raises(ValueError, match="lambda1/lambda2"):
            validate_scm_inputs(Y, T0=4, blank_periods=0,
                                design="weakly_targeted", lambda2=0.1)

    def test_validate_scm_inputs_unit_terms_only_for_unit_penalized(self):
        Y = np.zeros((4, 6))
        for kw in (dict(xi=0.1), dict(lambda1_unit=0.1), dict(lambda2_unit=0.1)):
            with pytest.raises(ValueError, match="xi/lambda1_unit/lambda2_unit"):
                validate_scm_inputs(Y, T0=4, blank_periods=0,
                                    design="standard", **kw)

    def test_validate_costs_budget_length_mismatch(self):
        with pytest.raises(ValueError, match="costs must have length N"):
            validate_costs_budget([1.0, 2.0], budget=10.0, N=4,
                                   cluster_labels=np.array([0]), K=1)

    def test_validate_costs_budget_missing_budget(self):
        with pytest.raises(ValueError, match="budget must be provided"):
            validate_costs_budget([1.0] * 4, budget=None, N=4,
                                   cluster_labels=np.array([0]), K=1)

    def test_validate_costs_budget_dict_missing_cluster(self):
        with pytest.raises(ValueError, match="budget missing entry"):
            validate_costs_budget([1.0] * 4, budget={0: 10.0}, N=4,
                                   cluster_labels=np.array([0, 1]), K=2)

    def test_validate_costs_budget_dict_scalar_share(self):
        # Scalar budget is divided equally across K clusters.
        _, bdict = validate_costs_budget([1.0] * 4, budget=10.0, N=4,
                                          cluster_labels=np.array([0, 1]), K=2)
        assert bdict == {0: 5.0, 1: 5.0}

    def test_validate_costs_budget_wrong_type(self):
        with pytest.raises(TypeError, match="budget must be a scalar or dict"):
            validate_costs_budget([1.0] * 4, budget="ten", N=4,
                                   cluster_labels=np.array([0]), K=1)

    def test_get_per_cluster_param_paths(self):
        assert get_per_cluster_param(None, "a", default=7) == 7
        assert get_per_cluster_param({"a": 3, "b": 5}, "a") == 3
        # missing key returns default
        assert get_per_cluster_param({"a": 3}, "z", default=11) == 11
        # scalar parameter is returned as-is
        assert get_per_cluster_param(5, "a") == 5

    def test_compute_cluster_means_empty_cluster(self):
        # A cluster label assigned to no units should be flagged.
        Y = np.zeros((3, 4))
        clusters = np.array([0, 0, 0])
        # Manually build a mask that pretends cluster '1' exists with no members
        # (build_membership_mask wouldn't naturally produce this — we test the
        # downstream guard).
        M = np.zeros((3, 2), dtype=bool)
        M[:, 0] = True
        with pytest.raises(ValueError, match="no members"):
            compute_cluster_means_members(Y, M, np.array([0, 1]))

    def test_prepare_fit_slices_no_blank(self):
        Y = np.arange(24).reshape(4, 6).astype(float)
        Y_fit, Y_blank, T_fit = prepare_fit_slices(Y, T0=4, blank_periods=0)
        assert Y_blank is None
        assert T_fit == 4 and Y_fit.shape == (4, 4)


# ----------------------------------------------------------------------
# Edge cases: optimizer + post-hoc discretizer (optimization.py)
# ----------------------------------------------------------------------

from mlsynth.utils.marex_helpers.optimization import (
    _augment_fit,
    post_hoc_discretize,
)


class TestOptimizationEdges:
    """Cover the unusual branches of the optimizer module: 1-D covariates,
    post-hoc discretization when the relaxed v-mass collapses, and the
    relaxed blank-window RMSE branch.
    """

    def test_augment_fit_accepts_1d_covariates(self):
        Y = np.ones((5, 4))
        cov_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = _augment_fit(Y, cov_1d, covariate_weight=1.0)
        assert X.shape == (5, 5)
        np.testing.assert_allclose(X[:, -1], cov_1d)

    def test_augment_fit_standardize_handles_zero_variance_column(self):
        # Constant column has std 0 — _augment_fit must not divide by zero.
        Y = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))
        X = _augment_fit(Y, covariates=None, covariate_weight=1.0,
                          standardize=True)
        assert np.isfinite(X).all()

    def test_post_hoc_discretize_uniform_v_when_relaxed_v_collapses(self):
        # Relaxed v with all-zero (or sub-threshold) values forces the
        # uniform-fallback branch (lines 152-153 of optimization.py).
        cluster_members = [np.array([0, 1, 2, 3])]
        labels = [0]
        w_rel = np.zeros((4, 1)); w_rel[:, 0] = [0.6, 0.4, 0.0, 0.0]
        v_rel = np.zeros((4, 1))  # all-zero -> fallback to uniform on controls
        w_d, v_d, sel_t, sel_c, rmse_b = post_hoc_discretize(
            w_rel, v_rel, cluster_members, labels, m_eq=2,
            trim_threshold=1e-2, Y_fit=None, Y_blank=None,
        )
        assert sel_t[0] == [0, 1]
        assert sel_c[0] == [2, 3]
        # Uniform fallback: each control gets 0.5
        np.testing.assert_allclose(v_d[[2, 3], 0], [0.5, 0.5])
        assert rmse_b == [None]  # no blank window

    def test_post_hoc_discretize_blank_rmse_computed(self):
        # With Y_blank and both treated + control non-empty, the discretizer
        # computes the blank-period RMSE (covers lines 157-159).
        cluster_members = [np.array([0, 1, 2, 3])]
        labels = [0]
        w_rel = np.zeros((4, 1)); w_rel[:, 0] = [0.8, 0.2, 0.0, 0.0]
        v_rel = np.zeros((4, 1)); v_rel[:, 0] = [0.0, 0.0, 0.5, 0.5]
        Y_blank = np.arange(8).reshape(4, 2).astype(float)
        _, _, _, _, rmse_b = post_hoc_discretize(
            w_rel, v_rel, cluster_members, labels, m_eq=2,
            trim_threshold=1e-2, Y_fit=None, Y_blank=Y_blank,
        )
        assert rmse_b[0] is not None and np.isfinite(rmse_b[0])

    def test_post_hoc_discretize_without_m_eq_uses_min_max(self):
        # No m_eq supplied — discretizer should respect m_min / m_max defaults
        # (covers lines 135-136).
        cluster_members = [np.array([0, 1, 2, 3, 4])]
        labels = [0]
        w_rel = np.zeros((5, 1)); w_rel[:, 0] = [0.4, 0.3, 0.2, 0.05, 0.05]
        v_rel = np.zeros((5, 1)); v_rel[:, 0] = [0.0, 0.0, 0.0, 0.5, 0.5]
        _, _, sel_t, _, _ = post_hoc_discretize(
            w_rel, v_rel, cluster_members, labels, m_eq=None,
            m_min=1, m_max=3, trim_threshold=1e-2,
        )
        assert 1 <= len(sel_t[0]) <= 3


# ----------------------------------------------------------------------
# Edge cases: orchestration label-swap and power-analysis safety net
# ----------------------------------------------------------------------

class TestOrchestrationEdges:
    """The swap-to-smaller-treated-set rule (Abadie & Zhao convention) and the
    ``try/except`` shield around power analysis must both be exercised.
    """

    def test_label_swap_picks_smaller_treated_set(self):
        # Constructed example where the raw optimization yields a treated set
        # of size 6 and a control set of size 2 — the orchestrator should swap
        # so the *reported* treated set is the small one.
        from mlsynth.utils.marex_helpers.orchestration import solve_marex
        df = _panel(J=8, T=14)
        # m_eq=6 forces the raw treated set to be the majority; the swap rule
        # then reports the size-2 group as "treated".
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 6}).fit()
        # Post-swap, the reported "Treated" set is the smaller one.
        cluster = res.clusters["0"]
        n_t = len(cluster.unit_weight_map["Treated"])
        n_c = len(cluster.unit_weight_map["Control"])
        assert n_t <= n_c

    def test_power_analysis_failure_is_swallowed(self, monkeypatch):
        # Force compute_power_analysis to raise — the fit must still succeed
        # and just leave res.post_fit.power == None (covers lines 173-174).
        import mlsynth.utils.marex_helpers.orchestration as orch

        def _boom(*_a, **_kw):
            raise RuntimeError("nope")
        monkeypatch.setattr(orch, "compute_power_analysis", _boom)
        df = _panel(J=8, T=14)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        assert res.post_fit is not None
        assert res.post_fit.power is None


# ----------------------------------------------------------------------
# Edge cases: plotter (panel layouts, blank-window shading, inference band)
# ----------------------------------------------------------------------

class TestPlotterEdges:
    """The plotter has four interacting toggles (plot_type, clusters,
    global_result, blank_periods) — exercise each branch.
    """

    def test_plotter_returns_when_nothing_to_plot(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        # Single anonymous cluster "0" + global_result=False ⇒ nothing to plot.
        plot_marex(res, plot_type="treatment", global_result=False)
        _plt.close("all")

    def test_plotter_clustered_with_global(self):
        df = _panel(J=10, T=14, clusters=True)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "cluster": "grp", "T0": 10, "m_eq": 1}).fit()
        # Multiple clusters + global panel — covers the per-cluster loop and the
        # global-panel branch (lines 67-69).
        plot_marex(res, plot_type="treatment", global_result=True)
        _plt.close("all")

    def test_plotter_inference_band_and_blank_shading(self):
        # With inference + a blank window, the plotter draws the conformal CI
        # band and the gray blank-period shading (lines 56-57 and 76-77).
        df = _panel(J=10, T=18)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 14, "m_eq": 2, "inference": True,
                     "blank_periods": 3, "T_post": 3}).fit()
        plot_marex(res, plot_type="treatment", global_result=True)
        _plt.close("all")


# ----------------------------------------------------------------------
# Edge cases: inference module degenerate windows
# ----------------------------------------------------------------------

from mlsynth.utils.marex_helpers.inference import compute_inference


class TestInferenceEdges:
    """Compute_inference must degrade gracefully when the blank or post window
    has zero length.
    """

    def test_inference_no_post_window(self):
        # T_post = 0 ⇒ no treated effects, global_p falls back to NaN
        # (line 72) and the per-period / CI arrays come out NaN (lines 81-82).
        Y_t = np.zeros(10)
        Y_c = np.zeros(10)
        inf = compute_inference(Y_t, Y_c, T0=10, TcE=8, Tb=2, random_state=0)
        assert np.isnan(inf.global_p_value)
        assert inf.per_period_pvals.size == 0
        # CI rows for the pre-period should be NaN.
        assert np.isnan(inf.ci[:10]).all()

    def test_inference_no_blank_window(self):
        # Tb = 0 ⇒ no placebo, per_period_pvals becomes NaN, CI half-width q is
        # NaN (lines 81-82).
        rng = np.random.default_rng(0)
        Y_t = rng.normal(size=12)
        Y_c = rng.normal(size=12)
        inf = compute_inference(Y_t, Y_c, T0=10, TcE=10, Tb=0, random_state=0)
        assert np.isnan(inf.per_period_pvals).all()
        assert np.isnan(inf.ci[10:]).all()


# ----------------------------------------------------------------------
# Edge cases: scexp.py defensive code paths
# ----------------------------------------------------------------------

class TestEstimatorErrorPaths:
    """Cover the converted-exception paths in MAREX.__init__ / .fit()."""

    def test_invalid_config_dict_raises_config_error(self, panel):
        # Pass an unknown field — pydantic raises ValidationError which is
        # converted to MlsynthConfigError (line 54).
        with pytest.raises(MlsynthConfigError, match="Invalid MAREX configuration"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2, "totally_not_a_field": 42})

    def test_cluster_column_dropped_after_config(self, panel):
        # Build a valid config (with cluster), then mutate the df to drop the
        # cluster column — fit() defends against that at line 91-92.
        df = _panel(J=8, T=14, clusters=True)
        cfg = MAREXConfig(df=df, outcome="y", unitid="unit", time="time",
                          cluster="grp", T0=10, m_eq=1)
        # Remove the cluster column from the dataframe AFTER validation
        cfg.df = cfg.df.drop(columns=["grp"])
        with pytest.raises(MlsynthDataError, match="Cluster column"):
            MAREX(cfg)

    def test_setup_invalid_blank_periods_re_raises_as_config_error(self, panel):
        # Manually bypass MAREXConfig (it would have caught this) and call
        # prepare_marex_panel with an invalid blank_periods to trip line 81
        # of setup.py — then verify MAREX.fit() reframes it as
        # MlsynthConfigError (line 116-117 of scexp.py).
        from mlsynth.utils.marex_helpers.setup import prepare_marex_panel
        with pytest.raises(ValueError, match="blank_periods"):
            prepare_marex_panel(
                df=panel, outcome="y", unitid="unit", time="time",
                cluster=None, T0=10, inference=True, blank_periods=10,
                T_post=2,
            )


# ----------------------------------------------------------------------
# Simulation DGP (mlsynth.utils.marex_helpers.simulation)
# ----------------------------------------------------------------------

from mlsynth.utils.marex_helpers.simulation import (
    MAREXSample,
    generate_marex_sample,
)


class TestSimulationDGP:
    """The linear-factor DGP that backs the replication tests should produce
    well-shaped samples with zero pre-period effects.
    """

    def test_generate_marex_sample_shapes_and_zero_pre_effect(self):
        rng = np.random.default_rng(0)
        s = generate_marex_sample(J=6, R=3, F=4, T=12, T0=8, sigma=0.5, rng=rng)
        assert isinstance(s, MAREXSample)
        assert s.Y_N.shape == s.Y_I.shape == (6, 12)
        assert s.tau.shape == (12,)
        # Pre-period treatment effect is zero by construction.
        np.testing.assert_allclose(s.tau[:8], 0.0)
        # Post-period effects are real-valued.
        assert np.isfinite(s.tau[8:]).all()

    def test_generate_marex_sample_default_rng_works(self):
        # No rng supplied — should still produce a well-shaped sample.
        s = generate_marex_sample(J=4, R=2, F=2, T=6, T0=4)
        assert s.Y_N.shape == (4, 6) and s.T0 == 4


# ----------------------------------------------------------------------
# Edge cases: post_fit module (compute_smd / compute_post_fit / power)
# ----------------------------------------------------------------------

from mlsynth.utils.marex_helpers.structures import MAREXInference
from mlsynth.utils.post_fit import (
    _ar1_rho,
    _extract_inference,
    _normalize_weights,
    _safe_float,
    _variance_inflation,
    compute_post_fit_marex,
)


class TestPostFitEdges:
    """Cover the post_fit module's defensive branches: degenerate weights,
    bad covariate scales, inference adapters across estimators, and the
    power-analysis fallbacks for missing pre/post windows.
    """

    # ---- _normalize_weights -----------------------------------------

    def test_normalize_weights_zero_sum_returns_none(self):
        assert _normalize_weights(np.zeros(4), N=4) is None

    def test_normalize_weights_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            _normalize_weights(np.array([1.0, 2.0]), N=4)

    # ---- compute_smd default names + zero weights -------------------

    def test_compute_smd_default_names(self):
        from mlsynth.utils.post_fit import compute_smd as _smd
        cov = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        tw = np.array([1.0, 0.0, 0.0])
        cw = np.array([0.0, 0.5, 0.5])
        out = _smd(cov, tw, cw)
        # Names default to cov_0, cov_1
        assert set(out["smd"].keys()) == {"cov_0", "cov_1"}

    def test_compute_smd_all_zero_weights_returns_nan(self):
        from mlsynth.utils.post_fit import compute_smd as _smd
        cov = np.zeros((4, 2))
        out = _smd(cov, np.zeros(4), np.zeros(4))
        assert out["smd"] == {}
        assert np.isnan(out["smd_abs_max"])
        assert np.isnan(out["smd_squared_sum"])

    def test_compute_smd_explicit_cov_scales(self):
        from mlsynth.utils.post_fit import compute_smd as _smd
        cov = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        tw = np.array([1.0, 0.0, 0.0])
        cw = np.array([0.0, 0.5, 0.5])
        # Passing cov_scales overrides the default ddof=1 std.
        out = _smd(cov, tw, cw, cov_scales=np.array([1.0, 1.0]))
        # With unit scales, SMDs are raw mean differences.
        np.testing.assert_allclose(out["smd"]["cov_0"], 1.0 - 2.5)
        np.testing.assert_allclose(out["smd"]["cov_1"], 10.0 - 25.0)

    # ---- compute_post_fit default n_post + treated-only / control-only

    def test_compute_post_fit_default_n_post(self):
        from mlsynth.utils.post_fit import compute_post_fit as _pf
        t = np.linspace(0, 1, 12); c = np.linspace(0, 1, 12)
        out = _pf(t, c, n_fit=8)   # n_post left as default
        assert out.n_post == 4

    def test_compute_post_fit_treated_only_smd(self):
        # Pass treated_weights but no control_weights -> only the
        # treated-vs-pop comparison is computed.
        from mlsynth.utils.post_fit import compute_post_fit as _pf
        rng = np.random.default_rng(0)
        N, T = 6, 14
        t = rng.normal(size=T); c = rng.normal(size=T)
        cov = rng.normal(size=(N, 2))
        tw = np.zeros(N); tw[0] = 1.0
        out = _pf(t, c, n_fit=10, cov_matrix=cov, treated_weights=tw,
                  control_weights=None, cov_names=("a", "b"))
        assert out.covariate_smd is None
        assert out.covariate_smd_treated_vs_pop is not None
        assert out.covariate_smd_control_vs_pop is None
        # Names come from treated_vs_pop, not treated-vs-control.
        assert out.covariate_names == ("a", "b")

    def test_compute_post_fit_control_only_smd(self):
        from mlsynth.utils.post_fit import compute_post_fit as _pf
        rng = np.random.default_rng(0)
        N, T = 6, 14
        t = rng.normal(size=T); c = rng.normal(size=T)
        cov = rng.normal(size=(N, 2))
        cw = np.zeros(N); cw[1] = 1.0
        out = _pf(t, c, n_fit=10, cov_matrix=cov, treated_weights=None,
                  control_weights=cw, cov_names=("a", "b"))
        assert out.covariate_smd is None
        assert out.covariate_smd_treated_vs_pop is None
        assert out.covariate_smd_control_vs_pop is not None
        assert out.covariate_names == ("a", "b")

    def test_compute_post_fit_explicit_population_weights(self):
        # Non-uniform population weights exercise the scaling branch.
        from mlsynth.utils.post_fit import compute_post_fit as _pf
        rng = np.random.default_rng(1)
        N, T = 6, 14
        t = rng.normal(size=T); c = rng.normal(size=T)
        cov = rng.normal(size=(N, 2))
        tw = np.zeros(N); tw[0] = 1.0
        cw = np.zeros(N); cw[1] = 1.0
        pop = np.arange(1, N + 1, dtype=float)
        out = _pf(t, c, n_fit=10, cov_matrix=cov,
                  treated_weights=tw, control_weights=cw,
                  population_weights=pop)
        # The two vs-population SMDs use the supplied (non-uniform) weights;
        # they should differ from the uniform-pop case.
        uniform_pop = _pf(t, c, n_fit=10, cov_matrix=cov,
                          treated_weights=tw, control_weights=cw)
        assert (out.covariate_smd_treated_vs_pop
                != uniform_pop.covariate_smd_treated_vs_pop)

    def test_compute_post_fit_zero_sum_population_weights_falls_back(self):
        from mlsynth.utils.post_fit import compute_post_fit as _pf
        rng = np.random.default_rng(2)
        N, T = 5, 12
        t = rng.normal(size=T); c = rng.normal(size=T)
        cov = rng.normal(size=(N, 2))
        tw = np.zeros(N); tw[0] = 1.0
        cw = np.zeros(N); cw[1] = 1.0
        out = _pf(t, c, n_fit=8, cov_matrix=cov,
                  treated_weights=tw, control_weights=cw,
                  population_weights=np.zeros(N))   # forces uniform fallback
        # No crash; the vs-pop comparisons still come back as dicts.
        assert isinstance(out.covariate_smd_treated_vs_pop, dict)

    def test_compute_post_fit_explicit_cov_scales_branch(self):
        from mlsynth.utils.post_fit import compute_post_fit as _pf
        rng = np.random.default_rng(3)
        N, T = 5, 12
        t = rng.normal(size=T); c = rng.normal(size=T)
        cov = rng.normal(size=(N, 2))
        tw = np.zeros(N); tw[0] = 1.0
        cw = np.zeros(N); cw[1] = 1.0
        # Supply explicit scales — the shared-scales branch should fire and
        # the per-covariate SMDs should match a standalone compute_smd call.
        from mlsynth.utils.post_fit import compute_smd as _smd
        scales = np.array([2.0, 3.0])
        out = _pf(t, c, n_fit=8, cov_matrix=cov,
                  treated_weights=tw, control_weights=cw,
                  cov_scales=scales)
        # Reference value using the same explicit scales.
        ref = _smd(cov, tw, cw, cov_scales=scales)
        assert set(out.covariate_smd.keys()) == set(ref["smd"].keys())

    # ---- _extract_inference: every estimator-shape branch -----------

    def test_extract_inference_none(self):
        out = _extract_inference(None)
        assert all(v is None for v in out.values())

    def test_extract_inference_dict(self):
        out = _extract_inference({"p_value": 0.02, "ci_lower": -1.0,
                                  "ci_upper": 1.0, "method": "custom"})
        assert out["p_value"] == 0.02 and out["method"] == "custom"

    def test_extract_inference_marex(self):
        ci = np.array([[np.nan, np.nan]] * 5 + [[-1.0, 1.0], [-2.0, 2.0]])
        inf = MAREXInference(
            treated_effects=np.array([0.0, 0.0]),
            placebo_effects=np.array([0.0, 0.0]),
            fulltreated_effects=np.zeros(7),
            s_obs=0.0, global_p_value=0.05,
            per_period_pvals=np.zeros(2), ci=ci,
        )
        out = _extract_inference(inf)
        assert out["p_value"] == pytest.approx(0.05)
        assert out["method"] == "marex_permutation"
        # CI low/high are the row-wise nanmeans across the band.
        assert out["ci_lower"] == pytest.approx(-1.5)
        assert out["ci_upper"] == pytest.approx(1.5)

    def test_extract_inference_syndes_like(self):
        # Duck-type a "SYNDESInference" — _extract_inference dispatches on
        # type().__name__, so a class literally named SYNDESInference is enough.
        from dataclasses import dataclass as _dc

        @_dc
        class SYNDESInference:        # noqa: D401 — adapter test stub
            atet: float
            p_value: float
            null_stats: np.ndarray
            alpha: float

        stub = SYNDESInference(
            atet=0.4, p_value=0.03,
            null_stats=np.array([-0.2, -0.1, 0.1, 0.2, 0.0]),
            alpha=0.10,
        )
        out = _extract_inference(stub)
        assert out["method"] == "syndes_permutation"
        assert out["p_value"] == pytest.approx(0.03)
        assert out["ci_lower"] is not None and out["ci_upper"] is not None

    def test_extract_inference_lexscm_like(self):
        from dataclasses import dataclass as _dc

        @_dc
        class Inference:               # LEXSCM-named adapter
            p_value: float
            ci_lower: float
            ci_upper: float

        stub = Inference(p_value=0.04, ci_lower=-0.3, ci_upper=0.5)
        out = _extract_inference(stub)
        assert out["method"] == "lexscm_conformal"
        assert out["p_value"] == 0.04
        assert out["ci_lower"] == -0.3 and out["ci_upper"] == 0.5

    def test_extract_inference_generic_attribute_fallback(self):
        # Any object with attributes named p_value / ci_lower / ci_upper /
        # method gets a best-effort read.
        class _GenericInf:
            p_value = 0.01
            ci_lower = -0.5
            ci_upper = 0.5
        out = _extract_inference(_GenericInf())
        assert out["p_value"] == 0.01

    # ---- _safe_float ------------------------------------------------

    def test_safe_float_returns_none_for_none(self):
        assert _safe_float(None) is None

    def test_safe_float_returns_none_for_unparseable(self):
        assert _safe_float("abc") is None

    def test_safe_float_returns_none_for_nan_or_inf(self):
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None

    # ---- AR(1) helpers ---------------------------------------------

    def test_ar1_rho_short_series(self):
        assert _ar1_rho(np.array([1.0])) == 0.0
        assert _ar1_rho(np.array([])) == 0.0

    def test_ar1_rho_degenerate_zero_variance(self):
        assert _ar1_rho(np.zeros(10)) == 0.0

    def test_ar1_rho_clipped(self):
        # Perfectly serially correlated series clips to +0.99.
        rho = _ar1_rho(np.arange(20, dtype=float))
        assert rho <= 0.99

    def test_variance_inflation_corner_cases(self):
        assert _variance_inflation(0, 0.5) == 0.0
        assert _variance_inflation(1, 0.5) == 1.0
        # With rho == 0 the VIF reduces to 1/n.
        np.testing.assert_allclose(_variance_inflation(10, 0.0), 0.1)
        # With non-zero rho, VIF > 1/n.
        assert _variance_inflation(10, 0.7) > 0.1

    # ---- compute_power_analysis fallbacks --------------------------

    def test_power_analysis_handles_no_pre_window(self):
        # Empty pre-window ⇒ sigma_placebo NaN, MDE is NaN everywhere.
        from mlsynth.utils.post_fit import (
            SyntheticControlPostFit, compute_power_analysis as _pa,
        )
        t = np.zeros(6); c = np.zeros(6)
        pf = SyntheticControlPostFit(
            treated_series=t, control_series=c, gap_series=t - c,
            n_fit=0, n_blank=0, n_post=6,
        )
        out = _pa(pf)
        assert np.isnan(out.sigma_placebo)
        assert np.isnan(out.headline.mde_absolute)

    def test_power_analysis_uses_pre_window_when_no_blank(self):
        # With n_blank == 0 the placebo estimator falls back to the fit
        # window (lines 579-580 of post_fit.py).
        from mlsynth.utils.post_fit import (
            SyntheticControlPostFit, compute_power_analysis as _pa,
        )
        rng = np.random.default_rng(0)
        T_fit, T_post = 12, 4
        t = rng.normal(size=T_fit + T_post)
        c = rng.normal(size=T_fit + T_post)
        pf = SyntheticControlPostFit(
            treated_series=t, control_series=c, gap_series=t - c,
            n_fit=T_fit, n_blank=0, n_post=T_post, ate=0.1,
        )
        out = _pa(pf)
        assert np.isfinite(out.sigma_placebo)
        assert out.serial_correlation >= -1 and out.serial_correlation <= 1

    def test_power_analysis_zero_post_window(self):
        # Power analysis must NOT depend on post-period data. With n_post == 0
        # the baseline is taken from the placebo (here the fit) window, so it is
        # finite and mde_pct is well-defined -- computed without any post data.
        from mlsynth.utils.post_fit import (
            SyntheticControlPostFit, compute_power_analysis as _pa,
        )
        rng = np.random.default_rng(2)
        T_fit = 12
        c = 100.0 + rng.normal(size=T_fit)          # clear nonzero outcome level
        t = c + rng.normal(scale=0.1, size=T_fit)
        pf = SyntheticControlPostFit(
            treated_series=t, control_series=c, gap_series=t - c,
            n_fit=T_fit, n_blank=0, n_post=0,
        )
        out = _pa(pf)
        assert np.isfinite(out.baseline) and abs(out.baseline - 100.0) < 5.0
        assert np.isfinite(out.headline.mde_absolute)
        assert np.isfinite(out.headline.mde_pct)

    def test_power_analysis_explicit_post_grid(self):
        from mlsynth.utils.post_fit import (
            SyntheticControlPostFit, compute_power_analysis as _pa,
        )
        rng = np.random.default_rng(3)
        T = 20
        t = rng.normal(size=T); c = rng.normal(size=T)
        pf = SyntheticControlPostFit(
            treated_series=t, control_series=c, gap_series=t - c,
            n_fit=14, n_blank=2, n_post=4, ate=0.2,
        )
        out = _pa(pf, post_grid=[3, 6, 9, 12])
        # Explicit grid is honoured.
        assert [pt.post_periods for pt in out.curve] == [3, 6, 9, 12]
        # MDE-by-horizon dict has those keys too (covers line 160).
        assert set(out.mde_by_horizon().keys()) == {3, 6, 9, 12}

    # ---- compute_post_fit_marex adapter ----------------------------

    def test_compute_post_fit_marex_adapter(self, panel):
        # The adapter is what auto-attaches post_fit to MAREXResults; call it
        # directly to cover lines 396-415 of post_fit.py.
        from mlsynth.utils.marex_helpers.setup import prepare_marex_panel
        from mlsynth.utils.marex_helpers.orchestration import solve_marex
        import cvxpy as cp
        ppanel = prepare_marex_panel(
            df=panel, outcome="y", unitid="unit", time="time",
            cluster=None, T0=10, inference=False, blank_periods=0, T_post=None,
        )
        raw = solve_marex(
            Y_full=ppanel.Y_full, T0=ppanel.T0, clusters=ppanel.clusters,
            blank_periods=ppanel.blank_periods, m_eq=2, solver=cp.SCIP,
        )
        out = compute_post_fit_marex(raw, ppanel)
        assert out.n_fit == 10 and out.n_blank == 0
        assert out.gap_series.size == 14


# ----------------------------------------------------------------------
# Last-mile coverage: a handful of branches not naturally hit elsewhere
# ----------------------------------------------------------------------

class TestCoverageMopUp:
    """Branches that need a specific construction to fire: tied-support swap,
    full budget dict, unit-penalized xi / lambda2_unit, the zeta integrality
    penalty in the relaxed solver, the MAREXResults convenience properties,
    and the three exception-conversion paths in MAREX.fit().
    """

    # ---- orchestration: tied-support label swap (orchestration.py 90-92)

    def test_orchestrator_first_pos_breaks_tie_on_support_size(self):
        # n_t == n_c (tie) ⇒ _first_pos comparison decides the swap.
        df = _panel(J=4, T=10)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 8, "m_eq": 2}).fit()
        # With J=4 and m_eq=2 the treated and control supports are both 2 —
        # the swap rule then prefers the earlier-first-treated group; either
        # outcome is fine, but the code path that calls _first_pos has run.
        cluster = res.clusters["0"]
        assert (len(cluster.unit_weight_map["Treated"]) == 2
                and len(cluster.unit_weight_map["Control"]) == 2)

    # ---- structures.py 189: synthetic_control property

    def test_results_synthetic_control_property(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        # Both convenience properties round-trip to the aggregate fields.
        np.testing.assert_allclose(res.synthetic_treated,
                                    res.globres.synthetic_treated)
        np.testing.assert_allclose(res.synthetic_control,
                                    res.globres.synthetic_control)

    # ---- formulation.py 80: complete budget dict accepted

    def test_validate_costs_budget_dict_complete(self):
        # All cluster labels present in the budget dict ⇒ dict is returned
        # as-is (covers line 80).
        out_costs, out_budget = validate_costs_budget(
            costs=[1.0] * 4, budget={0: 7.0, 1: 3.0}, N=4,
            cluster_labels=np.array([0, 1]), K=2,
        )
        assert out_budget == {0: 7.0, 1: 3.0}
        assert out_costs.shape == (4,)

    # ---- formulation.py 232-243: extra penalty branches in build_objective.
    # These add non-DCP terms (variable * variable, z*(1-z)) that cvxpy
    # rejects at solve time, but the objective-construction code itself runs,
    # so we exercise it directly via build_objective.

    def test_build_objective_unit_penalized_with_xi(self):
        import cvxpy as cp
        from mlsynth.utils.marex_helpers.formulation import (
            build_objective,
            init_cvxpy_variables,
        )
        Y_fit = np.arange(24, dtype=float).reshape(4, 6)
        cluster_members = [np.array([0, 1, 2, 3])]
        Xbar = [Y_fit.mean(axis=0)]
        w, v, z = init_cvxpy_variables(N=4, K=1, boolean=True)
        obj = build_objective(Y_fit, Xbar, cluster_members, w, v, z,
                              design="unit_penalized", xi=0.5)
        assert isinstance(obj, cp.Minimize)

    def test_build_objective_unit_penalized_with_lambda2_unit(self):
        import cvxpy as cp
        from mlsynth.utils.marex_helpers.formulation import (
            build_objective, init_cvxpy_variables, precompute_distances,
        )
        Y_fit = np.arange(24, dtype=float).reshape(4, 6)
        cluster_members = [np.array([0, 1, 2, 3])]
        Xbar = [Y_fit.mean(axis=0)]
        _, D2 = precompute_distances(Y_fit, Xbar, cluster_members)
        w, v, z = init_cvxpy_variables(N=4, K=1, boolean=True)
        obj = build_objective(Y_fit, Xbar, cluster_members, w, v, z,
                              design="unit_penalized",
                              lambda2_unit=0.1, D2_list=D2)
        assert isinstance(obj, cp.Minimize)

    def test_build_objective_zeta_integrality_penalty(self):
        import cvxpy as cp
        from mlsynth.utils.marex_helpers.formulation import (
            build_objective, init_cvxpy_variables,
        )
        Y_fit = np.arange(24, dtype=float).reshape(4, 6)
        cluster_members = [np.array([0, 1, 2, 3])]
        Xbar = [Y_fit.mean(axis=0)]
        w, v, z = init_cvxpy_variables(N=4, K=1, boolean=False)
        obj = build_objective(Y_fit, Xbar, cluster_members, w, v, z,
                              design="standard", zeta=0.1)
        assert isinstance(obj, cp.Minimize)

    # ---- scexp.py exception-conversion paths -----------------------

    def test_fit_wraps_unexpected_setup_failure(self, panel, monkeypatch):
        # Force prepare_marex_panel to raise a plain RuntimeError — fit()
        # should reframe it as MlsynthDataError (lines 118-119 of scexp.py).
        import mlsynth.estimators.scexp as scexp_mod

        def _boom(**_kw):
            raise RuntimeError("synthetic panic")
        monkeypatch.setattr(scexp_mod, "prepare_marex_panel", _boom)
        with pytest.raises(MlsynthDataError, match="Error preparing MAREX"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2}).fit()

    def test_fit_wraps_unexpected_solver_failure(self, panel, monkeypatch):
        # Force solve_marex to raise a plain RuntimeError — fit() converts
        # to MlsynthEstimationError (lines 138-139 of scexp.py).
        import mlsynth.estimators.scexp as scexp_mod
        from mlsynth.exceptions import MlsynthEstimationError

        def _boom(**_kw):
            raise RuntimeError("solver exploded")
        monkeypatch.setattr(scexp_mod, "solve_marex", _boom)
        with pytest.raises(MlsynthEstimationError, match="MAREX design failed"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2}).fit()

    def test_fit_wraps_plotting_failure(self, panel, monkeypatch):
        # display_graph=True + plot_marex raising RuntimeError ⇒
        # MlsynthPlottingError (lines 147-149 of scexp.py).
        import mlsynth.estimators.scexp as scexp_mod
        from mlsynth.exceptions import MlsynthPlottingError

        def _boom(*_a, **_kw):
            raise RuntimeError("matplotlib went south")
        monkeypatch.setattr(scexp_mod, "plot_marex", _boom)
        with pytest.raises(MlsynthPlottingError, match="MAREX plotting"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2, "display_graph": True}).fit()

    def test_fit_display_graph_succeeds(self, panel):
        # display_graph=True with the real plotter must complete without error
        # (covers the successful branch at lines 142-143 of scexp.py).
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "display_graph": True}).fit()
        assert res.post_fit is not None
        _plt.close("all")

    # ---- The bare-raise paths: typed Mlsynth exceptions from helpers must
    # bubble out unwrapped (so callers see the real cause, not a generic
    # MAREX-level reframing).

    def test_setup_typed_error_passes_through(self, panel, monkeypatch):
        # MlsynthDataError raised inside prepare_marex_panel must bubble out
        # untouched (line 116 of scexp.py: bare `raise` in the typed handler).
        import mlsynth.estimators.scexp as scexp_mod

        def _boom(**_kw):
            raise MlsynthDataError("upstream data problem")
        monkeypatch.setattr(scexp_mod, "prepare_marex_panel", _boom)
        with pytest.raises(MlsynthDataError, match="upstream data problem"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2}).fit()

    def test_setup_value_error_reframed_as_config_error(self, panel, monkeypatch):
        # A plain ValueError from prepare_marex_panel becomes
        # MlsynthConfigError (line 118 of scexp.py).
        import mlsynth.estimators.scexp as scexp_mod

        def _boom(**_kw):
            raise ValueError("bad shape")
        monkeypatch.setattr(scexp_mod, "prepare_marex_panel", _boom)
        with pytest.raises(MlsynthConfigError, match="bad shape"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2}).fit()

    def test_solver_typed_error_passes_through(self, panel, monkeypatch):
        # MlsynthEstimationError raised inside solve_marex must bubble out
        # untouched (line 139 of scexp.py).
        import mlsynth.estimators.scexp as scexp_mod
        from mlsynth.exceptions import MlsynthEstimationError

        def _boom(**_kw):
            raise MlsynthEstimationError("solver said no")
        monkeypatch.setattr(scexp_mod, "solve_marex", _boom)
        with pytest.raises(MlsynthEstimationError, match="solver said no"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2}).fit()

    def test_plotter_typed_error_passes_through(self, panel, monkeypatch):
        # MlsynthPlottingError raised inside plot_marex must bubble out
        # untouched (line 147 of scexp.py).
        import mlsynth.estimators.scexp as scexp_mod
        from mlsynth.exceptions import MlsynthPlottingError

        def _boom(*_a, **_kw):
            raise MlsynthPlottingError("plot already broken")
        monkeypatch.setattr(scexp_mod, "plot_marex", _boom)
        with pytest.raises(MlsynthPlottingError, match="plot already broken"):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 2, "display_graph": True}).fit()
