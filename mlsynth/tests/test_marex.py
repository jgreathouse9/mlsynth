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
