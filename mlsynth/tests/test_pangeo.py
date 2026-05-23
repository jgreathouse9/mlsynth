"""Tests for the PANGEO experimental-design estimator.

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): parallelism scoring + admissible-pair
  enumeration + best split.
* Layer 2 (data utilities): historical-panel ingestion + arm pools.
* Layer 3 (integration): the design exact-covers each arm, respects Q,
  and beats random assignment on pre-period parallelism (the whole point).
* Layer 4 (public API contracts): import, frozen results, config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PANGEO
from mlsynth.config_models import PANGEOConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.pangeo_helpers import (
    PangeoPower,
    PangeoResults,
    covariate_imbalance,
    enumerate_candidate_pairs,
    gap_variance,
    make_seasonal_sales_panel,
    parallelism_r2,
    prepare_pangeo_inputs,
    run_pangeo,
)
from mlsynth.utils.pangeo_helpers.parallelism import best_split


@pytest.fixture
def panel():
    return make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                     T=120, seed=0)


# ----------------------------------------------------------------------
# Layer 1: parallelism helpers
# ----------------------------------------------------------------------

class TestParallelism:
    def test_gap_variance_zero_for_parallel(self):
        # Two trajectories differing by a constant level -> perfectly parallel.
        a = np.array([1.0, 2.0, 3.0, 2.5])
        b = a + 7.0
        assert gap_variance(a, b) == pytest.approx(0.0, abs=1e-12)

    def test_gap_variance_positive_for_nonparallel(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])   # anti-parallel
        assert gap_variance(a, b) > 1.0

    def test_best_split_picks_parallel_halves(self):
        # 4 units: two pairs of parallel (level-shifted) trajectories.
        base1 = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        base2 = np.array([0.0, 1.0, 4.0, 1.0, 2.0])
        Ypre = np.vstack([base1, base1 + 10, base2, base2 + 10])
        members = np.array([0, 1, 2, 3])
        score, side_a, side_b = best_split(members, Ypre, max_size=2)
        # The optimal split groups {0 with 2} vs {1 with 3} OR {0,1} vs {2,3}?
        # The level-shifted duplicates are perfectly parallel, so any split that
        # pairs base1-types against base1-types gives a near-zero gap variance.
        assert score < 1e-6

    def test_enumerate_pairs_nonempty(self, panel):
        inp = prepare_pangeo_inputs(panel, "sales", "arm", "unit", "time")
        idx = inp.arm_units["A"]
        pairs = enumerate_candidate_pairs(idx, inp.Y, max_size=3)
        assert len(pairs) > 0
        for p in pairs:
            assert len(p["side_a"]) <= 3 and len(p["side_b"]) <= 3
            assert set(p["side_a"]).isdisjoint(p["side_b"])


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs(self, panel):
        inp = prepare_pangeo_inputs(panel, "sales", "arm", "unit", "time")
        assert inp.Y.shape == (18, 120)
        assert set(inp.arm_units) == {"A", "B", "C"}
        assert all(idx.size == 6 for idx in inp.arm_units.values())

    def test_small_arm_rejected(self):
        # Two arms with one geo each: enough units overall, but no arm can
        # form a supergeo pair.
        df = make_seasonal_sales_panel(units_per_arm=1, arms=("A", "B"),
                                       T=20, seed=0)
        with pytest.raises(MlsynthConfigError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time")

    def test_missing_column_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(panel, "nope", "arm", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: integration
# ----------------------------------------------------------------------

class TestDesign:
    def test_exact_cover_and_respects_Q(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "max_supergeo_size": 3, "display_graphs": False}).fit()
        for arm, d in res.arm_designs.items():
            covered = []
            for p in d.pairs:
                assert len(p.treatment) <= 3 and len(p.control) <= 3   # Q
                covered.extend(p.treatment + p.control)
            # Exact cover: every arm unit assigned exactly once.
            assert sorted(covered) == sorted(
                [u for u in res.assignment if u.startswith(arm)]
            )
            assert len(covered) == len(set(covered)) == d.n_units

    def test_beats_random_parallelism(self, panel):
        """The designed split is far more parallel than random assignment."""
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "max_supergeo_size": 3, "display_graphs": False}).fit()
        states = sorted(panel.unit.unique())
        times = sorted(panel.time.unique())
        Y = panel.pivot(index="unit", columns="time",
                        values="sales").loc[states, times].to_numpy()
        pos = {u: i for i, u in enumerate(states)}
        rng = np.random.default_rng(1)
        for arm, d in res.arm_designs.items():
            dgv = sum(gap_variance(Y[[pos[u] for u in p.treatment]].mean(0),
                                   Y[[pos[u] for u in p.control]].mean(0))
                      for p in d.pairs)
            arm_idx = [pos[u] for u in states
                       if panel[panel.unit == u].arm.iloc[0] == arm]
            rand = []
            for _ in range(200):
                perm = rng.permutation(arm_idx)
                h = len(perm) // 2
                rand.append(gap_variance(Y[perm[:h]].mean(0),
                                         Y[perm[h:2 * h]].mean(0)))
            assert dgv < np.median(rand)        # design beats typical random
            assert d.mean_parallelism_r2 > 0.5  # genuinely parallel

    def test_Q_one_recovers_matched_pairs(self, panel):
        """Q=1 forces singleton supergeos (classic matched pairs)."""
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "max_supergeo_size": 1, "display_graphs": False}).fit()
        for d in res.arm_designs.values():
            for p in d.pairs:
                assert len(p.treatment) == 1 and len(p.control) == 1


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import PANGEO as _P
        assert _P is PANGEO

    def test_results_frozen(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "display_graphs": False}).fit()
        assert isinstance(res, PangeoResults)
        with pytest.raises(Exception):
            res.max_supergeo_size = 99

    def test_invalid_Q_rejected(self, panel):
        with pytest.raises(MlsynthConfigError):
            PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "max_supergeo_size": 0, "display_graphs": False})


# ----------------------------------------------------------------------
# Covariate-augmented design
# ----------------------------------------------------------------------

@pytest.fixture
def panel_cov():
    return make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                     T=104, seed=0, covariates=True)


class TestCovariates:
    def test_inputs_carry_covariates(self, panel_cov):
        inp = prepare_pangeo_inputs(
            panel_cov, "sales", "arm", "unit", "time",
            covariates=["population", "income"])
        assert inp.covariates.shape == (18, 2)
        assert inp.covariate_names == ["population", "income"]
        assert inp.covariate_scales.shape == (2,)
        assert np.all(inp.covariate_scales > 0)

    def test_missing_covariate_rejected(self, panel_cov):
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(panel_cov, "sales", "arm", "unit", "time",
                                  covariates=["nope"])

    def test_smd_recorded_and_cover_preserved(self, panel_cov):
        res = PANGEO({"df": panel_cov, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                      "covariates": ["population", "income"],
                      "display_graphs": False}).fit()
        assert res.metadata["covariates"] == ["population", "income"]
        for arm, d in res.arm_designs.items():
            covered = [u for p in d.pairs for u in (p.treatment + p.control)]
            assert len(covered) == len(set(covered)) == d.n_units
            for p in d.pairs:
                assert set(p.covariate_smd) == {"population", "income"}

    def test_weighting_improves_covariate_balance(self, panel_cov):
        """Up-weighting covariates trades parallelism for better SMD balance."""
        base_cov = panel_cov.groupby("unit")[["population", "income"]].mean()
        scales = base_cov.std(ddof=0)

        def mean_abs_smd(res):
            vals = []
            for d in res.arm_designs.values():
                for p in d.pairs:
                    ca = base_cov.loc[p.treatment].mean()
                    cb = base_cov.loc[p.control].mean()
                    vals.append(np.abs((ca - cb) / scales).mean())
            return float(np.mean(vals))

        plain = PANGEO({"df": panel_cov, "outcome": "sales", "arm": "arm",
                        "unitid": "unit", "time": "time",
                        "max_supergeo_size": 3,
                        "display_graphs": False}).fit()
        balanced = PANGEO({"df": panel_cov, "outcome": "sales", "arm": "arm",
                           "unitid": "unit", "time": "time",
                           "max_supergeo_size": 3,
                           "covariates": ["population", "income"],
                           "covariate_weights": {"population": 25.0,
                                                 "income": 25.0},
                           "display_graphs": False}).fit()
        assert mean_abs_smd(balanced) <= mean_abs_smd(plain)


# ----------------------------------------------------------------------
# Power / MDE analysis
# ----------------------------------------------------------------------

class TestPower:
    def test_power_attached_by_default(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                      "display_graphs": False}).fit()
        assert isinstance(res.power, PangeoPower)
        assert res.power.power_target == 0.80
        assert -1.0 <= res.power.serial_correlation <= 1.0

    def test_can_disable_power(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "compute_power": False, "display_graphs": False}).fit()
        assert res.power is None

    def test_default_horizons_2_to_12(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "display_graphs": False}).fit()
        xs = [pt.post_periods for pt in res.power.program.points]
        assert xs == list(range(2, 13))

    def test_mde_positive_and_shrinks_with_horizon(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "display_graphs": False}).fit()
        pts = res.power.program.points
        assert all(np.isfinite(p.mde_pct) and p.mde_pct > 0 for p in pts)
        mdes = [p.mde_absolute for p in pts]
        # More post periods can only sharpen (or hold) the MDE.
        assert all(b <= a + 1e-9 for a, b in zip(mdes, mdes[1:]))

    def test_program_pooling_beats_arms(self, panel):
        """Pooling all arms detects no worse than the least-powerful arm."""
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                      "display_graphs": False}).fit()
        pw = res.power
        assert set(pw.arms) == set(res.arm_designs)
        prog2 = pw.program.points[0].mde_pct
        worst_arm = max(c.points[0].mde_pct for c in pw.arms.values())
        assert prog2 <= worst_arm + 1e-9

    def test_custom_horizons_and_target(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "power_target": 0.90, "power_post_periods": [4, 8],
                      "display_graphs": False}).fit()
        assert [pt.post_periods for pt in res.power.program.points] == [4, 8]
        assert res.power.power_target == 0.90

    def test_power_for_effect_in_unit_interval(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "display_graphs": False}).fit()
        pw = res.power
        small = pw.power_for_effect(1.0, 8)
        large = pw.power_for_effect(50.0, 8)
        assert 0.0 <= small <= large <= 1.0
        # A big effect should be near-certain to detect.
        assert large > 0.95

    def test_summary_table(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "display_graphs": False}).fit()
        tab = res.power.summary()
        assert "program_mde_pct" in tab.columns
        assert len(tab) == 11


# ----------------------------------------------------------------------
# Automatic Q selection
# ----------------------------------------------------------------------

class TestAutoQ:
    def test_auto_q_when_unspecified(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "display_graphs": False}).fit()
        assert res.metadata.get("q_auto_selected") is True
        assert res.metadata["q_selected"] == res.max_supergeo_size
        sweep = res.metadata["q_sweep"]
        assert len(sweep) >= 1
        # The chosen Q minimises mean program MDE over the feasible sweep.
        feasible = [s for s in sweep if s["feasible"]]
        best = min(feasible, key=lambda s: s["mean_program_mde_pct"])
        assert best["q"] == res.metadata["q_selected"]
        # Still a valid exact-cover design.
        for arm, d in res.arm_designs.items():
            covered = [u for p in d.pairs for u in (p.treatment + p.control)]
            assert len(covered) == len(set(covered)) == d.n_units

    def test_explicit_q_skips_auto(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 2,
                      "display_graphs": False}).fit()
        assert res.max_supergeo_size == 2
        assert "q_auto_selected" not in res.metadata


# ----------------------------------------------------------------------
# Realized ATT (post_col) + inference
# ----------------------------------------------------------------------

class TestEffects:
    def _panel(self, seed=0, n_post=8, covariates=False):
        return make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                         T=104, seed=seed, n_post=n_post,
                                         covariates=covariates)

    def test_design_identical_with_post(self):
        d = self._panel()
        pre = d[d.post_col == 0].drop(columns="post_col")
        cfg = dict(outcome="sales", arm="arm", unitid="unit", time="time",
                   max_supergeo_size=3, compute_power=False,
                   display_graphs=False)
        with_post = PANGEO({"df": d, "post_col": "post_col", **cfg}).fit()
        pre_only = PANGEO({"df": pre, **cfg}).fit()
        assert with_post.assignment == pre_only.assignment
        assert with_post.effects is not None
        assert pre_only.effects is None

    def test_effect_fields_and_summary(self):
        d = self._panel()
        res = PANGEO({"df": d, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "post_col": "post_col",
                      "max_supergeo_size": 3, "compute_power": False,
                      "display_graphs": False}).fit()
        e = res.effects
        assert set(e.arms) == set(res.arm_designs)
        for est in [e.program] + list(e.arms.values()):
            assert est.ci_lower <= est.att <= est.ci_upper
            assert 0.0 <= est.p_value <= 1.0
            assert est.n_post == 8
        tab = e.summary()
        assert "program" in tab.index and "att" in tab.columns

    def test_recovers_injected_effect(self):
        """A large constant additive effect is recovered and is significant."""
        d = self._panel(seed=3)
        des = PANGEO({"df": d, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "post_col": "post_col",
                      "max_supergeo_size": 3, "compute_power": False,
                      "display_graphs": False}).fit()
        treated = [u for u, a in des.assignment.items() if a == "treatment"]
        TAU = 5.0
        d2 = d.copy()
        mask = (d2.post_col == 1) & (d2.unit.isin(treated))
        d2.loc[mask, "sales"] += TAU
        e = PANGEO({"df": d2, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "post_col": "post_col",
                    "max_supergeo_size": 3, "compute_power": False,
                    "display_graphs": False}).fit().effects
        assert abs(e.program.att - TAU) < 1.5          # recovered (noisy)
        assert e.program.p_value < 0.05                # detected

    def test_population_weighted(self):
        d = self._panel(covariates=True)
        res = PANGEO({"df": d, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "post_col": "post_col",
                      "weight_col": "population", "max_supergeo_size": 3,
                      "compute_power": False, "display_graphs": False}).fit()
        assert res.effects.weighted is True

    def test_missing_post_col_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "post_col": "nope",
                    "display_graphs": False}).fit()


# ----------------------------------------------------------------------
# Augmented-DiD inference (Li & Van den Bulte 2022)
# ----------------------------------------------------------------------

class TestADIDInference:
    def _design_and_score(self, seed, tau, **sim):
        d = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                      T=104, seed=seed, n_post=8, **sim)
        cfg = dict(outcome="sales", arm="arm", unitid="unit", time="time",
                   post_col="post_col", max_supergeo_size=3,
                   compute_power=False, display_graphs=False)
        des = PANGEO({"df": d, **cfg}).fit()
        treated = [u for u, a in des.assignment.items() if a == "treatment"]
        d2 = d.copy()
        m = (d2.post_col == 1) & (d2.unit.isin(treated))
        d2.loc[m, "sales"] += tau
        return PANGEO({"df": d2, **cfg}).fit().effects, des.effects

    def test_se_scale_and_pvalue_present(self):
        eff, _ = self._design_and_score(0, 0.6)
        for est in [eff.program] + list(eff.arms.values()):
            assert est.se > 0
            assert est.ci_lower <= est.att <= est.ci_upper
            assert 0.0 <= est.p_value <= 1.0
        assert "scale" in eff.summary().columns

    def test_augment_toggle_sets_scale(self):
        d = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                      T=104, seed=1, n_post=8)
        cfg = dict(outcome="sales", arm="arm", unitid="unit", time="time",
                   post_col="post_col", max_supergeo_size=3,
                   compute_power=False, display_graphs=False)
        aug = PANGEO({"df": d, "att_augment": True, **cfg}).fit().effects
        plain = PANGEO({"df": d, "att_augment": False, **cfg}).fit().effects
        assert plain.program.scale == 1.0          # delta_2 forced to 1
        assert aug.program.scale != 1.0            # free augmentation
        assert aug.metadata["augment"] is True

    def test_trend_toggle_runs(self):
        d = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                      T=104, seed=2, n_post=8)
        res = PANGEO({"df": d, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "post_col": "post_col",
                      "att_trend": False, "max_supergeo_size": 3,
                      "compute_power": False, "display_graphs": False}).fit()
        assert res.effects.metadata["trend"] is False

    @pytest.mark.slow
    def test_nominal_coverage_on_stationary_gap(self):
        """Paper-faithful DGP (stationary factor, no trend/season): the
        prediction-variance CI is ~nominal and the point ATT unbiased.
        """
        TAU = 0.6
        cover, type1, atts = [], [], []
        for s in range(60):
            eff, null = self._design_and_score(
                s, TAU, factor="iid", season_amp=0.0, trend_sd=0.0)
            cover.append(eff.program.ci_lower <= TAU <= eff.program.ci_upper)
            atts.append(eff.program.att)
            type1.append(null.program.p_value < 0.05)
        assert abs(np.mean(atts) - TAU) < 0.1      # unbiased
        assert np.mean(cover) > 0.85               # ~0.93 expected
        assert np.mean(type1) < 0.20               # ~0.07 expected

    def test_pure_did_power_and_effects_coherent(self):
        """Att_augment=False gives plain DiD end to end: scale fixed at 1,
        finite SE, and a power MDE built from the plain-DiD residual.
        """
        d = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                      T=104, seed=0, n_post=8)
        res = PANGEO({"df": d, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "post_col": "post_col",
                      "max_supergeo_size": 3, "att_augment": False,
                      "att_trend": False, "display_graphs": False}).fit()
        assert res.effects.program.scale == 1.0
        assert np.isfinite(res.effects.program.se) and res.effects.program.se > 0
        assert all(np.isfinite(pt.mde_pct) and pt.mde_pct > 0
                   for pt in res.power.program.points)

    @pytest.mark.slow
    def test_planning_mde_calibrated_to_realized_se(self):
        """The held-out power residual matches the evaluation model, so the
        planning MDE tracks the realised SE on a stationary gap.
        """
        TAU = 0.6
        MULT = 2.8016                              # z_.975 + z_.80
        plan, real = [], []
        for s in range(40):
            d = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                          T=104, seed=s, n_post=8, factor="iid",
                                          season_amp=0.0, trend_sd=0.0)
            cfg = dict(outcome="sales", arm="arm", unitid="unit", time="time",
                       max_supergeo_size=3, display_graphs=False)
            des = PANGEO({"df": d[d.post_col == 0].drop(columns="post_col"),
                          **cfg}).fit()
            plan.append([pt.mde_pct for pt in des.power.program.points
                         if pt.post_periods == 8][0])
            e = PANGEO({"df": d, "post_col": "post_col", "compute_power": False,
                        **cfg}).fit().effects.program
            real.append(100 * MULT * e.se / e.baseline)
        ratio = np.mean(plan) / np.mean(real)
        assert 0.6 <= ratio <= 1.5                 # ~0.93 expected


@pytest.mark.parametrize("objective", ["ss_res", "r2", "weighted"])
def test_objective_options_run_and_cover(panel, objective):
    """Every score objective yields a valid exact-cover design."""
    res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                  "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                  "objective": objective, "display_graphs": False}).fit()
    assert res.metadata["objective"] == objective
    for arm, d in res.arm_designs.items():
        covered = [u for p in d.pairs for u in (p.treatment + p.control)]
        assert len(covered) == len(set(covered)) == d.n_units


# ----------------------------------------------------------------------
# Layer 1 (cont.): numerical-helper edge cases
# ----------------------------------------------------------------------

class TestHelperEdgeCases:
    def test_gap_variance_single_point_is_zero(self):
        # A length-1 trajectory has no shape: the level-removed gap is 0.
        assert gap_variance(np.array([5.0]), np.array([2.0])) ==\
            pytest.approx(0.0)

    def test_parallelism_r2_nan_for_constant_treatment(self):
        # No variation to explain -> R^2 undefined (NaN), not a crash.
        const = np.full(10, 3.0)
        assert np.isnan(parallelism_r2(const, np.arange(10.0)))

    def test_covariate_imbalance_known_value(self):
        # Standardised squared mean-difference: ((2-0)/1)^2 + ((4-1)/3)^2 = 5.
        a = np.array([2.0, 4.0]); b = np.array([0.0, 1.0])
        scales = np.array([1.0, 3.0]); weights = np.array([1.0, 1.0])
        assert covariate_imbalance(a, b, scales, weights) == pytest.approx(5.0)

    def test_enumerate_two_unit_arm_single_pair(self):
        # An arm of exactly two geos admits exactly one (1 vs 1) pair.
        Y = np.array([[1.0, 2.0, 3.0], [1.5, 2.4, 3.1]])
        pairs = enumerate_candidate_pairs(np.array([0, 1]), Y, max_size=3)
        assert len(pairs) == 1
        assert len(pairs[0]["side_a"]) == 1 and len(pairs[0]["side_b"]) == 1


# ----------------------------------------------------------------------
# Error paths: data ingestion (prepare_pangeo_inputs)
# ----------------------------------------------------------------------

class TestSetupErrors:
    @pytest.mark.parametrize("col", ["sales", "arm", "unit", "time"])
    def test_missing_required_column(self, panel, col):
        renamed = panel.rename(columns={col: f"{col}_x"})
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(renamed, "sales", "arm", "unit", "time")

    def test_nan_outcome_rejected(self, panel):
        df = panel.copy(); df.loc[0, "sales"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time")

    def test_unbalanced_panel_rejected(self, panel):
        df = panel.drop(panel.index[5])      # remove one unit-time cell
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time")

    def test_arm_varying_within_unit_rejected(self, panel):
        df = panel.copy()
        n = int((df.unit == "A0").sum())
        df.loc[df.unit == "A0", "arm"] = (["A", "B"] * (n // 2 + 1))[:n]
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time")

    def test_fewer_than_two_units_rejected(self):
        df = make_seasonal_sales_panel(units_per_arm=1, arms=("A",),
                                       T=20, seed=0)
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time")

    def test_missing_weight_column_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(panel, "sales", "arm", "unit", "time",
                                  weight_col="nope")

    def test_nan_weight_rejected(self, panel):
        df = panel.copy(); df["pop"] = 1.0; df.loc[0, "pop"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time",
                                  weight_col="pop")

    def test_nonpositive_weight_rejected(self, panel):
        df = panel.copy(); df["pop"] = 1.0
        df.loc[df.unit == "A0", "pop"] = -1.0
        with pytest.raises(MlsynthConfigError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time",
                                  weight_col="pop")

    def test_nan_covariate_rejected(self, panel):
        df = panel.copy(); df["z"] = 1.0; df.loc[0, "z"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time",
                                  covariates=["z"])


# ----------------------------------------------------------------------
# Error paths: configuration validation (loud, typed rejection)
# ----------------------------------------------------------------------

class TestConfigValidation:
    @pytest.mark.parametrize("override", [
        {"max_supergeo_size": 0},
        {"max_supergeo_size": -2},
        {"objective": "bogus"},
        {"min_pairs": 0},
        {"power_target": 0.0},
        {"power_target": 1.5},
        {"power_alpha": 0.0},
        {"power_alpha": 2.0},
        {"frac_E": 0.0},
        {"frac_E": 1.0},
        {"recency_decay": 0.0},
        {"recency_decay": 1.5},
    ])
    def test_invalid_config_value_rejected(self, panel, override):
        cfg = {"df": panel, "outcome": "sales", "arm": "arm", "unitid": "unit",
               "time": "time", "display_graphs": False, **override}
        with pytest.raises(MlsynthConfigError):
            PANGEO(cfg)

    @pytest.mark.parametrize("drop", ["outcome", "arm", "unitid", "time", "df"])
    def test_missing_required_field_rejected(self, panel, drop):
        cfg = {"df": panel, "outcome": "sales", "arm": "arm", "unitid": "unit",
               "time": "time", "display_graphs": False}
        cfg.pop(drop)
        with pytest.raises(MlsynthConfigError):
            PANGEO(cfg)


# ----------------------------------------------------------------------
# Error paths: estimation / infeasibility (raised, not silently empty)
# ----------------------------------------------------------------------

class TestEstimatorErrors:
    def test_infeasible_explicit_Q_raises(self):
        # Q=1 on an odd-sized arm has no 1-vs-1 exact cover.
        df = make_seasonal_sales_panel(units_per_arm=5, arms=("A", "B"),
                                       T=30, seed=0)
        with pytest.raises(MlsynthEstimationError):
            PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "max_supergeo_size": 1,
                    "display_graphs": False}).fit()

    def test_min_pairs_infeasible_raises(self, panel):
        # An arm of 6 with Q=3 yields at most 3 pairs; demanding 5 is infeasible.
        with pytest.raises(MlsynthEstimationError):
            PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                    "min_pairs": 5, "display_graphs": False}).fit()

    def test_no_pre_rows_raises(self, panel):
        df = panel.copy(); df["post_col"] = 1     # everything flagged post
        with pytest.raises(MlsynthDataError):
            PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "post_col": "post_col",
                    "display_graphs": False}).fit()

    def test_post_missing_design_unit_raises(self):
        df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                       T=40, seed=0, n_post=6)
        df = df[~((df.post_col == 1) & (df.unit == "A0"))]   # drop a unit post
        with pytest.raises(MlsynthDataError):
            PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "post_col": "post_col",
                    "max_supergeo_size": 3, "compute_power": False,
                    "display_graphs": False}).fit()

    def test_run_pangeo_rejects_nonpositive_q(self, panel):
        inp = prepare_pangeo_inputs(panel, "sales", "arm", "unit", "time")
        with pytest.raises(MlsynthConfigError):
            run_pangeo(inp, max_supergeo_size=0)


# ----------------------------------------------------------------------
# Edge cases: unusual-but-valid inputs behave as expected
# ----------------------------------------------------------------------

class TestEdgeCases:
    def test_integer_unit_and_time_ids(self, panel):
        df = panel.copy()
        df["unit"] = df["unit"].astype("category").cat.codes
        df["time"] = df["time"].astype(int)
        res = PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                      "compute_power": False, "display_graphs": False}).fit()
        covered = [u for d in res.arm_designs.values()
                   for p in d.pairs for u in (p.treatment + p.control)]
        assert len(covered) == len(set(covered)) == df.unit.nunique()

    def test_single_arm(self):
        df = make_seasonal_sales_panel(units_per_arm=6, arms=("A",),
                                       T=40, seed=0)
        res = PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                      "compute_power": False, "display_graphs": False}).fit()
        assert set(res.arm_designs) == {"A"}

    def test_odd_sized_arm_exact_cover(self):
        df = make_seasonal_sales_panel(units_per_arm=5, arms=("A", "B"),
                                       T=40, seed=0)
        res = PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                      "compute_power": False, "display_graphs": False}).fit()
        for d in res.arm_designs.values():
            covered = [u for p in d.pairs for u in (p.treatment + p.control)]
            assert len(covered) == len(set(covered)) == d.n_units == 5

    def test_q_larger_than_arm_is_clamped(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 99,
                      "compute_power": False, "display_graphs": False}).fit()
        for d in res.arm_designs.values():
            for p in d.pairs:                      # halves can't exceed the arm
                assert len(p.treatment) <= d.n_units
                assert len(p.control) <= d.n_units

    def test_single_post_period_effects(self):
        df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                       T=40, seed=0, n_post=1)
        res = PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "post_col": "post_col",
                      "max_supergeo_size": 3, "compute_power": False,
                      "display_graphs": False}).fit()
        assert res.effects.n_post == 1
        assert np.isfinite(res.effects.program.se)

    def test_deterministic(self, panel):
        cfg = {"df": panel, "outcome": "sales", "arm": "arm", "unitid": "unit",
               "time": "time", "max_supergeo_size": 3, "compute_power": False,
               "display_graphs": False}
        r1 = PANGEO(cfg).fit(); r2 = PANGEO(cfg).fit()
        assert r1.assignment == r2.assignment

    def test_config_object_accepted(self, panel):
        cfg = PANGEOConfig(df=panel, outcome="sales", arm="arm", unitid="unit",
                           time="time", max_supergeo_size=3,
                           compute_power=False, display_graphs=False)
        res = PANGEO(cfg).fit()
        assert isinstance(res, PangeoResults)

    def test_auto_q_skips_infeasible(self):
        # Odd arms: Q=1 is infeasible (no 1-vs-1 cover) and must be skipped,
        # not crash; a feasible Q is selected.
        df = make_seasonal_sales_panel(units_per_arm=5, arms=("A", "B"),
                                       T=40, seed=0)
        res = PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "display_graphs": False}).fit()
        sweep = {s["q"]: s["feasible"] for s in res.metadata["q_sweep"]}
        assert sweep.get(1) is False           # Q=1 infeasible for odd arms
        assert res.metadata["q_selected"] >= 2

    def test_min_pairs_respected(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "max_supergeo_size": 3,
                      "min_pairs": 2, "compute_power": False,
                      "display_graphs": False}).fit()
        for d in res.arm_designs.values():
            assert len(d.pairs) >= 2
