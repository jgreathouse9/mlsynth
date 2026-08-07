"""Tests for SDIDGEO: synthetic difference-in-differences geo experiment design.

The estimator scores candidate test regions by simulated power, so the tests are
layered: the SDID engine's invariants first, then one pseudo-experiment, then the
grid, then the end-to-end design on GeoLift's published test panel
(``basedata/geolift_test_data.csv``, 40 markets x 105 days).

Two engine properties carry the whole design and are checked against brute-force
alternatives, not asserted:

* neither SDID weight program reads the treated post block, so injecting an
  effect cannot move omega or lambda, and the effect-size sweep is exact
  arithmetic on the ATT;
* the placebo draws reassign control units, so the placebo sigma does not depend
  on the injected effect either.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SDIDGEO
from mlsynth.config_models import DesignResult, SDIDGEOConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.sdidgeo_helpers.engine import (
    placebo_sigma, sdid_att, sdid_fit_once)
from mlsynth.utils.sdidgeo_helpers.simulate import inject_effect, simulate_backtest
from mlsynth.utils.sdidgeo_helpers.batch import run_simulations
from mlsynth.utils.sdidgeo_helpers.shaping import aggregate_treated, donor_matrix
from mlsynth.utils.sdidgeo_helpers.windows import (
    backtest_pre_periods, backtest_treatment_window)

DATA = Path(__file__).resolve().parents[2] / "basedata" / "geolift_test_data.csv"
EFFECT_SIZES = [-0.20, -0.10, 0.0, 0.10, 0.20]


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def wide(panel):
    from mlsynth.utils.datautils import geoex_dataprep
    return geoex_dataprep(panel, "location", "date", "Y")["Ywide"]


@pytest.fixture(scope="module")
def arrays(wide):
    """One candidate's treated series, donor pool, and backtest."""
    units = list(wide.columns)
    candidate = frozenset(units[:2])
    treated = aggregate_treated(wide, candidate, how="mean").to_numpy()
    donors = donor_matrix(wide, candidate).to_numpy()
    n_periods = wide.shape[0]
    duration, sim = 14, 3
    n_pre = backtest_pre_periods(n_periods, duration, sim)
    start, end = backtest_treatment_window(n_periods, duration, sim)
    return treated, donors, n_periods, duration, sim, n_pre, start, end


class TestWindows:
    def test_backtest_window_matches_geolift_arithmetic(self):
        assert backtest_pre_periods(100, 15, 1) == 85
        assert backtest_treatment_window(100, 15, 1) == (85, 99)
        assert backtest_treatment_window(100, 15, 5) == (81, 95)

    def test_post_block_length_is_always_the_duration(self):
        for sim in range(1, 6):
            start, end = backtest_treatment_window(105, 14, sim)
            assert end - start + 1 == 14

    def test_backtest_off_the_panel_raises(self):
        with pytest.raises(MlsynthConfigError, match="runs off the start"):
            backtest_pre_periods(20, 15, 10)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_non_positive_integers_raise(self, bad):
        with pytest.raises(MlsynthConfigError):
            backtest_pre_periods(100, 15, bad)


class TestEngineInvariants:
    def test_fit_returns_finite_weights_on_the_simplex(self, arrays):
        treated, donors, _, _, _, n_pre, start, end = arrays
        fit = sdid_fit_once(treated, donors, n_pre, start, end, n_tr=2)
        assert fit.omega.shape[0] == donors.shape[1]
        assert fit.lam.shape[0] == n_pre
        for w in (fit.omega, fit.lam):
            assert np.all(np.isfinite(w))
            assert w.min() >= -1e-9
            assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.isfinite(fit.pre_rmspe) and fit.pre_rmspe >= 0.0
        assert np.isfinite(fit.zeta) and fit.zeta > 0.0

    def test_counterfactual_spans_the_panel(self, arrays):
        treated, donors, n_periods, _, _, n_pre, start, end = arrays
        fit = sdid_fit_once(treated, donors, n_pre, start, end, n_tr=2)
        assert fit.counterfactual.shape == (n_periods,)
        assert np.all(np.isfinite(fit.counterfactual))

    def test_weights_are_blind_to_the_treated_post_block(self, arrays):
        """The property the analytic sweep rests on."""
        treated, donors, _, _, _, n_pre, start, end = arrays
        base = sdid_fit_once(treated, donors, n_pre, start, end, n_tr=2)
        for es in (-0.5, 0.25, 3.0):
            moved = inject_effect(treated, start, end, es)
            other = sdid_fit_once(moved, donors, n_pre, start, end, n_tr=2)
            np.testing.assert_array_equal(other.omega, base.omega)
            np.testing.assert_array_equal(other.lam, base.lam)

    def test_att_shifts_analytically_with_the_injected_effect(self, arrays):
        treated, donors, _, _, _, n_pre, start, end = arrays
        fit = sdid_fit_once(treated, donors, n_pre, start, end, n_tr=2)
        tau0 = sdid_att(fit, treated, start, end)
        baseline = float(np.mean(treated[start:end + 1]))
        for es in (-0.3, 0.15, 0.75):
            moved = inject_effect(treated, start, end, es)
            assert sdid_att(fit, moved, start, end) == pytest.approx(
                tau0 + es * baseline, rel=1e-10, abs=1e-9)

    def test_placebo_sigma_is_positive_and_effect_invariant(self, arrays):
        treated, donors, _, _, _, n_pre, start, end = arrays
        base = placebo_sigma(treated, donors, n_pre, start, end,
                             n_draws=25, n_tr=2, seed=0)
        assert np.isfinite(base) and base > 0.0
        for es in (-0.4, 0.6):
            moved = inject_effect(treated, start, end, es)
            assert placebo_sigma(moved, donors, n_pre, start, end,
                                 n_draws=25, n_tr=2, seed=0) == base

    def test_matches_the_shipped_sdid_estimator(self, wide):
        """SDIDGEO's engine must be SDID, not a lookalike."""
        from mlsynth import SDID
        from mlsynth.config_models import SDIDConfig

        units = list(wide.columns)
        candidate = frozenset(units[:2])
        n_pre, duration = 80, 14
        start, end = n_pre, n_pre + duration - 1
        sub = wide.iloc[:end + 1]
        idx_name = sub.index.name or "index"
        long = (sub.reset_index()
                .melt(id_vars=idx_name, var_name="location", value_name="Y")
                .rename(columns={idx_name: "date"}))
        long["treated"] = ((long["location"].isin(candidate))
                           & (long["date"] >= sub.index[start])).astype(int)
        shipped = SDID(SDIDConfig(
            df=long, outcome="Y", treat="treated", unitid="location",
            time="date", display_graphs=False, vce="placebo", B=25, seed=0
        )).fit()

        treated = aggregate_treated(wide, candidate, how="mean").to_numpy()
        donors = donor_matrix(wide, candidate).to_numpy()
        fit = sdid_fit_once(treated[:end + 1], donors[:end + 1], n_pre,
                            start, end, n_tr=2)
        assert sdid_att(fit, treated[:end + 1], start, end) == pytest.approx(
            float(shipped.effects.att), rel=1e-6)

    def test_too_few_donors_gives_nan_sigma(self, arrays):
        treated, donors, _, _, _, n_pre, start, end = arrays
        assert np.isnan(placebo_sigma(treated, donors[:, :2], n_pre, start,
                                      end, n_draws=10, n_tr=2, seed=0))


class TestInjectEffect:
    def test_scales_only_the_post_block(self):
        y = np.arange(10.0)
        out = inject_effect(y, 4, 6, 0.5)
        np.testing.assert_array_equal(out[:4], y[:4])
        np.testing.assert_array_equal(out[7:], y[7:])
        np.testing.assert_allclose(out[4:7], y[4:7] * 1.5)

    def test_does_not_mutate_the_input(self):
        y = np.arange(10.0)
        before = y.copy()
        inject_effect(y, 2, 5, 0.3)
        np.testing.assert_array_equal(y, before)

    @pytest.mark.parametrize("window", [(-1, 3), (5, 4), (8, 12)])
    def test_invalid_window_raises(self, window):
        with pytest.raises(MlsynthConfigError):
            inject_effect(np.arange(10.0), window[0], window[1], 0.1)


class TestSimulateBacktest:
    def test_row_schema_and_one_row_per_effect_size(self, arrays):
        treated, donors, n_periods, duration, sim, *_ = arrays
        rows = simulate_backtest(treated, donors, n_periods, duration, sim,
                                 EFFECT_SIZES, n_draws=20, n_tr=2, seed=0)
        assert len(rows) == len(EFFECT_SIZES)
        expected = {"sim", "duration", "effect_size", "p_value",
                    "placebo_mean_effect", "detected_lift", "scaled_l2",
                    "pre_rmspe", "investment"}
        for row in rows:
            assert expected <= set(row)
            assert 0.0 <= row["p_value"] <= 1.0

    def test_p_value_peaks_at_no_effect(self, arrays):
        treated, donors, n_periods, duration, sim, *_ = arrays
        rows = simulate_backtest(treated, donors, n_periods, duration, sim,
                                 EFFECT_SIZES, n_draws=40, n_tr=2, seed=0)
        p = {r["effect_size"]: r["p_value"] for r in rows}
        assert p[0.0] == max(p.values())
        assert p[0.20] < p[0.10] and p[-0.20] < p[-0.10]

    def test_deterministic_under_a_fixed_seed(self, arrays):
        treated, donors, n_periods, duration, sim, *_ = arrays
        kw = dict(n_draws=20, n_tr=2, seed=7)
        a = simulate_backtest(treated, donors, n_periods, duration, sim,
                              EFFECT_SIZES, **kw)
        b = simulate_backtest(treated, donors, n_periods, duration, sim,
                              EFFECT_SIZES, **kw)
        assert [r["p_value"] for r in a] == [r["p_value"] for r in b]

    def test_investment_is_nan_without_cpic(self, arrays):
        treated, donors, n_periods, duration, sim, *_ = arrays
        rows = simulate_backtest(treated, donors, n_periods, duration, sim,
                                 [0.1], n_draws=10, n_tr=2, seed=0)
        assert np.isnan(rows[0]["investment"])

    def test_investment_scales_with_cpic_and_effect(self, arrays):
        treated, donors, n_periods, duration, sim, *_ = arrays
        total = treated * 2.0
        rows = simulate_backtest(treated, donors, n_periods, duration, sim,
                                 [0.1, 0.2], n_draws=10, n_tr=2, seed=0,
                                 cpic=3.0, treated_total=total)
        assert rows[1]["investment"] == pytest.approx(
            2.0 * rows[0]["investment"])

    def test_mismatched_panel_length_raises(self, arrays):
        treated, donors, n_periods, duration, sim, *_ = arrays
        with pytest.raises(MlsynthConfigError, match="n_periods"):
            simulate_backtest(treated[:-1], donors, n_periods, duration, sim,
                              EFFECT_SIZES, n_draws=10, n_tr=2, seed=0)


class TestRunSimulations:
    def test_cube_shape_and_columns(self, wide):
        units = list(wide.columns)
        cands = [frozenset(units[:2]), frozenset(units[2:4])]
        cube = run_simulations(wide, cands, [14], 2, EFFECT_SIZES,
                               n_draws=15, seed=0)
        assert len(cube) == len(cands) * 1 * 2 * len(EFFECT_SIZES)
        assert list(cube.columns)[:4] == ["candidate", "duration", "sim",
                                          "effect_size"]

    def test_empty_candidate_list_gives_an_empty_cube(self, wide):
        cube = run_simulations(wide, [], [14], 2, EFFECT_SIZES, n_draws=10)
        assert cube.empty and "p_value" in cube.columns

    @pytest.mark.parametrize("bad", [0, -3, 2.5, True])
    def test_bad_n_backtests_raises(self, wide, bad):
        units = list(wide.columns)
        with pytest.raises(MlsynthConfigError, match="n_backtests"):
            run_simulations(wide, [frozenset(units[:2])], [14], bad,
                            EFFECT_SIZES, n_draws=10)


class TestConfigValidation:
    def _base(self, panel, **kw):
        args = dict(df=panel, unitid="location", time="date", outcome="Y",
                    treatment_size=2, durations=[14], effect_sizes=[0.1],
                    n_backtests=2)
        args.update(kw)
        return SDIDGEOConfig(**args)

    def test_minimal_config_builds(self, panel):
        assert self._base(panel).treatment_size == 2

    def test_forbids_unknown_fields(self, panel):
        with pytest.raises(Exception):
            self._base(panel, not_a_real_option=1)

    @pytest.mark.parametrize("kw", [
        {"durations": []}, {"durations": [0]}, {"effect_sizes": []},
        {"n_backtests": 0}, {"alpha": 0.0}, {"alpha": 1.0},
        {"power_threshold": 1.5}, {"treatment_size": 0}, {"n_draws": 2},
    ])
    def test_invalid_values_raise(self, panel, kw):
        with pytest.raises(MlsynthConfigError):
            self._base(panel, **kw)

    def test_budget_without_cpic_raises(self, panel):
        with pytest.raises(MlsynthConfigError, match="cpic"):
            self._base(panel, budget=1000.0)


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def result(self, panel):
        return SDIDGEO(SDIDGEOConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=[-0.15, -0.10, 0.0,
                                                            0.10, 0.15, 0.20],
            n_backtests=3, n_draws=25, seed=0,
        )).fit()

    def test_returns_a_design_result(self, result):
        assert isinstance(result, DesignResult)

    def test_selects_a_test_region_of_the_requested_size(self, result):
        assert result.selected_units is not None
        assert len(result.selected_units) == 2
        assert result.assignment["treated"] == result.selected_units

    def test_shortlist_is_ranked_and_finite(self, result):
        shortlist = result.power
        assert not shortlist.empty
        assert shortlist["rank"].is_monotonic_increasing
        assert shortlist["mde"].notna().all()

    def test_design_weights_carry_both_sdid_vectors(self, result):
        weights = result.design_weights
        for vector in (weights.donor_weights, weights.time_weights):
            assert vector, "SDID reports donor and time weights alike"
            w = np.asarray(list(vector.values()), dtype=float)
            assert w.min() >= -1e-9
            assert w.sum() == pytest.approx(1.0, abs=1e-8)

    def test_donor_weights_exclude_the_treated_markets(self, result):
        assert not (set(result.design_weights.donor_weights)
                    & set(result.selected_units))

    def test_metadata_reports_the_search(self, result):
        meta = result.metadata
        assert meta["treatment_size"] == 2
        assert meta["n_candidates"] > 0
        assert meta["pre_periods"] > 0

    def test_report_is_absent_before_the_experiment_runs(self, result):
        assert result.report is None

    def test_reproducible(self, panel):
        kw = dict(df=panel, unitid="location", time="date", outcome="Y",
                  treatment_size=2, durations=[14], effect_sizes=[0.0, 0.15],
                  n_backtests=2, n_draws=15, seed=3)
        a = SDIDGEO(SDIDGEOConfig(**kw)).fit()
        b = SDIDGEO(SDIDGEOConfig(**kw)).fit()
        assert a.selected_units == b.selected_units

    def test_forced_and_forbidden_markets_are_respected(self, panel):
        units = sorted(panel["location"].unique())
        forced, banned = units[0], units[1]
        res = SDIDGEO(SDIDGEOConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=[0.0, 0.15],
            n_backtests=2, n_draws=15, seed=0,
            to_be_treated=[forced], not_to_be_treated=[banned],
        )).fit()
        assert forced in res.selected_units
        assert banned not in res.selected_units


class TestDegenerateInputs:
    def test_treatment_size_exceeding_the_panel_raises(self, panel):
        with pytest.raises(MlsynthConfigError):
            SDIDGEO(SDIDGEOConfig(
                df=panel, unitid="location", time="date", outcome="Y",
                treatment_size=500, durations=[14], effect_sizes=[0.1],
                n_backtests=2, n_draws=15,
            )).fit()

    def test_duration_longer_than_the_panel_raises(self, panel):
        with pytest.raises(MlsynthConfigError):
            SDIDGEO(SDIDGEOConfig(
                df=panel, unitid="location", time="date", outcome="Y",
                treatment_size=2, durations=[500], effect_sizes=[0.1],
                n_backtests=2, n_draws=15,
            )).fit()

    def test_unknown_forced_market_raises(self, panel):
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            SDIDGEO(SDIDGEOConfig(
                df=panel, unitid="location", time="date", outcome="Y",
                treatment_size=2, durations=[14], effect_sizes=[0.1],
                n_backtests=2, n_draws=15, to_be_treated=["atlantis"],
            )).fit()

    def test_candidate_with_no_donors_left_raises(self, wide):
        with pytest.raises(MlsynthDataError, match="No donor"):
            donor_matrix(wide, frozenset(wide.columns))

    def test_unknown_candidate_unit_raises(self, wide):
        with pytest.raises(MlsynthDataError):
            aggregate_treated(wide, frozenset(["nowhere"]))


class TestPlots:
    """The design plots: a power curve and the fit behind it."""

    @staticmethod
    def _agg():
        import matplotlib
        matplotlib.use("Agg")

    def test_design_plot_has_both_panels(self, panel):
        self._agg()
        from mlsynth import plot_sdidgeo_design

        result = SDIDGEO(SDIDGEOConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14],
            effect_sizes=[-0.2, -0.1, 0.0, 0.1, 0.2],
            n_backtests=3, n_draws=20, seed=0,
        )).fit()
        fig = plot_sdidgeo_design(result, power_threshold=0.8)
        assert len(fig.axes) == 2
        assert "power" in fig.axes[0].get_ylabel().lower()
        assert fig.axes[1].get_lines(), "the fit panel should draw the series"

    def test_mde_ranking_plot_marks_the_winner(self, panel):
        self._agg()
        from mlsynth import plot_mde_ranking

        result = SDIDGEO(SDIDGEOConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14],
            effect_sizes=[-0.2, -0.1, 0.0, 0.1, 0.2],
            n_backtests=3, n_draws=20, seed=0,
        )).fit()
        fig = plot_mde_ranking(result, top=5)
        bars = fig.axes[0].patches
        assert 0 < len(bars) <= 5
        reds = [b for b in bars if b.get_facecolor()[:3] == (1.0, 0.0, 0.0)]
        assert len(reds) == 1, "exactly the winning region is highlighted"

    def test_power_table_is_retained_for_the_curve(self, panel):
        result = SDIDGEO(SDIDGEOConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=[0.0, 0.15],
            n_backtests=2, n_draws=15, seed=0,
        )).fit()
        table = result.search.power_table
        assert not table.empty
        assert set(table["effect_size"]) == {0.0, 0.15}
        assert table["power"].between(0.0, 1.0).all()

    def test_plot_without_a_winner_raises(self, panel):
        self._agg()
        from mlsynth import plot_mde_ranking, plot_sdidgeo_design

        # An effect grid of only zero detects nothing, so nothing is ranked.
        result = SDIDGEO(SDIDGEOConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=[0.0],
            n_backtests=2, n_draws=15, seed=0,
        )).fit()
        assert result.selected_units is None
        with pytest.raises(ValueError, match="no winning design"):
            plot_sdidgeo_design(result)
        with pytest.raises(ValueError, match="shortlist is empty"):
            plot_mde_ranking(result)


class TestMultipleTreatmentSizes:
    """``treatment_size`` accepts a list, scanning several region sizes at once.

    GeoLift's ``N = c(2, 3, 4)``: candidates of every requested size are
    generated, scored on the same grid, and ranked against each other, so a
    three-market region competes with a two-market one on its MDE.
    """

    def _design(self, panel, size, **kw):
        args = dict(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=size, durations=[14],
            effect_sizes=[-0.2, -0.1, 0.0, 0.1, 0.2],
            n_backtests=2, n_draws=15, seed=0,
        )
        args.update(kw)
        return SDIDGEO(SDIDGEOConfig(**args)).fit()

    def test_scalar_still_works(self, panel):
        result = self._design(panel, 2)
        assert len(result.selected_units) == 2
        assert {len(c.units) for c in result.search.candidates} == {2}

    def test_list_generates_every_requested_size(self, panel):
        result = self._design(panel, [2, 3, 4])
        assert {len(c.units) for c in result.search.candidates} == {2, 3, 4}

    def test_shortlist_reports_the_size_of_each_candidate(self, panel):
        result = self._design(panel, [2, 3])
        shortlist = result.power
        assert "treatment_size" in shortlist.columns
        for _, row in shortlist.iterrows():
            assert row["treatment_size"] == len(row["candidate"])
        assert set(shortlist["treatment_size"]) <= {2, 3}

    def test_sizes_are_ranked_against_each_other(self, panel):
        """One ranking over the pooled field, not one ranking per size."""
        result = self._design(panel, [2, 3, 4])
        shortlist = result.power
        assert shortlist["rank"].is_monotonic_increasing
        assert len(shortlist) == shortlist["rank"].count()
        # The winner is whichever size scored best, so its size must be one of
        # the requested ones and need not be the smallest.
        assert len(result.selected_units) in {2, 3, 4}

    def test_metadata_records_the_scanned_sizes(self, panel):
        result = self._design(panel, [2, 4])
        assert result.metadata["treatment_sizes"] == [2, 4]

    def test_duplicates_and_order_are_normalised(self, panel):
        result = self._design(panel, [3, 2, 3])
        assert result.metadata["treatment_sizes"] == [2, 3]

    def test_each_size_uses_its_own_treated_count(self, panel):
        """A three-market region must be fit as three treated units.

        ``n_tr`` feeds the SDID ridge as ``(n_tr * T_post)^(1/4)``, so a size
        mix that ignored it would regularise every candidate identically.
        """
        result = self._design(panel, [2, 4])
        by_size = {len(c.units): c for c in result.search.candidates}
        assert set(by_size) == {2, 4}
        for size, design in by_size.items():
            assert len(design.units) == size
            assert not set(design.units) & set(design.weights.donor_weights)

    def test_forced_markets_larger_than_a_size_raise(self, panel):
        units = sorted(panel["location"].unique())
        with pytest.raises(MlsynthConfigError, match="to_be_treated"):
            self._design(panel, [2, 3], to_be_treated=units[:3])

    @pytest.mark.parametrize("bad", [[], [0], [2, -1], [2, 0]])
    def test_invalid_sizes_raise(self, panel, bad):
        with pytest.raises(MlsynthConfigError):
            self._design(panel, bad)

    def test_size_exceeding_the_panel_raises(self, panel):
        with pytest.raises(MlsynthConfigError):
            self._design(panel, [2, 500])


class TestPlaceboWarmStart:
    """The placebo loop chains its warm start; the sigma must not move.

    Consecutive draws differ only in which donors were reassigned, so each
    draw's weights seed the next. ``solve_simplex_qp`` discards an infeasible
    or wrongly sized seed, so the chain can only change pivot count -- and
    ``sdid_fit_once`` still solves each backtest cold, which gives an
    independent path to the same number.
    """

    @staticmethod
    def _cold_sigma(y, Y0, n_pre, start, end, *, n_draws, n_tr, seed):
        """The same draws, each refit from scratch through ``sdid_fit_once``."""
        Y0 = np.asarray(Y0, dtype=float)
        rng = np.random.default_rng(seed)
        taus = []
        for _ in range(n_draws):
            idx = rng.choice(Y0.shape[1], size=n_tr, replace=False)
            mask = np.ones(Y0.shape[1], dtype=bool)
            mask[idx] = False
            pseudo_y = Y0[:, idx].mean(axis=1)
            fit = sdid_fit_once(pseudo_y, Y0[:, mask], n_pre, start, end,
                                n_tr=n_tr)
            taus.append(sdid_att(fit, pseudo_y, start, end))
        return float(np.std(np.asarray(taus), ddof=1))

    @pytest.mark.parametrize("sim", [1, 3, 5])
    def test_chained_sigma_matches_cold_refitting(self, wide, sim):
        units = list(wide.columns)
        candidate = frozenset(units[:2])
        treated = aggregate_treated(wide, candidate, how="mean").to_numpy()
        donors = donor_matrix(wide, candidate).to_numpy()
        n_pre = wide.shape[0] - 14 - sim + 1
        start, end = n_pre, n_pre + 13
        kw = dict(n_draws=40, n_tr=2, seed=0)
        assert placebo_sigma(treated, donors, n_pre, start, end,
                             **kw) == pytest.approx(
            self._cold_sigma(treated, donors, n_pre, start, end, **kw),
            rel=1e-9)

    def test_still_deterministic_across_calls(self, arrays):
        treated, donors, _, _, _, n_pre, start, end = arrays
        kw = dict(n_draws=30, n_tr=2, seed=4)
        first = placebo_sigma(treated, donors, n_pre, start, end, **kw)
        second = placebo_sigma(treated, donors, n_pre, start, end, **kw)
        assert first == second


class TestPlanningMDE:
    """The winner's MDE is a selected minimum; the validation one is not.

    ``compute_rank`` returns the smallest MDE in the field, so the region most
    likely to be picked is the one whose estimate came out low by luck. The
    validation backtests sit deeper in history and take no part in the search, so
    re-scoring the winner on them gives a figure that was not selected on.
    """

    def _design(self, panel, **kw):
        args = dict(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14],
            effect_sizes=[-0.2, -0.1, 0.0, 0.1, 0.2],
            n_backtests=4, n_validation_backtests=4, n_draws=20, seed=0,
        )
        args.update(kw)
        return SDIDGEO(SDIDGEOConfig(**args)).fit()

    def test_both_numbers_are_reported(self, panel):
        result = self._design(panel)
        assert result.metadata["winner_mde_optimistic"] is not None
        assert result.metadata["winner_mde_planning"] is not None
        assert result.metadata["n_validation_backtests"] == 4
        assert result.search.winner.mde_planning == pytest.approx(
            result.metadata["winner_mde_planning"])

    def test_planning_mde_is_an_effect_size_from_the_grid(self, panel):
        result = self._design(panel)
        grid = [-0.2, -0.1, 0.0, 0.1, 0.2]
        assert float(result.metadata["winner_mde_planning"]) in grid

    def test_disabled_by_zero(self, panel):
        result = self._design(panel, n_validation_backtests=0)
        assert result.metadata["winner_mde_planning"] is None
        assert result.search.winner.mde_planning is None
        assert result.metadata["winner_mde_optimistic"] is not None

    def test_only_the_winner_is_rescored(self, panel):
        """Scoring every candidate on the validation would reintroduce selection."""
        result = self._design(panel)
        rescored = [c for c in result.search.candidates
                    if c.mde_planning is not None]
        assert len(rescored) == 1
        assert rescored[0].units == result.selected_units

    def test_none_when_the_panel_cannot_carry_the_backtests(self, panel):
        # 105 periods; duration 14 with 8 backtests needs 8 more backtests to
        # reach depth 16, which fits -- so ask for a depth the panel cannot hold.
        result = self._design(panel, n_backtests=4, n_validation_backtests=200)
        assert result.metadata["winner_mde_planning"] is None
        assert result.metadata["winner_mde_optimistic"] is not None

    def test_validation_uses_backtests_the_search_did_not(self, panel):
        """Changing only the validation depth must not move the selection."""
        a = self._design(panel, n_validation_backtests=4)
        b = self._design(panel, n_validation_backtests=6)
        assert a.selected_units == b.selected_units
        assert a.metadata["winner_mde_optimistic"] == b.metadata["winner_mde_optimistic"]

    @pytest.mark.parametrize("bad", [-1, -10])
    def test_negative_validation_backtests_raises(self, panel, bad):
        with pytest.raises(MlsynthConfigError, match="n_validation_backtests"):
            self._design(panel, n_validation_backtests=bad)

    def test_reproducible(self, panel):
        a = self._design(panel)
        b = self._design(panel)
        assert (a.metadata["winner_mde_planning"]
                == b.metadata["winner_mde_planning"])
