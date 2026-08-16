"""PPSCM reports the shape of the balance problem, not only its residual.

A pre-treatment imbalance near zero means two different things depending on how
many constraints the program had. With more donors than balance periods, the
simplex is wide enough that exact balance is generically attainable, and the
residual describes the geometry instead of the fit. With fewer, the residual is
the fit.

On ``diff-diff``'s own panels the two cases sit side by side. ``mpdta`` reports
``max_donors=480`` against ``balance_periods=1`` -- its earliest cohort adopts
in the second period, so one pre-period is all the binding cohort has -- and
reaches 3.4e-06. ``castle_doctrine`` reports 32 against 5 and cannot get below
2.1e-02. Before #476 the estimator reported those residuals identically.

The tests below fix what the estimator must report so a reader can tell them
apart, and are written against panels whose shape is chosen, not measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM


def panel(n_units=40, n_per=20, onsets=(12, 14), k=2, seed=0):
    """A staggered panel with a chosen number of periods and donors."""
    rng = np.random.default_rng(seed)
    assign = {u: g for i, g in enumerate(onsets) for u in range(i * k, (i + 1) * k)}
    rows = []
    for u in range(n_units):
        base, walk = rng.normal(10, 2), np.cumsum(rng.normal(0, 0.3, n_per))
        g = assign.get(u)
        for t in range(1, n_per + 1):
            on = g is not None and t >= g
            rows.append((u, t, base + walk[t - 1] + 0.5 * rng.normal() + 2.0 * on,
                         int(on)))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d"])


def fit(df, **over):
    return PPSCM({"df": df, "outcome": "y", "treat": "d", "unitid": "id",
                  "time": "time", "display_graphs": False,
                  "run_inference": False, **over}).fit()


class TestTheReportedShape:
    def test_the_design_reports_donors_and_balance_periods(self):
        d = fit(panel()).design
        assert isinstance(d.max_donors, int) and d.max_donors > 0
        assert isinstance(d.balance_periods, int) and d.balance_periods > 0

    def test_the_donor_count_is_the_widest_pool_any_cohort_had(self):
        """The widest pool decides whether the program could interpolate, so a
        cohort with a narrow pool must not mask one with a wide pool.

        Measured against ``eligible_donors`` and not against
        ``per_unit[...].donor_weights``: the latter holds the weights that came
        back non-zero, which on a wide pool is a small fraction of it, so it
        cannot see the quantity this field is about. The panel is shaped so the
        pools genuinely differ -- widely spaced cohorts and a short lead window
        let the late cohort serve as a donor for the early one, and not the
        reverse.
        """
        from mlsynth.utils.ppscm_helpers.engine import eligible_donors
        from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs

        df = panel(n_units=40, n_per=20, onsets=(5, 18), k=2)
        res = fit(df)
        inputs = prepare_ppscm_inputs(df, outcome="y", treat="d",
                                      unitid="id", time="time")
        pools = [len(eligible_donors(inputs.trt, int(a), res.design.n_leads,
                                     "window"))
                 for a in sorted({int(v) for v in inputs.trt[np.isfinite(inputs.trt)]})]
        assert min(pools) < max(pools), (
            "the panel must give the cohorts different pools or this cannot "
            f"distinguish widest from narrowest; got {pools}")
        assert res.design.max_donors == max(pools)

    def test_balance_periods_follows_n_lags(self):
        df = panel(n_per=20, onsets=(12, 14))
        assert fit(df, n_lags=3).design.balance_periods == 3
        assert fit(df, n_lags=6).design.balance_periods == 6

    def test_balance_periods_is_capped_by_the_earliest_cohorts_history(self):
        """A cohort adopting at t=4 has 3 pre-periods however many lags are
        asked for, and it is the cohort with the least history that binds."""
        df = panel(n_per=20, onsets=(4, 16), k=2)
        assert fit(df, n_lags=10).design.balance_periods == 3


class TestUnderdetermined:
    def test_a_wide_pool_against_few_periods_is_flagged(self):
        df = panel(n_units=40, n_per=20, onsets=(12, 14), k=2)
        d = fit(df, n_lags=2).design
        assert d.max_donors - 1 > d.balance_periods
        assert d.underdetermined is True

    def test_a_narrow_pool_against_many_periods_is_not(self):
        df = panel(n_units=8, n_per=20, onsets=(12, 14), k=2)
        d = fit(df, n_lags=12).design
        assert d.max_donors - 1 <= d.balance_periods
        assert d.underdetermined is False

    def test_the_flag_is_the_stated_comparison_and_not_a_heuristic(self):
        for n_units, n_lags in ((40, 2), (40, 15), (8, 12), (8, 2)):
            d = fit(panel(n_units=n_units), n_lags=n_lags).design
            assert d.underdetermined == (d.max_donors - 1 > d.balance_periods)

    def test_an_underdetermined_program_reaches_a_far_smaller_residual(self):
        """The reason the flag exists: the same panel, balanced against few
        periods and against many, gives residuals that are not comparable."""
        df = panel(n_units=40, n_per=20, onsets=(12, 14), k=2)
        wide = fit(df, n_lags=2).design
        tight = fit(df, n_lags=10).design
        # Both are underdetermined here -- 36 donors outruns even ten periods --
        # which is the point: the flag says the residuals are not comparable,
        # and it does not claim the tighter one is well determined.
        assert wide.underdetermined and tight.underdetermined
        assert wide.global_l2 < tight.global_l2

    def test_uniform_weights_solve_no_program_so_the_flag_is_about_the_pool(self):
        """``donor_weights="uniform"`` has nothing to interpolate with, so the
        shape is still reported and still describes the pool that was available."""
        d = fit(panel(), donor_weights="uniform").design
        assert d.max_donors > 0 and d.balance_periods > 0
        assert isinstance(d.underdetermined, bool)


class TestItTravelsWithTheResult:
    def test_the_shape_survives_inference(self):
        d = fit(panel(), run_inference=True, inference_method="bootstrap",
                n_boot=50).design
        assert d.max_donors > 0 and d.balance_periods > 0

    def test_time_cohort_reports_the_pooled_cohorts_shape(self):
        df = panel(n_units=40, onsets=(12, 12, 14), k=1)
        a = fit(df).design
        b = fit(df, time_cohort=True).design
        assert a.max_donors == b.max_donors
        assert a.balance_periods == b.balance_periods
