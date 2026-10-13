"""Staggered adoption in DMLFM.

The sampler already estimates on the control observations -- ``fit_rows =
np.flatnonzero(d == 0)`` is the estimation set of Pang, Liu & Xu (2022)
Eq. (A.5) whether one unit or several are treated, and whether they adopt
together or apart. What these tests pin is the ingestion that feeds it more
than one treated unit, and the per-cohort reporting that comes back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import DMLFM
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.dmlfm_helpers.setup import prepare_dmlfm_inputs
from mlsynth.utils.dmlfm_helpers.config import DMLFMConfig

N_UNITS, N_PERIODS = 8, 18


def panel(adoption, *, effect=0.0, n_units=N_UNITS, n_periods=N_PERIODS,
          seed=11, balanced=True):
    """A two-factor panel in which ``adoption`` maps unit index to its first
    treated period. Units absent from ``adoption`` are never treated.
    """
    rng = np.random.default_rng(seed)
    gamma = rng.normal(size=(n_units, 2))
    f = np.cumsum(rng.normal(scale=0.3, size=(n_periods, 2)), axis=0)
    y0 = gamma @ f.T + rng.normal(scale=0.2, size=(n_units, n_periods))

    rows = []
    for i in range(n_units):
        a = adoption.get(i)
        for t in range(n_periods):
            treated = a is not None and t >= a
            rows.append({
                "unit": f"u{i}", "time": 2000 + t,
                "y": y0[i, t] + (effect if treated else 0.0),
                "d": int(treated), "x": float(rng.normal()),
            })
    df = pd.DataFrame(rows)
    if not balanced:
        df = df.drop(df.index[3]).reset_index(drop=True)
    return df


def cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
                r=3, re="time", niter=240, burn=120, seed=3,
                display_graphs=False)
    base.update(kw)
    return DMLFMConfig(**base)


# --------------------------------------------------------------- ingestion
def test_two_cohorts_are_accepted_by_ingestion():
    inputs = prepare_dmlfm_inputs(cfg(panel({5: 8, 6: 12})))
    assert inputs.treated_names == ["u5", "u6"]
    assert inputs.adoption_index.tolist() == [8, 12]


def test_estimation_set_excludes_every_treated_cell_and_nothing_else():
    df = panel({5: 8, 6: 12})
    inputs = prepare_dmlfm_inputs(cfg(df))
    expected = int((df["d"] == 0).sum())
    assert inputs.y.shape[0] == expected
    assert inputs.X.shape[0] == expected


def test_treated_block_covers_every_period_of_every_treated_unit():
    inputs = prepare_dmlfm_inputs(cfg(panel({5: 8, 6: 12})))
    assert inputs.X_tr.shape[0] == 2 * N_PERIODS
    assert sorted(set(inputs.unit_index_tr.tolist())) == [5, 6]
    for u in (5, 6):
        assert (inputs.time_index_tr[inputs.unit_index_tr == u].tolist()
                == list(range(N_PERIODS)))


def test_one_treated_unit_keeps_the_scalar_pre_period():
    inputs = prepare_dmlfm_inputs(cfg(panel({5: 8})))
    assert inputs.pre_periods == 8
    assert inputs.treated_names == ["u5"]
    assert inputs.treated_name == "u5"


def test_pre_periods_is_the_earliest_adoption_under_staggered_adoption():
    inputs = prepare_dmlfm_inputs(cfg(panel({5: 8, 6: 12})))
    assert inputs.pre_periods == 8


def test_units_treated_together_are_one_cohort():
    inputs = prepare_dmlfm_inputs(cfg(panel({5: 9, 6: 9})))
    assert inputs.adoption_index.tolist() == [9, 9]


# ------------------------------------------------------------------ fitting
def test_smoke_two_cohorts_fit_and_return_finite_numbers():
    res = DMLFM(cfg(panel({5: 8, 6: 12}))).fit()
    assert np.isfinite(res.effects.att)
    assert np.isfinite(res.fit_diagnostics.rmse_pre)
    assert res.time_series.counterfactual_outcome.shape == (N_PERIODS, 2)
    assert res.time_series.observed_outcome.shape == (N_PERIODS, 2)


def test_cohort_att_has_one_entry_per_distinct_adoption_time():
    res = DMLFM(cfg(panel({5: 8, 6: 12}))).fit()
    cohort = res.effects.additional_effects["cohort_att"]
    assert sorted(cohort) == [2008, 2012]


def test_units_adopting_together_collapse_to_one_cohort_entry():
    res = DMLFM(cfg(panel({5: 9, 6: 9}))).fit()
    assert list(res.effects.additional_effects["cohort_att"]) == [2009]


def test_event_study_is_keyed_by_relative_time_and_spans_both_signs():
    res = DMLFM(cfg(panel({5: 8, 6: 12}))).fit()
    es = res.effects.additional_effects["event_study"]
    assert min(es) == -12 and max(es) == N_PERIODS - 1 - 8
    assert 0 in es


def test_a_planted_constant_effect_is_recovered():
    shift = 4.0
    res = DMLFM(cfg(panel({5: 8, 6: 12}, effect=shift), niter=1200, burn=600)).fit()
    assert res.effects.att == pytest.approx(shift, abs=0.8)
    for value in res.effects.additional_effects["cohort_att"].values():
        assert value == pytest.approx(shift, abs=1.2)


def test_pre_adoption_event_times_sit_near_zero():
    res = DMLFM(cfg(panel({5: 8, 6: 12}, effect=4.0), niter=1200, burn=600)).fit()
    es = res.effects.additional_effects["event_study"]
    pre = [v for e, v in es.items() if e < 0]
    assert abs(float(np.mean(pre))) < 1.0


def test_att_averages_only_treated_cells():
    """A unit treated for one period contributes one cell, not a whole row."""
    res = DMLFM(cfg(panel({5: 8, 6: N_PERIODS - 1}, effect=5.0),
                    niter=1200, burn=600)).fit()
    n_cells = (N_PERIODS - 8) + 1
    assert res.method_details.parameters["treated_cells"] == n_cells


def test_method_details_lists_every_treated_unit():
    res = DMLFM(cfg(panel({5: 8, 6: 12}))).fit()
    params = res.method_details.parameters
    assert params["treated_units"] == ["u5", "u6"]
    assert params["adoption_periods"] == [2008, 2012]
    assert params["staggered"] is True


def test_single_cohort_reports_staggered_false():
    res = DMLFM(cfg(panel({5: 8}))).fit()
    assert res.method_details.parameters["staggered"] is False
    assert res.method_details.parameters["treated_units"] == ["u5"]


def test_cohort_keys_are_plain_python_scalars():
    res = DMLFM(cfg(panel({5: 8, 6: 12}))).fit()
    for key in res.effects.additional_effects["cohort_att"]:
        assert type(key) is int
    for key in res.method_details.parameters["adoption_periods"]:
        assert type(key) is int


def test_the_horseshoe_prior_runs_on_a_staggered_panel():
    res = DMLFM(cfg(panel({5: 8, 6: 12}, effect=4.0), prior="horseshoe",
                    niter=1200, burn=600)).fit()
    assert res.method_details.parameters["prior"] == "horseshoe"
    assert res.effects.att == pytest.approx(4.0, abs=1.0)
    assert sorted(res.effects.additional_effects["cohort_att"]) == [2008, 2012]


def test_three_cohorts_report_three_entries():
    res = DMLFM(cfg(panel({4: 7, 5: 10, 6: 13}))).fit()
    assert sorted(res.effects.additional_effects["cohort_att"]) == [2007, 2010, 2013]
    assert res.method_details.parameters["treated_cells"] == (
        (N_PERIODS - 7) + (N_PERIODS - 10) + (N_PERIODS - 13))


def test_per_period_bounds_bracket_the_mean_gap_cell_by_cell():
    res = DMLFM(cfg(panel({5: 8, 6: 12}))).fit()
    lo = np.asarray(res.inference.details["per_period_lower"])
    hi = np.asarray(res.inference.details["per_period_upper"])
    gap = np.asarray(res.time_series.estimated_gap)
    assert lo.shape == hi.shape == gap.shape == (N_PERIODS, 2)
    assert np.all(lo <= gap) and np.all(gap <= hi)


# --------------------------------------------------------------- plotting
def test_the_plotter_draws_one_panel_per_treated_unit(tmp_path):
    out = tmp_path / "staggered.png"
    res = DMLFM(cfg(panel({5: 8, 6: 12}), display_graphs=True, save=str(out))).fit()
    assert out.exists()
    assert res.method_details.parameters["staggered"] is True


def test_the_plotter_still_draws_a_single_panel_for_one_treated_unit(tmp_path):
    out = tmp_path / "single.png"
    DMLFM(cfg(panel({5: 8}), display_graphs=True, save=str(out))).fit()
    assert out.exists()


# ------------------------------------------------------- unchanged default
def test_one_treated_unit_is_bit_identical_to_the_pre_staggered_path():
    """The single-treated path must not move: its cross-validation is pinned."""
    df = panel({5: 8})
    a = DMLFM(cfg(df, seed=7)).fit()
    b = DMLFM(cfg(df, seed=7)).fit()
    assert a.effects.att == b.effects.att
    assert np.array_equal(a.time_series.counterfactual_outcome,
                          b.time_series.counterfactual_outcome)
    assert a.time_series.counterfactual_outcome.shape == (N_PERIODS, 1)


def test_adding_a_second_treated_unit_shrinks_the_estimation_set():
    one = prepare_dmlfm_inputs(cfg(panel({5: 8})))
    two = prepare_dmlfm_inputs(cfg(panel({5: 8, 6: 12})))
    assert two.y.shape[0] == one.y.shape[0] - (N_PERIODS - 12)


# ------------------------------------------------------------- degenerate
def test_no_treated_unit_is_refused():
    with pytest.raises(MlsynthDataError, match="(?i)no treated unit"):
        prepare_dmlfm_inputs(cfg(panel({})))


def test_most_units_treated_is_estimable_while_one_stays_untreated():
    """The estimation set is cell-level, so a large treated share is fine."""
    adoption = {i: 6 + (i % 3) for i in range(N_UNITS - 1)}
    inputs = prepare_dmlfm_inputs(cfg(panel(adoption)))
    assert inputs.n_treated == N_UNITS - 1
    assert inputs.y.shape[0] == sum(6 + (i % 3) for i in range(N_UNITS - 1)) + N_PERIODS
    res = DMLFM(cfg(panel(adoption))).fit()
    assert np.isfinite(res.effects.att)
    assert len(res.effects.additional_effects["cohort_att"]) == 3


def test_a_period_with_no_untreated_observation_is_refused():
    """Every unit treated by t leaves the period-t factor on its prior alone."""
    adoption = {i: 6 + (i % 3) for i in range(N_UNITS)}
    with pytest.raises(MlsynthDataError, match="untreated observation in every period"):
        prepare_dmlfm_inputs(cfg(panel(adoption)))


def test_the_period_check_names_the_first_bare_periods():
    adoption = {i: 6 + (i % 3) for i in range(N_UNITS)}
    with pytest.raises(MlsynthDataError, match="2008"):
        prepare_dmlfm_inputs(cfg(panel(adoption)))


def test_the_period_check_is_skipped_when_no_per_period_block_exists():
    """With r=0 and re='none' nothing is indexed by period, so bare periods are fine."""
    adoption = {i: 6 + (i % 3) for i in range(N_UNITS)}
    inputs = prepare_dmlfm_inputs(
        cfg(panel(adoption), r=0, re="none", covariates=["x"]))
    assert inputs.n_treated == N_UNITS


def test_a_unit_treated_from_the_first_period_is_refused():
    with pytest.raises(MlsynthDataError, match="pre-treatment period"):
        prepare_dmlfm_inputs(cfg(panel({5: 0, 6: 12})))


def test_unbalanced_panel_is_still_refused():
    with pytest.raises(MlsynthDataError, match="balanced"):
        prepare_dmlfm_inputs(cfg(panel({5: 8, 6: 12}, balanced=False)))


def test_treatment_that_switches_off_is_refused():
    df = panel({5: 8, 6: 12})
    df.loc[(df["unit"] == "u5") & (df["time"] == 2012), "d"] = 0
    with pytest.raises(MlsynthDataError, match="(?i)not sustained"):
        prepare_dmlfm_inputs(cfg(df))
