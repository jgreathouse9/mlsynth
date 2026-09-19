"""Staggered-adoption Forward DID.

FDID (Li 2023) is written for one treated unit; Web Appendix C extends it to
several by running the method per unit and averaging. These tests pin the
additions that extension needs to be usable:

* donor eligibility enforced at selection as well as estimation, so a donor
  that adopts inside a treated unit's pre-window never enters its criterion;
* selection per unit, on the cohort mean, or on a convex combination;
* ``ATT(g, h)`` building blocks aggregated on a balanced event clock;
* a joint covariance across treated units, because they draw on one donor pool
  and their estimates are therefore dependent.

The panels here are deterministic wherever an exact answer is available, so the
assertions are on invariants and exact identities, not on tolerances around
simulated noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import FDID
from mlsynth.config_models import EffectResult
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.fdid_helpers import FDIDResults
from mlsynth.utils.fdid_helpers.config import FDIDConfig


# ─── fixtures ─────────────────────────────────────────────────────────────

def _factor(T: int) -> np.ndarray:
    """A deterministic, non-linear common path."""
    t = np.arange(T, dtype=float)
    return 5.0 + 0.3 * t + 2.0 * np.sin(t / 3.0)


def exact_panel(T: int = 20, adopt=(("A", 10), ("B", 12)), n_donor: int = 4,
                ramp: bool = True, anticipation: int = 0) -> pd.DataFrame:
    """Noiseless panel: every donor is the common path plus a level shift.

    Any single donor therefore satisfies parallel trends exactly, so forward
    selection stops at one donor and ``ATT(g, h)`` is recovered exactly.
    Effects ramp as ``tau(h) = 1 + h`` when ``ramp``, else a constant 1.
    """
    f = _factor(T)
    rows = []
    for j in range(n_donor):
        for t in range(T):
            rows.append({"unit": f"d{j}", "t": t, "y": f[t] + j, "w": 0})
    for k, (name, g) in enumerate(adopt):
        for t in range(T):
            y = f[t] + 100.0 * (k + 1)
            if t >= g:
                y += (1.0 + (t - g)) if ramp else 1.0
            elif anticipation and t >= g - anticipation:
                y += 0.5
            rows.append({"unit": name, "t": t, "y": y, "w": int(t >= g)})
    return pd.DataFrame(rows)


def noisy_panel(T: int = 40, adopt=(("A", 28), ("B", 32)), n_donor: int = 6,
                seed: int = 0) -> pd.DataFrame:
    """Factor panel with idiosyncratic noise, for the inference assertions."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal(T))
    rows = []
    for j in range(n_donor):
        load = 1.0 + (0.2 if j % 2 else -0.2)
        col = 3.0 * j + load * f + rng.standard_normal(T) * 0.5
        for t in range(T):
            rows.append({"unit": f"d{j}", "t": t, "y": col[t], "w": 0})
    for name, g in adopt:
        col = 50.0 + f + rng.standard_normal(T) * 0.5
        col[g:] += 2.0
        for t in range(T):
            rows.append({"unit": name, "t": t, "y": col[t], "w": int(t >= g)})
    return pd.DataFrame(rows)


def cfg(df: pd.DataFrame, **kw) -> dict:
    base = dict(df=df, outcome="y", treat="w", unitid="unit", time="t",
                display_graphs=False)
    base.update(kw)
    return base


# ─── smoke ────────────────────────────────────────────────────────────────

def test_staggered_panel_fits_and_returns_an_effect_result():
    res = FDID(cfg(exact_panel())).fit()
    assert isinstance(res, EffectResult)
    assert np.isfinite(res.att)
    assert len(res.units) == 2
    assert res.event_study.att.shape == res.event_study.horizons.shape


def test_single_treated_panel_still_returns_the_single_unit_container():
    df = exact_panel(adopt=(("A", 10),))
    res = FDID(cfg(df)).fit()
    assert isinstance(res, FDIDResults)
    assert hasattr(res, "fdid")


# ─── identification: the estimates are right on a noiseless panel ─────────

def test_event_study_recovers_a_known_ramp_exactly():
    res = FDID(cfg(exact_panel(ramp=True))).fit()
    h = res.event_study.horizons
    assert np.allclose(res.event_study.att, 1.0 + h, atol=1e-8)


def test_overall_att_is_the_mean_of_the_balanced_event_path():
    res = FDID(cfg(exact_panel(ramp=True))).fit()
    assert res.overall_att == pytest.approx(float(res.event_study.att.mean()),
                                            abs=1e-10)


def test_default_horizon_is_the_largest_every_cohort_supports():
    # T = 20, adoptions at 10, 12 and 15 -> the last cohort supports h <= 4.
    df = exact_panel(T=20, adopt=(("A", 10), ("B", 12), ("C", 15)))
    res = FDID(cfg(df)).fit()
    assert res.event_study.horizons.tolist() == [0, 1, 2, 3, 4]


def test_max_horizon_beyond_what_a_cohort_supports_is_rejected():
    df = exact_panel(T=20, adopt=(("A", 10), ("B", 15)))
    with pytest.raises(MlsynthEstimationError, match="horizon"):
        FDID(cfg(df, max_horizon=9)).fit()


# ─── condition II: donor eligibility ──────────────────────────────────────

def test_eventually_treated_units_never_enter_a_donor_set():
    df = exact_panel(T=20, adopt=(("A", 10), ("B", 12), ("C", 15)))
    res = FDID(cfg(df)).fit()
    treated = {"A", "B", "C"}
    for u in res.units:
        assert treated.isdisjoint(u.selected_names)


def test_a_panel_with_no_never_treated_donor_is_refused():
    rows = []
    f = _factor(12)
    for k, (name, g) in enumerate((("A", 6), ("B", 8))):
        for t in range(12):
            rows.append({"unit": name, "t": t, "y": f[t] + 10 * k,
                         "w": int(t >= g)})
    with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
        FDID(cfg(pd.DataFrame(rows))).fit()


# ─── condition III: anticipation ──────────────────────────────────────────

def test_anticipation_trims_the_window_the_intercept_is_taken_over():
    df = exact_panel(T=20, adopt=(("A", 10), ("B", 12)), anticipation=2)
    plain = FDID(cfg(df)).fit()
    trimmed = FDID(cfg(df, anticipation=2)).fit()
    assert trimmed.units[0].pre_periods == plain.units[0].pre_periods - 2
    # The anticipation bump contaminates the untrimmed intercept and not the
    # trimmed one, so trimming restores the true ramp.
    assert np.allclose(trimmed.event_study.att,
                       1.0 + trimmed.event_study.horizons, atol=1e-8)
    assert not np.allclose(plain.event_study.att,
                           1.0 + plain.event_study.horizons, atol=1e-8)


def test_anticipation_cannot_consume_the_whole_pre_window():
    df = exact_panel(T=20, adopt=(("A", 3), ("B", 12)))
    with pytest.raises(MlsynthEstimationError):
        FDID(cfg(df, anticipation=3)).fit()


# ─── condition IV: how donors are selected ────────────────────────────────

def test_pooled_selection_gives_every_cohort_member_one_donor_set():
    df = noisy_panel(adopt=(("A", 28), ("B", 28), ("C", 32), ("D", 32)))
    res = FDID(cfg(df, selection="pooled")).fit()
    by_cohort = {}
    for u in res.units:
        by_cohort.setdefault(u.adoption_time, []).append(tuple(u.selected_indices))
    for sets in by_cohort.values():
        assert len(set(sets)) == 1


def test_partial_pooling_interpolates_between_the_two_criteria():
    df = noisy_panel(adopt=(("A", 28), ("B", 28), ("C", 32), ("D", 32)))
    unit = FDID(cfg(df, selection="unit")).fit()
    pooled = FDID(cfg(df, selection="pooled")).fit()
    at_zero = FDID(cfg(df, selection="partial", pooling_weight=0.0)).fit()
    at_one = FDID(cfg(df, selection="partial", pooling_weight=1.0)).fit()
    for a, b in ((at_zero, unit), (at_one, pooled)):
        assert [u.selected_indices for u in a.units] == \
               [u.selected_indices for u in b.units]


def test_pooling_weight_outside_a_partial_fit_is_a_config_error():
    with pytest.raises(MlsynthConfigError, match="pooling_weight"):
        FDIDConfig(**cfg(exact_panel(), selection="unit", pooling_weight=0.3))


def test_pooling_weight_must_be_a_proportion():
    with pytest.raises(Exception):
        FDIDConfig(**cfg(exact_panel(), selection="partial", pooling_weight=1.7))


# ─── condition VI: the joint covariance ───────────────────────────────────

def test_aggregate_variance_prices_the_cross_unit_covariance():
    """Two units with perfectly correlated residuals and equal weights carry
    twice the standard error the independence formula reports."""
    from mlsynth.utils.fdid_helpers.staggered import aggregate_variance

    rng = np.random.default_rng(1)
    v = rng.standard_normal(200)
    residuals = np.column_stack([v, v])          # identical, so corr = 1
    C = np.zeros((2, 200))
    C[0, 100:] = 0.5 / 100
    C[1, 100:] = 0.5 / 100
    joint = aggregate_variance(residuals, C)
    indep = sum(np.var(residuals[:, i]) * (C[i] @ C[i]) for i in range(2))
    assert joint == pytest.approx(2.0 * indep, rel=1e-6)


def test_aggregate_variance_matches_independence_when_units_are_independent():
    from mlsynth.utils.fdid_helpers.staggered import aggregate_variance

    rng = np.random.default_rng(2)
    residuals = rng.standard_normal((4000, 2))
    C = np.zeros((2, 4000))
    C[0, 2000:] = 0.5 / 2000
    C[1, 2000:] = 0.5 / 2000
    joint = aggregate_variance(residuals, C)
    indep = sum(np.var(residuals[:, i]) * (C[i] @ C[i]) for i in range(2))
    assert joint == pytest.approx(indep, rel=0.05)


def test_lagged_variance_is_floored_at_the_contemporaneous_one():
    """Autocovariances of a demeaned stretch sum to about -gamma_0 over a
    block, so an unguarded lag sum can fall below the contemporaneous value."""
    from mlsynth.utils.fdid_helpers.staggered import aggregate_variance

    rng = np.random.default_rng(3)
    residuals = rng.standard_normal((80, 2))
    C = np.zeros((2, 80))
    C[:, 40:] = 0.5 / 40
    assert (aggregate_variance(residuals, C, lag=6)
            >= aggregate_variance(residuals, C) - 1e-12)


def test_reported_standard_errors_are_positive_and_finite():
    res = FDID(cfg(noisy_panel())).fit()
    assert res.overall_se > 0 and np.isfinite(res.overall_se)
    assert np.all(res.event_study.se > 0)
    assert np.all(np.isfinite(res.event_study.se))
    assert np.all(res.event_study.ci_lower < res.event_study.ci_upper)


# ─── contract ─────────────────────────────────────────────────────────────

def test_standard_submodels_are_populated_over_event_time():
    res = FDID(cfg(noisy_panel())).fit()
    assert res.effects is not None and res.time_series is not None
    assert res.inference is not None and res.method_details is not None
    assert res.effects.att == pytest.approx(res.overall_att)
    n = res.event_study.horizons.size
    assert np.asarray(res.time_series.time_periods).size == n


def test_per_unit_counterfactual_reproduces_the_gap():
    res = FDID(cfg(noisy_panel())).fit()
    for u in res.units:
        assert np.allclose(u.gap, u.observed - u.counterfactual, atol=1e-10)


def test_donor_weights_are_equal_and_sum_to_one():
    res = FDID(cfg(noisy_panel())).fit()
    for u in res.units:
        w = list(u.donor_weights.values())
        assert w and np.isclose(sum(w), 1.0)
        assert len(set(np.round(w, 12))) == 1


# ─── failure modes ────────────────────────────────────────────────────────

def test_a_cohort_without_enough_pre_periods_is_refused():
    df = exact_panel(T=20, adopt=(("A", 1), ("B", 12)))
    with pytest.raises(MlsynthEstimationError):
        FDID(cfg(df)).fit()


def test_unknown_selection_mode_is_a_config_error():
    with pytest.raises(Exception):
        FDIDConfig(**cfg(exact_panel(), selection="bogus"))


def test_negative_anticipation_is_a_config_error():
    with pytest.raises(Exception):
        FDIDConfig(**cfg(exact_panel(), anticipation=-1))


# ─── helper-level invariants and guards ───────────────────────────────────

def test_forward_selection_path_rejects_a_malformed_call():
    from mlsynth.utils.fdid_helpers import forward_selection_path

    donors = np.random.default_rng(0).standard_normal((10, 3))
    with pytest.raises(ValueError, match="at least one target"):
        forward_selection_path(donors, [])
    with pytest.raises(ValueError, match="does not match the pre-window"):
        forward_selection_path(donors, [np.zeros(9)])
    with pytest.raises(ValueError, match="one entry per target"):
        forward_selection_path(donors, [np.zeros(10)], weights=[0.5, 0.5])


def test_forward_selection_path_tolerates_a_flat_target():
    """A target with no variation has nothing to explain; the search must
    still return a complete ordering instead of dividing by zero."""
    from mlsynth.utils.fdid_helpers import forward_selection_path

    donors = np.random.default_rng(1).standard_normal((12, 4))
    order, path = forward_selection_path(donors, [np.ones(12)])
    assert sorted(order) == [0, 1, 2, 3]
    assert np.all(np.isfinite(path))


def test_cross_covariance_guards_its_inputs():
    from mlsynth.utils.fdid_helpers import residual_cross_covariances

    with pytest.raises(ValueError, match="non-empty"):
        residual_cross_covariances(np.empty((0, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        residual_cross_covariances(np.ones((5, 2)), lag=-1)


def test_aggregate_variance_rejects_a_mismatched_contrast():
    from mlsynth.utils.fdid_helpers import aggregate_variance

    with pytest.raises(ValueError, match="n_units"):
        aggregate_variance(np.ones((10, 2)), np.ones((3, 10)))


def test_event_time_weights_drop_units_the_panel_does_not_reach():
    from mlsynth.utils.fdid_helpers import event_time_weights

    g = np.array([5, 8])
    assert event_time_weights(g, 0, 10).tolist() == [0.5, 0.5]
    assert event_time_weights(g, 3, 10).tolist() == [1.0, 0.0]   # 8 + 3 = 11
    assert event_time_weights(g, 9, 10).tolist() == [0.0, 0.0]


def test_contrast_matrix_skips_units_absent_at_a_horizon():
    """A unit carrying zero weight must contribute no pre-window term either,
    or its intercept would leak sampling error into an estimate it is not in.
    """
    from mlsynth.utils.fdid_helpers.staggered import _contrast_matrix

    g = np.array([5, 8])
    C = _contrast_matrix(g, [5, 8], [0], T=10, horizon=3)
    assert np.allclose(C[1], 0.0)
    assert C[0].sum() == pytest.approx(0.0)      # +1 post, -1 spread over pre


def test_an_explicit_max_horizon_within_reach_is_honoured():
    res = FDID(cfg(exact_panel(T=20, adopt=(("A", 10), ("B", 12))),
                   max_horizon=3)).fit()
    assert res.event_study.horizons.tolist() == [0, 1, 2, 3]


def test_staggered_inputs_need_a_cohort():
    from mlsynth.utils.fdid_helpers import prepare_staggered_inputs

    with pytest.raises(MlsynthEstimationError, match="cohorts"):
        prepare_staggered_inputs({"cohorts": {}})


def test_prepare_panel_wraps_a_broken_panel_as_a_data_error():
    from mlsynth.utils.fdid_helpers import prepare_panel

    with pytest.raises(MlsynthDataError):
        prepare_panel(pd.DataFrame({"unit": [], "t": [], "y": [], "w": []}),
                      "y", "w", "unit", "t")
    ragged = exact_panel().drop(index=0)          # unbalanced: one unit short
    with pytest.raises(MlsynthDataError):
        prepare_panel(ragged, "y", "w", "unit", "t")


def test_lrvar_lag_outside_a_hac_fit_is_a_config_error():
    with pytest.raises(MlsynthConfigError, match="lrvar_lag"):
        FDIDConfig(**cfg(exact_panel(), lrvar_lag=3))


def test_hac_inference_prices_autocovariances_into_the_aggregate():
    plain = FDID(cfg(noisy_panel())).fit()
    hac = FDID(cfg(noisy_panel(), inference="hac", lrvar_lag=3)).fit()
    assert hac.overall_att == pytest.approx(plain.overall_att)
    assert hac.overall_se >= plain.overall_se - 1e-12
    assert hac.metadata["lrvar_lag"] == 3


def test_convenience_accessors_agree_with_the_typed_fields():
    res = FDID(cfg(noisy_panel())).fit()
    assert res.att_se == res.overall_se
    assert res.att_by_unit() == {u.unit_name: u.att for u in res.units}
    assert res.donors_by_unit() == {u.unit_name: list(u.selected_names)
                                    for u in res.units}


def test_metadata_records_the_configuration_actually_used():
    res = FDID(cfg(noisy_panel(adopt=(("A", 28), ("B", 28), ("C", 32), ("D", 32))),
                   selection="partial", pooling_weight=0.25)).fit()
    assert res.metadata["pooling_weight"] == 0.25
    assert res.metadata["n_cohorts"] == 2
    assert res.metadata["anticipation"] == 0
    assert res.selection == "partial"


def test_a_result_built_with_submodels_already_set_keeps_them():
    """The validator fills the standardized surface once; a caller that has
    supplied it must not have it overwritten."""
    from mlsynth.config_models import EffectsResults

    res = FDID(cfg(noisy_panel())).fit()
    rebuilt = res.__class__(
        inputs=res.inputs, units=res.units, event_study=res.event_study,
        overall_att=res.overall_att, overall_se=res.overall_se,
        overall_ci=res.overall_ci, selection=res.selection,
        effects=EffectsResults(att=999.0),
    )
    assert rebuilt.effects.att == 999.0


def test_hac_without_an_explicit_lag_resolves_a_non_zero_default():
    """A 'hac' fit that priced no autocovariance would be a silent no-op."""
    res = FDID(cfg(noisy_panel(), inference="hac")).fit()
    assert res.metadata["inference"] == "hac"
    assert res.metadata["lrvar_lag"] > 0


def test_the_staggered_default_prices_autocovariances():
    """The aggregate averages over horizons, so leaving the autocorrelation
    unpriced divides its variance by the horizon count for free."""
    res = FDID(cfg(noisy_panel())).fit()
    assert res.metadata["inference"] == "hac"
    assert res.metadata["lrvar_lag"] > 0


def test_an_explicit_analytic_request_is_honoured():
    res = FDID(cfg(noisy_panel(), inference="analytic")).fit()
    assert res.metadata["inference"] == "analytic"
    assert res.metadata["lrvar_lag"] == 0


def test_the_single_treated_path_keeps_the_analytic_default():
    """Li's Proposition 2.1 is the single-unit contract; only the staggered
    aggregate changes default."""
    res = FDID(cfg(exact_panel(adopt=(("A", 10),)))).fit()
    assert res.fdid.inference_method == "analytic"


# ─── event-study plot ─────────────────────────────────────────────────────

def test_plot_draws_the_event_study_with_its_confidence_band():
    """The generic counterfactual plot would draw the effect path against a
    fabricated zero series and no band; a staggered result owes an event
    study."""
    import matplotlib
    matplotlib.use("Agg")

    res = FDID(cfg(noisy_panel())).fit()
    ax = res.plot()
    line = next(ln for ln in ax.get_lines() if ln.get_label() == "Effect")
    assert np.allclose(line.get_xdata(), res.event_study.horizons)
    assert np.allclose(line.get_ydata(), res.event_study.att)
    # the shaded band is a filled collection, not a line
    assert ax.collections, "no confidence band was drawn"
    assert ax.get_xlabel() == "Event time"


def test_plot_helper_returns_its_axes_and_shows_nothing():
    """Computation and presentation stay separate: the helper hands back the
    axes and leaves displaying to the caller."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mlsynth.utils.fdid_helpers import plot_fdid_staggered

    res = FDID(cfg(noisy_panel())).fit()
    calls = []
    original, plt.show = plt.show, lambda *a, **k: calls.append(1)
    try:
        ax = plot_fdid_staggered(res)
    finally:
        plt.show = original
    assert ax is not None
    assert not calls


def test_plot_accepts_an_existing_axis_and_cosmetic_overrides():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = FDID(cfg(noisy_panel())).fit()
    _, ax = plt.subplots()
    returned = res.plot(ax=ax, title="custom")
    assert returned is ax
    assert ax.get_title() == "custom"


def test_plot_saves_when_the_config_asks_for_it(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    res = FDID(cfg(noisy_panel())).fit()
    target = tmp_path / "es.png"
    res.plot(save=str(target))
    assert target.exists() and target.stat().st_size > 0

    res.plot(save=True)                      # bare True picks the default name
    assert (Path.cwd() / "fdid_event_study.png").exists()
    (Path.cwd() / "fdid_event_study.png").unlink()


def test_plot_displays_only_when_the_config_asks_for_it():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = FDID(cfg(noisy_panel())).fit()
    calls = []
    original, plt.show = plt.show, lambda *a, **k: calls.append(1)
    try:
        res.plot()
        assert not calls
        res.plot(display=True)
        assert len(calls) == 1
    finally:
        plt.show = original


def test_lagged_variance_matches_the_univariate_block_mean_variance():
    """One unit whose contrast is a block mean reduces to the case
    block_mean_variance already handles, which pins the lagged terms against
    validated code. A formulation that failed to align the two time indices
    would agree at lag 0 and diverge at every lag above it."""
    from mlsynth.utils.fdid_helpers import (
        aggregate_variance, block_mean_variance, residual_autocovariances,
    )

    rng = np.random.default_rng(7)
    n, rho = 400, 0.8
    v = np.empty(n)
    v[0] = rng.standard_normal()
    for t in range(1, n):
        v[t] = rho * v[t - 1] + rng.standard_normal()
    block = 40
    C = np.zeros((1, n))
    C[0, :block] = 1.0 / block

    for lag in (0, 1, 3, 8):
        gamma = residual_autocovariances(v, lag)
        assert aggregate_variance(v[:, None], C, lag) == pytest.approx(
            block_mean_variance(gamma, block), rel=1e-10)


def test_positive_autocorrelation_widens_the_aggregate():
    """A persistent residual must cost precision; a lag term that cancelled
    would leave the variance at its lag-0 value."""
    from mlsynth.utils.fdid_helpers import aggregate_variance

    rng = np.random.default_rng(8)
    n = 600
    v = np.empty(n)
    v[0] = rng.standard_normal()
    for t in range(1, n):
        v[t] = 0.8 * v[t - 1] + rng.standard_normal()
    C = np.zeros((1, n))
    C[0, :50] = 1.0 / 50
    assert aggregate_variance(v[:, None], C, 6) > \
        2.0 * aggregate_variance(v[:, None], C, 0)
