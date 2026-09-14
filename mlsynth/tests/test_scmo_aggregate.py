"""The aggregate effect across a domain's outcomes, and its permutation p-value.

Tian, Lee & Panchenko (2026) summarize a domain of related outcomes with the
index of Kling, Liebman & Katz (2007): standardize each outcome's estimated
effect by that outcome's cross-sectional SD, average over the periods in a
window, and average across the outcomes (Online Appendix B.3.2),

    tau_i(t1, t2) = (1/K) sum_k  mean_{t in [t1, t2]} tau_it,k / sigma_k.

Its p-value is the same permutation test one level up (B.3.3): each unit's
aggregate ratio is its post-treatment loss summed over outcomes over its
pre-treatment loss summed over outcomes, and the treated unit's rank in that
ranking is the p-value,

    r_i = sum_k R^post_i,k / sigma_k  /  sum_k R^pre_i,k / sigma_k.

Levels: smoke, unit invariants, edge, failure.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.scmo import SCMO
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.scmo_helpers import CONCATENATED, prepare_scmo_inputs
from mlsynth.utils.scmo_helpers.aggregate import (
    DomainAggregate,
    OutcomeGaps,
    aggregate_domain,
    outcome_gaps,
)
from mlsynth.utils.scmo_helpers.inference import permutation_inference

N_UNITS, T_PRE, T_POST = 6, 5, 4
TREATED = 0


def _outcome(name, *, treated_effect=0.0, sigma=1.0, seed=0, periods=None,
             n_units=N_UNITS, t_post=T_POST) -> OutcomeGaps:
    """A synthetic outcome: small placebo gaps, plus an effect on the treated."""
    rng = np.random.default_rng(seed)
    gaps = rng.normal(scale=0.1, size=(n_units, t_post))
    gaps[TREATED] += treated_effect
    pre = np.abs(rng.normal(scale=0.1, size=n_units)) + 0.1
    if periods is None:
        periods = np.arange(t_post)
    return OutcomeGaps(name=name, periods=np.asarray(periods), gaps=gaps,
                       pre_rmspe=pre, sigma=sigma)


# --------------------------------------------------------------------------- smoke

def test_aggregate_smoke():
    agg = aggregate_domain([_outcome("a", seed=1), _outcome("b", seed=2)],
                           treated_idx=TREATED)
    assert isinstance(agg, DomainAggregate)
    assert np.isfinite(agg.tau)
    assert 0.0 < agg.p_value <= 1.0
    assert agg.ratios.shape == (N_UNITS,)
    assert agg.outcomes == ("a", "b")


def test_windows_report_one_value_each():
    windows = [(0, 1), (2, 3)]
    agg = aggregate_domain([_outcome("a", seed=1), _outcome("b", seed=2)],
                           treated_idx=TREATED, windows=windows)
    assert agg.tau_by_window.shape == (2,)
    assert agg.p_by_window.shape == (2,)
    assert np.all(np.isfinite(agg.tau_by_window))
    assert np.all((agg.p_by_window > 0) & (agg.p_by_window <= 1))


# ------------------------------------------------------------------ unit invariants

def test_one_outcome_is_its_own_standardized_mean_gap():
    o = _outcome("a", treated_effect=2.0, sigma=4.0, seed=3)
    agg = aggregate_domain([o], treated_idx=TREATED)
    assert agg.tau == pytest.approx(o.gaps[TREATED].mean() / o.sigma)


def test_the_index_averages_over_outcomes():
    a, b = _outcome("a", treated_effect=1.0, seed=4), _outcome("b", treated_effect=3.0, seed=5)
    agg = aggregate_domain([a, b], treated_idx=TREATED)
    each = [o.gaps[TREATED].mean() / o.sigma for o in (a, b)]
    assert agg.tau == pytest.approx(float(np.mean(each)))


def test_the_index_is_free_of_the_outcome_scale():
    """Standardizing by sigma_k is what lets outcomes in different units be
    averaged: rescaling an outcome leaves the index where it was."""
    base = _outcome("a", treated_effect=1.0, sigma=2.0, seed=6)
    scaled = OutcomeGaps(name="a", periods=base.periods, gaps=base.gaps * 50.0,
                         pre_rmspe=base.pre_rmspe * 50.0, sigma=base.sigma * 50.0)
    other = _outcome("b", treated_effect=0.5, seed=7)
    assert aggregate_domain([base, other], treated_idx=TREATED).tau == pytest.approx(
        aggregate_domain([scaled, other], treated_idx=TREATED).tau)


def test_one_window_over_everything_is_the_overall_index():
    outcomes = [_outcome("a", treated_effect=1.0, seed=8), _outcome("b", seed=9)]
    agg = aggregate_domain(outcomes, treated_idx=TREATED,
                           windows=[(0, T_POST - 1)])
    assert agg.tau_by_window[0] == pytest.approx(agg.tau)


def test_a_window_reads_only_its_own_periods():
    o = _outcome("a", seed=10)
    o.gaps[TREATED] = np.array([10.0, 10.0, 0.0, 0.0])
    agg = aggregate_domain([o], treated_idx=TREATED, windows=[(0, 1), (2, 3)])
    assert agg.tau_by_window[0] == pytest.approx(10.0 / o.sigma)
    assert agg.tau_by_window[1] == pytest.approx(0.0)


def test_windows_are_inclusive_at_both_ends():
    """The appendix's windows share their endpoints, as their own code's
    ``dates1[t-1]:dates1[t]`` does."""
    o = _outcome("a", seed=11)
    o.gaps[TREATED] = np.array([0.0, 6.0, 0.0, 0.0])
    agg = aggregate_domain([o], treated_idx=TREATED, windows=[(0, 1), (1, 2)])
    assert agg.tau_by_window[0] == pytest.approx(3.0)   # periods 0 and 1
    assert agg.tau_by_window[1] == pytest.approx(3.0)   # periods 1 and 2


def test_outcomes_may_be_observed_on_different_grids():
    """A domain mixes frequencies: each outcome contributes the periods it has
    inside the window, and an outcome with none contributes nothing."""
    daily = _outcome("daily", seed=12, periods=[0, 1, 2, 3])
    daily.gaps[TREATED] = np.array([4.0, 4.0, 4.0, 4.0])
    quarterly = _outcome("quarterly", seed=13, periods=[0, 10, 20, 30], t_post=4)
    quarterly.gaps[TREATED] = np.array([8.0, 0.0, 0.0, 0.0])
    agg = aggregate_domain([daily, quarterly], treated_idx=TREATED,
                           windows=[(0, 3), (10, 30)])
    assert agg.tau_by_window[0] == pytest.approx((4.0 + 8.0) / 2)
    assert agg.tau_by_window[1] == pytest.approx(0.0)    # only the quarterly is in it


def test_the_p_value_is_a_rank_among_the_units():
    agg = aggregate_domain([_outcome("a", seed=14), _outcome("b", seed=15)],
                           treated_idx=TREATED)
    assert agg.p_value == pytest.approx(float(np.mean(agg.ratios >= agg.treated_ratio)))
    assert (agg.p_value * N_UNITS) == pytest.approx(round(agg.p_value * N_UNITS))


def test_an_effect_in_every_outcome_ranks_the_treated_unit_first():
    outcomes = [_outcome("a", treated_effect=20.0, seed=16),
                _outcome("b", treated_effect=20.0, seed=17)]
    agg = aggregate_domain(outcomes, treated_idx=TREATED, alternative="greater")
    assert agg.p_value == pytest.approx(1.0 / N_UNITS)


def test_a_domain_with_no_effect_does_not_rank_first():
    agg = aggregate_domain([_outcome("a", seed=18), _outcome("b", seed=19)],
                           treated_idx=TREATED)
    assert agg.p_value > 1.0 / N_UNITS


def test_the_one_sided_test_ignores_the_wrong_direction():
    outcomes = [_outcome("a", treated_effect=20.0, seed=20),
                _outcome("b", treated_effect=20.0, seed=21)]
    assert aggregate_domain(outcomes, treated_idx=TREATED,
                            alternative="less").p_value == pytest.approx(1.0)


def test_eta_compresses_the_ratios():
    outcomes = [_outcome("a", treated_effect=20.0, seed=22), _outcome("b", seed=23)]
    agg = aggregate_domain(outcomes, treated_idx=TREATED, eta=1e9)
    np.testing.assert_allclose(agg.ratios, 1.0, atol=1e-6)


# ----------------------------------------------------------------------- edge

def test_a_single_outcome_agrees_with_its_own_permutation_test():
    """With one outcome the aggregate ratio is that outcome's ratio, so the two
    code paths must return the same p-value. The guards are stated in different
    units -- the permutation test takes eta in the outcome's own units, the
    aggregate in standardized ones -- so the equivalent of eta is eta * sigma.
    """
    panel = _real_panel(effect=40.0)
    inputs = prepare_scmo_inputs(panel, unitid="unit", time="time", outcome="y1",
                                 spec=_SPEC, treated_unit="u0", intervention_time=T_PRE)
    sigma = float(np.mean(inputs.Y[:, T_PRE:].std(axis=0, ddof=1)))
    placebo = permutation_inference(inputs, CONCATENATED, alternative="greater",
                                    eta=0.05 * sigma)
    agg = aggregate_domain([outcome_gaps("y1", inputs, placebo)],
                           treated_idx=inputs.treated_idx, alternative="greater",
                           eta=0.05)
    assert agg.p_value == pytest.approx(placebo.p_value)
    np.testing.assert_allclose(agg.ratios, placebo.ratios, rtol=1e-9)


def test_two_units_give_the_coarsest_possible_p_value():
    outcomes = [_outcome("a", seed=24, n_units=2), _outcome("b", seed=25, n_units=2)]
    assert aggregate_domain(outcomes, treated_idx=TREATED).p_value in (0.5, 1.0)


def test_an_outcome_whose_gaps_are_all_zero_contributes_nothing():
    a = _outcome("a", treated_effect=6.0, seed=26)
    b = _outcome("b", seed=27)
    b.gaps[:] = 0.0
    agg = aggregate_domain([a, b], treated_idx=TREATED)
    assert agg.tau == pytest.approx(a.gaps[TREATED].mean() / a.sigma / 2)


# ----------------------------------------------------------------------- failure

def test_an_empty_domain_raises():
    with pytest.raises(MlsynthConfigError, match="at least one outcome"):
        aggregate_domain([], treated_idx=TREATED)


def test_outcomes_covering_different_unit_counts_raise():
    with pytest.raises(MlsynthConfigError, match="same units"):
        aggregate_domain([_outcome("a", seed=28), _outcome("b", seed=29, n_units=5)],
                         treated_idx=TREATED)


def test_a_non_positive_sigma_raises():
    with pytest.raises(MlsynthConfigError, match="sigma"):
        aggregate_domain([_outcome("a", sigma=0.0, seed=30)], treated_idx=TREATED)


def test_a_window_with_no_periods_raises():
    with pytest.raises(MlsynthConfigError, match="no period"):
        aggregate_domain([_outcome("a", seed=31)], treated_idx=TREATED,
                         windows=[(100, 200)])


def test_an_unknown_alternative_raises():
    with pytest.raises(MlsynthConfigError, match="alternative"):
        aggregate_domain([_outcome("a", seed=32)], treated_idx=TREATED,
                         alternative="sideways")


def test_a_negative_eta_raises():
    with pytest.raises(MlsynthConfigError, match="eta"):
        aggregate_domain([_outcome("a", seed=33)], treated_idx=TREATED, eta=-1.0)


# --------------------------------------------- building an outcome from a fit

_SPEC = {"year": list(range(T_PRE)), "vars": {"a": "y1", "b": "y2"}}


def _real_panel(effect: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    T = T_PRE + T_POST
    factors = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    rows = []
    for i in range(N_UNITS):
        load = rng.uniform(0.5, 1.5, size=2)
        for t in range(T):
            base = 20.0 + factors[t] @ load
            treated = int(i == 0 and t >= T_PRE)
            rows.append({"unit": f"u{i}", "time": t, "treat": treated,
                         "y1": base + rng.normal(scale=0.2) + (effect if treated else 0.0),
                         "y2": 1.1 * base + rng.normal(scale=0.2)})
    return pd.DataFrame(rows)


def test_outcome_gaps_reads_the_fit():
    panel = _real_panel(effect=10.0)
    inputs = prepare_scmo_inputs(panel, unitid="unit", time="time", outcome="y1",
                                 spec=_SPEC, treated_unit="u0", intervention_time=T_PRE)
    placebo = permutation_inference(inputs, CONCATENATED)
    og = outcome_gaps("y1", inputs, placebo)
    assert og.name == "y1"
    assert og.gaps.shape == (N_UNITS, T_POST)
    assert og.pre_rmspe.shape == (N_UNITS,)
    np.testing.assert_allclose(og.periods, inputs.time_index.labels[T_PRE:])
    # sigma is the average cross-sectional SD of the outcome over the post periods
    expected = float(np.mean(inputs.Y[:, T_PRE:].std(axis=0, ddof=1)))
    assert og.sigma == pytest.approx(expected)
    # the treated row is that unit's gap series
    np.testing.assert_allclose(og.gaps[inputs.treated_idx], placebo.post_gaps[inputs.treated_idx])


def test_a_domain_reads_end_to_end_from_two_fits():
    """Two outcomes on one panel, aggregated: the path the COVID application
    takes, where a domain's outcomes share a matching matrix."""
    panel = _real_panel(effect=10.0)
    fits = []
    for outcome in ("y1", "y2"):
        res = SCMO({"df": panel, "outcome": outcome, "treat": "treat",
                    "unitid": "unit", "time": "time", "spec": _SPEC,
                    "schemes": [CONCATENATED], "inference": "placebo",
                    "display_graphs": False}).fit()
        fits.append(outcome_gaps(outcome, res.inputs, res._primary.placebo))
    agg = aggregate_domain(fits, treated_idx=0, alternative="greater")
    assert agg.outcomes == ("y1", "y2")
    assert np.isfinite(agg.tau) and 0.0 < agg.p_value <= 1.0


def test_the_guard_enters_both_sides_of_the_ratio():
    """The appendix's footnote puts eta on the numerator and the denominator, so
    it can reorder units whose pre-treatment losses differ: here the treated
    unit has no post-treatment gap but a tight pre-treatment fit, and the guard
    moves it from last to first.

    Tian-Lee-Panchenko's own script applies the guard to both sides for the
    per-outcome and overall tests, but leaves it off the numerator when it
    aggregates inside a window. mlsynth is consistent with the footnote, which
    is why three of that application's 33 window p-values differ -- all three in
    windows where the treated unit's aggregate statistic is about zero.
    """
    o = OutcomeGaps(name="a", periods=np.array([0, 1]),
                    gaps=np.array([[0.0, 0.0], [0.05, 0.05]]),
                    pre_rmspe=np.array([0.1, 1.0]), sigma=1.0)
    window = [(0, 1)]
    unguarded = aggregate_domain([o], treated_idx=0, windows=window,
                                 alternative="greater", eta=0.0)
    guarded = aggregate_domain([o], treated_idx=0, windows=window,
                               alternative="greater", eta=1.0)
    assert unguarded.p_by_window[0] == pytest.approx(1.0)     # 0/0.1 is the smallest
    assert guarded.p_by_window[0] == pytest.approx(0.5)       # 1/1.1 is the largest
