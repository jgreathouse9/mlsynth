"""Triage for a control market that went wrong after a design was locked.

A weighted design fixes its control weights before the experiment runs. When
an outside event hits one of those markets afterwards, the estimation error is
exactly the weight that market carries times the size of the event. That is
arithmetic, so the first three questions an analyst should ask are arithmetic
too: what weight does the market carry, what does the event cost at that
weight, and how large would the event have to be to overturn the conclusion.

These tests cover the four levels the repository asks for -- smoke, the
invariants, the degenerate inputs, and the failures -- for
``mlsynth.utils.contamination``. The generative versions of the invariants are
in ``test_contamination_properties.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.contamination import (
    ContaminationReport,
    contamination_report,
    control_exposure,
)


def _v(*w):
    return np.asarray(w, dtype=float)


# --------------------------------------------------------------------- smoke
def test_report_runs_end_to_end_and_is_the_right_shape():
    rep = contamination_report(_v(0.5, 0.3, 0.2), market=0, shock=-0.4, att=0.1)
    assert isinstance(rep, ContaminationReport)
    for field in ("exposure", "bias", "breakdown_shock", "max_weight",
                  "effective_sample_size", "herfindahl", "n_carrying_weight"):
        assert np.isfinite(getattr(rep, field))


def test_report_without_a_shock_or_an_effect_still_reports_the_weight():
    rep = contamination_report(_v(0.5, 0.3, 0.2), market=1)
    assert rep.exposure == pytest.approx(0.3)
    assert rep.bias is None
    assert rep.breakdown_shock is None


# ----------------------------------------------------------------- invariants
def test_bias_is_minus_weight_times_shock():
    rep = contamination_report(_v(0.6, 0.4), market=1, shock=-2.0)
    assert rep.bias == pytest.approx(-0.4 * -2.0)


def test_breakdown_shock_times_exposure_recovers_the_effect():
    rep = contamination_report(_v(0.85, 0.15), market=1, att=3.0)
    assert rep.breakdown_shock * rep.exposure == pytest.approx(3.0)


def test_uniform_weights_give_an_effective_sample_size_of_n():
    rep = contamination_report(np.full(8, 1 / 8), market=0)
    assert rep.effective_sample_size == pytest.approx(8.0)
    assert rep.n_carrying_weight == 8


def test_herfindahl_is_the_reciprocal_of_effective_sample_size():
    rep = contamination_report(_v(0.5, 0.25, 0.25), market=0)
    assert rep.herfindahl == pytest.approx(1.0 / rep.effective_sample_size)


def test_a_market_carrying_no_weight_costs_nothing_however_large_the_event():
    rep = contamination_report(_v(1.0, 0.0), market=1, shock=-1e6, att=0.2)
    assert rep.exposure == 0.0
    assert rep.bias == 0.0
    assert rep.breakdown_shock == float("inf")
    assert rep.carries_weight is False


def test_carries_weight_is_true_once_the_market_carries_weight():
    assert contamination_report(_v(0.9, 0.1), market=1).carries_weight is True


# ---------------------------------------------------------------- edge cases
def test_a_single_market_carries_the_whole_design():
    rep = contamination_report(_v(1.0), market=0, shock=0.5, att=1.0)
    assert rep.exposure == pytest.approx(1.0)
    assert rep.effective_sample_size == pytest.approx(1.0)
    assert rep.breakdown_shock == pytest.approx(1.0)


def test_a_zero_effect_breaks_down_at_a_shock_of_zero():
    rep = contamination_report(_v(0.5, 0.5), market=0, att=0.0)
    assert rep.breakdown_shock == pytest.approx(0.0)


def test_breakdown_shock_uses_the_size_of_the_effect_not_its_sign():
    pos = contamination_report(_v(0.8, 0.2), market=1, att=2.0)
    neg = contamination_report(_v(0.8, 0.2), market=1, att=-2.0)
    assert pos.breakdown_shock == pytest.approx(neg.breakdown_shock)


def test_weights_within_tolerance_of_one_are_accepted():
    contamination_report(_v(0.5, 0.5 + 5e-10), market=0)


# ------------------------------------------------------------------ failures
@pytest.mark.parametrize(
    "weights, market, fragment",
    [
        (_v(0.5, 0.4), 0, "sum to 1"),
        (_v(0.5, 0.6), 0, "sum to 1"),
        (_v(1.2, -0.2), 0, "negative"),
        (_v(0.5, 0.5), 2, "out of range"),
        (_v(0.5, 0.5), -1, "out of range"),
        (np.array([], dtype=float), 0, "at least one"),
        (_v(np.nan, 1.0), 0, "finite"),
        (_v(np.inf, 1.0), 0, "finite"),
    ],
)
def test_bad_input_raises_a_translated_error_naming_the_problem(weights, market, fragment):
    with pytest.raises(MlsynthDataError, match=fragment):
        contamination_report(weights, market=market)


def test_a_two_dimensional_weight_array_is_refused():
    with pytest.raises(MlsynthDataError, match="one-dimensional"):
        contamination_report(np.eye(2), market=0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_shock_that_is_not_finite_is_refused(bad):
    with pytest.raises(MlsynthDataError, match="finite"):
        contamination_report(_v(0.5, 0.5), market=0, shock=bad)


@pytest.mark.parametrize("bad", [np.nan, -np.inf])
def test_an_effect_that_is_not_finite_is_refused(bad):
    with pytest.raises(MlsynthDataError, match="finite"):
        contamination_report(_v(0.5, 0.5), market=0, att=bad)


def test_the_report_is_frozen():
    rep = contamination_report(_v(0.5, 0.5), market=0)
    with pytest.raises(Exception):
        rep.exposure = 0.9


def test_a_weight_at_the_solver_tolerance_is_treated_as_no_weight():
    """A denormal weight is not a market, and inverting it is not a number.

    The breakdown shock is ``|att| / v_k``, which at ``v_k = 1e-300`` either
    overflows or reports a figure no reader can act on. The report draws the
    same line here that it draws for ``carries_weight``.
    """
    rep = contamination_report(_v(1.0, 1e-300), market=1, att=1.0, shock=5.0)
    assert rep.carries_weight is False
    assert rep.breakdown_shock == float("inf")
    assert rep.bias == pytest.approx(0.0, abs=1e-290)


# --------------------------------------------- reading a real design result
@pytest.fixture(scope="module")
def marex_design():
    """One small MAREX design, reused by the tests that read a result."""
    from mlsynth import MAREX
    from mlsynth.utils.marex_helpers.config import MAREXConfig

    rng = np.random.default_rng(7)
    J, T, T0 = 10, 26, 20
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    load = rng.uniform(0.5, 1.5, size=(2, J))
    Y = f @ load + rng.normal(scale=0.2, size=(T, J)) + 10.0
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), T),
                       "time": np.tile(np.arange(T), J),
                       "y": Y.T.reshape(-1)})
    res = MAREX(MAREXConfig(df=df, outcome="y", unitid="unit", time="time",
                            T0=T0, program_type="MIQP", display_graph=False,
                            inference=False, m_eq=2)).fit()
    return res, Y, T0


def test_control_exposure_reads_the_design_surface(marex_design):
    res, _, _ = marex_design
    weight_map = res.design_weights.summary_stats["control_weights_agg"]
    market = next(iter(weight_map))
    rep = control_exposure(res, market=market)
    assert rep.name == str(market)
    assert rep.exposure == pytest.approx(weight_map[market])
    assert rep.n_carrying_weight == len(weight_map)


def test_control_exposure_picks_up_the_designs_own_effect(marex_design):
    res, _, _ = marex_design
    weight_map = res.design_weights.summary_stats["control_weights_agg"]
    market = max(weight_map, key=weight_map.get)
    rep = control_exposure(res, market=market)
    att = res.report.effects.att
    assert rep.breakdown_shock == pytest.approx(abs(att) / rep.exposure)


def test_a_market_outside_the_control_group_has_no_exposure(marex_design):
    res, _, _ = marex_design
    treated = next(iter(res.design_weights.donor_weights))
    rep = control_exposure(res, market=treated, shock=100.0)
    assert rep.exposure == 0.0
    assert rep.bias == 0.0
    assert rep.carries_weight is False


def test_the_reported_bias_is_the_shift_a_shock_puts_on_the_estimate(marex_design):
    """The identity the module rests on, measured on a real design.

    Shifting one control market's post-period level by ``pi`` moves the
    synthetic control by ``v_k * pi`` and so moves the treated-minus-control
    estimate by ``-v_k * pi``. The report claims that number before anything
    is re-estimated; here it is checked against the re-estimation.
    """
    res, Y, T0 = marex_design
    weight_map = res.design_weights.summary_stats["control_weights_agg"]
    treated_map = res.design_weights.donor_weights
    J = Y.shape[1]
    v = np.zeros(J)
    for unit, x in weight_map.items():
        v[int(unit)] = x
    w = np.zeros(J)
    for unit, x in treated_map.items():
        w[int(unit)] = x

    market = max(weight_map, key=weight_map.get)
    k, pi = int(market), -0.85
    post = slice(T0, Y.shape[0])
    clean = float(np.mean(Y[post] @ w - Y[post] @ v))
    Yc = Y.copy()
    Yc[post, k] += pi
    dirty = float(np.mean(Yc[post] @ w - Yc[post] @ v))

    rep = control_exposure(res, market=market, shock=pi)
    assert dirty - clean == pytest.approx(rep.bias, abs=1e-12)


def test_a_result_without_control_weights_is_refused():
    class NotADesign:
        design_weights = None

    with pytest.raises(MlsynthDataError, match="no control-weight map"):
        control_exposure(NotADesign(), market="anywhere")


def test_a_result_with_no_reported_effect_omits_the_breakdown_point():
    from mlsynth.config_models import WeightsResults

    class BareDesign:
        design_weights = WeightsResults(
            donor_weights={"a": 1.0},
            summary_stats={"control_weights_agg": {"b": 0.6, "c": 0.4}},
        )
        report = None

    rep = control_exposure(BareDesign(), market="b", shock=1.0)
    assert rep.breakdown_shock is None
    assert rep.bias == pytest.approx(-0.6)


def test_the_triage_helpers_are_exported_at_the_top_level():
    import mlsynth

    assert mlsynth.contamination_report is contamination_report
    assert mlsynth.control_exposure is control_exposure
    assert mlsynth.ContaminationReport is ContaminationReport
