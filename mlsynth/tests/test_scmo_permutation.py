"""Permutation (placebo) inference for SCMO, and the demeaned appendix DGP.

The test in Tian, Lee & Panchenko (2026) Online Appendix B.3.3 is Abadie's:
refit the synthetic control with every unit in turn playing the treated one,
rank the post-to-pre-treatment RMSPE ratios, and read the treated unit's rank as
a p-value. The appendix's simulation (Table B.1) reports the rejection rate of
that test, so the procedure has to exist before the table can be reproduced.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.scmo import SCMO
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.scmo_helpers import (
    AVERAGED,
    CONCATENATED,
    SEPARATE,
    PlaceboInference,
    permutation_inference,
    prepare_scmo_inputs,
)
from mlsynth.utils.scmo_helpers.inference import rmspe_ratio
from mlsynth.utils.scmo_helpers.simulation import simulate_tian_demeaned, to_panel

N_UNITS = 6
T_PRE, T_POST = 6, 2
PRE_YEARS = list(range(1960, 1960 + T_PRE))
SPEC = {"year": PRE_YEARS, "vars": {"a": "y1", "b": "y2"}}


def _panel(effect: float = 0.0, effect_last_only: bool = False) -> pd.DataFrame:
    """Six units on two common factors; ``effect`` hits the treated unit's
    primary outcome in the post-period."""
    rng = np.random.default_rng(4)
    T = T_PRE + T_POST
    factors = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    rows = []
    for i in range(N_UNITS):
        load = rng.uniform(0.5, 1.5, size=2)
        for t in range(T):
            year = 1960 + t
            base = 20.0 + factors[t] @ load
            treated = int(i == 0 and t >= T_PRE)
            y1 = base + rng.normal(scale=0.2)
            if treated and (not effect_last_only or t == T - 1):
                y1 += effect
            rows.append({"unit": f"u{i}", "time": year, "treat": treated,
                         "y1": y1, "y2": base * 1.1 + rng.normal(scale=0.2)})
    return pd.DataFrame(rows)


def _inputs(df: pd.DataFrame):
    return prepare_scmo_inputs(df, unitid="unit", time="time", outcome="y1",
                               spec=SPEC, treated_unit="u0",
                               intervention_time=1960 + T_PRE)


# --- the ratio statistic ---------------------------------------------------

def test_rmspe_ratio_is_post_over_pre():
    assert rmspe_ratio(2.0, 1.0, eta=0.0) == pytest.approx(2.0)


def test_rmspe_ratio_eta_enters_both_sides():
    assert rmspe_ratio(2.0, 1.0, eta=1.0) == pytest.approx(1.5)


def test_rmspe_ratio_large_eta_pulls_every_unit_to_one():
    assert rmspe_ratio(9.0, 1.0, eta=1e6) == pytest.approx(1.0, abs=1e-4)


def test_rmspe_ratio_of_a_perfect_pre_fit_is_infinite():
    """A zero pre-treatment RMSPE is the degenerate case the appendix adds eta
    for; without eta the ratio is infinite, not a division error."""
    assert rmspe_ratio(1.0, 0.0, eta=0.0) == np.inf


def test_rmspe_ratio_zero_over_zero_is_zero():
    """No post-treatment gap ranks last however good the pre-fit was."""
    assert rmspe_ratio(0.0, 0.0, eta=0.0) == 0.0


def test_rmspe_ratio_rejects_negative_eta():
    with pytest.raises(MlsynthConfigError, match="eta"):
        rmspe_ratio(1.0, 1.0, eta=-0.5)


# --- the permutation test --------------------------------------------------

def test_permutation_smoke():
    res = permutation_inference(_inputs(_panel()), CONCATENATED)
    assert isinstance(res, PlaceboInference)
    assert 0.0 < res.p_value <= 1.0
    assert res.ratios.shape == (N_UNITS,)
    assert res.pre_rmspe.shape == (N_UNITS,) and res.post_rmspe.shape == (N_UNITS,)
    assert res.per_period_p.shape == (T_POST,)
    assert res.per_period_ratios.shape == (N_UNITS, T_POST)
    assert np.all(np.isfinite(res.ratios))


def test_p_value_is_a_rank_among_the_units():
    res = permutation_inference(_inputs(_panel()), CONCATENATED)
    assert res.p_value == pytest.approx(np.mean(res.ratios >= res.treated_ratio))
    assert (res.p_value * N_UNITS) == pytest.approx(round(res.p_value * N_UNITS))


@pytest.mark.parametrize("scheme", [CONCATENATED, AVERAGED, SEPARATE])
def test_every_scheme_supports_the_test(scheme):
    res = permutation_inference(_inputs(_panel()), scheme)
    assert 0.0 < res.p_value <= 1.0


def test_a_large_effect_ranks_the_treated_unit_first():
    res = permutation_inference(_inputs(_panel(effect=60.0)), CONCATENATED)
    assert res.p_value == pytest.approx(1.0 / N_UNITS)
    assert res.treated_ratio == res.ratios.max()


def test_no_effect_does_not_rank_the_treated_unit_first():
    res = permutation_inference(_inputs(_panel(effect=0.0)), CONCATENATED)
    assert res.p_value > 1.0 / N_UNITS


def test_large_eta_pulls_every_ratio_to_one():
    """eta compresses the ratios toward 1, which is how the appendix keeps a
    unit with a near-zero pre-treatment RMSPE from dominating the ranking. It
    compresses the statistic; it does not erase the ordering."""
    res = permutation_inference(_inputs(_panel(effect=60.0)), CONCATENATED, eta=1e9)
    np.testing.assert_allclose(res.ratios, 1.0, atol=1e-6)
    assert 0.0 < res.p_value <= 1.0


def test_one_sided_test_ignores_the_wrong_direction():
    """The appendix tests a one-sided alternative. An effect of the opposite
    sign leaves the treated unit's statistic at zero, so it ranks last."""
    up = _inputs(_panel(effect=60.0))
    assert permutation_inference(up, CONCATENATED, alternative="greater").p_value \
        == pytest.approx(1.0 / N_UNITS)
    assert permutation_inference(up, CONCATENATED, alternative="less").p_value \
        == pytest.approx(1.0)


def test_per_period_p_values_localize_the_effect():
    """An effect only in the final period is significant there, not before."""
    res = permutation_inference(
        _inputs(_panel(effect=60.0, effect_last_only=True)), CONCATENATED)
    assert res.per_period_p[-1] == pytest.approx(1.0 / N_UNITS)
    assert res.per_period_p[0] > res.per_period_p[-1]


def test_demeaned_permutation_runs():
    df = _panel()
    inp = prepare_scmo_inputs(df, unitid="unit", time="time", outcome="y1",
                              spec=SPEC, treated_unit="u0",
                              intervention_time=1960 + T_PRE, demean=True)
    res = permutation_inference(inp, CONCATENATED, demean=True)
    assert 0.0 < res.p_value <= 1.0


def test_two_units_give_the_coarsest_possible_p_value():
    df = _panel()
    df = df[df["unit"].isin(["u0", "u1"])]
    inp = prepare_scmo_inputs(df, unitid="unit", time="time", outcome="y1",
                              spec=SPEC, treated_unit="u0",
                              intervention_time=1960 + T_PRE)
    assert permutation_inference(inp, CONCATENATED).p_value in (0.5, 1.0)


def test_unknown_alternative_raises():
    with pytest.raises(MlsynthConfigError, match="alternative"):
        permutation_inference(_inputs(_panel()), CONCATENATED, alternative="sideways")


# --- reaching it from the estimator ----------------------------------------

def _fit(df: pd.DataFrame, **extra):
    return SCMO({"df": df, "outcome": "y1", "treat": "treat", "unitid": "unit",
                 "time": "time", "spec": SPEC, "schemes": [CONCATENATED],
                 "display_graphs": False, **extra}).fit()

def test_inference_defaults_to_conformal():
    res = _fit(_panel())
    assert res.inference.method == "conformal (CWZ)"
    assert res.inference.ci_lower is not None


def test_placebo_inference_is_selectable():
    res = _fit(_panel(effect=60.0), inference="placebo")
    assert res.inference.method == "placebo (post/pre RMSPE ratio)"
    assert res.inference.p_value == pytest.approx(1.0 / N_UNITS)
    assert res._primary.placebo is not None
    assert res._primary.placebo.ratios.shape == (N_UNITS,)


def test_placebo_inference_reports_no_interval():
    """The permutation test ranks units; it does not invert to an interval, and
    the result says so instead of carrying the conformal one."""
    res = _fit(_panel(), inference="placebo")
    assert res.inference.ci_lower is None and res.inference.ci_upper is None


def test_placebo_options_reach_the_helper():
    flat = _fit(_panel(effect=60.0), inference="placebo", placebo_eta=1e9)
    assert flat._primary.placebo.eta == 1e9
    np.testing.assert_allclose(flat._primary.placebo.ratios, 1.0, atol=1e-6)
    one_sided = _fit(_panel(effect=60.0), inference="placebo",
                     placebo_alternative="less")
    assert one_sided._primary.placebo.alternative == "less"
    assert one_sided.inference.p_value == pytest.approx(1.0)


def test_unknown_inference_choice_is_rejected():
    with pytest.raises(Exception) as excinfo:
        _fit(_panel(), inference="bootstrap")
    assert "inference" in str(excinfo.value)


def test_negative_placebo_eta_is_rejected():
    with pytest.raises(Exception) as excinfo:
        _fit(_panel(), inference="placebo", placebo_eta=-1.0)
    assert "placebo_eta" in str(excinfo.value)


# --- the demeaned appendix DGP (Table B.1) ---------------------------------

def test_simulate_tian_demeaned_shapes():
    rng = np.random.default_rng(0)
    outcomes, N, TT, treated, Z = simulate_tian_demeaned(rng, T0=5, K=3, d=1.0)
    assert N == 30 and TT == 6 and treated == 0
    assert len(outcomes) == 3 and all(Y.shape == (30, 6) for Y in outcomes)
    assert Z.shape == (30, 2)


def test_simulate_tian_demeaned_is_reproducible():
    a = simulate_tian_demeaned(np.random.default_rng(7), T0=4, K=2)[0]
    b = simulate_tian_demeaned(np.random.default_rng(7), T0=4, K=2)[0]
    for Ya, Yb in zip(a, b):
        np.testing.assert_allclose(Ya, Yb)


def test_d_bounds_the_treated_unit_predictors():
    """``d`` shrinks the support of the treated unit's predictors: at d = 0 it
    sits at the center of the donors' support, the well-inside-the-hull case."""
    _o, _N, _TT, _tr, Z = simulate_tian_demeaned(np.random.default_rng(1), T0=5, K=2, d=0.0)
    np.testing.assert_allclose(Z[0], 0.0, atol=1e-12)
    assert np.abs(Z[1:]).max() > 0.1
    _o, _N, _TT, _tr, Z1 = simulate_tian_demeaned(np.random.default_rng(1), T0=5, K=2, d=1.0)
    assert np.abs(Z1[0]).max() > 0.0


def test_tau_lands_on_the_treated_post_period():
    common = dict(T0=4, K=2, d=1.0)
    null = simulate_tian_demeaned(np.random.default_rng(3), **common)[0]
    effect = simulate_tian_demeaned(np.random.default_rng(3), tau=5.0, **common)[0]
    assert effect[0][0, -1] - null[0][0, -1] == pytest.approx(5.0)
    np.testing.assert_allclose(effect[0][1:], null[0][1:])
    np.testing.assert_allclose(effect[0][0, :-1], null[0][0, :-1])


def test_outcomes_carry_stable_level_differences():
    """The appendix DGP gives each outcome a large mean of its own, which is
    what makes demeaning bite."""
    outcomes, _N, _TT, _tr, _Z = simulate_tian_demeaned(
        np.random.default_rng(5), T0=10, K=5, d=1.0)
    assert float(np.std([Y.mean() for Y in outcomes])) > 1.0


def test_to_panel_carries_the_observed_predictors():
    outcomes, N, TT, treated, Z = simulate_tian_demeaned(
        np.random.default_rng(2), T0=3, K=2, d=1.0)
    df = to_panel(outcomes, N, TT, treated, predictors=Z)
    assert {"z0", "z1"} <= set(df.columns)
    per_unit = df.groupby("unit")["z0"].nunique()
    assert (per_unit == 1).all()
    assert len(df) == N * TT
