"""Inference across a registry of designs (Ferman, Pinto and Possebom 2020).

The functions are array-level: gaps are ``(T, N)`` (periods by units), a stack
of designs is ``(D, T, N)``, and a weight stack is ``(D, N, N)`` whose row ``j``
holds unit ``j``'s donor weights over every unit (zero on itself). They are
checked here against closed forms, against invariants over their input domain,
and against the authors' own R run (``test_fpp_*``).
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.sensitivity_helpers.inference import (
    ConfidenceBand,
    donohue,
    dube_zipperer,
    fp_band,
    identified_effect_range,
    joint_pvalue,
    permutation_pvalue,
    ratio_statistic,
    synthetic_gaps,
)

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)


def masks(T, n_pre):
    pre = np.arange(T) < n_pre
    return pre, ~pre


# --- ratio_statistic -------------------------------------------------------


def test_ratio_statistic_is_post_over_pre_mean_square():
    gaps = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 0.5], [3.0, 0.5]])
    pre, post = masks(4, 2)
    np.testing.assert_allclose(ratio_statistic(gaps, pre, post), [9.0, 0.0625])
    np.testing.assert_allclose(ratio_statistic(gaps, pre, post, root=True), [3.0, 0.25])


def test_ratio_statistic_marks_units_without_a_statistic():
    gaps = np.array([[0.0, 1.0, np.nan], [0.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    pre, post = masks(3, 2)
    out = ratio_statistic(gaps, pre, post)
    assert np.isnan(out[0]) and np.isnan(out[2]) and out[1] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "pre, post",
    [
        (np.array([True, True, False]), np.array([False, False, False])),
        (np.array([False, False, False]), np.array([True, True, True])),
        (np.array([True, True, False]), np.array([False, True, True])),
        (np.array([True, False]), np.array([False, True])),
    ],
)
def test_ratio_statistic_rejects_empty_overlapping_or_misshaped_windows(pre, post):
    with pytest.raises(MlsynthConfigError):
        ratio_statistic(np.ones((3, 2)), pre, post)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.lists(finite, min_size=3, max_size=3), min_size=6, max_size=6),
    st.floats(min_value=0.01, max_value=100),
)
def test_ratio_statistic_is_scale_free(rows, scale):
    gaps = np.array(rows)
    pre, post = masks(6, 3)
    a = ratio_statistic(gaps, pre, post)
    b = ratio_statistic(scale * gaps, pre, post)
    np.testing.assert_allclose(a, b, rtol=1e-9, equal_nan=True)


# --- permutation_pvalue ----------------------------------------------------


def test_pvalue_is_share_at_least_as_large():
    stat = np.array([5.0, 1.0, 7.0, 5.0, 2.0])
    assert permutation_pvalue(stat, 0) == pytest.approx(3 / 5)  # 5, 7, 5
    assert permutation_pvalue(stat, 2) == pytest.approx(1 / 5)
    assert permutation_pvalue(stat, 1) == pytest.approx(1.0)


def test_pvalue_drops_units_without_a_statistic():
    stat = np.array([9.0, np.nan, 1.0, 3.0])
    assert permutation_pvalue(stat, 0) == pytest.approx(1 / 3)


@pytest.mark.parametrize("position", [-1, 4])
def test_pvalue_rejects_positions_outside_the_panel(position):
    with pytest.raises(MlsynthConfigError):
        permutation_pvalue(np.ones(4), position)


def test_pvalue_reports_a_treated_unit_without_a_statistic():
    with pytest.raises(MlsynthDataError):
        permutation_pvalue(np.array([np.nan, 1.0, 2.0]), 0)


@settings(max_examples=80, deadline=None)
@given(st.lists(finite, min_size=2, max_size=25), st.data())
def test_pvalue_invariants(values, data):
    stat = np.array(values)
    n = len(stat)
    p = permutation_pvalue(stat, 0)
    assert 1 / n - 1e-12 <= p <= 1 + 1e-12
    # Relabelling the other units leaves it unchanged.
    order = data.draw(st.permutations(list(range(1, n))))
    assert permutation_pvalue(stat[[0, *order]], 0) == pytest.approx(p)
    # A larger treated statistic never raises it.
    bigger = stat.copy()
    bigger[0] += data.draw(st.floats(min_value=0, max_value=1e3))
    assert permutation_pvalue(bigger, 0) <= p + 1e-12


# --- joint_pvalue ----------------------------------------------------------


def test_joint_pvalue_ranks_the_mean_across_designs():
    stats = np.array([[4.0, 1.0], [1.0, 5.0], [2.0, 2.0]])  # means 2.5, 3, 2
    assert joint_pvalue(stats, 0) == pytest.approx(2 / 3)
    assert joint_pvalue(stats, 1) == pytest.approx(1 / 3)


def test_joint_pvalue_averages_over_the_designs_a_unit_has():
    stats = np.array([[3.0, 3.0], [np.nan, 4.0], [1.0, 1.0]])
    assert joint_pvalue(stats, 0) == pytest.approx(2 / 3)


def test_joint_pvalue_requires_a_unit_by_design_matrix():
    with pytest.raises(MlsynthDataError):
        joint_pvalue(np.ones(4), 0)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.lists(finite, min_size=3, max_size=3), min_size=2, max_size=12), st.data())
def test_joint_pvalue_invariants(rows, data):
    stats = np.array(rows)
    single = joint_pvalue(stats[:, :1], 0)
    assert single == pytest.approx(permutation_pvalue(stats[:, 0], 0))
    order = data.draw(st.permutations([0, 1, 2]))
    assert joint_pvalue(stats[:, order], 0) == pytest.approx(joint_pvalue(stats, 0))


# --- choice criteria -------------------------------------------------------


def test_dube_zipperer_averages_placebo_post_mse_and_ignores_the_treated():
    gaps = np.zeros((2, 3, 3))
    gaps[0, 2, :] = [100.0, 1.0, 3.0]  # placebo post MSE 1 and 9 -> mean 5
    gaps[1, 2, :] = [0.0, 2.0, 2.0]  # -> mean 4
    post = np.array([False, False, True])
    np.testing.assert_allclose(dube_zipperer(gaps, post, 0), [5.0, 4.0])
    gaps[:, :, 0] = 1e6
    np.testing.assert_allclose(dube_zipperer(gaps, post, 0), [5.0, 4.0])


def test_dube_zipperer_skips_placebos_without_a_fit():
    gaps = np.zeros((1, 2, 3))
    gaps[0, 1, :] = [0.0, 2.0, np.nan]
    np.testing.assert_allclose(dube_zipperer(gaps, np.array([False, True]), 0), [4.0])


def test_donohue_is_the_treated_mse_over_the_window():
    gaps = np.zeros((2, 4, 2))
    gaps[0, 1:3, 0] = [1.0, 3.0]
    gaps[1, 1:3, 0] = [2.0, 2.0]
    gaps[:, :, 1] = 1e6
    window = np.array([False, True, True, False])
    np.testing.assert_allclose(donohue(gaps, window, 0), [5.0, 4.0])


def test_criteria_require_a_design_stack():
    with pytest.raises(MlsynthDataError):
        dube_zipperer(np.ones((3, 2)), np.array([False, True, True]), 0)
    with pytest.raises(MlsynthConfigError):
        donohue(np.ones((1, 3, 2)), np.array([False, False, False]), 0)


# --- synthetic_gaps --------------------------------------------------------


def test_synthetic_gaps_are_outcome_minus_weighted_donors():
    Y = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]])
    W = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [np.nan] * 3])
    G = synthetic_gaps(Y, W)
    np.testing.assert_allclose(G[:, 0], [1.0 - 3.0, 2.0 - 4.0])
    np.testing.assert_allclose(G[:, 1], [2.0 - 4.0, 3.0 - 5.0])
    assert np.isnan(G[:, 2]).all()


def test_synthetic_gaps_reject_weight_on_the_unit_itself():
    with pytest.raises(MlsynthDataError):
        synthetic_gaps(np.ones((2, 2)), np.array([[0.5, 0.5], [1.0, 0.0]]))


# --- fp_band ---------------------------------------------------------------


def panel(seed=0, T=12, N=8, n_pre=8, effect=0.0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(0, 1, (T, N)).cumsum(axis=0) + 10
    Y[n_pre:, 0] += effect
    W = np.zeros((2, N, N))
    for d in range(2):
        for j in range(N):
            w = rng.dirichlet(np.ones(N - 1))
            W[d, j, [k for k in range(N) if k != j]] = w
    return Y, W


def test_fp_band_brackets_the_average_effect_and_has_no_pre_period_width():
    Y, W = panel(effect=5.0)
    band = fp_band(Y, W, 0, n_pre=8, significance=2 / 8, precision=12)
    assert isinstance(band, ConfidenceBand)
    gaps = np.stack([synthetic_gaps(Y, W[d])[:, 0] for d in range(2)])
    np.testing.assert_allclose(band.average, gaps.mean(axis=0))
    assert np.isnan(band.lower[:8]).all() and np.isnan(band.upper[:8]).all()
    assert band.shift_lower <= band.shift_upper
    np.testing.assert_allclose(band.lower[8:], band.average[8:] + band.shift_lower)
    np.testing.assert_allclose(band.upper[8:], band.average[8:] + band.shift_upper)


@settings(max_examples=12, deadline=None)
@given(st.integers(0, 10_000), st.floats(min_value=-50, max_value=50))
def test_fp_band_moves_with_the_treated_post_outcomes(seed, delta):
    Y, W = panel(seed)
    a = fp_band(Y, W, 0, n_pre=8, significance=2 / 8, precision=10)
    Y2 = Y.copy()
    Y2[8:, 0] += delta
    b = fp_band(Y2, W, 0, n_pre=8, significance=2 / 8, precision=10)
    assert b.shift_lower == pytest.approx(a.shift_lower)
    assert b.shift_upper == pytest.approx(a.shift_upper)
    np.testing.assert_allclose(b.average[8:], a.average[8:] + delta)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(significance=0.0),
        dict(significance=1.0),
        dict(precision=-1),
        dict(n_pre=0),
        dict(n_pre=12),
    ],
)
def test_fp_band_rejects_invalid_settings(kwargs):
    Y, W = panel()
    args = dict(significance=0.25, precision=5, n_pre=8) | kwargs
    with pytest.raises(MlsynthConfigError):
        fp_band(Y, W, 0, **args)


def test_fp_band_reports_a_misshaped_weight_stack():
    Y, W = panel()
    with pytest.raises(MlsynthDataError):
        fp_band(Y, W[:, :, :-1], 0, n_pre=8, significance=0.25)


# --- identified_effect_range -----------------------------------------------


def test_unconstrained_range_spans_the_donor_vertices():
    y_post = np.array([10.0, 12.0])
    Y0_post = np.array([[1.0, 4.0, 9.0], [3.0, 6.0, 11.0]])
    lo, hi = identified_effect_range(
        np.zeros((1, 3)), np.array([0.2, 0.3, 0.5]), y_post, Y0_post
    )
    effects = [np.mean(y_post - Y0_post[:, k]) for k in range(3)]
    assert lo == pytest.approx(min(effects)) and hi == pytest.approx(max(effects))


def test_full_rank_fit_identifies_the_effect():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 4))
    w = np.array([0.1, 0.2, 0.3, 0.4])
    y_post, Y0_post = rng.normal(size=3), rng.normal(size=(3, 4))
    lo, hi = identified_effect_range(A, w, y_post, Y0_post)
    att = np.mean(y_post - Y0_post @ w)
    assert lo == pytest.approx(att, abs=1e-6) and hi == pytest.approx(att, abs=1e-6)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 5))
def test_identified_range_contains_the_solved_effect(seed, K):
    rng = np.random.default_rng(seed)
    J = 6
    A = rng.normal(size=(K, J))
    w = rng.dirichlet(np.ones(J))
    y_post, Y0_post = rng.normal(size=4), rng.normal(size=(4, J))
    lo, hi = identified_effect_range(A, w, y_post, Y0_post)
    att = np.mean(y_post - Y0_post @ w)
    assert lo <= att + 1e-7 and att <= hi + 1e-7


def test_identified_range_reports_inconsistent_shapes():
    with pytest.raises(MlsynthDataError):
        identified_effect_range(np.ones((2, 3)), np.ones(4) / 4, np.ones(2), np.ones((2, 3)))
    with pytest.raises(MlsynthDataError):
        identified_effect_range(np.ones((2, 3)), np.ones(3) / 3, np.ones(2), np.ones((3, 3)))


# --- Ferman, Pinto and Possebom (2020), West Germany ------------------------
#
# The authors' replication_germany.R, run live under Synth 1.1-5, traced every
# synth() call (benchmarks/studies/fpp_cherry_picking). These tests rebuild
# Table 3, the joint test, both choice rules and the Figure 2 band from the
# captured donor weights alone, so they check the inference above against the
# authors' own numbers without R.

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_TRACE = _ROOT / "benchmarks" / "reference" / "fpp_cherry_picking" / "germany_trace.json"
_CODE = [f"{k}{c}" for k in "1234567" for c in "ab"]
# The code's "a" carries the covariates; the paper's tables call that "b".
_PAPER = {s: s[0] + ("b" if s[1] == "a" else "a") for s in _CODE}
_TABLE3 = {"1a": 0.059, "1b": 0.059, "2a": 0.059, "2b": 0.118, "3a": 0.118,
           "3b": 0.059, "4a": 0.059, "4b": 0.059, "5a": 0.118, "5b": 0.059,
           "6a": 0.588, "6b": 0.059, "7a": 0.353, "7b": 0.059}


@pytest.fixture(scope="module")
def germany():
    d = pd.read_stata(_ROOT / "basedata" / "repgermany.dta")
    d["index"] = d["index"].astype(int)
    units = list(dict.fromkeys(d["index"]))
    Y = (d.pivot(index="year", columns="index", values="gdp")
         .loc[1960.0:2003.0, units].to_numpy(float))
    trace = json.loads(_TRACE.read_text())
    N = len(units)
    W = np.full((len(_CODE), N, N), np.nan)
    calls = trace["synth_calls"]
    # One capture per specification; training and main calls alternate per
    # unit, and the main call's weights are the design's. 7a skips Austria.
    for part in range(1, len(_CODE) + 1):
        block = [c for c in calls if c["part"] == part]
        for train, main in zip(block[0::2], block[1::2]):
            assert train["treat"] == main["treat"]
            j = units.index(int(main["treat"]))
            W[part - 1, j] = 0.0
            W[part - 1, j, [units.index(int(c)) for c in main["controls"]]] = main["w"]
    results = np.asarray(trace["objects"]["results"]["values"], float)
    years = np.arange(1960, 2004)
    return Y, W, units.index(7), years, results


def _gaps(Y, W):
    return np.stack([synthetic_gaps(Y, Wd) for Wd in W])


def test_fpp_table3_pvalues_match_the_authors_run(germany):
    Y, W, pos, years, results = germany
    G = _gaps(Y, W)
    pre, post = years <= 1989, years >= 1990
    for d, spec in enumerate(_CODE):
        stat = ratio_statistic(G[d], pre, post, root=True)
        p = permutation_pvalue(stat, pos)
        assert p == pytest.approx(results[d, 0], abs=1e-12), spec
        # The table prints 1/17 for 7a, which skips Austria; its code divides by 16.
        assert p == pytest.approx(_TABLE3[_PAPER[spec]], abs=0.004), spec


def test_fpp_joint_test_rejects_at_one_over_seventeen(germany):
    Y, W, pos, years, _ = germany
    G = _gaps(Y, W)
    pre, post = years <= 1989, years >= 1990
    as_coded = np.column_stack([ratio_statistic(G[d], pre, post, root=True)
                                for d in range(len(_CODE))])
    assert joint_pvalue(as_coded, pos) == pytest.approx(1 / 17)
    one_to_five = [d for d, s in enumerate(_CODE) if s[0] in "12345"]
    as_described = np.column_stack([ratio_statistic(G[d], pre, post)
                                    for d in one_to_five])
    assert joint_pvalue(as_described, pos) == pytest.approx(1 / 17)


def test_fpp_choice_rules_pick_the_papers_specifications(germany):
    Y, W, pos, years, _ = germany
    G = _gaps(Y, W)
    one_to_five = [d for d, s in enumerate(_CODE) if s[0] in "12345"]
    dz = dube_zipperer(G, years >= 1990, pos)
    assert _CODE[int(np.argmin(dz))] == "6b"  # the authors pass over it
    assert _PAPER[_CODE[one_to_five[int(np.argmin(dz[one_to_five]))]]] == "1a"
    daw = donohue(G[one_to_five], (years >= 1981) & (years <= 1989), pos)
    assert _PAPER[_CODE[one_to_five[int(np.argmin(daw))]]] == "2b"


def test_fpp_figure2_band(germany):
    Y, W, pos, years, _ = germany
    # cs_germany.R overwrites specifications 2 and 3 with specification 1's
    # weights, so its "average of 1 to 5" is this list.
    as_coded = [_CODE.index(s) for s in
                ["1a", "1b", "1a", "1b", "1a", "1b", "4a", "4b", "5a", "5b"]]
    n_pre = int(np.flatnonzero(years == 1990)[0]) + 1
    band = fp_band(Y, W[as_coded], pos, n_pre=n_pre, significance=2 / 17 + 0.001)
    # The authors' own figure output: 2003 band about -5,720 to -2,780 around -4,440.
    assert band.average[-1] == pytest.approx(-4429.389006817, abs=1e-6)
    assert band.lower[-1] == pytest.approx(-5740.641499997, abs=1e-6)
    assert band.upper[-1] == pytest.approx(-2777.175044449, abs=1e-6)
    one_to_five = [d for d, s in enumerate(_CODE) if s[0] in "12345"]
    fixed = fp_band(Y, W[one_to_five], pos, n_pre=n_pre, significance=2 / 17 + 0.001)
    assert fixed.shift_lower == pytest.approx(-1139.659645103, abs=1e-6)
    assert fixed.shift_upper == pytest.approx(1422.545714356, abs=1e-6)


def test_fp_band_reports_a_side_without_rejection():
    Y, W = panel()
    with pytest.warns(RuntimeWarning, match="No rejection"):
        band = fp_band(Y, W, 0, n_pre=8, significance=0.01, precision=2, max_steps=4)
    assert band.shift_upper == np.inf and band.shift_lower == -np.inf
    assert np.isposinf(band.upper[8:]).all() and np.isneginf(band.lower[8:]).all()


def test_fp_band_needs_the_treated_unit_fit_in_every_design():
    Y, W = panel()
    W[1, 0] = np.nan
    with pytest.raises(MlsynthDataError, match="every design"):
        fp_band(Y, W, 0, n_pre=8, significance=0.25)


def test_identified_range_reports_nonfinite_inputs_and_failed_programs():
    with pytest.raises(MlsynthDataError, match="finite"):
        identified_effect_range(np.array([[np.nan, 1.0]]), np.array([0.5, 0.5]),
                                np.ones(1), np.ones((1, 2)))
    from unittest.mock import patch

    class Failed:
        status, message = 2, "infeasible"

    with patch("mlsynth.utils.sensitivity_helpers.inference.linprog", return_value=Failed()):
        with pytest.raises(MlsynthEstimationError, match="infeasible"):
            identified_effect_range(np.ones((1, 2)), np.array([0.5, 0.5]),
                                    np.ones(1), np.ones((1, 2)))


def test_misshaped_gap_and_weight_arrays_are_reported():
    pre, post = masks(3, 2)
    with pytest.raises(MlsynthDataError, match="periods, units"):
        ratio_statistic(np.ones(3), pre, post)
    with pytest.raises(MlsynthDataError, match="for 2 units"):
        synthetic_gaps(np.ones((3, 2)), np.zeros((3, 3)))


def test_ratio_statistic_survives_tiny_and_huge_gaps():
    # Squaring these in their own units underflows or overflows.
    gaps = np.array([[1.8e-161, 1e200], [0.0, 1e200], [3.6e-161, 2e200]])
    pre, post = masks(3, 2)
    np.testing.assert_allclose(ratio_statistic(gaps, pre, post), [8.0, 4.0])
