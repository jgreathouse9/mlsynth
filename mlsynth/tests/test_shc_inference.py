"""Tests for mlsynth.utils.shc_helpers.inference.

Covers the Andrews-Genton agnostic conformal bands (:func:`ag_conformal`),
the Chen-Yang-Yang conformal permutation test (:func:`shc_conformal_test`),
and the orchestration helper (:func:`run_conformal_inference`).

Assertions are structural (shapes, padding, sum/quantile relationships,
ordering of bounds, determinism under a fixed seed) rather than magic
numbers copied from current output.

The validation/error paths assert the documented :class:`MlsynthDataError` /
:class:`MlsynthConfigError` exceptions.
"""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.helperutils import IndexSet
from mlsynth.utils.shc_helpers.inference import (
    ag_conformal,
    run_conformal_inference,
    shc_conformal_test,
)
from mlsynth.utils.shc_helpers.structures import (
    SHCDesign,
    SHCInference,
    SHCInputs,
)


# --------------------------------------------------------------------------
# ag_conformal -- success path
# --------------------------------------------------------------------------
def test_ag_conformal_shapes_and_padding():
    rng = np.random.default_rng(0)
    actual = rng.standard_normal(8)
    pred_pre = actual + 0.1 * rng.standard_normal(8)
    pred_post = rng.standard_normal(5)
    lower, upper = ag_conformal(actual, pred_pre, pred_post, miscoverage_rate=0.1)

    # Full series = pre-period padding (T_pre) + post-period bounds (T_post).
    assert lower.shape == (8 + 5,)
    assert upper.shape == (8 + 5,)
    # First T_pre entries are pad_value (NaN by default).
    assert np.all(np.isnan(lower[:8]))
    assert np.all(np.isnan(upper[:8]))
    # Post-period bounds are finite.
    assert np.all(np.isfinite(lower[8:]))
    assert np.all(np.isfinite(upper[8:]))
    # Lower bound never exceeds upper bound.
    assert np.all(lower[8:] <= upper[8:])
    # Output is 1D.
    assert lower.ndim == 1 and upper.ndim == 1


def test_ag_conformal_interval_centered_on_prediction_plus_mean_residual():
    # The post bounds straddle (prediction + mean_residual) symmetrically.
    actual = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
    pred_pre = np.array([10.5, 11.5, 10.5, 12.5, 11.5])
    pred_post = np.array([14.0, 15.0, 14.5])
    lower, upper = ag_conformal(actual, pred_pre, pred_post, miscoverage_rate=0.1)

    mean_resid = np.mean(actual - pred_pre)
    centers = pred_post + mean_resid
    mids = 0.5 * (lower[5:] + upper[5:])
    assert np.allclose(mids, centers)
    # Half-width is identical across post periods (depends only on residuals).
    half_widths = 0.5 * (upper[5:] - lower[5:])
    assert np.allclose(half_widths, half_widths[0])
    assert half_widths[0] > 0


def test_ag_conformal_custom_pad_value():
    actual = np.array([1.0, 2.0, 3.0])
    pred_pre = np.array([1.0, 2.0, 3.0])
    pred_post = np.array([4.0, 5.0])
    lower, upper = ag_conformal(actual, pred_pre, pred_post, pad_value=-999.0)
    assert np.all(lower[:3] == -999.0)
    assert np.all(upper[:3] == -999.0)


def test_ag_conformal_smaller_miscoverage_gives_wider_band():
    # Lower alpha (higher coverage) -> wider interval via log(2/alpha).
    rng = np.random.default_rng(1)
    actual = rng.standard_normal(10)
    pred_pre = actual + 0.2 * rng.standard_normal(10)
    pred_post = rng.standard_normal(4)
    lo90, up90 = ag_conformal(actual, pred_pre, pred_post, miscoverage_rate=0.10)
    lo99, up99 = ag_conformal(actual, pred_pre, pred_post, miscoverage_rate=0.01)
    width90 = up90[10:] - lo90[10:]
    width99 = up99[10:] - lo99[10:]
    assert np.all(width99 >= width90)


# --------------------------------------------------------------------------
# ag_conformal -- error paths
# --------------------------------------------------------------------------
def test_ag_conformal_length_mismatch_raises():
    with pytest.raises(MlsynthDataError):
        ag_conformal(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))


def test_ag_conformal_empty_pre_raises():
    with pytest.raises(MlsynthDataError):
        ag_conformal(np.array([]), np.array([]), np.array([1.0]))


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, 1.5, -0.1])
def test_ag_conformal_bad_miscoverage_raises(bad_rate):
    with pytest.raises(MlsynthConfigError):
        ag_conformal(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            miscoverage_rate=bad_rate,
        )


# --------------------------------------------------------------------------
# shc_conformal_test -- success path
# --------------------------------------------------------------------------
def test_shc_conformal_test_keys_and_structure():
    rng = np.random.default_rng(0)
    pre = rng.standard_normal(30)
    post = rng.standard_normal(5)
    res = shc_conformal_test(pre, post, num_resamples=500, random_state=0)

    assert set(res) == {
        "test_statistic",
        "p_value",
        "critical_values",
        "reject",
        "null_distribution",
        "num_resamples",
        "levels",
    }
    # Test statistic = n^{-1/2} * sum|post|.
    expected_stat = np.sum(np.abs(post)) / np.sqrt(post.size)
    assert np.isclose(res["test_statistic"], expected_stat)

    assert res["null_distribution"].shape == (500,)
    assert res["num_resamples"] == 500
    assert res["levels"] == (0.01, 0.05, 0.10)
    # p-value is a valid probability.
    assert 0.0 <= res["p_value"] <= 1.0
    # p_value == fraction of null >= statistic.
    assert np.isclose(
        res["p_value"], np.mean(res["null_distribution"] >= res["test_statistic"])
    )


def test_shc_conformal_test_critical_values_and_reject():
    rng = np.random.default_rng(0)
    pre = rng.standard_normal(40)
    post = rng.standard_normal(6)
    res = shc_conformal_test(pre, post, num_resamples=800, random_state=0)

    cvs = res["critical_values"]
    # Lower significance level -> higher upper-tail critical value.
    assert cvs[0.01] >= cvs[0.05] >= cvs[0.10]
    # reject decision is consistent with statistic > critical value.
    for lvl in (0.01, 0.05, 0.10):
        assert res["reject"][lvl] == (res["test_statistic"] > cvs[lvl])
        assert isinstance(res["reject"][lvl], bool)


def test_shc_conformal_test_custom_levels():
    rng = np.random.default_rng(2)
    pre = rng.standard_normal(20)
    post = rng.standard_normal(4)
    res = shc_conformal_test(pre, post, num_resamples=200, levels=(0.2,), random_state=1)
    assert res["levels"] == (0.2,)
    assert set(res["critical_values"]) == {0.2}
    assert set(res["reject"]) == {0.2}


def test_shc_conformal_test_determinism():
    pre = np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.4])
    post = np.array([0.5, -0.6, 0.2])
    a = shc_conformal_test(pre, post, num_resamples=300, random_state=7)
    b = shc_conformal_test(pre, post, num_resamples=300, random_state=7)
    assert a["test_statistic"] == b["test_statistic"]
    assert a["p_value"] == b["p_value"]
    assert np.array_equal(a["null_distribution"], b["null_distribution"])


def test_shc_conformal_test_accepts_list_and_2d_inputs():
    # ravel/asarray should accept python lists and flatten 2D column vectors.
    res = shc_conformal_test(
        [[0.1], [0.2], [-0.3]], [[0.4], [0.5]], num_resamples=100, random_state=0
    )
    assert np.isfinite(res["test_statistic"])
    assert res["null_distribution"].shape == (100,)


def test_shc_conformal_test_empty_pre_raises():
    with pytest.raises(MlsynthDataError):
        shc_conformal_test(np.array([]), np.array([1.0]))


def test_shc_conformal_test_empty_post_raises():
    with pytest.raises(MlsynthDataError):
        shc_conformal_test(np.array([1.0]), np.array([]))


# --------------------------------------------------------------------------
# run_conformal_inference -- orchestration
# --------------------------------------------------------------------------
def _make_inputs(T=12, T0=9, m=3):
    return SHCInputs(
        time_index=IndexSet.from_labels(np.arange(T)),
        y=np.arange(T, dtype=float),
        T0=T0,
        m=m,
        treated_label="A",
    )


def _make_design(T0=9, m=3, n=3):
    return SHCDesign(
        bandwidth=1.0,
        latent_pre=np.arange(T0, dtype=float) + 0.1,
        weights=np.array([1.0]),
        selected_blocks=[0],
        block_weights={"b0": 1.0},
        counterfactual_window=np.zeros(m + n),
        use_augmented=False,
    )


def test_run_conformal_inference_returns_inference_object():
    T, T0, m = 12, 9, 3
    n = T - T0
    inputs = _make_inputs(T, T0, m)
    design = _make_design(T0, m, n)
    rng = np.random.default_rng(0)
    observed = rng.standard_normal(m + n)
    counterfactual = observed - 0.2

    res = run_conformal_inference(
        inputs, design, observed, counterfactual, num_resamples=200, random_state=0
    )
    assert isinstance(res, SHCInference)
    assert res.method == "conformal_permutation"
    # confidence_level = 1 - miscoverage_rate (default 0.10).
    assert np.isclose(res.confidence_level, 0.90)
    # Post-period conformal bands have length n.
    assert res.conformal_lower.shape == (n,)
    assert res.conformal_upper.shape == (n,)
    assert np.all(res.conformal_lower <= res.conformal_upper)
    assert res.null_distribution.shape == (200,)
    assert 0.0 <= res.p_value <= 1.0
    assert res.num_resamples == 200


def test_run_conformal_inference_exact_method_moving_block():
    T, T0, m = 12, 9, 3
    n = T - T0
    inputs = _make_inputs(T, T0, m)
    design = _make_design(T0, m, n)
    rng = np.random.default_rng(1)
    observed = rng.standard_normal(m + n)
    counterfactual = observed - 0.2

    res = run_conformal_inference(
        inputs, design, observed, counterfactual,
        method="exact", permutation_scheme="moving_block",
    )
    assert isinstance(res, SHCInference)
    assert res.method == "conformal_exact_moving_block"
    # moving block enumerates T = T0 + n cyclic shifts.
    assert res.null_distribution.shape == (T0 + n,)
    assert res.num_resamples == T0 + n
    assert 0.0 <= res.p_value <= 1.0


def test_run_conformal_inference_exact_method_iid():
    T, T0, m = 12, 9, 3
    n = T - T0
    inputs = _make_inputs(T, T0, m)
    design = _make_design(T0, m, n)
    rng = np.random.default_rng(2)
    observed = rng.standard_normal(m + n)
    counterfactual = observed - 0.2

    res = run_conformal_inference(
        inputs, design, observed, counterfactual,
        method="exact", permutation_scheme="iid", num_permutations=250,
    )
    assert res.method == "conformal_exact_iid"
    assert res.num_resamples == 250


def test_run_conformal_inference_bad_method_raises():
    inputs = _make_inputs()
    design = _make_design()
    observed = np.zeros(6)
    counterfactual = np.zeros(6)
    with pytest.raises(MlsynthConfigError):
        run_conformal_inference(
            inputs, design, observed, counterfactual, method="bogus",
        )


def test_run_conformal_inference_custom_miscoverage_rate():
    T, T0, m = 12, 9, 3
    n = T - T0
    inputs = _make_inputs(T, T0, m)
    design = _make_design(T0, m, n)
    observed = np.linspace(0, 1, m + n)
    counterfactual = observed - 0.05
    res = run_conformal_inference(
        inputs,
        design,
        observed,
        counterfactual,
        miscoverage_rate=0.05,
        num_resamples=150,
        levels=(0.05,),
        random_state=3,
    )
    assert np.isclose(res.confidence_level, 0.95)
    assert set(res.critical_values) == {0.05}
    assert set(res.reject) == {0.05}


def test_run_conformal_inference_determinism():
    T, T0, m = 12, 9, 3
    n = T - T0
    inputs = _make_inputs(T, T0, m)
    design = _make_design(T0, m, n)
    observed = np.arange(m + n, dtype=float)
    counterfactual = observed - 0.3
    a = run_conformal_inference(
        inputs, design, observed, counterfactual, num_resamples=200, random_state=5
    )
    b = run_conformal_inference(
        inputs, design, observed, counterfactual, num_resamples=200, random_state=5
    )
    assert a.test_statistic == b.test_statistic
    assert a.p_value == b.p_value
    assert np.array_equal(a.null_distribution, b.null_distribution)
