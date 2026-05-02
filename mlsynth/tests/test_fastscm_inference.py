import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.inference import (
    compute_post_inference,
    compute_moving_block_conformal_ci,
)
from mlsynth.utils.fast_scm_helpers.structure import (
    SEDCandidate,
    Identification,
    WeightVectors,
    PredictionVectors,
    Losses,
)

# =========================================================
# FIXTURE HELPERS
# =========================================================

class DummySolution:
    label = "test"


def make_candidate(residuals_B, effects_post):
    return SEDCandidate(
        identification=Identification(
            solution=DummySolution(),
            treated_idx=np.array([0]),
        ),
        weights=WeightVectors(
            treated=np.array([1.0]),
            control=np.array([0.2, 0.8]),
        ),
        predictions=PredictionVectors(
            synthetic_treated=np.array([]),
            synthetic_control=np.array([]),
            effects=np.array(effects_post),
            residuals_E=np.array([]),
            residuals_B=np.array(residuals_B),
        ),
        losses=Losses(0, 0, 0, 0, 0, 0, 0),
    )

# =========================================================
# POST INFERENCE (SMOKE + EDGE)
# =========================================================

def test_compute_post_inference_basic_deterministic():
    cand = make_candidate(
        residuals_B=np.array([1.0, -1.0, 0.5]),
        effects_post=np.array([2.0, 2.0, 2.0]),
    )

    post_idx = np.array([0, 1, 2])

    out = compute_post_inference(
        candidate=cand,
        post_idx=post_idx,
        n_perms=200,
        seed=123,
    )

    assert out.inference.p_value is not None
    assert 0.0 <= out.inference.p_value <= 1.0
    assert np.isclose(out.inference.ate, 2.0)


def test_compute_post_inference_empty_post():
    cand = make_candidate(
        residuals_B=np.array([1.0, 2.0]),
        effects_post=np.array([]),
    )

    out = compute_post_inference(
        candidate=cand,
        post_idx=np.array([], dtype=int),
    )

    # Correct expectation for empty post-period
    assert out.inference.ate == 0.0
    assert out.inference.p_value is None

def test_compute_post_inference_extreme_case_all_equal():
    cand = make_candidate(
        residuals_B=np.array([1.0, 1.0, 1.0]),
        effects_post=np.array([1.0, 1.0]),
    )

    out = compute_post_inference(
        candidate=cand,
        post_idx=np.array([0, 1]),
        n_perms=100,
        seed=0,
    )

    # Perfectly identical distributions should yield high p-values
    assert out.inference.p_value >= 0.5

# =========================================================
# MOVING BLOCK CONFORMAL CI
# =========================================================

def test_conformal_ci_basic_shape_and_bounds():
    cand = make_candidate(
        residuals_B=np.array([1.0, -1.0, 2.0, -2.0, 1.5]),
        effects_post=np.array([1.0, 2.0, 1.5]),
    )

    out = compute_moving_block_conformal_ci(
        candidate=cand,
        post_idx=np.array([0, 1, 2]),
        alpha=0.1,
        seed=42,
    )

    assert out.inference.ci_lower is not None
    assert out.inference.ci_upper is not None
    assert out.inference.ci_lower <= out.inference.ci_upper


def test_conformal_ci_empty_post_returns_nan():
    cand = make_candidate(
        residuals_B=np.array([1.0, 2.0]),
        effects_post=np.array([]),
    )

    out = compute_moving_block_conformal_ci(
        candidate=cand,
        post_idx=np.array([]),
    )

    assert np.isnan(out.inference.ci_lower)
    assert np.isnan(out.inference.ci_upper)


def test_conformal_ci_contains_ate_in_normal_case():
    cand = make_candidate(
        residuals_B=np.random.normal(0, 1, 50),
        effects_post=np.random.normal(1.0, 0.5, 20),
    )

    out = compute_moving_block_conformal_ci(
        candidate=cand,
        post_idx=np.arange(20),
        alpha=0.2,
        seed=1,
    )

    ate = float(np.mean(cand.predictions.effects))

    # sanity check: interval should usually contain point estimate
    assert out.inference.ci_lower <= ate <= out.inference.ci_upper

# =========================================================
# SMOKE: FULL PIPELINE COMPATIBILITY
# =========================================================

def test_pipeline_no_crash_smoke():
    cand = make_candidate(
        residuals_B=np.random.randn(10),
        effects_post=np.random.randn(5),
    )

    post_idx = np.array([0, 1, 2, 3, 4])

    c1 = compute_post_inference(cand, post_idx, n_perms=50, seed=0)
    c2 = compute_moving_block_conformal_ci(c1, post_idx, alpha=0.1)

    assert c2.inference.p_value is not None
    assert c2.inference.ci_lower is not None




import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.inference import (
    compute_post_inference,
    compute_moving_block_conformal_ci,
)
from mlsynth.utils.fast_scm_helpers.structure import (
    SEDCandidate,
    PredictionVectors,
    Inference,
    Losses,
    WeightVectors,
    Identification,
)

# =========================================================
# HELPERS
# =========================================================

class DummySolution:
    label = "stress"


def make_candidate(residuals_B, effects_post):
    return SEDCandidate(
        identification=Identification(
            solution=DummySolution(),
            treated_idx=np.array([0]),
        ),
        weights=WeightVectors(
            treated=np.array([1.0]),
            control=np.array([0.0, 0.5, 0.5]),
        ),
        predictions=PredictionVectors(
            synthetic_treated=np.array([]),
            synthetic_control=np.array([]),
            effects=effects_post,
            residuals_E=np.array([]),
            residuals_B=np.array(residuals_B),
        ),
        losses=Losses(0, 0, 0, 0, 0, 0, 0),
        inference=Inference(),
    )


# =========================================================
# POST INFERENCE STRESS TESTS
# =========================================================

def test_post_inference_determinism():
    cand1 = make_candidate(
        residuals_B=np.random.randn(50),
        effects_post=np.random.randn(10),
    )
    cand2 = make_candidate(
        residuals_B=cand1.predictions.residuals_B.copy(),
        effects_post=cand1.predictions.effects.copy(),
    )

    out1 = compute_post_inference(cand1, np.arange(10), n_perms=500, seed=123)
    out2 = compute_post_inference(cand2, np.arange(10), n_perms=500, seed=123)

    assert np.isclose(out1.inference.p_value, out2.inference.p_value)
    assert np.isclose(out1.inference.ate, out2.inference.ate)


def test_post_inference_all_zero_effects():
    cand = make_candidate(
        residuals_B=np.zeros(20),
        effects_post=np.zeros(5),
    )

    out = compute_post_inference(cand, np.arange(5), n_perms=200)

    assert out.inference.ate == 0.0
    assert 0.0 <= out.inference.p_value <= 1.0


def test_post_inference_constant_signal():
    cand = make_candidate(
        residuals_B=np.ones(30),
        effects_post=np.ones(10),
    )

    out = compute_post_inference(cand, np.arange(10), n_perms=200)

    assert np.isfinite(out.inference.p_value)
    assert out.inference.ate == 1.0


def test_post_inference_small_post_sample():
    cand = make_candidate(
        residuals_B=np.random.randn(100),
        effects_post=np.array([1.0]),
    )

    out = compute_post_inference(cand, np.array([0]), n_perms=200)

    assert np.isfinite(out.inference.p_value)
    assert out.inference.ate == 1.0


def test_post_inference_empty_post_graceful():
    cand = make_candidate(
        residuals_B=np.random.randn(20),
        effects_post=np.array([]),
    )

    out = compute_post_inference(cand, np.array([]), n_perms=200)

    assert out.inference.ate == 0.0
    assert out.inference.p_value is None or np.isnan(out.inference.p_value)


# =========================================================
# CONFORMAL CI STRESS TESTS
# =========================================================

def test_conformal_ci_ordering():
    cand = make_candidate(
        residuals_B=np.random.randn(50),
        effects_post=np.random.randn(10),
    )

    out = compute_moving_block_conformal_ci(cand, np.arange(10))

    assert out.inference.ci_lower <= out.inference.ci_upper


def test_conformal_ci_finite():
    cand = make_candidate(
        residuals_B=np.random.randn(50),
        effects_post=np.random.randn(10),
    )

    out = compute_moving_block_conformal_ci(cand, np.arange(10))

    assert np.isfinite(out.inference.ci_lower)
    assert np.isfinite(out.inference.ci_upper)


def test_conformal_ci_empty_post():
    cand = make_candidate(
        residuals_B=np.random.randn(50),
        effects_post=np.array([]),
    )

    out = compute_moving_block_conformal_ci(cand, np.array([]))

    assert np.isnan(out.inference.ci_lower)
    assert np.isnan(out.inference.ci_upper)


def test_conformal_ci_constant_signal():
    cand = make_candidate(
        residuals_B=np.ones(50),
        effects_post=np.ones(10),
    )

    out = compute_moving_block_conformal_ci(cand, np.arange(10))

    assert out.inference.ci_upper >= out.inference.ci_lower


# =========================================================
# ROBUSTNESS EDGE CASES
# =========================================================

def test_mixed_nan_residuals():
    cand = make_candidate(
        residuals_B=np.array([np.nan, 1.0, 2.0, np.nan]),
        effects_post=np.array([1.0, 2.0, 3.0]),
    )

    out = compute_post_inference(cand, np.arange(3), n_perms=100)

    # Should not crash
    assert np.isfinite(out.inference.ate) or np.isnan(out.inference.ate)


def test_large_permutation_stability():
    cand = make_candidate(
        residuals_B=np.random.randn(200),
        effects_post=np.random.randn(50),
    )

    out = compute_post_inference(cand, np.arange(50), n_perms=2000, seed=999)

    assert 0.0 <= out.inference.p_value <= 1.0
