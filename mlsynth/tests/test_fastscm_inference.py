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
