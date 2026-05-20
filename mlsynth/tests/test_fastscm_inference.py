"""Tests for moving-block conformal inference in fast_scm_helpers.

The companion ``compute_post_inference`` function was removed during the
LEXSCM refactor (its responsibilities were folded into
``fast_scm_setup._run_post_intervention_updates``), so this test module
now exercises only the surviving public surface,
:func:`compute_moving_block_conformal_ci`.
"""

import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.inference import compute_moving_block_conformal_ci
from mlsynth.utils.fast_scm_helpers.structure import (
    Identification,
    Inference,
    Losses,
    PredictionVectors,
    SEDCandidate,
    WeightVectors,
)


# =========================================================
# FIXTURE HELPERS
# =========================================================

class _DummySolution:
    label = "test"


def make_candidate(residuals_B, effects_post, synthetic_treated=None):
    """Build a minimal SEDCandidate exercising the conformal CI path."""
    effects_arr = np.asarray(effects_post)
    if synthetic_treated is None:
        # The conformal CI uses ``synthetic_treated`` to derive the
        # percentage-lift baseline. A constant non-zero series works.
        synthetic_treated = np.ones_like(effects_arr, dtype=float)

    return SEDCandidate(
        identification=Identification(
            solution=_DummySolution(),
            treated_idx=np.array([0]),
        ),
        weights=WeightVectors(
            treated=np.array([1.0]),
            control=np.array([0.2, 0.8]),
        ),
        predictions=PredictionVectors(
            synthetic_treated=np.asarray(synthetic_treated, dtype=float),
            synthetic_control=np.zeros_like(effects_arr, dtype=float),
            effects=effects_arr.astype(float),
            residuals_E=np.array([]),
            residuals_B=np.asarray(residuals_B, dtype=float),
        ),
        losses=Losses(0, 0, 0, 0, 0, 0, 0),
        inference=Inference(),
    )


# =========================================================
# BASIC SHAPE
# =========================================================

def test_conformal_ci_basic_shape_and_bounds():
    cand = make_candidate(
        residuals_B=[1.0, -1.0, 2.0, -2.0, 1.5],
        effects_post=[1.0, 2.0, 1.5],
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
        residuals_B=[1.0, 2.0],
        effects_post=[],
    )

    out = compute_moving_block_conformal_ci(
        candidate=cand,
        post_idx=np.array([]),
    )

    assert np.isnan(out.inference.ci_lower)
    assert np.isnan(out.inference.ci_upper)


def test_conformal_ci_contains_ate_in_normal_case():
    rng = np.random.default_rng(1)
    cand = make_candidate(
        residuals_B=rng.normal(0, 1, 50),
        effects_post=rng.normal(1.0, 0.5, 20),
    )

    out = compute_moving_block_conformal_ci(
        candidate=cand,
        post_idx=np.arange(20),
        alpha=0.2,
        seed=1,
    )

    ate = float(np.mean(cand.predictions.effects))
    assert out.inference.ci_lower <= ate <= out.inference.ci_upper


# =========================================================
# DETERMINISM
# =========================================================

def test_conformal_ci_deterministic_with_same_seed():
    rng = np.random.default_rng(3)
    residuals = rng.normal(0, 1, 30)
    effects = rng.normal(0.5, 0.3, 10)

    out1 = compute_moving_block_conformal_ci(
        candidate=make_candidate(residuals_B=residuals, effects_post=effects),
        post_idx=np.arange(10),
        alpha=0.1,
        seed=7,
    )
    out2 = compute_moving_block_conformal_ci(
        candidate=make_candidate(residuals_B=residuals, effects_post=effects),
        post_idx=np.arange(10),
        alpha=0.1,
        seed=7,
    )

    assert out1.inference.ci_lower == out2.inference.ci_lower
    assert out1.inference.ci_upper == out2.inference.ci_upper
    assert out1.inference.p_value == out2.inference.p_value


# =========================================================
# STRESS / EDGE CASES
# =========================================================

def test_conformal_ci_ordering():
    rng = np.random.default_rng(0)
    cand = make_candidate(
        residuals_B=rng.normal(0, 1, 50),
        effects_post=rng.normal(0, 1, 10),
    )

    out = compute_moving_block_conformal_ci(cand, np.arange(10))

    assert out.inference.ci_lower <= out.inference.ci_upper


def test_conformal_ci_finite():
    rng = np.random.default_rng(0)
    cand = make_candidate(
        residuals_B=rng.normal(0, 1, 50),
        effects_post=rng.normal(0, 1, 10),
    )

    out = compute_moving_block_conformal_ci(cand, np.arange(10))

    assert np.isfinite(out.inference.ci_lower)
    assert np.isfinite(out.inference.ci_upper)


def test_conformal_ci_empty_post():
    rng = np.random.default_rng(0)
    cand = make_candidate(
        residuals_B=rng.normal(0, 1, 50),
        effects_post=[],
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
