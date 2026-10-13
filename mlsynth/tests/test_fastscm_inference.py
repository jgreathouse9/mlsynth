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


# =========================================================
# THREE DEFECTS FOUND BY READING THE IMPLEMENTATION
# =========================================================
# Measured context: inverting a constant-effect sharp null returns an empty
# acceptance set for 4 per cent of units at a 100-period blank window and 14 per
# cent at 24, so the empty case is ordinary and has to be reported as itself.
# Conditional on returning an interval the procedure covers at its nominal
# level (0.89 measured against 0.90 over 450 draws per cell); a fabricated
# fallback interval is outside that guarantee and carries no coverage at all.

def _ar1(rng, n, rho, sd=1.0):
    e = rng.normal(0.0, sd, n)
    out = np.empty(n)
    out[0] = e[0] / np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + e[t]
    return out


def test_an_empty_acceptance_set_is_reported_not_replaced():
    """No theta accepted means no constant effect fits, which is a result.

    Substituting ``observed_ate +/- 4 * std_err_proxy`` returns something that
    looks like a conformal interval, is not one, and carries no coverage
    guarantee. The caller cannot tell the two apart from the fields alone.
    """
    rng = np.random.default_rng(4)
    blank = rng.normal(0.0, 0.01, 60)          # a null with almost no spread
    post = np.array([40.0, -38.0, 41.0, -39.0, 42.0, -41.0])  # nothing constant fits
    cand = make_candidate(blank, post)
    with pytest.warns(UserWarning, match="no constant"):
        out = compute_moving_block_conformal_ci(cand, np.arange(post.size), alpha=0.10)
    assert np.isnan(out.inference.ci_lower)
    assert np.isnan(out.inference.ci_upper)
    assert out.inference.conformal_empty is True


def test_a_returned_interval_is_flagged_as_non_empty():
    rng = np.random.default_rng(5)
    cand = make_candidate(_ar1(rng, 60, 0.1), _ar1(rng, 8, 0.1))
    out = compute_moving_block_conformal_ci(cand, np.arange(8), alpha=0.10)
    assert out.inference.conformal_empty is False
    assert np.isfinite(out.inference.ci_lower)


def test_the_grid_is_wide_enough_not_to_clip_the_interval():
    """Under dependence the search range has to be charged on effective periods.

    ``std(e_B) / sqrt(n_post)`` treats the post window as independent, which
    understates the standard error and can leave the accepted set touching the
    grid boundary -- an interval truncated by the search, not by the data.
    """
    rng = np.random.default_rng(6)
    blank = _ar1(rng, 80, 0.7, sd=2.0)
    post = _ar1(rng, 8, 0.7, sd=2.0)
    cand = make_candidate(blank, post)
    out = compute_moving_block_conformal_ci(cand, np.arange(8), alpha=0.10)
    span = out.inference.ci_upper - out.inference.ci_lower
    assert span < out.inference.conformal_grid_span * 0.98, (
        "accepted set reaches the grid edge, so the interval is clipped by the "
        "search range rather than determined by the data")


def test_total_lift_is_the_quantity_the_gap_is_measured_in():
    """The gap is for the weighted synthetic treated unit, so the total is
    gap x periods. The old multiplier read ``candidate.treated``, which
    SEDCandidate does not define, so it silently resolved to 1 -- the right
    answer reached by a route that would have broken the moment the attribute
    appeared."""
    rng = np.random.default_rng(7)
    post = _ar1(rng, 8, 0.1) + 3.0
    cand = make_candidate(_ar1(rng, 60, 0.1), post)
    out = compute_moving_block_conformal_ci(cand, np.arange(8), alpha=0.10)
    assert out.inference.total_lift == pytest.approx(float(np.mean(post)) * 8)
