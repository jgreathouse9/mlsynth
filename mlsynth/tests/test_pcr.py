"""Tests for the shared PCR kernel (:mod:`mlsynth.utils.pcr.core`).

``pcr_weights`` is reached by SI, ClusterSC, RSC and SCMO, but until now only
transitively and only on well-conditioned panels. The degenerate half of its
input domain -- a design whose numerical rank falls below the requested rank --
was untested, which is where the pseudo-inverse divides by a singular value at
or near zero.
"""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pcr import hsvt, pcr_weights


def _well_conditioned(m=40, n=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n)), rng.standard_normal(m)


def _rank_deficient(m=40, n=12, true_rank=3, seed=0, exact_zeros=False):
    """A design whose numerical rank is ``true_rank`` but which has ``n`` columns."""
    rng = np.random.default_rng(seed)
    if exact_zeros:
        design = rng.standard_normal((m, n))
        design[:, true_rank:] = 0.0
        return design
    basis = rng.standard_normal((m, true_rank))
    return basis @ rng.standard_normal((true_rank, n))


# --- smoke -----------------------------------------------------------------


def test_pcr_weights_smoke():
    """Runs end to end on a minimal valid input and returns a finite vector."""
    design, target = _well_conditioned()
    weights = pcr_weights(design, target, rank=3)
    assert weights.shape == (design.shape[1],)
    assert np.all(np.isfinite(weights))


# --- unit ------------------------------------------------------------------


def test_pcr_weights_matches_closed_form():
    """w = sum_{i<=r} (1/s_i) v_i u_i' y, the definition in the docstring."""
    design, target = _well_conditioned()
    U, s, Vt = np.linalg.svd(design, full_matrices=False)
    for rank in (1, 2, 4, 6):
        expected = sum((1.0 / s[i]) * Vt[i] * (U[:, i] @ target) for i in range(rank))
        np.testing.assert_allclose(pcr_weights(design, target, rank), expected,
                                   rtol=0, atol=1e-12)


def test_pcr_weights_at_full_rank_is_the_pseudo_inverse():
    """At full rank the truncation is vacuous, so PCR is the least-squares fit."""
    design, target = _well_conditioned()
    np.testing.assert_allclose(pcr_weights(design, target, rank=design.shape[1]),
                               np.linalg.pinv(design) @ target, rtol=1e-10, atol=1e-12)


def test_pcr_weights_reconstructs_a_target_in_the_donor_span():
    """A target that is exactly a donor combination is reproduced by the fit."""
    design, _ = _well_conditioned()
    truth = np.array([0.3, -0.7, 1.1, 0.0, 0.5, -0.2])
    target = design @ truth
    weights = pcr_weights(design, target, rank=design.shape[1])
    np.testing.assert_allclose(design @ weights, target, rtol=1e-8, atol=1e-8)


def test_pcr_weights_is_scale_equivariant_in_the_target():
    """Doubling the target doubles the weights -- the map is linear in y."""
    design, target = _well_conditioned()
    np.testing.assert_allclose(pcr_weights(design, 2.5 * target, rank=4),
                               2.5 * pcr_weights(design, target, rank=4),
                               rtol=1e-10, atol=1e-12)


def test_pcr_weights_accepts_a_column_vector_target():
    """``target`` is ravelled, so an (m, 1) column behaves like an (m,) vector."""
    design, target = _well_conditioned()
    np.testing.assert_allclose(pcr_weights(design, target.reshape(-1, 1), rank=3),
                               pcr_weights(design, target, rank=3),
                               rtol=0, atol=0)


# --- edge ------------------------------------------------------------------


@pytest.mark.parametrize("exact_zeros", [False, True])
def test_pcr_weights_bounded_when_rank_exceeds_numerical_rank(exact_zeros):
    """The degenerate case: more directions requested than the design carries.

    With ``exact_zeros`` the spare columns are identically zero, so the spare
    singular values are exactly 0 and the unguarded quotient is 0/0. Without
    it they are ~1e-16 and the unguarded quotient explodes instead. Both must
    stay finite and bounded.
    """
    design = _rank_deficient(true_rank=3, exact_zeros=exact_zeros)
    rng = np.random.default_rng(1)
    target = rng.standard_normal(design.shape[0])

    reference = pcr_weights(design, target, rank=3)
    for rank in (4, 6, 9, 12):
        weights = pcr_weights(design, target, rank=rank)
        assert np.all(np.isfinite(weights))
        assert np.max(np.abs(weights)) <= 1e3 * max(np.max(np.abs(reference)), 1.0)


def test_pcr_weights_truncates_to_the_numerical_rank():
    """Asking beyond the numerical rank adds no directions, so the fit is stable."""
    design = _rank_deficient(true_rank=3)
    rng = np.random.default_rng(2)
    target = rng.standard_normal(design.shape[0])
    at_rank = pcr_weights(design, target, rank=3)
    for rank in (5, 8, 12):
        np.testing.assert_allclose(design @ pcr_weights(design, target, rank=rank),
                                   design @ at_rank, rtol=1e-6, atol=1e-6)


def test_pcr_weights_collinear_donors_stay_bounded():
    """Two exactly duplicated donors -- the classic collinear donor pool."""
    rng = np.random.default_rng(3)
    design = rng.standard_normal((30, 4))
    design = np.column_stack([design, design[:, 0]])       # donor 4 duplicates donor 0
    target = rng.standard_normal(30)
    weights = pcr_weights(design, target, rank=5)
    assert np.all(np.isfinite(weights))
    assert np.max(np.abs(weights)) < 1e3


def test_pcr_weights_single_donor():
    """A one-column design is a scalar projection, not a degenerate case."""
    rng = np.random.default_rng(4)
    design = rng.standard_normal((20, 1))
    target = rng.standard_normal(20)
    weights = pcr_weights(design, target, rank=1)
    assert weights.shape == (1,)
    np.testing.assert_allclose(weights, np.linalg.lstsq(design, target, rcond=None)[0],
                               rtol=1e-10, atol=1e-12)


def test_pcr_weights_rank_above_matrix_dimensions_is_clamped():
    """``hsvt`` already clamps to min(m, n); asking for more is not an error."""
    design, target = _well_conditioned(m=10, n=3)
    np.testing.assert_allclose(pcr_weights(design, target, rank=99),
                               pcr_weights(design, target, rank=3),
                               rtol=0, atol=0)


def test_pcr_weights_rank_below_one_is_clamped():
    """``hsvt`` floors the rank at 1, so a zero or negative rank is not an error."""
    design, target = _well_conditioned()
    for rank in (0, -5):
        np.testing.assert_allclose(pcr_weights(design, target, rank=rank),
                                   pcr_weights(design, target, rank=1),
                                   rtol=0, atol=0)


# --- failure ---------------------------------------------------------------


def test_pcr_weights_all_zero_design_raises_translated_error():
    """No direction survives thresholding, so the fit is reported, not returned."""
    design = np.zeros((15, 4))
    target = np.random.default_rng(5).standard_normal(15)
    with pytest.raises(MlsynthEstimationError, match="rank"):
        pcr_weights(design, target, rank=2)


def test_pcr_weights_non_2d_design_raises_translated_error():
    """A 1-D design is a caller error and surfaces as an mlsynth exception."""
    target = np.random.default_rng(6).standard_normal(10)
    with pytest.raises(MlsynthEstimationError):
        pcr_weights(np.arange(10.0), target, rank=1)


def test_hsvt_non_2d_raises_translated_error():
    with pytest.raises(MlsynthEstimationError, match="2D"):
        hsvt(np.arange(10.0), rank=1)


# --- regression guard ------------------------------------------------------


def test_well_conditioned_weights_are_unchanged_by_the_rank_guard():
    """The guard must be inert whenever every retained direction carries signal.

    Pins the exact closed-form value so a future change to the thresholding
    cannot silently move results for SI / ClusterSC / RSC / SCMO.
    """
    design, target = _well_conditioned(m=50, n=8, seed=11)
    U, s, Vt = np.linalg.svd(design, full_matrices=False)
    assert s[-1] / s[0] > 1e-6, "fixture must be well conditioned"
    for rank in range(1, 9):
        expected = sum((1.0 / s[i]) * Vt[i] * (U[:, i] @ target) for i in range(rank))
        np.testing.assert_allclose(pcr_weights(design, target, rank), expected,
                                   rtol=0, atol=1e-13)
