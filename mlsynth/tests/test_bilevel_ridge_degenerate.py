"""Ridge augmentation when the centred donor matrix carries no information.

Augmented SCM (Ben-Michael, Feller & Rothstein 2021) centres the outcomes each
period by the donor mean and fits a ridge correction on what is left. With a
single donor the donor mean per period *is* that donor, so the centred matrix is
identically zero: there is no cross-donor variation for the correction to act
on, and the augmentation is mathematically zero whatever penalty is chosen.

Before this, that case crashed. ``generate_lambdas`` takes the squared largest
singular value of the centred matrix as ``lambda_max``, which is zero, so every
penalty on the grid is zero; the ridge path then divides zero by zero and the
whole cross-validation curve is NaN; ``best_lambda`` compares against a NaN
threshold, selects nothing, and calls ``.max()`` on an empty array. The user saw
``ValueError: zero-size array to reduction operation maximum which has no
identity`` from inside numpy, with nothing naming the cause.

The claim tested here is that a rank-zero matching matrix is a degeneracy with a
known answer, not an error: the augmentation contributes nothing, so the
result is the base simplex fit, reported as such.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.engine import BilevelSCM
from mlsynth.utils.bilevel.ridge_augment import (
    best_lambda,
    build_matching,
    generate_lambdas,
    ridge_augment_weights,
)


def _panel(n_pre=15, n_donors=1, seed=0):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(100.0, 5.0, size=(n_pre, n_donors))
    y = Y0[:, 0] + rng.normal(0.0, 1.0, size=n_pre)
    return y, Y0


# ---------------------------------------------------------------------------
# The premise
# ---------------------------------------------------------------------------

class TestPremise:
    def test_single_donor_centres_to_exactly_zero(self):
        y, Y0 = _panel()
        B, _ = build_matching(y, Y0)
        assert np.abs(B).max() == 0.0

    def test_zero_matrix_gives_a_zero_lambda_grid(self):
        grid = generate_lambdas(np.zeros((15, 1)))
        assert np.all(grid == 0.0)

    def test_two_donors_are_not_degenerate(self):
        y, Y0 = _panel(n_donors=2)
        B, _ = build_matching(y, Y0)
        assert np.abs(B).max() > 0.0


# ---------------------------------------------------------------------------
# The fix: a known answer, not a crash
# ---------------------------------------------------------------------------

class TestSingleDonorAugmentation:
    def test_ridge_augment_returns_the_base_fit(self):
        y, Y0 = _panel()
        res = ridge_augment_weights(y, Y0)
        np.testing.assert_allclose(res.W, res.W_base, atol=1e-12)
        np.testing.assert_allclose(res.W_ridge, 0.0, atol=1e-12)

    def test_lambda_is_zero_and_finite(self):
        y, Y0 = _panel()
        res = ridge_augment_weights(y, Y0)
        assert np.isfinite(res.lambda_) and res.lambda_ == 0.0

    def test_weights_are_finite_and_sum_to_one(self):
        y, Y0 = _panel()
        res = ridge_augment_weights(y, Y0)
        assert np.all(np.isfinite(res.W))
        assert float(np.sum(res.W)) == pytest.approx(1.0, abs=1e-8)

    def test_through_the_public_engine(self):
        y, Y0 = _panel()
        A, B = y - y.mean(), Y0 - Y0.mean(axis=0)
        res = BilevelSCM(augment="ridge").fit(A, B)
        assert np.all(np.isfinite(np.asarray(res.W)))
        assert np.isfinite(res.pre_rmspe)

    @pytest.mark.parametrize("n_donors", [1, 2, 3, 5])
    def test_donor_counts_around_the_degeneracy(self, n_donors):
        # The guard must bite at J=1 and nowhere else.
        y, Y0 = _panel(n_donors=n_donors, seed=n_donors)
        res = ridge_augment_weights(y, Y0)
        assert np.all(np.isfinite(res.W))
        if n_donors == 1:
            assert res.lambda_ == 0.0
        else:
            assert res.lambda_ > 0.0

    def test_duplicated_donor_is_also_degenerate(self):
        # Two identical donors centre to zero as surely as one does, so the
        # guard has to key on the matrix and not on the column count.
        rng = np.random.default_rng(3)
        col = rng.normal(100.0, 5.0, size=(15, 1))
        Y0 = np.hstack([col, col])
        y = col[:, 0] + rng.normal(0.0, 1.0, size=15)
        res = ridge_augment_weights(y, Y0)
        assert np.all(np.isfinite(res.W))
        assert res.lambda_ == 0.0


# ---------------------------------------------------------------------------
# best_lambda's own contract
# ---------------------------------------------------------------------------

class TestBestLambda:
    def test_selects_among_finite_entries(self):
        lambdas = np.array([10.0, 5.0, 1.0])
        mean = np.array([np.nan, 2.0, 3.0])
        se = np.array([0.0, 0.5, 0.5])
        assert best_lambda(lambdas, mean, se, min_1se=False) == 5.0

    def test_all_non_finite_raises_a_translated_error(self):
        from mlsynth.exceptions import MlsynthEstimationError
        lambdas = np.array([1.0, 2.0])
        nan = np.array([np.nan, np.nan])
        with pytest.raises(MlsynthEstimationError, match="cross-validation"):
            best_lambda(lambdas, nan, nan)

    def test_ordinary_curve_is_unchanged(self):
        # The 1-SE rule: the largest lambda within one standard error of the
        # minimum. This must not move.
        lambdas = np.array([100.0, 10.0, 1.0])
        mean = np.array([1.2, 1.0, 1.15])
        se = np.array([0.3, 0.3, 0.3])
        assert best_lambda(lambdas, mean, se, min_1se=True) == 100.0
        assert best_lambda(lambdas, mean, se, min_1se=False) == 10.0
