import numpy as np
import pytest
from mlsynth.utils.inferutils import ag_conformal


def test_ag_conformal_basic():
    y_true_pre = np.array([1.0, 2.0, 3.0])
    y_pred_pre = np.array([0.9, 2.1, 2.9])
    y_pred_post = np.array([3.5, 4.0])

    lower, upper = ag_conformal(y_true_pre, y_pred_pre, y_pred_post, alpha=0.1, pad_value=-999)

    residuals = y_true_pre - y_pred_pre
    mu_hat = np.mean(residuals)
    sigma2_hat = np.var(residuals, ddof=1)
    delta = np.sqrt(2 * sigma2_hat * np.log(2 / 0.1))

    expected_lower = y_pred_post + mu_hat - delta
    expected_upper = y_pred_post + mu_hat + delta

    expected_pad = np.full(len(y_true_pre), -999)

    expected_lower_full = np.concatenate([expected_pad, expected_lower])
    expected_upper_full = np.concatenate([expected_pad, expected_upper])

    np.testing.assert_allclose(lower, expected_lower_full, rtol=1e-6)
    np.testing.assert_allclose(upper, expected_upper_full, rtol=1e-6)
