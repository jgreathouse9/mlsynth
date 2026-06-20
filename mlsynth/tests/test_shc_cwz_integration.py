"""Config + end-to-end tests for selecting exact CWZ inference in SHC.

Pins the new ``SHCConfig`` knobs (``inference_method``, ``permutation_scheme``,
``num_permutations``) and that ``SHC.fit()`` routes through the exact
permutation test when asked, while the default stays the CYY bootstrap (backward
compatible).
"""

import numpy as np
import pytest

from mlsynth import SHC
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.shc_helpers import simulate_shc_panel


def _panel(seed=0):
    df, _ = simulate_shc_panel(
        m=10, h=2, n=8, P=10, sigma=0.1, w_f=(1, 0), regular=True, seed=seed
    )
    return df


def _base(df, **extra):
    cfg = {
        "df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
        "time": "time", "m": 10, "display_graphs": False,
    }
    cfg.update(extra)
    return cfg


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------
def test_default_inference_is_bootstrap():
    res = SHC(_base(_panel())).fit()
    assert res.inference.method == "conformal_permutation"


def test_exact_moving_block_runs_and_labels_method():
    res = SHC(_base(_panel(), inference_method="exact",
                    permutation_scheme="moving_block")).fit()
    assert "exact" in res.inference.method
    assert "moving_block" in res.inference.method
    assert 0.0 <= res.inference.p_value <= 1.0
    assert np.isfinite(res.inference_detail.test_statistic)


def test_exact_iid_runs():
    res = SHC(_base(_panel(), inference_method="exact",
                    permutation_scheme="iid", num_permutations=500)).fit()
    assert "iid" in res.inference.method
    assert 0.0 <= res.inference.p_value <= 1.0


def test_bad_inference_method_raises():
    with pytest.raises(MlsynthConfigError):
        SHC(_base(_panel(), inference_method="nope"))


def test_bad_permutation_scheme_raises():
    with pytest.raises(MlsynthConfigError):
        SHC(_base(_panel(), permutation_scheme="nope"))


def test_bad_num_permutations_raises():
    with pytest.raises(MlsynthConfigError):
        SHC(_base(_panel(), inference_method="exact",
                  permutation_scheme="iid", num_permutations=1))


def test_extra_field_still_forbidden():
    # extra="forbid" on the config must reject unknown kwargs.
    with pytest.raises(Exception):
        SHC(_base(_panel(), not_a_real_field=3))


def test_moving_block_pvalue_is_multiple_of_one_over_T():
    # n + T0 cyclic shifts -> p-value lands on a 1/T grid.
    df = _panel()
    res = SHC(_base(df, inference_method="exact",
                    permutation_scheme="moving_block")).fit()
    null = res.inference_detail.null_distribution
    T = null.shape[0]
    grid_val = round(res.inference.p_value * T)
    assert res.inference.p_value == pytest.approx(grid_val / T)
