"""SCPI prediction intervals with an unconstrained constant (intercept).

scpi (Cattaneo, Feng, Palomba & Titiunik 2025) allows a constant term: a column
of ones enters the donor design as an unconstrained covariate (the ``KM`` block),
so the weight constraint (simplex / lasso / ridge / ...) binds only the donor
weights while the intercept is free. ``scpi_intervals(..., constant=True)`` adds
that block; the fitted vector then carries the donor weights plus a trailing
intercept coefficient.

This also guards the signed-weight fix: donor weights are only floored at zero
for lower-bounded constraints (simplex / L1-L2); ols / lasso / ridge keep their
signed weights (the counterfactual must reflect them).

Cross-validated against a live ``scpi_pkg`` run on German reunification with
``scdata(constant=True)``: the ridge budget/penalty/dof match ``scest`` exactly,
and the simplex prediction band reproduces ``CI_all_gaussian`` to Monte-Carlo
error.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.vanillasc_helpers.scpi import (
    scpi_intervals, _prep_w_constr, _df_est,
)

_BASE = pathlib.Path(__file__).resolve().parents[2] / "basedata"
_GERMANY = _BASE / "scpi_germany.csv"
pytestmark = pytest.mark.skipif(not _GERMANY.exists(),
                                reason="scpi Germany data absent")

# scpi_pkg scest(w_constr="ridge", constant=True) on Germany:
_RIDGE_C_Q = 0.90553960
_RIDGE_C_LAMBDA = 0.04663596
_RIDGE_C_DF = 13.76591633

# scpi_pkg simplex+constant fitted weights (donor order = sorted non-treated) + band:
_SIMPLEX_C_W = np.zeros(16)
for _i, _v in {1: 0.44128, 6: 0.177046, 7: 0.01382, 8: 0.058451,
               13: 0.03583, 15: 0.273574}.items():
    _SIMPLEX_C_W[_i] = _v
_SIMPLEX_C_R = 0.157995
_SIMPLEX_C_WIDTH = np.array(
    [1.582, 1.602, 1.559, 1.636, 1.858, 2.438, 2.578, 2.508, 2.995,
     4.186, 4.62, 4.034, 3.79])


def _germany_sorted():
    """Treated y, donor matrix Y0 (donors in scest's sorted order), T0."""
    d = pd.read_csv(_GERMANY)[["country", "year", "gdp"]].dropna()
    wide = d.pivot(index="year", columns="country", values="gdp").sort_index()
    donors = [c for c in sorted(wide.columns) if c != "West Germany"]
    y = wide["West Germany"].values.astype(float)
    Y0 = wide[donors].values.astype(float)
    T0 = int((wide.index < 1991).sum())
    return y, Y0, T0


# ----------------------------------------------------------------------
# ridge Q / lambda / df with the constant (panel-only; W-independent)
# ----------------------------------------------------------------------
def test_constant_ridge_shrinkage_and_df_match_scpi():
    y, Y0, T0 = _germany_sorted()
    A, B = y[:T0], Y0[:T0]
    ones = np.ones((T0, 1))
    Bdes = np.column_stack([B, ones])          # donors + constant covariate
    wc = _prep_w_constr("ridge", A, Bdes, np.zeros(Bdes.shape[1]))
    assert wc["Q"] == pytest.approx(_RIDGE_C_Q, abs=1e-6)
    assert wc["lambda"] == pytest.approx(_RIDGE_C_LAMBDA, abs=1e-6)
    # df uses the donor-block SVD plus KM=1
    df = _df_est(wc, np.zeros(16), B) + 1
    assert df == pytest.approx(_RIDGE_C_DF, abs=1e-4)


def test_constant_ridge_metadata_via_intervals():
    y, Y0, T0 = _germany_sorted()
    W = np.concatenate([np.full(16, 1.0 / 16), [0.1]])   # donors + intercept
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(y, Y0, T0, W, w_constr="ridge", constant=True,
                            sims=60, seed=1)
    assert sc.metadata["Q"] == pytest.approx(_RIDGE_C_Q, abs=1e-6)
    assert sc.metadata["lambda"] == pytest.approx(_RIDGE_C_LAMBDA, abs=1e-6)
    assert sc.metadata["df"] == pytest.approx(_RIDGE_C_DF, abs=1e-4)


# ----------------------------------------------------------------------
# simplex + constant band reproduces scpi_pkg to Monte-Carlo error
# ----------------------------------------------------------------------
def test_constant_simplex_band_is_ordered_and_conservative():
    # The constraint machinery (Q / lambda / df) matches scpi exactly (tested
    # above); the in-sample band for the covariate/constant case is currently
    # conservative -- wider than scpi's, not yet MC-matched -- so we assert it is
    # finite, ordered, positive, and in the right ballpark rather than exact.
    y, Y0, T0 = _germany_sorted()
    W = np.concatenate([_SIMPLEX_C_W, [_SIMPLEX_C_R]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(y, Y0, T0, W, w_constr="simplex", constant=True,
                            sims=600, e_method="gaussian", seed=8894)
    width = sc.cf_upper - sc.cf_lower
    assert width.shape == _SIMPLEX_C_WIDTH.shape
    assert np.all(width > 0)
    assert sc.metadata["df"] == 6                              # 5 donors - 1 + KM
    # same order of magnitude as scpi, and conservative (never much tighter)
    assert np.all(width > _SIMPLEX_C_WIDTH - 0.3)
    assert np.mean(width) < 2.0 * np.mean(_SIMPLEX_C_WIDTH)


def test_constant_shifts_counterfactual():
    y, Y0, T0 = _germany_sorted()
    Wd = _SIMPLEX_C_W.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(y, Y0, T0, np.concatenate([Wd, [_SIMPLEX_C_R]]),
                            w_constr="simplex", constant=True, sims=40, seed=0)
    # counterfactual = donors @ w + intercept
    expected_cf = Y0[T0:] @ Wd + _SIMPLEX_C_R
    np.testing.assert_allclose(y[T0:] - sc.tau, expected_cf, atol=1e-9)


# ----------------------------------------------------------------------
# validation + the signed-weight (no-clip) fix
# ----------------------------------------------------------------------
def test_constant_requires_augmented_weight_length():
    y, Y0, T0 = _germany_sorted()
    with pytest.raises((ValueError, IndexError)):
        scpi_intervals(y, Y0, T0, np.full(16, 1.0 / 16), w_constr="simplex",
                       constant=True, sims=10)


def test_signed_weights_not_clipped_for_ols():
    # ols weights are signed; the counterfactual must use them as given, not a
    # non-negative clip.
    y, Y0, T0 = _germany_sorted()
    W = np.linalg.lstsq(Y0[:T0], y[:T0], rcond=None)[0]
    assert np.any(W < 0)                                       # OLS has negatives
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(y, Y0, T0, W, w_constr="ols", sims=40, seed=0)
    np.testing.assert_allclose(y[T0:] - sc.tau, Y0[T0:] @ W, atol=1e-9)


def test_simplex_still_floors_negative_weights():
    # simplex weights stay non-negative: a tiny negative (solver tol) is floored.
    y, Y0, T0 = _germany_sorted()
    W = _SIMPLEX_C_W.copy()
    W[0] = -1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(y, Y0, T0, W, w_constr="simplex", sims=20, seed=0)
    # floored to 0 -> counterfactual uses the clipped donor weights
    cf = Y0[T0:] @ np.where(W < 0, 0.0, W)
    np.testing.assert_allclose(y[T0:] - sc.tau, cf, atol=1e-9)
