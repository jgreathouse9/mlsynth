"""Tests for the flexible SPSC bases: configurable detrend + time-varying ATT.

The SPSC estimator (Park & Tchetgen Tchetgen 2025) lets the user pick the
detrend basis (``detrend.ft``) and the ATT basis (``att.ft``). mlsynth exposes
these as ``spsc_detrend_basis`` / ``spsc_detrend_degree`` and ``spsc_att_degree``.
These tests pin: back-compatibility of the defaults, the time-varying effect
path and its per-period SE, the polynomial detrend, config validation, and a
regression guard that the linear-detrend + linear-ATT configuration reproduces
the authors' California (Proposition 99) example value-for-value.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import PROXIMAL
from mlsynth.config_models import PROXIMALConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.proximal_helpers.spsc.estimation import (
    _att_basis,
    _build_detrend_matrix,
    estimate_spsc,
)

_SMOKING = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                        "smoking_data.csv")


def _toy(seed=0, T=24, T0=14, N=6, effect=2.0):
    rng = np.random.default_rng(seed)
    f = np.linspace(0, 2, T) + 0.3 * np.sin(np.linspace(0, 5, T))
    W = np.column_stack([rng.normal(5, 1) + rng.normal(1, 0.2) * f
                         + rng.normal(0, 0.1, T) for _ in range(N)])
    y = W @ (np.ones(N) / N) + rng.normal(0, 0.05, T)
    y[T0:] += effect
    return y, W, T0


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_att_basis_constant_is_indicator():
    B = _att_basis(T=10, T0=6, att_degree=0)
    assert B.shape == (10, 1)
    assert np.array_equal(B[:, 0], np.r_[np.zeros(6), np.ones(4)])


def test_att_basis_linear_columns():
    B = _att_basis(T=10, T0=6, att_degree=1)
    assert B.shape == (10, 2)
    assert np.array_equal(B[6:, 1], np.array([1.0, 2.0, 3.0, 4.0]))  # within-post index
    assert np.all(B[:6] == 0)


def test_poly_detrend_basis_is_vandermonde():
    D = _build_detrend_matrix(T0=6, T=10, df=5, basis="poly", degree=1)
    assert D.shape == (10, 2)
    assert np.array_equal(D[:, 0], np.ones(10))
    assert np.array_equal(D[:, 1], np.arange(1, 11))


# --------------------------------------------------------------------------- #
# Engine behaviour
# --------------------------------------------------------------------------- #
def test_att_degree_zero_backcompat():
    """The att_degree=0 default reproduces the legacy call bit-for-bit."""
    y, W, T0 = _toy(seed=1)
    legacy = estimate_spsc(y, W, T0, detrend=True)
    deg0 = estimate_spsc(y, W, T0, detrend=True, att_degree=0)
    for a, b in zip(legacy, deg0):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_att_degree_one_returns_path():
    y, W, T0 = _toy(seed=2)
    out = estimate_spsc(y, W, T0, detrend=True, att_degree=1)
    effect_path, path_se = out[6], out[7]
    T1 = len(y) - T0
    assert effect_path.shape == (T1,) and path_se.shape == (T1,)
    assert np.isclose(out[2], float(np.mean(effect_path)))      # att == path mean
    assert np.all(np.isfinite(path_se))


def test_constant_path_is_flat():
    y, W, T0 = _toy(seed=3)
    out = estimate_spsc(y, W, T0, detrend=False, att_degree=0)
    assert np.allclose(out[6], out[2])                          # flat path at att


def test_poly_detrend_differs_from_bspline():
    y, W, T0 = _toy(seed=4)
    bs = estimate_spsc(y, W, T0, detrend=True, detrend_basis="bspline")[2]
    poly = estimate_spsc(y, W, T0, detrend=True, detrend_basis="poly",
                         detrend_degree=1)[2]
    assert not np.isclose(bs, poly)


def test_invalid_att_degree_raises():
    y, W, T0 = _toy()
    with pytest.raises(ValueError):
        estimate_spsc(y, W, T0, att_degree=-1)


def test_invalid_detrend_degree_raises():
    y, W, T0 = _toy()
    with pytest.raises(ValueError):
        estimate_spsc(y, W, T0, detrend_basis="poly", detrend_degree=0)


# --------------------------------------------------------------------------- #
# Config validation + end-to-end plumbing
# --------------------------------------------------------------------------- #
def _cfg(df, **kw):
    base = dict(df=df, outcome="cigsale", treat="treat", unitid="state",
                time="year", donors=[s for s in dict.fromkeys(df["state"])
                                     if s != "California"],
                methods=["SPSC"], display_graphs=False)
    base.update(kw)
    return PROXIMALConfig(**base)


def test_detrend_basis_pattern_validated():
    df = pd.read_csv(_SMOKING)
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1988)).astype(int)
    with pytest.raises((MlsynthConfigError, Exception)):
        _cfg(df, spsc_detrend_basis="loess")


def test_negative_att_degree_config_rejected():
    df = pd.read_csv(_SMOKING)
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1988)).astype(int)
    with pytest.raises((MlsynthConfigError, Exception)):
        _cfg(df, spsc_att_degree=-1)


def test_end_to_end_path_in_metadata():
    df = pd.read_csv(_SMOKING)
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1988)).astype(int)
    res = PROXIMAL(_cfg(df, spsc_att_degree=1, spsc_detrend_basis="poly",
                        spsc_detrend_degree=1, spsc_lambda=0.0)).fit()
    fit = res.spsc
    assert "ATT1" in fit.metadata["variant"]
    assert fit.metadata["effect_path"].shape == (13,)
    assert fit.metadata["effect_path_se"].shape == (13,)


# --------------------------------------------------------------------------- #
# Regression guard: reproduce the authors' California example value-for-value
# --------------------------------------------------------------------------- #
def test_california_linear_config_reproduces_reference():
    """Linear detrend + linear ATT on Prop 99 reproduces the qkrcks0218/SPSC
    California path (lambda fixed at 10**0), value-for-value."""
    df = pd.read_csv(_SMOKING)
    donors = [s for s in dict.fromkeys(df["state"]) if s != "California"]
    W = np.column_stack([df["cigsale"][df["state"] == s].to_numpy(float)
                         for s in donors])
    y = df["cigsale"][df["state"] == "California"].to_numpy(float)
    out = estimate_spsc(y, W, 18, detrend=True, ridge_lambda=0.0, att_degree=1,
                        detrend_basis="poly", detrend_degree=1)
    ref = np.array([-4.845, -7.382, -9.918, -12.455, -14.991, -17.528, -20.064,
                    -22.601, -25.138, -27.674, -30.211, -32.747, -35.284])
    ref_se = np.array([0.0020, 0.0038, 0.0056, 0.0074, 0.0092, 0.0110, 0.0127,
                       0.0145, 0.0163, 0.0181, 0.0199, 0.0217, 0.0235])
    np.testing.assert_allclose(out[6], ref, atol=2e-3)
    np.testing.assert_allclose(out[7], ref_se, atol=2e-4)
