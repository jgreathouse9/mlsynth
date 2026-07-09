"""Cointegrated SCPI prediction intervals for the single-unit synthetic control.

VanillaSC's ``inference="scpi"`` now supports scpi's ``cointegrated_data=True``
(Cattaneo, Feng & Titiunik 2021): when the outcome and donors are cointegrated
(I(1) levels), the in-sample ``E[u]`` and out-of-sample ``E[e]`` uncertainty
models are fit on first differences of the donor design, dropping the first
pre-period. The point counterfactual is unchanged; only the prediction bands
move.

Differential check against a live ``scpi_pkg`` run on German reunification
(``scdata(..., cointegrated_data=True)`` then ``scpi(..., e_method="gaussian")``,
seed 8894, sims 2000), the setup of the Mendez ``python_scpi`` tutorial. The
reference synthetic-prediction-band widths (``CI_all_gaussian``) are recorded
below; mlsynth reproduces them to Monte-Carlo error, and its non-cointegrated
default reproduces the levels band.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.config_models import VanillaSCConfig

_BASE = pathlib.Path(__file__).resolve().parents[2] / "basedata"
_GERMANY = _BASE / "scpi_germany.csv"

# scpi_pkg CI_all_gaussian band widths, seed 8894, sims 2000, e_method gaussian.
_SCPI_COINT_WIDTH = np.array(
    [2.304, 1.070, 1.167, 1.356, 2.256, 2.100, 2.385, 2.745, 2.751,
     4.888, 6.488, 4.007, 4.071])
_SCPI_LEVELS_WIDTH = np.array(
    [1.999, 1.931, 2.214, 2.375, 2.933, 3.531, 3.537, 3.660, 4.102,
     5.419, 6.428, 5.695, 5.203])

pytestmark = pytest.mark.skipif(not _GERMANY.exists(),
                                reason="scpi Germany data absent")


def _panel():
    d = pd.read_csv(_GERMANY)[["country", "year", "gdp"]].dropna()
    d["status"] = ((d.country == "West Germany") & (d.year >= 1991)).astype(int)
    return d


def _fit(cointegrated, sims=800):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({"df": _panel(), "outcome": "gdp", "treat": "status",
                          "unitid": "country", "time": "year",
                          "display_graphs": False, "inference": "scpi",
                          "scpi_sims": sims, "seed": 8894,
                          "scpi_e_method": "gaussian",
                          "scpi_cointegrated": cointegrated}).fit()


def _band_widths(res):
    det = res.inference.details
    lo = np.asarray(det["counterfactual_lower"], float)
    hi = np.asarray(det["counterfactual_upper"], float)
    return hi - lo


@pytest.fixture(scope="module")
def coint():
    return _fit(True)


@pytest.fixture(scope="module")
def levels():
    return _fit(False)


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_accepts_scpi_cointegrated():
    cfg = VanillaSCConfig(df=_panel(), outcome="gdp", treat="status",
                          unitid="country", time="year", inference="scpi",
                          scpi_cointegrated=True, display_graphs=False)
    assert cfg.scpi_cointegrated is True


def test_default_is_non_cointegrated():
    cfg = VanillaSCConfig(df=_panel(), outcome="gdp", treat="status",
                          unitid="country", time="year", display_graphs=False)
    assert cfg.scpi_cointegrated is False


# ----------------------------------------------------------------------
# cointegration moves the bands, not the point estimate
# ----------------------------------------------------------------------
def test_point_estimate_unchanged_by_cointegration(coint, levels):
    # same donors/weights and same counterfactual: only the intervals differ
    assert coint.weights.donor_weights.keys() == levels.weights.donor_weights.keys()
    for k in coint.weights.donor_weights:
        assert coint.weights.donor_weights[k] == pytest.approx(
            levels.weights.donor_weights[k], abs=1e-9)
    cf_c = np.asarray(coint.inference.details["counterfactual_lower"]) \
        - np.asarray(coint.inference.details["in_sample_lower"]) \
        - np.asarray(coint.inference.details["out_of_sample_lower"])
    cf_l = np.asarray(levels.inference.details["counterfactual_lower"]) \
        - np.asarray(levels.inference.details["in_sample_lower"]) \
        - np.asarray(levels.inference.details["out_of_sample_lower"])
    np.testing.assert_allclose(cf_c, cf_l, atol=1e-9)


def test_cointegration_changes_the_band(coint, levels):
    wc = _band_widths(coint)
    wl = _band_widths(levels)
    # the bands genuinely differ (early years narrow markedly under cointegration)
    assert np.max(np.abs(wc - wl)) > 0.5
    assert wc[1] < wl[1] - 0.3            # 1992: coint ~1.07 vs levels ~1.93


# ----------------------------------------------------------------------
# differential cross-validation vs scpi_pkg
# ----------------------------------------------------------------------
def test_cointegrated_matches_scpi_reference(coint):
    wc = _band_widths(coint)
    # reproduces scpi's cointegrated band to MC error, and matches it far better
    # than it matches the levels band
    assert np.max(np.abs(wc - _SCPI_COINT_WIDTH)) < 0.6
    assert (np.mean(np.abs(wc - _SCPI_COINT_WIDTH))
            < np.mean(np.abs(wc - _SCPI_LEVELS_WIDTH)))


def test_levels_default_matches_scpi_reference(levels):
    wl = _band_widths(levels)
    assert np.max(np.abs(wl - _SCPI_LEVELS_WIDTH)) < 0.6
    assert (np.mean(np.abs(wl - _SCPI_LEVELS_WIDTH))
            < np.mean(np.abs(wl - _SCPI_COINT_WIDTH)))
