"""Tests for the over-identified, unit-instrument Proximal Inference method (PIOID).

The ``PIOID`` method fits the over-identified proximal outcome bridge of Shi,
Li, Yu, Miao, Kuchibhotla, Hu & Tchetgen Tchetgen (2026, JASA) --
``omega = (W'Z Z'W)^{-1} W'Z Z'Y`` on the pre-period, using a *distinct set of
donor units* ``Z`` (the ``outcome_instruments``) as instruments for the donor
pool ``W`` (the ``donors``), a single outcome variable throughout. This is the
configuration the manuscript's German-reunification application uses, and it
differs from the just-identified variable-proxy ``PI`` method.

Validated value-for-value against the authors' manuscript code
(``KenLi93/proximal_sc_manuscript``, ``NC_nocov``) on ``scpi_germany``: the
average post-reunification effect is -1.7091 (thousand USD), i.e. the paper's
-1709 USD headline.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import PROXIMAL
from mlsynth.config_models import PROXIMALConfig
from mlsynth.exceptions import MlsynthConfigError

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED = "West Germany"
_W = ["Austria", "Italy", "Japan", "Netherlands", "Switzerland", "USA"]


def _germany() -> pd.DataFrame:
    df = pd.read_csv(_BASE / "scpi_germany.csv")[["country", "year", "gdp"]].dropna()
    df["treat"] = ((df["country"] == _TREATED) & (df["year"] > 1990)).astype(int)
    return df


def _pioid_cfg(df: pd.DataFrame) -> dict:
    Z = [c for c in df["country"].unique() if c not in _W + [_TREATED]]
    return {
        "df": df, "outcome": "gdp", "treat": "treat",
        "unitid": "country", "time": "year",
        "donors": _W, "outcome_instruments": Z,
        "methods": ["PIOID"], "display_graphs": False,
    }


def test_pioid_reproduces_manuscript_germany_att() -> None:
    """PIOID on scpi_germany matches the paper's PI ATT (-1709 USD) to the dollar."""
    df = _germany()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PROXIMAL(_pioid_cfg(df)).fit()
    att = res.att_by_method()["PIOID"]
    # scpi_germany GDP is in thousands of USD; -1.7091 thousand = -1709.1 USD.
    assert att == pytest.approx(-1.709137, abs=1e-3)


def test_pioid_counterfactual_uses_only_donors() -> None:
    """The PIOID counterfactual is W @ omega -- the instruments Z do not enter it."""
    df = _germany()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PROXIMAL(_pioid_cfg(df)).fit()
    fit = res.methods["PIOID"]
    assert set(fit.donor_weights.keys()) == set(_W)
    assert np.isfinite(fit.counterfactual).all()


def test_pioid_requires_outcome_instruments() -> None:
    """PIOID without outcome_instruments raises a translated config error."""
    df = _germany()
    with pytest.raises(MlsynthConfigError):
        PROXIMALConfig(**{
            "df": df, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year",
            "donors": _W, "methods": ["PIOID"], "display_graphs": False,
        })


def test_pioid_does_not_require_donorproxies() -> None:
    """PIOID needs no 'donorproxies' variable (unlike the just-identified PI)."""
    df = _germany()
    cfg = PROXIMALConfig(**_pioid_cfg(df))
    assert "PIOID" in cfg.methods
    assert not cfg.vars.get("donorproxies")
