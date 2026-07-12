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


def test_pioid_se_matches_manuscript_gmm_ci() -> None:
    """At the manuscript's Newey-West lag (default 10), PIOID's HAC SE reproduces
    the paper's GMM PI 90% CI of (-2806, -616) USD."""
    df = _germany()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = PROXIMAL(_pioid_cfg(df)).fit().methods["PIOID"]
    # SE in thousands of USD; the paper's 90% CI implies SE ~= 0.666.
    assert fit.att_se == pytest.approx(0.6655, abs=5e-3)
    z90 = 1.6448536269514722
    lo, hi = fit.att - z90 * fit.att_se, fit.att + z90 * fit.att_se
    assert lo * 1000 == pytest.approx(-2804, abs=15)   # paper -2806
    assert hi * 1000 == pytest.approx(-614, abs=15)     # paper -616


def test_pioid_hac_lag_is_configurable_and_affects_only_se() -> None:
    """Changing pioid_hac_lag moves the SE but not the point estimate."""
    df = _germany()
    cfg2 = dict(_pioid_cfg(df), pioid_hac_lag=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f10 = PROXIMAL(_pioid_cfg(df)).fit().methods["PIOID"]
        f2 = PROXIMAL(cfg2).fit().methods["PIOID"]
    assert f2.att == pytest.approx(f10.att, abs=1e-9)   # point estimate unchanged
    assert f2.att_se != pytest.approx(f10.att_se, abs=1e-3)  # SE moves with the lag


def test_pioid_counterfactual_uses_only_donors() -> None:
    """The PIOID counterfactual is W @ omega -- the instruments Z do not enter it."""
    df = _germany()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PROXIMAL(_pioid_cfg(df)).fit()
    fit = res.methods["PIOID"]
    assert set(fit.donor_weights.keys()) == set(_W)
    assert np.isfinite(fit.counterfactual).all()


def test_pioid_simplex_reproduces_manuscript_cpi_att() -> None:
    """PIOID with pioid_simplex=True (the authors' cPI) matches the paper's cPI ATT
    (-1719 USD) to the dollar, with simplex donor weights and no GMM SE."""
    df = _germany()
    cfg = dict(_pioid_cfg(df), pioid_simplex=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = PROXIMAL(cfg).fit().methods["PIOID"]
    assert fit.att == pytest.approx(-1.718900, abs=1e-3)   # -1719 USD (thousands)
    w = np.array(list(fit.donor_weights.values()))
    assert (w >= -1e-8).all() and abs(w.sum() - 1.0) < 1e-6  # on the simplex
    assert fit.att_se is None                               # constrained: no GMM SE


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
