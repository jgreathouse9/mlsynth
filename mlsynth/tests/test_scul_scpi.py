"""SCUL prediction intervals via the generalized scpi lasso engine.

scpi (Cattaneo, Feng, Palomba & Titiunik 2025) Table 3 pairs the lasso weight
constraint with Chernozhukov, Wuthrich & Zhu (2021); SCUL (Hollingsworth & Wing
2022) *is* a lasso synthetic control -- signed coefficients on standardized
donors plus an intercept. SCUL can now route its prediction intervals through
VanillaSC's generalized ``scpi_intervals`` with the lasso constraint and a
constant (the intercept), in addition to its placebo p-value.

Layered per ``agents/agents_tests.md``:

* smoke: ``compute_scpi_pi=True`` yields a finite, ordered ATT interval via
  ``.fit()`` on the California Prop 99 panel;
* invariant: the reported constraint is lasso and the degrees of freedom equal
  ``#nonzero + 1`` (lasso df plus the intercept ``KM``);
* config: the new fields validate and default correctly.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SCUL
from mlsynth.config_models import SCULConfig

_PANEL = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                      "california_panel.csv")


def _cfg(**kw):
    df = pd.read_csv(_PANEL)
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    base = dict(df=df, outcome="cigsale", treat="treat", unitid="state",
                time="year", display_graphs=False, inference=False)
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_scpi_defaults():
    cfg = SCULConfig(**_cfg())
    assert cfg.compute_scpi_pi is False


def test_config_scpi_accepts():
    cfg = SCULConfig(**_cfg(compute_scpi_pi=True, scpi_sims=50))
    assert cfg.compute_scpi_pi is True
    assert cfg.scpi_sims == 50


# ----------------------------------------------------------------------
# .fit() integration (result comes from .fit())
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def scpi_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SCUL(_cfg(compute_scpi_pi=True, scpi_sims=120)).fit()


def test_scul_scpi_ci_populated(scpi_fit):
    lo, hi = scpi_fit.att_ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi >= lo
    assert "scpi" in scpi_fit.inference.method.lower()


def test_scul_scpi_reports_lasso_and_constant(scpi_fit):
    sc = scpi_fit.fit.scpi
    assert sc is not None
    assert sc.constraint == "lasso"
    # lasso df = #nonzero donor weights + KM (the intercept)
    nz = int(np.sum(np.abs(scpi_fit.fit.weights) > 1e-10))
    assert sc.df == pytest.approx(nz + 1, abs=1e-9)


def test_scul_scpi_bands_ordered(scpi_fit):
    sc = scpi_fit.fit.scpi
    assert np.all(np.asarray(sc.cf_upper) >= np.asarray(sc.cf_lower) - 1e-8)
    # simultaneous band never tighter than pointwise
    wp = np.asarray(sc.cf_upper) - np.asarray(sc.cf_lower)
    ws = np.asarray(sc.cf_upper_simul) - np.asarray(sc.cf_lower_simul)
    assert np.mean(ws) >= np.mean(wp) - 1e-6


def test_placebo_still_available_without_scpi():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SCUL(_cfg(inference=True)).fit()
    assert res.inference is not None
    assert res.inference.p_value is not None
