"""SparseSC prediction intervals via the generalized scpi simplex engine.

SparseSC (Vives-i-Bastida) fits simplex donor weights (``w >= 0``, ``sum w = 1``)
-- the L1 penalty selects *predictors* (the V-weights), not donors. Its synthetic
control is therefore an Abadie convex combination, so scpi's (Cattaneo, Feng,
Palomba & Titiunik 2025) simplex weight-constraint prediction intervals apply
directly. SparseSC can now route model-based prediction intervals through
VanillaSC's generalized ``scpi_intervals`` under the simplex constraint, in
addition to its placebo / conformal inference.

Layered per ``agents/agents_tests.md``:

* config: the new fields validate and default off;
* smoke: ``compute_scpi_pi=True`` yields a finite, ordered ATT interval;
* invariant: the reported constraint is simplex and df == ``#nonzero - 1``
  (simplex degrees of freedom, no covariate/constant block);
* the placebo / conformal path still works without scpi.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SparseSC
from mlsynth.config_models import SparseSCConfig

from test_sparse_sc import _factor_panel, COVS


def _cfg(**kw):
    df = _factor_panel()
    base = dict(df=df, outcome="y", treat="tr", unitid="unit", time="year",
                covariates=COVS, display_graphs=False,
                inference_method="none", run_inference=False)
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_scpi_defaults():
    cfg = SparseSCConfig(**_cfg())
    assert cfg.compute_scpi_pi is False


def test_config_scpi_accepts():
    cfg = SparseSCConfig(**_cfg(compute_scpi_pi=True, scpi_sims=50))
    assert cfg.compute_scpi_pi is True
    assert cfg.scpi_sims == 50


# ----------------------------------------------------------------------
# .fit() integration
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def scpi_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = dict(df=_factor_panel(), outcome="y", treat="tr", unitid="unit",
                   time="year", covariates=COVS, display_graphs=False,
                   inference_method="none", run_inference=False,
                   compute_scpi_pi=True, scpi_sims=120)
        return SparseSC(cfg).fit()


def test_sparsesc_scpi_ci_populated(scpi_fit):
    lo, hi = scpi_fit.att_ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi >= lo
    assert "scpi" in scpi_fit.inference.method.lower()


def test_sparsesc_scpi_reports_simplex(scpi_fit):
    sc = scpi_fit.scpi
    assert sc is not None
    assert sc.constraint == "simplex"
    # simplex df = #nonzero donor weights - 1 (no covariate/constant block)
    nz = int(np.sum(np.abs(np.asarray(scpi_fit.design.w)) > 1e-6))
    assert sc.df == pytest.approx(nz - 1, abs=1e-9)


def test_sparsesc_scpi_bands_ordered(scpi_fit):
    sc = scpi_fit.scpi
    assert np.all(np.asarray(sc.cf_upper) >= np.asarray(sc.cf_lower) - 1e-8)
    wp = np.asarray(sc.cf_upper) - np.asarray(sc.cf_lower)
    ws = np.asarray(sc.cf_upper_simul) - np.asarray(sc.cf_lower_simul)
    assert np.mean(ws) >= np.mean(wp) - 1e-6


def test_conformal_still_available_without_scpi():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SparseSC(_cfg(run_inference=True, inference_method="conformal")).fit()
    assert res.inference is not None
    assert res.scpi is None
