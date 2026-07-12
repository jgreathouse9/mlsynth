"""TSSC prediction intervals via the generalized scpi engine, per variant.

TSSC (Li & Shankar 2023) fits four SC-class variants distinguished by their
weight constraints, each of which maps onto scpi's (Cattaneo, Feng, Palomba &
Titiunik 2025) weight-constraint family:

    SC    : w >= 0, sum w = 1            -> scpi simplex
    MSCa  : intercept + w >= 0, sum w = 1 -> scpi simplex + constant
    MSCb  : w >= 0  (no adding-up)        -> scpi ols
    MSCc  : intercept + w >= 0            -> scpi ols + constant

The sum-constrained variants (SC / MSCa) map exactly; the no-adding-up variants
(MSCb / MSCc) carry bare non-negativity, which scpi's family does not include, so
they map to scpi's ols compatible set -- the nonnegativity is not re-imposed on
the PI set, giving a (slightly conservative) band. With ``compute_scpi_pi`` each
variant fit gains a ``scpi`` band alongside its Li (2020) subsampling ATT CI.

Layered per ``agents/agents_tests.md``: config validation, a per-variant
constraint-mapping invariant, band coherence, and the recommended-variant
surfacing; the subsampling CI is unaffected when scpi is off.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsynth import TSSC
from mlsynth.config_models import TSSCConfig

from test_tssc import _make_panel

_EXPECT_CONSTRAINT = {"SC": "simplex", "MSCa": "simplex",
                      "MSCb": "ols", "MSCc": "ols"}


def _cfg(**kw):
    base = dict(df=_make_panel(treated_effect=3.0, seed=3), outcome="y",
                unitid="unitid", time="time", treat="treat", display_graphs=False)
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_scpi_defaults():
    cfg = TSSCConfig(**_cfg())
    assert cfg.compute_scpi_pi is False


def test_config_scpi_accepts():
    cfg = TSSCConfig(**_cfg(compute_scpi_pi=True, scpi_sims=50))
    assert cfg.compute_scpi_pi is True
    assert cfg.scpi_sims == 50


# ----------------------------------------------------------------------
# .fit() integration
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def scpi_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TSSC(dict(df=_make_panel(treated_effect=3.0, seed=3), outcome="y",
                         unitid="unitid", time="time", treat="treat",
                         display_graphs=False, compute_scpi_pi=True,
                         scpi_sims=60)).fit()


@pytest.mark.parametrize("variant", ["SC", "MSCa", "MSCb", "MSCc"])
def test_variant_maps_to_expected_scpi_constraint(scpi_fit, variant):
    fit = scpi_fit.variants[variant]
    assert fit.scpi is not None
    assert fit.scpi.constraint == _EXPECT_CONSTRAINT[variant]


@pytest.mark.parametrize("variant", ["SC", "MSCa", "MSCb", "MSCc"])
def test_variant_scpi_bands_are_coherent(scpi_fit, variant):
    sc = scpi_fit.variants[variant].scpi
    lo, hi = sc.att_pi
    assert np.isfinite(lo) and np.isfinite(hi) and hi >= lo
    assert np.all(np.asarray(sc.cf_upper) >= np.asarray(sc.cf_lower) - 1e-8)
    ws = np.asarray(sc.cf_upper_simul) - np.asarray(sc.cf_lower_simul)
    wp = np.asarray(sc.cf_upper) - np.asarray(sc.cf_lower)
    assert np.mean(ws) >= np.mean(wp) - 1e-6


def test_recommended_variant_scpi_surfaced(scpi_fit):
    rec = scpi_fit.selection.recommended
    assert scpi_fit.variants[rec].scpi is not None
    # the recommended variant's scpi band is surfaced at top level
    assert scpi_fit.scpi is not None
    assert scpi_fit.scpi.constraint == _EXPECT_CONSTRAINT[rec]


def test_subsampling_ci_intact_without_scpi():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TSSC(_cfg()).fit()
    for v in res.variants.values():
        assert v.scpi is None
        lo, hi = v.att_ci
        assert np.isfinite(lo) and np.isfinite(hi)
