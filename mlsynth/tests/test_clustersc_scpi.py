"""CLUSTERSC RSC / PCR inference via the generalized scpi prediction intervals.

scpi (Cattaneo, Feng, Palomba & Titiunik 2025) Table 3 assigns the ridge weight
constraint to Amjad, Kim, Shah & Shen (2018) Robust Synthetic Control. CLUSTERSC's
PCR (RSC) and RPCA fits can now route their prediction intervals through
VanillaSC's generalized ``scpi_intervals`` under a chosen constraint (ridge by
default), instead of -- or alongside -- the Shen / CFT paths.

Layered per ``agents/agents_tests.md``:

* smoke: ``compute_scpi_pi=True`` produces a finite, ordered ATT interval via
  ``.fit()`` on a synthetic factor panel;
* invariants: the reported constraint / df are scpi's; the interval is the
  standardized ``att_ci``; the simultaneous band is never tighter than pointwise;
* config: the new fields validate and default correctly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import CLUSTERSC
from mlsynth.config_models import CLUSTERSCConfig
from mlsynth.utils.clustersc_helpers.scpi_pi import (
    ScpiPIInference,
    scpi_pi_inference,
)


def _factor_panel(J=12, T_pre=18, T_post=8, r=2, tau_true=1.0, seed=0):
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, r))
    lam = rng.standard_normal((J + 1, r))
    eps = rng.standard_normal((T, J + 1)) * 0.4
    Y = (F @ lam.T + eps)
    Y[T_pre:, 0] += tau_true
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]),
             "D": int(j == 0 and t >= T_pre)}
            for j in range(J + 1) for t in range(T)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _factor_panel()


def _cfg(panel, **kw):
    return {"df": panel, "outcome": "y", "treat": "D", "unitid": "unit",
            "time": "time", "display_graphs": False, **kw}


# ----------------------------------------------------------------------
# helper-level unit test (independent of CLUSTERSC wiring)
# ----------------------------------------------------------------------
def test_scpi_pi_inference_helper_ridge():
    rng = np.random.default_rng(0)
    T, J, T0 = 30, 5, 20
    Y0 = rng.normal(size=(T, J)) + np.linspace(0, 3, T)[:, None]
    W = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    y = Y0 @ W + rng.normal(scale=0.1, size=T)
    y[T0:] += 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = scpi_pi_inference(y, Y0, T0, W, constraint="ridge",
                                sims=100, alpha=0.05, seed=1)
    assert isinstance(obj, ScpiPIInference)
    assert obj.constraint == "ridge"
    lo, hi = obj.att_pi
    assert np.isfinite(lo) and np.isfinite(hi) and hi >= lo
    assert obj.df > 0
    assert len(obj.pi_lower) == T - T0
    wp = np.asarray(obj.cf_upper) - np.asarray(obj.cf_lower)
    ws = np.asarray(obj.cf_upper_simul) - np.asarray(obj.cf_lower_simul)
    assert np.mean(ws) >= np.mean(wp) - 1e-6


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_scpi_defaults(panel):
    cfg = CLUSTERSCConfig(**_cfg(panel))
    assert cfg.compute_scpi_pi is False
    assert cfg.scpi_constraint == "ridge"


def test_config_scpi_accepts_constraints(panel):
    cfg = CLUSTERSCConfig(**_cfg(panel, compute_scpi_pi=True,
                                 scpi_constraint="ols", scpi_sims=50))
    assert cfg.compute_scpi_pi is True
    assert cfg.scpi_constraint == "ols"
    assert cfg.scpi_sims == 50


# ----------------------------------------------------------------------
# .fit() integration (the hard rule: result comes from .fit())
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def pcr_scpi(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CLUSTERSC(_cfg(panel, method="pcr", compute_scpi_pi=True,
                              scpi_constraint="ridge", scpi_sims=100,
                              random_state=0)).fit()


def test_pcr_scpi_ci_populated(pcr_scpi):
    lo, hi = pcr_scpi.att_ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi >= lo
    assert "scpi" in pcr_scpi.inference.method.lower()


def test_pcr_scpi_reports_ridge(pcr_scpi):
    ci = pcr_scpi.cluster_inference
    assert ci.scpi is not None
    assert ci.scpi.constraint == "ridge"
    assert ci.scpi.df > 0


def test_pcr_scpi_populates_canonical_band(pcr_scpi):
    ts = pcr_scpi.time_series
    assert ts.has_prediction_interval is True
    assert len(ts.counterfactual_lower) == len(ts.observed_outcome)
    post = np.isfinite(ts.counterfactual_lower)
    assert post.any() and not post.all()
    assert np.all(ts.counterfactual_upper[post] >= ts.counterfactual_lower[post])
    assert ts.prediction_interval_kind == "scpi:ridge"


def test_rpca_scpi_runs(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC(_cfg(panel, method="rpca", k_clusters=1,
                             compute_scpi_pi=True, scpi_constraint="simplex",
                             scpi_sims=60, random_state=0)).fit()
    lo, hi = res.att_ci
    assert np.isfinite(lo) and np.isfinite(hi) and hi >= lo
    assert res.cluster_inference.scpi.constraint == "simplex"


def test_simplex_pcr_scpi_runs(panel):
    # PCR with SIMPLEX weights: the coherent pairing is the simplex constraint
    # (df = #nonzero - 1). Shen CIs don't apply to simplex weights, so scpi is
    # the natural inference here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC(_cfg(panel, method="pcr", pcr_objective="SIMPLEX",
                             compute_scpi_pi=True, scpi_constraint="simplex",
                             scpi_sims=80, random_state=0)).fit()
    lo, hi = res.att_ci
    assert np.isfinite(lo) and np.isfinite(hi) and hi >= lo
    assert res.cluster_inference.scpi is not None
    assert res.cluster_inference.scpi.constraint == "simplex"
    assert "scpi" in res.inference.method.lower()
