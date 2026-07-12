"""Generalized ``scpi_intervals`` weight-constraint family.

Cattaneo, Feng, Palomba & Titiunik (2025, JSS ``scpi``), Table 2. The
single-unit prediction-interval engine now supports the full constraint family
-- ols / simplex / lasso / ridge / L1-L2 -- not just the simplex. The in-sample
QCQP compatible set and the effective degrees of freedom follow scpi's
``local_geom`` / ``df_EST`` per constraint; the out-of-sample component is
constraint-independent. The ridge constraint is scpi's Table-3 inference setting
for Amjad, Kim, Shah & Shen (2018) Robust Synthetic Control, which CLUSTERSC's
RSC / PCR path uses.

Layered per ``agents/agents_tests.md``:

* Layer 1: the constraint spec (``_prep_w_constr``), localisation
  (``_local_geom``) and degrees of freedom (``_df_est``) reproduce scpi.
* Layer 2: each constraint yields finite, ordered bands; simplex is bit-identical
  to the pre-generalization output (regression guard).
* Layer 4: invalid constraints raise the translated error.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.datautils import dataprep
from mlsynth.estimators.vanillasc import VanillaSC
from mlsynth.utils.vanillasc_helpers.scpi import (
    scpi_intervals,
    _prep_w_constr,
    _local_geom,
    _df_est,
)

_BASE = pathlib.Path(__file__).resolve().parents[2] / "basedata"
_GERMANY = _BASE / "scpi_germany.csv"

pytestmark = pytest.mark.skipif(not _GERMANY.exists(),
                                reason="scpi Germany data absent")

# --- pre-generalization simplex band (seed 8894, sims 400, gaussian, levels) ---
_SIMPLEX_CF_LOWER = np.array(
    [20.3177, 21.1043, 21.4297, 22.3441, 23.064, 23.6876, 24.3602, 25.4945,
     26.3521, 26.8267, 27.1345, 28.5057, 29.5494])
_SIMPLEX_CF_UPPER = np.array(
    [22.2359, 22.9433, 23.5098, 24.5582, 25.7911, 27.0189, 27.7183, 28.9706,
     30.2508, 32.0892, 33.3491, 34.0511, 34.6439])
_SIMPLEX_DF = 5
_SIMPLEX_RHO = 0.07164          # scpi's default type-2 regularisation

# --- scpi_pkg ridge reference (scest w_constr={"name":"ridge"}, Germany) ---
_RIDGE_Q = 0.66876212
_RIDGE_LAMBDA = 0.09607606
_RIDGE_DF = 11.46267270


def _germany():
    d = pd.read_csv(_GERMANY)[["country", "year", "gdp"]].dropna()
    d["status"] = ((d.country == "West Germany") & (d.year >= 1991)).astype(int)
    prep = dataprep(d, "country", "year", "gdp", "status")
    y = prep["y"].ravel()
    Y0 = prep["donor_matrix"]
    pre = prep["pre_periods"]
    res = VanillaSC({"df": d, "outcome": "gdp", "treat": "status",
                     "unitid": "country", "time": "year",
                     "display_graphs": False}).fit()
    W = np.asarray([res.weights.donor_weights.get(c, 0.0)
                    for c in prep["donor_names"]], float)
    return y, Y0, pre, W


@pytest.fixture(scope="module")
def germany():
    return _germany()


def _run(germany, **kw):
    y, Y0, pre, W = germany
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scpi_intervals(y, Y0, pre, W, sims=400, u_alpha=0.05,
                              e_alpha=0.05, e_method="gaussian",
                              cointegrated=False, seed=8894, **kw)


# ----------------------------------------------------------------------
# Layer 2: simplex regression guard (bit-identical to before)
# ----------------------------------------------------------------------
def test_default_w_constr_is_simplex(germany):
    sc = _run(germany)
    assert sc.metadata["w_constr"] == "simplex"


def test_simplex_bit_identical_regression_guard(germany):
    # both the default and the explicit "simplex" reproduce the snapshot exactly
    for wc in (None, "simplex"):
        sc = _run(germany) if wc is None else _run(germany, w_constr=wc)
        np.testing.assert_allclose(sc.cf_lower, _SIMPLEX_CF_LOWER, atol=1e-3)
        np.testing.assert_allclose(sc.cf_upper, _SIMPLEX_CF_UPPER, atol=1e-3)
        assert sc.metadata["df"] == _SIMPLEX_DF
        assert sc.metadata["rho"] == pytest.approx(_SIMPLEX_RHO, abs=1e-5)


# ----------------------------------------------------------------------
# Layer 1: ridge Q / lambda / df reproduce scpi (panel-only, W-independent)
# ----------------------------------------------------------------------
def test_ridge_Q_lambda_reproduce_scpi(germany):
    y, Y0, pre, W = germany
    A = y[:pre]
    B = Y0[:pre]
    wc = _prep_w_constr("ridge", A, B, W)
    assert wc["name"] == "ridge"
    assert wc["p"] == "L2" and wc["dir"] == "<="
    assert wc["Q"] == pytest.approx(_RIDGE_Q, abs=1e-6)
    assert wc["lambda"] == pytest.approx(_RIDGE_LAMBDA, abs=1e-6)


def test_ridge_df_reproduces_scpi(germany):
    y, Y0, pre, W = germany
    A = y[:pre]
    B = Y0[:pre]
    wc = _prep_w_constr("ridge", A, B, W)
    df = _df_est(wc, W, B)
    assert df == pytest.approx(_RIDGE_DF, abs=1e-5)


def test_ridge_metadata_carries_constraint(germany):
    sc = _run(germany, w_constr="ridge")
    assert sc.metadata["w_constr"] == "ridge"
    assert sc.metadata["Q"] == pytest.approx(_RIDGE_Q, abs=1e-6)
    assert sc.metadata["lambda"] == pytest.approx(_RIDGE_LAMBDA, abs=1e-6)
    assert sc.metadata["df"] == pytest.approx(_RIDGE_DF, abs=1e-5)


# ----------------------------------------------------------------------
# Layer 1: df_EST per constraint (ols=J, lasso=#nonzero, simplex=#nonzero-1)
# ----------------------------------------------------------------------
def test_df_rules_per_constraint(germany):
    y, Y0, pre, W = germany
    A = y[:pre]
    B = Y0[:pre]
    J = B.shape[1]
    nz = int(np.sum(np.abs(W) >= 1e-6))
    assert _df_est(_prep_w_constr("ols", A, B, W), W, B) == pytest.approx(J)
    assert _df_est(_prep_w_constr("simplex", A, B, W), W, B) == pytest.approx(nz - 1)
    assert _df_est(_prep_w_constr("lasso", A, B, W), W, B) == pytest.approx(nz)


# ----------------------------------------------------------------------
# Layer 1: localisation bounds (ridge/ols use all donors; no lb)
# ----------------------------------------------------------------------
def test_local_geom_active_set(germany):
    y, Y0, pre, W = germany
    A = y[:pre]
    B = Y0[:pre]
    rho = 0.03
    lg_r = _local_geom(_prep_w_constr("ridge", A, B, W), W, rho, B)
    assert lg_r.idxw.all()             # ridge keeps every donor active
    assert not lg_r.use_lb             # no lower bound
    lg_s = _local_geom(_prep_w_constr("simplex", A, B, W), W, rho, B)
    assert lg_s.use_lb                 # simplex has the localised lb
    assert lg_s.has_sum


# ----------------------------------------------------------------------
# Layer 2: constraint-family smoke (finite, ordered, sane df)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["ols", "simplex", "lasso", "ridge", "L1-L2"])
def test_constraint_family_smoke(germany, name):
    sc = _run(germany, w_constr=name)
    assert np.all(np.isfinite(sc.cf_lower))
    assert np.all(np.isfinite(sc.cf_upper))
    assert np.all(sc.cf_upper >= sc.cf_lower - 1e-8)
    assert np.all(sc.upper >= sc.lower - 1e-8)
    assert sc.metadata["att_upper"] >= sc.metadata["att_lower"] - 1e-8
    assert 0 < sc.metadata["df"] < pre_len(germany)


def pre_len(germany):
    return germany[2]


# ----------------------------------------------------------------------
# Layer 2: simultaneous (joint-coverage) bands
# ----------------------------------------------------------------------
def test_simultaneous_present_and_ordered(germany):
    sc = _run(germany)
    for arr in (sc.cf_lower_simul, sc.cf_upper_simul,
                sc.lower_simul, sc.upper_simul):
        assert np.all(np.isfinite(arr))
    assert np.all(sc.cf_upper_simul >= sc.cf_lower_simul - 1e-8)
    assert "eps_joint" in sc.metadata


def test_simultaneous_wider_than_pointwise(germany):
    sc = _run(germany)
    w_point = sc.cf_upper - sc.cf_lower
    w_simul = sc.cf_upper_simul - sc.cf_lower_simul
    # joint coverage over the whole horizon is never tighter than pointwise
    assert np.all(w_simul >= w_point - 1e-6)
    assert np.mean(w_simul) > np.mean(w_point)      # strictly wider on average


@pytest.mark.parametrize("name", ["ols", "ridge", "lasso"])
def test_simultaneous_wider_across_constraints(germany, name):
    sc = _run(germany, w_constr=name)
    w_point = sc.cf_upper - sc.cf_lower
    w_simul = sc.cf_upper_simul - sc.cf_lower_simul
    assert np.mean(w_simul) >= np.mean(w_point) - 1e-6


def test_ridge_under_cointegration_runs(germany):
    # cointegration drops the first pre-period from the Q/Sigma design; the
    # ridge shrinkage rule-of-thumb must align the treated pre-outcome to it.
    y, Y0, pre, W = germany
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(y, Y0, pre, W, w_constr="ridge", sims=200,
                            u_alpha=0.05, e_alpha=0.05, e_method="gaussian",
                            cointegrated=True, seed=8894)
    assert np.all(np.isfinite(sc.cf_lower)) and np.all(np.isfinite(sc.cf_upper))
    assert np.all(sc.cf_upper >= sc.cf_lower - 1e-8)


# ----------------------------------------------------------------------
# Layer 4: invalid constraints raise
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["bogus", {"name": "nope"}, {"p": "L3"}])
def test_invalid_w_constr_raises(germany, bad):
    with pytest.raises((ValueError, KeyError)):
        _run(germany, w_constr=bad)
