"""Tests for the Partially Pooled SCM estimator (augsynth::multisynth port)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.config_models import PPSCMConfig
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.ppscm_helpers.engine import fit_feff, run_multisynth
from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs
from mlsynth.utils.ppscm_helpers.structures import (
    PPSCMEventStudy, PPSCMInputs, PPSCMResults,
)


def _staggered_panel(*, seed: int = 0, adoption_offsets=(10, 15, 20),
                     N_donors: int = 8, T: int = 30,
                     true_effect: float = -3.0, noise: float = 0.4) -> pd.DataFrame:
    """Staggered panel from the paper's linear factor DGP."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 2))
    loadings_donors = rng.standard_normal((N_donors, 2)) * 0.5
    loadings_treated = loadings_donors.mean(axis=0)
    records = []
    for j, T_j in enumerate(adoption_offsets):
        base_load = loadings_treated + 0.1 * rng.standard_normal(2)
        series = factors @ base_load + rng.standard_normal(T) * noise
        series[T_j:] += true_effect
        for t in range(T):
            records.append({"unit": f"treated_{j}", "year": 2000 + t,
                            "y": float(series[t]), "tr": int(t >= T_j)})
    for dd in range(N_donors):
        series = factors @ loadings_donors[dd] + rng.standard_normal(T) * noise
        for t in range(T):
            records.append({"unit": f"d_{dd}", "year": 2000 + t,
                            "y": float(series[t]), "tr": 0})
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _staggered_panel()


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="tr", unitid="unit", time="year",
                display_graphs=False, run_inference=False)
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Layer 1: formatting + fixed effects
# --------------------------------------------------------------------------- #
def test_prepare_inputs(panel):
    inp = prepare_ppscm_inputs(panel, outcome="y", treat="tr", unitid="unit", time="year")
    assert isinstance(inp, PPSCMInputs)
    assert inp.Xy.shape[0] == inp.n
    assert np.isfinite(inp.trt).sum() == 3            # three treated cohorts
    assert (~np.isfinite(inp.trt)).sum() == 8         # eight never-treated controls
    assert inp.n_pre == 20                            # periods before last adoption (offset 20)


def test_fit_feff_two_way_keys_and_shape():
    rng = np.random.default_rng(0)
    Xy = rng.standard_normal((6, 8))
    trt = np.array([3, 5, np.inf, np.inf, np.inf, np.inf], float)
    res = fit_feff(Xy, trt, {3, 5}, fixedeff=True)
    assert set(res.keys()) == {3, 5}
    assert res[3].shape == Xy.shape
    # without fixed effects only the control time-mean is removed
    res0 = fit_feff(Xy, trt, {3, 5}, fixedeff=False)
    ever = np.isfinite(trt)
    assert np.allclose(np.nanmean(res0[3][~ever], axis=0), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Layer 2: engine
# --------------------------------------------------------------------------- #
def test_run_multisynth_recovers_effect(panel):
    inp = prepare_ppscm_inputs(panel, outcome="y", treat="tr", unitid="unit", time="year")
    fit = run_multisynth(inp.Xy, inp.trt, inp.n_pre, n_leads=inp.Xy.shape[1] - inp.n_pre,
                         n_lags=inp.n_pre, fixedeff=True, time_cohort=False, nu=None)
    assert fit["att"] < -1.0                          # true effect -3
    assert np.isfinite(fit["nu_used"]) and fit["nu_used"] >= 0
    for w in fit["weights"].values():
        assert w.sum() == pytest.approx(1.0, abs=1e-4) and np.all(w >= -1e-7)


def test_time_cohort_and_separate_both_run(panel):
    inp = prepare_ppscm_inputs(panel, outcome="y", treat="tr", unitid="unit", time="year")
    H, L = inp.Xy.shape[1] - inp.n_pre, inp.n_pre
    sep = run_multisynth(inp.Xy, inp.trt, inp.n_pre, H, L, time_cohort=False, nu=None)
    coh = run_multisynth(inp.Xy, inp.trt, inp.n_pre, H, L, time_cohort=True, nu=None)
    assert np.isfinite(sep["att"]) and np.isfinite(coh["att"])


def test_fixed_nu_respected(panel):
    inp = prepare_ppscm_inputs(panel, outcome="y", treat="tr", unitid="unit", time="year")
    fit = run_multisynth(inp.Xy, inp.trt, inp.n_pre, inp.Xy.shape[1] - inp.n_pre,
                         inp.n_pre, nu=0.5)
    assert fit["nu_used"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Layer 3: estimator end-to-end
# --------------------------------------------------------------------------- #
def test_fit_returns_results(panel):
    res = PPSCM(_cfg(panel)).fit()
    assert isinstance(res, PPSCMResults)
    assert isinstance(res.event_study, PPSCMEventStudy)
    assert res.att < 0
    assert res.event_study.tau.shape == res.event_study.horizons.shape
    assert all(0 <= w <= 1.0 + 1e-6 for dd in res.donor_weights.values() for w in dd.values())


def test_time_cohort_flag(panel):
    res = PPSCM(_cfg(panel, time_cohort=True)).fit()
    assert res.design.time_cohort is True
    assert np.isfinite(res.att)


def test_jackknife_inference_runs(panel):
    res = PPSCM(_cfg(panel, run_inference=True)).fit()
    assert res.inference.method == "jackknife"
    assert np.isfinite(res.inference.se) and res.inference.se > 0
    lo, hi = res.inference.ci
    assert lo < res.att < hi


def test_requires_treated_unit(panel):
    df = panel.copy(); df["tr"] = 0
    with pytest.raises(MlsynthDataError):
        PPSCM(_cfg(df)).fit()


# --------------------------------------------------------------------------- #
# Layer 4: regression against the augsynth multisynth vignette
# --------------------------------------------------------------------------- #
_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "Teachingaugsynth.scv")


@pytest.mark.skipif(not os.path.exists(_DATA), reason="Paglayan data not present")
def test_matches_augsynth_vignette():
    df = pd.read_csv(_DATA)
    d = df[~df.State.isin(["DC", "WI"])].copy()
    d = d[(d.year >= 1959) & (d.year <= 1997)].copy()
    d["cbr"] = (d["year"] >= d["YearCBrequired"].fillna(np.inf)).astype(int)
    common = dict(df=d, outcome="lnppexpend", treat="cbr", unitid="State",
                  time="year", display_graphs=False, run_inference=False)
    # default: augsynth nu = 0.2607, Average ATT = -0.011
    r = PPSCM(common).fit()
    assert r.design.nu_used == pytest.approx(0.2607, abs=2e-3)
    assert r.att == pytest.approx(-0.011, abs=1.5e-3)
    # time cohort: augsynth nu = 0.3939, Average ATT = -0.018
    rt = PPSCM({**common, "time_cohort": True}).fit()
    assert rt.design.nu_used == pytest.approx(0.3939, abs=2e-3)
    assert rt.att == pytest.approx(-0.018, abs=2e-3)
