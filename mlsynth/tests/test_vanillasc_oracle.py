"""Tests for ``VanillaSC(oracle_weights=...)`` -- user-specified donor weights.

When ``oracle_weights`` is supplied, VanillaSC skips the weight optimization and
uses the given weights directly. This exposes the "oracle" case (known weights)
of the CWZ (2025) simulations through the public API, and lets a practitioner
plug in externally computed weights. The defining check: handing back the
*fitted* SC weights must reproduce the fitted run exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.inferutils import debiased_sc_ttest

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"


def _basque():
    f = _BASEDATA / "basque_jasa.csv"
    if not f.exists():
        pytest.skip("basque_jasa.csv not available")
    df = pd.read_csv(f)
    df = df[df.regionname != "Spain (Espana)"].copy()
    df["treated"] = ((df.regionname == "Basque Country (Pais Vasco)")
                     & (df.year >= 1975)).astype(int)
    return df


_BASE = dict(outcome="gdpcap", treat="treated", unitid="regionname", time="year",
             backend="outcome-only", display_graphs=False)


def test_oracle_weights_reproduce_fitted_run():
    # Fit, then feed the fitted weights back as oracle: identical results, no solve.
    df = _basque()
    fit = VanillaSC({"df": df, "inference": False, **_BASE}).fit()
    w = {str(k): float(v) for k, v in fit.weights.donor_weights.items()}

    orc = VanillaSC({"df": df, "inference": False, "oracle_weights": w, **_BASE}).fit()
    np.testing.assert_allclose(orc.effects.att, fit.effects.att, atol=1e-9)
    np.testing.assert_allclose(orc.time_series.counterfactual_outcome,
                               fit.time_series.counterfactual_outcome, atol=1e-9)
    assert orc.method_details.parameters_used["backend"] == "oracle"
    for k, v in w.items():
        assert abs(orc.weights.donor_weights[k] - v) < 1e-9


def test_oracle_weights_skip_optimization_via_ttest():
    # The t-test "Oracle" case: cross-fit with known weights (no per-fold refit).
    df = _basque()
    piv = df.pivot(index="year", columns="regionname", values="gdpcap").sort_index()
    treated = "Basque Country (Pais Vasco)"
    donors = [c for c in piv.columns if c != treated]
    y = piv[treated].to_numpy()
    Y0 = piv[donors].to_numpy()
    T0 = int((piv.index < 1975).sum())
    T1 = int((piv.index >= 1975).sum())
    fit = VanillaSC({"df": df, "inference": False, **_BASE}).fit()
    wdict = {str(k): float(v) for k, v in fit.weights.donor_weights.items()}
    wvec = np.array([wdict.get(d, 0.0) for d in donors])

    orc = VanillaSC({"df": df, "inference": "ttest", "ttest_K": 3, "alpha": 0.1,
                     "oracle_weights": wdict, **_BASE}).fit()
    direct = debiased_sc_ttest(y, Y0, T0, T1, K=3, alpha=0.1,
                               weight_fn=lambda idx: wvec)
    np.testing.assert_allclose(orc.inference.details["att_debiased"],
                               direct["att"], atol=1e-9)
    np.testing.assert_allclose(orc.inference.ci_lower, direct["ci_lower"], atol=1e-9)


def test_oracle_weights_missing_donors_default_zero():
    df = _basque()
    orc = VanillaSC({"df": df, "inference": False,
                     "oracle_weights": {"Cataluna": 0.8, "Madrid (Comunidad De)": 0.2},
                     **_BASE}).fit()
    w = orc.weights.donor_weights
    assert abs(w["Cataluna"] - 0.8) < 1e-9 and abs(w["Madrid (Comunidad De)"] - 0.2) < 1e-9
    assert abs(w.get("Aragon", 0.0)) < 1e-9          # unlisted donor -> 0


def test_oracle_weights_unknown_donor_raises():
    df = _basque()
    with pytest.raises(MlsynthDataError):
        VanillaSC({"df": df, "inference": False,
                   "oracle_weights": {"Atlantis": 1.0}, **_BASE}).fit()


def test_oracle_weights_incompatible_inference_raises():
    df = _basque()
    with pytest.raises(MlsynthConfigError):
        VanillaSC({"df": df, "inference": "scpi",
                   "oracle_weights": {"Cataluna": 1.0}, **_BASE}).fit()
