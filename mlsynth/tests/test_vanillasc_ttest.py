"""Integration tests for ``VanillaSC(inference="ttest")``.

Wires the Chernozhukov, Wuthrich & Zhu (2025) debiased SC t-test
(``utils/inferutils.debiased_sc_ttest``) into VanillaSC: the per-fold weights
are refit with the configured backend on each pre-period block-complement.
Pinned on the two canonical SC datasets — Basque terrorism and California
Proposition 99 (the 38-control-state ADH pool, full predictor spec + lags).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"


def _basque_df():
    f = _BASEDATA / "basque_jasa.csv"
    if not f.exists():
        pytest.skip("basque_jasa.csv not available")
    df = pd.read_csv(f)
    df = df[df.regionname != "Spain (Espana)"].copy()
    df["treated"] = ((df.regionname == "Basque Country (Pais Vasco)")
                     & (df.year >= 1975)).astype(int)
    return df


def _prop99_df():
    f = _BASEDATA / "augmented_cali_long.csv"
    if not f.exists():
        pytest.skip("augmented_cali_long.csv not available")
    return pd.read_csv(f)


def test_ttest_returns_inference_contract():
    df = _basque_df()
    res = VanillaSC({"df": df, "outcome": "gdpcap", "treat": "treated",
                     "unitid": "regionname", "time": "year",
                     "backend": "outcome-only", "inference": "ttest",
                     "ttest_K": 3, "alpha": 0.1, "display_graphs": False}).fit()
    inf = res.inference
    assert inf is not None
    assert "Chernozhukov" in inf.method
    assert inf.ci_lower < inf.ci_upper
    assert inf.confidence_level == pytest.approx(0.9)
    d = inf.details
    for k in ("att_debiased", "att_naive", "se", "tstat", "dof", "K", "r", "tau_k"):
        assert k in d
    assert d["dof"] == 2 and d["K"] == 3 and len(d["tau_k"]) == 3


def test_ttest_basque_outcome_only_pinned():
    df = _basque_df()
    res = VanillaSC({"df": df, "outcome": "gdpcap", "treat": "treated",
                     "unitid": "regionname", "time": "year",
                     "backend": "outcome-only", "inference": "ttest",
                     "ttest_K": 3, "alpha": 0.1, "display_graphs": False}).fit()
    d = res.inference.details
    np.testing.assert_allclose(d["att_debiased"], -0.6575, atol=5e-3)
    np.testing.assert_allclose(d["se"], 0.1391, atol=5e-3)
    assert res.inference.ci_upper < 0                    # significant at 10%
    assert res.inference.p_value < 0.1


def test_ttest_prop99_outcome_only_pinned():
    d99 = _prop99_df()
    res = VanillaSC({"df": d99, "outcome": "cigsale", "treat": "Proposition 99",
                     "unitid": "state", "time": "year",
                     "backend": "outcome-only", "inference": "ttest",
                     "ttest_K": 3, "alpha": 0.1, "display_graphs": False}).fit()
    d = res.inference.details
    # ADH Prop 99: ~ -19 packs/capita; debiased ~ -18, significant.
    np.testing.assert_allclose(d["att_debiased"], -17.99, atol=0.1)
    np.testing.assert_allclose(d["se"], 1.563, atol=0.1)
    assert res.inference.ci_upper < 0
    assert d["att_naive"] < -18                          # the canonical naive gap


def test_ttest_prop99_mscmt_full_spec_with_lags():
    d99 = _prop99_df()
    covs = ["p_cig", "loginc", "pct15-24", "pc_beer", "smk_75", "smk_80", "smk_88"]
    win = {"p_cig": (1980, 1988), "loginc": (1980, 1988),
           "pct15-24": (1980, 1988), "pc_beer": (1984, 1988)}
    res = VanillaSC({"df": d99, "outcome": "cigsale", "treat": "Proposition 99",
                     "unitid": "state", "time": "year", "backend": "mscmt",
                     "canonical_v": "min.loss.w", "covariates": covs,
                     "covariate_windows": win, "inference": "ttest",
                     "ttest_K": 3, "alpha": 0.1, "seed": 1,
                     "display_graphs": False}).fit()
    d = res.inference.details
    # Full ADH predictor spec + 3 outcome lags; debiased ATT in the -19..-12 band.
    assert -25.0 < d["att_debiased"] < -10.0
    assert res.inference.ci_upper < 0
    assert d["r"] == 6
