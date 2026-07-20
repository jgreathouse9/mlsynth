"""Tests for SDID's synthetic triple-difference (SC-DDD) mode.

The ``subgroup`` / ``target_subgroup`` config switches SDID into the Zhuang
(2024, arXiv:2409.12353) synthetic triple difference: the outcome is demeaned
by the non-target subgroup within each treatment-group-by-time cell, then SDID
runs on the transformed outcome. Path A reproduces Feldman & Semprini (2026)'s
Virginia HPV mandate result (SC-DDD = +1.559, cross-validated against the
authors' Stata ``sdid``); the remaining tests pin the transform, the
mode-equals-manual equivalence, and the config/failure contract.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import SDID
from mlsynth.config_models import SDIDConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.sdid_helpers.setup import apply_ddd_transform

_HERE = os.path.dirname(__file__)
_HPV = os.path.abspath(os.path.join(_HERE, "..", "..", "basedata",
                                    "hpv_cervical_ddd.csv"))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _hpv_frame():
    d = pd.read_csv(_HPV)
    d["treated"] = ((d.state == "Virginia") & (d.year >= 2016)
                    & (d.age == "20-24")).astype(int)
    return d


@pytest.fixture(scope="module")
def hpv():
    return _hpv_frame()


def _synth_ddd_panel(n_units=8, T=12, T0=8, seed=0):
    """Panel with a target and one non-target subgroup; unit 0 treated."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.standard_normal(T)) * 0.3
    rows = []
    for i in range(n_units):
        load = rng.standard_normal()
        base_t = 5.0 + load * common + rng.standard_normal(T) * 0.2   # target
        base_n = 3.0 + 0.5 * load * common + rng.standard_normal(T) * 0.2  # non-target
        for t in range(T):
            treated_cell = (i == 0 and t >= T0)
            yt = base_t[t] + (2.0 if treated_cell else 0.0)
            rows.append({"unit": f"u{i}", "year": 2000 + t, "grp": "T",
                         "y": float(yt), "treat": int(treated_cell)})
            rows.append({"unit": f"u{i}", "year": 2000 + t, "grp": "N",
                         "y": float(base_n[t]), "treat": 0})
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="year",
                display_graphs=False)
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Path A — HPV reproduction
# --------------------------------------------------------------------------- #
def test_path_a_hpv_scddd(hpv):
    """SC-DDD reproduces the paper's +1.559 (Stata sdid cross-validation)."""
    res = SDID({"df": hpv, "outcome": "cervix_adj", "treat": "treated",
                "unitid": "state", "time": "year",
                "subgroup": "age", "target_subgroup": "20-24",
                "display_graphs": False}).fit()
    assert res.effects.att == pytest.approx(1.559, abs=0.05)
    assert res.method_details.method_name == "SDID-DDD"
    # placebo CI excludes zero (paper: 0.418, 2.699)
    assert res.inference.ci_lower > 0


def test_path_a_naive_scdd_is_null(hpv):
    """Ordinary SDID on the untransformed 20-24 outcome is ~null (paper 0.252)."""
    sub = hpv[hpv.age == "20-24"].copy()
    res = SDID({"df": sub, "outcome": "cervix_adj", "treat": "treated",
                "unitid": "state", "time": "year", "display_graphs": False}).fit()
    assert res.effects.att == pytest.approx(0.252, abs=0.05)
    assert res.method_details.method_name == "SDID"


# --------------------------------------------------------------------------- #
# Transform + equivalence
# --------------------------------------------------------------------------- #
def test_transform_matches_definition():
    """apply_ddd_transform yields W = Y - non-target group-by-time mean."""
    df = _synth_ddd_panel()
    reduced, col = apply_ddd_transform(
        df, "y", "treat", "unit", "year", "grp", "T")
    assert col == "y__ddd"
    # one row per (unit, year); only target rows survive
    assert len(reduced) == df.unit.nunique() * df.year.nunique()
    # hand-check a control-unit cell: W = Y_target - mean(non-target controls, t)
    yr = 2003
    ctrl_nontarget = df[(df.grp == "N") & (df.unit != "u0") & (df.year == yr)]
    exp_mean = ctrl_nontarget.y.mean()
    u1_target = df[(df.grp == "T") & (df.unit == "u1") & (df.year == yr)].y.iloc[0]
    got = reduced[(reduced.unit == "u1") & (reduced.year == yr)][col].iloc[0]
    assert got == pytest.approx(u1_target - exp_mean)


def test_ddd_mode_equals_manual_transform_plus_sdid():
    """DDD mode == manual transform fed to ordinary SDID (identical ATT)."""
    df = _synth_ddd_panel()
    reduced, col = apply_ddd_transform(
        df, "y", "treat", "unit", "year", "grp", "T")
    manual = SDID(_cfg(reduced, outcome=col)).fit()
    mode = SDID(_cfg(df, subgroup="grp", target_subgroup="T")).fit()
    assert mode.effects.att == pytest.approx(manual.effects.att, abs=1e-9)


def test_treated_group_uses_own_nontarget_mean():
    """The ever-treated unit is demeaned by its own non-target rows, not pooled."""
    df = _synth_ddd_panel()
    reduced, col = apply_ddd_transform(
        df, "y", "treat", "unit", "year", "grp", "T")
    yr = 2003
    va_nontarget = df[(df.grp == "N") & (df.unit == "u0") & (df.year == yr)].y.mean()
    va_target = df[(df.grp == "T") & (df.unit == "u0") & (df.year == yr)].y.iloc[0]
    got = reduced[(reduced.unit == "u0") & (reduced.year == yr)][col].iloc[0]
    assert got == pytest.approx(va_target - va_nontarget)


# --------------------------------------------------------------------------- #
# Ordinary SDID untouched
# --------------------------------------------------------------------------- #
def test_plain_sdid_unaffected():
    df = _synth_ddd_panel()
    sub = df[df.grp == "T"].copy()
    res = SDID(_cfg(sub)).fit()
    assert res.method_details.method_name == "SDID"
    assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# Config / data contract
# --------------------------------------------------------------------------- #
def test_subgroup_without_target_rejected():
    df = _synth_ddd_panel()
    with pytest.raises(MlsynthConfigError):
        SDIDConfig(**_cfg(df, subgroup="grp"))


def test_target_without_subgroup_rejected():
    df = _synth_ddd_panel()
    with pytest.raises(MlsynthConfigError):
        SDIDConfig(**_cfg(df, target_subgroup="T"))


def test_subgroup_column_missing_rejected():
    df = _synth_ddd_panel()
    with pytest.raises(MlsynthConfigError):
        SDIDConfig(**_cfg(df, subgroup="ghost", target_subgroup="T"))


def test_target_value_absent_rejected():
    df = _synth_ddd_panel()
    with pytest.raises(MlsynthConfigError):
        SDIDConfig(**_cfg(df, subgroup="grp", target_subgroup="ZZZ"))


def test_no_nontarget_rows_rejected():
    """A subgroup column with only the target value has nothing to demean by."""
    df = _synth_ddd_panel()
    df = df[df.grp == "T"].copy()
    with pytest.raises(MlsynthConfigError):
        SDIDConfig(**_cfg(df, subgroup="grp", target_subgroup="T"))


def test_nonnumeric_outcome_rejected():
    df = _synth_ddd_panel()
    df["y"] = df["y"].astype(object)
    df.loc[df.index[0], "y"] = "oops"
    with pytest.raises(MlsynthDataError):
        SDID(_cfg(df, subgroup="grp", target_subgroup="T")).fit()


def test_missing_nontarget_cell_rejected():
    """A group-by-time cell with a target but no non-target row is undefined."""
    df = _synth_ddd_panel()
    # Drop the treated unit's non-target rows in one year -> (grp=1, that year)
    # has a target row but nothing to demean by.
    mask = (df.unit == "u0") & (df.grp == "N") & (df.year == 2003)
    df = df[~mask].copy()
    with pytest.raises(MlsynthDataError):
        SDID(_cfg(df, subgroup="grp", target_subgroup="T")).fit()


def test_nonunique_target_rejected():
    """Two target rows per (unit, year) is ambiguous."""
    df = _synth_ddd_panel()
    dup = df[df.grp == "T"].copy()
    df2 = pd.concat([df, dup], ignore_index=True)   # duplicate target rows
    with pytest.raises(MlsynthDataError):
        SDID(_cfg(df2, subgroup="grp", target_subgroup="T")).fit()
