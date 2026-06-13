"""ROLLDID — rolling-transformation DiD (Lee & Wooldridge 2026, small-N exact).

Validation is Path A: reproduce the paper's two empirical applications exactly —
California Prop 99 (common timing, Table 3) and castle laws (staggered, §7.2) —
plus config-error and plotting smoke. Clean-room from the paper equations;
``basedata/smoking_data.csv`` and ``basedata/castle.csv`` are shipped in-repo so
the suite is network-free.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import ROLLDID
from mlsynth.config_models import BaseEstimatorResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------

def _smoking():
    d = pd.read_csv("basedata/smoking_data.csv")
    d["logcig"] = np.log(d["cigsale"])           # paper uses log per-capita sales
    return d


def _smoking_cfg(**over):
    base = dict(df=_smoking(), outcome="logcig", treat="Proposition 99",
                unitid="state", time="year", display_graphs=False)
    base.update(over)
    return base


def _castle():
    d = pd.read_csv("basedata/castle.csv")
    # W_it = 1 for an eventually-treated state from its adoption year on
    d["W"] = ((d["effyear"].notna()) & (d["year"] >= d["effyear"])).astype(int)
    return d


def _castle_cfg(**over):
    base = dict(df=_castle(), outcome="l_homicide", treat="W",
                unitid="state", time="year", display_graphs=False)
    base.update(over)
    return base


# ----------------------------------------------------------------------------
# Prop 99 — common timing (Table 3), exact reproduction
# ----------------------------------------------------------------------------

def test_prop99_demean_att_and_se():
    res = ROLLDID(_smoking_cfg(rolling="demean", inference="exact")).fit()
    assert isinstance(res, BaseEstimatorResults)
    assert res.effects.att == pytest.approx(-0.422, abs=5e-3)        # paper -0.422
    assert res.inference.standard_error == pytest.approx(0.121, abs=5e-3)
    assert res.transformation == "demean"
    assert res.design == "common"
    assert res.n_treated == 1 and res.n_control == 38


def test_prop99_detrend_att_se_and_exact_p():
    res = ROLLDID(_smoking_cfg(rolling="detrend", inference="exact")).fit()
    assert res.effects.att == pytest.approx(-0.227, abs=5e-3)        # paper -0.227
    assert res.inference.standard_error == pytest.approx(0.094, abs=5e-3)
    assert res.inference.p_value == pytest.approx(0.021, abs=2e-3)   # paper 0.021
    assert res.inference.method == "exact-t"


def test_prop99_per_period_event_study():
    res = ROLLDID(_smoking_cfg(rolling="demean")).fit()
    pp = res.per_period
    assert isinstance(pp, pd.DataFrame)
    g = lambda yr: float(pp.loc[pp["time"] == yr, "att"].iloc[0])
    assert g(1989) == pytest.approx(-0.168, abs=5e-3)
    assert g(1995) == pytest.approx(-0.484, abs=5e-3)
    assert g(2000) == pytest.approx(-0.667, abs=5e-3)
    # the aggregated ATT is the mean of the per-period effects
    assert res.effects.att == pytest.approx(pp["att"].mean(), abs=1e-9)


def test_prop99_detrend_per_period():
    res = ROLLDID(_smoking_cfg(rolling="detrend")).fit()
    pp = res.per_period
    g = lambda yr: float(pp.loc[pp["time"] == yr, "att"].iloc[0])
    assert g(2000) == pytest.approx(-0.403, abs=5e-3)               # paper -0.403


# ----------------------------------------------------------------------------
# castle laws — staggered (§7.2), exact reproduction
# ----------------------------------------------------------------------------

def test_castle_demean_aggregated():
    res = ROLLDID(_castle_cfg(rolling="demean", inference="exact")).fit()
    assert res.design == "staggered"
    assert res.effects.att == pytest.approx(0.092, abs=3e-3)        # paper 0.092
    assert res.inference.standard_error == pytest.approx(0.057, abs=3e-3)
    assert res.n_treated == 21 and res.n_control == 29


def test_castle_detrend_hc3():
    res = ROLLDID(_castle_cfg(rolling="detrend", inference="hc3")).fit()
    assert res.effects.att == pytest.approx(0.067, abs=3e-3)        # paper 0.067
    assert res.inference.standard_error == pytest.approx(0.055, abs=3e-3)
    assert res.inference.method == "hc3"


def test_castle_per_cohort_present():
    res = ROLLDID(_castle_cfg(rolling="demean")).fit()
    pc = res.per_cohort
    assert isinstance(pc, pd.DataFrame)
    assert set(pc["cohort"]) == {2005.0, 2006.0, 2007.0, 2008.0, 2009.0}


# ----------------------------------------------------------------------------
# inference variants
# ----------------------------------------------------------------------------

def test_randomization_inference_runs_and_bounded():
    res = ROLLDID(_smoking_cfg(rolling="detrend", inference="ri", ri_reps=200,
                               seed=0)).fit()
    assert res.inference.method == "ri"
    assert 0.0 <= res.inference.p_value <= 1.0
    assert res.inference.p_value < 0.2                              # strong effect


def test_hc3_changes_se_not_point():
    # use castle (21 treated) — HC3 is well-defined with a handful of treated
    base = ROLLDID(_castle_cfg(rolling="detrend", inference="exact")).fit()
    hc3 = ROLLDID(_castle_cfg(rolling="detrend", inference="hc3")).fit()
    assert hc3.effects.att == pytest.approx(base.effects.att, abs=1e-9)
    assert hc3.inference.standard_error != base.inference.standard_error
    assert np.isfinite(hc3.inference.standard_error)


def test_hc3_single_treated_raises():
    """HC3 is degenerate with one treated unit (leverage 1) — reported, not NaN."""
    with pytest.raises(MlsynthConfigError, match="HC3"):
        ROLLDID(_smoking_cfg(rolling="detrend", inference="hc3")).fit()


# ----------------------------------------------------------------------------
# config validation
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("over,msg", [
    ({"rolling": "bogus"}, "rolling"),
    ({"inference": "bootstrap"}, "inference"),
    ({"alpha": 1.5}, "alpha"),
    ({"ri_reps": 0}, "ri_reps"),
])
def test_config_validation(over, msg):
    with pytest.raises(MlsynthConfigError, match=msg):
        ROLLDID(_smoking_cfg(**over))


def test_rejects_non_config_non_dict():
    with pytest.raises(MlsynthConfigError):
        ROLLDID(42)


def test_no_treated_units_raises():
    d = _smoking()
    d["Proposition 99"] = False                                    # nobody treated
    with pytest.raises((MlsynthConfigError, MlsynthDataError)):
        ROLLDID(_smoking_cfg(df=d)).fit()


# ----------------------------------------------------------------------------
# setup edge / failure cases (direct, since the base config guards columns)
# ----------------------------------------------------------------------------

from mlsynth.utils.rolldid_helpers.setup import rolldid_setup


def _toy(treat_map=None, drop=None, y=None):
    """Small balanced long panel: u0,u1 treated at t=2; u2,u3 never."""
    rows = []
    for u in ("u0", "u1", "u2", "u3"):
        for t in range(4):
            w = int(u in ("u0", "u1") and t >= 2)
            rows.append({"unit": u, "time": t, "y": float(t + (u == "u0")), "w": w})
    df = pd.DataFrame(rows)
    return df


def test_setup_missing_column_raises():
    with pytest.raises(MlsynthConfigError, match="nope"):
        rolldid_setup(_toy(), "unit", "time", "y", "nope")


def test_setup_non_binary_treat_raises():
    df = _toy(); df.loc[0, "w"] = 2
    with pytest.raises(MlsynthDataError, match="0/1"):
        rolldid_setup(df, "unit", "time", "y", "w")


def test_setup_unbalanced_raises():
    df = _toy().iloc[1:]                                   # drop a unit x time cell
    with pytest.raises(MlsynthDataError, match="unbalanced"):
        rolldid_setup(df, "unit", "time", "y", "w")


def test_setup_non_absorbing_raises():
    df = _toy()
    df.loc[(df.unit == "u0") & (df.time == 3), "w"] = 0    # turns back off
    with pytest.raises(MlsynthDataError, match="absorbing"):
        rolldid_setup(df, "unit", "time", "y", "w")


def test_setup_no_never_treated_raises():
    df = _toy()
    df.loc[df.unit.isin(("u2", "u3")) & (df.time >= 2), "w"] = 1   # everyone treated
    with pytest.raises(MlsynthDataError, match="never-treated"):
        rolldid_setup(df, "unit", "time", "y", "w")


# ----------------------------------------------------------------------------
# plotting smoke
# ----------------------------------------------------------------------------

def test_plot_event_study_smoke():
    import matplotlib
    matplotlib.use("Agg")
    res = ROLLDID(_smoking_cfg(rolling="demean", display_graphs=True)).fit()
    assert isinstance(res, BaseEstimatorResults)


def test_plot_staggered_smoke():
    import matplotlib
    matplotlib.use("Agg")
    res = ROLLDID(_castle_cfg(rolling="demean", display_graphs=True)).fit()
    assert res.per_cohort is not None
