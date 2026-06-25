"""TDD for the Truncated History (TH) robustness diagnostic.

Spoelstra et al. (2025), Economics Letters 257, 112701. TH re-estimates an SC
effect on truncated pre-treatment windows and profiles the ATT vs the
pretreatment horizon.

Layers:
 * structural/mode/failure tests use a controllable fake estimator (fast,
   deterministic) so the diagnostic is tested in isolation;
 * an integration test runs the real ``SDID`` and pins the left-TH profile
   against Spoelstra et al. Table 1 (California tobacco), where it reproduces the
   reported ATEs to the decimal.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import truncated_history, SDID, VanillaSC
from mlsynth.utils.truncated_history import (
    TruncatedHistoryResult,
    TruncatedHistoryWindow,
)
from mlsynth.config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
)
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


# --------------------------------------------------------------- fixtures ----

def _panel(n_pre=10, n_post=4, n_donors=5, seed=0):
    """A small balanced panel: unit 'T' treated after the pre-period."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + n_pre + n_post))
    treat_start = years[n_pre]
    rows = []
    f = rng.normal(size=len(years))
    for j in range(n_donors):
        s = f * rng.normal() + rng.normal(scale=0.3, size=len(years)) + 10
        for t, y in zip(years, s):
            rows.append({"unit": f"d{j}", "year": t, "y": y, "treat": 0})
    treated = f * 1.0 + rng.normal(scale=0.1, size=len(years)) + 10
    for i, (t, y) in enumerate(zip(years, treated)):
        post = t >= treat_start
        rows.append({"unit": "T", "year": t, "y": y + (-2.0 if post else 0.0),
                     "treat": int(post)})
    return pd.DataFrame(rows), years[:n_pre]


def _cfg(df):
    return {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit", "time": "year"}


# Controllable fake estimator: ATT is a function of the kept pre-period count,
# so tests can dial stability deterministically.
_ATT_FN = {"f": lambda n_pre: -20.0}


class _Fake:
    def __init__(self, config):
        self.config = config

    def fit(self):
        df = self.config["df"]
        time, treat = self.config["time"], self.config["treat"]
        start = df[df[treat] == 1][time].min()
        n_pre = int(df[df[time] < start][time].nunique())
        att = float(_ATT_FN["f"](n_pre))
        return BaseEstimatorResults(
            effects=EffectsResults(att=att),
            fit_diagnostics=FitDiagnosticsResults(rmse_pre=0.5),
            inference=InferenceResults(p_value=0.04, standard_error=1.5),
        )


# --------------------------------------------------------------- structure ----

def test_left_th_smoke_and_profile_shape():
    df, pre = _panel()
    res = truncated_history(_Fake, _cfg(df), mode="left", min_pre=2)
    assert isinstance(res, TruncatedHistoryResult)
    assert res.mode == "left"
    # left drops 1..(len(pre)-min_pre) earliest periods
    assert len(res.profile) == len(pre) - 2
    assert all(isinstance(w, TruncatedHistoryWindow) for w in res.profile)
    # n_pre strictly decreases as more early periods are dropped
    npre = [w.n_pre_periods for w in res.profile]
    assert npre == sorted(npre, reverse=True)
    assert npre[0] == len(pre) - 1


def test_window_counts_per_mode():
    df, pre = _panel(n_pre=8)
    P, mp = len(pre), 2
    counts = {
        "left": P - mp,
        "right": P - mp,
        "loo": P,
        "l2o": P * (P - 1) // 2,
    }
    for mode, n in counts.items():
        res = truncated_history(_Fake, _cfg(df), mode=mode, min_pre=mp)
        assert len(res.profile) == n, mode


def test_random_mode_is_seeded_and_bounded():
    df, _ = _panel(n_pre=8)
    a = truncated_history(_Fake, _cfg(df), mode="random", n_random=10, seed=7)
    b = truncated_history(_Fake, _cfg(df), mode="random", n_random=10, seed=7)
    assert [w.label for w in a.profile] == [w.label for w in b.profile]   # reproducible
    assert all(w.n_pre_periods >= 2 for w in a.profile)


def test_dropped_periods_actually_removed():
    df, pre = _panel()
    captured = {}

    class _Spy(_Fake):
        def fit(self):
            yrs = set(self.config["df"][self.config["time"]].unique())
            captured.setdefault("seen", []).append(yrs)
            return super().fit()

    truncated_history(_Spy, _cfg(df), mode="loo", min_pre=2)
    # each loo window drops exactly one distinct pre-year
    dropped = [set(pre) - (s & set(pre)) for s in captured["seen"][1:]]  # skip full fit
    assert all(len(d) == 1 for d in dropped)
    assert {next(iter(d)) for d in dropped} == set(pre)


def test_extracts_mspe_pvalue_se():
    df, _ = _panel()
    res = truncated_history(_Fake, _cfg(df), mode="left")
    w = res.profile[0]
    assert w.pre_mspe == pytest.approx(0.25)         # rmse_pre=0.5 -> 0.25
    assert w.p_value == pytest.approx(0.04)
    assert w.standard_error == pytest.approx(1.5)


# --------------------------------------------------------------- stability ----

def test_stable_when_att_flat():
    df, _ = _panel()
    _ATT_FN["f"] = lambda n_pre: -20.0
    res = truncated_history(_Fake, _cfg(df), mode="left")
    assert res.stable is True
    assert res.att_min == res.att_max == -20.0
    assert "stable" in res.stability_note


def test_unstable_when_att_changes_sign():
    df, _ = _panel()
    _ATT_FN["f"] = lambda n_pre: -20.0 if n_pre > 6 else +20.0   # flips with truncation
    res = truncated_history(_Fake, _cfg(df), mode="left")
    assert res.stable is False
    assert "sign" in res.stability_note
    _ATT_FN["f"] = lambda n_pre: -20.0                            # restore


def test_unstable_when_spread_large():
    df, _ = _panel()
    _ATT_FN["f"] = lambda n_pre: -10.0 * n_pre                   # big drift, same sign
    res = truncated_history(_Fake, _cfg(df), mode="left", stability_tol=0.05)
    assert res.stable is False
    assert "interval" in res.stability_note
    _ATT_FN["f"] = lambda n_pre: -20.0


# --------------------------------------------------------------- failures ----

def test_invalid_mode_raises():
    df, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        truncated_history(_Fake, _cfg(df), mode="sideways")


def test_missing_df_raises():
    with pytest.raises(MlsynthConfigError):
        truncated_history(_Fake, {"outcome": "y", "treat": "t", "unitid": "u", "time": "yr"})


def test_missing_column_key_raises():
    df, _ = _panel()
    cfg = _cfg(df); del cfg["treat"]
    with pytest.raises(MlsynthConfigError):
        truncated_history(_Fake, cfg, mode="left")


def test_no_treated_rows_raises():
    df, _ = _panel()
    df = df.assign(treat=0)
    with pytest.raises(MlsynthDataError):
        truncated_history(_Fake, _cfg(df), mode="left")


def test_min_pre_too_large_raises():
    df, pre = _panel(n_pre=4)
    with pytest.raises(MlsynthConfigError):
        truncated_history(_Fake, _cfg(df), mode="left", min_pre=10)


def test_min_pre_bad_value_raises():
    df, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        truncated_history(_Fake, _cfg(df), mode="left", min_pre=0)


def test_loo_yields_no_windows_when_min_pre_blocks_it():
    df, _ = _panel(n_pre=3)
    with pytest.raises(MlsynthConfigError):
        truncated_history(_Fake, _cfg(df), mode="loo", min_pre=3)


def test_l2o_no_windows_past_main_guard():
    # 3 pre-periods, min_pre=2: passes the main (min_pre+1) guard, but l2o drops
    # two, leaving 1 < min_pre -> empty -> the 'no windows' error.
    df, _ = _panel(n_pre=3)
    with pytest.raises(MlsynthConfigError):
        truncated_history(_Fake, _cfg(df), mode="l2o", min_pre=2)


def test_treatment_from_first_period_has_no_pre():
    df, _ = _panel()
    df = df.assign(treat=1)                                   # treated in every period
    with pytest.raises(MlsynthDataError):
        truncated_history(_Fake, _cfg(df), mode="left")


def test_estimator_error_is_propagated():
    from mlsynth.exceptions import MlsynthEstimationError

    class _Boom:
        def __init__(self, config): pass
        def fit(self):
            raise MlsynthEstimationError("boom")

    df, _ = _panel()
    with pytest.raises(MlsynthEstimationError):
        truncated_history(_Boom, _cfg(df), mode="left")


# --------------------------------------------------------------- contract ----

def test_result_models_frozen():
    df, _ = _panel()
    res = truncated_history(_Fake, _cfg(df), mode="left")
    with pytest.raises(Exception):
        res.stable = True
    with pytest.raises(Exception):
        res.profile[0].att = 0.0


# ------------------------------------------------------------- integration ----

_P99 = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "P99data.csv")


@pytest.mark.skipif(not os.path.exists(_P99), reason="Prop 99 data absent")
def test_sdid_left_th_reproduces_spoelstra_table1():
    df = pd.read_csv(os.path.abspath(_P99)).rename(columns={"cigsale": "y"})
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    p99_cfg = {"df": df, "outcome": "y", "treat": "treat", "unitid": "state", "time": "year"}
    res = truncated_history(SDID, p99_cfg, mode="left", min_pre=2)
    by_label = {w.label: w.att for w in res.profile}
    # Spoelstra et al. Table 1 (SDID column), full sample + 1971-1974 starts.
    assert res.att_full == pytest.approx(-15.6, abs=0.3)
    assert by_label["1971-1988"] == pytest.approx(-16.3, abs=0.3)
    assert by_label["1972-1988"] == pytest.approx(-16.7, abs=0.4)
    assert by_label["1974-1988"] == pytest.approx(-17.2, abs=0.4)
    assert res.stable is True                                  # SDID is stable under left-TH


def test_vanillasc_left_th_runs_end_to_end():
    df, _ = _panel(n_pre=8, seed=3)
    res = truncated_history(VanillaSC, _cfg(df), mode="left", min_pre=3)
    assert np.isfinite(res.att_full)
    assert all(np.isfinite(w.att) for w in res.profile)
    assert res.profile[0].pre_mspe is not None
