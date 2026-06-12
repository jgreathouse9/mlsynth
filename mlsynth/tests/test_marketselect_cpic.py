"""CPIC (cost-per-incremental-conversion) budget layer for GEOLIFT.

Faithful to GeoLift's ``pvalueCalc`` /  ``GeoLiftMarketSelection``:

    investment <- cpic * sum(data_aux$Y[data_aux$D == 1]) * es

i.e. investment = cpic x effect_size x (summed treated volume over the lookback
window, baseline/pre-injection), then candidates whose investment exceeds the
budget are dropped.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.geolift_helpers.marketselect.helpers.simulate import simulate_lookback
from mlsynth.utils.geolift_helpers.marketselect.helpers.windows import (
    lookback_treatment_window,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.aggregate import compute_rank


def _sim_panel(n=20, J=3, seed=1):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(n, J)) + np.arange(n)[:, None] * 0.3
    treated = Y0 @ np.array([0.5, 0.3, 0.2]) + rng.normal(scale=0.05, size=n)
    return treated, Y0


# === investment formula (simulate) ===

def test_investment_matches_geolift_formula():
    """investment = cpic * es * sum(treated_total over the window)."""
    treated, Y0 = _sim_panel()
    treated_total = treated * 2.0                      # summed (2 units) vs the mean fit
    cpic = 7.5
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 0.05, 0.1],
                             augment="ridge", ns=20, seed=0,
                             cpic=cpic, treated_total=treated_total)
    start, end = lookback_treatment_window(20, 4, 1)
    window_vol = float(treated_total[start:end + 1].sum())
    for r in rows:
        assert r["investment"] == pytest.approx(cpic * r["effect_size"] * window_vol)


def test_investment_uses_total_not_fit_series():
    """The volume is the summed treated_total, not the (mean) fit series."""
    treated, Y0 = _sim_panel()
    treated_total = treated * 3.0
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.1],
                             augment="ridge", ns=20, seed=0,
                             cpic=2.0, treated_total=treated_total)
    start, end = lookback_treatment_window(20, 4, 1)
    assert rows[0]["investment"] == pytest.approx(
        2.0 * 0.1 * float(treated_total[start:end + 1].sum()))
    # would be 1/3 of that if it wrongly used the fit series
    assert rows[0]["investment"] != pytest.approx(
        2.0 * 0.1 * float(treated[start:end + 1].sum()))


def test_investment_zero_effect_is_zero():
    treated, Y0 = _sim_panel()
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.0],
                             augment="ridge", ns=20, seed=0,
                             cpic=7.5, treated_total=treated)
    assert rows[0]["investment"] == 0.0


def test_investment_nan_when_no_cpic():
    treated, Y0 = _sim_panel()
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.1], augment="ridge", ns=20, seed=0)
    assert np.isnan(rows[0]["investment"])


# === budget gate (compute_rank) ===

def _power_table_with_investment():
    """Two candidates, both detectable at es=0.1; A cheap, B over budget."""
    rows = []
    for cand, inv in [(frozenset({"A"}), 5_000.0), (frozenset({"B"}), 500_000.0)]:
        for es in (0.0, 0.05, 0.1):
            rows.append({
                "candidate": cand, "duration": 10, "effect_size": es,
                "power": 0.0 if es < 0.1 else 0.9,
                "placebo_mean_effect": es, "detected_lift": es,
                "scaled_l2": 0.1, "pre_rmspe": 1.0,
                "investment": inv * es / 0.1,           # scales with es
            })
    return pd.DataFrame(rows)


def test_budget_drops_overbudget_candidate():
    pt = _power_table_with_investment()
    ranked = compute_rank(pt, power_threshold=0.8, budget=100_000.0)
    cands = {next(iter(c)) for c in ranked["candidate"]}
    assert "A" in cands and "B" not in cands           # B's MDE investment 500k > 100k


def test_no_budget_keeps_all_detectable():
    pt = _power_table_with_investment()
    ranked = compute_rank(pt, power_threshold=0.8)     # budget=None
    cands = {next(iter(c)) for c in ranked["candidate"]}
    assert cands == {"A", "B"}


def test_rank_table_carries_investment():
    pt = _power_table_with_investment()
    ranked = compute_rank(pt, power_threshold=0.8)
    assert "investment" in ranked.columns
    a = ranked[ranked["candidate"] == frozenset({"A"})].iloc[0]
    assert a["investment"] == pytest.approx(5_000.0)   # at the MDE (es=0.1)


# === config validation ===

def _cfg(**kw):
    from mlsynth.utils.geolift_helpers.config import GeoLiftConfig
    base = dict(df=pd.DataFrame({"Y": [1.0, 2], "location": ["a", "a"], "date": [1, 2]}),
                outcome="Y", unitid="location", time="date",
                treatment_size=1, durations=[2], effect_sizes=[0.0, 0.1])
    base.update(kw)
    return GeoLiftConfig(**base)


def test_config_accepts_cpic_and_budget():
    cfg = _cfg(cpic=7.5, budget=100_000.0)
    assert cfg.cpic == 7.5 and cfg.budget == 100_000.0


def test_config_negative_cpic_raises():
    with pytest.raises(MlsynthConfigError, match="cpic"):
        _cfg(cpic=-1.0)


def test_config_budget_requires_cpic():
    with pytest.raises(MlsynthConfigError, match="budget requires cpic"):
        _cfg(budget=100_000.0)


def test_config_nonpositive_budget_raises():
    with pytest.raises(MlsynthConfigError, match="budget"):
        _cfg(cpic=7.5, budget=0.0)


# === realize cost ===

def test_realize_reports_cost():
    from mlsynth.utils.geolift_helpers.marketselect.realize import realize_design
    rng = np.random.default_rng(0)
    T = 22
    cols = {u: np.arange(T) * 0.3 + rng.normal(scale=0.5, size=T) + i
            for i, u in enumerate("ABCDE")}
    Ywide = pd.DataFrame(cols, index=pd.Index(range(T), name="time"))
    rep = realize_design(Ywide, frozenset({"A", "B"}), pre_periods=18,
                         how="sum", augment="ridge", ns=40, seed=0,
                         fixed_effects=True, cpic=7.5)
    ss = rep.weights.summary_stats
    assert ss["cpic"] == 7.5
    # cost = cpic * summed incremental over the post window
    incr = float(np.sum(rep.time_series.estimated_gap[18:]))
    assert ss["cost"] == pytest.approx(7.5 * incr, rel=1e-6)
