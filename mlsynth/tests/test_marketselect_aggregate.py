import pytest
import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.marketselect.helpers.aggregate import (
    compute_power,
    compute_mde,
    compute_rank,
)


def _make_cube():
    """Cube for one candidate / duration, 2 sims, 3 effect sizes."""
    cand = frozenset(["A", "B"])
    pvals = {0.0: [0.6, 0.4], 0.5: [0.02, 0.3], 1.0: [0.01, 0.04]}
    rows = []
    for es, ps in pvals.items():
        for sim, p in zip([1, 2], ps):
            rows.append({"candidate": cand, "duration": 4, "sim": sim,
                         "effect_size": es, "p_value": p,
                         "placebo_mean_effect": es * 10, "detected_lift": es,
                         "scaled_l2": 0.3, "pre_rmspe": 1.0})
    return pd.DataFrame(rows)


# === compute_power ===

def test_compute_power_detection_rate():
    out = compute_power(_make_cube(), alpha=0.1)
    p = out.set_index("effect_size")["power"]
    assert p[0.0] == 0.0          # 0.6,0.4 -> none < 0.1
    assert p[0.5] == 0.5          # 0.02<0.1, 0.3 not
    assert p[1.0] == 1.0          # 0.01,0.04 both < 0.1


def test_compute_power_one_row_per_group_and_avg_metrics():
    out = compute_power(_make_cube(), alpha=0.1)
    assert len(out) == 3
    assert (out["scaled_l2"] == 0.3).all()
    assert (out["pre_rmspe"] == 1.0).all()
    assert out.set_index("effect_size")["placebo_mean_effect"][0.5] == 5.0


def test_compute_power_alpha_threshold():
    out = compute_power(_make_cube(), alpha=0.015)
    p = out.set_index("effect_size")["power"]
    assert p[1.0] == 0.5          # only 0.01 < 0.015
    assert p[0.5] == 0.0


def test_compute_power_empty():
    cols = ["candidate", "duration", "effect_size", "p_value",
            "placebo_mean_effect", "detected_lift", "scaled_l2", "pre_rmspe"]
    out = compute_power(pd.DataFrame(columns=cols))
    assert out.empty


# === compute_mde ===

def _power_table(power_by_es):
    rows = [{"candidate": frozenset(["A"]), "duration": 4,
             "effect_size": es, "power": pw} for es, pw in power_by_es.items()]
    return pd.DataFrame(rows)


def test_compute_mde_smallest_positive():
    pt = _power_table({0.0: 0.05, 0.1: 0.5, 0.2: 0.85, 0.3: 0.95})
    assert compute_mde(pt, power_threshold=0.8)["mde"].iloc[0] == pytest.approx(0.2)


def test_compute_mde_prefers_smaller_magnitude_negative():
    pt = _power_table({-0.1: 0.9, 0.3: 0.9, 0.1: 0.2})
    assert compute_mde(pt, power_threshold=0.8)["mde"].iloc[0] == pytest.approx(-0.1)


def test_compute_mde_tie_prefers_positive():
    pt = _power_table({-0.1: 0.9, 0.1: 0.9})
    assert compute_mde(pt, power_threshold=0.8)["mde"].iloc[0] == pytest.approx(0.1)


def test_compute_mde_none_detectable_is_nan():
    pt = _power_table({0.1: 0.2, 0.2: 0.5})
    assert np.isnan(compute_mde(pt, power_threshold=0.8)["mde"].iloc[0])


def test_compute_mde_empty():
    out = compute_mde(pd.DataFrame(columns=["candidate", "duration", "effect_size", "power"]))
    assert out.empty


# === compute_rank ===

def _rank_power_table():
    X, Y = frozenset(["A", "B"]), frozenset(["C", "D"])
    rows = [
        # X: MDE 0.1, low power-at-MDE, perfect recovery -> best on all three
        {"candidate": X, "duration": 4, "effect_size": 0.0, "power": 0.0, "detected_lift": 0.0, "scaled_l2": 0.2, "pre_rmspe": 1.0, "placebo_mean_effect": 0.0},
        {"candidate": X, "duration": 4, "effect_size": 0.1, "power": 0.81, "detected_lift": 0.1, "scaled_l2": 0.2, "pre_rmspe": 1.0, "placebo_mean_effect": 1.0},
        {"candidate": X, "duration": 4, "effect_size": 0.3, "power": 0.99, "detected_lift": 0.3, "scaled_l2": 0.2, "pre_rmspe": 1.0, "placebo_mean_effect": 3.0},
        # Y: MDE 0.3, higher power, recovery error 0.05
        {"candidate": Y, "duration": 4, "effect_size": 0.0, "power": 0.0, "detected_lift": 0.0, "scaled_l2": 0.5, "pre_rmspe": 2.0, "placebo_mean_effect": 0.0},
        {"candidate": Y, "duration": 4, "effect_size": 0.1, "power": 0.5, "detected_lift": 0.08, "scaled_l2": 0.5, "pre_rmspe": 2.0, "placebo_mean_effect": 0.8},
        {"candidate": Y, "duration": 4, "effect_size": 0.3, "power": 0.95, "detected_lift": 0.25, "scaled_l2": 0.5, "pre_rmspe": 2.0, "placebo_mean_effect": 2.5},
    ]
    return pd.DataFrame(rows)


def test_compute_rank_best_candidate_first():
    out = compute_rank(_rank_power_table(), power_threshold=0.8)
    assert out.iloc[0]["candidate"] == frozenset(["A", "B"])
    assert out.iloc[0]["rank"] == 1
    assert out.iloc[-1]["candidate"] == frozenset(["C", "D"])


def _row(out, cand):
    return out[out["candidate"].apply(lambda c: c == cand)].iloc[0]


def test_compute_rank_mde_and_abszero():
    out = compute_rank(_rank_power_table(), power_threshold=0.8)
    X, Y = frozenset(["A", "B"]), frozenset(["C", "D"])
    assert _row(out, X)["mde"] == pytest.approx(0.1)
    assert _row(out, Y)["mde"] == pytest.approx(0.3)
    assert _row(out, X)["abs_lift_in_zero"] == pytest.approx(0.0)
    assert _row(out, Y)["abs_lift_in_zero"] == pytest.approx(0.05)


def test_compute_rank_components_exclude_scaled_l2():
    out = compute_rank(_rank_power_table(), power_threshold=0.8)
    rank_cols = {c for c in out.columns if c.startswith("rank")}
    assert rank_cols == {"rank_mde", "rank_pvalue", "rank_abszero", "rank"}
    assert "scaled_l2" in out.columns and "pre_rmspe" in out.columns   # reported only


def test_compute_rank_pvalue_is_ascending_in_power():
    """The GeoLift quirk: lower power-at-MDE -> lower (better) rank_pvalue."""
    out = compute_rank(_rank_power_table(), power_threshold=0.8)
    X, Y = frozenset(["A", "B"]), frozenset(["C", "D"])
    assert _row(out, X)["power"] < _row(out, Y)["power"]
    assert _row(out, X)["rank_pvalue"] < _row(out, Y)["rank_pvalue"]


def test_compute_rank_drops_nondetectable():
    Z = frozenset(["E"])
    extra = pd.DataFrame([
        {"candidate": Z, "duration": 4, "effect_size": 0.0, "power": 0.0, "detected_lift": 0.0, "scaled_l2": 0.9, "pre_rmspe": 3.0, "placebo_mean_effect": 0.0},
        {"candidate": Z, "duration": 4, "effect_size": 0.3, "power": 0.3, "detected_lift": 0.1, "scaled_l2": 0.9, "pre_rmspe": 3.0, "placebo_mean_effect": 1.0},
    ])
    out = compute_rank(pd.concat([_rank_power_table(), extra], ignore_index=True), power_threshold=0.8)
    assert Z not in set(out["candidate"])


def test_compute_rank_empty():
    assert compute_rank(pd.DataFrame(columns=_POWER_COLUMNS_FOR_TEST)).empty


_POWER_COLUMNS_FOR_TEST = ["candidate", "duration", "effect_size", "power",
                          "placebo_mean_effect", "detected_lift", "scaled_l2", "pre_rmspe"]


def test_compute_rank_all_nondetectable_returns_empty():
    """No candidate clears the power threshold -> empty ranked frame."""
    weak = _rank_power_table().assign(power=0.1)   # nothing detectable
    out = compute_rank(weak, power_threshold=0.8)
    assert out.empty
    assert "rank" in out.columns
