import pytest
import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.marketselect.helpers.aggregate import (
    compute_power,
    compute_mde,
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
                         "placebo_mean_effect": es * 10, "scaled_l2": 0.3,
                         "pre_rmspe": 1.0})
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
            "placebo_mean_effect", "scaled_l2", "pre_rmspe"]
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
