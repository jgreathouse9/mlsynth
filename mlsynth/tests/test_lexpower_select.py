"""Tests for the rebuilt power (lexpower) and recommendation (lexselect) stages."""
import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.lexpower import compute_mde, detectability_curve
from mlsynth.utils.fast_scm_helpers.lexselect import DesignMetrics, select_design


# ============================== lexpower ==============================

def test_null_calibrated_at_alpha():
    rng = np.random.default_rng(0)
    pool = [rng.standard_normal(50) for _ in range(5)]
    for alpha in (0.05, 0.10):
        r = compute_mde(pool, n_post=5, alpha=alpha, random_state=1)
        rej0 = r["curve"][0][1]            # power at tau=0 == size of the test
        assert abs(rej0 - alpha) < 0.03


def test_power_monotone_and_mde_finite():
    rng = np.random.default_rng(1)
    pool = [rng.standard_normal(60) for _ in range(4)]
    r = compute_mde(pool, n_post=6, random_state=2)
    ps = [p for _, p in r["curve"]]
    assert all(ps[i] <= ps[i + 1] + 0.05 for i in range(len(ps) - 1))
    assert r["feasible"] and np.isfinite(r["mde_sd"])


def test_mde_decreases_with_horizon():
    rng = np.random.default_rng(2)
    pool = [rng.standard_normal(80) for _ in range(5)]
    dc = detectability_curve(pool, [2, 4, 8, 16], random_state=3)
    cs = dc["curve_sd"]
    assert cs[2] >= cs[4] - 0.05 >= cs[8] - 0.10 >= cs[16] - 0.15


def test_scale_invariance_in_sd_units():
    rng = np.random.default_rng(3)
    pool = [rng.standard_normal(50) for _ in range(4)]
    r1 = compute_mde(pool, 4, random_state=5)
    r10 = compute_mde([10 * s for s in pool], 4, random_state=5)
    assert abs(r1["mde_sd"] - r10["mde_sd"]) < 1e-6
    assert abs(r10["mde_abs"] - 10 * r1["mde_abs"]) < 1e-4


def test_zero_mean_outcome_does_not_break():
    rng = np.random.default_rng(4)
    r = compute_mde([rng.standard_normal(40) - rng.standard_normal(40).mean()],
                    4, random_state=6)
    assert np.isfinite(r["c_alpha"]) and r["feasible"]


def test_autocorrelation_raises_mde():
    rng = np.random.default_rng(5)
    iid = [rng.standard_normal(80) for _ in range(5)]
    ar = []
    for _ in range(5):
        e = np.zeros(80)
        for t in range(1, 80):
            e[t] = 0.7 * e[t - 1] + rng.standard_normal()
        ar.append(e)
    r_iid = compute_mde(iid, 6, random_state=7)
    r_ar = compute_mde(ar, 6, random_state=7)
    assert r_ar["mde_sd"] > r_iid["mde_sd"]      # serial dependence -> harder


def test_huge_noise_marks_infeasible_not_crash():
    rng = np.random.default_rng(6)
    # tiny window + heavy noise: MDE may exceed the grid -> infeasible, no crash
    r = compute_mde([rng.standard_normal(6) * 5], n_post=2, max_sd=1.0,
                    random_state=8)
    assert r["feasible"] in (True, False)
    assert np.isfinite(r["c_alpha"])


# ========================= percentage MDE =========================

def test_mde_pct_matches_abs_over_baseline():
    rng = np.random.default_rng(10)
    pool = [rng.standard_normal(60) for _ in range(4)]   # sigma ~ 1
    baseline = 100.0                                       # >> sigma -> trustworthy
    r = compute_mde(pool, 6, baseline=baseline, random_state=2)
    assert r["feasible"] and np.isfinite(r["mde_pct"])
    assert r["baseline"] == baseline
    assert abs(r["mde_pct"] - 100.0 * r["mde_abs"] / baseline) < 1e-9


def test_mde_pct_nan_without_baseline():
    rng = np.random.default_rng(11)
    r = compute_mde([rng.standard_normal(60) for _ in range(4)], 6, random_state=2)
    assert np.isnan(r["mde_pct"]) and np.isnan(r["baseline"])


def test_mde_pct_guarded_for_near_zero_baseline():
    # baseline below the default floor (one residual SD ~ 1) -> percentage withheld
    rng = np.random.default_rng(12)
    pool = [rng.standard_normal(60) for _ in range(4)]
    r = compute_mde(pool, 6, baseline=0.2, random_state=2)
    assert np.isnan(r["mde_pct"])           # not a trustworthy level -> NaN
    assert np.isfinite(r["mde_abs"])        # absolute/SD metrics unaffected


def test_baseline_floor_override_forces_percentage():
    rng = np.random.default_rng(13)
    pool = [rng.standard_normal(60) for _ in range(4)]
    r = compute_mde(pool, 6, baseline=0.2, baseline_floor=1e-6, random_state=2)
    assert np.isfinite(r["mde_pct"])
    assert abs(r["mde_pct"] - 100.0 * r["mde_abs"] / 0.2) < 1e-9


def test_detectability_curve_emits_pct():
    rng = np.random.default_rng(14)
    pool = [rng.standard_normal(80) for _ in range(5)]
    level = np.full(40, 50.0)               # flat counterfactual level >> sigma
    dc = detectability_curve(pool, [4, 8], baseline_series=level, random_state=3)
    assert set(dc["curve_pct"]) == {4, 8}
    for w in (4, 8):
        d = dc["details"][w]
        if d["feasible"]:
            assert abs(dc["curve_pct"][w] - 100.0 * d["mde_abs"] / 50.0) < 1e-9
    # no baseline_series -> percentages are NaN, SD curve still present
    dc0 = detectability_curve(pool, [4, 8], random_state=3)
    assert all(np.isnan(v) for v in dc0["curve_pct"].values())
    assert np.isfinite(dc0["curve_sd"][8])


# ============================== lexselect ==============================

def _d(i, imb, mde, feas=True, stab=0.1, cost=0.0):
    return DesignMetrics(design_id=f"D{i}", indices=[i], imbalance=imb,
                         mde_sd=mde, mde_abs=mde, mde_feasible=feas,
                         stability=stab, total_cost=cost)


def test_lexicographic_balance_then_power():
    designs = [
        _d(1, 0.10, 2.0),              # best balance, weak power
        _d(2, 0.11, 1.0),              # within tol of best, strong power  <- winner
        _d(3, 0.105, 0.5, feas=False), # in gate but power infeasible -> excluded
        _d(4, 0.50, 0.2),              # outside balance gate despite best power
    ]
    rec = select_design(designs, imbalance_tol=0.25)   # gate ceil = 0.125
    assert rec.status == "OK"
    assert rec.winner.design_id == "D2"     # in-gate, smallest feasible MDE
    assert "D4" not in {d.design_id for d in rec.shortlist}  # gated out


def test_power_not_established_fallback():
    designs = [_d(1, 0.10, np.inf, feas=False), _d(2, 0.20, np.inf, feas=False)]
    rec = select_design(designs, imbalance_tol=0.25)
    assert rec.status == "POWER_NOT_ESTABLISHED"
    assert rec.winner.design_id == "D1"     # best balance, no crash


def test_pareto_frontier():
    designs = [_d(1, 0.1, 2.0), _d(2, 0.2, 1.0), _d(3, 0.3, 0.5), _d(4, 0.25, 2.5)]
    rec = select_design(designs, imbalance_tol=10.0)   # gate keeps all
    # D4 is dominated by D2 (lower imbalance AND lower mde); others are Pareto
    assert "D4" not in rec.pareto_ids
    assert {"D1", "D2", "D3"}.issubset(set(rec.pareto_ids))


def test_empty_input():
    rec = select_design([])
    assert rec.status == "EMPTY" and rec.winner is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
