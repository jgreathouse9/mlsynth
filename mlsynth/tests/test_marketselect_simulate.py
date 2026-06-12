import pytest
import numpy as np

from mlsynth.utils.geolift_helpers.marketselect.helpers.simulate import (
    inject_effect,
    simulate_lookback,
)
from mlsynth.exceptions import MlsynthConfigError


# === inject_effect ===

def test_inject_effect_es_zero_is_copy():
    t = np.array([1.0, 2, 3, 4, 5])
    out = inject_effect(t, 3, 4, 0.0)
    np.testing.assert_allclose(out, t)
    assert out is not t


def test_inject_effect_scales_post_block():
    t = np.array([1.0, 2, 3, 4, 5])
    out = inject_effect(t, 3, 4, 0.5)            # indices 3,4 *= 1.5
    np.testing.assert_allclose(out, [1, 2, 3, 6, 7.5])


def test_inject_effect_does_not_mutate_input():
    t = np.array([1.0, 2, 3, 4, 5])
    _ = inject_effect(t, 0, 1, 1.0)
    np.testing.assert_allclose(t, [1, 2, 3, 4, 5])


def test_inject_effect_negative_effect():
    t = np.array([10.0, 10, 10, 10])
    out = inject_effect(t, 2, 3, -0.2)           # *= 0.8
    np.testing.assert_allclose(out, [10, 10, 8, 8])


@pytest.mark.parametrize("start,end", [(2, 1), (0, 4), (-1, 2)])
def test_inject_effect_bad_range_raises(start, end):
    t = np.array([1.0, 2, 3, 4])
    with pytest.raises(MlsynthConfigError):
        inject_effect(t, start, end, 0.5)


# === simulate_lookback ===

def _sim_panel(n=20, J=3, seed=1):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(n, J)) + np.arange(n)[:, None] * 0.3
    w = np.array([0.5, 0.3, 0.2])
    treated = Y0 @ w + rng.normal(scale=0.05, size=n)   # near-perfectly synthesizable
    return treated, Y0


def test_simulate_lookback_smoke():
    treated, Y0 = _sim_panel()
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 0.5],
                             augment="ridge", ns=30, seed=0)
    assert len(rows) == 2
    keys = {"sim", "duration", "effect_size", "p_value", "placebo_mean_effect", "detected_lift", "scaled_l2", "pre_rmspe"}
    assert all(keys <= set(r) for r in rows)
    assert all(0.0 <= r["p_value"] <= 1.0 for r in rows)
    assert rows[0]["effect_size"] == 0.0 and rows[1]["effect_size"] == 0.5
    assert rows[0]["sim"] == 1 and rows[0]["duration"] == 4


def test_simulate_lookback_larger_effect_lowers_pvalue():
    treated, Y0 = _sim_panel()
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 3.0],
                             augment="ridge", ns=200, seed=0)
    assert rows[1]["p_value"] < rows[0]["p_value"]


def test_simulate_lookback_placebo_effect_grows_with_es():
    treated, Y0 = _sim_panel()
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 0.5],
                             augment="ridge", ns=30, seed=0)
    assert rows[1]["placebo_mean_effect"] > rows[0]["placebo_mean_effect"]


def test_simulate_lookback_metrics_constant_across_effects():
    """One fit -> scaled_l2 / pre_rmspe identical for every effect size."""
    treated, Y0 = _sim_panel()
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 0.5, 1.0],
                             augment="ridge", ns=20, seed=0)
    assert len({r["scaled_l2"] for r in rows}) == 1
    assert len({r["pre_rmspe"] for r in rows}) == 1


def test_simulate_lookback_off_start_window_raises():
    treated, Y0 = _sim_panel(n=6)
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        simulate_lookback(treated, Y0, 6, 4, 3, [0.0], augment="ridge", ns=10)


def test_simulate_lookback_row_count_mismatch_raises():
    treated, Y0 = _sim_panel(n=20)
    with pytest.raises(MlsynthConfigError, match="n_periods"):
        simulate_lookback(treated, Y0, 19, 4, 1, [0.0], augment="ridge", ns=10)


# === optimization equivalence: CV-once == per-es refit ===

from mlsynth.utils.geolift_helpers.marketselect.helpers.windows import (
    lookback_pre_periods,
    lookback_treatment_window,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.fit import fit_augsynth_once
from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue


def test_simulate_lookback_cv_once_equals_per_es_refit():
    """The injection lands only on the post block, so the pre-period the CV sees
    is identical across effect sizes; re-CVing per es returns the same lambda and
    hence the identical conformal p-value. Proves the CV-once optimization exact."""
    treated, Y0 = _sim_panel()
    n_periods, duration, sim = 20, 4, 1
    effect_sizes = [0.0, 0.5, 2.0]
    ns, seed = 100, 0
    fast = simulate_lookback(treated, Y0, n_periods, duration, sim, effect_sizes,
                             augment="ridge", ns=ns, seed=seed)
    n_pre = lookback_pre_periods(n_periods, duration, sim)
    start, end = lookback_treatment_window(n_periods, duration, sim)
    t = np.asarray(treated, dtype=float)
    Y0a = np.asarray(Y0, dtype=float)
    for row, es in zip(fast, effect_sizes):
        refit = fit_augsynth_once(t[:n_pre], Y0a[:n_pre], augment="ridge")   # re-CV
        ti = inject_effect(t, start, end, es)
        p_naive = conformal_pvalue(ti, Y0a, n_pre, lambda_=refit.lambda_,
                                   q=1.0, ns=ns, seed=seed)
        assert row["p_value"] == pytest.approx(p_naive)


def test_simulate_lookback_simplex_path_runs():
    """augment=None (simplex point fit, lambda_=None -> ridge conformal) is valid."""
    treated, Y0 = _sim_panel()
    rows = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 1.0],
                             augment=None, ns=20, seed=0)
    assert len(rows) == 2
    assert all(0.0 <= r["p_value"] <= 1.0 for r in rows)


def test_simulate_lookback_block_conformal_seed_invariant():
    """conformal_type='block' is deterministic -> p-values don't depend on seed."""
    treated, Y0 = _sim_panel()
    a = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 1.0],
                          augment="ridge", ns=50, seed=0, conformal_type="block")
    b = simulate_lookback(treated, Y0, 20, 4, 1, [0.0, 1.0],
                          augment="ridge", ns=50, seed=777, conformal_type="block")
    assert [r["p_value"] for r in a] == [r["p_value"] for r in b]
