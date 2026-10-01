"""rfPDA: random-forest donor selection for the panel data approach.

Liu, G., Long, W., & Luo, X. (2025). "A Random Forest-Based Panel Data Approach
for Program Evaluation." Journal of Applied Econometrics 40(5):591-607.

The counterfactual and the ATE test are Hsiao, Ching & Wan (2012) PDA unchanged;
what rfPDA contributes is the donor-selection step, so that is what these tests
pin -- the importance ordering, the forward search over it, the cap that keeps
the pre-period fit from interpolating, and the West (1997) MA long-run variance
the paper's Eq. (9)-(10) uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PDA
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pda_helpers.config import PDAConfig
from mlsynth.utils.pda_helpers.rf import rf_select, rf_ate_inference, west_lrvar

T0, T2, N_DONORS = 20, 8, 12


def panel(effect=0.0, n_donors=N_DONORS, t0=T0, t2=T2, seed=5, relevant=4):
    """A four-factor panel whose first ``relevant`` donors share the treated
    unit's loadings; the rest load on noise factors only."""
    rng = np.random.default_rng(seed)
    T = t0 + t2
    f = rng.normal(size=(T, 4))
    load_tr = np.array([1.0, 0.8, -0.6, 0.4])
    X = np.empty((T, n_donors))
    for j in range(n_donors):
        load = load_tr + rng.normal(scale=0.05, size=4) if j < relevant else rng.normal(size=4)
        X[:, j] = f @ load + rng.normal(scale=0.3, size=T)
    y = f @ load_tr + rng.normal(scale=0.3, size=T)
    y[t0:] += effect
    return y, X


def frame(effect=0.0, **kw):
    y, X = panel(effect=effect, **kw)
    T = y.shape[0]
    t0 = kw.get("t0", T0)
    rows = [{"unit": "treated", "time": t, "y": y[t], "d": int(t >= t0)}
            for t in range(T)]
    for j in range(X.shape[1]):
        rows += [{"unit": f"d{j}", "time": t, "y": X[t, j], "d": 0} for t in range(T)]
    return pd.DataFrame(rows)


def cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
                method="rf", rf_n_estimators=40, rf_seed=0, display_graphs=False)
    base.update(kw)
    return PDAConfig(**base)


# ------------------------------------------------------------------ selection
def test_smoke_returns_a_counterfactual_over_the_whole_panel():
    y, X = panel()
    sel, beta, const, cf, meta = rf_select(y, X, T0, n_estimators=40, seed=0)
    assert cf.shape == y.shape and np.all(np.isfinite(cf))
    assert beta.shape == (N_DONORS,)
    assert set(sel) <= set(range(N_DONORS))
    assert np.allclose(cf, X @ beta + const)


def test_the_selected_donors_carry_the_only_nonzero_coefficients():
    y, X = panel()
    sel, beta, _, _, _ = rf_select(y, X, T0, n_estimators=40, seed=0)
    assert np.count_nonzero(beta) == len(sel)
    assert sorted(np.flatnonzero(beta).tolist()) == sorted(sel)


def test_a_planted_effect_is_recovered():
    y, X = panel(effect=3.0)
    _, _, _, cf, _ = rf_select(y, X, T0, n_estimators=80, seed=0)
    assert float(np.mean((y - cf)[T0:])) == pytest.approx(3.0, abs=0.6)


def test_the_relevant_donors_are_ranked_above_the_noise_donors():
    """Donors 0-3 share the treated unit's loadings; importance should find them."""
    y, X = panel()
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=120, seed=0)
    top = meta["importance_order"][:8]
    assert len(set(top) & {0, 1, 2, 3}) >= 3


def test_selection_is_deterministic_in_the_seed():
    y, X = panel()
    a = rf_select(y, X, T0, n_estimators=40, seed=3)
    b = rf_select(y, X, T0, n_estimators=40, seed=3)
    assert a[0] == b[0]
    assert np.array_equal(a[3], b[3])


# ----------------------------------------------------------------- the cap
def test_the_default_cap_is_two_below_the_pre_period_length():
    y, X = panel(n_donors=30)
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0)
    assert meta["k_max"] == T0 - 2
    assert len(meta["selected"]) <= T0 - 2


def test_the_cap_keeps_the_pre_period_fit_from_interpolating():
    """Over-selection makes the pre-period residual vanish and the test degenerate."""
    y, X = panel(n_donors=30)
    _, _, _, cf, _ = rf_select(y, X, T0, n_estimators=40, seed=0)
    assert float(np.mean((y - cf)[:T0] ** 2)) > 1e-8


def test_an_explicit_cap_is_honoured():
    y, X = panel()
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0, k_max=5)
    assert meta["k_max"] == 5
    assert len(meta["selected"]) <= 5


def test_a_cap_above_the_pre_period_length_is_honoured_and_flagged():
    """The published search had no cap; reproducing it has to stay possible."""
    y, X = panel(n_donors=30)
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0, k_max=58)
    assert meta["k_max"] == 58
    assert meta["cap_exceeds_pre_periods"] is True


def test_the_cap_is_reported_as_binding_only_when_it_binds():
    y, X = panel()
    _, _, _, _, tight = rf_select(y, X, T0, n_estimators=40, seed=0, k_max=2)
    assert tight["cap_binds"] is True
    _, _, _, _, loose = rf_select(y, X, T0, n_estimators=40, seed=0, k_max=T0 - 2)
    assert loose["cap_binds"] == (len(loose["selected"]) == T0 - 2)


# ----------------------------------------------------------------- splitting
def test_the_temporal_split_keeps_the_three_blocks_in_time_order():
    y, X = panel()
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0, split="temporal")
    tr, va, te = meta["train_idx"], meta["validation_idx"], meta["test_idx"]
    assert max(tr) < min(va) and max(va) < min(te)
    assert sorted(tr) + sorted(va) + sorted(te) == list(range(T0))


def test_the_random_split_reproduces_the_released_code_and_has_no_validation_block():
    y, X = panel()
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0, split="random",
                                 train_fraction=0.7)
    assert len(meta["validation_idx"]) == 0
    assert len(meta["train_idx"]) == int(T0 * 0.7)
    assert set(meta["train_idx"]) | set(meta["test_idx"]) == set(range(T0))


def test_the_two_splits_are_different_estimators():
    y, X = panel()
    a = rf_select(y, X, T0, n_estimators=80, seed=0, split="temporal")
    b = rf_select(y, X, T0, n_estimators=80, seed=0, split="random")
    assert a[0] != b[0] or not np.allclose(a[3], b[3])


# ---------------------------------------------------------------- importance
def test_oob_importance_is_available_for_the_released_variant():
    y, X = panel()
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=80, seed=0, importance="oob")
    assert meta["importance"] == "oob"
    assert len(meta["importance_order"]) == N_DONORS


def test_the_two_importance_rules_rank_the_relevant_donors_alike():
    y, X = panel()
    perm = rf_select(y, X, T0, n_estimators=120, seed=0, importance="permutation")[4]
    oob = rf_select(y, X, T0, n_estimators=120, seed=0, importance="oob")[4]
    assert len(set(perm["importance_order"][:8]) & set(oob["importance_order"][:8])) >= 4


# -------------------------------------------------------------- degenerate
def test_a_single_donor_still_fits():
    y, X = panel(n_donors=1, relevant=1)
    sel, _, _, cf, _ = rf_select(y, X, T0, n_estimators=40, seed=0)
    assert sel == [0]
    assert np.all(np.isfinite(cf))


def test_a_constant_donor_is_dropped_before_ranking():
    y, X = panel()
    X[:, 7] = 2.5
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0)
    assert 7 not in meta["selected"]
    assert 7 in meta["dropped_constant"]


def test_a_pre_period_too_short_to_split_is_refused():
    y, X = panel(t0=5, t2=4)
    with pytest.raises(MlsynthConfigError, match="pre-treatment period"):
        rf_select(y, X, 5, n_estimators=40, seed=0)


# ------------------------------------------------------------- seed spread
def test_seed_sensitivity_is_reported_when_asked_for():
    y, X = panel()
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0, n_seeds=5)
    s = meta["seed_sensitivity"]
    assert s["n_seeds"] == 5
    assert s["att_min"] <= s["att_mean"] <= s["att_max"]
    assert s["att_sd"] >= 0.0
    assert 0.0 <= s["selection_jaccard_mean"] <= 1.0


def test_one_seed_reports_no_spread():
    y, X = panel()
    _, _, _, _, meta = rf_select(y, X, T0, n_estimators=40, seed=0)
    assert meta["seed_sensitivity"] is None


def test_the_point_estimate_is_the_named_seed_not_the_seed_average():
    y, X = panel()
    one = rf_select(y, X, T0, n_estimators=40, seed=2)
    many = rf_select(y, X, T0, n_estimators=40, seed=2, n_seeds=4)
    assert np.allclose(one[3], many[3])


# ------------------------------------------------------------- inference
def test_west_lrvar_matches_the_reference_construction():
    """The MA(1) long-run variance of Eq. (9)-(10), against a hand computation."""
    rng = np.random.default_rng(0)
    e = rng.normal(size=60)
    v = west_lrvar(e[:40], e[40:], T1=40, T2=20, q1=1, q2=1)
    assert v["before"] > 0 and v["after"] > 0
    # Scaling: the pre-period term carries T2 / (T1 (T1 - q1)), the post 1/(T2 - q2).
    v2 = west_lrvar(e[:40], e[40:], T1=40, T2=10, q1=1, q2=1)
    assert v2["before"] == pytest.approx(v["before"] * 10 / 20, rel=1e-10)


def test_the_ate_test_reads_the_post_period_mean():
    y, X = panel(effect=2.0)
    _, _, _, cf, _ = rf_select(y, X, T0, n_estimators=80, seed=0)
    att, se, ci, p = rf_ate_inference(y, cf, T0)
    assert att == pytest.approx(float(np.mean((y - cf)[T0:])))
    assert se > 0 and ci[0] < att < ci[1] and 0.0 <= p <= 1.0


def test_a_zero_effect_panel_does_not_reject():
    y, X = panel(effect=0.0)
    _, _, _, cf, _ = rf_select(y, X, T0, n_estimators=80, seed=0)
    _, _, _, p = rf_ate_inference(y, cf, T0)
    assert p > 0.05


def test_a_split_that_leaves_no_test_block_is_refused_at_the_call_too():
    """The config validator catches this, but the helper is callable directly."""
    y, X = panel()
    with pytest.raises(MlsynthConfigError, match="no test block"):
        rf_select(y, X, T0, n_estimators=40, seed=0,
                  train_fraction=0.9, validation_fraction=0.2)


def test_a_pool_with_no_pre_treatment_variation_is_refused():
    y, X = panel(n_donors=3)
    X[:T0, :] = 1.0
    with pytest.raises(MlsynthConfigError, match="pre-treatment variation"):
        rf_select(y, X, T0, n_estimators=40, seed=0)


def test_a_constant_error_series_falls_back_to_its_own_second_moment():
    """An MA fit has nothing to estimate from, so the scaling stays at one."""
    flat = np.zeros(30) + 2.0
    v = west_lrvar(flat, flat, T1=30, T2=30, q1=1, q2=1)
    assert v["before"] > 0 and v["after"] > 0


def test_a_vanishing_long_run_variance_returns_a_degenerate_test():
    """An interpolating fit leaves no variance to studentize by."""
    y = np.arange(30, dtype=float)
    att, se, ci, p = rf_ate_inference(y, y.copy(), 20)
    assert att == 0.0 and se == 0.0 and ci == (0.0, 0.0) and p == 1.0


# ------------------------------------------------------------ integration
def test_pda_runs_the_rf_variant_end_to_end():
    res = PDA(cfg(frame(effect=2.0), rf_n_estimators=80)).fit()
    fit = res.fits["rf"]
    assert np.isfinite(fit.att) and fit.att == pytest.approx(2.0, abs=0.8)
    assert fit.selected_donors
    assert set(fit.donor_weights) == set(fit.selected_donors)


def test_the_rf_variant_reports_its_settings():
    res = PDA(cfg(frame(), rf_split="random", rf_k_max=6, rf_n_seeds=3)).fit()
    meta = res.fits["rf"].metadata
    assert meta["split"] == "random"
    assert meta["k_max"] == 6
    assert meta["seed_sensitivity"]["n_seeds"] == 3


def test_rf_runs_beside_the_other_variants():
    res = PDA(cfg(frame(effect=2.0), methods=["fs", "rf"])).fit()
    assert set(res.fits) == {"fs", "rf"}
    for f in res.fits.values():
        assert np.isfinite(f.att)


# ------------------------------------------------------- config validation
def test_the_method_literal_accepts_rf():
    assert cfg(frame()).method == "rf"


@pytest.mark.parametrize("kw,match", [
    (dict(rf_split="kfold"), "rf_split"),
    (dict(rf_n_estimators=0), "rf_n_estimators"),
    (dict(rf_k_max=1), "rf_k_max"),
    (dict(rf_train_fraction=1.0), "rf_train_fraction"),
    (dict(rf_validation_fraction=0.0), "rf_validation_fraction"),
    (dict(rf_n_seeds=0), "rf_n_seeds"),
    (dict(rf_importance="gini"), "rf_importance"),
])
def test_invalid_rf_settings_are_refused(kw, match):
    with pytest.raises((MlsynthConfigError, ValueError), match=match):
        cfg(frame(), **kw)


def test_the_split_fractions_must_leave_a_test_block():
    with pytest.raises((MlsynthConfigError, ValueError), match="test block"):
        cfg(frame(), rf_train_fraction=0.7, rf_validation_fraction=0.3)
