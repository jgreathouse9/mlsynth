"""Inference for each treated unit of a MAREX design, and eq. (11)'s aggregate.

``unit_inference=True`` gives every treated unit its own synthetic control,
Abadie and Zhao's eq. (10): a simplex fit on the fitting periods from the
untreated units of its cluster. MAREX's inference then runs on each unit's
contrast ``u_j = y_j - Y_0 v_j``, and the family of treated units gets
Bonferroni-adjusted p-values and a joint test. Eq. (11)'s aggregate,
``sum_j w_j u_j``, is reported beside the shared contrast with its own
inference.

Levels: smoke, unit invariants, edge, failure.
"""
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.marex_helpers.inference import compute_inference
from mlsynth.utils.marex_helpers.unit_inference import joint_p_value
from mlsynth.utils.solvers.active_set import solve_simplex_qp


def _panel(N=18, T=60, seed=1, n_clusters=3):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    Y = (rng.normal(size=(N, 2)) @ F.T + 0.3 * rng.standard_normal((N, T))
         + 10.0 + rng.normal(0, 1, (N, 1)))
    rows = [{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]),
             "grp": f"c{j % n_clusters}"} for j in range(N) for t in range(T)]
    return pd.DataFrame(rows)


_CFG = dict(outcome="y", unitid="unit", time="time", T0=50, m_eq=2,
            cluster="grp", inference=True, blank_periods=20, T_post=10,
            backend="exact")
_TE = _CFG["T0"] - _CFG["blank_periods"]


def _fit(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, **_CFG)
    cfg.update(kw)
    return MAREX(cfg).fit()


def _units(res):
    return res.globres.unit_level.units


def _rows(res):
    """Row index of each unit label in Y_full."""
    labels = sorted(f"u{j:02d}" for j in range(res.globres.Y_full.shape[0]))
    return {u: i for i, u in enumerate(labels)}


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_every_treated_unit_gets_inference():
    res = _fit(unit_inference=True)
    treated = set(res.selected_units)
    assert set(_units(res)) == {str(u) for u in treated}
    for ui in _units(res).values():
        assert 0 < ui.inference.global_p_value <= 1
        assert np.all(np.isfinite(ui.inference.ci[_CFG["T0"]:]))
    assert 0 < res.globres.unit_level.joint_p_value <= 1


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_it_is_off_by_default():
    assert _fit().globres.unit_level is None


def test_a_units_control_is_its_clusters_untreated_units_fitted_on_the_fit_window():
    res = _fit(unit_inference=True)
    Y = res.globres.Y_full
    row = _rows(res)
    for c in res.clusters.values():
        idx = np.array([row[u] for u in c.members])
        pool = idx[np.asarray(c.treated_weights)[idx] <= 1e-8]
        for j in idx[np.asarray(c.treated_weights)[idx] > 1e-8]:
            label = f"u{j:02d}"
            ui = _units(res)[label]
            want = solve_simplex_qp(Y[pool, :_TE].T, Y[j, :_TE])
            got = np.array([ui.control_weights.get(f"u{i:02d}", 0.0) for i in pool])
            np.testing.assert_array_equal(got, want)
            assert set(ui.control_weights) <= {f"u{i:02d}" for i in pool}
            assert ui.cluster == c.label
            np.testing.assert_allclose(ui.synthetic_control, want @ Y[pool],
                                       rtol=1e-12)


def test_a_units_inference_is_compute_inference_on_its_contrast():
    res = _fit(unit_inference=True)
    Y = res.globres.Y_full
    row = _rows(res)
    label, ui = next(iter(_units(res).items()))
    j = row[label]
    want = compute_inference(Y[j], ui.synthetic_control, _CFG["T0"], _TE,
                             _CFG["blank_periods"], alpha=0.05,
                             random_state=42, T_post=_CFG["T_post"])
    np.testing.assert_array_equal(ui.inference.ci, want.ci)
    assert ui.inference.global_p_value == want.global_p_value
    np.testing.assert_array_equal(ui.inference.per_period_pvals,
                                  want.per_period_pvals)


def test_bonferroni_multiplies_by_the_family_size_and_caps_at_one():
    res = _fit(unit_inference=True)
    units = _units(res)
    m = len(units)
    assert m == 6                                  # three clusters, two each
    for ui in units.values():
        assert ui.p_bonferroni == min(1.0, m * ui.inference.global_p_value)


def test_eq11_is_the_weighted_sum_of_the_unit_contrasts():
    res = _fit(unit_inference=True)
    g = res.globres
    row = _rows(res)
    w = np.asarray(g.treated_weights_agg)
    control = sum(w[row[u]] * ui.synthetic_control for u, ui in _units(res).items())
    np.testing.assert_allclose(g.unit_level.eq11_control, control, rtol=1e-12)
    np.testing.assert_allclose(g.unit_level.eq11.fulltreated_effects,
                               g.synthetic_treated - control, rtol=1e-12,
                               atol=1e-12)


def test_under_inversion_eq11_shares_the_aggregate_band_for_the_total():
    """The inversion refits the treated series on the untreated units and never
    reads a control, so the shared contrast and eq. (11) get one band; only the
    point, each contrast's own total, differs."""
    res = _fit(unit_inference=True, cumulative_band=True,
               cumulative_method="inversion")
    shared = res.globres.inference.cumulative_band
    eq11 = res.globres.unit_level.eq11.cumulative_band
    assert (eq11.lower, eq11.upper) == (shared.lower, shared.upper)
    assert eq11.point == pytest.approx(
        float(res.globres.unit_level.eq11.treated_effects.sum()), rel=1e-12)
    assert eq11.point != shared.point


def test_the_shared_inference_is_untouched():
    plain = _fit().globres.inference
    with_units = _fit(unit_inference=True).globres.inference
    np.testing.assert_array_equal(plain.ci, with_units.ci)
    assert plain.global_p_value == with_units.global_p_value


def test_the_level_and_window_reach_every_unit():
    res = _fit(unit_inference=True, alpha=0.10, T_post=6)
    for ui in _units(res).values():
        assert ui.inference.alpha == 0.10
        assert ui.inference.treated_effects.size == 6
    assert res.globres.unit_level.eq11.alpha == 0.10


# --------------------------------------------------------------------------- #
# the joint test
# --------------------------------------------------------------------------- #
def _windows(m=3, n=9, h=3, seed=0):
    return np.random.default_rng(seed).normal(size=(m, n)) * [[1.0], [5.0], [0.2]][:m]


def _enumerated_joint(U, h):
    Z = np.abs(U) / np.mean(np.abs(U), axis=1, keepdims=True)
    n = U.shape[1]
    t_obs = np.max(np.mean(Z[:, n - h:], axis=1))
    combos = np.array(list(itertools.combinations(range(n), h)))
    t = np.max(np.mean(Z[:, combos], axis=2), axis=0)
    return float(np.mean(t >= t_obs - 1e-12 * (1 + t_obs)))


def test_the_joint_p_value_is_exact_when_the_arrangements_are_few():
    U = _windows()
    p = joint_p_value(U, 3, max_combinations=84, rng=np.random.default_rng(0))
    assert p == pytest.approx(_enumerated_joint(U, 3), abs=1e-15)
    assert p >= 1 / 84


def test_the_joint_p_value_by_draws_counts_the_observed_arrangement():
    U = _windows(n=30, h=8)
    p = joint_p_value(U, 8, max_combinations=200, rng=np.random.default_rng(4))
    rng = np.random.default_rng(4)
    pi = np.array([rng.choice(30, size=8, replace=False) for _ in range(200)])
    Z = np.abs(U) / np.mean(np.abs(U), axis=1, keepdims=True)
    t_obs = np.max(np.mean(Z[:, 22:], axis=1))
    t = np.max(np.mean(Z[:, pi], axis=2), axis=0)
    k = int(np.sum(t >= t_obs - 1e-12 * (1 + t_obs)))
    assert p == pytest.approx((1 + k) / 201, abs=1e-15)


def test_the_joint_statistic_ignores_each_units_scale():
    U = _windows()
    a = joint_p_value(U, 3, max_combinations=84, rng=np.random.default_rng(0))
    b = joint_p_value(U * np.array([[1e3], [1e-2], [7.0]]), 3,
                      max_combinations=84, rng=np.random.default_rng(0))
    assert a == b


def test_a_unit_with_no_variation_does_not_break_the_joint_test():
    U = _windows()
    U[1] = 0.0
    p = joint_p_value(U, 3, max_combinations=84, rng=np.random.default_rng(0))
    assert 1 / 84 <= p <= 1


def test_the_joint_p_value_through_fit_matches_the_contrasts():
    res = _fit(unit_inference=True)
    T0, Tb, H = _CFG["T0"], _CFG["blank_periods"], _CFG["T_post"]
    U = np.array([np.r_[ui.inference.placebo_effects, ui.inference.treated_effects]
                  for ui in _units(res).values()])
    want = joint_p_value(U, H, max_combinations=1000,
                         rng=np.random.default_rng(42))
    assert res.globres.unit_level.joint_p_value == want


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_a_cluster_with_one_control_gives_its_units_that_control():
    df = _panel(N=6, n_clusters=2)                  # three per cluster
    res = _fit(df, m_eq=2, unit_inference=True)
    for ui in _units(res).values():
        assert list(ui.control_weights.values()) == [1.0]


def test_no_blank_window_gives_no_unit_inference():
    res = _fit(unit_inference=True, blank_periods=0)
    assert res.globres.unit_level is None


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_unit_inference_needs_inference():
    with pytest.raises(MlsynthConfigError, match="unit_inference"):
        _fit(unit_inference=True, inference=False)


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_unit_inference_must_be_a_bool(bad):
    with pytest.raises(MlsynthConfigError, match="unit_inference must be"):
        _fit(unit_inference=bad)


def test_the_joint_test_refuses_a_post_block_longer_than_the_window():
    with pytest.raises(MlsynthDataError, match="T_post"):
        joint_p_value(_windows(), 10, max_combinations=10,
                      rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# the blank-period bias check
# --------------------------------------------------------------------------- #
from scipy import stats

from mlsynth.utils.marex_helpers.unit_inference import (blank_check,
                                                        unit_level_inference)
from mlsynth.utils.post_fit import _ar1_rho, _effective_n, _variance_inflation


def _one_cluster(target, donors, T0=40, Tb=16, H=6):
    """A unit-level result for one treated unit against the given donors."""
    Y = np.vstack([target, donors])
    w = np.zeros(Y.shape[0])
    w[0] = 1.0
    return unit_level_inference(
        Y, [("c0", np.arange(Y.shape[0]), w)], w,
        [f"u{i}" for i in range(Y.shape[0])], T0=T0, TcE=T0 - Tb, Tb=Tb,
        alpha=0.1, max_combinations=200, random_state=0, T_post=None
    ).units["u0"]


def test_the_check_is_the_blank_mean_with_an_ar1_standard_error():
    res = _fit(unit_inference=True)
    Y = res.globres.Y_full
    row = _rows(res)
    Tb = _CFG["blank_periods"]
    for label, ui in _units(res).items():
        b = ui.inference.placebo_effects
        chk = ui.blank_check
        rho = _ar1_rho(b)
        se = float(np.std(b, ddof=1) * np.sqrt(_variance_inflation(b.size, rho)))
        assert chk.n == Tb
        assert chk.mean == pytest.approx(float(b.mean()), rel=1e-12)
        assert chk.rho == pytest.approx(rho, rel=1e-12)
        assert chk.se == pytest.approx(se, rel=1e-12)
        assert chk.t == pytest.approx(chk.mean / chk.se, rel=1e-12)
        df = _effective_n(b.size, rho) - 1.0
        assert chk.p_value == pytest.approx(2 * stats.t.sf(abs(chk.t), df), rel=1e-9)
        level = np.mean(np.abs(Y[row[label], _TE:_CFG["T0"]]))
        assert chk.relative == pytest.approx(chk.mean / level, rel=1e-12)


def test_a_unit_outside_its_donors_hull_is_flagged():
    """Donor levels 10 to 20 and a unit at 30: no blend reaches it."""
    rng = np.random.default_rng(0)
    T = 46
    common = rng.normal(0, 1, T).cumsum() * 0.2
    donors = np.array([lvl + common + 0.3 * rng.standard_normal(T)
                       for lvl in np.linspace(10, 20, 6)])
    outside = 30 + common + 0.3 * rng.standard_normal(T)
    chk = _one_cluster(outside, donors).blank_check
    assert chk.mean > 5 and chk.t > 10 and chk.p_value < 1e-6
    assert chk.relative == pytest.approx(chk.mean / np.mean(np.abs(outside[24:40])))


def test_a_unit_inside_its_donors_hull_is_not_flagged():
    rng = np.random.default_rng(1)
    T = 46
    common = rng.normal(0, 1, T).cumsum() * 0.2
    donors = np.array([lvl + common + 0.3 * rng.standard_normal(T)
                       for lvl in np.linspace(10, 20, 6)])
    inside = 15 + common + 0.3 * rng.standard_normal(T)
    chk = _one_cluster(inside, donors).blank_check
    assert abs(chk.t) < 3 and chk.p_value > 0.01


def test_one_blank_period_leaves_the_spread_undefined():
    chk = blank_check(np.array([0.7]), np.array([10.0]))
    assert chk.mean == 0.7 and chk.n == 1
    assert np.isnan(chk.se) and np.isnan(chk.t) and np.isnan(chk.p_value)


def test_a_constant_offset_with_no_noise_is_certain():
    chk = blank_check(np.full(8, 2.0), np.full(8, 10.0))
    assert chk.se == 0.0 and chk.t == np.inf and chk.p_value == 0.0
    assert chk.relative == pytest.approx(0.2)
    below = blank_check(np.full(8, -2.0), np.full(8, 10.0))
    assert below.t == -np.inf and below.p_value == 0.0


def test_the_share_is_of_the_size_of_the_level():
    """A negative outcome keeps the bias's sign in the share."""
    chk = blank_check(np.array([1.0, 2.0, 3.0]), np.full(3, -10.0))
    assert chk.relative == pytest.approx(0.2)


def test_an_exact_fit_is_not_flagged():
    chk = blank_check(np.zeros(8), np.full(8, 10.0))
    assert chk.t == 0.0 and chk.p_value == 1.0


def test_a_zero_level_leaves_the_share_undefined():
    chk = blank_check(np.array([0.1, -0.2, 0.3]), np.zeros(3))
    assert np.isnan(chk.relative)
