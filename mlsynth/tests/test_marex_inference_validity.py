"""MAREX's inference at the level it reports (#744).

Five defects, each pinned here:

* The pointwise band took ``np.quantile`` of the absolute blank-period effects,
  which interpolates below the sample maximum. Split conformal prediction calls
  for the ``ceil((Tb + 1)(1 - alpha))``-th order statistic, which exists only
  when ``Tb >= ceil(1 / alpha) - 1``; below that the band is unbounded.
* The default blank window, 30% of the pre-period, fell below that floor on
  any panel with ``T0 < 64``. The default now holds at least the floor of blank
  periods and at least ``max(2 r, 10)`` fitting periods, ``r`` the effective
  rank of the pre-period panel, and refuses a pre-period too short for both.
* The global p-value was a Monte Carlo mean that could reach zero. It is exact
  when the arrangements number at most ``max_combinations`` and ``(1 + k) /
  (1 + B)`` otherwise.
* The per-period p-values were ``k / Tb``, which can reach zero. They are
  ``(1 + k) / (Tb + 1)``.
* ``T_post`` was checked by the config and then ignored by the inference, and
  ``alpha`` set the MDE alone while the inference ran at 0.05.

Levels: smoke, unit invariants, edge, failure.
"""
import itertools
import math
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal.quantile import split_conformal_quantile
from mlsynth.utils.marex_helpers.inference import compute_inference
from mlsynth.utils.marex_helpers.windows import (default_blank_periods,
                                                 effective_rank,
                                                 min_blank_periods)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _factor_panel(N=30, T=40, r=2, scale=5.0, noise=0.3, seed=0, effects=True):
    """A units-by-periods array with ``r`` strong factors."""
    rng = np.random.default_rng(seed)
    Y = scale * rng.normal(size=(N, r)) @ rng.normal(size=(r, T))
    if effects:
        Y = Y + rng.normal(size=(N, 1)) * 10 + rng.normal(size=(1, T)) * 10
    return Y + noise * rng.standard_normal((N, T))


def _long(Y):
    N, T = Y.shape
    return pd.DataFrame([{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
                         for j in range(N) for t in range(T)])


def _contrast(Tb=20, T_post=8, TE=24, seed=1, effect=1.5, extra=0):
    """A synthetic treated and control pair with ``extra`` periods past the post."""
    rng = np.random.default_rng(seed)
    T = TE + Tb + T_post + extra
    base = np.cumsum(rng.normal(0, 0.3, T)) + 10.0
    y_t = base + rng.standard_normal(T)
    y_t[TE + Tb:] += effect
    return y_t, base, TE + Tb, TE, Tb


def _infer(Tb=20, T_post=8, TE=24, seed=1, alpha=0.05, **kw):
    y_t, y_c, T0, te, tb = _contrast(Tb, T_post, TE, seed)
    return compute_inference(y_t, y_c, T0, te, tb, alpha=alpha, random_state=0,
                             **kw)


# --------------------------------------------------------------------------- #
# the two floors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha,floor", [(0.05, 19), (0.10, 9), (0.01, 99),
                                         (0.20, 4), (0.30, 3), (0.50, 1)])
def test_the_blank_floor_is_the_fewest_periods_with_a_conformal_quantile(alpha,
                                                                         floor):
    assert min_blank_periods(alpha) == floor
    assert np.isfinite(split_conformal_quantile(np.arange(1.0, floor + 1), alpha))
    if floor > 1:
        assert np.isinf(split_conformal_quantile(np.arange(1.0, floor), alpha))


def test_the_effective_rank_of_an_exact_factor_panel_is_its_rank():
    assert effective_rank(_factor_panel(r=3, noise=0.0)) == 3


def test_the_effective_rank_finds_strong_factors_under_noise():
    assert effective_rank(_factor_panel(r=2, noise=0.3)) == 2


def test_unit_and_period_effects_are_not_factors():
    """A common trend is fitted by any weights that sum to one."""
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(20, 1)) * 10 + rng.normal(size=(1, 30)) * 10
    assert effective_rank(Y) == 0


def test_a_constant_panel_has_no_factors():
    assert effective_rank(np.full((10, 20), 3.0)) == 0


@pytest.mark.parametrize("shape", [(1, 30), (30, 1), (2, 3)])
def test_a_tiny_panel_gives_a_rank_without_error(shape):
    r = effective_rank(np.random.default_rng(0).normal(size=shape))
    assert 0 <= r <= max(min(shape) - 1, 0)


def test_noise_alone_on_few_units_does_not_count_every_dimension():
    """Eight noisy units leave seven dimensions after centring; two are factors."""
    assert effective_rank(_factor_panel(N=8, T=60, r=2, noise=0.3)) == 2


@pytest.mark.parametrize("T0,want", [(100, 30), (40, 19), (29, 19)])
def test_the_default_blank_window_holds_the_floor(T0, want):
    assert default_blank_periods(_factor_panel(T=T0, r=1), 0.05) == want


def test_a_larger_alpha_lowers_the_floor():
    assert default_blank_periods(_factor_panel(T=20, r=1), 0.10) == 9


def test_the_fitting_floor_caps_the_blank_window():
    """At alpha 0.5 the 30% target, 3 of 12, would leave 9 fitting periods."""
    assert default_blank_periods(_factor_panel(T=12, r=1), 0.5) == 2


def test_a_richer_panel_needs_more_fitting_periods():
    """Six factors need twelve fitting periods, so 31 pre-periods, not 29."""
    Y = _factor_panel(N=30, T=31, r=6, noise=0.05)
    assert effective_rank(Y) == 6
    assert default_blank_periods(Y, 0.05) == 19
    with pytest.raises(MlsynthConfigError):
        default_blank_periods(Y[:, :30], 0.05)


def test_a_pre_period_too_short_for_both_floors_is_refused():
    with pytest.raises(MlsynthConfigError) as err:
        default_blank_periods(_factor_panel(T=28, r=1), 0.05)
    msg = str(err.value)
    for part in ("19 blank periods", "10 fitting periods", "alpha=0.05",
                 "T0=28", "effective rank", "alpha=0.1", "blank_periods"):
        assert part in msg, part


# --------------------------------------------------------------------------- #
# the pointwise band
# --------------------------------------------------------------------------- #
def test_the_band_is_the_conformal_order_statistic():
    inf = _infer(Tb=40)
    b = np.sort(np.abs(inf.placebo_effects))
    k = math.ceil(41 * 0.95)
    half = inf.ci[-1, 1] - inf.treated_effects[-1]
    assert half == pytest.approx(b[k - 1], rel=1e-12)
    assert half > np.quantile(b, 0.95)            # the old, narrower reading


def test_at_the_floor_the_band_is_the_largest_blank_effect():
    inf = _infer(Tb=19)
    half = inf.ci[-1, 1] - inf.treated_effects[-1]
    assert half == pytest.approx(np.max(np.abs(inf.placebo_effects)), rel=1e-12)


def test_below_the_floor_the_band_is_unbounded_and_says_so():
    with pytest.warns(UserWarning, match="19 blank periods"):
        inf = _infer(Tb=12)
    T0 = 24 + 12
    assert np.all(inf.ci[T0:, 0] == -np.inf)
    assert np.all(inf.ci[T0:, 1] == np.inf)
    assert np.isfinite(inf.global_p_value)
    assert np.all(np.isfinite(inf.per_period_pvals))


def test_the_level_sets_the_floor():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inf = _infer(Tb=12, alpha=0.10)
    assert np.all(np.isfinite(inf.ci[36:]))


def test_no_blank_window_still_gives_no_band_and_no_warning():
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inf = compute_inference(rng.normal(size=12), rng.normal(size=12),
                                T0=10, TcE=10, Tb=0, random_state=0)
    assert np.isnan(inf.ci[10:]).all()


# --------------------------------------------------------------------------- #
# p-values
# --------------------------------------------------------------------------- #
def test_per_period_p_values_count_the_post_period_itself():
    inf = _infer(Tb=20)
    b = np.abs(inf.placebo_effects)
    want = [(1 + np.sum(b >= abs(e))) / 21 for e in inf.treated_effects]
    np.testing.assert_allclose(inf.per_period_pvals, want, rtol=0, atol=0)


def test_an_effect_beyond_every_placebo_has_p_one_over_tb_plus_one():
    y_t, y_c, T0, te, tb = _contrast(Tb=20)
    y_t[T0:] += 1e3
    inf = compute_inference(y_t, y_c, T0, te, tb, random_state=0)
    assert np.all(inf.per_period_pvals == 1 / 21)
    assert inf.global_p_value > 0


def _enumerated(inf, H):
    allx = np.abs(np.concatenate([inf.placebo_effects, inf.treated_effects]))
    combos = np.array(list(itertools.combinations(range(allx.size), H)))
    s = allx[combos].mean(axis=1)
    return float(np.mean(s >= np.mean(np.abs(inf.treated_effects)) - 1e-12))


def test_the_global_p_value_is_exact_when_the_arrangements_are_few():
    """Six blank and three post periods: 84 arrangements, all of them used."""
    y_t, y_c, T0, te, tb = _contrast(Tb=6, T_post=3)
    p = [compute_inference(y_t, y_c, T0, te, tb, random_state=s).global_p_value
         for s in (0, 1, 2)]
    assert p[0] == p[1] == p[2]
    inf = compute_inference(y_t, y_c, T0, te, tb, random_state=0)
    assert p[0] == pytest.approx(_enumerated(inf, 3), abs=1e-15)
    assert round(p[0] * 84) == pytest.approx(p[0] * 84, abs=1e-9)
    assert p[0] >= 1 / 84


def test_the_exact_branch_includes_its_boundary():
    y_t, y_c, T0, te, tb = _contrast(Tb=6, T_post=3)
    a = compute_inference(y_t, y_c, T0, te, tb, max_combinations=84,
                          random_state=0).global_p_value
    b = compute_inference(y_t, y_c, T0, te, tb, max_combinations=84,
                          random_state=7).global_p_value
    assert a == b


def _replayed(inf, H, B, seed):
    allx = np.concatenate([inf.placebo_effects, inf.treated_effects])
    rng = np.random.default_rng(seed)
    pi = np.array([rng.choice(allx.size, size=H, replace=False)
                   for _ in range(B)])
    s = np.mean(np.abs(allx[pi]), axis=1)
    k = int(np.sum(s >= np.mean(np.abs(inf.treated_effects)) - 1e-12))
    return (1 + k) / (1 + B)


def test_the_global_p_value_by_draws_counts_the_observed_arrangement():
    y_t, y_c, T0, te, tb = _contrast(Tb=30, T_post=8)
    inf = compute_inference(y_t, y_c, T0, te, tb, max_combinations=200,
                            random_state=3)
    assert inf.global_p_value == pytest.approx(_replayed(inf, 8, 200, 3),
                                               abs=1e-15)


def test_one_arrangement_short_of_the_count_draws_instead():
    y_t, y_c, T0, te, tb = _contrast(Tb=6, T_post=3)
    inf = compute_inference(y_t, y_c, T0, te, tb, max_combinations=83,
                            random_state=5)
    assert inf.global_p_value == pytest.approx(_replayed(inf, 3, 83, 5),
                                               abs=1e-15)


def test_a_huge_effect_never_gives_a_zero_global_p_value():
    y_t, y_c, T0, te, tb = _contrast(Tb=30, T_post=8)
    y_t[T0:] += 1e3
    inf = compute_inference(y_t, y_c, T0, te, tb, max_combinations=500,
                            random_state=0)
    assert inf.global_p_value == pytest.approx(1 / 501)


# --------------------------------------------------------------------------- #
# T_post
# --------------------------------------------------------------------------- #
def test_t_post_limits_the_window_the_inference_reads():
    y_t, y_c, T0, te, tb = _contrast(Tb=20, T_post=4, extra=6)
    inf = compute_inference(y_t, y_c, T0, te, tb, random_state=0, T_post=4,
                            cumulative_band=True, cumulative_n_sim=300)
    assert inf.treated_effects.size == 4
    assert inf.per_period_pvals.size == 4
    assert inf.ci.shape == (y_t.size, 2)
    assert np.all(np.isfinite(inf.ci[T0:T0 + 4]))
    assert np.isnan(inf.ci[T0 + 4:]).all()
    assert inf.fulltreated_effects.size == y_t.size
    assert inf.cumulative_band.horizon == 4
    assert inf.cumulative_band.point == pytest.approx(
        float((y_t - y_c)[T0:T0 + 4].sum()))


def test_t_post_reaches_the_inversion_band():
    y_t, y_c, T0, te, tb = _contrast(Tb=20, T_post=4, extra=6)
    rng = np.random.default_rng(2)
    controls = y_c[None, :] + rng.normal(0, 1, (4, y_t.size))
    inf = compute_inference(y_t, y_c, T0, te, tb, random_state=0, T_post=4,
                            cumulative_band=True, cumulative_method="inversion",
                            control_outcomes=controls)
    assert inf.cumulative_band.horizon == 4
    assert inf.cumulative_band.n_reference == tb + 4


def test_t_post_defaults_to_every_post_period():
    y_t, y_c, T0, te, tb = _contrast(Tb=20, T_post=4, extra=6)
    inf = compute_inference(y_t, y_c, T0, te, tb, random_state=0)
    assert inf.treated_effects.size == 10


@pytest.mark.parametrize("bad", [0, 11, -1])
def test_t_post_outside_the_post_period_is_refused(bad):
    y_t, y_c, T0, te, tb = _contrast(Tb=20, T_post=4, extra=6)
    with pytest.raises(MlsynthDataError, match="T_post"):
        compute_inference(y_t, y_c, T0, te, tb, T_post=bad)


@pytest.mark.parametrize("bad", [True, 2.0, "4"])
def test_t_post_must_be_an_integer(bad):
    y_t, y_c, T0, te, tb = _contrast(Tb=20, T_post=4, extra=6)
    with pytest.raises(MlsynthConfigError, match="T_post"):
        compute_inference(y_t, y_c, T0, te, tb, T_post=bad)


# --------------------------------------------------------------------------- #
# through fit()
# --------------------------------------------------------------------------- #
_PANEL = _long(_factor_panel(N=10, T=50, r=2, scale=1.0, noise=0.4, seed=3))
_BASE = dict(df=_PANEL, outcome="y", unitid="unit", time="time", m_eq=2,
             backend="exact")


def _fit(**kw):
    return MAREX({**_BASE, **kw}).fit()


def test_the_config_alpha_reaches_the_inference_and_the_power_curve():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = _fit(T0=40, inference=True, blank_periods=10, alpha=0.10)
    inf = res.globres.inference
    assert inf.alpha == 0.10
    assert np.all(np.isfinite(inf.ci[40:]))
    assert res.post_fit.power.alpha == 0.10


def test_the_default_alpha_is_unchanged():
    res = _fit(T0=40, inference=True, blank_periods=20)
    assert res.globres.inference.alpha == 0.05
    assert res.post_fit.power.alpha == 0.05


def test_the_default_blank_window_follows_the_floors():
    cfg = MAREXConfig(**_BASE, T0=40, inference=True)
    assert cfg.blank_periods == 19
    assert _fit(T0=40, inference=True).study.blank_periods == 19


def test_the_default_blank_window_reads_the_configured_level():
    """At alpha 0.10 twenty pre-periods hold 9 blanks and 11 fitting periods."""
    assert MAREXConfig(**_BASE, T0=20, inference=True, alpha=0.10).blank_periods == 9


def test_a_short_pre_period_is_refused_by_default():
    with pytest.raises(MlsynthConfigError, match="fitting periods"):
        MAREXConfig(**_BASE, T0=20, inference=True)


def test_an_explicit_short_blank_window_is_honoured_with_an_unbounded_band():
    with pytest.warns(UserWarning, match="blank periods"):
        res = _fit(T0=20, inference=True, blank_periods=4, T_post=5)
    inf = res.globres.inference
    assert res.study.blank_periods == 4
    assert np.all(np.isinf(inf.ci[20:25]))
    assert np.isfinite(inf.global_p_value)
    assert res.post_fit.ci_lower is None and res.post_fit.ci_upper is None


def test_t_post_shorter_than_the_post_period_through_fit():
    res = _fit(T0=40, inference=True, blank_periods=20, T_post=4)
    assert res.globres.inference.treated_effects.size == 4
    for c in res.clusters.values():
        assert c.inference.treated_effects.size == 4


def test_no_inference_carves_no_blank_window():
    assert MAREXConfig(**_BASE, T0=20).blank_periods == 0


# --------------------------------------------------------------------------- #
# the plot
# --------------------------------------------------------------------------- #
def test_an_unbounded_band_is_not_drawn():
    from mlsynth.utils.marex_helpers.plotter import plot_marex

    with pytest.warns(UserWarning):
        res = _fit(T0=20, inference=True, blank_periods=4, T_post=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plot_marex(res, plot_type="treatment")
    for ax in plt.gcf().axes:
        for coll in ax.collections:
            for path in coll.get_paths():
                assert np.isfinite(path.vertices).all()
