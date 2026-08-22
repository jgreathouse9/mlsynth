"""``MAREXConfig.alpha`` and ``power_target`` reaching the fit that uses them.

``scexp.py`` passed neither to ``solve_marex``, so two consumers inside it ran at
the signature defaults whatever the caller configured: ``compute_inference``,
which sets the placebo band's half-width to the ``1 - alpha`` quantile of the
absolute blank-period effects, and ``compute_power_analysis``, which sets the
primary design's MDE. Setting ``alpha=0.2`` returned a 95 percent band and a
5 percent MDE.

``build_marex_pool`` on the ``top_K`` path did receive both, so the pool's MDE and
the primary design's MDE were computed at different levels in the same result.

Levels: unit invariants (the level reaches each consumer, and they agree), edge,
regression (the 0.05 / 0.80 default is unmoved).
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX


def _panel(J=10, T=24, seed=11):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.4 * rng.standard_normal((J, T))
    return pd.DataFrame([{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
                         for j in range(J) for t in range(T)])


_CFG = dict(outcome="y", unitid="unit", time="time", T0=18, m_eq=2,
            inference=True, blank_periods=6, T_post=6)


def _res(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, **_CFG)
    cfg.update(kw)
    return MAREX(cfg).fit()


def _q(res):
    """The band's half-width, read off the returned interval."""
    inf = res.globres.inference
    return float(inf.ci[-1, 1] - inf.treated_effects[-1])


# --------------------------------------------------------------------------- #
# the level reaches the placebo band
# --------------------------------------------------------------------------- #
def test_a_looser_level_gives_a_narrower_band():
    assert _q(_res(alpha=0.20)) < _q(_res(alpha=0.01))


def test_the_band_is_the_quantile_at_the_configured_level():
    for alpha in (0.01, 0.10, 0.20):
        res = _res(alpha=alpha)
        expected = float(np.quantile(
            np.abs(res.globres.inference.placebo_effects), 1.0 - alpha))
        assert _q(res) == pytest.approx(expected), alpha


def test_the_inference_records_the_configured_level():
    assert _res(alpha=0.10).globres.inference.alpha == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# the level reaches the primary MDE
# --------------------------------------------------------------------------- #
def test_a_looser_level_gives_a_smaller_mde():
    """MDE scales with ``z_{1-alpha/2} + z_{power}``, so a looser level detects a
    smaller effect."""
    assert (_res(alpha=0.20).post_fit.power.headline.mde_absolute
            < _res(alpha=0.01).post_fit.power.headline.mde_absolute)


def test_the_power_analysis_records_the_configured_level():
    assert _res(alpha=0.10).post_fit.power.alpha == pytest.approx(0.10)


def test_a_lower_power_target_gives_a_smaller_mde():
    """``power_target`` was stuck at the same signature default as ``alpha``."""
    assert (_res(power_target=0.50).post_fit.power.headline.mde_absolute
            < _res(power_target=0.95).post_fit.power.headline.mde_absolute)


def test_the_power_analysis_records_the_configured_power_target():
    assert _res(power_target=0.90).post_fit.power.power_target == pytest.approx(0.90)


# --------------------------------------------------------------------------- #
# the two MDEs in one result agree on the level
# --------------------------------------------------------------------------- #
def test_the_primary_and_pool_mdes_are_computed_at_the_same_level():
    """``build_marex_pool`` always received the configured level; the primary
    design's power analysis did not, so one result carried two levels."""
    res = _res(alpha=0.20, top_K=3)
    assert res.post_fit.power.alpha == pytest.approx(0.20)
    assert res.pool, "top_K > 1 should populate the pool"


# --------------------------------------------------------------------------- #
# regression: the default is where it was
# --------------------------------------------------------------------------- #
def test_the_default_level_is_the_one_that_was_hard_wired():
    """Every result predating this fix was computed at 0.05 / 0.80, so an
    unconfigured fit has to land in exactly the same place."""
    default, explicit = _res(), _res(alpha=0.05, power_target=0.80)
    assert _q(default) == pytest.approx(_q(explicit))
    assert (default.post_fit.power.headline.mde_absolute
            == pytest.approx(explicit.post_fit.power.headline.mde_absolute))
    assert default.globres.inference.alpha == pytest.approx(0.05)


def test_the_permutation_p_value_does_not_move_with_the_level():
    """The level sets the band and the MDE. The permutation p-value is a tail
    probability of the observed statistic, so it is the same number whatever
    level it is later compared against."""
    a, b = _res(alpha=0.01), _res(alpha=0.20)
    assert (a.globres.inference.global_p_value
            == pytest.approx(b.globres.inference.global_p_value))
    assert np.allclose(a.globres.inference.per_period_pvals,
                       b.globres.inference.per_period_pvals)
