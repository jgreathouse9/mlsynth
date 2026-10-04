r"""TBR's posterior attached to a measured TBRMM design.

The point estimates come from the augmented DiD fit (Li and Van den Bulte 2022,
eqn 2.4). Their uncertainty is TBR's, because under Kerman, Wang and Vaver's
flat prior on :math:`(\alpha, \beta, \log\sigma)` the two are the same
regression: the posterior of the cumulative effect is a t on :math:`n-2` degrees
of freedom with the scale their eqn 6 gives, which is algebraically the OLS
prediction-error standard deviation.

Two facts drive most of what is asserted here.

The cumulative effect is the primitive and the ATT is a rescaling of it by a
known constant, so the interval divides and the degrees of freedom do not move.
The constant is ``n_treated * n_post``, not ``n_post``, because TBRMM pools by
averaging over geos as well as periods.

The group's uncertainty does not decompose into the markets'. Eqn 6's variance
has a coefficient term growing as :math:`T^2` and a test-noise term growing as
:math:`T`, and across geos the residuals are correlated. Both mean the group
posterior has to come from a regression spanning what is correlated -- the
summed treated series -- and cannot be assembled from the per-market posteriors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBRMM
from mlsynth.config_models import TBRMMConfig


def _panel(n_units: int = 8, n_pre: int = 40, n_post: int = 10,
           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) + 50.0
    level = rng.uniform(8.0, 20.0, n_units)
    loading = rng.uniform(0.7, 1.3, n_units)
    Y = level[None, :] + loading[None, :] * factor[:, None] + rng.normal(0, 1.0, (T, n_units))
    units = [f"g{j:02d}" for j in range(n_units)]
    return pd.DataFrame([{"geo": units[j], "t": t, "sales": float(Y[t, j]),
                          "post": int(t >= n_pre)}
                         for j in range(n_units) for t in range(T)])


def _cfg(df, **kw):
    base = dict(df=df, outcome="sales", unitid="geo", time="t",
                max_treatment_size=3, n_test=8)
    base.update(kw)
    return TBRMMConfig(**base)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_a_measured_design_carries_a_posterior():
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect

    assert eff.posterior is not None
    assert all(m.posterior is not None for m in eff.market_effects)
    assert eff.posterior.df == 40 - 2


def test_no_post_window_means_no_posterior():
    res = TBRMM(_cfg(_panel().drop(columns=["post"]))).fit()
    assert res.report is None
    assert all(d.effect is None for d in res.designs)


# ---------------------------------------------------------------------------
# The interval is a rescaling of the cumulative one
# ---------------------------------------------------------------------------

def test_the_att_interval_is_the_cumulative_one_over_n_treated_times_n_post():
    """The ATT divides by geos AND periods, which is where a factor of N hides."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect
    k = eff.n_treated * eff.n_post

    assert eff.posterior.att_lower == pytest.approx(eff.posterior.total_lower / k, rel=1e-12)
    assert eff.posterior.att_upper == pytest.approx(eff.posterior.total_upper / k, rel=1e-12)


def test_each_market_interval_divides_by_its_own_periods_only():
    """A single geo's ATT averages over periods, not over geos."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect

    for m in eff.market_effects:
        assert m.posterior.att_lower == pytest.approx(
            m.posterior.total_lower / eff.n_post, rel=1e-12)


def test_the_interval_brackets_the_point_estimate():
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect

    assert eff.posterior.total_lower < eff.total_effect < eff.posterior.total_upper
    assert eff.posterior.att_lower < eff.report.att < eff.posterior.att_upper
    for m in eff.market_effects:
        assert m.posterior.total_lower < m.total_effect < m.posterior.total_upper


# ---------------------------------------------------------------------------
# The group posterior is its own fit
# ---------------------------------------------------------------------------

def test_the_group_scale_is_not_the_markets_combined_as_independent():
    """Treated geos co-move, so combining their scales in quadrature is wrong."""
    res = TBRMM(_cfg(_panel(), post_col="post", max_treatment_size=3)).fit()
    eff = next(d.effect for d in res.designs if d.k >= 2)

    quadrature = float(np.sqrt(sum(m.posterior.scale ** 2 for m in eff.market_effects)))
    assert eff.posterior.scale != pytest.approx(quadrature, rel=1e-6)


def test_the_group_cumulative_is_the_sum_of_the_markets():
    """OLS is linear in y, so the point estimates do reconcile exactly."""
    res = TBRMM(_cfg(_panel(), post_col="post", max_treatment_size=3)).fit()
    eff = next(d.effect for d in res.designs if d.k >= 2)

    assert eff.total_effect == pytest.approx(
        sum(m.total_effect for m in eff.market_effects), rel=1e-9)


# ---------------------------------------------------------------------------
# The level
# ---------------------------------------------------------------------------

def test_a_wider_level_gives_a_wider_interval():
    narrow = TBRMM(_cfg(_panel(), post_col="post", level=0.50)).fit().recommended.effect
    wide = TBRMM(_cfg(_panel(), post_col="post", level=0.99)).fit().recommended.effect

    assert wide.posterior.total_lower < narrow.posterior.total_lower
    assert wide.posterior.total_upper > narrow.posterior.total_upper
    assert wide.posterior.scale == pytest.approx(narrow.posterior.scale, rel=1e-12)


def test_the_level_is_recorded_on_the_posterior():
    res = TBRMM(_cfg(_panel(), post_col="post", level=0.80)).fit()
    assert res.recommended.effect.posterior.level == pytest.approx(0.80)


def test_the_default_level_is_the_papers():
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    assert res.recommended.effect.posterior.level == pytest.approx(0.90)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_an_impossible_level_is_refused(bad):
    with pytest.raises(Exception):
        _cfg(_panel(), post_col="post", level=bad)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_report_inference_carries_the_interval_in_the_atts_units():
    """report.effects.att and report.inference must be on one scale."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    inf, eff = res.report.inference, res.recommended.effect

    assert inf is not None
    assert inf.ci_lower == pytest.approx(eff.posterior.att_lower, rel=1e-12)
    assert inf.ci_upper == pytest.approx(eff.posterior.att_upper, rel=1e-12)
    assert inf.ci_lower < res.report.effects.att < inf.ci_upper


def test_report_inference_names_its_method():
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    assert res.report.inference.method == "tbr_posterior"


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def test_probability_of_direction_is_a_probability():
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect

    for p in [eff.posterior] + [m.posterior for m in eff.market_effects]:
        assert 0.0 <= p.prob_direction <= 1.0
        assert p.prob_direction >= 0.5, "it is the mass on the estimate's own side"


def test_a_large_injected_effect_is_called_with_near_certainty():
    df = _panel()
    design = TBRMM(_cfg(df, post_col="post")).fit().recommended
    hit = df["geo"].isin(design.treatment_units) & (df["post"] == 1)
    df.loc[hit, "sales"] = df.loc[hit, "sales"] + 60.0

    eff = TBRMM(_cfg(df, post_col="post")).fit().recommended.effect
    assert eff.posterior.prob_direction > 0.999
    assert eff.posterior.total_lower > 0.0
