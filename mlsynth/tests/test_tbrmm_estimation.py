"""TBRMM's estimation phase: market-level effects when a post window is given.

TBRMM selects a treatment/control split from pretest data. When the panel also
carries a post window, the chosen split is measured on it with the augmented
DiD regression of Li and Van den Bulte (2022, eqn 2.4), fitted per treated
market against the control-group average and pooled as the average of the
per-market effects (their Appendix C, case (i)). The pooled effect is therefore
the mean of the market-level effects by construction, which is the invariant
most of these tests assert.

Without a post window the estimator's behaviour is unchanged: a design, and no
effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBRMM
from mlsynth.config_models import TBRMMConfig
from mlsynth.exceptions import MlsynthDataError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _panel(n_units: int = 8, n_pre: int = 40, n_post: int = 10,
           seed: int = 0) -> pd.DataFrame:
    """A small balanced panel with a common factor, marked pre/post."""
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) + 50.0
    level = rng.uniform(8.0, 20.0, n_units)
    loading = rng.uniform(0.7, 1.3, n_units)
    Y = level[None, :] + loading[None, :] * factor[:, None] + rng.normal(0, 1.0, (T, n_units))
    units = [f"g{j:02d}" for j in range(n_units)]
    rows = [{"geo": units[j], "t": t, "sales": float(Y[t, j]),
             "post": int(t >= n_pre)}
            for j in range(n_units) for t in range(T)]
    return pd.DataFrame(rows)


def _inject(df: pd.DataFrame, units, tau: float) -> pd.DataFrame:
    """Add a constant additive effect to the named units' post periods."""
    out = df.copy()
    hit = out["geo"].isin(list(units)) & (out["post"] == 1)
    out.loc[hit, "sales"] = out.loc[hit, "sales"] + tau
    return out


def _cfg(df: pd.DataFrame, **kw) -> TBRMMConfig:
    base = dict(df=df, outcome="sales", unitid="geo", time="t",
                max_treatment_size=3, n_test=8)
    base.update(kw)
    return TBRMMConfig(**base)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_post_window_populates_report():
    """With a post column the recommended design is measured."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()

    assert res.report is not None, "a post window should produce a report"
    att = res.report.effects.att
    assert att is not None and np.isfinite(att)


def test_design_only_when_no_post_window():
    """Without a post column nothing is measured -- the prior behaviour."""
    res = TBRMM(_cfg(_panel().drop(columns=["post"]))).fit()

    assert res.report is None
    assert all(d.effect is None for d in res.designs)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_pooled_effect_is_the_mean_of_market_effects():
    """Appendix C: the pooled ATT is the average of the per-market ATTs."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect

    per_market = np.array([m.att for m in eff.market_effects], dtype=float)
    assert per_market.size == len(res.recommended.treatment_units)
    assert eff.att == pytest.approx(float(per_market.mean()), rel=1e-10)


def test_every_candidate_design_is_measured():
    """The whole menu is measured, not only the recommendation."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()

    assert len(res.designs) >= 1
    for d in res.designs:
        assert d.effect is not None, f"design k={d.k} carries no effect"
        assert len(d.effect.market_effects) == len(d.treatment_units)


def test_market_effects_name_the_treated_markets():
    """Every treated market appears once, and nothing else does."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect

    named = [m.unit for m in eff.market_effects]
    assert sorted(named) == sorted(res.recommended.treatment_units)
    assert len(named) == len(set(named))


def test_recovers_an_injected_additive_effect():
    """A constant tau added to the treated post periods comes back as tau."""
    tau = 6.0
    design = TBRMM(_cfg(_panel(), post_col="post")).fit().recommended
    treated = list(design.treatment_units)

    # The post window is excluded from design scoring, so injecting into it
    # leaves the search -- and therefore the chosen split -- untouched.
    res = TBRMM(_cfg(_inject(_panel(), treated, tau), post_col="post")).fit()
    assert sorted(res.recommended.treatment_units) == sorted(treated)
    eff = res.recommended.effect

    assert eff.att == pytest.approx(tau, abs=1.0)
    for m in eff.market_effects:
        assert m.att == pytest.approx(tau, abs=1.5)


def test_zero_effect_panel_reads_near_zero():
    """With no injected effect the measured ATT sits near zero."""
    res = TBRMM(_cfg(_panel(), post_col="post")).fit()
    eff = res.recommended.effect

    scale = float(np.std(_panel()["sales"]))
    assert abs(eff.att) < 0.5 * scale


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_treated_market_pools_to_itself():
    """One treated market: the pooled effect is that market's effect."""
    df = _panel()
    res = TBRMM(_cfg(df, post_col="post", max_treatment_size=1)).fit()
    eff = res.recommended.effect

    assert len(eff.market_effects) == 1
    assert eff.att == pytest.approx(eff.market_effects[0].att, rel=1e-12)


def test_all_post_flags_zero_leaves_report_empty():
    """A post column that marks nothing is a design-only run, not a zero effect."""
    df = _panel()
    df["post"] = 0
    res = TBRMM(_cfg(df, post_col="post")).fit()

    assert res.report is None
    assert all(d.effect is None for d in res.designs)


def test_one_post_period_is_measured():
    """A single realized period is enough to read an effect."""
    res = TBRMM(_cfg(_panel(n_post=1), post_col="post")).fit()

    assert res.report is not None
    assert np.isfinite(res.recommended.effect.att)


# ---------------------------------------------------------------------------
# Failures are reported, not swallowed
# ---------------------------------------------------------------------------

def test_post_window_shorter_than_the_pretest_requirement_still_designs():
    """A thin post window does not break the design half."""
    res = TBRMM(_cfg(_panel(n_pre=40, n_post=2), post_col="post")).fit()

    assert res.recommended is not None
    assert res.recommended.treatment_units


def test_non_block_post_column_raises():
    """post must stay block-assigned; a ragged flag is a data error."""
    df = _panel()
    df.loc[(df["geo"] == "g00") & (df["t"] == 5), "post"] = 1

    with pytest.raises(MlsynthDataError):
        TBRMM(_cfg(df, post_col="post")).fit()


# ---------------------------------------------------------------------------
# The measurement helper's own refusals
# ---------------------------------------------------------------------------

def test_measure_design_refuses_an_empty_treatment_group():
    from mlsynth.utils.tbrmm_helpers.estimate import measure_design
    from mlsynth.exceptions import MlsynthEstimationError

    pre = np.arange(20.0).reshape(10, 2)
    post = np.arange(6.0).reshape(3, 2)
    with pytest.raises(MlsynthEstimationError, match="no treated geo"):
        measure_design(pre, post, ["a", "b"], [], ["a", "b"])


def test_measure_design_refuses_an_empty_control_group():
    from mlsynth.utils.tbrmm_helpers.estimate import measure_design
    from mlsynth.exceptions import MlsynthEstimationError

    pre = np.arange(20.0).reshape(10, 2)
    post = np.arange(6.0).reshape(3, 2)
    with pytest.raises(MlsynthEstimationError, match="no control geo"):
        measure_design(pre, post, ["a", "b"], ["a"], [])


def test_measure_design_refuses_an_unidentified_scale():
    """A control average that never moves identifies no scale, and says so."""
    from mlsynth.utils.tbrmm_helpers.estimate import measure_design
    from mlsynth.exceptions import MlsynthEstimationError

    rng = np.random.default_rng(1)
    pre = np.column_stack([rng.normal(10, 1, 12), np.full(12, 7.0)])
    post = np.column_stack([rng.normal(10, 1, 4), np.full(4, 7.0)])
    with pytest.raises(MlsynthEstimationError, match="not identified"):
        measure_design(pre, post, ["t", "c"], ["t"], ["c"])
