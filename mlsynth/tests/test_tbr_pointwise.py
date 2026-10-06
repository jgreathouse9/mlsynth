r"""TBR reports the per-period effect with its own posterior.

The cumulative effect :math:`\Delta(t)` answers what the campaign did by time
:math:`t`. It cannot answer what it did *in* a period, and a per-period number
without its interval cannot be read: an effect indistinguishable from zero
looks the same as one that is not.

Equation 6 supplies the interval with no new algebra. At a horizon of one
period the cumulative effect is the per-period effect and :math:`\bar{x}_1` is
:math:`x_t`, so the scale reduces to
:math:`s\sqrt{v_a + 2 x_t v_{ab} + v_b x_t^2 + 1}`. These tests hold the
per-period field to that identity, to agreement with the cumulative path, and
to the reading the paper's Figure 3 gives its pretest half: the model's
residuals, centred on zero.

Levels: smoke, unit invariants, a property over the panel domain, edge cases,
failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.utils.tbr_helpers.posterior import cumulative_posterior, fit_pretest
from mlsynth.utils.tbr_helpers.structures import PointwiseEffect

N_PRE, N_POST, N_GEOS = 30, 8, 8
TREATED = {0, 1, 2}


def _panel(n_pre: int = N_PRE, n_post: int = N_POST, n_geos: int = N_GEOS,
           seed: int = 0, effect: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) + 60.0
    level = rng.uniform(8.0, 20.0, n_geos)
    load = rng.uniform(0.7, 1.3, n_geos)
    Y = level[None, :] + load[None, :] * factor[:, None] + rng.normal(0, 1.0, (T, n_geos))
    for j in TREATED:
        if j < n_geos:
            Y[n_pre:, j] += effect
    return pd.DataFrame([
        {"geo": f"g{j:02d}", "t": t, "sales": float(Y[t, j]),
         "post": int(t >= n_pre), "is_treat": int(j in TREATED),
         "is_ctrl": int(j not in TREATED)}
        for j in range(n_geos) for t in range(T)])


def _fit(df: pd.DataFrame, **kw):
    base = dict(df=df, outcome="sales", unitid="geo", time="t",
                treatment_col="is_treat", control_col="is_ctrl",
                post_col="post")
    base.update(kw)
    return TBR(TBRConfig(**base)).fit().report


def _aggregates(df: pd.DataFrame):
    wide = df.pivot_table(index="t", columns="geo", values="sales").sort_index()
    cols = list(wide.columns)
    tre = [cols.index(f"g{j:02d}") for j in sorted(TREATED)]
    ctl = [j for j in range(len(cols)) if j not in tre]
    arr = wide.to_numpy(dtype=float)
    return arr[:, tre].sum(axis=1), arr[:, ctl].sum(axis=1)


# --------------------------------------------------------------------------- smoke
def test_a_named_fit_carries_the_per_period_posterior():
    rep = _fit(_panel())
    assert isinstance(rep.pointwise, PointwiseEffect)
    n = N_PRE + N_POST
    for field in ("estimate", "scale", "lower", "upper", "periods"):
        assert len(getattr(rep.pointwise, field)) == n
    assert np.isfinite(rep.pointwise.estimate).all()
    assert (np.asarray(rep.pointwise.scale) > 0.0).all()


# ------------------------------------------------------------------ unit invariants
def test_the_estimate_is_the_prediction_error_of_the_pretest_fit():
    df = _panel(effect=5.0)
    rep = _fit(df)
    y, x = _aggregates(df)
    fit = fit_pretest(y[:N_PRE], x[:N_PRE])
    direct = y - (fit.alpha + fit.beta * x)
    assert rep.pointwise.estimate == pytest.approx(direct.tolist(), abs=1e-9)


def test_the_scale_is_equation_six_at_one_horizon():
    df = _panel()
    rep = _fit(df)
    y, x = _aggregates(df)
    fit = fit_pretest(y[:N_PRE], x[:N_PRE])
    for t in (0, 5, N_PRE, N_PRE + 3):
        _, scale = cumulative_posterior(fit, y[t:t + 1], x[t:t + 1])
        assert rep.pointwise.scale[t] == pytest.approx(float(scale[-1]), rel=1e-12)


def test_the_per_period_effects_sum_to_the_cumulative_one():
    rep = _fit(_panel(effect=4.0))
    post = np.asarray(rep.pointwise.estimate)[N_PRE:]
    assert post.sum() == pytest.approx(rep.cumulative.estimate[-1], rel=1e-10)


def test_the_interval_brackets_the_estimate_everywhere():
    pw = _fit(_panel(effect=3.0)).pointwise
    lo = np.asarray(pw.lower); hi = np.asarray(pw.upper)
    est = np.asarray(pw.estimate)
    assert (lo <= est).all() and (est <= hi).all()
    assert (hi > lo).all()


def test_the_level_and_degrees_of_freedom_match_the_cumulative_path():
    rep = _fit(_panel(), level=0.8)
    assert rep.pointwise.level == pytest.approx(0.8)
    assert rep.pointwise.df == rep.cumulative.df == rep.tbr_fit.df


def test_the_periods_are_the_whole_panel_not_only_the_test_window():
    rep = _fit(_panel())
    assert list(rep.pointwise.periods) == list(rep.time_series.time_periods)
    assert len(rep.pointwise.periods) > len(rep.cumulative.periods)


def test_a_wider_level_gives_a_wider_interval():
    narrow = _fit(_panel(), level=0.5).pointwise
    wide = _fit(_panel(), level=0.95).pointwise
    a = np.asarray(wide.upper) - np.asarray(wide.lower)
    b = np.asarray(narrow.upper) - np.asarray(narrow.lower)
    assert (a > b).all()


def test_the_pretest_half_reads_as_the_models_residuals():
    """Figure 3's caption calls it a visual diagnostic. It behaves like one."""
    pw = _fit(_panel(effect=6.0)).pointwise
    pre = np.asarray(pw.estimate)[:N_PRE]
    assert abs(pre.mean()) < 1e-8
    lo = np.asarray(pw.lower)[:N_PRE]
    hi = np.asarray(pw.upper)[:N_PRE]
    assert 0.7 <= np.mean((lo <= 0) & (0 <= hi)) <= 1.0


# ---------------------------------------------------------------- the variance knob
def test_the_variance_choice_reaches_the_per_period_band():
    iid = _fit(_panel(effect=4.0)).pointwise
    hac = _fit(_panel(effect=4.0), variance="hac").pointwise
    assert hac.estimate == pytest.approx(iid.estimate, rel=1e-12)
    assert hac.scale != pytest.approx(iid.scale, rel=1e-9)


def test_an_explicit_bandwidth_reaches_the_per_period_band():
    a = _fit(_panel(), variance="hac", hac_bandwidth=1).pointwise
    b = _fit(_panel(), variance="hac", hac_bandwidth=5).pointwise
    assert a.scale != pytest.approx(b.scale, rel=1e-9)


# ------------------------------------------------------------------------- property
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(n_pre=st.integers(min_value=6, max_value=40),
       n_post=st.integers(min_value=1, max_value=12),
       n_geos=st.integers(min_value=4, max_value=10),
       seed=st.integers(min_value=0, max_value=2 ** 31 - 1))
def test_the_identity_holds_over_the_panel_domain(n_pre, n_post, n_geos, seed):
    df = _panel(n_pre=n_pre, n_post=n_post, n_geos=n_geos, seed=seed)
    rep = _fit(df)
    assume(not rep.tbr_fit.rank_deficient)
    y, x = _aggregates(df)
    fit = fit_pretest(y[:n_pre], x[:n_pre])

    est = np.asarray(rep.pointwise.estimate)
    assert est == pytest.approx((y - (fit.alpha + fit.beta * x)).tolist(),
                                abs=1e-7, rel=1e-9)
    assert est[n_pre:].sum() == pytest.approx(rep.cumulative.estimate[-1],
                                              rel=1e-8)
    assert (np.asarray(rep.pointwise.scale) > 0.0).all()


# ------------------------------------------------------------------------ edge cases
def test_a_single_post_period_still_gets_a_band():
    rep = _fit(_panel(n_post=1))
    assert len(rep.pointwise.estimate) == N_PRE + 1
    assert rep.pointwise.estimate[-1] == pytest.approx(
        rep.cumulative.estimate[-1], rel=1e-10)


def test_a_constant_control_pretest_still_reports_a_band():
    """The group fit tolerates this and records it; the band comes with it."""
    df = _panel()
    pre = df["t"] < N_PRE
    df.loc[pre & (df["is_ctrl"] == 1), "sales"] = 100.0
    rep = _fit(df)
    assert rep.tbr_fit.rank_deficient is True
    assert rep.pointwise is not None
    assert np.isfinite(rep.pointwise.estimate).all()


def test_the_shortest_admissible_pretest_still_gets_a_band():
    rep = _fit(_panel(n_pre=4, n_post=2))
    assert len(rep.pointwise.estimate) == 6
    assert (np.asarray(rep.pointwise.scale) > 0.0).all()


# --------------------------------------------------------------------------- failure
def test_the_container_is_frozen():
    pw = _fit(_panel()).pointwise
    with pytest.raises(Exception):
        pw.level = 0.5


def test_the_plotter_draws_the_band_on_the_per_period_panel():
    """The panel the paper's Figure 3 shows with an interval."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection

    from mlsynth.utils.tbr_helpers.plotter import plot_tbr

    fig = plot_tbr(_fit(_panel(effect=5.0)))
    middle = fig.axes[1]
    bands = [c for c in middle.collections if isinstance(c, PolyCollection)]
    assert bands, "the per-period panel has no filled interval"


def test_the_interval_is_the_t_quantile_on_n_minus_two_degrees_of_freedom():
    """Not the pretest length: the posterior is a t on ``n - 2``.

    Without this the band is merely narrower, which every other assertion here
    tolerates, so this is the one that fails if the degrees of freedom drift.
    """
    from scipy import stats

    rep = _fit(_panel(effect=4.0), level=0.9)
    pw = rep.pointwise
    assert pw.df == N_PRE - 2

    est = np.asarray(pw.estimate)
    scale = np.asarray(pw.scale)
    lo = stats.t.ppf(0.05, pw.df, loc=est, scale=scale)
    hi = stats.t.ppf(0.95, pw.df, loc=est, scale=scale)
    assert np.asarray(pw.lower) == pytest.approx(lo, rel=1e-12)
    assert np.asarray(pw.upper) == pytest.approx(hi, rel=1e-12)

    wrong = stats.t.ppf(0.05, rep.tbr_fit.n_pretest, loc=est, scale=scale)
    assert not np.allclose(np.asarray(pw.lower), wrong)
