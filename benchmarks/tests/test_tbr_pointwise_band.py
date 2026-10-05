"""The pointwise posterior the Figure 3 panel draws.

``mlsynth``'s TBR result carries the cumulative posterior and the fitted
relation, and no pointwise posterior, so the estimator's own plotter draws the
per-period difference as a bare line. Equation 6 supplies the band anyway: at a
horizon of one period the cumulative effect *is* the pointwise effect, and its
scale reduces to ``s sqrt(v_a + 2 x_t v_ab + v_b x_t^2 + 1)``.

These tests hold that identity, so the figure's middle panel is the library's
own posterior at one horizon and not a second implementation of the paper's
algebra. If the pointwise posterior is ever put on the result, this is the
check the new field has to meet.

Levels: smoke, unit invariants, a property over the domain, edge, failure.
"""
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from benchmarks.studies.tbr_geo import figure3
from mlsynth.utils.tbr_helpers.posterior import cumulative_posterior, fit_pretest


def _series(n_pre, n_post, seed, spread=5.0, noise=2.0):
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    x = 100.0 + rng.normal(0.0, spread, n)
    y = 20.0 + 1.3 * x + rng.normal(0.0, noise, n)
    return y, x


# --------------------------------------------------------------------------- smoke
def test_a_one_period_window_returns_a_location_and_a_scale():
    y, x = _series(30, 5, 0)
    loc, scale = cumulative_posterior(fit_pretest(y[:30], x[:30]), y[30:31], x[30:31])
    assert loc.shape == scale.shape == (1,)
    assert np.isfinite(loc).all() and scale[0] > 0.0


# ------------------------------------------------------------------ unit invariants
def test_the_one_period_effect_is_the_prediction_error():
    """Equation 4 at one horizon is ``y_t - (alpha + beta x_t)``."""
    y, x = _series(40, 6, 3)
    fit = fit_pretest(y[:40], x[:40])
    for t in range(40, 46):
        loc, _ = cumulative_posterior(fit, y[t:t + 1], x[t:t + 1])
        assert loc[-1] == pytest.approx(y[t] - (fit.alpha + fit.beta * x[t]),
                                        abs=1e-10)


def test_the_cumulative_effect_is_the_pointwise_effects_summed():
    y, x = _series(40, 10, 5)
    fit = fit_pretest(y[:40], x[:40])
    whole, _ = cumulative_posterior(fit, y[40:], x[40:])
    parts = sum(cumulative_posterior(fit, y[t:t + 1], x[t:t + 1])[0][-1]
                for t in range(40, 50))
    assert whole[-1] == pytest.approx(parts, rel=1e-10)


def test_the_first_cumulative_scale_is_the_first_pointwise_scale():
    """The cumulative at horizon one and the pointwise at that period coincide."""
    y, x = _series(40, 8, 7)
    fit = fit_pretest(y[:40], x[:40])
    _, whole = cumulative_posterior(fit, y[40:], x[40:])
    _, first = cumulative_posterior(fit, y[40:41], x[40:41])
    assert whole[0] == pytest.approx(first[-1], rel=1e-12)


def test_the_pointwise_scale_exceeds_the_fitted_residual_scale():
    """It carries the new period's own noise on top of the parameter uncertainty."""
    y, x = _series(40, 6, 11)
    fit = fit_pretest(y[:40], x[:40])
    s = float(np.sqrt(fit.sigma_sq))
    for t in range(40, 46):
        _, scale = cumulative_posterior(fit, y[t:t + 1], x[t:t + 1])
        assert scale[-1] > s


# ------------------------------------------------------------------------- property
@settings(max_examples=40, deadline=None)
@given(n_pre=st.integers(min_value=6, max_value=60),
       n_post=st.integers(min_value=1, max_value=15),
       seed=st.integers(min_value=0, max_value=2 ** 31 - 1),
       spread=st.floats(min_value=0.5, max_value=50.0),
       noise=st.floats(min_value=0.1, max_value=20.0))
def test_the_identity_holds_over_the_panel_domain(n_pre, n_post, seed, spread,
                                                  noise):
    y, x = _series(n_pre, n_post, seed, spread, noise)
    fit = fit_pretest(y[:n_pre], x[:n_pre])
    assume(not fit.rank_deficient)
    for t in range(n_pre, n_pre + n_post):
        loc, scale = cumulative_posterior(fit, y[t:t + 1], x[t:t + 1])
        direct = y[t] - (fit.alpha + fit.beta * x[t])
        assert loc[-1] == pytest.approx(direct, abs=1e-8, rel=1e-10)
        assert scale[-1] > 0.0


# ------------------------------------------------------------------ the figure's use
def test_the_figure_panels_are_consistent_with_each_other():
    """Panel (b) summed over the test window is panel (c)'s final value."""
    df, treated = figure3.simulate()
    phi, lo, hi = figure3.pointwise(df, treated)
    assert len(phi) == figure3.N_PRE + figure3.N_INTERVENTION + figure3.N_COOLDOWN
    assert (lo <= phi).all() and (phi <= hi).all()

    from mlsynth import TBR
    from mlsynth.config_models import TBRConfig
    report = TBR(TBRConfig(
        df=df, outcome="revenue", unitid="geo", time="t",
        treatment_col="is_treat", control_col="is_ctrl", post_col="post",
        cooldown_col="cooldown", level=figure3.LEVEL)).fit().report
    post = np.arange(len(phi)) >= figure3.N_PRE
    assert phi[post].sum() == pytest.approx(report.cumulative.estimate[-1],
                                            rel=1e-10)


def test_the_pretest_panel_reads_as_a_residual_diagnostic():
    """Centred on zero, with bands covering it at about the nominal rate."""
    df, treated = figure3.simulate()
    phi, lo, hi = figure3.pointwise(df, treated)
    pre = np.arange(len(phi)) < figure3.N_PRE
    assert abs(phi[pre].mean()) < 1e-8
    covered = np.mean((lo[pre] <= 0) & (0 <= hi[pre]))
    assert 0.75 <= covered <= 1.0


# --------------------------------------------------------------------------- failure
def test_a_constant_control_pretest_is_reported_as_rank_deficient():
    y, x = _series(30, 5, 2)
    fit = fit_pretest(y[:30], np.full(30, 7.0))
    assert fit.rank_deficient is True
