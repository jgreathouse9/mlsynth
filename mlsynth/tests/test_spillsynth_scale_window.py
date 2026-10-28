"""The scale-free fit and the held-out window of SPILLSYNTH's iterative and iscm.

Both methods rebuild a spillover-free series for each affected unit from a
per-unit synthetic control. Two options change that fit (#932, step 2).

* ``scale``: each series is divided by its mean over the fitting window before
  the simplex fit, and the synthetic is taken back to levels by the target's
  mean. The synthetic is then ``sum_j (w_j m_k / m_j) y_j``: still linear in the
  donors, with level weights in place of the simplex weights. A simplex average
  of the donors cannot reach a unit larger than all of them; the scale-free fit
  can, and it is unchanged when any unit's series is rescaled.
* ``holdout``: every fit stops ``holdout`` periods before treatment and the
  correction starts there. The held-out periods are then a placebo window for
  the corrected estimator, and their gap carries the correction's own
  prediction error.

The properties: the defaults reproduce the methods as they were; the level
weights are the map the synthetic applies to the donors; rescaling a control
leaves the scale-free estimate unchanged; the treated unit's estimate does not
depend on the size or sign of a spillover on the affected units from the start
of the correction on; the fits do not read the held-out window.

Levels: smoke, unit invariants, edge, failure, properties. The mutants are
catalogued under ``spillsynth-scale-window`` in ``tools/mutation/targets.toml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import SPILLSYNTH
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.spillsynth_helpers import build_unit_sc, run_iscm, run_iterative
from mlsynth.utils.spillsynth_helpers.setup import prepare_spillsynth_inputs

_SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
_METHODS = {"iterative": run_iterative, "iscm": run_iscm}


def _levels(seed, J, T, spread=1.0, noise=0.03):
    """Weekly sales in levels: a common seasonal path times each unit's size.

    Sizes span a factor of ``e^(2 spread)``, and row 0 is the largest unit, so a
    simplex average of the others cannot reach it.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    path = 100 + 10 * np.sin(2 * np.pi * t / 13) + np.cumsum(rng.normal(size=T))
    size = np.exp(rng.uniform(-spread, spread, size=J + 1))
    size[0] = size.max() * 2.0
    return size[:, None] * path[None, :] * (1 + noise * rng.normal(size=(J + 1, T)))


def _frame(Y, T0):
    units = [f"u{j:02d}" for j in range(Y.shape[0])]
    return pd.DataFrame([{"unit": u, "time": t, "y": float(Y[j, t]),
                          "d": int(j == 0 and t >= T0)}
                         for j, u in enumerate(units) for t in range(Y.shape[1])])


def _inputs(Y, T0, affected=("u01", "u02")):
    return prepare_spillsynth_inputs(df=_frame(Y, T0), outcome="y", treat="d",
                                     unitid="unit", time="time",
                                     affected_units=list(affected))


def _row(inputs, label):
    names = [inputs.treated_label, *inputs.affected_labels, *inputs.clean_labels]
    return names.index(label)


def _fit_unit(Y, T0, **kw):
    return build_unit_sc(0, np.arange(1, Y.shape[0]), Y, T0, **kw)


# --------------------------------------------------------------------------- #
# The per-unit fit
# --------------------------------------------------------------------------- #
def test_the_scale_free_fit_returns_a_synthetic_in_levels():
    Y = _levels(0, J=6, T=40)
    w, cf, gap, pre_rmspe, sol = _fit_unit(Y, 30, scale=True)
    assert sol is None and w.shape == (6,) and np.all(np.isfinite(cf))
    np.testing.assert_allclose(cf + gap, Y[0], rtol=1e-12)
    assert pre_rmspe == pytest.approx(np.sqrt(np.mean(gap[:30] ** 2)), rel=1e-12)


@settings(max_examples=60, deadline=None)
@given(seed=_SEEDS, J=st.integers(1, 7), T0=st.integers(2, 20))
def test_the_level_weights_are_the_map_the_synthetic_applies(seed, J, T0):
    """``cf = Y0 w`` with ``w = w_simplex m_k / m_j``, over every period."""
    Y = _levels(seed, J, T0 + 5)
    w, cf, _, _, _ = _fit_unit(Y, T0, scale=True)
    np.testing.assert_allclose(cf, Y[1:].T @ w, rtol=1e-10)
    m = Y[:, :T0].mean(axis=1)
    simplex = w * m[1:] / m[0]
    assert simplex.min() >= 0.0 and simplex.sum() == pytest.approx(1.0, abs=1e-9)


@settings(max_examples=60, deadline=None)
@given(seed=_SEEDS, J=st.integers(1, 7), T0=st.integers(2, 20),
       factor=st.floats(1e-3, 1e3), which=st.integers(0, 7))
def test_rescaling_a_unit_rescales_only_its_own_part(seed, J, T0, factor, which):
    """A donor rescaled leaves the synthetic unchanged; the target rescaled
    rescales it. The plain simplex fit has neither property."""
    Y = _levels(seed, J, T0 + 5)
    _, cf, _, _, _ = _fit_unit(Y, T0, scale=True)
    k = which % (J + 1)
    Ys = Y.copy()
    Ys[k] *= factor
    _, cf_s, _, _, _ = _fit_unit(Ys, T0, scale=True)
    expected = cf * (factor if k == 0 else 1.0)
    np.testing.assert_allclose(cf_s, expected, rtol=1e-7)


@settings(max_examples=40, deadline=None)
@given(seed=_SEEDS, J=st.integers(1, 7), T0=st.integers(2, 20))
def test_the_level_shift_has_nothing_left_to_absorb_once_scaled(seed, J, T0):
    """Every scaled series has mean one over the fitting window, so centring
    shifts target and donors by the same constant and the shift is zero."""
    Y = _levels(seed, J, T0 + 5)
    _, cf, _, _, _ = _fit_unit(Y, T0, scale=True)
    _, cf_i, _, _, _ = _fit_unit(Y, T0, scale=True, intercept=True)
    np.testing.assert_allclose(cf_i, cf, rtol=1e-8)


def test_a_unit_larger_than_every_donor_is_reached_only_once_scaled():
    """Twice the largest donor's size: the simplex average falls short by about
    half the target, the scale-free fit reaches it to the noise."""
    Y = _levels(3, J=6, T=40, noise=0.0)
    _, _, _, plain, _ = _fit_unit(Y, 30)
    _, _, _, scaled, _ = _fit_unit(Y, 30, scale=True)
    level = Y[0, :30].mean()
    assert plain > 0.3 * level
    assert scaled < 1e-9 * level


def test_a_single_donor_is_rescaled_to_the_target():
    Y = _levels(4, J=1, T=20)
    w, cf, _, _, _ = _fit_unit(Y, 12, scale=True)
    ratio = Y[0, :12].mean() / Y[1, :12].mean()
    np.testing.assert_allclose(w, [ratio], rtol=1e-12)
    np.testing.assert_allclose(cf, ratio * Y[1], rtol=1e-12)


@pytest.mark.parametrize("row", [0, 2])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_series_without_a_positive_mean_cannot_be_scaled(row, value):
    Y = _levels(0, J=3, T=20)
    Y[row, :12] = value
    with pytest.raises(MlsynthDataError, match="pre-period mean"):
        _fit_unit(Y, 12, scale=True)


def test_the_scale_free_fit_is_for_outcomes_not_predictors():
    Y = _levels(0, J=3, T=20)
    with pytest.raises(MlsynthConfigError, match="outcome-only"):
        _fit_unit(Y, 12, scale=True, predictors=np.ones((4, 2)),
                  predictor_names=["a", "b"])


# --------------------------------------------------------------------------- #
# The pipelines: defaults
# --------------------------------------------------------------------------- #
def _same(a, b):
    for name in ("att", "att_scm", "pre_rmspe"):
        assert getattr(a, name) == getattr(b, name)
    for name in ("gap", "gap_scm", "counterfactual", "treated_synthetic_pre"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.donor_weights == b.donor_weights


@pytest.mark.parametrize("method", sorted(_METHODS))
@pytest.mark.parametrize("intercept", [False, True])
def test_the_defaults_are_the_methods_as_they_were(method, intercept):
    inputs = _inputs(_levels(1, J=8, T=40), T0=30)
    run = _METHODS[method]
    base = run(inputs, intercept=intercept)
    _same(run(inputs, intercept=intercept, scale=False, holdout=0), base)
    assert base.scale_free is False and base.holdout_periods == 0
    assert base.holdout_gap.shape == (0,) and np.isnan(base.holdout_rmspe)


# --------------------------------------------------------------------------- #
# The pipelines: invariants
# --------------------------------------------------------------------------- #
def _plant(inputs, Y, T0, start, deltas):
    """Add a spillover path to each affected unit from ``start`` on."""
    Yp = Y.copy()
    for label, d in zip(inputs.affected_labels, deltas):
        j = int(label[1:])
        Yp[j, start:] += d
    return Yp


@settings(max_examples=25, deadline=None)
@given(seed=_SEEDS, method=st.sampled_from(sorted(_METHODS)),
       scale=st.booleans(), holdout=st.integers(0, 6),
       amp=st.floats(-80.0, 80.0), shape_seed=_SEEDS)
def test_the_estimate_ignores_the_spillovers_size_and_sign(seed, method, scale,
                                                           holdout, amp,
                                                           shape_seed):
    """The affected units' outcomes from the start of the correction on never
    enter the treated unit's estimate: iterative replaces them, and in iscm a
    path added to unit ``k`` moves the gaps by ``Omega[:, k]`` times it, which
    ``Omega^{-1}`` returns to unit ``k`` alone."""
    T0, T = 30, 40
    Y = _levels(seed, J=8, T=T)
    inputs = _inputs(Y, T0)
    start = T0 - holdout
    rng = np.random.default_rng(shape_seed)
    deltas = [amp * Y[int(lab[1:]), :start].mean() / 100 * rng.normal(size=T - start)
              for lab in inputs.affected_labels]
    run = _METHODS[method]
    base = run(inputs, scale=scale, holdout=holdout)
    hit = run(_inputs(_plant(inputs, Y, T0, start, deltas), T0),
              scale=scale, holdout=holdout)
    level = float(np.abs(Y[0]).max())
    np.testing.assert_allclose(hit.gap, base.gap, rtol=0, atol=1e-8 * level)
    np.testing.assert_allclose(hit.holdout_gap, base.holdout_gap, rtol=0,
                               atol=1e-8 * level)
    if method == "iscm":
        for lab, d in zip(inputs.affected_labels, deltas):
            np.testing.assert_allclose(hit.spillover_panel[lab] - base.spillover_panel[lab],
                                       d[holdout:], rtol=0, atol=1e-8 * level)


@settings(max_examples=25, deadline=None)
@given(seed=_SEEDS, method=st.sampled_from(sorted(_METHODS)),
       holdout=st.integers(0, 6), factor=st.floats(1e-2, 1e2),
       which=st.integers(1, 8))
def test_the_scale_free_estimate_does_not_see_a_controls_size(seed, method,
                                                               holdout, factor,
                                                               which):
    T0 = 30
    Y = _levels(seed, J=8, T=40)
    run = _METHODS[method]
    base = run(_inputs(Y, T0), scale=True, holdout=holdout)
    Ys = Y.copy()
    Ys[which] *= factor
    moved = run(_inputs(Ys, T0), scale=True, holdout=holdout)
    level = float(np.abs(Y[0]).max())
    np.testing.assert_allclose(moved.gap, base.gap, rtol=0, atol=1e-6 * level)
    assert moved.att == pytest.approx(base.att, abs=1e-6 * level)


@settings(max_examples=25, deadline=None)
@given(seed=_SEEDS, method=st.sampled_from(sorted(_METHODS)),
       scale=st.booleans(), holdout=st.integers(1, 6), shape_seed=_SEEDS)
def test_the_fits_do_not_read_the_held_out_window(seed, method, scale, holdout,
                                                  shape_seed):
    """Moving the treated unit's held-out outcomes moves its held-out gap by the
    same amount and nothing else: no weight, no post-period gap."""
    T0 = 30
    Y = _levels(seed, J=8, T=40)
    run = _METHODS[method]
    base = run(_inputs(Y, T0), scale=scale, holdout=holdout)
    bump = np.random.default_rng(shape_seed).normal(size=holdout) * Y[0].mean() / 10
    Yb = Y.copy()
    Yb[0, T0 - holdout:T0] += bump
    moved = run(_inputs(Yb, T0), scale=scale, holdout=holdout)
    level = float(np.abs(Y[0]).max())
    np.testing.assert_allclose(moved.gap, base.gap, rtol=0, atol=1e-8 * level)
    np.testing.assert_allclose(moved.holdout_gap - base.holdout_gap, bump, rtol=0,
                               atol=1e-8 * level)
    assert moved.donor_weights.keys() == base.donor_weights.keys()
    for k, v in base.donor_weights.items():
        assert moved.donor_weights[k] == pytest.approx(v, abs=1e-9 * max(1.0, abs(v)))


@pytest.mark.parametrize("method", sorted(_METHODS))
@pytest.mark.parametrize("scale", [False, True])
def test_the_held_out_window_is_reported_as_a_placebo(method, scale):
    T0, holdout = 30, 6
    inputs = _inputs(_levels(2, J=8, T=40), T0)
    fit = _METHODS[method](inputs, scale=scale, holdout=holdout)
    assert fit.holdout_periods == holdout and fit.scale_free is scale
    assert fit.holdout_gap.shape == fit.holdout_gap_scm.shape == (holdout,)
    assert fit.holdout_rmspe == pytest.approx(np.sqrt(np.mean(fit.holdout_gap ** 2)))
    # the pre-period synthetic carries the corrected held-out counterfactual
    np.testing.assert_allclose(inputs.Y[0, T0 - holdout:T0] - fit.holdout_gap,
                               fit.treated_synthetic_pre[T0 - holdout:], rtol=1e-12)
    # and the reported fit is the fitting window's
    fit_gap = inputs.Y[0, :T0 - holdout] - fit.treated_synthetic_pre[:T0 - holdout]
    assert fit.pre_rmspe == pytest.approx(np.sqrt(np.mean(fit_gap ** 2)), rel=1e-9)
    assert fit.gap.shape == (inputs.T1,)
    assert fit.bilevel_solver == ("scale-free" if scale else "outcome-only")


@pytest.mark.parametrize("scale", [False, True])
def test_the_restricted_pool_comparison_is_fit_the_same_way(scale):
    """iscm's cost-of-exclusion diagnostic refits the treated unit on the clean
    controls alone, on the same window and with the same scaling."""
    T0, holdout = 30, 6
    inputs = _inputs(_levels(2, J=8, T=40), T0)
    fit = run_iscm(inputs, scale=scale, holdout=holdout)
    clean = np.arange(inputs.p + 1, inputs.N)
    _, _, _, expected, _ = build_unit_sc(0, clean, inputs.Y, T0 - holdout,
                                         scale=scale)
    assert fit.pre_rmspe_restricted == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_the_corrected_placebo_clears_a_spillover_the_naive_one_carries(method):
    """No effect anywhere, a spillover on both affected units from the start of
    the held-out window: the naive held-out gap carries the spillover through
    their weight, and the corrected one is back near the noise."""
    T0, holdout = 30, 6
    Y = _levels(5, J=8, T=40, noise=0.01)
    inputs = _inputs(Y, T0)
    deltas = [0.4 * Y[int(lab[1:]), :T0].mean() * np.ones(40 - (T0 - holdout))
              for lab in inputs.affected_labels]
    hit = _METHODS[method](_inputs(_plant(inputs, Y, T0, T0 - holdout, deltas), T0),
                           scale=True, holdout=holdout)
    naive = np.sqrt(np.mean(hit.holdout_gap_scm ** 2))
    assert hit.holdout_rmspe < 0.25 * naive


# --------------------------------------------------------------------------- #
# Failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", sorted(_METHODS))
@pytest.mark.parametrize("holdout", [30, 31])
def test_a_held_out_window_that_leaves_nothing_to_fit_is_refused(method, holdout):
    inputs = _inputs(_levels(0, J=6, T=40), T0=30)
    with pytest.raises(MlsynthDataError, match="no period to fit"):
        _METHODS[method](inputs, holdout=holdout)


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_a_negative_held_out_window_is_refused(method):
    inputs = _inputs(_levels(0, J=6, T=40), T0=30)
    with pytest.raises(MlsynthDataError, match="holdout"):
        _METHODS[method](inputs, holdout=-1)


# --------------------------------------------------------------------------- #
# SPILLSYNTH exposes both
# --------------------------------------------------------------------------- #
def _cfg(Y, T0, **kw):
    cfg = dict(df=_frame(Y, T0), outcome="y", treat="d", unitid="unit",
               time="time", affected_units=["u01", "u02"], display_graphs=False)
    cfg.update(kw)
    return cfg


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_spillsynth_passes_both_options_through(method):
    Y, T0 = _levels(6, J=8, T=40), 30
    res = SPILLSYNTH(_cfg(Y, T0, method=method, scale_free=True,
                          holdout_periods=4)).fit()
    direct = _METHODS[method](_inputs(Y, T0), scale=True, holdout=4)
    fit = getattr(res, method)
    assert fit.scale_free is True and fit.holdout_periods == 4
    assert res.att == pytest.approx(direct.att, rel=1e-12)
    np.testing.assert_allclose(fit.holdout_gap, direct.holdout_gap, rtol=1e-12)
    assert len(res.time_series.counterfactual_outcome) == Y.shape[1]


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_spillsynth_by_default_neither_scales_nor_holds_out(method):
    Y, T0 = _levels(6, J=8, T=40), 30
    res = SPILLSYNTH(_cfg(Y, T0, method=method)).fit()
    fit = getattr(res, method)
    assert fit.scale_free is False and fit.holdout_periods == 0
    assert res.att == _METHODS[method](_inputs(Y, T0)).att


def test_the_scale_free_estimate_is_closer_when_the_treated_unit_is_largest():
    """No spillover, a lift of 5% on a unit twice the largest control: the
    simplex counterfactual cannot reach it, the scale-free one can."""
    Y, T0 = _levels(7, J=8, T=40, noise=0.005), 30
    truth = 0.05 * Y[0, T0:]
    Y[0, T0:] += truth
    plain = SPILLSYNTH(_cfg(Y, T0, method="iterative")).fit()
    scaled = SPILLSYNTH(_cfg(Y, T0, method="iterative", scale_free=True)).fit()
    assert abs(scaled.att - truth.mean()) < 0.1 * abs(plain.att - truth.mean())


@pytest.mark.parametrize("kw, word", [
    (dict(method="cd", scale_free=True), "scale_free"),
    (dict(method="grossi", holdout_periods=2), "holdout_periods"),
    (dict(method="iterative", scale_free=True, covariates=["y"]), "outcome-only"),
    (dict(method="iscm", holdout_periods=2, covariates=["y"]), "outcome-only"),
])
def test_the_options_are_refused_where_they_do_not_apply(kw, word):
    Y, T0 = _levels(0, J=6, T=40), 30
    with pytest.raises(MlsynthConfigError, match=word):
        SPILLSYNTH(_cfg(Y, T0, **kw))


def test_a_negative_held_out_window_is_refused_by_the_config():
    Y, T0 = _levels(0, J=6, T=40), 30
    with pytest.raises(MlsynthConfigError):
        SPILLSYNTH(_cfg(Y, T0, method="iscm", holdout_periods=-1))


def test_a_held_out_window_as_long_as_the_pre_period_is_reported():
    Y, T0 = _levels(0, J=6, T=40), 30
    with pytest.raises(MlsynthDataError, match="no period to fit"):
        SPILLSYNTH(_cfg(Y, T0, method="iterative", holdout_periods=30)).fit()
