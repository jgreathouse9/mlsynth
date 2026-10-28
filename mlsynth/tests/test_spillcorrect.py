"""SPILLCORRECT: correcting known contaminated controls in a fitted design (#932).

A design's estimate is its treated composite less its control composite. A
control unit ``k`` carrying a spillover ``gamma_k(t)`` biases that estimate by
``-v_k gamma_k(t)``, with ``v_k`` the unit's control weight. SPILLCORRECT keeps
the design (both composites) and rebuilds each contaminated, weighted control
from the clean controls:

* ``iterative`` (Melnychuk 2024): each contaminated control's own synthetic
  control replaces its outcomes from the start of the correction on;
* ``inclusive`` (Di Stefano and Mellace 2024): each contaminated control's
  synthetic control draws on the clean controls, the treated composite and the
  other contaminated controls, and the cross-weight system recovers the effect
  and every spillover;
* ``restricted``: the design re-solved with the contaminated controls dropped
  from the control pool, the comparator Di Stefano and Mellace recommend
  reporting beside a correction.

By default each fit is scale-free and stops at the end of the design's fitting
window, and the correction starts there, so the design's held-out (blank)
periods, on which its band and test are calibrated, carry the correction's own
prediction error. The design's inference is then recomputed on the corrected
panel with the design's own settings.

Levels: smoke, unit invariants, edge, failure, properties. The mutants are
catalogued under ``spillcorrect-*`` in ``tools/mutation/targets.toml``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import MAREX, SPILLCORRECT
from mlsynth.config_models import BaseEstimatorResults, DesignResult
from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)
from mlsynth.utils.marex_helpers.inference import compute_inference
from mlsynth.utils.spillcorrect_helpers.correct import (inclusive_correction,
                                                        iterative_correction)
from mlsynth.utils.spillcorrect_helpers.plotter import plot_spillcorrect
from mlsynth.utils.spillcorrect_helpers.structures import (SpillCorrectEstimate,
                                                           SpillCorrectResults)

# 20 blank periods: the pointwise band needs 19 at the 5% level
T0, BLANK, POST = 48, 20, 12
_DESIGN = dict(outcome="y", unitid="unit", time="time", T0=T0,
               blank_periods=BLANK, T_post=POST, m_eq=2, backend="exact",
               inference=True)
# the design's own configuration, without what the correction supplies
_DESIGN_CONFIG = {k: v for k, v in _DESIGN.items() if k not in ("outcome", "unitid", "time")}


def _geo(J=14, T=T0 + POST, seed=0, noise=0.02):
    """Weekly sales in levels: two demand factors scaled by each region's size."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    f1 = 100 + 8 * np.sin(2 * np.pi * t / 13) + np.cumsum(rng.normal(0, 0.5, T))
    f2 = np.cumsum(rng.normal(0, 1, T))
    size = np.exp(rng.uniform(-0.7, 0.7, J))
    load = rng.normal(0, 0.3, J)
    Y = size[:, None] * (f1 + load[:, None] * f2) * (1 + noise * rng.normal(size=(J, T)))
    return pd.DataFrame([{"unit": f"g{j:02d}", "time": s, "y": float(Y[j, s])}
                         for j in range(J) for s in range(T)])


def _fit_design(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MAREX(dict(df=df, **{**_DESIGN, **kw})).fit()


@pytest.fixture(scope="module")
def base():
    return _geo()


@pytest.fixture(scope="module")
def design(base):
    return _fit_design(base)


@pytest.fixture(scope="module")
def sparse(base):
    """A weakly-targeted design, which leaves some controls without weight."""
    d = _fit_design(base, design="weakly_targeted")
    assert [u for u in d.composite.pool if u not in d.composite.control]
    return d


def _heaviest(design, n):
    c = design.composite.control
    return sorted(c, key=c.get, reverse=True)[:n]


def _shift(df, unit, start, path):
    out = df.copy()
    rows = (out["unit"] == unit) & (out["time"] >= start)
    out.loc[rows, "y"] = out.loc[rows, "y"].to_numpy() + np.asarray(path)
    return out


def _lift(df, design, share=0.05):
    out = df.copy()
    for u in design.composite.treated:
        rows = (out["unit"] == u) & (out["time"] >= T0)
        out.loc[rows, "y"] *= 1 + share
    return out


def _correct(df, design, affected, **kw):
    cfg = dict(df=df, outcome="y", unitid="unit", time="time", design=design,
               affected_units=list(affected))
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLCORRECT(cfg).fit()


def _matrix(df, labels):
    return df.pivot(index="unit", columns="time", values="y").loc[labels].to_numpy().copy()


# --------------------------------------------------------------------------- #
# Smoke and the result contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["iterative", "inclusive"])
def test_a_correction_is_an_effect_result(base, design, method):
    res = _correct(_lift(base, design), design, _heaviest(design, 2), method=method)
    assert isinstance(res, SpillCorrectResults) and isinstance(res, BaseEstimatorResults)
    assert np.isfinite(res.att) and res.att == pytest.approx(res.effects.att)
    assert res.att == pytest.approx(res.corrected.att)
    assert res.time_series.counterfactual_outcome.shape == (T0 + POST,)
    assert res.inference.p_value == res.corrected.p_value
    assert res.method_details.method_name == f"SPILLCORRECT/{method}"
    assert not res.weights.is_empty
    for name in res.weights.weights_at or []:
        assert getattr(res, name) is not None
    assert res.model_dump(include={"effects", "inference"})["effects"]["att"] is not None
    for arm in (res.corrected, res.uncorrected):
        assert isinstance(arm, SpillCorrectEstimate)
        assert arm.effect.shape == arm.counterfactual.shape == (POST,)
        assert arm.ci.shape == (POST, 2) and arm.placebo.shape == (BLANK,)


def test_the_plot_returns_a_figure(base, design):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = _correct(_lift(base, design), design, _heaviest(design, 1))
    fig = plot_spillcorrect(res)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The uncorrected arm is the design's own estimate
# --------------------------------------------------------------------------- #
def test_the_uncorrected_arm_reproduces_the_designs_inference(base, design):
    res = _correct(base, design, _heaviest(design, 1))
    g = design.globres
    np.testing.assert_allclose(res.uncorrected.effect,
                               (g.synthetic_treated - g.synthetic_control)[T0:], atol=1e-10)
    assert res.uncorrected.p_value == g.inference.global_p_value
    np.testing.assert_allclose(res.uncorrected.ci, g.inference.ci[T0:], atol=1e-10)


def test_the_corrected_inference_is_the_designs_inference_on_the_corrected_panel(base, design):
    """The corrected arm is what the design's own inference reports once the
    contaminated controls' outcomes are replaced by their rebuilt series."""
    res = _correct(_lift(base, design), design, _heaviest(design, 2))
    c, w = design.composite, design.windows
    Y = _matrix(_lift(base, design), c.members)
    pos = {u: i for i, u in enumerate(c.members)}
    start = w.fit_end
    for u, path in res.spillover.items():
        Y[pos[u], start:] -= path
    wt = np.array([c.treated.get(u, 0.0) for u in c.members])
    wc = np.array([c.control.get(u, 0.0) for u in c.members])
    inf = compute_inference(wt @ Y, wc @ Y, w.T0, w.fit_end, w.blank_periods,
                            control_outcomes=Y[[pos[u] for u in c.pool]],
                            **design.inference_settings.inference_kwargs())
    assert res.corrected.p_value == inf.global_p_value
    np.testing.assert_allclose(res.corrected.ci, inf.ci[T0:], atol=1e-9)
    np.testing.assert_allclose(res.corrected.effect, (wt @ Y - wc @ Y)[T0:], atol=1e-9)


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #
@settings(max_examples=20, deadline=None)
@given(method=st.sampled_from(["iterative", "inclusive"]), scale=st.booleans(),
       window=st.sampled_from(["blank_and_post", "post"]),
       amp=st.floats(-0.5, 0.5), shape_seed=st.integers(0, 2**32 - 1))
def test_the_estimate_ignores_the_spillovers_size_and_sign(base, design, method,
                                                           scale, window, amp,
                                                           shape_seed):
    """Whatever reaches the contaminated controls from the start of the
    correction on, the corrected effect, band and p-value do not move: the
    iterative method replaces those outcomes, and in the inclusive system a path
    added to unit ``k`` returns whole to its own spillover estimate."""
    ks = _heaviest(design, 2)
    start = design.windows.fit_end if window == "blank_and_post" else T0
    world = _lift(base, design)
    hit = world
    rng = np.random.default_rng(shape_seed)
    for k in ks:
        level = float(base.loc[base.unit == k, "y"].mean())
        hit = _shift(hit, k, start, amp * level * (1 + 0.3 * rng.normal(size=T0 + POST - start)))
    a = _correct(world, design, ks, method=method, scale=scale, window=window)
    b = _correct(hit, design, ks, method=method, scale=scale, window=window)
    scale_ = float(np.abs(a.time_series.observed_outcome).max())
    np.testing.assert_allclose(b.corrected.effect, a.corrected.effect, rtol=0, atol=1e-8 * scale_)
    np.testing.assert_allclose(b.corrected.placebo, a.corrected.placebo, rtol=0, atol=1e-8 * scale_)
    np.testing.assert_allclose(b.corrected.ci, a.corrected.ci, rtol=0, atol=1e-8 * scale_)
    assert b.corrected.p_value == a.corrected.p_value


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), factor=st.floats(0.2, 5.0),
       amp=st.floats(-50.0, 50.0), start=st.integers(4, 20))
def test_a_control_in_the_clean_controls_span_is_rebuilt_exactly(seed, factor, amp, start):
    """A contaminated control that is a multiple of a clean one comes back
    exactly, spillover and all removed, under the scale-free fit."""
    rng = np.random.default_rng(seed)
    T = start + 10
    clean = 50 + np.cumsum(rng.normal(size=(4, T)), axis=1)
    truth = factor * clean[2]
    observed = truth.copy()
    observed[start:] += amp * (1 + rng.normal(size=T - start))
    Y = np.vstack([clean, observed])
    fix = iterative_correction(Y, ks=[4], clean=[0, 1, 2, 3], start=start, scale=True)
    np.testing.assert_allclose(fix.cleaned[4], truth, rtol=1e-9)
    np.testing.assert_allclose(fix.spillover[0], observed[start:] - truth[start:],
                               rtol=0, atol=1e-8 * abs(truth).max())


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), m=st.integers(1, 3), start=st.integers(6, 20),
       amp=st.floats(-20.0, 20.0))
def test_the_two_corrections_agree_when_the_clean_controls_span_every_control(seed, m,
                                                                               start, amp):
    """When each contaminated control is a combination of the clean ones, both
    rebuild it exactly, so both recover the same treated effect."""
    rng = np.random.default_rng(seed)
    T, J = start + 8, 5
    clean = 50 + np.cumsum(rng.normal(size=(J, T)), axis=1)
    mix = rng.dirichlet(np.ones(J), size=m)
    truth = mix @ clean
    observed = truth + np.vstack([np.r_[np.zeros(start), amp * rng.normal(size=T - start)]
                                  for _ in range(m)])
    # the treated composite outside the clean span, so no exact rebuild leans on it
    treated = 200 + np.cumsum(rng.normal(size=T)) + np.r_[np.zeros(start), 3.0 * np.ones(T - start)]
    Y = np.vstack([clean, observed])
    ks = list(range(J, J + m))
    v = np.full(m, 0.6 / m)
    control = 0.4 * clean[2] + v @ observed
    it = iterative_correction(Y, ks=ks, clean=list(range(J)), start=start, scale=False)
    inc = inclusive_correction(Y, ks=ks, clean=list(range(J)), start=start, scale=False,
                               treated_series=treated, gap=treated - control, v=v)
    np.testing.assert_allclose(it.cleaned[ks], truth, rtol=1e-6)
    np.testing.assert_allclose(inc.cleaned[ks], truth, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Units, windows and arms
# --------------------------------------------------------------------------- #
def test_the_correction_reads_the_spillover_off_the_contaminated_controls(base, design):
    """A planted spillover is what comes back as the unit's spillover path, to
    the rebuilding fit's error."""
    k = _heaviest(design, 1)[0]
    start = design.windows.fit_end
    level = float(base.loc[base.unit == k, "y"].mean())
    planted = 0.3 * level * np.ones(T0 + POST - start)
    res = _correct(_shift(_lift(base, design), k, start, planted), design, [k])
    assert res.spillover[k].shape == planted.shape
    err = np.abs(res.spillover[k] - planted).mean()
    assert err < 0.1 * planted.mean()
    assert res.cleaning_rmspe[k] < 0.1 * level


def test_the_correction_removes_the_bias_the_spillover_adds(base, design):
    """With a 5% lift and a large spillover on the two heaviest controls, the
    uncorrected estimate is off by about their weight times the spillover; the
    corrected one is back near the clean estimate."""
    ks = _heaviest(design, 2)
    start = design.windows.fit_end
    world = _lift(base, design)
    clean = _correct(world, design, ks).uncorrected.att
    hit = world
    for k in ks:
        level = float(base.loc[base.unit == k, "y"].mean())
        hit = _shift(hit, k, start, 0.4 * level * np.ones(T0 + POST - start))
    res = _correct(hit, design, ks)
    assert abs(res.uncorrected.att - clean) > 5 * abs(res.corrected.att - clean)


def test_the_post_window_fits_on_the_whole_pre_period(base, design):
    res = _correct(_lift(base, design), design, _heaviest(design, 1), window="post")
    assert res.correction_start == T0
    assert all(p.shape == (POST,) for p in res.spillover.values())


def test_the_default_window_starts_at_the_end_of_the_fitting_window(base, design):
    res = _correct(_lift(base, design), design, _heaviest(design, 1))
    assert res.correction_start == design.windows.fit_end == T0 - BLANK
    assert all(p.shape == (BLANK + POST,) for p in res.spillover.values())
    assert (res.method, res.scale, res.window) == ("iterative", True, "blank_and_post")


def test_the_inclusive_method_reports_each_controls_load_on_the_treated_composite(base, design):
    ks = _heaviest(design, 2)
    res = _correct(_lift(base, design), design, ks, method="inclusive")
    assert set(res.composite_weight) == set(ks)
    assert all(np.isfinite(x) for x in res.composite_weight.values())
    assert res.omega_det is not None and abs(res.omega_det) > 0
    it = _correct(_lift(base, design), design, ks)
    assert it.composite_weight is None and it.omega_det is None


def test_the_restricted_arm_is_the_design_re_solved_without_the_controls(base, design):
    ks = _heaviest(design, 2)
    world = _lift(base, design)
    res = _correct(world, design, ks, design_config=_DESIGN_CONFIG)
    assert res.restricted is not None
    labels = design.composite.members
    A = pd.DataFrame(0.0, index=labels, columns=labels)
    A.loc[ks, list(design.composite.treated)] = 1.0
    A.loc[list(design.composite.treated), ks] = 1.0
    ref = _fit_design(world, to_be_treated=list(design.composite.treated), adjacency=A,
                      exclude_bordering_donors=True)
    assert all(ref.composite.control.get(k, 0.0) == 0.0 for k in ks)
    g = ref.globres
    np.testing.assert_allclose(res.restricted.effect,
                               (g.synthetic_treated - g.synthetic_control)[T0:], atol=1e-10)
    assert res.restricted.p_value == g.inference.global_p_value
    assert res.restricted_weights == pytest.approx(ref.composite.control)


def test_without_the_design_config_there_is_no_restricted_arm(base, design):
    res = _correct(_lift(base, design), design, _heaviest(design, 1))
    assert res.restricted is None and res.restricted_weights is None


def test_the_restricted_method_reports_the_restricted_arm(base, design):
    ks = _heaviest(design, 1)
    res = _correct(_lift(base, design), design, ks, method="restricted",
                   design_config=_DESIGN_CONFIG)
    assert res.att == res.restricted.att
    assert res.method_details.method_name == "SPILLCORRECT/restricted"


def test_the_global_test_can_be_changed(base, design):
    res = _correct(_lift(base, design), design, _heaviest(design, 1), global_test="cyclic")
    p = res.corrected.p_value
    assert 0 < p <= 1 and abs(p * (BLANK + POST) - round(p * (BLANK + POST))) < 1e-9


def test_a_design_without_inference_still_gets_its_corrected_effect(base):
    d = _fit_design(base, inference=False)
    res = _correct(_lift(base, d), d, _heaviest(d, 1))
    assert np.isfinite(res.att)
    assert res.corrected.ci is None and res.corrected.p_value is None
    assert res.inference is None or res.inference.p_value is None


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_an_unweighted_affected_control_is_kept_out_of_the_donors_not_corrected(base, sparse):
    design = sparse
    pool = [u for u in design.composite.pool if u not in design.composite.control]
    k, z = _heaviest(design, 1)[0], pool[0]
    with pytest.warns(UserWarning, match="no control weight"):
        res = SPILLCORRECT(dict(df=_lift(base, design), outcome="y", unitid="unit",
                                time="time", design=design, affected_units=[k, z])).fit()
    assert res.corrected_units == [k] and res.skipped_units == [z]
    assert res.n_corrected == 1 and z not in res.spillover
    assert z not in res.donor_units


def test_with_no_weighted_affected_control_there_is_nothing_to_correct(base, sparse):
    design = sparse
    pool = [u for u in design.composite.pool if u not in design.composite.control]
    with pytest.warns(UserWarning, match="no control weight"):
        res = SPILLCORRECT(dict(df=_lift(base, design), outcome="y", unitid="unit",
                                time="time", design=design, affected_units=pool[:1])).fit()
    assert res.n_corrected == 0
    np.testing.assert_array_equal(res.corrected.effect, res.uncorrected.effect)


def test_correcting_many_controls_points_to_design_time_exclusion(base, design):
    ks = _heaviest(design, 4)
    with pytest.warns(UserWarning, match="exclude"):
        SPILLCORRECT(dict(df=_lift(base, design), outcome="y", unitid="unit",
                          time="time", design=design, affected_units=ks)).fit()


def test_a_few_light_controls_draw_no_warning(base, design):
    k = _heaviest(design, 1)[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SPILLCORRECT(dict(df=_lift(base, design), outcome="y", unitid="unit",
                          time="time", design=design, affected_units=[k])).fit()
    assert not [w for w in caught if "SPILLCORRECT" in str(w.message)]


def test_a_singular_inclusive_system_is_named():
    """A control carrying the whole control composite whose own synthetic is the
    treated composite: ``det Omega = 1 - v l = 0``."""
    rng = np.random.default_rng(0)
    T, start = 30, 20
    clean = 50 + np.cumsum(rng.normal(size=(3, T)), axis=1)
    treated = clean[0] * 1.0
    k = treated * 1.0
    Y = np.vstack([clean, k])
    with pytest.raises(MlsynthEstimationError, match="singular"):
        inclusive_correction(Y, ks=[3], clean=[1, 2], start=start, scale=False,
                             treated_series=treated, gap=treated - k, v=np.array([1.0]))


# --------------------------------------------------------------------------- #
# Failure
# --------------------------------------------------------------------------- #
def _cfg(base, design, **kw):
    cfg = dict(df=base, outcome="y", unitid="unit", time="time", design=design,
               affected_units=_heaviest(design, 1))
    cfg.update(kw)
    return cfg


def test_a_treated_unit_cannot_be_declared_affected(base, design):
    t = list(design.composite.treated)[0]
    with pytest.raises(MlsynthConfigError, match="treated"):
        SPILLCORRECT(_cfg(base, design, affected_units=[t]))


@pytest.mark.parametrize("affected, word", [([], "at least one"),
                                            (["nowhere"], "not among"),
                                            (["g01", "g01"], "twice")])
def test_the_affected_units_must_name_distinct_design_units(base, design, affected, word):
    with pytest.raises(MlsynthConfigError, match=word):
        SPILLCORRECT(_cfg(base, design, affected_units=affected))


def test_a_design_without_its_faces_is_refused(base):
    with pytest.raises(MlsynthConfigError, match="windows"):
        SPILLCORRECT(dict(df=base, outcome="y", unitid="unit", time="time",
                          design=DesignResult(), affected_units=["g01"]))


def test_the_restricted_method_needs_the_design_config(base, design):
    with pytest.raises(MlsynthConfigError, match="design_config"):
        SPILLCORRECT(_cfg(base, design, method="restricted"))


@pytest.mark.parametrize("key", ["df", "to_be_treated", "adjacency"])
def test_the_design_config_cannot_carry_what_the_correction_sets(base, design, key):
    with pytest.raises(MlsynthConfigError, match=key):
        SPILLCORRECT(_cfg(base, design, design_config={**_DESIGN_CONFIG, key: 1}))


def test_a_new_global_test_needs_the_designs_inference(base):
    d = _fit_design(base, inference=False)
    with pytest.raises(MlsynthConfigError, match="inference"):
        SPILLCORRECT(_cfg(base, d, global_test="cyclic"))


def test_a_panel_the_design_does_not_reproduce_on_is_refused(base, design):
    """The design was fit on its fitting window; a panel that moved there is a
    different panel, and the design's weights are not its design."""
    moved = _shift(base, _heaviest(design, 1)[0], 0, 5.0 * np.ones(T0 + POST))
    with pytest.raises(MlsynthDataError, match="reproduce"):
        SPILLCORRECT(_cfg(moved, design)).fit()


def test_a_panel_missing_a_design_unit_is_refused(base, design):
    k = _heaviest(design, 1)[0]
    with pytest.raises(MlsynthDataError, match="missing"):
        SPILLCORRECT(_cfg(base[base.unit != k].reset_index(drop=True), design,
                          affected_units=_heaviest(design, 2)[1:])).fit()


def test_a_panel_shorter_than_the_design_is_refused(base, design):
    short = base[base.time < T0 + POST - 1].reset_index(drop=True)
    with pytest.raises(MlsynthDataError, match="periods"):
        SPILLCORRECT(_cfg(short, design)).fit()


def test_every_control_affected_leaves_no_clean_donor(base, design):
    every = list(design.composite.pool)
    with pytest.raises(MlsynthDataError, match="clean"):
        SPILLCORRECT(_cfg(base, design, affected_units=every)).fit()


# --------------------------------------------------------------------------- #
# The estimator's surface
# --------------------------------------------------------------------------- #
def test_a_config_object_and_a_dict_give_the_same_estimate(base, design):
    from mlsynth.utils.spillcorrect_helpers.config import SPILLCORRECTConfig
    world = _lift(base, design)
    cfg = _cfg(world, design)
    a = SPILLCORRECT(SPILLCORRECTConfig(**cfg)).fit()
    b = SPILLCORRECT(cfg).fit()
    assert a.att == b.att and a.corrected.p_value == b.corrected.p_value


def test_an_unexpected_failure_is_reported_as_an_estimation_error(base, design, monkeypatch):
    import mlsynth.estimators.spillcorrect as module

    def boom(config):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "run_spillcorrect", boom)
    with pytest.raises(MlsynthEstimationError, match="boom"):
        SPILLCORRECT(_cfg(base, design)).fit()


def test_only_the_designs_own_inference_is_recomputed(base, design):
    from mlsynth.config_models import DesignInferenceSettings
    other = design.model_copy(update={"inference_settings": DesignInferenceSettings(alpha=0.05)})
    with pytest.raises(MlsynthConfigError, match="MAREX's inference"):
        SPILLCORRECT(_cfg(base, other)).fit()


def test_the_restricted_arm_runs_the_global_test_asked_for(base, design):
    res = _correct(_lift(base, design), design, _heaviest(design, 1),
                   design_config=_DESIGN_CONFIG, global_test="cyclic")
    p = res.restricted.p_value
    assert abs(p * (BLANK + POST) - round(p * (BLANK + POST))) < 1e-9


def test_the_plot_draws_every_arm_on_the_callers_axes(base, design):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = _correct(_lift(base, design), design, _heaviest(design, 1),
                   design_config=_DESIGN_CONFIG)
    fig, ax = plt.subplots()
    assert plot_spillcorrect(res, ax=ax) is fig
    assert len(ax.get_legend().get_texts()) == 3
    plt.close(fig)


@pytest.mark.parametrize("kw", [dict(method="waterfall"), dict(window="pre"), dict(scale="often")])
def test_a_value_outside_a_fields_type_is_a_config_error(base, design, kw):
    with pytest.raises(MlsynthConfigError, match="Invalid SPILLCORRECT configuration"):
        SPILLCORRECT(_cfg(base, design, **kw))


def test_the_inversion_band_refits_on_the_corrected_controls(base):
    """MAREX's inversion band refits the counterfactual on the control
    outcomes, so the corrected band reads the corrected controls."""
    d = _fit_design(base, cumulative_band=True, cumulative_method="inversion")
    ks = _heaviest(d, 2)
    start = d.windows.fit_end
    hit = _lift(base, d)
    for k in ks:
        level = float(base.loc[base.unit == k, "y"].mean())
        hit = _shift(hit, k, start, 0.4 * level * np.ones(T0 + POST - start))
    res = _correct(hit, d, ks)
    c, w = d.composite, d.windows
    Y = _matrix(hit, c.members)
    pos = {u: i for i, u in enumerate(c.members)}
    for u, path in res.spillover.items():
        Y[pos[u], start:] -= path
    wt = np.array([c.treated.get(u, 0.0) for u in c.members])
    wc = np.array([c.control.get(u, 0.0) for u in c.members])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inf = compute_inference(wt @ Y, wc @ Y, w.T0, w.fit_end, w.blank_periods,
                                control_outcomes=Y[[pos[u] for u in c.pool]],
                                **d.inference_settings.inference_kwargs())
    got, want = res.corrected.inference.cumulative_band, inf.cumulative_band
    assert (got.lower, got.upper) == pytest.approx((want.lower, want.upper), abs=1e-8)


@pytest.mark.parametrize("seed", range(6))
def test_the_inclusive_estimates_solve_the_cross_weight_system(seed):
    """Each contaminated control's gap is its spillover less its loads on the
    treated composite's effect and on the other controls' spillovers, with the
    loads read off its own synthetic control."""
    from mlsynth.utils.spillsynth_helpers.iscm.weights import build_unit_sc
    rng = np.random.default_rng(seed)
    T, start, J = 30, 20, 4
    clean = 50 + np.cumsum(rng.normal(size=(J, T)), axis=1)
    treated = 50 + np.cumsum(rng.normal(size=T))
    ks_series = np.vstack([0.6 * treated + 0.4 * clean[i] + rng.normal(0, 0.3, T)
                           for i in range(2)])
    ks_series[:, start:] += rng.normal(0, 2, (2, T - start))
    Y = np.vstack([clean, ks_series])
    ks, v = [J, J + 1], np.array([0.3, 0.2])
    control = 0.5 * clean[0] + v @ ks_series
    fix = inclusive_correction(Y, ks=ks, clean=list(range(J)), start=start, scale=True,
                               treated_series=treated, gap=treated - control, v=v)
    assert np.all(fix.composite_weight > 0.05)
    theta = (treated - control)[start:] + v @ fix.spillover
    aug = np.vstack([Y, treated[None, :]])
    for i, k in enumerate(ks):
        other = ks[1 - i]
        pool = np.array(list(range(J)) + [Y.shape[0], other])
        w, _, gap_k, _, _ = build_unit_sc(k, pool, aug, start, scale=True)
        rhs = fix.spillover[i] - w[J] * theta - w[J + 1] * fix.spillover[1 - i]
        np.testing.assert_allclose(gap_k[start:], rhs, rtol=0, atol=1e-8)
        assert fix.composite_weight[i] == pytest.approx(w[J])
