"""The typed faces a design result carries for whatever reads it after the fit.

``DesignResult`` holds the assignment, the weights, the power and the realised
report. A step that re-estimates the effect afterwards (a spillover correction,
#932) also needs to know where the design's periods fall, which composites it
compared, and how it computed its inference, and it needs them typed:

* ``windows`` (:class:`DesignWindows`): the fitting window, the held-out
  (blank) window and the post window;
* ``composite`` and ``cluster_composites`` (:class:`DesignComposite`): the
  treated and control weights by unit label, for the whole design and for each
  of its clusters, with the units each one drew from;
* ``inference_settings`` (:class:`DesignInferenceSettings`): the settings the
  design's inference ran with.

The test that matters most: from those faces and the panel alone, MAREX's own
inference is reproduced exactly, for the aggregate and for every cluster.

Levels: smoke, unit invariants, edge, failure, properties.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import MAREX
from mlsynth.config_models import (DesignComposite, DesignInferenceSettings,
                                   DesignResult, DesignWindows)
from mlsynth.utils.marex_helpers.inference import compute_inference
from mlsynth.utils.marex_helpers.structures import MAREXInferenceSettings


def _panel(J=12, T=48, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    lam = rng.normal(size=(J, 2))
    Y = lam @ F.T + 0.4 * rng.standard_normal((J, T))
    return pd.DataFrame([{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]),
                          "grp": "a" if j < J // 2 else "b"}
                         for j in range(J) for t in range(T)])


_CFG = dict(outcome="y", unitid="unit", time="time", T0=40, inference=True,
            blank_periods=18, T_post=8, backend="exact")


def _fit(**kw):
    cfg = dict(df=_panel(), m_eq=2, **_CFG)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MAREX(cfg).fit()


def _matrix(res, labels):
    df = _panel()
    return df.pivot(index="unit", columns="time", values="y").loc[labels].to_numpy()


# --------------------------------------------------------------------------- #
# the models
# --------------------------------------------------------------------------- #

def test_windows_split_the_timeline_into_fit_blank_and_post():
    w = DesignWindows(T0=40, fit_end=22, post_periods=8)
    assert w.blank_periods == 18
    assert (w.fit, w.blank, w.post) == (slice(0, 22), slice(22, 40), slice(40, 48))


def test_windows_without_a_blank_window():
    w = DesignWindows(T0=30, fit_end=30, post_periods=0)
    assert w.blank_periods == 0 and w.post == slice(30, 30)


@pytest.mark.parametrize("kw, word", [
    (dict(T0=10, fit_end=11, post_periods=2), "fit_end"),
    (dict(T0=-1, fit_end=0, post_periods=2), "T0"),
    (dict(T0=10, fit_end=-1, post_periods=2), "fit_end"),
    (dict(T0=10, fit_end=5, post_periods=-1), "post_periods"),
    (dict(T0=10, fit_end=5, post_periods=2, time_labels=list(range(11))), "time_labels"),
])
def test_windows_refuse_an_inconsistent_timeline(kw, word):
    with pytest.raises(ValueError, match=word):
        DesignWindows(**kw)


def test_windows_are_frozen():
    w = DesignWindows(T0=10, fit_end=5, post_periods=2)
    with pytest.raises(Exception):
        w.T0 = 3


def test_a_composite_holds_disjoint_simplex_weights():
    c = DesignComposite(treated={"a": 0.25, "b": 0.75}, control={"c": 1.0},
                        members=["a", "b", "c", "d"])
    assert c.treated_labels == ["a", "b"] and c.control_labels == ["c"]
    assert c.pool == ["c", "d"]


@pytest.mark.parametrize("kw, word", [
    (dict(treated={"a": 1.0}, control={"a": 1.0}), "both"),
    (dict(treated={"a": 1.2, "b": -0.2}, control={"c": 1.0}), "negative"),
    (dict(treated={"a": 0.5}, control={"c": 1.0}), "sum"),
    (dict(treated={}, control={"c": 1.0}), "treated"),
    (dict(treated={"a": 1.0}, control={"c": 1.0}, members=["a"]), "members"),
])
def test_a_composite_refuses_weights_off_the_simplex(kw, word):
    with pytest.raises(ValueError, match=word):
        DesignComposite(**kw)


def test_a_composite_accepts_a_solvers_rounding_and_refuses_more():
    # MAREX's MIP backend returned control weights summing to 1 + 5.0e-8.
    DesignComposite(treated={"a": 1.0}, control={"c": 0.5, "d": 0.5 + 4.98912602e-8})
    with pytest.raises(ValueError, match="sum"):
        DesignComposite(treated={"a": 1.0}, control={"c": 0.5, "d": 0.5 + 1e-4})


def test_without_recorded_members_the_pool_is_the_control_composite():
    c = DesignComposite(treated={"a": 1.0}, control={"c": 0.4, "d": 0.6})
    assert c.members is None and c.pool == ["c", "d"]


def test_the_base_settings_pass_only_the_level():
    assert DesignInferenceSettings(alpha=0.05).inference_kwargs() == {"alpha": 0.05}


def test_inference_settings_need_a_level_in_the_unit_interval():
    assert DesignInferenceSettings(alpha=0.1).alpha == 0.1
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            DesignInferenceSettings(alpha=bad)


def test_a_design_without_the_faces_is_still_a_design():
    d = DesignResult()
    assert d.windows is None and d.composite is None
    assert d.cluster_composites is None and d.inference_settings is None


@settings(max_examples=200, deadline=None)
@given(T0=st.integers(0, 300), data=st.data())
def test_the_windows_partition_the_timeline(T0, data):
    fit_end = data.draw(st.integers(0, T0))
    post = data.draw(st.integers(0, 50))
    w = DesignWindows(T0=T0, fit_end=fit_end, post_periods=post)
    idx = np.arange(T0 + post)
    parts = np.concatenate([idx[w.fit], idx[w.blank], idx[w.post]])
    np.testing.assert_array_equal(parts, idx)
    assert w.blank_periods == T0 - fit_end


# --------------------------------------------------------------------------- #
# MAREX fills them
# --------------------------------------------------------------------------- #

def test_marex_fills_every_face():
    res = _fit()
    assert isinstance(res.windows, DesignWindows)
    assert isinstance(res.composite, DesignComposite)
    assert isinstance(res.inference_settings, MAREXInferenceSettings)
    assert isinstance(res.inference_settings, DesignInferenceSettings)
    assert set(res.cluster_composites) == set(res.clusters)


def test_the_windows_are_the_designs():
    res = _fit()
    w = res.windows
    assert (w.T0, w.fit_end, w.blank_periods, w.post_periods) == (40, 22, 18, 8)
    assert list(w.time_labels) == list(range(48))
    # the free-form keys stay, as mirrors of the typed face
    assert (res.metadata["T0"], res.metadata["blank_periods"]) == (w.T0, w.blank_periods)


def test_the_composite_is_the_aggregate_design():
    res = _fit()
    g, c = res.globres, res.composite
    labels = sorted(_panel()["unit"].unique())
    tw = {u: float(x) for u, x in zip(labels, g.treated_weights_agg) if x > 1e-8}
    cw = {u: float(x) for u, x in zip(labels, g.control_weights_agg) if x > 1e-8}
    assert c.treated == pytest.approx(tw) and c.control == pytest.approx(cw)
    assert c.members == labels
    assert c.treated == pytest.approx(res.design_weights.donor_weights)


def test_each_cluster_composite_is_that_clusters_design():
    res = _fit(cluster="grp", m_eq=1)
    assert len(res.cluster_composites) == 2
    for label, cl in res.clusters.items():
        c = res.cluster_composites[label]
        assert c.treated == pytest.approx(cl.unit_weight_map["Treated"])
        assert c.control == pytest.approx(cl.unit_weight_map["Control"])
        assert c.members == [str(u) for u in cl.members]


def test_the_settings_are_the_ones_the_inference_ran_with():
    res = _fit(global_test="cyclic", alpha=0.1, cumulative_band=True,
               cumulative_method="inversion")
    s = res.inference_settings
    assert (s.alpha, s.T_post, s.global_test) == (0.1, 8, "cyclic")
    assert (s.cumulative_band, s.cumulative_method) == (True, "inversion")


def test_the_drawn_subsets_settings_are_recorded():
    # 26 blank and post periods choose 8 in far more than 1,000 ways, so eq. (17)
    # draws its subsets, and reproducing it needs the count and the seed.
    s = _fit().inference_settings
    assert (s.max_combinations, s.random_state) == (1000, 42)


def test_without_inference_there_are_no_settings_but_the_design_is_still_described():
    res = _fit(inference=False)
    assert res.inference_settings is None
    assert res.windows is not None and res.composite is not None


# --------------------------------------------------------------------------- #
# the faces alone reproduce MAREX's inference
# --------------------------------------------------------------------------- #

def _reproduce(res, composite):
    w, s = res.windows, res.inference_settings
    T = w.T0 + w.post_periods
    labels = composite.members
    Y = _matrix(res, labels)[:, :T]
    pos = {u: i for i, u in enumerate(labels)}
    wt = np.zeros(len(labels)); wc = np.zeros(len(labels))
    for u, x in composite.treated.items():
        wt[pos[u]] = x
    for u, x in composite.control.items():
        wc[pos[u]] = x
    pool = Y[[pos[u] for u in composite.pool]]
    return compute_inference(wt @ Y, wc @ Y, w.T0, w.fit_end, w.blank_periods,
                             control_outcomes=pool, **s.inference_kwargs())


def _same(a, b):
    assert a.global_p_value == b.global_p_value
    np.testing.assert_allclose(a.ci, b.ci, rtol=0, atol=1e-10)
    np.testing.assert_allclose(a.per_period_pvals, b.per_period_pvals, atol=0)
    if b.cumulative_band is not None:
        assert (a.cumulative_band.lower, a.cumulative_band.upper) == pytest.approx(
            (b.cumulative_band.lower, b.cumulative_band.upper), abs=1e-9)


@pytest.mark.parametrize("extra", [
    {},
    {"global_test": "cyclic"},
    {"cumulative_band": True, "cumulative_method": "inversion"},
    {"cumulative_band": True, "cumulative_block": 2, "cumulative_n_sim": 500},
    {"alpha": 0.1},
])
def test_the_faces_reproduce_the_aggregate_inference(extra):
    res = _fit(**extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _same(_reproduce(res, res.composite), res.globres.inference)


def test_the_faces_reproduce_every_clusters_inference():
    res = _fit(cluster="grp", m_eq=1, cumulative_band=True, cumulative_method="inversion")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, cl in res.clusters.items():
            _same(_reproduce(res, res.cluster_composites[label]), cl.inference)
        _same(_reproduce(res, res.composite), res.globres.inference)
