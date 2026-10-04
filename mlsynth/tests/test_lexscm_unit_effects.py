r"""Per-treated-unit effects on the LEXSCM result.

``fit()`` reports the design's aggregate: one synthetic treated against one
synthetic control, and the gap between them. Abadie and Zhao's equation (10)
also speaks of each treated unit against its own synthetic control, and
:func:`unit_level_cumulative` takes exactly that -- period-by-unit gaps. Before
this, a caller had to rebuild those by hand from ``control_weight_dict`` and the
window layout, which is three chances to disagree with the estimator about which
periods are the fit window.

The tests below pin the helper that computes them and the field that stores
them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import LEXSCM
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.fast_scm_helpers.unit_effects import per_treated_unit_effects


@pytest.fixture
def panel():
    """A factor panel with a known treated/control split and window layout."""
    rng = np.random.default_rng(5)
    n, T = 14, 48
    F = np.cumsum(rng.normal(0, 0.4, (T, 2)), axis=0)
    L = rng.normal(size=(n, 2))
    Y = 100 + rng.normal(0, 4, n) + F @ L.T + rng.normal(0, 1.0, (T, n))
    return Y, [f"u{j:02d}" for j in range(n)]


# ------------------------------------------------------------------ smoke
def test_one_entry_per_treated_unit_with_weight(panel):
    Y, names = panel
    out = per_treated_unit_effects(
        Y, names, treated={"u00": 0.6, "u03": 0.4}, control={"u07": 0.5, "u09": 0.5},
        fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40), post_idx=np.arange(40, 48))
    assert [e.unit for e in out] == ["u00", "u03"]
    for e in out:
        assert np.isfinite(e.gap_fit).all() and np.isfinite(e.gap_blank).all()
        assert np.isfinite(e.gap_post).all()
        assert isinstance(e.approximability_ok, bool)


def test_window_lengths_follow_the_layout(panel):
    Y, names = panel
    e = per_treated_unit_effects(
        Y, names, treated={"u00": 1.0}, control={"u07": 0.5, "u09": 0.5},
        fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40),
        post_idx=np.arange(40, 48))[0]
    assert e.gap_fit.shape == (30,) and e.gap_blank.shape == (10,)
    assert e.gap_post.shape == (8,)


# ------------------------------------------------------------------- unit
def test_each_unit_gets_its_own_simplex_weights_over_the_control_pool(panel):
    Y, names = panel
    out = per_treated_unit_effects(
        Y, names, treated={"u00": 0.6, "u03": 0.4},
        control={"u07": 0.3, "u09": 0.3, "u11": 0.4},
        fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40), post_idx=np.arange(40, 48))
    for e in out:
        w = np.array(list(e.donor_weights.values()))
        assert set(e.donor_weights) <= {"u07", "u09", "u11"}
        assert (w >= -1e-9).all() and abs(w.sum() - 1.0) < 1e-7
    assert out[0].donor_weights != out[1].donor_weights, \
        "two different treated units should not get identical synthetic controls"


def test_the_fit_is_the_optimum_over_that_pool(panel):
    """The per-unit weights minimise fit-window SSE; a feasible neighbour is worse."""
    Y, names = panel
    pool = ["u07", "u09", "u11", "u12"]
    e = per_treated_unit_effects(
        Y, names, treated={"u00": 1.0}, control={k: 0.25 for k in pool},
        fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40),
        post_idx=np.arange(40, 48))[0]
    sse = float(e.gap_fit @ e.gap_fit)
    cols = [names.index(k) for k in pool]
    w = np.array([e.donor_weights.get(k, 0.0) for k in pool])
    rng = np.random.default_rng(0)
    for _ in range(40):                      # feasible simplex neighbours
        d = rng.normal(0, 0.05, len(pool)); d -= d.mean()
        cand = np.clip(w + d, 0, None)
        if cand.sum() <= 0: continue
        cand /= cand.sum()
        g = Y[np.arange(0, 30)][:, cols] @ cand - Y[np.arange(0, 30), names.index("u00")]
        assert float(g @ g) >= sse - 1e-7


def test_a_zero_weight_treated_unit_is_left_out(panel):
    Y, names = panel
    out = per_treated_unit_effects(
        Y, names, treated={"u00": 1.0, "u03": 0.0}, control={"u07": 1.0},
        fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40), post_idx=np.arange(40, 48))
    assert [e.unit for e in out] == ["u00"]


# -------------------------------------------------------------- edge cases
def test_a_single_donor_gives_that_donor_all_the_weight(panel):
    Y, names = panel
    e = per_treated_unit_effects(
        Y, names, treated={"u00": 1.0}, control={"u07": 1.0},
        fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40),
        post_idx=np.arange(40, 48))[0]
    assert e.donor_weights == pytest.approx({"u07": 1.0})
    expected = Y[np.arange(0, 30), names.index("u07")] - Y[np.arange(0, 30), names.index("u00")]
    assert e.gap_fit == pytest.approx(-expected)


def test_an_empty_post_window_is_allowed(panel):
    Y, names = panel
    e = per_treated_unit_effects(
        Y, names, treated={"u00": 1.0}, control={"u07": 0.5, "u09": 0.5},
        fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40),
        post_idx=np.arange(0, 0))[0]
    assert e.gap_post.shape == (0,)


# ----------------------------------------------------------------- failure
@pytest.mark.parametrize("kw,match", [
    (dict(treated={}), "at least one treated"),
    (dict(control={}), "at least one control"),
    (dict(treated={"nope": 1.0}), "not in the panel"),
    (dict(control={"nope": 1.0}), "not in the panel"),
])
def test_an_impossible_request_is_refused_with_a_reason(panel, kw, match):
    Y, names = panel
    args = dict(treated={"u00": 1.0}, control={"u07": 1.0},
                fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40),
                post_idx=np.arange(40, 48))
    args.update(kw)
    with pytest.raises(MlsynthDataError, match=match):
        per_treated_unit_effects(Y, names, **args)


def test_a_treated_unit_in_its_own_donor_pool_is_refused(panel):
    """It would reproduce itself exactly and report a zero gap."""
    Y, names = panel
    with pytest.raises(MlsynthDataError, match="both treated and control"):
        per_treated_unit_effects(
            Y, names, treated={"u00": 1.0}, control={"u00": 0.5, "u07": 0.5},
            fit_idx=np.arange(0, 30), blank_idx=np.arange(30, 40),
            post_idx=np.arange(40, 48))


# ------------------------------------------------------ the stored result
def test_fit_stores_one_effect_per_selected_unit(panel):
    Y, names = panel
    T, n = Y.shape
    df = pd.DataFrame([{"market": names[j], "week": t, "y": Y[t, j],
                        "eligible": 1, "post": int(t >= T - 8)}
                       for j in range(n) for t in range(T)])
    res = LEXSCM(dict(df=df, outcome="y", unitid="market", time="week",
                      candidate_col="eligible", post_col="post",
                      m=2, top_K=6, verbose=False)).fit()
    eff = res.unit_effects
    assert len(eff) == len(res.search.winner.treated_weight_dict)
    assert {e.unit for e in eff} == set(res.search.winner.treated_weight_dict)
    lay = res.panel.time
    for e in eff:
        assert e.gap_blank.shape == (lay.n_blank,)
        assert e.gap_post.shape == (lay.n_post,)
        # the full-precision weight, not the display-rounded dict: a design
        # reapplied from a rounded weight is a different design
        assert e.weight == pytest.approx(
            res.search.winner.treated_weight_dict_full[e.unit], rel=1e-12)
        assert e.weight != res.search.winner.treated_weight_dict[e.unit] or \
            e.weight == pytest.approx(round(e.weight, 3), abs=0)


def test_the_stored_effects_feed_the_cumulative_path_directly(panel):
    """The reason the field exists: no hand-rebuilding to reach eq. (10)."""
    from mlsynth.utils.fast_scm_helpers.post_inference import unit_level_cumulative
    Y, names = panel
    T, n = Y.shape
    df = pd.DataFrame([{"market": names[j], "week": t, "y": Y[t, j],
                        "eligible": 1, "post": int(t >= T - 8)}
                       for j in range(n) for t in range(T)])
    res = LEXSCM(dict(df=df, outcome="y", unitid="market", time="week",
                      candidate_col="eligible", post_col="post",
                      m=2, top_K=6, verbose=False)).fit()
    eff = res.unit_effects
    post = np.column_stack([e.gap_post for e in eff])
    blank = np.column_stack([e.gap_blank for e in eff])
    w = np.array([e.weight for e in eff]); w = w / w.sum()
    out = unit_level_cumulative(post, blank, w, level=0.90)
    assert len(out.per_unit) == len(eff)
    assert len(out.aggregate) == res.panel.time.n_post
    agg = out.aggregate[-1]
    # The band is a pivot, `observed - quantile(null)`, so the estimate is not
    # required to sit inside it: a placebo pool whose block sums are one-sided
    # shifts the whole interval off the estimate, which is the pool's bias
    # being subtracted and not an inconsistency.
    assert agg.lower <= agg.upper
    assert np.isfinite([agg.lower, agg.estimate, agg.upper]).all()
    assert agg.estimate == pytest.approx(
        float(np.sum(post @ (w / w.sum()))), rel=1e-9), \
        "the aggregate estimate is the weighted sum of the per-unit paths"
