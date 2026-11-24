"""SYNDES ingestion through ``datautils.geoex_dataprep`` (#1138, part 1).

SYNDES is a design estimator: it chooses which units to treat, so the panel
has no treatment column and ``dataprep`` cannot split it. ``geoex_dataprep`` is
the design-side sibling: a balanced ``time x unit`` outcome matrix, refusing a
duplicate or absent unit-period and a missing or non-finite outcome, and an
optional ``post_col`` whose post periods must be one block at the end of the
panel.

The last rule is a change. SYNDES used to accept post periods anywhere, and its
permutation inference and plotter stack ``Y_pre`` over ``Y_post``, which puts
the periods out of time order when a post period sits between pre periods.

``_reference`` is the old pivot, kept as the oracle the new path must match on
every panel both accept.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.syndes_helpers import setup as SETUP
from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs
from mlsynth.utils.syndes_helpers.structures import SYNDESInputs

COLS = dict(outcome="y", unitid="unit", time="time")


def _reference(df, outcome, unitid, time, T0=None, post_col=None):
    """The pre-#1138 pivot and split, for panels both paths accept."""
    ordered = df.sort_values([time, unitid])
    Ywide = ordered.pivot(index=time, columns=unitid, values=outcome).sort_index()
    t = np.asarray(Ywide.index.to_list())
    u = np.asarray(Ywide.columns.to_list())
    Y = Ywide.to_numpy(dtype=float)
    if post_col is not None:
        post = (ordered[[time, post_col]].drop_duplicates(subset=[time])
                .set_index(time).reindex(t)[post_col].astype(bool).to_numpy())
        T0 = int((~post).sum())
    elif T0 is None:
        T0 = Y.shape[0]
    return dict(Y_pre=Y[:T0], Y_post=Y[T0:] if T0 < Y.shape[0] else None,
                units=u, times=t, pre=t[:T0],
                post=t[T0:] if T0 < Y.shape[0] else None)


def _panel(n_units=5, T=8, n_post=2, seed=0, names="int", shuffle=True, t0=2000):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            rows.append(dict(unit=u if names == "int" else f"m{u:02d}",
                             time=t0 + t, y=rng.normal() + u + 0.1 * t,
                             post=int(t >= T - n_post)))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _labels(index_set):
    return None if index_set is None else np.asarray(index_set.labels)


def _assert_same(inputs, ref):
    assert isinstance(inputs, SYNDESInputs)
    assert inputs.Y_pre.dtype == np.float64
    np.testing.assert_array_equal(inputs.Y_pre, ref["Y_pre"])
    if ref["Y_post"] is None:
        assert inputs.Y_post is None and inputs.post_time_index is None
    else:
        np.testing.assert_array_equal(inputs.Y_post, ref["Y_post"])
        np.testing.assert_array_equal(_labels(inputs.post_time_index), ref["post"])
    np.testing.assert_array_equal(_labels(inputs.unit_index), ref["units"])
    np.testing.assert_array_equal(_labels(inputs.time_index), ref["times"])
    np.testing.assert_array_equal(_labels(inputs.pre_time_index), ref["pre"])


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_setup_goes_through_geoex_dataprep(self, monkeypatch):
        calls = []
        real = SETUP.geoex_dataprep

        def spy(*a, **k):
            calls.append(k.get("post_col"))
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "geoex_dataprep", spy)
        prepare_syndes_inputs(_panel(), **COLS, T0=6)
        assert calls == [None]

    def test_post_col_is_validated_by_geoex_dataprep(self, monkeypatch):
        calls = []
        real = SETUP.geoex_dataprep

        def spy(*a, **k):
            calls.append(k.get("post_col"))
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "geoex_dataprep", spy)
        prepare_syndes_inputs(_panel(), **COLS, post_col="post")
        assert "post" in calls

    def test_setup_does_not_pivot(self):
        # The setup module itself must not reshape the panel. A runtime patch of
        # DataFrame.pivot cannot test this: geoex_dataprep's _wide_pivot falls
        # back to df.pivot on pandas versions whose fast path would change the
        # array layout, so the patch fires inside geoex_dataprep there.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw,split", [
        (dict(), dict(post_col="post")),
        (dict(), dict(T0=6)),
        (dict(), dict()),                                   # design only
        (dict(n_post=0), dict(post_col="post")),            # post_col, no post
        (dict(names="str"), dict(post_col="post")),
        (dict(shuffle=False, t0=-4), dict(T0=2)),           # two pre periods
        (dict(T=3, n_post=1), dict(post_col="post")),
    ])
    def test_inputs_match_the_old_pivot(self, kw, split):
        df = _panel(**kw)
        _assert_same(prepare_syndes_inputs(df, **COLS, **split),
                     _reference(df, **COLS, **split))

    def test_extra_columns_are_ignored(self):
        df = _panel().assign(region="r", other=1.0)
        _assert_same(prepare_syndes_inputs(df, **COLS, post_col="post"),
                     _reference(df, **COLS, post_col="post"))


# --------------------------------------------------------------------------- #
# refusals now come from geoex_dataprep and balance
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_post_periods_between_pre_periods_are_refused(self):
        df = _panel(shuffle=False)
        df.loc[df.time == 2002, "post"] = 1      # a post period inside the pre window
        with pytest.raises(MlsynthConfigError, match="contiguous"):
            prepare_syndes_inputs(df, **COLS, post_col="post")

    def test_post_periods_at_the_start_are_refused(self):
        df = _panel(shuffle=False, n_post=0)
        df.loc[df.time <= 2001, "post"] = 1
        with pytest.raises(MlsynthConfigError, match="contiguous"):
            prepare_syndes_inputs(df, **COLS, post_col="post")

    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time == 2003), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_syndes_inputs(df, **COLS, post_col="post")

    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = -np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_syndes_inputs(df, **COLS, T0=6)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 3) & (df.time == 2004))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_syndes_inputs(df, **COLS, T0=6)

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = pd.concat([df, df.iloc[[3]]])
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_syndes_inputs(df, **COLS, T0=6)

    def test_a_missing_outcome_column_is_refused(self):
        with pytest.raises(MlsynthDataError, match="not found"):
            prepare_syndes_inputs(_panel().drop(columns="y"), **COLS, T0=6)


# --------------------------------------------------------------------------- #
# SYNDES's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_unknown_post_col(self):
        with pytest.raises(MlsynthDataError, match="not present"):
            prepare_syndes_inputs(_panel(), **COLS, post_col="nope")

    def test_every_period_post_is_refused(self):
        df = _panel(n_post=8)
        with pytest.raises(MlsynthConfigError, match="every period"):
            prepare_syndes_inputs(df, **COLS, post_col="post")

    def test_post_col_undefined_for_a_period(self):
        df = _panel(shuffle=False).astype({"post": float})
        df.loc[df.time == 2003, "post"] = np.nan
        with pytest.raises(MlsynthDataError, match="every time period"):
            prepare_syndes_inputs(df, **COLS, post_col="post")

    @pytest.mark.parametrize("T0", [0, -1, 9])
    def test_T0_out_of_range(self, T0):
        with pytest.raises(MlsynthConfigError, match="T0 must be between"):
            prepare_syndes_inputs(_panel(), **COLS, T0=T0)

    def test_one_pre_period_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="two pre-treatment"):
            prepare_syndes_inputs(_panel(), **COLS, T0=1)

    def test_one_unit_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="two units"):
            prepare_syndes_inputs(_panel(n_units=1), **COLS, T0=6)


# --------------------------------------------------------------------------- #
# properties over the domain, invalid post patterns included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(2, 7))
    T = draw(st.integers(2, 10))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, n_post=0, seed=seed, names=names)
    how = draw(st.sampled_from(["post_col", "T0", "none"]))
    if how == "T0":
        return df, dict(T0=draw(st.integers(1, T))), "ok"
    if how == "none":
        return df, dict(), "ok"
    post = draw(st.lists(st.booleans(), min_size=T, max_size=T))
    df["post"] = df["time"].map({2000 + t: int(p) for t, p in enumerate(post)})
    n_pre = post.index(True) if True in post else T
    if all(post):
        kind = "all-post"
    elif any(post[n_pre:]) and not all(post[n_pre:]):
        kind = "split"
    else:
        kind = "ok"
    return df, dict(post_col="post"), kind


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_geoex_path_matches_the_old_pivot_or_refuses(case):
    df, split, kind = case
    if kind == "all-post":
        with pytest.raises(MlsynthConfigError, match="every period"):
            prepare_syndes_inputs(df, **COLS, **split)
        return
    if kind == "split":
        with pytest.raises(MlsynthConfigError, match="contiguous"):
            prepare_syndes_inputs(df, **COLS, **split)
        return
    ref = _reference(df, **COLS, **split)
    if ref["Y_pre"].shape[0] < 2:
        with pytest.raises(MlsynthDataError, match="two pre-treatment"):
            prepare_syndes_inputs(df, **COLS, **split)
        return
    inputs = prepare_syndes_inputs(df, **COLS, **split)
    _assert_same(inputs, ref)
    # The pre and post blocks are the panel's periods in time order.
    times = _labels(inputs.time_index)
    pre = _labels(inputs.pre_time_index)
    post = _labels(inputs.post_time_index)
    stacked = pre if post is None else np.concatenate([pre, post])
    np.testing.assert_array_equal(stacked, times)
