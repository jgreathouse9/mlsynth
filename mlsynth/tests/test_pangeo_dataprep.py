"""PANGEO ingestion through ``datautils.geoex_dataprep`` (#1138, part 1).

PANGEO designs an experiment: it pairs geos into supergeos before anyone is
treated, so the panel has no treatment column and ``dataprep`` cannot split it.
Both of its outcome matrices, the design panel and the post-period panel the
realized effect is read from, now come from ``geoex_dataprep``. That refuses a
duplicate or absent unit-period and a missing or non-finite outcome; the old
pivots let an infinite outcome through and raised a raw pandas ``ValueError``
on a duplicate cell.

The post panel is cut to the design's units before it is read, so a unit that
appears only after the design (and may have gaps) is ignored, as before.

``_reference_Y`` and ``_reference_post`` are the old pivots, kept as the oracle
the new path must match on every panel both accept.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.pangeo_helpers import setup as SETUP
from mlsynth.utils.pangeo_helpers.setup import build_post_matrix, prepare_pangeo_inputs

COLS = dict(outcome="y", arm="arm", unitid="unit", time="time")


def _reference_Y(df, outcome, unitid, time):
    """The pre-#1138 design pivot: units sorted, times sorted, (N, T)."""
    t = np.array(sorted(df[time].unique()))
    u = sorted(df[unitid].unique())
    wide = df.pivot(index=unitid, columns=time, values=outcome)
    return wide.loc[u, t].to_numpy(dtype=float), u, t


def _reference_post(post_df, unit_names, outcome, unitid, time):
    """The pre-#1138 post pivot, aligned with the design's units."""
    t = np.array(sorted(post_df[time].unique()))
    wide = post_df.pivot(index=unitid, columns=time, values=outcome)
    return wide.loc[unit_names, t].to_numpy(dtype=float), t


def _panel(n_units=8, T=10, seed=0, names="int", shuffle=True, t0=2000,
           arms=("A",), extra=False):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            rows.append(dict(unit=u if names == "int" else f"g{u:02d}",
                             time=t0 + t, y=rng.normal() + u + 0.1 * t,
                             arm=arms[u % len(arms)], pop=1.0 + u,
                             x=rng.normal()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _check(df, **kw):
    inputs = prepare_pangeo_inputs(df, **COLS, **kw)
    Y, u, t = _reference_Y(df, "y", "unit", "time")
    assert inputs.Y.dtype == np.float64
    np.testing.assert_array_equal(inputs.Y, Y)
    assert list(inputs.unit_names) == list(u)
    np.testing.assert_array_equal(np.asarray(inputs.time_labels), t)
    assert np.asarray(inputs.time_labels).dtype == t.dtype
    return inputs


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_both_matrices_come_from_geoex_dataprep(self, monkeypatch):
        calls = []
        real = SETUP.geoex_dataprep

        def spy(*a, **k):
            calls.append(len(a[0]))
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "geoex_dataprep", spy)
        df = _panel()
        inputs = prepare_pangeo_inputs(df[df.time < 2008], **COLS)
        build_post_matrix(df[df.time >= 2008], inputs, "y", "unit", "time")
        assert len(calls) == 2

    def test_setup_does_not_pivot(self):
        # A runtime patch of DataFrame.pivot cannot test this: geoex_dataprep's
        # _wide_pivot falls back to df.pivot on some pandas versions.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),
        dict(names="str"),
        dict(arms=("A", "B")),
        dict(shuffle=False, t0=-5),
        dict(n_units=2, T=3),
    ])
    def test_design_matrix_matches_the_old_pivot(self, kw):
        _check(_panel(**kw))

    def test_covariates_and_weights_are_unchanged(self):
        df = _panel(arms=("A", "B"))
        inputs = _check(df, covariates=["x"], weight_col="pop")
        base = df.groupby("unit")["x"].mean().loc[inputs.unit_names].to_numpy()
        np.testing.assert_array_equal(inputs.covariates[:, 0], base)
        np.testing.assert_array_equal(
            inputs.weights, df.groupby("unit")["pop"].mean().loc[inputs.unit_names].to_numpy())
        assert set(inputs.arm_units) == {"A", "B"}

    @pytest.mark.parametrize("names", ["int", "str"])
    def test_post_matrix_matches_the_old_pivot(self, names):
        df = _panel(names=names)
        pre, post = df[df.time < 2007], df[df.time >= 2007]
        inputs = prepare_pangeo_inputs(pre, **COLS)
        Y_post, t = build_post_matrix(post, inputs, "y", "unit", "time")
        Y_ref, t_ref = _reference_post(post, inputs.unit_names, "y", "unit", "time")
        assert Y_post.dtype == np.float64
        np.testing.assert_array_equal(Y_post, Y_ref)
        np.testing.assert_array_equal(t, t_ref)

    def test_a_unit_only_in_the_post_period_is_ignored(self):
        df = _panel()
        pre, post = df[df.time < 2007], df[df.time >= 2007]
        stray = pd.DataFrame([dict(unit=99, time=2008, y=1.0, arm="A", pop=1.0, x=0.0)])
        post = pd.concat([post, stray])   # 99 is absent in 2007 and 2009
        inputs = prepare_pangeo_inputs(pre, **COLS)
        Y_post, _ = build_post_matrix(post, inputs, "y", "unit", "time")
        assert Y_post.shape == (8, 3)

    def test_fit_smoke(self):
        from mlsynth import PANGEO
        df = _panel(n_units=8, T=12)
        df["post"] = (df.time >= 2009).astype(int)
        res = PANGEO(dict(df=df, outcome="y", arm="arm", unitid="unit", time="time",
                          post_col="post", display_graphs=False)).fit()
        assert np.isfinite(res.effects.program.att)


# --------------------------------------------------------------------------- #
# refusals now come from geoex_dataprep and balance
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2003), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_pangeo_inputs(df, **COLS)

    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time == 2001), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_pangeo_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 3) & (df.time == 2004))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_pangeo_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = pd.concat([df, df.iloc[[4]]])
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_pangeo_inputs(df, **COLS)

    def test_a_non_finite_post_outcome_is_refused(self):
        df = _panel(shuffle=False)
        pre, post = df[df.time < 2007], df[df.time >= 2007].copy()
        post.loc[(post.unit == 0) & (post.time == 2008), "y"] = -np.inf
        inputs = prepare_pangeo_inputs(pre, **COLS)
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            build_post_matrix(post, inputs, "y", "unit", "time")

    def test_a_gap_in_the_post_panel_is_refused(self):
        df = _panel(shuffle=False)
        pre, post = df[df.time < 2007], df[df.time >= 2007]
        post = post[~((post.unit == 5) & (post.time == 2008))]
        inputs = prepare_pangeo_inputs(pre, **COLS)
        with pytest.raises(MlsynthDataError, match="balanced"):
            build_post_matrix(post, inputs, "y", "unit", "time")

    def test_a_duplicate_post_cell_is_refused(self):
        df = _panel(shuffle=False)
        pre, post = df[df.time < 2007], df[df.time >= 2007]
        inputs = prepare_pangeo_inputs(pre, **COLS)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            build_post_matrix(pd.concat([post, post.iloc[[0]]]), inputs, "y", "unit", "time")


# --------------------------------------------------------------------------- #
# PANGEO's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_missing_column(self):
        with pytest.raises(MlsynthDataError, match="missing"):
            prepare_pangeo_inputs(_panel().drop(columns="arm"), **COLS)

    def test_one_unit_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="at least 2 units"):
            prepare_pangeo_inputs(_panel(n_units=1), **COLS)

    def test_a_design_unit_absent_from_the_post_period(self):
        df = _panel(shuffle=False)
        pre, post = df[df.time < 2007], df[df.time >= 2007]
        inputs = prepare_pangeo_inputs(pre, **COLS)
        with pytest.raises(MlsynthDataError, match="absent from the post period"):
            build_post_matrix(post[post.unit != 6], inputs, "y", "unit", "time")

    def test_missing_post_outcome_column(self):
        df = _panel(shuffle=False)
        inputs = prepare_pangeo_inputs(df[df.time < 2007], **COLS)
        with pytest.raises(MlsynthDataError, match="missing in post"):
            build_post_matrix(df[df.time >= 2007].drop(columns="y"), inputs, "y", "unit", "time")


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(2, 8))
    T = draw(st.integers(2, 9))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, seed=seed, names=names)
    defect = draw(st.sampled_from(["none", "none", "nan", "inf", "gap", "dup"]))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = np.inf
    elif defect == "gap" and len(df) > 1:
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    else:
        defect = "none"
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_geoex_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(df, **COLS)
        return
    inputs = _check(df)
    # Each arm's units index rows of Y whose names carry that arm.
    for a, idx in inputs.arm_units.items():
        assert all(inputs.arm_of[inputs.unit_names[i]] == a for i in idx)
