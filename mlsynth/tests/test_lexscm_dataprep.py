"""LEXSCM ingestion through ``datautils.geoex_dataprep`` (#1138, part 1).

LEXSCM is a design estimator: it has no treatment column, only an optional
``post_col`` marking the periods after the experiment starts. Its outcome and
covariate matrices were pivoted by hand in
``fast_scm_helpers.fast_scm_setup.build_Y_matrix`` / ``build_Z_matrix`` (the
pre-period rows, and the post-period rows when there are any), and the pivots
were not checked: a non-finite outcome or a missing covariate reached the design
search. Each matrix is now read through ``geoex_dataprep``, so the rows passed
in get its refusals: an absent or duplicate unit-period and a missing or
non-finite value. A covariate's refusal is raised again under its name.

``_reference_*`` are the old pivots, kept as the oracle the new path must match
on every panel both accept.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import LEXSCM
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.fast_scm_helpers import fast_scm_setup as SETUP
from mlsynth.utils.fast_scm_helpers.fast_scm_setup import build_Y_matrix, build_Z_matrix
from mlsynth.utils.fast_scm_helpers.structure import IndexSet


def _reference_Y(df, outcome, time, unitid, unit_index):
    return (df.pivot(index=time, columns=unitid, values=outcome)
            .reindex(columns=unit_index).to_numpy(dtype=float))


def _reference_Z(df, covariates, time, unitid, unit_index):
    rows = []
    for col in covariates:
        wide = df.pivot(index=time, columns=unitid, values=col).reindex(columns=unit_index)
        if wide.nunique(axis=0).max() <= 1:
            rows.append(wide.iloc[0:1, :].to_numpy(dtype=float))
        else:
            rows.append(wide.mean(axis=0).to_numpy(dtype=float).reshape(1, -1))
    return np.vstack(rows)


def _panel(n_units=8, T=12, T_post=4, seed=0, names="str", shuffle=True, t0=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        share = rng.uniform()
        for t in range(T):
            rows.append(dict(unit=f"u{u:02d}" if names == "str" else u, time=t0 + t,
                             y=100.0 + rng.normal() + 0.3 * u, share=share,
                             promo=float(rng.integers(0, 2)), post=int(t >= T - T_post),
                             candidate=int(u < max(3, n_units // 2))))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _units(df):
    return IndexSet.from_labels(sorted(df["unit"].unique()))


def _assert_same(got, want):
    assert got.dtype == want.dtype == np.float64
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)
    # The design search sums over these matrices, so the layout is kept too.
    assert got.flags["C_CONTIGUOUS"] == want.flags["C_CONTIGUOUS"]
    assert got.flags["F_CONTIGUOUS"] == want.flags["F_CONTIGUOUS"]


def _mean_tolerance(wide):
    """Rounding bound for a column mean over T periods."""
    return 4 * np.finfo(float).eps * wide.shape[0] * (np.nanmax(np.abs(wide)) + 1.0)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_every_matrix_is_read_through_geoex_dataprep(self, monkeypatch):
        calls = []
        real = SETUP.geoex_dataprep

        def spy(*a, **k):
            calls.append(a[3])
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "geoex_dataprep", spy)
        df = _panel()
        build_Y_matrix(df, "y", "time", "unit", _units(df))
        build_Z_matrix(df, ["share", "promo"], "time", "unit", _units(df))
        assert calls == ["y", "share", "promo"]

    def test_setup_does_not_pivot(self):
        # A runtime patch of DataFrame.pivot cannot test this: dataprep's
        # _wide_pivot falls back to df.pivot on some pandas versions.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameMatrices:
    @pytest.mark.parametrize("kw", [
        dict(), dict(names="int"), dict(shuffle=False), dict(t0=-5),
        dict(n_units=2, T=3, T_post=1), dict(n_units=40, T=30, seed=3),
        dict(n_units=40, T=30, names="int", seed=4),
    ])
    def test_outcome_matrix_matches_the_old_pivot(self, kw):
        df = _panel(**kw)
        for rows in (df, df[df.post == 0], df[df.post == 1]):
            _assert_same(build_Y_matrix(rows, "y", "time", "unit", _units(df)),
                         _reference_Y(rows, "y", "time", "unit", _units(df)))

    @pytest.mark.parametrize("kw", [dict(), dict(names="int"), dict(n_units=40, T=30, seed=5)])
    def test_covariate_matrix_matches_the_old_pivot(self, kw):
        df = _panel(**kw)
        pre = df[df.post == 0]
        got = build_Z_matrix(pre, ["share", "promo"], "time", "unit", _units(df))
        want = _reference_Z(pre, ["share", "promo"], "time", "unit", _units(df))
        assert got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_array_equal(got[0], want[0])          # invariant: a value
        np.testing.assert_allclose(got[1], want[1], rtol=0,     # varying: a mean
                                   atol=_mean_tolerance(pre[["promo"]].to_numpy()))

    @pytest.mark.parametrize("label", ["datetime", "str", "float"])
    def test_period_labels(self, label):
        df = _panel()
        df["time"] = {"datetime": pd.to_datetime("2020-01-06") + pd.to_timedelta(7 * df["time"], unit="D"),
                      "str": "w" + df["time"].map("{:02d}".format),
                      "float": df["time"] / 4.0}[label]
        _assert_same(build_Y_matrix(df, "y", "time", "unit", _units(df)),
                     _reference_Y(df, "y", "time", "unit", _units(df)))

    def test_a_unit_order_other_than_sorted_is_kept(self):
        df = _panel()
        order = IndexSet.from_labels(sorted(df["unit"].unique(), reverse=True))
        _assert_same(build_Y_matrix(df, "y", "time", "unit", order),
                     _reference_Y(df, "y", "time", "unit", order))

    def test_covariate_rows_follow_a_unit_order_other_than_sorted(self):
        df = _panel()
        order = IndexSet.from_labels(sorted(df["unit"].unique(), reverse=True))
        got = build_Z_matrix(df, ["share"], "time", "unit", order)
        np.testing.assert_array_equal(got, _reference_Z(df, ["share"], "time", "unit", order))

    def test_no_covariates(self):
        df = _panel()
        assert build_Z_matrix(df, None, "time", "unit", _units(df)) is None
        assert build_Z_matrix(df, [], "time", "unit", _units(df)) is None

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        build_Y_matrix(df, "y", "time", "unit", _units(df))
        build_Z_matrix(df, ["share"], "time", "unit", _units(df))
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        df = _panel(n_units=10, T=30, T_post=6)
        res = LEXSCM(dict(df=df, outcome="y", unitid="unit", time="time",
                          candidate_col="candidate", m=2, post_col="post",
                          covariates=["share"], top_K=3, n_sims=20,
                          verbose=False)).fit()
        assert res.search.winner is not None


# --------------------------------------------------------------------------- #
# refusals now come from geoex_dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    @pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
    def test_a_non_finite_outcome_is_refused(self, value):
        # The old pivot passed it to the design search.
        df = _panel(shuffle=False)
        df.loc[(df.unit == "u02") & (df.time == 3), "y"] = value
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            build_Y_matrix(df, "y", "time", "unit", _units(df))

    def test_a_non_finite_post_outcome_is_refused_by_the_estimator(self):
        df = _panel(n_units=10, T=30, T_post=6, shuffle=False)
        df.loc[(df.unit == "u02") & (df.time == 27), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            LEXSCM(dict(df=df, outcome="y", unitid="unit", time="time",
                        candidate_col="candidate", m=2, post_col="post",
                        top_K=3, n_sims=20, verbose=False)).fit()

    def test_an_absent_unit_period_is_refused(self):
        # The old pivot filled the cell with NaN.
        df = _panel(shuffle=False)
        df = df[~((df.unit == "u05") & (df.time == 4))]
        with pytest.raises(MlsynthDataError, match="not strongly balanced"):
            build_Y_matrix(df, "y", "time", "unit", _units(df))

    def test_a_post_flag_that_varies_within_a_period_is_refused(self):
        # A unit-level post flag leaves the pre-period rows unbalanced, and
        # the old pivot filled the gaps with NaN.
        df = _panel(n_units=10, T=30, T_post=6, shuffle=False)
        df.loc[(df.unit == "u01") & (df.time == 23), "post"] = 1
        with pytest.raises(MlsynthDataError, match="not strongly balanced"):
            LEXSCM(dict(df=df, outcome="y", unitid="unit", time="time",
                        candidate_col="candidate", m=2, post_col="post",
                        top_K=3, n_sims=20, verbose=False)).fit()

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            build_Y_matrix(pd.concat([df, df.iloc[[3]]]), "y", "time", "unit", _units(df))

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_a_missing_or_non_finite_covariate_is_named(self, value):
        # The old pivot's nunique and mean skipped a NaN.
        df = _panel(shuffle=False)
        df.loc[(df.unit == "u04") & (df.time == 2), "promo"] = value
        with pytest.raises(MlsynthDataError, match="Covariate 'promo'") as info:
            build_Z_matrix(df, ["share", "promo"], "time", "unit", _units(df))
        assert "Missing outcome values" in str(info.value)
        assert isinstance(info.value.__cause__, MlsynthDataError)

    def test_a_missing_covariate_column_is_named(self):
        df = _panel()
        with pytest.raises(MlsynthDataError, match="Covariate 'nope'"):
            build_Z_matrix(df, ["nope"], "time", "unit", _units(df))


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
DEFECTS = ["none", "none", "none", "nan_y", "inf_y", "gap", "dup", "nan_x", "inf_x"]


@st.composite
def cases(draw):
    n_units = draw(st.integers(2, 9))
    T = draw(st.integers(2, 10))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, T_post=0, seed=seed, names=names)
    defect = draw(st.sampled_from(DEFECTS))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan_y":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf_y":
        df.iloc[i, df.columns.get_loc("y")] = draw(st.sampled_from([np.inf, -np.inf]))
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "nan_x":
        df.iloc[i, df.columns.get_loc("promo")] = np.nan
    elif defect == "inf_x":
        df.iloc[i, df.columns.get_loc("promo")] = np.inf
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_geoex_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    units = IndexSet.from_labels(sorted(set(df["unit"])))
    if defect in ("nan_x", "inf_x"):
        build_Y_matrix(df, "y", "time", "unit", units)
        with pytest.raises(MlsynthDataError, match="Covariate 'promo'"):
            build_Z_matrix(df, ["promo"], "time", "unit", units)
        return
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            build_Y_matrix(df, "y", "time", "unit", units)
        return
    Y = build_Y_matrix(df, "y", "time", "unit", units)
    _assert_same(Y, _reference_Y(df, "y", "time", "unit", units))
    Z = build_Z_matrix(df, ["share", "promo"], "time", "unit", units)
    ref = _reference_Z(df, ["share", "promo"], "time", "unit", units)
    np.testing.assert_array_equal(Z[0], ref[0])
    np.testing.assert_allclose(Z[1], ref[1], rtol=0, atol=_mean_tolerance(df[["promo"]].to_numpy()))
    # Rows are sorted periods and columns the requested unit order.
    assert Y.shape == (df["time"].nunique(), len(units))
    assert np.isfinite(Y).all() and np.isfinite(Z).all()
