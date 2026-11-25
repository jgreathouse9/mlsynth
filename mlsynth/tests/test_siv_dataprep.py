"""SIV ingestion through ``datautils.geoex_dataprep`` (#1138, part 1).

SIV used to pivot the long panel itself: the outcome ``Y``, the treatment
intensity ``R`` and the instrument ``Z``. ``R`` and ``Z`` are continuous, so
``dataprep``, which accepts only a 0/1 treatment, cannot read them; each of
the three columns is read as the outcome of its own ``geoex_dataprep`` call on
the same panel, and a refusal for ``R`` or ``Z`` is raised again under the
column's name. SIV gets the shared refusals: a missing or non-finite value in
any of the three columns, an absent or duplicate unit-period, and a row
without a unit or period. SIV's own checks (three units, ``T0`` or
``post_col``, ``T0_train``) stay on top.

``_reference`` is the old setup, kept as the oracle the new path must match on
every panel both accept. The setup only reorders and transposes the three
matrices, so every comparison is exact.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import SIV
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.siv_helpers import setup as SETUP
from mlsynth.utils.siv_helpers.setup import prepare_siv_inputs
from mlsynth.utils.siv_helpers.structures import SIVInputs

COLS = dict(outcome="y", treat="r", instrument="z", unitid="unit", time="time")


def _reference(df, outcome, treat, instrument, unitid, time, T0=None, post_col=None,
               T0_train=None):
    """The pre-#1138 setup: three pivots of the unit-time sorted panel."""
    ordered = df.sort_values([unitid, time]).copy()
    Y_wide = ordered.pivot(index=time, columns=unitid, values=outcome).sort_index()
    R_wide = ordered.pivot(index=time, columns=unitid, values=treat).sort_index()
    Z_wide = ordered.pivot(index=time, columns=unitid, values=instrument).sort_index()
    Y = Y_wide.to_numpy(dtype=float).T
    R = R_wide.to_numpy(dtype=float).T
    Z = Z_wide.to_numpy(dtype=float).T
    if post_col is not None:
        post_by_time = (ordered[[time, post_col]].drop_duplicates(subset=[time])
                        .set_index(time).reindex(Y_wide.index)[post_col]
                        .astype(bool).to_numpy())
        T0 = int(np.argmax(post_by_time))
    if T0_train is None:
        T0_train = max(2, int(0.75 * T0))
    return SIVInputs(
        Y=Y, R=R, Z=Z,
        unit_index=IndexSet.from_labels(Y_wide.columns.to_list()),
        time_index=IndexSet.from_labels(Y_wide.index.to_list()),
        T0=int(T0), T0_train=int(T0_train),
        has_pre_treatment=bool(np.any(R[:, :T0] != 0)),
        has_pre_instrument=bool(np.any(Z[:, :T0] != 0)),
    )


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _panel(n_units=8, n_periods=12, t0=8, seed=0, names="int", shuffle=True,
           start=2000, pre_r=False, pre_z=False):
    """Balanced long panel with a continuous treatment intensity and instrument.

    ``r`` and ``z`` are zero before ``t0`` (a shift-share design) unless
    ``pre_r`` / ``pre_z`` give them pre-period variation. ``post`` marks the
    periods from ``t0`` on.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        share = rng.uniform()
        for t in range(n_periods):
            post = t >= t0
            z = share * rng.uniform() if (post or pre_z) else 0.0
            r = 0.5 * z + 0.1 * rng.normal() if (post or pre_r) else 0.0
            rows.append(dict(unit=_name(u, names), time=start + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t - 0.16 * r,
                             r=r, z=z, post=int(post)))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, SIVInputs)
    for f in ("Y", "R", "Z"):
        got, want = getattr(inputs, f), getattr(ref, f)
        assert got.dtype == want.dtype, f"{f}: dtype {got.dtype} vs {want.dtype}"
        assert got.shape == want.shape, f
        np.testing.assert_array_equal(got, want, err_msg=f)
        # The SC solves and the projections sum over this layout.
        assert got.flags["C_CONTIGUOUS"] == want.flags["C_CONTIGUOUS"], f
        assert got.flags["F_CONTIGUOUS"] == want.flags["F_CONTIGUOUS"], f
    for f in ("unit_index", "time_index"):
        got, want = getattr(inputs, f).labels, getattr(ref, f).labels
        np.testing.assert_array_equal(got, want, err_msg=f)
        assert got.dtype == want.dtype, f
        assert getattr(inputs, f).label_to_idx == getattr(ref, f).label_to_idx, f
    for f in ("T0", "T0_train", "has_pre_treatment", "has_pre_instrument"):
        assert getattr(inputs, f) == getattr(ref, f), f


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_each_column_is_read_through_geoex_dataprep(self, monkeypatch):
        calls = []
        real = SETUP.geoex_dataprep

        def spy(df, unitid, time, col, **k):
            calls.append((unitid, time, col))
            return real(df, unitid, time, col, **k)

        monkeypatch.setattr(SETUP, "geoex_dataprep", spy)
        prepare_siv_inputs(_panel(), **COLS, T0=8)
        assert calls == [("unit", "time", "y"), ("unit", "time", "r"), ("unit", "time", "z")]

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
        dict(names="str", t0=6),
        dict(shuffle=False, start=-5, t0=11),                # one post-period
        dict(pre_r=True),
        dict(pre_z=True, names="str"),
        dict(pre_r=True, pre_z=True),
        dict(n_units=3, t0=4),                               # three units
        dict(n_units=40, n_periods=20, t0=14, seed=3),
        dict(n_units=40, n_periods=20, names="str", t0=10, seed=4, pre_z=True),
    ])
    @pytest.mark.parametrize("by", ["T0", "post_col"])
    def test_inputs_match_the_old_setup(self, kw, by):
        df = _panel(**kw)
        t0 = kw.get("t0", 8)
        arg = dict(T0=t0) if by == "T0" else dict(post_col="post")
        _assert_same(prepare_siv_inputs(df, **COLS, **arg), _reference(df, **COLS, **arg))

    def test_t0_train_is_kept(self):
        df = _panel()
        _assert_same(prepare_siv_inputs(df, **COLS, T0=8, T0_train=5),
                     _reference(df, **COLS, T0=8, T0_train=5))

    @pytest.mark.parametrize("label", ["datetime", "str", "float", "float32", "int32"])
    @pytest.mark.parametrize("by", ["T0", "post_col"])
    def test_period_labels_keep_their_type(self, label, by):
        df = _panel()
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0,
                      "float32": (step / 4.0).astype(np.float32),
                      "int32": step.astype(np.int32)}[label]
        arg = dict(T0=8) if by == "T0" else dict(post_col="post")
        _assert_same(prepare_siv_inputs(df, **COLS, **arg), _reference(df, **COLS, **arg))

    def test_a_post_col_read_from_the_first_sorted_unit(self):
        # post_col is read per period from the first unit in sorted order, as
        # before; a unit that disagrees later in the order does not move T0.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 5) & (df.time == 2006), "post"] = 1
        _assert_same(prepare_siv_inputs(df, **COLS, post_col="post"),
                     _reference(df, **COLS, post_col="post"))
        assert prepare_siv_inputs(df, **COLS, post_col="post").T0 == 8

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _assert_same(prepare_siv_inputs(df, **COLS, T0=8), _reference(df, **COLS, T0=8))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_siv_inputs(df, **COLS, post_col="post")
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = SIV(dict(df=_panel(n_units=12, n_periods=16, t0=10), **COLS, T0=10,
                       mode="siv", display_graphs=False)).fit()
        assert np.isfinite(res.theta_hat)


# --------------------------------------------------------------------------- #
# refusals now come from geoex_dataprep and balance
# --------------------------------------------------------------------------- #
class TestRefusals:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_missing_or_non_finite_outcome_is_refused(self, bad):
        # The old setup let an infinite value through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = bad
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_siv_inputs(df, **COLS, T0=8)

    @pytest.mark.parametrize("col,role", [("r", "treatment"), ("z", "instrument")])
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_a_missing_or_non_finite_r_or_z_is_refused_by_name(self, col, role, bad):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2009), col] = bad
        with pytest.raises(MlsynthDataError,
                           match=f"SIV's {role} column '{col}' is incomplete") as info:
            prepare_siv_inputs(df, **COLS, T0=8)
        # The cause is kept, in the message and as the exception's cause.
        assert "Missing outcome values" in str(info.value)
        assert isinstance(info.value.__cause__, MlsynthDataError)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_siv_inputs(df, **COLS, T0=8)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_siv_inputs(pd.concat([df, df.iloc[[7]]]), **COLS, T0=8)

    def test_the_estimator_reports_a_duplicate(self):
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            SIV(dict(df=pd.concat([df, df.iloc[[7]]]), **COLS, T0=8,
                     display_graphs=False)).fit()

    def test_a_missing_key_is_refused(self):
        df = _panel(shuffle=False).astype({"time": float})
        df.iloc[5, df.columns.get_loc("time")] = np.nan
        with pytest.raises(MlsynthDataError, match="key columns"):
            prepare_siv_inputs(df, **COLS, T0=8)


# --------------------------------------------------------------------------- #
# SIV's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "r", "z", "unit", "time"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match="Missing required columns"):
            prepare_siv_inputs(df, **COLS, T0=8)

    @pytest.mark.parametrize("n_units", [1, 2])
    def test_fewer_than_three_units(self, n_units):
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_siv_inputs(_panel(n_units=n_units), **COLS, T0=8)

    @pytest.mark.parametrize("T0", [0, 12, 13])
    def test_t0_out_of_range(self, T0):
        with pytest.raises(MlsynthConfigError, match=r"T0 must be in \[1, 11\]"):
            prepare_siv_inputs(_panel(), **COLS, T0=T0)

    def test_neither_t0_nor_post_col(self):
        with pytest.raises(MlsynthConfigError, match="Either T0 or post_col"):
            prepare_siv_inputs(_panel(), **COLS)

    def test_post_col_absent(self):
        with pytest.raises(MlsynthConfigError, match="post_col 'nope' is not in df"):
            prepare_siv_inputs(_panel(), **COLS, post_col="nope")

    @pytest.mark.parametrize("value", [0, 1])
    def test_post_col_marks_one_side_only(self, value):
        df = _panel()
        df["post"] = value
        with pytest.raises(MlsynthConfigError, match="both pre and post"):
            prepare_siv_inputs(df, **COLS, post_col="post")

    @pytest.mark.parametrize("T0_train", [1, 8, 9])
    def test_t0_train_out_of_range(self, T0_train):
        with pytest.raises(MlsynthConfigError, match="T0_train must lie"):
            prepare_siv_inputs(_panel(), **COLS, T0=8, T0_train=T0_train)


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(3, 9))
    T = draw(st.integers(4, 10))
    t0 = draw(st.integers(1, T - 1))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, n_periods=T, t0=t0, seed=seed, names=names,
                pre_r=draw(st.booleans()), pre_z=draw(st.booleans()))
    defect = draw(st.sampled_from(
        ["none", "none", "none", "nan", "inf", "gap", "dup"]))
    col = draw(st.sampled_from(["y", "r", "z"]))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan":
        df.iloc[i, df.columns.get_loc(col)] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc(col)] = -np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    by = draw(st.sampled_from(["T0", "post_col"]))
    return df, defect, col, (dict(T0=t0) if by == "T0" else dict(post_col="post"))


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_geoex_path_matches_the_old_setup_or_refuses(case):
    df, defect, col, arg = case
    if defect != "none":
        with pytest.raises(MlsynthDataError) as info:
            prepare_siv_inputs(df, **COLS, **arg)
        if defect in ("nan", "inf") and col != "y":
            assert f"'{col}' is incomplete" in str(info.value)
        return
    inputs = prepare_siv_inputs(df, **COLS, **arg)
    _assert_same(inputs, _reference(df, **COLS, **arg))
    # The three matrices share one unit by period grid, units and periods sorted.
    assert inputs.Y.shape == inputs.R.shape == inputs.Z.shape
    assert list(inputs.unit_index.labels) == sorted(df["unit"].unique())
    assert 1 <= inputs.T0 < inputs.Y.shape[1]
