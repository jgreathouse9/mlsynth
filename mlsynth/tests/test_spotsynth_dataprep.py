"""SPOTSYNTH ingestion through ``datautils.dataprep`` (#1138, part 1).

SPOTSYNTH used to pivot the long panel itself, outcome and treatment both. It
now takes the outcome matrix, the time labels, the treated unit and its
adoption period from ``balance`` and ``dataprep``. It gets the same refusals as
every other estimator: a missing or non-finite outcome cell, an absent or
duplicate unit-period, a non-binary treatment, and a treatment that switches
off after it starts. SPOTSYNTH's own checks (three units, one treated unit,
three pre-periods, no missing treatment value) stay on top.

``_reference`` is the old setup, kept as the oracle the new path must match on
every panel both accept. The setup only slices and reorders the outcome
matrix, so every comparison is exact.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import SPOTSYNTH
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.spotsynth_helpers import setup as SETUP
from mlsynth.utils.spotsynth_helpers.setup import prepare_spotsynth_inputs
from mlsynth.utils.spotsynth_helpers.structures import SpotSynthInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, outcome, treat, unitid, time):
    """The pre-#1138 setup: units sorted, times sorted, D read cell by cell."""
    time_labels = np.array(sorted(df[time].unique()))
    unit_names = sorted(df[unitid].unique())
    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    Dmat = (d_wide.loc[unit_names, time_labels].to_numpy() != 0).astype(int)
    ti = int(np.where(Dmat.max(axis=1) == 1)[0][0])
    T0 = int(np.argmax(Dmat.max(axis=0) == 1))
    donor_rows = [i for i in range(len(unit_names)) if i != ti]
    return SpotSynthInputs(
        y=Y[ti], D=Y[donor_rows].T, T0=T0,
        donor_names=[unit_names[i] for i in donor_rows],
        treated_name=unit_names[ti], time_labels=time_labels,
    )


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _panel(n_units=8, n_periods=12, treated=(0,), t0=6, seed=0, names="int",
           shuffle=True, start=2000):
    """Balanced long panel; the units in ``treated`` adopt at period index ``t0``."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            d = int(u in treated and t >= t0)
            rows.append(dict(unit=_name(u, names), time=start + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, SpotSynthInputs)
    for f in ("y", "D"):
        got, want = getattr(inputs, f), getattr(ref, f)
        assert got.dtype == want.dtype, f"{f}: dtype {got.dtype} vs {want.dtype}"
        assert got.shape == want.shape, f
        np.testing.assert_array_equal(got, want, err_msg=f)
        # The screen's and the solver's sums depend on the layout as well.
        assert got.flags["C_CONTIGUOUS"] == want.flags["C_CONTIGUOUS"], f
        assert got.flags["F_CONTIGUOUS"] == want.flags["F_CONTIGUOUS"], f
    assert inputs.T0 == ref.T0
    assert inputs.donor_names == ref.donor_names
    assert [type(u) for u in inputs.donor_names] == [type(u) for u in ref.donor_names]
    assert inputs.treated_name == ref.treated_name
    assert type(inputs.treated_name) is type(ref.treated_name)
    np.testing.assert_array_equal(np.asarray(inputs.time_labels), ref.time_labels)
    assert np.asarray(inputs.time_labels).dtype == ref.time_labels.dtype


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_setup_calls_dataprep_once(self, monkeypatch):
        calls = []
        real = SETUP.dataprep

        def spy(*a, **k):
            calls.append(k.get("keys"))
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "dataprep", spy)
        prepare_spotsynth_inputs(_panel(), **COLS)
        assert len(calls) == 1
        assert calls[0] is not None          # balance()'s key scan is reused

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
class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),
        dict(treated=(3,), t0=8),
        dict(names="str", treated=(5,)),
        dict(shuffle=False, start=-5, t0=11),                # last-period adoption
        dict(t0=3),                                          # three pre-periods
        dict(n_units=3, treated=(2,)),                       # two donors
        dict(n_units=40, n_periods=20, treated=(17,), t0=14, seed=3),
        dict(n_units=40, n_periods=20, names="str", treated=(0,), t0=10, seed=4),
    ])
    def test_inputs_match_the_old_setup(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_spotsynth_inputs(df, **COLS), _reference(df, **COLS))

    @pytest.mark.parametrize("label", ["datetime", "str", "float", "float32", "int32"])
    def test_period_labels_keep_their_type(self, label):
        # The old labels were np.array(sorted(...)): Timestamps as objects,
        # strings as a fixed-width unicode array, and 32-bit periods (a Stata
        # file's float32 time column) kept at 32 bits.
        df = _panel()
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0,
                      "float32": (step / 4.0).astype(np.float32),
                      "int32": step.astype(np.int32)}[label]
        _assert_same(prepare_spotsynth_inputs(df, **COLS), _reference(df, **COLS))

    @pytest.mark.parametrize("dtype", [bool, float])
    def test_a_bool_or_float_treatment_column_is_accepted(self, dtype):
        # The German and Californian benchmark panels carry a bool treatment.
        df = _panel()
        df["d"] = df["d"].astype(dtype)
        _assert_same(prepare_spotsynth_inputs(df, **COLS), _reference(df, **COLS))

    def test_an_integer_outcome_is_read_as_float(self):
        df = _panel()
        df["y"] = (100 * df["y"]).round().astype(int)
        _assert_same(prepare_spotsynth_inputs(df, **COLS), _reference(df, **COLS))

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _assert_same(prepare_spotsynth_inputs(df, **COLS), _reference(df, **COLS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_spotsynth_inputs(df, **COLS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = SPOTSYNTH(dict(df=_panel(n_units=12, n_periods=30, t0=22), **COLS,
                             inference="frequentist", display_graphs=False)).fit()
        assert np.isfinite(res.att)
        assert res.inputs.treated_name == 0


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old setup checked only for NaN and let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2009), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_spotsynth_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_missing_key_is_refused(self):
        df = _panel(shuffle=False).astype({"time": float})
        df.iloc[5, df.columns.get_loc("time")] = np.nan
        with pytest.raises(MlsynthDataError, match="key columns"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup read only the first treated period and accepted this.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, n_periods=30, t0=22)
        df.loc[(df.unit == 0) & (df.time == 2025), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            SPOTSYNTH(dict(df=df, **COLS, inference="frequentist",
                           display_graphs=False)).fit()

    @pytest.mark.parametrize("dose", [2, 0.5, -1])
    def test_a_non_binary_treatment_is_refused(self, dose):
        # The old setup read any non-zero value as treated.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[df.d == 1, "d"] = dose
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_a_missing_treatment_cell_is_refused(self):
        # dataprep's binary check skips NaN; SPOTSYNTH refuses it first.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 5) & (df.time == 2002), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="Treatment column contains NaN"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError, match="No treated unit"):
            prepare_spotsynth_inputs(_panel(treated=()), **COLS)

    def test_one_treated_unit_at_the_first_period(self):
        # dataprep's own refusal on the single-treated path.
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_spotsynth_inputs(_panel(t0=0), **COLS)


# --------------------------------------------------------------------------- #
# SPOTSYNTH's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match=f"Missing columns: {col}"):
            prepare_spotsynth_inputs(df, **COLS)

    @pytest.mark.parametrize("n_units", [1, 2])
    def test_fewer_than_three_units(self, n_units):
        with pytest.raises(MlsynthDataError, match="at least 2 donors"):
            prepare_spotsynth_inputs(_panel(n_units=n_units), **COLS)

    @pytest.mark.parametrize("treated,t0,n", [((0, 1), 6, 2), ((0, 2, 5), 0, 3),
                                              ((1, 4), 4, 2)])
    def test_more_than_one_treated_unit(self, treated, t0, n):
        df = _panel(treated=treated, t0=t0)
        with pytest.raises(MlsynthDataError, match=f"single treated unit. Found {n};"):
            prepare_spotsynth_inputs(df, **COLS)

    def test_staggered_treated_units_are_counted(self):
        df = _panel(shuffle=False, treated=(0, 1))
        df.loc[(df.unit == 1) & (df.time == 2006), "d"] = 0
        with pytest.raises(MlsynthDataError, match="Found 2;"):
            prepare_spotsynth_inputs(df, **COLS)

    @pytest.mark.parametrize("t0", [1, 2])
    def test_fewer_than_three_pre_periods(self, t0):
        with pytest.raises(MlsynthDataError, match="at least 3 pre-intervention"):
            prepare_spotsynth_inputs(_panel(t0=t0), **COLS)


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(3, 9))
    T = draw(st.integers(5, 10))
    t0 = draw(st.integers(3, T - 1))
    treated = draw(st.integers(0, n_units - 1))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, n_periods=T, treated=(treated,), t0=t0, seed=seed,
                names=names)
    defect = draw(st.sampled_from(
        ["none", "none", "none", "nan", "inf", "gap", "dup", "off", "dose", "dnan",
         "second"]))
    i = draw(st.integers(0, len(df) - 1))
    u = _name(treated, names)
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = -np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "off":
        if t0 >= T - 1:
            defect = "none"
        else:
            df.loc[(df.unit == u) & (df.time == 2000 + T - 1), "d"] = 0
    elif defect == "dose":
        df.loc[(df.unit == u) & (df.time == 2000 + T - 1), "d"] = 2
    elif defect == "dnan":
        df["d"] = df["d"].astype(float)
        df.iloc[i, df.columns.get_loc("d")] = np.nan
    elif defect == "second":
        other = _name((treated + 1) % n_units, names)
        df.loc[(df.unit == other) & (df.time == 2000 + T - 1), "d"] = 1
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_setup_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_spotsynth_inputs(df, **COLS)
        return
    inputs = prepare_spotsynth_inputs(df, **COLS)
    _assert_same(inputs, _reference(df, **COLS))
    # The treated unit and the donors partition the panel, the donors sorted.
    assert sorted(inputs.donor_names + [inputs.treated_name]) == sorted(df.unit.unique())
    assert inputs.donor_names == sorted(inputs.donor_names)
    assert inputs.D.shape == (len(inputs.time_labels), len(inputs.donor_names))
