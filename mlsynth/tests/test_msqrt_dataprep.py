"""MSQRT ingestion through ``datautils.dataprep`` (#1138, part 1).

MSQRT used to pivot the long panel itself, outcome and treatment both. It now
takes the outcome matrix, the time labels and each treated unit's adoption
period from ``balance`` and ``dataprep``. It gets the same refusals as every
other estimator: a missing or non-finite outcome cell, an absent or duplicate
unit-period, a non-binary treatment, and a treatment that switches off after
it starts. MSQRT's own checks (two pre-periods, two never-treated donors, a
block design, no unit treated in the first period) stay on top.

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

from mlsynth import MSQRT
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.msqrt_helpers import setup as SETUP
from mlsynth.utils.msqrt_helpers.setup import prepare_msqrt_inputs
from mlsynth.utils.msqrt_helpers.structures import MSQRTInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, outcome, treat, unitid, time, weight_col=None):
    """The pre-#1138 setup: units sorted, times sorted, D read cell by cell."""
    time_labels = np.array(sorted(df[time].unique()))
    unit_names = sorted(df[unitid].unique())
    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = d_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    T0 = int(np.argmax(D.max(axis=0) == 1))
    T = time_labels.size
    treated_idx = np.where(D.max(axis=1) == 1)[0]
    control_idx = np.where(D.max(axis=1) == 0)[0]
    treated_names = [unit_names[i] for i in treated_idx]
    control_names = [unit_names[i] for i in control_idx]
    unit_weights = None
    if weight_col is not None:
        wmap = df.drop_duplicates(subset=[unitid]).set_index(unitid)[weight_col]
        unit_weights = {str(u): float(wmap.get(u)) for u in treated_names}
    return MSQRTInputs(
        Y_pre=Y[np.ix_(treated_idx, np.arange(T0))].T,
        Y_post=Y[np.ix_(treated_idx, np.arange(T0, T))].T,
        X_pre=Y[np.ix_(control_idx, np.arange(T0))].T,
        X_post=Y[np.ix_(control_idx, np.arange(T0, T))].T,
        treated_names=treated_names, control_names=control_names,
        time_labels=time_labels, T0=int(T0), unit_weights=unit_weights,
    )


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _panel(n_units=8, n_periods=12, treated=(0, 1), t0=6, seed=0, names="int",
           shuffle=True, start=2000):
    """Balanced long panel; the units in ``treated`` adopt at period index ``t0``."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            d = int(u in treated and t >= t0)
            rows.append(dict(unit=_name(u, names), time=start + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d,
                             pop=10.0 + u))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, MSQRTInputs)
    for f in ("Y_pre", "Y_post", "X_pre", "X_post"):
        got, want = getattr(inputs, f), getattr(ref, f)
        assert got.dtype == want.dtype, f"{f}: dtype {got.dtype} vs {want.dtype}"
        assert got.shape == want.shape, f
        np.testing.assert_array_equal(got, want, err_msg=f)
        # The solver's sums depend on the layout as well as the values.
        assert got.flags["C_CONTIGUOUS"] == want.flags["C_CONTIGUOUS"], f
        assert got.flags["F_CONTIGUOUS"] == want.flags["F_CONTIGUOUS"], f
    assert inputs.T0 == ref.T0
    for f in ("treated_names", "control_names"):
        got, want = getattr(inputs, f), getattr(ref, f)
        assert got == want, f
        assert [type(u) for u in got] == [type(u) for u in want], f
    np.testing.assert_array_equal(np.asarray(inputs.time_labels), ref.time_labels)
    assert np.asarray(inputs.time_labels).dtype == ref.time_labels.dtype
    assert inputs.unit_weights == ref.unit_weights


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
        prepare_msqrt_inputs(_panel(), **COLS)
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
        dict(),                                              # a block of two
        dict(treated=(3,), t0=8),                            # one treated unit
        dict(names="str", treated=(1, 4, 6)),
        dict(shuffle=False, start=-5, t0=11),                # last-period adoption
        dict(t0=2),                                          # two pre-periods
        dict(n_units=4, treated=(0, 3)),                     # two donors
        dict(n_units=40, n_periods=20, treated=tuple(range(0, 40, 3)), t0=14, seed=3),
        dict(n_units=40, n_periods=20, names="str", treated=(5,), t0=10, seed=4),
    ])
    def test_inputs_match_the_old_setup(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_msqrt_inputs(df, **COLS), _reference(df, **COLS))

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
        _assert_same(prepare_msqrt_inputs(df, **COLS), _reference(df, **COLS))

    @pytest.mark.parametrize("names", ["int", "str"])
    def test_weight_col_is_read_as_before(self, names):
        df = _panel(names=names, treated=(0, 2, 5))
        _assert_same(prepare_msqrt_inputs(df, **COLS, weight_col="pop"),
                     _reference(df, **COLS, weight_col="pop"))

    def test_a_float_treatment_column_is_accepted(self):
        df = _panel()
        df["d"] = df["d"].astype(float)
        _assert_same(prepare_msqrt_inputs(df, **COLS), _reference(df, **COLS))

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _assert_same(prepare_msqrt_inputs(df, **COLS), _reference(df, **COLS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_msqrt_inputs(df, **COLS, weight_col="pop")
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = MSQRT(dict(df=_panel(n_units=10, n_periods=24, t0=18), **COLS,
                         lambda_=0.5, display_graphs=False, inference=False)).fit()
        assert np.isfinite(res.att)
        assert res.inputs.m == 2 and res.inputs.n == 8


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_msqrt_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old setup checked only for NaN and let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2009), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_msqrt_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_msqrt_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_msqrt_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_missing_key_is_refused(self):
        df = _panel(shuffle=False).astype({"time": float})
        df.iloc[5, df.columns.get_loc("time")] = np.nan
        with pytest.raises(MlsynthDataError, match="key columns"):
            prepare_msqrt_inputs(df, **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup read only each unit's first treated period and
        # accepted this.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_msqrt_inputs(df, **COLS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, n_periods=20, t0=14)
        df.loc[(df.unit == 1) & (df.time == 2016), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            MSQRT(dict(df=df, **COLS, display_graphs=False, inference=False)).fit()

    def test_a_non_binary_treatment_is_refused(self):
        # {0, 1, 2}: the old setup counted a unit whose maximum was 2 as a
        # never-treated donor.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time >= 2008), "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_msqrt_inputs(df, **COLS)

    def test_a_missing_treatment_cell_is_refused(self):
        # dataprep refuses it and names the cell; the setup has no check of
        # its own.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 5) & (df.time == 2002), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match=r"Missing treatment values: 1 of .*"
                           r"the first at unit '5', time 2002\."):
            prepare_msqrt_inputs(df, **COLS)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError, match="No treated"):
            prepare_msqrt_inputs(_panel(treated=()), **COLS)


# --------------------------------------------------------------------------- #
# MSQRT's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match=f"Required column '{col}' missing"):
            prepare_msqrt_inputs(df, **COLS)

    @pytest.mark.parametrize("n_units,treated", [(3, (0, 1)), (2, (0,)), (1, (0,)),
                                                 (3, (0, 1, 2))])
    def test_fewer_than_two_donors(self, n_units, treated):
        with pytest.raises(MlsynthDataError, match="at least 2 never-treated donor"):
            prepare_msqrt_inputs(_panel(n_units=n_units, treated=treated), **COLS)

    def test_staggered_adoption_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time.isin([2006, 2007])), "d"] = 0
        with pytest.raises(MlsynthDataError, match="block design"):
            prepare_msqrt_inputs(df, **COLS)

    def test_one_treated_unit_at_the_first_period(self):
        # dataprep's own refusal on the single-treated path.
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_msqrt_inputs(_panel(treated=(0,), t0=0), **COLS)

    def test_a_block_at_the_first_period(self):
        # dataprep's cohort path does not refuse it; MSQRT does.
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_msqrt_inputs(_panel(t0=0), **COLS)

    @pytest.mark.parametrize("treated", [(0,), (0, 1)])
    def test_one_pre_period_is_too_few(self, treated):
        with pytest.raises(MlsynthDataError, match="at least 2 pre-treatment periods"):
            prepare_msqrt_inputs(_panel(treated=treated, t0=1), **COLS)

    def test_weight_col_missing_column(self):
        with pytest.raises(MlsynthDataError, match="weight_col 'nope' missing"):
            prepare_msqrt_inputs(_panel(), **COLS, weight_col="nope")

    def test_weight_col_missing_value(self):
        df = _panel()
        df.loc[df.unit == 1, "pop"] = np.nan
        with pytest.raises(MlsynthDataError, match="no value for treated unit"):
            prepare_msqrt_inputs(df, **COLS, weight_col="pop")

    def test_weight_col_negative(self):
        df = _panel()
        df.loc[df.unit == 0, "pop"] = -1.0
        with pytest.raises(MlsynthDataError, match="negative"):
            prepare_msqrt_inputs(df, **COLS, weight_col="pop")

    def test_weight_col_zero_sum(self):
        df = _panel()
        df["pop"] = 0.0
        with pytest.raises(MlsynthDataError, match="sums to zero"):
            prepare_msqrt_inputs(df, **COLS, weight_col="pop")


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_treated = draw(st.integers(1, 4))
    n_units = n_treated + draw(st.integers(2, 6))
    T = draw(st.integers(4, 10))
    t0 = draw(st.integers(2, T - 1))
    treated = tuple(sorted(draw(st.sets(st.integers(0, n_units - 1),
                                        min_size=n_treated, max_size=n_treated))))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, n_periods=T, treated=treated, t0=t0, seed=seed,
                names=names)
    defect = draw(st.sampled_from(
        ["none", "none", "none", "nan", "inf", "gap", "dup", "off", "dose", "dnan"]))
    i = draw(st.integers(0, len(df) - 1))
    u = _name(draw(st.sampled_from(treated)), names)
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
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_setup_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_msqrt_inputs(df, **COLS)
        return
    inputs = prepare_msqrt_inputs(df, **COLS)
    _assert_same(inputs, _reference(df, **COLS))
    # The treated and control units partition the panel, both sorted, and the
    # pre and post blocks partition the periods at T0.
    assert sorted(inputs.treated_names + inputs.control_names) == sorted(df.unit.unique())
    assert inputs.Y_pre.shape[0] + inputs.Y_post.shape[0] == len(inputs.time_labels)
    assert inputs.X_pre.shape[0] == inputs.T0 == inputs.Y_pre.shape[0]
