"""SpSyDiD ingestion through ``datautils.dataprep`` (#1138, part 1).

SpSyDiD used to pivot the long panel itself, outcome and treatment both. It
now takes the outcome matrix, the time labels and each directly treated
unit's adoption period from ``balance`` and ``dataprep``, and builds the
treatment matrix ``D`` from the adoption periods. Its treatment column was
already required to be 0/1, so ``dataprep``'s binary refusal replaces its own.
It gets the same refusals as every other estimator: a missing or non-finite
outcome cell, an absent or duplicate unit-period, a non-binary treatment, and
a treatment that switches off after it starts. SpSyDiD's own checks (the
``unit_order`` matching the panel, the spatial matrix, a pure control, no unit
treated in the first period) stay on top.

Staggered adoption is accepted as before: ``T0`` is the first period any unit
is treated and every later adopter is treated in ``D`` from its own start.

``_reference`` is the old setup, kept as the oracle the new path must match on
every panel both accept. The exposure ``W D`` is a matrix product whose
summation order depends on the layout of ``D``; with ``D`` built in the old
layout it is compared exactly.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import SpSyDiD
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.spsydid_helpers import setup as SETUP
from mlsynth.utils.spsydid_helpers.setup import prepare_spsydid_inputs
from mlsynth.utils.spsydid_helpers.spatial import row_standardize
from mlsynth.utils.spsydid_helpers.structures import SpSyDiDInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, outcome, treat, unitid, time, spatial_matrix, unit_order=None,
               row_standardize_spatial=True):
    """The pre-#1138 setup: units sorted (or in unit_order), D read cell by cell."""
    unit_names = sorted(df[unitid].unique()) if unit_order is None else list(unit_order)
    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    outcome_wide = df.pivot(index=unitid, columns=time, values=outcome)
    treat_wide = df.pivot(index=unitid, columns=time, values=treat)
    outcome_wide = outcome_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    treat_wide = treat_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    W = np.asarray(spatial_matrix, dtype=float).copy()
    if row_standardize_spatial:
        W = row_standardize(W)
    exposure = W @ treat_wide
    direct_mask = (treat_wide > 0).any(axis=1)
    spillover_mask = (~direct_mask) & ((exposure > 1e-12).any(axis=1))
    pure_mask = ~(direct_mask | spillover_mask)
    any_treated_at_t = (treat_wide > 0).any(axis=0)
    return SpSyDiDInputs(
        outcome_matrix=outcome_wide, treatment_matrix=treat_wide, spatial_matrix=W,
        exposure_matrix=exposure, unit_names=unit_names, time_labels=time_labels,
        T=T, T0=int(np.argmax(any_treated_at_t)),
        direct_indices=np.where(direct_mask)[0],
        spillover_indices=np.where(spillover_mask)[0],
        pure_control_indices=np.where(pure_mask)[0],
    )


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _line_W(n):
    """Unit i's neighbours are i - 1 and i + 1, in sorted-unit order."""
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def _panel(n_units=8, n_periods=12, adopt=None, seed=0, names="int",
           shuffle=True, start=2000):
    """Balanced long panel; ``adopt`` maps unit index -> adoption period index."""
    adopt = {0: 6} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            d = int(u in adopt and t >= adopt[u])
            rows.append(dict(unit=_name(u, names), time=start + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, SpSyDiDInputs)
    for f in ("outcome_matrix", "treatment_matrix", "spatial_matrix", "exposure_matrix",
              "direct_indices", "spillover_indices", "pure_control_indices"):
        got, want = getattr(inputs, f), getattr(ref, f)
        assert got.dtype == want.dtype, f"{f}: dtype {got.dtype} vs {want.dtype}"
        assert got.shape == want.shape, f
        np.testing.assert_array_equal(got, want, err_msg=f)
        # The regression's and the exposure's sums depend on the layout too.
        assert got.flags["C_CONTIGUOUS"] == want.flags["C_CONTIGUOUS"], f
        assert got.flags["F_CONTIGUOUS"] == want.flags["F_CONTIGUOUS"], f
    assert (inputs.T, inputs.T0) == (ref.T, ref.T0)
    assert inputs.unit_names == ref.unit_names
    assert [type(u) for u in inputs.unit_names] == [type(u) for u in ref.unit_names]
    np.testing.assert_array_equal(np.asarray(inputs.time_labels), ref.time_labels)
    assert np.asarray(inputs.time_labels).dtype == ref.time_labels.dtype


def _both(df, W, **kw):
    return (prepare_spsydid_inputs(df, **COLS, spatial_matrix=W, **kw),
            _reference(df, **COLS, spatial_matrix=W, **kw))


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
        prepare_spsydid_inputs(_panel(), **COLS, spatial_matrix=_line_W(8))
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
        dict(),                                              # one treated unit
        dict(names="str", adopt={0: 8}),
        dict(shuffle=False, start=-5, adopt={0: 11}),        # last-period adoption
        dict(adopt={0: 5, 1: 5}),                            # one cohort of two
        dict(adopt={0: 4, 1: 9}),                            # staggered
        dict(n_units=40, n_periods=20, adopt={0: 12}, seed=3),
        dict(n_units=40, n_periods=20, names="str", seed=4,
             adopt={0: 10, 1: 10, 20: 14, 21: 17}),          # 40, staggered
    ])
    def test_inputs_match_the_old_setup(self, kw):
        df = _panel(**kw)
        n = df["unit"].nunique()
        _assert_same(*_both(df, _line_W(n)))

    def test_a_unit_order_is_kept(self):
        # W's rows follow the caller's unit order, not the sorted one.
        df = _panel(n_units=8, adopt={7: 6})
        order = [3, 7, 0, 5, 1, 6, 2, 4]
        inputs, ref = _both(df, _line_W(8), unit_order=order)
        _assert_same(inputs, ref)
        assert inputs.unit_names == order
        assert list(inputs.direct_indices) == [1]

    def test_an_unstandardised_matrix_is_kept(self):
        df = _panel()
        _assert_same(*_both(df, _line_W(8), row_standardize_spatial=False))

    @pytest.mark.parametrize("label", ["datetime", "str", "float", "float32", "int32"])
    def test_period_labels_keep_their_type(self, label):
        # The old labels were np.array(sorted(...)): Timestamps as objects,
        # strings as a fixed-width unicode array, 32-bit periods at 32 bits.
        df = _panel()
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0,
                      "float32": (step / 4.0).astype(np.float32),
                      "int32": step.astype(np.int32)}[label]
        _assert_same(*_both(df, _line_W(8)))

    @pytest.mark.parametrize("dtype", [bool, float])
    def test_a_bool_or_float_treatment_column_is_accepted(self, dtype):
        df = _panel()
        df["d"] = df["d"].astype(dtype)
        _assert_same(*_both(df, _line_W(8)))

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _assert_same(*_both(df, _line_W(8)))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_spsydid_inputs(df, **COLS, spatial_matrix=_line_W(8))
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = SpSyDiD(dict(df=_panel(n_units=10, n_periods=16, adopt={0: 10}), **COLS,
                           spatial_matrix=_line_W(10), display_graphs=False)).fit()
        assert np.isfinite(res.att)
        assert res.inputs.T0 == 10


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def _prep(self, df, n=8):
        return prepare_spsydid_inputs(df, **COLS, spatial_matrix=_line_W(n))

    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            self._prep(df)

    def test_a_non_finite_outcome_is_refused(self):
        # The old setup checked only for NaN and let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2009), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            self._prep(df)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            self._prep(df)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            self._prep(pd.concat([df, df.iloc[[7]]]))

    def test_a_missing_key_is_refused(self):
        df = _panel(shuffle=False).astype({"time": float})
        df.iloc[5, df.columns.get_loc("time")] = np.nan
        with pytest.raises(MlsynthDataError, match="key columns"):
            self._prep(df)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup kept the 0 in D and fitted it.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            self._prep(df)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, n_units=10, n_periods=16, adopt={0: 10, 1: 12})
        df.loc[(df.unit == 1) & (df.time == 2014), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            SpSyDiD(dict(df=df, **COLS, spatial_matrix=_line_W(10),
                         display_graphs=False)).fit()

    @pytest.mark.parametrize("dose", [2, 0.5, -1])
    def test_a_non_binary_treatment_is_refused(self, dose):
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[df.d == 1, "d"] = dose
        with pytest.raises(MlsynthDataError, match="binary"):
            self._prep(df)

    def test_a_missing_treatment_cell_is_refused(self):
        # dataprep refuses it and names the cell; the setup has no check of
        # its own.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 5) & (df.time == 2002), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match=r"Missing treatment values: 1 of .*"
                           r"the first at unit '5', time 2002\."):
            self._prep(df)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError, match="No treated units"):
            self._prep(_panel(adopt={}))

    def test_one_treated_unit_at_the_first_period(self):
        # dataprep's own refusal on the single-treated path.
        with pytest.raises(MlsynthDataError, match="0 pre-periods"):
            self._prep(_panel(adopt={0: 0}))


# --------------------------------------------------------------------------- #
# SpSyDiD's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match=f"Required column '{col}' missing"):
            prepare_spsydid_inputs(df, **COLS, spatial_matrix=_line_W(8))

    def test_a_unit_order_that_does_not_match(self):
        with pytest.raises(MlsynthDataError, match="unit_order does not match"):
            prepare_spsydid_inputs(_panel(), **COLS, spatial_matrix=_line_W(8),
                                   unit_order=[0, 1, 2, 3, 4, 5, 6, 99])

    def test_a_matrix_of_the_wrong_size(self):
        with pytest.raises(MlsynthDataError):
            prepare_spsydid_inputs(_panel(), **COLS, spatial_matrix=_line_W(7))

    def test_a_cohort_at_the_first_period(self):
        # dataprep's cohort path does not refuse it; SpSyDiD does.
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_spsydid_inputs(_panel(adopt={0: 0, 1: 5}), **COLS,
                                   spatial_matrix=_line_W(8))

    def test_no_pure_controls(self):
        W = np.ones((6, 6)) - np.eye(6)
        with pytest.raises(MlsynthDataError, match="No pure controls"):
            prepare_spsydid_inputs(_panel(n_units=6), **COLS, spatial_matrix=W)

    def test_one_pre_period_is_enough(self):
        df = _panel(adopt={0: 1, 1: 4})
        inputs, ref = _both(df, _line_W(8))
        _assert_same(inputs, ref)
        assert inputs.T0 == 1


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(5, 9))
    T = draw(st.integers(3, 9))
    # Treated units at the start of the line, so units 3 and above are pure.
    adopt = {u: draw(st.integers(1, T - 1)) for u in range(draw(st.integers(1, 2)))}
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, n_periods=T, adopt=adopt, seed=seed, names=names)
    defect = draw(st.sampled_from(
        ["none", "none", "none", "nan", "inf", "gap", "dup", "off", "dose", "dnan"]))
    i = draw(st.integers(0, len(df) - 1))
    u = draw(st.sampled_from(sorted(adopt)))
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = -np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "off":
        if adopt[u] >= T - 1:
            defect = "none"
        else:
            df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + T - 1), "d"] = 0
    elif defect == "dose":
        df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + T - 1), "d"] = 2
    elif defect == "dnan":
        df["d"] = df["d"].astype(float)
        df.iloc[i, df.columns.get_loc("d")] = np.nan
    order = draw(st.sampled_from([None, "reversed"]))
    return df, defect, n_units, order


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_setup_or_refuses(case):
    df, defect, n_units, order = case
    kw = {} if order is None else dict(unit_order=sorted(df["unit"].unique())[::-1])
    W = _line_W(n_units)
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_spsydid_inputs(df, **COLS, spatial_matrix=W, **kw)
        return
    inputs, ref = _both(df, W, **kw)
    _assert_same(inputs, ref)
    # D is absorbing, T0 is the first treated period of any unit, and the
    # three classes partition the units.
    D = inputs.treatment_matrix
    assert (np.diff(D, axis=1) >= 0).all()
    assert D[:, :inputs.T0].sum() == 0 and D[:, inputs.T0].max() == 1
    parts = np.r_[inputs.direct_indices, inputs.spillover_indices, inputs.pure_control_indices]
    assert sorted(parts) == list(range(n_units))
