"""SNN ingestion through ``datautils.dataprep`` (#1138, part 1).

SNN used to pivot the long panel itself. It now takes the outcome matrix, the
time labels and each treated unit's adoption period from ``balance`` and
``dataprep``, so it gets the same refusals as every other estimator: a missing
or non-finite outcome cell, an absent or duplicate unit-period, a non-binary
treatment, and a treatment that switches off after it starts. The last one is a
change: the old pivot read the treatment cell by cell and accepted a treatment
that stops. The setup's own docstring described a unit treated from its adoption
period onward, and the estimator page imputes the treated post-treatment cells,
with one-time and staggered adoption as its panel designs.

The matrix-completion engine, ``snn_complete``, still takes a matrix with
``NaN`` for its missing entries; only the panel path goes through ``dataprep``.

``_reference`` below is the old pivot, kept as the oracle the new path must
match on every panel both accept.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import SNN
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.snn_helpers import setup as SETUP
from mlsynth.utils.snn_helpers import snn_complete
from mlsynth.utils.snn_helpers.setup import prepare_snn_inputs
from mlsynth.utils.snn_helpers.structures import SNNInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, outcome, treat, unitid, time):
    """The pre-#1138 pivot: units sorted, times sorted, D read cell by cell."""
    time_labels = np.array(sorted(df[time].unique()))
    unit_names = sorted(df[unitid].unique())
    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = d_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    T0 = int(np.argmax(D.max(axis=0) == 1))
    return dict(Y=Y, D=D, treated_idx=np.where(D.max(axis=1) == 1)[0], T0=T0,
                unit_names=list(unit_names), time_labels=time_labels)


def _panel(n_units=8, n_periods=12, adopt=None, seed=0, names="int",
           shuffle=True, t0=2000, boolean=False):
    """Balanced long panel; ``adopt`` maps unit index -> adoption period index."""
    adopt = {0: 6} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            d = int(u in adopt and t >= adopt[u])
            rows.append(dict(unit=u if names == "int" else f"s{u:02d}",
                             time=t0 + t,
                             y=rng.normal() + u + 0.2 * t + 1.5 * d,
                             d=bool(d) if boolean else d))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, SNNInputs)
    for f in ("Y", "D", "treated_idx"):
        got = getattr(inputs, f)
        assert got.dtype == ref[f].dtype, f"{f}: dtype {got.dtype} vs {ref[f].dtype}"
        np.testing.assert_array_equal(got, ref[f], err_msg=f)
    assert inputs.T0 == ref["T0"]
    assert list(inputs.unit_names) == ref["unit_names"]
    np.testing.assert_array_equal(np.asarray(inputs.time_labels), ref["time_labels"])
    assert np.asarray(inputs.time_labels).dtype == ref["time_labels"].dtype


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
        prepare_snn_inputs(_panel(), **COLS)
        assert len(calls) == 1
        assert calls[0] is not None          # the balance() key scan is passed on

    def test_setup_does_not_pivot(self):
        # The setup module itself must not reshape the panel. A runtime patch of
        # DataFrame.pivot cannot test this: dataprep's _wide_pivot falls back to
        # df.pivot on pandas versions whose fast path would change the array
        # layout, so the patch fires inside dataprep on those versions.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),                                         # one treated unit
        dict(adopt={0: 6, 1: 6}),                       # one cohort, two units
        dict(adopt={0: 5, 1: 7, 2: 7, 3: 10}),          # staggered
        dict(names="str", adopt={0: 4, 3: 9}),          # string ids
        dict(shuffle=False, adopt={2: 1}),              # adoption at period 1
        dict(n_periods=7, adopt={0: 6}),                # one post period
        dict(t0=-3, adopt={1: 3}),                      # negative time labels
        dict(boolean=True, adopt={0: 5, 4: 8}),         # boolean treatment, as Prop 99
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_snn_inputs(df, **COLS), _reference(df, **COLS))

    def test_extra_columns_are_ignored(self):
        df = _panel().assign(other=1.0, region="r")
        _assert_same(prepare_snn_inputs(df, **COLS), _reference(df, **COLS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel(adopt={0: 5, 1: 8})
        before = df.copy()
        prepare_snn_inputs(df, **COLS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = SNN(dict(df=_panel(adopt={0: 6, 1: 8}), display_graphs=False,
                       **COLS)).fit()
        assert np.isfinite(res.att)

    def test_the_completion_engine_still_takes_missing_entries(self):
        X = np.array([[1.0, 2.0, np.nan], [2.0, 4.0, 6.0], [3.0, np.nan, 9.0]])
        completed, feasible = snn_complete(X)
        assert completed.shape == X.shape and feasible.shape == X.shape


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_treatment_that_switches_off_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time >= 2009), "d"] = 0   # on at 2006, off at 2009
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_snn_inputs(df, **COLS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, adopt={0: 6, 1: 6})
        df.loc[(df.unit == 1) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            SNN(dict(df=df, display_graphs=False, **COLS)).fit()

    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            SNN(dict(df=df, display_graphs=False, **COLS)).fit()

    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_snn_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 4) & (df.time == 2003))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            SNN(dict(df=df, display_graphs=False, **COLS)).fit()

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_snn_inputs(pd.concat([df, df.iloc[[5]]]), **COLS)

    def test_a_non_binary_treatment_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[df.d == 1, "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_snn_inputs(df, **COLS)


# --------------------------------------------------------------------------- #
# SNN's own requirements, kept on top of dataprep
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time"])
    def test_missing_column(self, col):
        with pytest.raises(MlsynthDataError, match=f"Required column '{col}' missing"):
            prepare_snn_inputs(_panel().drop(columns=col), **COLS)

    def test_two_units_are_too_few(self):
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_snn_inputs(_panel(n_units=2), **COLS)

    def test_one_control_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="2 control"):
            prepare_snn_inputs(_panel(n_units=3, adopt={0: 5, 1: 5}), **COLS)

    def test_two_controls_are_enough(self):
        df = _panel(n_units=3, adopt={0: 5})
        _assert_same(prepare_snn_inputs(df, **COLS), _reference(df, **COLS))

    def test_treatment_at_the_first_period_is_refused(self):
        with pytest.raises(MlsynthDataError):
            prepare_snn_inputs(_panel(adopt={0: 0}), **COLS)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError):
            prepare_snn_inputs(_panel(adopt={}), **COLS)


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(3, 9))
    n_periods = draw(st.integers(2, 10))
    n_treated = draw(st.integers(1, n_units - 2))
    adopt = {u: draw(st.integers(1, n_periods - 1)) for u in range(n_treated)}
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    boolean = draw(st.booleans())
    df = _panel(n_units=n_units, n_periods=n_periods, adopt=adopt, seed=seed,
                names=names, boolean=boolean)
    defect = draw(st.sampled_from(
        ["none", "none", "none", "nan", "inf", "gap", "dup", "off", "nonbinary"]))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = -np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "nonbinary":
        df["d"] = df["d"].astype(int) * 2
    elif defect == "off":
        u = draw(st.sampled_from(sorted(adopt)))
        if adopt[u] < n_periods - 1:
            stop = draw(st.integers(adopt[u] + 1, n_periods - 1))
            name = u if names == "int" else f"s{u:02d}"
            df.loc[(df.unit == name) & (df.time >= 2000 + stop), "d"] = (
                False if boolean else 0)
        else:
            defect = "none"
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_snn_inputs(df, **COLS)
        return
    inputs = prepare_snn_inputs(df, **COLS)
    _assert_same(inputs, _reference(df, **COLS))
    # D is absorbing, T0 is the first adoption, and at least two units are
    # never treated.
    assert np.all(np.diff(inputs.D, axis=1) >= 0)
    assert inputs.T0 == int(np.argmax(inputs.D.max(axis=0) == 1)) >= 1
    assert inputs.N - inputs.treated_idx.size >= 2
