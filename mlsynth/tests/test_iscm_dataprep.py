"""ISCM ingestion through ``datautils.dataprep`` (#1138, part 1).

ISCM used to pivot the long panel itself, outcome and treatment both. It now
takes the outcome matrix, the time labels and each treated unit's adoption
period from ``balance`` and ``dataprep``, and builds the treatment matrix ``D``
from the adoption periods. It gets the same refusals as every other estimator:
a missing or non-finite outcome cell, an absent or duplicate unit-period, a
non-binary treatment, and a treatment that switches off after it starts.

Staggered adoption is accepted as before. The old setup took ``T0`` as the
first period any unit is treated and counted every unit treated from then on
as treated; ``dataprep`` returns a ``CohortPanel`` for such a panel and ``D``
is rebuilt from its cohorts, so the inputs are unchanged.

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

from mlsynth import ISCM
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.iscm_helpers import setup as SETUP
from mlsynth.utils.iscm_helpers.setup import prepare_iscm_inputs
from mlsynth.utils.iscm_helpers.structures import ISCMInputs

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
    return dict(Y=Y, D=D, T0=T0, unit_names=list(unit_names),
                time_labels=time_labels,
                treated_idx=np.where(D[:, T0:].max(axis=1) == 1)[0])


def _panel(n_units=8, n_periods=12, adopt=None, seed=0, names="int",
           shuffle=True, t0=2000):
    """Balanced long panel; ``adopt`` maps unit index -> adoption period index."""
    adopt = {0: 6} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            d = int(u in adopt and t >= adopt[u])
            rows.append(dict(unit=u if names == "int" else f"s{u:02d}",
                             time=t0 + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d,
                             x=rng.normal()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _assert_same(inputs, ref):
    assert isinstance(inputs, ISCMInputs)
    for f in ("Y", "D", "treated_idx"):
        got = getattr(inputs, f)
        assert got.dtype == ref[f].dtype, f"{f}: dtype {got.dtype} vs {ref[f].dtype}"
        assert got.shape == ref[f].shape, f
        np.testing.assert_array_equal(got, ref[f], err_msg=f)
    # The old Y was C-ordered (N, T); the solver's sums depend on the layout.
    assert inputs.Y.flags["C_CONTIGUOUS"] == ref["Y"].flags["C_CONTIGUOUS"]
    assert inputs.T0 == ref["T0"]
    assert inputs.unit_names == ref["unit_names"]
    assert [type(u) for u in inputs.unit_names] == [type(u) for u in ref["unit_names"]]
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
        prepare_iscm_inputs(_panel(), **COLS)
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
        dict(names="str", adopt={3: 8}),
        dict(shuffle=False, t0=-5, adopt={2: 11}),           # last-period adoption
        dict(adopt={0: 5, 1: 5}),                            # one cohort of two
        dict(adopt={0: 4, 1: 6, 4: 9}),                      # staggered, three cohorts
        dict(names="str", adopt={u: 3 + (u % 4) for u in range(8)}),  # all treated
        dict(n_units=40, n_periods=20, adopt={0: 12}, seed=3),           # 40 units
        dict(n_units=40, n_periods=20, names="str", seed=4,
             adopt={0: 10, 5: 10, 9: 14, 33: 17}),                       # 40, staggered
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_iscm_inputs(df, **COLS), _reference(df, **COLS))

    def test_staggered_adoption_starts_at_the_first_cohort(self):
        # ISCM's T0 is the first period any unit is treated; a later cohort
        # is treated in D from its own start and counted among treated_idx.
        df = _panel(adopt={0: 4, 2: 4, 5: 9})
        inputs = prepare_iscm_inputs(df, **COLS)
        assert inputs.T0 == 4
        assert list(inputs.treated_idx) == [0, 2, 5]
        assert inputs.D[5, :9].sum() == 0 and inputs.D[5, 9:].min() == 1
        assert inputs.D[[1, 3, 4, 6, 7]].sum() == 0

    @pytest.mark.parametrize("label", ["datetime", "str", "float", "float32", "int32"])
    def test_period_labels_keep_their_type(self, label):
        # The old labels were np.array(sorted(...)): Timestamps as objects,
        # strings as a fixed-width unicode array, and a 32-bit period column
        # (a Stata float32 year, say) kept at 32 bits.
        df = _panel(adopt={1: 7})
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0,
                      "float32": (step / 4.0).astype("float32"),
                      "int32": df["time"].astype("int32")}[label]
        _assert_same(prepare_iscm_inputs(df, **COLS), _reference(df, **COLS))

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _assert_same(prepare_iscm_inputs(df, **COLS), _reference(df, **COLS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_iscm_inputs(df, **COLS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = ISCM(dict(df=_panel(n_periods=20, adopt={0: 14}), **COLS,
                        display_graphs=False, inference=False)).fit()
        assert np.isfinite(res.att)

    def test_staggered_fit_smoke(self):
        res = ISCM(dict(df=_panel(n_periods=20, adopt={0: 12, 1: 15}), **COLS,
                        display_graphs=False, inference=False)).fit()
        assert np.isfinite(res.att)
        assert res.metadata["n_treated"] == 2


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_iscm_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old setup checked only for NaN and let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2009), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_iscm_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_iscm_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_iscm_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old pivot read D cell by cell and accepted this.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_iscm_inputs(df, **COLS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, n_periods=20, adopt={0: 12, 1: 14})
        df.loc[(df.unit == 1) & (df.time == 2016), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            ISCM(dict(df=df, **COLS, display_graphs=False, inference=False)).fit()

    def test_a_non_binary_treatment_is_refused(self):
        # {0, 1, 2}: the old setup kept the 2 in D.
        df = _panel(shuffle=False, adopt={0: 6, 1: 6})
        df.loc[(df.unit == 1) & (df.time >= 2008), "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_iscm_inputs(df, **COLS)

    def test_a_missing_treatment_cell_is_refused(self):
        # dataprep's binary check skips NaN; the old pivot refused it as an
        # absent cell, and ISCM keeps that refusal.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 5) & (df.time == 2002), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="Treatment column"):
            prepare_iscm_inputs(df, **COLS)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError, match="No treated"):
            prepare_iscm_inputs(_panel(adopt={}), **COLS)


# --------------------------------------------------------------------------- #
# ISCM's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match=f"Required column '{col}' missing"):
            prepare_iscm_inputs(df, **COLS)

    @pytest.mark.parametrize("n_units", [1, 2])
    def test_fewer_than_three_units(self, n_units):
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_iscm_inputs(_panel(n_units=n_units), **COLS)

    def test_three_units_are_enough(self):
        df = _panel(n_units=3)
        _assert_same(prepare_iscm_inputs(df, **COLS), _reference(df, **COLS))

    def test_one_treated_unit_at_the_first_period(self):
        # dataprep's own refusal on the single-treated path.
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_iscm_inputs(_panel(adopt={0: 0}), **COLS)

    def test_a_cohort_at_the_first_period(self):
        # dataprep's cohort path does not refuse it; ISCM does.
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_iscm_inputs(_panel(adopt={0: 0, 1: 5}), **COLS)

    @pytest.mark.parametrize("adopt", [{0: 1}, {0: 1, 1: 5}])
    def test_one_pre_period_is_too_few(self, adopt):
        with pytest.raises(MlsynthDataError, match="at least 2 pre-treatment periods"):
            prepare_iscm_inputs(_panel(adopt=adopt), **COLS)

    @pytest.mark.parametrize("adopt", [{0: 2}, {0: 2, 1: 5}])
    def test_two_pre_periods_are_enough(self, adopt):
        df = _panel(adopt=adopt)
        inputs = prepare_iscm_inputs(df, **COLS)
        assert inputs.T0 == 2
        _assert_same(inputs, _reference(df, **COLS))


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(3, 9))
    T = draw(st.integers(4, 10))
    n_treated = draw(st.integers(1, n_units))
    adopt = {u: draw(st.integers(2, T - 1)) for u in range(n_treated)}
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, n_periods=T, adopt=adopt, seed=seed, names=names)
    defect = draw(st.sampled_from(
        ["none", "none", "none", "nan", "inf", "gap", "dup", "off", "dose", "dnan"]))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = -np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "off":
        u = draw(st.sampled_from(sorted(adopt)))
        if adopt[u] >= T - 1:
            defect = "none"
        else:
            df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + T - 1), "d"] = 0
    elif defect == "dose":
        u = draw(st.sampled_from(sorted(adopt)))
        df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + T - 1), "d"] = 2
    elif defect == "dnan":
        df["d"] = df["d"].astype(float)
        df.iloc[i, df.columns.get_loc("d")] = np.nan
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_iscm_inputs(df, **COLS)
        return
    inputs = prepare_iscm_inputs(df, **COLS)
    _assert_same(inputs, _reference(df, **COLS))
    # D is absorbing, T0 is the first treated period of any unit, and the
    # treated units are exactly the rows of D with a treated cell.
    assert (np.diff(inputs.D, axis=1) >= 0).all()
    assert inputs.D[:, :inputs.T0].sum() == 0 and inputs.D[:, inputs.T0].max() == 1
    np.testing.assert_array_equal(inputs.treated_idx, np.flatnonzero(inputs.D.any(axis=1)))
