"""SSC ingestion through ``datautils.dataprep`` (#1138, part 1).

SSC used to pivot the long panel itself, outcome and treatment both. It now
takes the outcome matrix, the time labels and each treated unit's adoption
period from ``balance`` and ``dataprep``, and builds the treatment matrix ``D``
and the adoption vector from the adoption periods. It gets the same refusals as
every other estimator: a missing or non-finite outcome cell, an absent or
duplicate unit-period, a non-binary treatment, and a treatment that switches
off after it starts. SSC's own checks (three units, two clean pre-periods, no
unit treated in the first period, no missing treatment value) stay on top.

Staggered adoption is what SSC is for: ``dataprep`` returns a ``CohortPanel``
for such a panel, and ``D`` is rebuilt from its cohorts.

``_reference`` is the old pivot, kept as the oracle the new path must match on
every panel both accept. Nothing in the setup sums or averages, so the
comparison is exact.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import SSC
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.ssc_helpers import setup as SETUP
from mlsynth.utils.ssc_helpers.setup import prepare_ssc_inputs
from mlsynth.utils.ssc_helpers.structures import SSCInputs

ARGS = ("y", "d", "unit", "time")


def _reference(df, outcome, treat, unitid, time):
    """The pre-#1138 pivot: units sorted, times sorted, D read cell by cell."""
    time_labels = np.array(sorted(df[time].unique()))
    unit_names = sorted(df[unitid].unique())
    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = (d_wide.loc[unit_names, time_labels].to_numpy() != 0).astype(int)
    T0 = int(np.argmax(D.max(axis=0) == 1))
    adoption = np.full(len(unit_names), -1, dtype=int)
    for i in range(len(unit_names)):
        on = np.where(D[i] == 1)[0]
        if on.size:
            adoption[i] = int(on[0])
    return SSCInputs(Y=Y, D=D, T0=T0, unit_names=list(unit_names),
                     time_labels=time_labels,
                     treated_idx=np.where(adoption >= 0)[0], adoption=adoption)


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _panel(n_units=8, n_periods=12, adopt=None, seed=0, names="int",
           shuffle=True, t0=2000):
    """Balanced long panel; ``adopt`` maps unit index -> adoption period index."""
    adopt = {0: 6} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            d = int(u in adopt and t >= adopt[u])
            rows.append(dict(unit=_name(u, names), time=t0 + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(got, ref):
    assert isinstance(got, SSCInputs)
    for f in ("Y", "D", "treated_idx", "adoption", "time_labels"):
        a, b = getattr(got, f), getattr(ref, f)
        assert a.dtype == b.dtype, f"{f}: dtype {a.dtype} vs {b.dtype}"
        assert a.shape == b.shape, f
        # The solver's sums depend on the layout, so it is pinned too.
        assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"], f
        assert a.flags["F_CONTIGUOUS"] == b.flags["F_CONTIGUOUS"], f
        np.testing.assert_array_equal(a, b, err_msg=f)
    assert got.T0 == ref.T0
    assert got.unit_names == ref.unit_names
    assert [type(u) for u in got.unit_names] == [type(u) for u in ref.unit_names]


def _both(df):
    got = prepare_ssc_inputs(df, *ARGS)
    _assert_same(got, _reference(df, *ARGS))
    return got


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
        prepare_ssc_inputs(_panel(adopt={0: 4, 3: 7}), *ARGS)
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
        _both(_panel(**kw))

    def test_staggered_adoption_starts_at_the_first_cohort(self):
        inputs = _both(_panel(adopt={0: 4, 2: 4, 5: 9}))
        assert inputs.T0 == 4
        assert list(inputs.treated_idx) == [0, 2, 5]
        assert list(inputs.adoption) == [4, -1, 4, -1, -1, 9, -1, -1]
        assert inputs.D[5, :9].sum() == 0 and inputs.D[5, 9:].min() == 1

    @pytest.mark.parametrize("label", ["datetime", "str", "float", "float32", "int32"])
    def test_period_labels_keep_their_type(self, label):
        # A 32-bit period column stays 32-bit, as np.array(sorted(...)) kept it.
        df = _panel(adopt={1: 7})
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0,
                      "float32": (step / 4.0).astype("float32"),
                      "int32": df["time"].astype("int32")}[label]
        _both(df)

    def test_a_boolean_treatment_is_read_as_before(self):
        df = _panel(adopt={0: 5, 2: 8})
        df["d"] = df["d"].astype(bool)
        _both(df)

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _both(df)

    def test_the_caller_frame_is_not_changed(self):
        df = _panel(adopt={0: 5, 2: 8})
        before = df.copy()
        prepare_ssc_inputs(df, *ARGS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        df = _panel(n_units=10, n_periods=30, adopt={0: 20, 1: 24, 2: 24})
        res = SSC(dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
                       display_graphs=False)).fit()
        assert np.isfinite(res.att)


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_ssc_inputs(df, *ARGS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old setup checked only for NaN and let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2009), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_ssc_inputs(df, *ARGS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_ssc_inputs(df, *ARGS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_ssc_inputs(pd.concat([df, df.iloc[[7]]]), *ARGS)

    def test_a_treatment_that_switches_off_is_refused(self):
        df = _panel(shuffle=False, adopt={0: 4, 1: 6})
        df.loc[(df.unit == 1) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_ssc_inputs(df, *ARGS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, n_units=10, n_periods=30, adopt={0: 20, 1: 24})
        df.loc[(df.unit == 1) & (df.time == 2027), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            SSC(dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
                     display_graphs=False)).fit()

    @pytest.mark.parametrize("value", [2, -1])
    def test_a_non_binary_treatment_is_refused(self, value):
        # The old setup read any nonzero code as treated.
        df = _panel(shuffle=False, adopt={0: 6, 1: 6})
        df.loc[(df.unit == 1) & (df.time >= 2008), "d"] = value
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_ssc_inputs(df, *ARGS)

    def test_a_missing_treatment_cell_is_refused(self):
        # dataprep's binary check skips NaN; the old pivot refused it as an
        # absent cell, and SSC keeps that refusal.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 5) & (df.time == 2002), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="Treatment column"):
            prepare_ssc_inputs(df, *ARGS)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError, match="No treated"):
            prepare_ssc_inputs(_panel(adopt={}), *ARGS)


# --------------------------------------------------------------------------- #
# SSC's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match=f"Required column '{col}' missing"):
            prepare_ssc_inputs(df, *ARGS)

    @pytest.mark.parametrize("n_units", [1, 2])
    def test_fewer_than_three_units(self, n_units):
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_ssc_inputs(_panel(n_units=n_units), *ARGS)

    def test_three_units_are_enough(self):
        _both(_panel(n_units=3))

    def test_one_treated_unit_at_the_first_period(self):
        # dataprep's own refusal on the single-treated path.
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_ssc_inputs(_panel(adopt={0: 0}), *ARGS)

    def test_a_cohort_at_the_first_period(self):
        # dataprep's cohort path does not refuse it; SSC does.
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_ssc_inputs(_panel(adopt={0: 0, 1: 5}), *ARGS)

    @pytest.mark.parametrize("adopt", [{0: 1}, {0: 1, 1: 5}])
    def test_one_pre_period_is_too_few(self, adopt):
        with pytest.raises(MlsynthDataError, match="at least 2 clean pre-treatment"):
            prepare_ssc_inputs(_panel(adopt=adopt), *ARGS)

    @pytest.mark.parametrize("adopt", [{0: 2}, {0: 2, 1: 5}])
    def test_two_pre_periods_are_enough(self, adopt):
        assert _both(_panel(adopt=adopt)).T0 == 2


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
        df.iloc[i, df.columns.get_loc("d")] = 2
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
            prepare_ssc_inputs(df, *ARGS)
        return
    inputs = _both(df)
    # D is absorbing, T0 is the first treated period of any unit, each unit's
    # adoption is its first treated column, and the treated units are exactly
    # the rows of D with a treated cell.
    assert (np.diff(inputs.D, axis=1) >= 0).all()
    assert inputs.D[:, :inputs.T0].sum() == 0 and inputs.D[:, inputs.T0].max() == 1
    for i, a in enumerate(inputs.adoption):
        assert (a == -1) == (inputs.D[i].sum() == 0)
        if a >= 0:
            assert inputs.D[i, a] == 1 and inputs.D[i, :a].sum() == 0
    np.testing.assert_array_equal(inputs.treated_idx, np.flatnonzero(inputs.D.any(axis=1)))
