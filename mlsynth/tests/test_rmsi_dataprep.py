"""RMSI ingestion through ``datautils.dataprep`` (#1138, part 1).

RMSI used to pivot the long panel itself, outcome and treatment both. It now
takes the outcome matrix, the time labels and each treated unit's adoption
period from ``balance`` and ``dataprep``, and builds the treatment matrix ``D``
from the adoption periods. It gets the same refusals as every other estimator:
a missing or non-finite outcome cell, an absent or duplicate unit-period, a
non-binary treatment, and a treatment that switches off after it starts.
RMSI's own checks (three units, two pre-periods, two never-treated controls,
a block design, no unit treated in the first period) stay on top.

The side information is unchanged: ``X`` and ``Z`` are still the unit and
period means of the covariate columns, which may hold missing values.

``_reference`` is the old setup, kept as the oracle the new path must match on
every panel both accept. The covariate means are computed by the same pandas
call in both, on the same frame, so every comparison is exact.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import RMSI
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.rmsi_helpers import setup as SETUP
from mlsynth.utils.rmsi_helpers.setup import prepare_rmsi_inputs
from mlsynth.utils.rmsi_helpers.structures import RMSIInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, outcome, treat, unitid, time, unit_covariates=None,
               time_covariates=None):
    """The pre-#1138 setup: units sorted, times sorted, D read cell by cell."""
    unit_covariates = list(unit_covariates or [])
    time_covariates = list(time_covariates or [])
    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    unit_names = sorted(df[unitid].unique())
    N = len(unit_names)
    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    d_wide = df.pivot(index=unitid, columns=time, values=treat)
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = (d_wide.loc[unit_names, time_labels].to_numpy() != 0).astype(int)
    T0 = int(np.argmax(D.max(axis=0) == 1))
    if unit_covariates:
        X = np.column_stack([
            df.groupby(unitid, observed=True)[c].mean().loc[unit_names].to_numpy(dtype=float)
            for c in unit_covariates])
    else:
        X = np.zeros((N, 0))
    if time_covariates:
        Z = np.column_stack([
            df.groupby(time, observed=True)[c].mean().loc[time_labels].to_numpy(dtype=float)
            for c in time_covariates])
    else:
        Z = np.zeros((T, 0))
    return RMSIInputs(
        Y=Y, D=D, X=X, Z=Z, T0=T0, treated_idx=np.where(D.max(axis=1) == 1)[0],
        control_idx=np.where(D.max(axis=1) == 0)[0], unit_names=list(unit_names),
        time_labels=time_labels, unit_covariates=unit_covariates,
        time_covariates=time_covariates,
    )


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _panel(n_units=8, n_periods=12, treated=(0, 1), t0=6, seed=0, names="int",
           shuffle=True, start=2000):
    """Balanced long panel; the units in ``treated`` adopt at period index ``t0``.

    ``xu`` is roughly constant within a unit and ``zt`` within a period.
    """
    rng = np.random.default_rng(seed)
    zt = rng.normal(size=n_periods)
    rows = []
    for u in range(n_units):
        xu = rng.normal()
        for t in range(n_periods):
            d = int(u in treated and t >= t0)
            rows.append(dict(unit=_name(u, names), time=start + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d,
                             xu=xu + 1e-3 * rng.normal(), zt=zt[t] + 1e-3 * rng.normal()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, RMSIInputs)
    for f in ("Y", "D", "X", "Z", "treated_idx", "control_idx"):
        got, want = getattr(inputs, f), getattr(ref, f)
        assert got.dtype == want.dtype, f"{f}: dtype {got.dtype} vs {want.dtype}"
        assert got.shape == want.shape, f
        np.testing.assert_array_equal(got, want, err_msg=f)
    # The old Y was C-ordered (N, T); the solver's sums depend on the layout.
    assert inputs.Y.flags["C_CONTIGUOUS"] == ref.Y.flags["C_CONTIGUOUS"]
    assert inputs.T0 == ref.T0
    assert inputs.unit_names == ref.unit_names
    assert [type(u) for u in inputs.unit_names] == [type(u) for u in ref.unit_names]
    np.testing.assert_array_equal(np.asarray(inputs.time_labels), ref.time_labels)
    assert np.asarray(inputs.time_labels).dtype == ref.time_labels.dtype
    assert inputs.unit_covariates == ref.unit_covariates
    assert inputs.time_covariates == ref.time_covariates


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
        prepare_rmsi_inputs(_panel(), **COLS)
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
COVS = dict(unit_covariates=["xu"], time_covariates=["zt"])


class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),                                              # a block of two
        dict(treated=(3,), t0=8),                            # one treated unit
        dict(names="str", treated=(1, 4, 6)),
        dict(shuffle=False, start=-5, t0=11),                # last-period adoption
        dict(t0=2),                                          # two pre-periods
        dict(n_units=4, treated=(0, 3)),                     # two controls
        dict(n_units=40, n_periods=20, treated=tuple(range(0, 40, 3)), t0=14, seed=3),
        dict(n_units=40, n_periods=20, names="str", treated=(5,), t0=10, seed=4),
    ])
    @pytest.mark.parametrize("covs", [{}, COVS])
    def test_inputs_match_the_old_setup(self, kw, covs):
        df = _panel(**kw)
        _assert_same(prepare_rmsi_inputs(df, **COLS, **covs), _reference(df, **COLS, **covs))

    @pytest.mark.parametrize("label", ["datetime", "str", "float", "float32", "int32"])
    def test_period_labels_keep_their_type(self, label):
        # The old labels were np.array(sorted(...)): Timestamps as objects,
        # strings as a fixed-width unicode array, and 32-bit periods (a Stata
        # file's float32 time column) kept at 32 bits. Z is read at these labels.
        df = _panel()
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0,
                      "float32": (step / 4.0).astype(np.float32),
                      "int32": step.astype(np.int32)}[label]
        _assert_same(prepare_rmsi_inputs(df, **COLS, **COVS),
                     _reference(df, **COLS, **COVS))

    def test_missing_covariate_values_are_averaged_over_as_before(self):
        # Side information is not an outcome: a NaN covariate cell is skipped
        # by the mean, as it was.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time < 2004), "xu"] = np.nan
        df.loc[(df.time == 2003) & (df.unit < 3), "zt"] = np.nan
        _assert_same(prepare_rmsi_inputs(df, **COLS, **COVS),
                     _reference(df, **COLS, **COVS))

    def test_a_float_treatment_column_is_accepted(self):
        df = _panel()
        df["d"] = df["d"].astype(float)
        _assert_same(prepare_rmsi_inputs(df, **COLS), _reference(df, **COLS))

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _assert_same(prepare_rmsi_inputs(df, **COLS), _reference(df, **COLS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_rmsi_inputs(df, **COLS, **COVS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = RMSI(dict(df=_panel(n_units=12, n_periods=24, t0=18), **COLS, **COVS,
                        display_graphs=False)).fit()
        assert np.isfinite(res.att)
        assert res.inputs.treated_idx.tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_rmsi_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old setup checked only for NaN and let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2009), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_rmsi_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_rmsi_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_rmsi_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_missing_key_is_refused(self):
        df = _panel(shuffle=False).astype({"time": float})
        df.iloc[5, df.columns.get_loc("time")] = np.nan
        with pytest.raises(MlsynthDataError, match="key columns"):
            prepare_rmsi_inputs(df, **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup read only each unit's first treated period and kept
        # the 0 in D.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_rmsi_inputs(df, **COLS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, n_periods=20, t0=14)
        df.loc[(df.unit == 1) & (df.time == 2016), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            RMSI(dict(df=df, **COLS, display_graphs=False)).fit()

    @pytest.mark.parametrize("dose", [2, 0.5, -1])
    def test_a_non_binary_treatment_is_refused(self, dose):
        # The old setup read any non-zero value as treated.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[df.d == 1, "d"] = dose
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_rmsi_inputs(df, **COLS)

    def test_a_missing_treatment_cell_is_refused(self):
        # dataprep's binary check skips NaN; the old pivot refused it as an
        # absent cell, and RMSI keeps that refusal.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 5) & (df.time == 2002), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="Treatment column"):
            prepare_rmsi_inputs(df, **COLS)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError, match="No treated"):
            prepare_rmsi_inputs(_panel(treated=()), **COLS)


# --------------------------------------------------------------------------- #
# RMSI's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time", "xu", "zt"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match=f"Missing columns: {col}"):
            prepare_rmsi_inputs(df, **COLS, **COVS)

    @pytest.mark.parametrize("n_units,treated", [(2, (0,)), (1, (0,))])
    def test_fewer_than_three_units(self, n_units, treated):
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_rmsi_inputs(_panel(n_units=n_units, treated=treated), **COLS)

    @pytest.mark.parametrize("n_units,treated", [(3, (0, 1)), (3, (0, 1, 2)),
                                                 (4, (0, 1, 3))])
    def test_fewer_than_two_controls(self, n_units, treated):
        with pytest.raises(MlsynthDataError, match="at least 2 never-treated control"):
            prepare_rmsi_inputs(_panel(n_units=n_units, treated=treated), **COLS)

    def test_staggered_adoption_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time.isin([2006, 2007])), "d"] = 0
        with pytest.raises(MlsynthDataError, match="block design"):
            prepare_rmsi_inputs(df, **COLS)

    def test_one_treated_unit_at_the_first_period(self):
        # dataprep's own refusal on the single-treated path.
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_rmsi_inputs(_panel(treated=(0,), t0=0), **COLS)

    def test_a_block_at_the_first_period(self):
        # dataprep's cohort path does not refuse it; RMSI does.
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_rmsi_inputs(_panel(t0=0), **COLS)

    @pytest.mark.parametrize("treated", [(0,), (0, 1)])
    def test_one_pre_period_is_too_few(self, treated):
        with pytest.raises(MlsynthDataError, match="at least 2 pre-treatment periods"):
            prepare_rmsi_inputs(_panel(treated=treated, t0=1), **COLS)


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
    covs = draw(st.sampled_from([{}, COVS]))
    return df, defect, covs


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_setup_or_refuses(case):
    df, defect, covs = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_rmsi_inputs(df, **COLS, **covs)
        return
    inputs = prepare_rmsi_inputs(df, **COLS, **covs)
    _assert_same(inputs, _reference(df, **COLS, **covs))
    # D is absorbing and a block: every treated row switches on at T0, and
    # the treated and control rows partition the units.
    assert (np.diff(inputs.D, axis=1) >= 0).all()
    assert inputs.D[:, :inputs.T0].sum() == 0
    assert (inputs.D[inputs.treated_idx, inputs.T0:] == 1).all()
    assert inputs.D[inputs.control_idx].sum() == 0
    assert sorted(np.r_[inputs.treated_idx, inputs.control_idx]) == list(range(len(inputs.unit_names)))
