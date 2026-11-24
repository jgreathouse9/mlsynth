"""MicroSynth ingestion through ``datautils.dataprep`` (#1138, part 1).

MicroSynth already balanced the panel; it then pivoted the post-period rows
itself and checked only those cells for missing outcomes. It now reads the
post-period outcome matrix from ``dataprep``, so it gets the same refusals as
every other estimator: a missing or non-finite outcome in any period, a
non-binary treatment, and a treatment that switches off after it starts.

``_reference_post`` is the old pivot, kept as the oracle the new path must
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

from mlsynth import MicroSynth
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.microsynth_helpers import setup as SETUP
from mlsynth.utils.microsynth_helpers.setup import prepare_microsynth_inputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference_post(df, treated, controls, cohort_time):
    """The pre-#1138 post-period pivot, one row per unit."""
    post = df[df["time"] >= cohort_time]
    times = np.sort(post["time"].unique())

    def block(units):
        return (post[post["unit"].isin(units)]
                .pivot(index="unit", columns="time", values="y")
                .reindex(index=units, columns=times).to_numpy(dtype=float))

    Y_T, Y_C = block(treated), block(controls)
    if Y_T.shape[1] == 1:
        Y_T, Y_C = Y_T[:, 0], Y_C[:, 0]
    return Y_T, Y_C


def _panel(n_c=12, n_t=4, T0=5, T_post=3, seed=0, names="int", shuffle=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_c + n_t):
        treated = i >= n_c
        x = rng.normal()
        for t in range(T0 + T_post):
            post = treated and t >= T0
            rows.append(dict(unit=i if names == "int" else f"b{i:03d}", time=t,
                             y=rng.normal() + x + 0.1 * t + (1.0 if post else 0.0),
                             d=int(post), x=x))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _check(df, **kw):
    kw = dict(dict(covariates=["x"]), **kw)
    inputs = prepare_microsynth_inputs(df, **COLS, **kw)
    Y_T, Y_C = _reference_post(df, inputs.treated_unit_names,
                               inputs.control_unit_names, inputs.cohort_time)
    assert inputs.Y_T.dtype == np.float64 and inputs.Y_C.dtype == np.float64
    np.testing.assert_array_equal(inputs.Y_T, Y_T)
    np.testing.assert_array_equal(inputs.Y_C, Y_C)
    return inputs


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_setup_calls_dataprep_once(self, monkeypatch):
        calls = []
        real = SETUP.dataprep

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "dataprep", spy)
        prepare_microsynth_inputs(_panel(), **COLS, covariates=["x"])
        assert len(calls) == 1

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
        dict(T_post=1),                       # one post period: vectors
        dict(n_t=1),                          # one treated unit
        dict(names="str", shuffle=False),
    ])
    def test_post_outcomes_match_the_old_pivot(self, kw):
        _check(_panel(**kw))

    def test_lagged_outcomes_are_unchanged(self):
        df = _panel()
        inputs = _check(df, outcome_lag_periods=[1, 3])
        lag = df[df.time == 3].set_index("unit")["y"]
        np.testing.assert_array_equal(inputs.lag_T_raw[:, 1],
                                      lag.reindex(inputs.treated_unit_names).to_numpy())

    def test_fit_smoke(self):
        res = MicroSynth(dict(df=_panel(), **COLS, covariates=["x"],
                              run_inference=False, display_graphs=False)).fit()
        assert np.all(np.isfinite(res.design.w))


# --------------------------------------------------------------------------- #
# refusals now come from dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_pre_period_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 1), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])

    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 13) & (df.time == 6), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])

    def test_a_treatment_that_switches_off_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 12) & (df.time == 7), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])

    def test_a_non_binary_treatment_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[df.d == 1, "d"] = 2
        with pytest.raises(MlsynthDataError):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 4) & (df.time == 6))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])


# --------------------------------------------------------------------------- #
# MicroSynth's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_two_cohorts_are_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 15) & (df.time == 5), "d"] = 0   # starts one period later
        with pytest.raises(MlsynthDataError, match="single cohort"):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])

    def test_no_control_unit(self):
        df = _panel(n_c=0, n_t=3)
        with pytest.raises(MlsynthDataError, match="No control"):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])

    def test_no_constraint(self):
        with pytest.raises(MlsynthDataError, match="balancing constraint"):
            prepare_microsynth_inputs(_panel(), **COLS, covariates=[])


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_c = draw(st.integers(2, 10))
    n_t = draw(st.integers(1, 5))
    T0 = draw(st.integers(1, 6))
    T_post = draw(st.integers(1, 4))
    names = draw(st.sampled_from(["int", "str"]))
    df = _panel(n_c=n_c, n_t=n_t, T0=T0, T_post=T_post,
                seed=draw(st.integers(0, 10_000)), names=names)
    defect = draw(st.sampled_from(["none", "none", "nan", "inf", "gap", "off"]))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = -np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "off":
        if T_post < 2:
            defect = "none"
        else:
            unit = df.loc[df.d == 1, "unit"].iloc[0]
            df.loc[(df.unit == unit) & (df.time == T0 + T_post - 1), "d"] = 0
    return df, defect


@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_microsynth_inputs(df, **COLS, covariates=["x"])
        return
    inputs = _check(df)
    # Every post period is a column, in time order, and none is pre-period.
    assert inputs.T_post == df.loc[df.time >= inputs.cohort_time, "time"].nunique()
