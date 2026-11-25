"""CTSC ingestion through ``datautils.geoex_dataprep`` (#1138, part 1).

CTSC used to pivot the outcome and each of its ``K`` treatment variables
itself. Its treatments are continuous or multi-valued intensities, which
``dataprep`` refuses as a non-binary treatment, so the panel is read through
``geoex_dataprep``, the treatment-agnostic reader: once for the outcome, and
once for each treatment variable read as that call's outcome. CTSC now gets the
same refusals as every other estimator: a duplicate or absent unit-period, a
missing key, and a missing or non-finite value of the outcome or of any
treatment variable, the last raised under the variable's name.

``_reference`` is the old pivot, kept as the oracle the new path must match on
every panel both accept, down to array layout, label dtype and the type of each
unit label.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ctsc_helpers import setup as SETUP
from mlsynth.utils.ctsc_helpers.setup import prepare_ctsc_inputs
from mlsynth.utils.ctsc_helpers.structures import CTSCInputs

COLS = dict(outcome="y", unitid="unit", time="time")


def _reference(df, outcome, treatment_vars, unitid, time, population_col=None):
    """The pre-#1138 setup, verbatim after its column and NaN checks."""
    treatment_vars = list(treatment_vars)
    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    unit_names = sorted(df[unitid].unique())
    n = len(unit_names)
    y_wide = df.pivot(index=unitid, columns=time, values=outcome)
    Y = y_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    D = np.empty((n, T, len(treatment_vars)), dtype=float)
    for k, col in enumerate(treatment_vars):
        w = df.pivot(index=unitid, columns=time, values=col)
        D[:, :, k] = w.loc[unit_names, time_labels].to_numpy(dtype=float)
    if population_col is not None:
        pop = (df.groupby(unitid, observed=True)[population_col].first()
               .loc[unit_names].to_numpy(float))
        population_weights = pop / pop.sum()
    else:
        population_weights = np.full(n, 1.0 / n)
    return dict(Y=Y, D=D, unit_names=list(unit_names), time_labels=time_labels,
                treatment_names=treatment_vars, population_weights=population_weights)


def _unit(u, names):
    return u if names == "int" else f"s{u:02d}"


def _time(t, times, t0):
    if times == "int":
        return t0 + t
    if times == "date":
        return pd.Timestamp("2000-01-01") + pd.DateOffset(months=3 * t)
    return f"q{t:03d}"


def _panel(n_units=6, T=10, K=2, seed=0, names="int", times="int", shuffle=True,
           t0=1990):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            row = dict(unit=_unit(u, names), time=_time(t, times, t0),
                       pop=float(1 + (7 * u) % 5))
            dose = [rng.normal() + 0.1 * t, float(rng.integers(0, 4)), rng.normal()]
            for k in range(K):
                row[f"d{k}"] = dose[k]
            row["y"] = rng.normal() + u + sum(0.5 * dose[k] for k in range(K))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _tv(K):
    return [f"d{k}" for k in range(K)]


def _assert_same(inputs, ref):
    assert isinstance(inputs, CTSCInputs)
    for name in ("Y", "D"):
        got, want = getattr(inputs, name), ref[name]
        assert got.dtype == np.float64
        assert got.flags["C_CONTIGUOUS"], name
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(inputs.time_labels, ref["time_labels"])
    assert inputs.time_labels.dtype == ref["time_labels"].dtype
    assert inputs.unit_names == ref["unit_names"]
    assert [type(u) for u in inputs.unit_names] == [type(u) for u in ref["unit_names"]]
    assert list(inputs.treatment_names) == ref["treatment_names"]
    np.testing.assert_array_equal(inputs.population_weights, ref["population_weights"])


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    @pytest.mark.parametrize("K", [1, 3])
    def test_outcome_and_each_treatment_are_read_by_geoex_dataprep(self, monkeypatch, K):
        calls = []
        real = SETUP.geoex_dataprep

        def spy(df, unitid, time, column, **kw):
            calls.append(column)
            return real(df, unitid, time, column, **kw)

        monkeypatch.setattr(SETUP, "geoex_dataprep", spy)
        prepare_ctsc_inputs(_panel(K=K), treatment_vars=_tv(K), **COLS)
        assert calls == ["y", *_tv(K)]

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
        dict(),                                            # int ids, shuffled, K=2
        dict(K=1, names="str"),
        dict(K=3, shuffle=False, t0=-4),
        dict(names="str", times="date"),                   # Timestamp periods
        dict(times="str", K=1),                            # string periods
        dict(n_units=3, T=2, K=1),                         # the smallest panel CTSC takes
        dict(n_units=40, T=15, seed=7),                    # 40 units
        dict(n_units=40, T=12, names="str", times="date", seed=3),
    ])
    @pytest.mark.parametrize("pop", [None, "pop"])
    def test_inputs_match_the_old_pivot(self, kw, pop):
        df = _panel(**kw)
        K = kw.get("K", 2)
        args = dict(treatment_vars=_tv(K), population_col=pop, **COLS)
        _assert_same(prepare_ctsc_inputs(df, **args), _reference(df, **args))

    def test_rows_are_units_and_columns_are_periods(self):
        df = _panel(n_units=5, T=7, K=2, names="str", seed=2)
        inputs = prepare_ctsc_inputs(df, treatment_vars=_tv(2), **COLS)
        cell = df.set_index(["unit", "time"])
        for i, u in enumerate(inputs.unit_names):
            for t, label in enumerate(inputs.time_labels):
                assert inputs.Y[i, t] == cell.loc[(u, label), "y"]
                for k in range(2):
                    assert inputs.D[i, t, k] == cell.loc[(u, label), f"d{k}"]
        assert inputs.unit_names == sorted(inputs.unit_names)
        assert list(inputs.time_labels) == sorted(inputs.time_labels)

    def test_population_weights_follow_the_unit_order(self):
        df = _panel(n_units=5, names="str", seed=4)
        inputs = prepare_ctsc_inputs(df, treatment_vars=_tv(2), population_col="pop", **COLS)
        pop = df.groupby("unit")["pop"].first()
        want = np.array([pop[u] for u in inputs.unit_names]) / pop.sum()
        np.testing.assert_allclose(inputs.population_weights, want, rtol=0, atol=1e-15)

    def test_integer_treatments_are_read_as_floats(self):
        df = _panel(K=2)
        df["d1"] = df["d1"].astype(int)
        args = dict(treatment_vars=_tv(2), **COLS)
        _assert_same(prepare_ctsc_inputs(df, **args), _reference(df, **args))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_ctsc_inputs(df, treatment_vars=_tv(2), population_col="pop", **COLS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        from mlsynth import CTSC
        res = CTSC(dict(df=_panel(n_units=6, T=20), treat="d0", treatment_vars=_tv(2),
                        display_graphs=False, inference=False, **COLS)).fit()
        assert res.average_effect.shape == (2,)
        assert np.isfinite(res.average_effect).all()


# --------------------------------------------------------------------------- #
# refusals now come from balance and geoex_dataprep
# --------------------------------------------------------------------------- #
def _set(df, column, value, unit=2, period=3):
    df = df.copy()
    df.loc[(df.unit == unit) & (df.time == 1990 + period), column] = value
    return df


class TestRefusals:
    def test_a_missing_outcome_is_refused(self):
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_ctsc_inputs(_set(_panel(), "y", np.nan), treatment_vars=_tv(2), **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_ctsc_inputs(_set(_panel(), "y", np.inf), treatment_vars=_tv(2), **COLS)

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("column", ["d0", "d1"])
    def test_an_incomplete_treatment_is_refused_under_its_name(self, value, column):
        df = _set(_panel(), column, value)
        with pytest.raises(MlsynthDataError, match=f"Treatment variable '{column}'") as info:
            prepare_ctsc_inputs(df, treatment_vars=_tv(2), **COLS)
        # The refusal is reported with dataprep's diagnosis attached, not replaced.
        assert isinstance(info.value.__cause__, MlsynthDataError)
        assert "Missing outcome values" in str(info.value)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 4) & (df.time == 1995))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_ctsc_inputs(df, treatment_vars=_tv(2), **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_ctsc_inputs(pd.concat([df, df.iloc[[7]]]), treatment_vars=_tv(2), **COLS)

    def test_a_missing_key_is_refused(self):
        df = _panel(shuffle=False)
        df["time"] = df["time"].astype(float)
        df.iloc[5, df.columns.get_loc("time")] = np.nan
        with pytest.raises(MlsynthDataError, match="key columns"):
            prepare_ctsc_inputs(df, treatment_vars=_tv(2), **COLS)

    def test_the_estimator_reports_a_duplicate_as_a_data_error(self):
        from mlsynth import CTSC
        df = _panel(shuffle=False)
        df = pd.concat([df, df.iloc[[3]]])
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            CTSC(dict(df=df, treat="d0", treatment_vars=_tv(2), display_graphs=False,
                      inference=False, **COLS)).fit()


# --------------------------------------------------------------------------- #
# CTSC's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_no_treatment_variable(self):
        with pytest.raises(MlsynthConfigError, match="at least one treatment"):
            prepare_ctsc_inputs(_panel(), treatment_vars=[], **COLS)

    @pytest.mark.parametrize("missing", ["y", "unit", "time", "d1"])
    def test_an_absent_column_is_named(self, missing):
        df = _panel().drop(columns=missing)
        with pytest.raises(MlsynthDataError, match=f"Required column '{missing}' missing"):
            prepare_ctsc_inputs(df, treatment_vars=_tv(2), **COLS)

    @pytest.mark.parametrize("n_units", [1, 2])
    def test_fewer_than_three_units(self, n_units):
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_ctsc_inputs(_panel(n_units=n_units), treatment_vars=_tv(2), **COLS)

    def test_three_units_are_enough(self):
        inputs = prepare_ctsc_inputs(_panel(n_units=3), treatment_vars=_tv(2), **COLS)
        assert inputs.n == 3

    def test_absent_population_column(self):
        with pytest.raises(MlsynthDataError, match="Population column 'nope'"):
            prepare_ctsc_inputs(_panel(), treatment_vars=_tv(2), population_col="nope", **COLS)

    @pytest.mark.parametrize("bad", ["negative", "zero"])
    def test_invalid_population_weights(self, bad):
        df = _panel()
        df["pop"] = -1.0 if bad == "negative" else 0.0
        with pytest.raises(MlsynthDataError, match="non-negative and sum > 0"):
            prepare_ctsc_inputs(df, treatment_vars=_tv(2), population_col="pop", **COLS)

    def test_uniform_weights_by_default(self):
        inputs = prepare_ctsc_inputs(_panel(n_units=7), treatment_vars=_tv(2), **COLS)
        np.testing.assert_array_equal(inputs.population_weights, np.full(7, 1 / 7))


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
DEFECTS = ["none", "none", "none", "nan_y", "inf_y", "nan_d", "inf_d", "gap", "dup"]


@st.composite
def cases(draw):
    n_units = draw(st.integers(3, 8))
    T = draw(st.integers(2, 8))
    K = draw(st.integers(1, 3))
    names = draw(st.sampled_from(["int", "str"]))
    times = draw(st.sampled_from(["int", "date", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, K=K, seed=seed, names=names, times=times)
    defect = draw(st.sampled_from(DEFECTS))
    i = draw(st.integers(0, len(df) - 1))
    k = draw(st.integers(0, K - 1))
    if defect == "nan_y":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf_y":
        df.iloc[i, df.columns.get_loc("y")] = -np.inf
    elif defect == "nan_d":
        df.iloc[i, df.columns.get_loc(f"d{k}")] = np.nan
    elif defect == "inf_d":
        df.iloc[i, df.columns.get_loc(f"d{k}")] = np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    pop = draw(st.sampled_from([None, "pop"]))
    return df, K, k, pop, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_geoex_path_matches_the_old_pivot_or_refuses(case):
    df, K, k, pop, defect = case
    args = dict(treatment_vars=_tv(K), population_col=pop, **COLS)
    if defect != "none":
        match = f"Treatment variable 'd{k}'" if defect in ("nan_d", "inf_d") else None
        with pytest.raises(MlsynthDataError, match=match):
            prepare_ctsc_inputs(df, **args)
        return
    inputs = prepare_ctsc_inputs(df, **args)
    _assert_same(inputs, _reference(df, **args))
    # Shapes agree with the panel, and the weights are a distribution.
    n, T = df["unit"].nunique(), df["time"].nunique()
    assert inputs.Y.shape == (n, T) and inputs.D.shape == (n, T, K)
    assert np.isfinite(inputs.Y).all() and np.isfinite(inputs.D).all()
    assert inputs.population_weights.sum() == pytest.approx(1.0, abs=1e-12)
