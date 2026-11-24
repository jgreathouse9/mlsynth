"""PPSCM ingestion through ``datautils.dataprep`` (#1138, part 1).

PPSCM used to pivot the panel itself and check only the window before the last
adoption for missing outcomes. It now takes the outcome matrix, the time labels
and each cohort's adoption period from ``balance`` and ``dataprep``, so it gets
the same refusals as every other estimator: a missing or non-finite outcome
anywhere in the panel, an absent or duplicate unit-period, a non-binary
treatment, and a treatment that switches off after it starts.

``_reference`` is the old pivot (augsynth's ``format_data_stag`` as ported),
kept as the oracle the new path must match on every panel both accept.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.ppscm_helpers import setup as SETUP
from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs
from mlsynth.utils.ppscm_helpers.structures import PPSCMInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, outcome, treat, unitid, time):
    """The pre-#1138 setup's outcome matrix and adoption bookkeeping."""
    t_vec = np.sort(pd.unique(df[time]))
    units = np.sort(pd.unique(df[unitid]))
    adopt = {}
    for u, g in df.groupby(unitid, observed=True):
        yrs = g.loc[g[treat] == 1, time]
        adopt[u] = yrs.min() if len(yrs) else np.inf
    finite = [a for a in adopt.values() if not (isinstance(a, float) and np.isinf(a))]
    trt = np.array([
        (int(np.where(t_vec == adopt[u])[0][0])
         if not (isinstance(adopt[u], float) and np.isinf(adopt[u])) else np.inf)
        for u in units], dtype=float)
    t_int = max(finite)
    wide = df.pivot(index=unitid, columns=time, values=outcome).reindex(index=units)
    return dict(Xy=wide.to_numpy(dtype=float), trt=trt,
                n_pre=int((t_vec < t_int).sum()), time_labels=t_vec, units=units,
                intervention_time=t_int)


def _panel(n_units=8, T=12, adopt=None, seed=0, names="int", shuffle=True, t0=1990):
    adopt = {0: 6, 1: 6, 2: 8} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            d = int(u in adopt and t >= adopt[u])
            rows.append(dict(unit=u if names == "int" else f"s{u:02d}", time=t0 + t,
                             y=rng.normal() + u + 0.1 * t + 0.5 * d, d=d,
                             x=rng.normal()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, PPSCMInputs)
    assert inputs.Xy.dtype == np.float64
    np.testing.assert_array_equal(inputs.Xy, ref["Xy"])
    np.testing.assert_array_equal(inputs.trt, ref["trt"])
    assert inputs.n_pre == ref["n_pre"]
    np.testing.assert_array_equal(inputs.time_labels, ref["time_labels"])
    assert np.asarray(inputs.time_labels).dtype == ref["time_labels"].dtype
    np.testing.assert_array_equal(inputs.units, ref["units"])
    assert inputs.intervention_time == ref["intervention_time"]


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
        prepare_ppscm_inputs(_panel(), **COLS)
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
        dict(),                                         # three cohorts
        dict(adopt={0: 6}),                             # one treated unit
        dict(adopt={0: 5, 1: 5}),                       # one cohort of two
        dict(names="str", adopt={0: 4, 3: 9}),
        dict(shuffle=False, t0=-6, adopt={2: 2, 4: 11}),  # last period adoption
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_ppscm_inputs(df, **COLS), _reference(df, **COLS))

    def test_covariates_are_unchanged(self):
        df = _panel()
        inputs = prepare_ppscm_inputs(df, **COLS, covariates=["x"])
        first = df[df.time < 1996].groupby("unit")["x"].mean().reindex(inputs.units)
        np.testing.assert_array_equal(inputs.Z[:, 0], first.to_numpy())
        assert inputs.cov_names == ("x",)

    def test_fit_smoke(self):
        from mlsynth import PPSCM
        res = PPSCM(dict(df=_panel(), **COLS, display_graphs=False)).fit()
        assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_post_period_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 5) & (df.time == 2000), "y"] = np.nan    # after the last adoption
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_ppscm_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 1992), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_ppscm_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_ppscm_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_ppscm_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time >= 1999), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_ppscm_inputs(df, **COLS)

    def test_a_non_binary_treatment_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[df.d == 1, "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_ppscm_inputs(df, **COLS)


# --------------------------------------------------------------------------- #
# PPSCM's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_unknown_covariate(self):
        with pytest.raises(MlsynthDataError, match="Covariate column"):
            prepare_ppscm_inputs(_panel(), **COLS, covariates=["nope"])

    def test_no_treated_unit(self):
        with pytest.raises(MlsynthDataError):
            prepare_ppscm_inputs(_panel(adopt={}), **COLS)

    def test_every_unit_treated(self):
        df = _panel(n_units=3, adopt={0: 5, 1: 6, 2: 7})
        with pytest.raises(MlsynthDataError):
            prepare_ppscm_inputs(df, **COLS)

    def test_one_pre_period_before_the_last_adoption_is_too_few(self):
        with pytest.raises(MlsynthDataError):
            prepare_ppscm_inputs(_panel(adopt={0: 1}), **COLS)

    def test_missing_covariate_values(self):
        df = _panel(shuffle=False)
        df.loc[df.unit == 3, "x"] = np.nan
        with pytest.raises(MlsynthDataError, match="covariates have missing values"):
            prepare_ppscm_inputs(df, **COLS, covariates=["x"])


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(3, 9))
    T = draw(st.integers(4, 10))
    n_treated = draw(st.integers(1, n_units - 1))
    adopt = {u: draw(st.integers(2, T - 1)) for u in range(n_treated)}
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, adopt=adopt, seed=seed, names=names)
    defect = draw(st.sampled_from(["none", "none", "nan", "inf", "gap", "dup", "off"]))
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
            name = u if names == "int" else f"s{u:02d}"
            df.loc[(df.unit == name) & (df.time == 1990 + T - 1), "d"] = 0
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_ppscm_inputs(df, **COLS)
        return
    inputs = prepare_ppscm_inputs(df, **COLS)
    _assert_same(inputs, _reference(df, **COLS))
    # Every treated unit's adoption index is a period after the first, and
    # n_pre is the last adoption.
    finite = inputs.trt[np.isfinite(inputs.trt)]
    assert finite.min() >= 1 and inputs.n_pre == int(finite.max())
