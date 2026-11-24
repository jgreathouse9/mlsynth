"""MC-NNM ingestion through ``datautils.dataprep`` (#1138, part 1).

MC-NNM used to pivot the long panel itself. It now takes the outcome matrix,
the time labels and the adoption times from ``dataprep``, so it gets the same
refusals as every other estimator: a missing or non-finite outcome cell, an
unbalanced panel, and a treatment that switches off after it starts. The last
one is a change: the old pivot built its mask cell by cell and accepted a
treatment that stops, while the docs describe one-time and staggered adoption
only (Athey, Bayati, Doudchenko, Imbens and Khosravi, "Matrix Completion
Methods for Causal Panel Data Models", JASA 116 (2021), section 2).

``_reference`` below is the old pivot, kept as the oracle the new path must
match on every panel both accept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import MCNNM
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.mcnnm_helpers import setup as SETUP
from mlsynth.utils.mcnnm_helpers.setup import prepare_mcnnm_inputs
from mlsynth.utils.mcnnm_helpers.structures import MCNNMInputs

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
    return dict(Y=Y, mask=(D == 0).astype(float), D=D,
                treated_idx=np.where(D.max(axis=1) == 1)[0], T0=T0,
                unit_names=list(unit_names), time_labels=time_labels)


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
                             y=rng.normal() + u + 0.2 * t + 1.5 * d, d=d))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, MCNNMInputs)
    for f in ("Y", "mask", "D", "treated_idx"):
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
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "dataprep", spy)
        prepare_mcnnm_inputs(df=_panel(), **COLS)
        assert len(calls) == 1

    def test_setup_does_not_pivot(self, monkeypatch):
        def no_pivot(*a, **k):  # pragma: no cover - fails the test if reached
            raise AssertionError("MC-NNM setup pivoted the panel itself")

        df = _panel()
        monkeypatch.setattr(pd.DataFrame, "pivot", no_pivot)
        monkeypatch.setattr(pd.DataFrame, "pivot_table", no_pivot)
        # dataprep builds its matrices without DataFrame.pivot, so any call
        # here came from the estimator.
        prepare_mcnnm_inputs(df=df, **COLS)


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
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_mcnnm_inputs(df=df, **COLS), _reference(df, **COLS))

    def test_fit_smoke(self):
        res = MCNNM(dict(df=_panel(adopt={0: 6, 1: 8}), display_graphs=False,
                         **COLS)).fit()
        assert np.isfinite(res.effects.att)

    def test_extra_columns_are_ignored(self):
        df = _panel().assign(other=1.0, region="r")
        _assert_same(prepare_mcnnm_inputs(df=df, **COLS), _reference(df, **COLS))


# --------------------------------------------------------------------------- #
# refusals now come from dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_treatment_that_switches_off_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time >= 2009), "d"] = 0   # on at 2006, off at 2009
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_mcnnm_inputs(df=df, **COLS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, adopt={0: 6, 1: 6})
        df.loc[(df.unit == 1) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            MCNNM(dict(df=df, display_graphs=False, **COLS)).fit()

    def test_a_missing_outcome_cell_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2004), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            MCNNM(dict(df=df, display_graphs=False, **COLS)).fit()

    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_mcnnm_inputs(df=df, **COLS)

    def test_an_unbalanced_panel_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 4) & (df.time == 2003))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            MCNNM(dict(df=df, display_graphs=False, **COLS)).fit()

    def test_a_duplicate_cell_is_refused(self):
        df = _panel(shuffle=False)
        df = pd.concat([df, df.iloc[[5]]])
        with pytest.raises(MlsynthDataError):
            prepare_mcnnm_inputs(df=df, **COLS)

    def test_a_non_binary_treatment_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[df.d == 1, "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_mcnnm_inputs(df=df, **COLS)


# --------------------------------------------------------------------------- #
# MC-NNM's own requirements, kept on top of dataprep
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_missing_column(self):
        with pytest.raises(MlsynthDataError, match="missing"):
            prepare_mcnnm_inputs(df=_panel().drop(columns="d"), **COLS)

    def test_two_units_are_too_few(self):
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_mcnnm_inputs(df=_panel(n_units=2), **COLS)

    def test_one_control_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="2 control units"):
            prepare_mcnnm_inputs(df=_panel(n_units=3, adopt={0: 5, 1: 5}), **COLS)

    def test_treatment_at_the_first_period_is_refused(self):
        with pytest.raises(MlsynthDataError):
            prepare_mcnnm_inputs(df=_panel(adopt={0: 0}), **COLS)

    def test_no_treated_cell_is_refused(self):
        with pytest.raises(MlsynthDataError):
            prepare_mcnnm_inputs(df=_panel(adopt={}), **COLS)


# --------------------------------------------------------------------------- #
# properties over the domain, invalid paths included
# --------------------------------------------------------------------------- #
@st.composite
def panels(draw):
    n_units = draw(st.integers(3, 9))
    n_periods = draw(st.integers(2, 10))
    n_treated = draw(st.integers(1, max(1, n_units - 2)))
    adopt = {u: draw(st.integers(1, n_periods - 1)) for u in range(n_treated)}
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, n_periods=n_periods, adopt=adopt, seed=seed,
                names=names)
    switch_off = draw(st.booleans())
    if switch_off:
        u = draw(st.sampled_from(sorted(adopt)))
        if adopt[u] < n_periods - 1:
            stop = draw(st.integers(adopt[u] + 1, n_periods - 1))
            name = u if names == "int" else f"s{u:02d}"
            df.loc[(df.unit == name) & (df.time >= 2000 + stop), "d"] = 0
        else:
            switch_off = False
    return df, switch_off


@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=panels())
def test_dataprep_path_matches_the_old_pivot_or_refuses_a_switch_off(case):
    df, switch_off = case
    if switch_off:
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_mcnnm_inputs(df=df, **COLS)
        return
    inputs = prepare_mcnnm_inputs(df=df, **COLS)
    _assert_same(inputs, _reference(df, **COLS))
    # The mask is the complement of D, D is absorbing, and T0 is the first
    # adoption.
    np.testing.assert_array_equal(inputs.mask, 1.0 - inputs.D)
    assert np.all(np.diff(inputs.D, axis=1) >= 0)
    assert inputs.T0 == int(np.argmax(inputs.D.max(axis=0) == 1))
