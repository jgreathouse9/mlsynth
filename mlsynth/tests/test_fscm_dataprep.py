"""FSCM ingestion through ``datautils.dataprep`` (#1138, part 1).

FSCM used to pivot the panel itself, read the treated unit and its first
treated period from the rows with ``treat == 1``, and check only for a NaN cell
after the pivot. It now takes the outcome matrix, the time labels, the treated
unit and its adoption period from ``balance`` and ``dataprep``, so it gets the
same refusals as every other estimator: a missing or non-finite outcome, an
absent or duplicate unit-period, a non-binary treatment, and a treatment that
switches off after it starts. Each covariate is read through ``dataprep`` too,
as the outcome of its own call with ``missing_outcomes="keep"`` on the same key
scan: FSCM fills a covariate's gaps and averages it over a per-covariate window,
which ``dataprep``'s covariate block does not do, so a NaN covariate cell has to
reach FSCM's fill while an infinite one is refused under the covariate's name.

``_reference`` is the old setup, kept as the oracle the new path must match on
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

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.fscm_helpers import setup as SETUP
from mlsynth.utils.fscm_helpers.setup import prepare_fscm_inputs
from mlsynth.utils.fscm_helpers.structures import FSCMInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, *, unitid, time, outcome, treat, covariates=None,
               covariate_windows=None, match_periods=None):
    """The pre-#1138 setup, verbatim apart from the return type."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError("no treated rows")
    treated_units = pd.unique(treated_rows[unitid])
    if len(treated_units) != 1:
        raise MlsynthDataError("not one treated unit")
    treated_unit = treated_units[0]
    intervention_time = treated_rows.loc[treated_rows[unitid] == treated_unit, time].min()
    windows = dict(covariate_windows or {})

    times = np.sort(pd.unique(df[time]))
    time_index = IndexSet.from_labels(times)
    wide = df.pivot(index=time, columns=unitid, values=outcome).reindex(times)
    if wide.isna().any().any():
        raise MlsynthDataError("incomplete")
    donors = [u for u in wide.columns if u != treated_unit]
    y = wide[treated_unit].to_numpy(dtype=float)
    Y = wide[donors].to_numpy(dtype=float)
    T0 = int(np.sum(times < intervention_time))
    if T0 < 4:
        raise MlsynthDataError("too few pre-periods")

    pre_mask = times < intervention_time
    cov_treated = np.empty(0)
    cov_donors = np.empty((len(donors), 0))
    cov_names = list(covariates or [])
    if cov_names:
        treated_vals, donor_vals = [], []
        for c in cov_names:
            cwide = df.pivot(index=time, columns=unitid, values=c).reindex(times)
            cwide = cwide.ffill().bfill()
            if cwide[[treated_unit, *donors]].isna().any().any():
                raise MlsynthDataError("covariate entirely missing")
            if c in windows:
                start, end = windows[c]
                mask = (times >= start) & (times <= end)
                if not mask.any():
                    raise MlsynthDataError("window selects nothing")
            else:
                mask = pre_mask
            treated_vals.append(float(cwide.loc[mask, treated_unit].mean()))
            donor_vals.append(cwide.loc[mask, donors].mean().to_numpy(dtype=float))
        cov_treated = np.asarray(treated_vals)
        cov_donors = np.column_stack(donor_vals)

    match_idx = np.empty(0, dtype=int)
    match_labels = []
    if match_periods:
        resolved = []
        for p in match_periods:
            if p not in time_index.label_to_idx:
                raise MlsynthDataError("match period not found")
            ti = int(time_index.label_to_idx[p])
            if ti >= T0:
                raise MlsynthDataError("match period not pre-treatment")
            resolved.append(ti)
        match_idx = np.array(sorted(set(resolved)), dtype=int)
        match_labels = list(match_periods)
    return dict(units=np.asarray(donors), times=times, y=y, Y=Y, T0=T0,
                treated=treated_unit, cov_treated=cov_treated, cov_donors=cov_donors,
                cov_names=cov_names, match_idx=match_idx, match_periods=match_labels,
                intervention_time=intervention_time)


def _panel(n_units=6, T=12, t_treat=7, treated=0, seed=0, names="int",
           shuffle=True, t0=1990, x_gaps=False):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            d = int(u == treated and t >= t_treat)
            x = rng.normal() + u
            if x_gaps and t % 3 == u % 3:
                x = np.nan
            rows.append(dict(unit=u if names == "int" else f"s{u:02d}", time=t0 + t,
                             y=rng.normal() + u + 0.1 * t - 0.5 * d, d=d, x=x,
                             z=rng.normal()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(inputs, ref):
    assert isinstance(inputs, FSCMInputs)
    np.testing.assert_array_equal(inputs.unit_index.labels, ref["units"])
    np.testing.assert_array_equal(inputs.time_index.labels, ref["times"])
    assert inputs.time_index.labels.dtype == ref["times"].dtype
    assert inputs.time_index.label_to_idx == {t: i for i, t in enumerate(ref["times"])}
    assert inputs.y.dtype == inputs.Y.dtype == np.float64
    np.testing.assert_array_equal(inputs.y, ref["y"])
    np.testing.assert_array_equal(inputs.Y, ref["Y"])
    assert inputs.T0 == ref["T0"]
    assert inputs.treated_label == ref["treated"]
    np.testing.assert_array_equal(inputs.cov_treated, ref["cov_treated"])
    np.testing.assert_array_equal(inputs.cov_donors, ref["cov_donors"])
    assert inputs.cov_donors.shape == ref["cov_donors"].shape
    assert inputs.covariate_names == ref["cov_names"]
    np.testing.assert_array_equal(inputs.match_idx, ref["match_idx"])
    assert inputs.match_periods == ref["match_periods"]
    assert inputs.metadata["outcome"] == "y"
    assert inputs.metadata["intervention_time"] == ref["intervention_time"]
    assert type(inputs.metadata["intervention_time"]) is type(ref["intervention_time"])


def _both(df, **kw):
    return prepare_fscm_inputs(df, **COLS, **kw), _reference(df, **COLS, **kw)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_setup_calls_balance_and_dataprep_once_on_one_scan(self, monkeypatch):
        calls = []
        real_balance, real_dataprep = SETUP.balance, SETUP.dataprep

        def balance_spy(*a, **k):
            keys = real_balance(*a, **k)
            calls.append(("balance", keys))
            return keys

        def dataprep_spy(*a, **k):
            calls.append(("dataprep", k.get("keys")))
            return real_dataprep(*a, **k)

        monkeypatch.setattr(SETUP, "balance", balance_spy)
        monkeypatch.setattr(SETUP, "dataprep", dataprep_spy)
        prepare_fscm_inputs(_panel(), **COLS)
        assert [name for name, _ in calls] == ["balance", "dataprep"]
        assert calls[1][1] is calls[0][1]

    def test_the_estimator_reaches_dataprep(self, monkeypatch):
        from mlsynth import FSCM
        calls = []
        real = SETUP.dataprep

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "dataprep", spy)
        FSCM(dict(df=_panel(), **COLS, display_graphs=False)).fit()
        assert len(calls) == 1

    def test_each_covariate_is_one_dataprep_call_that_keeps_gaps(self, monkeypatch):
        calls = []
        real_balance, real_dataprep = SETUP.balance, SETUP.dataprep
        scans = []

        def balance_spy(*a, **k):
            scans.append(real_balance(*a, **k))
            return scans[-1]

        def dataprep_spy(*a, **k):
            calls.append((a, k))
            return real_dataprep(*a, **k)

        monkeypatch.setattr(SETUP, "balance", balance_spy)
        monkeypatch.setattr(SETUP, "dataprep", dataprep_spy)
        df = _panel(x_gaps=True)
        prepare_fscm_inputs(df, **COLS, covariates=["x", "z"])
        assert len(scans) == 1 and len(calls) == 3
        (a0, k0), *cov_calls = calls
        assert a0[3] == "y" and k0.get("missing_outcomes", "refuse") == "refuse"
        assert [a[3] for a, _ in cov_calls] == ["x", "z"]
        for a, k in cov_calls:
            assert a[0] is df and a[1:3] == ("unit", "time") and a[4] == "d"
            assert k["missing_outcomes"] == "keep"
            assert k["keys"] is scans[0]
            assert set(k) == {"keys", "missing_outcomes"}

    def test_setup_reads_nothing_private_from_datautils(self):
        tree = ast.parse(inspect.getsource(SETUP))
        imported = {(node.module, alias.name) for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) for alias in node.names}
        private = {(m, n) for m, n in imported
                   if m and m.split(".")[-1] == "datautils" and n.startswith("_")}
        assert not private, private
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {(node.value.id, node.attr) for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)}
        assert "_wide_pivot" not in names
        assert not {a for a in attrs if a[0] == "datautils" and a[1].startswith("_")}, attrs

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
        dict(names="str", treated=3),
        dict(shuffle=False, t0=-6, treated=5, t_treat=11),   # treated in the last period only
        dict(n_units=2, t_treat=4),                          # one donor, four pre-periods
        dict(T=20, t_treat=15, treated=2, seed=4),
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        _assert_same(*_both(_panel(**kw)))

    def test_datetime_periods(self):
        df = _panel()
        df["time"] = pd.to_datetime(df["time"].astype(str))
        _assert_same(*_both(df))

    def test_boolean_treatment(self):
        df = _panel()
        df["d"] = df["d"].astype(bool)
        _assert_same(*_both(df))

    def test_integer_outcome(self):
        df = _panel()
        df["y"] = (df["y"] * 100).round().astype(int)
        _assert_same(*_both(df))

    def test_covariates_windows_and_match_periods(self):
        df = _panel(x_gaps=True, names="str")
        _assert_same(*_both(df, covariates=["x", "z"],
                            covariate_windows={"x": (1991, 1994)},
                            match_periods=[1995, 1991, 1995]))

    def test_covariate_window_reaching_past_the_intervention(self):
        _assert_same(*_both(_panel(x_gaps=True), covariates=["x"],
                            covariate_windows={"x": (1990, 2001)}))

    def test_fit_smoke(self):
        from mlsynth import FSCM
        res = FSCM(dict(df=_panel(T=16, t_treat=10), **COLS, display_graphs=False,
                        covariates=["x"], match_periods=[1992])).fit()
        assert np.isfinite(res.att)
        assert res.counterfactual.shape == (16,)


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2000), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_fscm_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old pivot checked only for NaN, so an infinite donor outcome
        # reached the solver.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 1992), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_fscm_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 2) & (df.time == 1995))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_fscm_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_fscm_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup read the first treated period and fitted the rest.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2000), "d"] = 0
        _reference(df, **COLS)                     # the old setup accepted it
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_fscm_inputs(df, **COLS)

    def test_a_non_binary_treatment_is_refused(self):
        # The old setup read only treat == 1, so a 2 was an untreated period.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 1991), "d"] = 2
        _reference(df, **COLS)
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_fscm_inputs(df, **COLS)

    def test_no_treated_unit_is_refused(self):
        df = _panel()
        df["d"] = 0
        with pytest.raises(MlsynthDataError, match="No treated units"):
            prepare_fscm_inputs(df, **COLS)

    def test_no_donor_is_refused(self):
        with pytest.raises(MlsynthDataError, match="No donor units"):
            prepare_fscm_inputs(_panel(n_units=1), **COLS)

    def test_treatment_at_the_first_period_is_refused(self):
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_fscm_inputs(_panel(t_treat=0), **COLS)

    def test_the_estimator_reports_the_refusal(self):
        from mlsynth import FSCM
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 1992), "y"] = -np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            FSCM(dict(df=df, **COLS, display_graphs=False)).fit()


# --------------------------------------------------------------------------- #
# FSCM's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_two_treated_units(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time >= 1997), "d"] = 1
        with pytest.raises(MlsynthDataError, match="exactly one treated unit; found 2"):
            prepare_fscm_inputs(df, **COLS)

    def test_two_cohorts_of_treated_units(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time >= 1999), "d"] = 1
        df.loc[(df.unit == 4) & (df.time >= 1999), "d"] = 1
        with pytest.raises(MlsynthDataError, match="exactly one treated unit; found 3"):
            prepare_fscm_inputs(df, **COLS)

    def test_three_pre_periods_are_too_few(self):
        with pytest.raises(MlsynthDataError, match="at least four pre-treatment"):
            prepare_fscm_inputs(_panel(t_treat=3), **COLS)

    def test_four_pre_periods_are_enough(self):
        assert prepare_fscm_inputs(_panel(t_treat=4), **COLS).T0 == 4

    def test_a_covariate_missing_for_a_unit(self):
        df = _panel(shuffle=False)
        df.loc[df.unit == 3, "x"] = np.nan
        with pytest.raises(MlsynthDataError, match="entirely missing"):
            prepare_fscm_inputs(df, **COLS, covariates=["x"])

    def test_an_infinite_covariate_is_refused_under_its_name(self):
        # The old pivot averaged the infinity into the predictor.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 1993), "z"] = np.inf
        _reference(df, **COLS, covariates=["x", "z"])      # the old setup accepted it
        with pytest.raises(MlsynthDataError, match="Covariate 'z'") as err:
            prepare_fscm_inputs(df, **COLS, covariates=["x", "z"])
        assert "infinite" in str(err.value)
        assert isinstance(err.value.__cause__, MlsynthDataError)

    def test_the_estimator_reports_an_infinite_covariate(self):
        # The old path refused it downstream, as "X1 contains NaN or infinite
        # values", without naming the covariate.
        from mlsynth import FSCM
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 1991), "x"] = -np.inf
        with pytest.raises(MlsynthDataError, match="Covariate 'x'"):
            FSCM(dict(df=df, **COLS, display_graphs=False, covariates=["x"])).fit()

    def test_a_covariate_gap_at_every_cell_but_one_is_filled(self):
        # Only one observed cell per unit: the fill spreads it to every period.
        df = _panel(shuffle=False)
        df.loc[df.time != 1994, "x"] = np.nan
        inputs, ref = _both(df, covariates=["x"])
        _assert_same(inputs, ref)
        expect = df[df.time == 1994].set_index("unit")["x"]
        np.testing.assert_allclose(inputs.cov_donors[:, 0],
                                   expect.loc[inputs.unit_index.labels].to_numpy(),
                                   rtol=1e-14)

    def test_a_window_selecting_no_period(self):
        with pytest.raises(MlsynthDataError, match="selects no periods"):
            prepare_fscm_inputs(_panel(), **COLS, covariates=["x"],
                                covariate_windows={"x": (3000, 3001)})

    def test_a_match_period_not_in_the_panel(self):
        with pytest.raises(MlsynthDataError, match="not found"):
            prepare_fscm_inputs(_panel(), **COLS, match_periods=[1800])

    def test_a_match_period_after_the_intervention(self):
        with pytest.raises(MlsynthDataError, match="not pre-treatment"):
            prepare_fscm_inputs(_panel(), **COLS, match_periods=[1997])

    def test_the_last_pre_period_can_be_matched(self):
        inputs = prepare_fscm_inputs(_panel(), **COLS, match_periods=[1996])
        assert inputs.match_idx.tolist() == [6]


class TestDeriveTreatment:
    """``derive_treatment`` stays exported; the setup no longer calls it."""

    def test_it_agrees_with_the_setup(self):
        df = _panel(names="str", treated=2)
        inputs = prepare_fscm_inputs(df, **COLS)
        assert SETUP.derive_treatment(df, "unit", "time", "d") == (
            inputs.treated_label, inputs.metadata["intervention_time"])

    def test_no_treated_row(self):
        df = _panel()
        df["d"] = 0
        with pytest.raises(MlsynthDataError, match="No treated rows"):
            SETUP.derive_treatment(df, "unit", "time", "d")

    def test_two_treated_units(self):
        df = _panel(shuffle=False)
        df.loc[df.unit == 1, "d"] = 1
        with pytest.raises(MlsynthDataError, match="found 2"):
            SETUP.derive_treatment(df, "unit", "time", "d")


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
DEFECTS = ["none", "none", "none", "nan", "inf", "gap", "dup", "off", "two", "nonbinary",
           "xinf"]


@st.composite
def cases(draw):
    n_units = draw(st.integers(2, 7))
    T = draw(st.integers(6, 12))
    t_treat = draw(st.integers(4, T - 1))
    treated = draw(st.integers(0, n_units - 1))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, t_treat=t_treat, treated=treated, seed=seed,
                names=names, x_gaps=draw(st.booleans()))
    kw = {}
    if draw(st.booleans()):
        kw["covariates"] = draw(st.sampled_from([["x"], ["z"], ["x", "z"]]))
        if draw(st.booleans()):
            a = draw(st.integers(0, T - 1))
            b = draw(st.integers(a, T - 1))
            kw["covariate_windows"] = {kw["covariates"][0]: (1990 + a, 1990 + b)}
    if draw(st.booleans()):
        kw["match_periods"] = draw(st.lists(st.integers(1990, 1990 + t_treat - 1),
                                            min_size=1, max_size=3))
    defect = draw(st.sampled_from(DEFECTS))
    i = draw(st.integers(0, len(df) - 1))
    name = treated if names == "int" else f"s{treated:02d}"
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = draw(st.sampled_from([np.inf, -np.inf]))
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "off":
        df.loc[(df.unit == name) & (df.time == 1990 + T - 1), "d"] = 0
    elif defect == "two":
        if n_units < 3:
            defect = "none"
        else:
            other = (treated + 1) % n_units
            other = other if names == "int" else f"s{other:02d}"
            df.loc[(df.unit == other) & (df.time >= 1990 + t_treat), "d"] = 1
    elif defect == "nonbinary":
        df.loc[(df.unit == name) & (df.time == 1990 + t_treat), "d"] = 2
    elif defect == "xinf":
        df.iloc[i, df.columns.get_loc("x")] = draw(st.sampled_from([np.inf, -np.inf]))
        kw["covariates"] = [*kw.get("covariates", []), "x"] if "x" not in kw.get(
            "covariates", []) else kw["covariates"]
    return df, kw, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, kw, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError) as err:
            prepare_fscm_inputs(df, **COLS, **kw)
        if defect == "xinf":
            assert "Covariate 'x'" in str(err.value)
        return
    try:
        ref = _reference(df, **COLS, **kw)
    except MlsynthDataError:
        # The old setup's own refusals (a covariate gap no fill reaches, a
        # window or match period outside the panel) still stand.
        with pytest.raises(MlsynthDataError):
            prepare_fscm_inputs(df, **COLS, **kw)
        return
    inputs = prepare_fscm_inputs(df, **COLS, **kw)
    _assert_same(inputs, ref)
    # The treated unit's adoption is the first period of the post block, and
    # every donor is a unit of the panel other than the treated one.
    assert inputs.time_index.labels[inputs.T0] == inputs.metadata["intervention_time"]
    assert inputs.treated_label not in set(inputs.unit_index.labels.tolist())
    assert inputs.Y.shape == (len(inputs.time_index.labels), len(inputs.unit_index.labels))
