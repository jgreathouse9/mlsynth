"""CPDA ingestion through ``datautils.dataprep`` (#1138, part 1).

CPDA used to pivot the outcome and every covariate itself and check the pivots
for NaN cells only. It now takes the outcome matrix, the time labels, the donor
order and the pre-period count from ``balance`` and ``dataprep``, and reads each
covariate's (period by unit) matrix through ``dataprep`` as well, so it gets the
same refusals as every other estimator: a missing or non-finite outcome or
covariate anywhere in the panel, an absent or duplicate unit-period, a
non-binary treatment, and a treatment that switches off after it starts.

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

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.cpda_helpers import setup as SETUP
from mlsynth.utils.cpda_helpers.setup import prepare_cpda_inputs
from mlsynth.utils.cpda_helpers.structures import CPDAInputs
from mlsynth.utils.datautils import PanelKeys

COLS = dict(unitid="unit", time="time", outcome="y", treat="d")
COVS = ["x0", "x1"]


def _reference(df, unitid, time, outcome, treat, covariates):
    """The pre-#1138 setup's arrays and labels."""
    treated_rows = df[df[treat] == 1]
    treated = pd.unique(treated_rows[unitid])[0]
    t_int = treated_rows.loc[treated_rows[unitid] == treated, time].min()
    times = np.sort(pd.unique(df[time]))
    wide = df.pivot(index=time, columns=unitid, values=outcome).reindex(times)
    donors = [u for u in wide.columns if u != treated]

    def cube(order):
        return np.stack([
            df.pivot(index=time, columns=unitid, values=c).reindex(times)[order]
            .to_numpy(dtype=float) for c in covariates], axis=2)

    return dict(
        y=wide[treated].to_numpy(dtype=float), x=cube([treated])[:, 0, :],
        Yco=wide[donors].to_numpy(dtype=float), Xco=cube(donors),
        T0=int(np.sum(times < t_int)), treated_label=treated, donors=donors,
        times=times, intervention_time=t_int,
    )


def _panel(n_donors=6, T=10, T0=6, treated=0, seed=0, names="int",
           shuffle=True, t0=2000, k=2):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_donors + 1):
        for t in range(T):
            row = dict(unit=u if names == "int" else f"s{u:02d}", time=t0 + t,
                       y=rng.normal() + u + 0.1 * t,
                       d=int(u == treated and t >= T0))
            for j in range(k):
                row[f"x{j}"] = rng.normal() + j
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _name(u, names="int"):
    return u if names == "int" else f"s{u:02d}"


def _assert_same(inputs, ref):
    assert isinstance(inputs, CPDAInputs)
    for field in ("y", "x", "Yco", "Xco"):
        got = getattr(inputs, field)
        assert got.dtype == np.float64, field
        assert got.shape == ref[field].shape, field
        np.testing.assert_array_equal(got, ref[field], err_msg=field)
    assert inputs.T0 == ref["T0"]
    assert inputs.treated_label == ref["treated_label"]
    np.testing.assert_array_equal(inputs.unit_index.labels, np.asarray(ref["donors"]))
    np.testing.assert_array_equal(inputs.time_index.labels, ref["times"])
    assert np.asarray(inputs.time_index.labels).dtype == ref["times"].dtype
    assert inputs.metadata["intervention_time"] == ref["intervention_time"]


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_every_matrix_is_read_through_dataprep_with_balances_keys(self, monkeypatch):
        scans, calls = [], []
        real_balance, real_dataprep = SETUP.balance, SETUP.dataprep

        def balance_spy(*a, **k):
            scans.append(real_balance(*a, **k))
            return scans[-1]

        def dataprep_spy(*a, **k):
            calls.append((a[3], k.get("keys")))
            return real_dataprep(*a, **k)

        monkeypatch.setattr(SETUP, "balance", balance_spy)
        monkeypatch.setattr(SETUP, "dataprep", dataprep_spy)
        prepare_cpda_inputs(_panel(), **COLS, covariates=COVS)
        assert len(scans) == 1 and isinstance(scans[0], PanelKeys)
        # The outcome once, then each covariate once, all on the one scan.
        assert [c for c, _ in calls] == ["y", *COVS]
        assert all(keys is scans[0] for _, keys in calls)

    def test_setup_does_not_pivot(self):
        # A runtime patch of DataFrame.pivot cannot test this: dataprep's
        # _wide_pivot falls back to df.pivot on some pandas versions.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called

    def test_the_estimator_scans_the_panel_once(self, monkeypatch):
        import mlsynth.estimators.cpda as EST
        from mlsynth import CPDA
        from mlsynth.utils import datautils

        scans = []
        real = PanelKeys.__init__

        def counting(self, *a, **k):
            scans.append(1)
            real(self, *a, **k)

        monkeypatch.setattr(datautils.PanelKeys, "__init__", counting)
        res = CPDA(dict(df=_panel(T=20, T0=14), **COLS, covariates=COVS,
                        display_graphs=False)).fit()
        assert np.isfinite(res.effects.att)
        assert len(scans) == 1, "the panel's keys were scanned more than once"
        assert not hasattr(EST, "balance")


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),
        dict(names="str"),
        dict(shuffle=False),
        dict(treated=4),                                 # treated unit not first
        dict(names="str", treated=3, seed=3),
        dict(n_donors=1),                                # a single donor
        dict(T0=2),                                      # the fewest pre-periods
        dict(T0=9),                                      # adoption in the last period
        dict(t0=-5, k=1),                                # negative times, one covariate
        dict(k=3, seed=7),
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        covs = [f"x{j}" for j in range(kw.get("k", 2))]
        _assert_same(prepare_cpda_inputs(df, **COLS, covariates=covs),
                     _reference(df, **COLS, covariates=covs))

    def test_datetime_times_match_the_old_pivot(self):
        df = _panel()
        df["time"] = pd.to_datetime(df["time"].astype(str) + "-01-01")
        _assert_same(prepare_cpda_inputs(df, **COLS, covariates=COVS),
                     _reference(df, **COLS, covariates=COVS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_cpda_inputs(df, **COLS, covariates=COVS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        from mlsynth import CPDA
        res = CPDA(dict(df=_panel(T=20, T0=14), **COLS, covariates=COVS,
                        display_graphs=False)).fit()
        assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2008), "y"] = np.nan    # post-period
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old NaN check let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 5) & (df.time == 2004))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old setup's pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_cpda_inputs(pd.concat([df, df.iloc[[7]]]), **COLS, covariates=COVS)

    def test_a_non_binary_treatment_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2009), "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup read it from its first treated period.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2008), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_a_missing_or_non_finite_covariate_is_named(self, value):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 4) & (df.time == 2003), "x1"] = value
        with pytest.raises(MlsynthDataError, match="Covariate 'x1'") as info:
            prepare_cpda_inputs(df, **COLS, covariates=COVS)
        # dataprep's count of the bad cells is carried in the message.
        assert "Missing outcome values" in str(info.value)
        assert isinstance(info.value.__cause__, MlsynthDataError)


# --------------------------------------------------------------------------- #
# CPDA's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_a_missing_column_is_named(self):
        with pytest.raises(MlsynthDataError, match="nope"):
            prepare_cpda_inputs(_panel(), **COLS, covariates=["x0", "nope"])

    def test_no_treated_unit(self):
        df = _panel()
        df["d"] = 0
        with pytest.raises(MlsynthDataError, match="No treated rows"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    def test_two_treated_units_in_one_cohort(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time >= 2006), "d"] = 1
        with pytest.raises(MlsynthDataError, match="exactly one treated unit"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    def test_two_treated_units_in_two_cohorts(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time >= 2004), "d"] = 1
        with pytest.raises(MlsynthDataError, match="exactly one treated unit"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)

    def test_one_pre_period_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="at least two pre-treatment periods; found 1"):
            prepare_cpda_inputs(_panel(T0=1), **COLS, covariates=COVS)

    def test_treatment_in_the_first_period(self):
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_cpda_inputs(_panel(T0=0), **COLS, covariates=COVS)

    def test_no_donor_unit(self):
        df = _panel()
        df = df[df.unit == 0]
        with pytest.raises(MlsynthDataError, match="at least one donor unit"):
            prepare_cpda_inputs(df, **COLS, covariates=COVS)


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
DEFECTS = ["none", "none", "none", "nan_y", "inf_y", "nan_x", "inf_x", "gap",
           "dup", "off", "nonbinary"]


@st.composite
def cases(draw):
    n_donors = draw(st.integers(1, 6))
    T = draw(st.integers(3, 9))
    T0 = draw(st.integers(2, T - 1))
    treated = draw(st.integers(0, n_donors))
    names = draw(st.sampled_from(["int", "str"]))
    k = draw(st.integers(1, 3))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_donors=n_donors, T=T, T0=T0, treated=treated, seed=seed,
                names=names, k=k)
    defect = draw(st.sampled_from(DEFECTS))
    i = draw(st.integers(0, len(df) - 1))
    cov = f"x{draw(st.integers(0, k - 1))}"
    if defect == "nan_y":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf_y":
        df.iloc[i, df.columns.get_loc("y")] = -np.inf
    elif defect == "nan_x":
        df.iloc[i, df.columns.get_loc(cov)] = np.nan
    elif defect == "inf_x":
        df.iloc[i, df.columns.get_loc(cov)] = np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "off":
        # A treated period after the first switches off. A zero in the first
        # treated period only moves the adoption, so it is not a defect.
        if T0 == T - 1:
            defect = "none"
        else:
            t = draw(st.integers(T0 + 1, T - 1))
            df.loc[(df.unit == _name(treated, names)) & (df.time == 2000 + t), "d"] = 0
    elif defect == "nonbinary":
        df.loc[(df.unit == _name(treated, names)) & (df.time == 2000 + T - 1), "d"] = 2
    return df, [f"x{j}" for j in range(k)], defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, covs, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_cpda_inputs(df, **COLS, covariates=covs)
        return
    inputs = prepare_cpda_inputs(df, **COLS, covariates=covs)
    _assert_same(inputs, _reference(df, **COLS, covariates=covs))
    # The pre-period is the treated unit's untreated run, and the donors and the
    # treated unit partition the panel's units.
    assert inputs.T0 == int((df.loc[df.unit == inputs.treated_label, "d"] == 0).sum())
    assert inputs.N + 1 == df["unit"].nunique()
    assert inputs.Xco.shape == (inputs.T, inputs.N, len(covs))
