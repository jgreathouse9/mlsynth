"""DSCAR ingestion through ``datautils.dataprep`` with missing cells kept (#1146).

DSCAR (Zheng and Chen 2024) used to pivot the long panel itself, once for the
outcome, once for the treatment and once per covariate. It now reads the
outcome and treatment through ``dataprep(missing_outcomes="keep")`` on the key
scan ``balance`` took, and each covariate and the lag column through their own
``dataprep`` call on the same scan, so no pivot is left in the setup.

DSCAR matches each period on complete cases: a donor missing its current
outcome, its lag or a covariate at a period is left out of that period's match
(``pipeline._build_period_targets``, mirroring the R reference's "select
records with complete variables"), and a treated unit's missing covariate or
lag is averaged through. The Beijing panels of the paper's application carry
such gaps (5 and 9 missing PM2.5 cells, gaps in ``pm25_lag1``, 56 missing
values in each red-alert covariate). The setup keeps every missing cell as
``NaN`` where the old pivot put it, so the complete-case logic sees the same
inputs and gives the same numbers.

One case the method does not define is a missing outcome of a treated unit
after the intervention: the treated mean at that period is the average of the
treated outcomes, and DSCAR's ATT is the post-period mean of the gap. The old
pivot let the ``NaN`` through and the ATT came out ``NaN``. The setup now
refuses it and names the unit and period.

Behaviour that changes with ``dataprep``: an absent unit-period row, a duplicate
unit-period, an infinite outcome or covariate, a missing or non-binary treatment
and a treatment that switches off are refused with ``MlsynthDataError``. The old
pivot read an absent row or an infinite value as a missing cell, read any
non-zero treatment as treated, and let a duplicate reach pandas as a raw
``ValueError``.

``_reference`` below is the old setup, verbatim, kept as the oracle.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import DSCAR
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.dscar_helpers import setup as SETUP
from mlsynth.utils.dscar_helpers import prepare_dsc_inputs, run_dsc
from mlsynth.utils.dscar_helpers.structures import DSCARInputs

BASE = Path(__file__).resolve().parents[2] / "basedata"
COLS = dict(outcome="y", treat="treat", unitid="unit", time="year")
EXTRA = dict(exog_covariates=["x1", "x2"], lagged_outcome="y_lag1")


def _reference(df, *, outcome, treat, unitid, time, exog_covariates=None,
               lagged_outcome=None):
    """The pre-#1146 setup, verbatim except for its error messages."""
    needed = [outcome, treat, unitid, time]
    if exog_covariates:
        needed.extend(exog_covariates)
    if lagged_outcome:
        needed.append(lagged_outcome)
    df = df.loc[:, list(dict.fromkeys(needed))].copy()
    treated_set = set(df.loc[df[treat] != 0, unitid].unique())
    donor_set = set(df[unitid].unique()) - treated_set
    treated_labels = tuple(sorted(treated_set, key=str))
    donor_labels = tuple(sorted(donor_set, key=str))
    unit_order = [*treated_labels, *donor_labels]
    n_treated = len(treated_labels)
    Y_wide = (df.pivot(index=time, columns=unitid, values=outcome)
                .reindex(columns=unit_order).sort_index())
    time_labels = Y_wide.index.to_numpy()
    T = Y_wide.shape[0]
    Y = Y_wide.to_numpy(dtype=float).T
    treat_wide = (df.pivot(index=time, columns=unitid, values=treat)
                    .reindex(columns=unit_order).sort_index()
                    .to_numpy(dtype=float).T)
    T0 = int(np.where(treat_wide[0] > 0)[0].min())
    Y_lag1 = np.full((Y.shape[0], T), np.nan)
    Y_lag1[:, 1:] = Y[:, :-1]
    if lagged_outcome is not None:
        Y_lag1[:, 0] = (df[df[time] == np.sort(time_labels)[0]]
                        .set_index(unitid)[lagged_outcome]
                        .reindex(unit_order).to_numpy(dtype=float))
    exog_cols = tuple(exog_covariates or ())
    p = len(exog_cols)
    if p > 0:
        X = np.empty((Y.shape[0], T, p), dtype=float)
        for k, col in enumerate(exog_cols):
            X[:, :, k] = (df.pivot(index=time, columns=unitid, values=col)
                            .reindex(columns=unit_order).sort_index()
                            .to_numpy(dtype=float).T)
    else:
        X = np.zeros((Y.shape[0], T, 0), dtype=float)
    return DSCARInputs(
        Y=Y, Y_lag1=Y_lag1, X=X, var_names=exog_cols, y_name=outcome,
        treated_labels=treated_labels, donor_labels=donor_labels,
        time_labels=time_labels, N=Y.shape[0], T=T, T0=T0, T1=T - T0,
        n_treated=n_treated,
    )


def _panel(n_units=8, n_periods=12, treated=(0,), t0=6, seed=0, names="str",
           shuffle=True):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        prev = rng.normal()
        for t in range(n_periods):
            d = int(u in treated and t >= t0)
            x1, x2 = rng.normal(size=2)
            y = 0.5 * x1 - 0.3 * x2 + 0.6 * prev + 0.3 * rng.normal() + 2.0 * d
            rows.append(dict(unit=f"u{u}" if names == "str" else u, year=2000 + t,
                             y=y, x1=x1, x2=x2, y_lag1=prev, treat=d))
            prev = y
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _plant(df, cells, col="y", value=np.nan):
    out = df.copy()
    for u, t in cells:
        out.loc[(out.unit == u) & (out.year == t), col] = value
    return out


def _assert_same(got, ref):
    assert isinstance(got, DSCARInputs)
    for f in ("Y", "Y_lag1", "X"):
        a, b = getattr(got, f), getattr(ref, f)
        assert a.dtype == b.dtype and a.shape == b.shape, f
        np.testing.assert_array_equal(a, b, err_msg=f)
    np.testing.assert_array_equal(got.time_labels, ref.time_labels)
    assert got.time_labels.dtype == ref.time_labels.dtype
    for f in ("var_names", "y_name", "treated_labels", "donor_labels",
              "N", "T", "T0", "T1", "n_treated"):
        assert getattr(got, f) == getattr(ref, f), f
    assert [type(x) for x in got.treated_labels + got.donor_labels] == \
        [type(x) for x in ref.treated_labels + ref.donor_labels]


def _beijing(alert):
    df = pd.read_csv(BASE / f"beijing_pm25_{alert}_alert.csv")
    df["treat_indicator"] = ((df["alert_if"] == 1) & (df["hour_eps"] > 48)).astype(int)
    return df


BEIJING = dict(outcome="pm25", treat="treat_indicator", unitid="id_eps", time="hour_eps",
               exog_covariates=["WSPM", "humi", "dewp", "pres"], lagged_outcome="pm25_lag1")


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_every_series_goes_through_dataprep_keeping_missing_cells(self, monkeypatch):
        calls = []
        real = SETUP.dataprep

        def spy(df, unitid, time, column, treat, *a, **k):
            calls.append((column, k.get("missing_outcomes"), k.get("keys")))
            return real(df, unitid, time, column, treat, *a, **k)

        monkeypatch.setattr(SETUP, "dataprep", spy)
        prepare_dsc_inputs(_panel(), **COLS, **EXTRA)
        assert [c[0] for c in calls] == ["y", "x1", "x2", "y_lag1"]
        assert all(c[1] == "keep" for c in calls)
        assert all(c[2] is not None and c[2] is calls[0][2] for c in calls)

    def test_setup_does_not_pivot(self):
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called


# --------------------------------------------------------------------------- #
# the old inputs, missing cells included
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),
        dict(treated=(0, 3)),
        dict(names="int", treated=(2, 11, 5), n_units=13),
        dict(shuffle=False, t0=2),
        dict(n_periods=7, t0=6),
    ])
    def test_complete_panels(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_dsc_inputs(df, **COLS, **EXTRA), _reference(df, **COLS, **EXTRA))

    def test_without_covariates_or_lag(self):
        df = _panel()
        _assert_same(prepare_dsc_inputs(df, **COLS), _reference(df, **COLS))

    def test_missing_outcome_covariate_and_lag_cells_stay_nan(self):
        df = _panel(treated=(0, 1))
        df = _plant(df, [("u3", 2004), ("u0", 2002), ("u5", 2011)])
        df = _plant(df, [("u4", 2003), ("u0", 2009)], col="x1")
        df = _plant(df, [("u6", 2000), ("u1", 2000)], col="y_lag1")
        got = prepare_dsc_inputs(df, **COLS, **EXTRA)
        _assert_same(got, _reference(df, **COLS, **EXTRA))
        assert int(np.isnan(got.Y).sum()) == 3
        assert int(np.isnan(got.X[:, :, 0]).sum()) == 2
        # the lag at t is the outcome at t - 1, so a missing outcome is a missing lag too
        assert int(np.isnan(got.Y_lag1).sum()) == 2 + 3 - 1   # (u5, 2011) is the last period

    @pytest.mark.parametrize("alert", ["orange", "red"])
    def test_the_beijing_panels(self, alert):
        df = _beijing(alert)
        got = prepare_dsc_inputs(df, **BEIJING)
        _assert_same(got, _reference(df, **BEIJING))
        assert np.isnan(got.Y).sum() == {"orange": 5, "red": 9}[alert]

    def test_the_caller_frame_is_not_changed(self):
        df = _plant(_panel(), [("u3", 2004)])
        before = df.copy()
        prepare_dsc_inputs(df, **COLS, **EXTRA)
        pd.testing.assert_frame_equal(df, before)


class TestTheFitSeesTheMissingCells:
    def test_a_donor_missing_its_outcome_is_left_out_of_that_period(self):
        df = _plant(_panel(), [("u3", 2004)])
        res = DSCAR(dict(df=df, display_graphs=False, **COLS, **EXTRA)).fit()
        j = res.inputs.donor_labels.index("u3")
        t = list(res.inputs.time_labels).index(2004)
        assert res.fit.weights[t, j] == 0.0
        assert res.fit.weights[t + 1, j] == 0.0          # its lag is missing there
        assert np.isfinite(res.att)

    def test_a_treated_pre_period_cell_is_accepted(self):
        df = _plant(_panel(treated=(0, 1)), [("u0", 2003)])
        res = DSCAR(dict(df=df, display_graphs=False, **COLS, **EXTRA)).fit()
        assert np.isfinite(res.att)

    @pytest.mark.parametrize("alert", ["orange", "red"])
    def test_the_beijing_fit_equals_the_fit_on_the_old_inputs(self, alert):
        df = _beijing(alert)
        new = run_dsc(prepare_dsc_inputs(df, **BEIJING))
        old = run_dsc(_reference(df, **BEIJING))
        assert new.att == old.att
        np.testing.assert_array_equal(new.Y0_hat, old.Y0_hat)
        np.testing.assert_array_equal(new.weights, old.weights)
        np.testing.assert_array_equal(new.Y_treated_mean, old.Y_treated_mean)


# --------------------------------------------------------------------------- #
# the treated post-treatment rule
# --------------------------------------------------------------------------- #
class TestTreatedPostCell:
    def test_refused_naming_the_unit_and_period(self):
        with pytest.raises(MlsynthDataError, match=r"treated unit 'u0'.*time 2008"):
            prepare_dsc_inputs(_plant(_panel(), [("u0", 2008)]), **COLS, **EXTRA)

    def test_the_estimator_reports_it(self):
        df = _plant(_panel(treated=(0, 2)), [("u2", 2011), ("u2", 2007)])
        with pytest.raises(MlsynthDataError, match=r"2 treated post-treatment cell.*'u2'.*time 2007"):
            DSCAR(dict(df=df, display_graphs=False, **COLS, **EXTRA)).fit()

    def test_the_intervention_period_itself_counts(self):
        with pytest.raises(MlsynthDataError, match=r"treated post-treatment.*time 2006"):
            prepare_dsc_inputs(_plant(_panel(), [("u0", 2006)]), **COLS, **EXTRA)

    def test_a_treated_unit_s_missing_covariate_after_treatment_is_accepted(self):
        df = _plant(_panel(), [("u0", 2008)], col="x1")
        _assert_same(prepare_dsc_inputs(df, **COLS, **EXTRA), _reference(df, **COLS, **EXTRA))


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_an_infinite_outcome(self):
        with pytest.raises(MlsynthDataError, match="infinite"):
            prepare_dsc_inputs(_plant(_panel(), [("u3", 2004)], value=np.inf), **COLS, **EXTRA)

    def test_an_infinite_covariate_is_named(self):
        df = _plant(_panel(), [("u3", 2004)], col="x2", value=-np.inf)
        with pytest.raises(MlsynthDataError, match=r"Covariate 'x2'.*infinite"):
            prepare_dsc_inputs(df, **COLS, **EXTRA)

    def test_an_infinite_lag_is_named(self):
        df = _plant(_panel(), [("u3", 2000)], col="y_lag1", value=np.inf)
        with pytest.raises(MlsynthDataError, match=r"Lagged outcome 'y_lag1'.*infinite"):
            prepare_dsc_inputs(df, **COLS, **EXTRA)

    def test_an_absent_row(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == "u4") & (df.year == 2003))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            DSCAR(dict(df=df, display_graphs=False, **COLS, **EXTRA)).fit()

    def test_a_duplicate_row(self):
        df = _panel()
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_dsc_inputs(pd.concat([df, df.iloc[[3]]]), **COLS, **EXTRA)

    def test_a_missing_treatment(self):
        df = _panel(shuffle=False).astype({"treat": float})
        df.loc[(df.unit == "u3") & (df.year == 2004), "treat"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing treatment values"):
            prepare_dsc_inputs(df, **COLS, **EXTRA)

    def test_a_non_binary_treatment(self):
        df = _panel()
        df["treat"] = df["treat"] * 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_dsc_inputs(df, **COLS, **EXTRA)

    def test_a_treatment_that_switches_off(self):
        df = _panel()
        df.loc[(df.unit == "u0") & (df.year >= 2010), "treat"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_dsc_inputs(df, **COLS, **EXTRA)

    def test_staggered_adoption_keeps_its_message(self):
        df = _panel(treated=(0, 1))
        df.loc[(df.unit == "u1") & (df.year < 2009), "treat"] = 0
        with pytest.raises(MlsynthDataError, match="common intervention time"):
            prepare_dsc_inputs(df, **COLS, **EXTRA)

    def test_every_unit_treated(self):
        with pytest.raises(MlsynthDataError, match="no donor"):
            prepare_dsc_inputs(_panel(n_units=3, treated=(0, 1, 2)), **COLS, **EXTRA)

    def test_one_pre_period(self):
        with pytest.raises(MlsynthDataError, match="T0 >= 2"):
            prepare_dsc_inputs(_panel(t0=1), **COLS, **EXTRA)

    def test_an_unknown_column(self):
        with pytest.raises(MlsynthDataError, match="not in df"):
            prepare_dsc_inputs(_panel(), **COLS, exog_covariates=["nope"])

    def test_an_empty_panel(self):
        with pytest.raises(MlsynthDataError, match="empty"):
            prepare_dsc_inputs(_panel().iloc[:0], **COLS, **EXTRA)


# --------------------------------------------------------------------------- #
# planted missingness on an AR(1) panel with a known effect
# --------------------------------------------------------------------------- #
def _sim(seed, n_units=20, n_periods=30, t0=20, tau=2.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_units, n_periods)) * 0.5
    eps = rng.standard_normal((n_units, n_periods)) * 0.3
    Y = np.zeros((n_units, n_periods))
    Y[:, 0] = rng.standard_normal(n_units)
    for t in range(1, n_periods):
        Y[:, t] = 0.5 * x[:, t] + 0.6 * Y[:, t - 1] + eps[:, t]
    D = np.zeros((n_units, n_periods), dtype=int)
    D[0, t0:] = 1
    Y = Y + tau * D
    lag0 = rng.standard_normal(n_units)
    return pd.DataFrame(dict(
        unit=np.repeat([f"u{i:02d}" for i in range(n_units)], n_periods),
        year=np.tile(np.arange(n_periods), n_units),
        y=Y.ravel(), x1=x.ravel(),
        y_lag1=np.column_stack([lag0, Y[:, :-1]]).ravel(), treat=D.ravel()))


def _drop_at_random(df, frac, seed):
    """NaN a ``frac`` share of the outcome cells that are not treated post-treatment."""
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(df["treat"].to_numpy() == 0)
    out = df.copy()
    out.iloc[rng.choice(cand, int(round(frac * cand.size)), replace=False),
             out.columns.get_loc("y")] = np.nan
    return out


# Mean absolute ATT error over eight seeds (tau = 2, one treated unit, noise
# sd 0.3) for each share of non-treated-post outcome cells removed at random.
# The fit gives 0.23 with no cell removed, 0.24 at 10% and 0.25 at 30%: the
# error is the noise of one treated unit over ten post periods, and the
# complete-case match keeps it there as donors drop out of single periods. The
# bound is 1.5 times the error with no cell removed.
SIM_BOUNDS = {0.0: 0.35, 0.10: 0.35, 0.30: 0.35}


@pytest.mark.parametrize("frac", sorted(SIM_BOUNDS))
def test_planted_missingness_stays_near_the_truth(frac):
    errs = []
    for seed in range(8):
        df = _drop_at_random(_sim(seed), frac, 100 + seed)
        res = DSCAR(dict(df=df, display_graphs=False, **COLS, exog_covariates=["x1"],
                         lagged_outcome="y_lag1")).fit()
        errs.append(abs(res.att - 2.0))
    assert np.mean(errs) <= SIM_BOUNDS[frac], (frac, np.round(errs, 3))


# --------------------------------------------------------------------------- #
# properties over random panels with random missing cells
# --------------------------------------------------------------------------- #
@st.composite
def missing_panels(draw):
    n_units = draw(st.integers(3, 8))
    n_periods = draw(st.integers(4, 10))
    t0 = draw(st.integers(2, n_periods - 1))
    n_treated = draw(st.integers(1, n_units - 1))
    df = _panel(n_units=n_units, n_periods=n_periods, treated=tuple(range(n_treated)),
                t0=t0, seed=draw(st.integers(0, 10_000)),
                names=draw(st.sampled_from(["str", "int"])))
    for col in ("y", "x1", "x2", "y_lag1"):
        k = draw(st.integers(0, 3))
        rows = draw(st.lists(st.integers(0, len(df) - 1), min_size=k, max_size=k, unique=True))
        df.iloc[rows, df.columns.get_loc(col)] = np.nan
    return df


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=missing_panels())
def test_property_the_old_inputs_or_the_treated_post_cell_is_refused(df):
    if (df.y.isna() & (df.treat == 1)).any():
        with pytest.raises(MlsynthDataError, match="treated post-treatment"):
            prepare_dsc_inputs(df, **COLS, **EXTRA)
        return
    got = prepare_dsc_inputs(df, **COLS, **EXTRA)
    _assert_same(got, _reference(df, **COLS, **EXTRA))
    # the treated block comes first and has no missing post-period outcome
    assert np.isfinite(got.Y[:got.n_treated, got.T0:]).all()
    assert int(np.isnan(got.Y).sum()) == int(df.y.isna().sum())
