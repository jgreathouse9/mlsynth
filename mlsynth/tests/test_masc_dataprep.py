"""MASC ingestion through ``datautils.dataprep`` (#1138, part 1).

MASC used to pivot the long panel itself. It now takes the outcome matrix, the
treated unit, the time labels and the adoption period from ``balance`` and
``dataprep``, and gets the same refusals as every other estimator: a missing or
non-finite outcome cell, an absent or duplicate unit-period, a non-binary
treatment, and a treatment that switches off after it starts. MASC's own checks
(one treated unit, two pre-periods, the covariate columns) stay on top.

Each covariate panel is read the same way, as the outcome of its own
``dataprep`` call with ``missing_outcomes="keep"``. A covariate need not be
observed in every period: MASC fills each unit's gaps forward and then backward,
and the Basque application relies on it (its sector shares are observed in a
handful of years), so a NaN covariate cell is kept for the fill. An infinite
covariate value is refused under the covariate's name; the old setup passed it
to the fit.

``_reference`` is the old setup, verbatim, kept as the oracle the new path must
match on every panel both accept. Nothing in the setup sums or averages, so the
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

from mlsynth import MASC
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.masc_helpers import setup as SETUP
from mlsynth.utils.masc_helpers.setup import prepare_masc_inputs
from mlsynth.utils.masc_helpers.structures import MASCInputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


def _reference(df, *, outcome, treat, unitid, time, covariates=None,
               covariate_windows=None):
    """The pre-#1138 setup, verbatim."""
    treated_rows = df.loc[df[treat] == 1, [unitid, time]]
    if treated_rows.empty:
        raise MlsynthDataError("none found")
    treated_units = treated_rows[unitid].unique()
    if treated_units.size != 1:
        raise MlsynthDataError("single treated unit")
    treated_label = treated_units[0]
    intervention_time = treated_rows[time].min()
    all_units = list(df[unitid].unique())
    never_treated = [
        u for u in all_units
        if u != treated_label and df.loc[df[unitid] == u, treat].sum() == 0
    ]
    Y_wide = df.pivot(index=time, columns=unitid, values=outcome).sort_index()
    time_index = Y_wide.index.to_numpy()
    Y_treated = Y_wide[treated_label].to_numpy(dtype=float)
    Y_donors = Y_wide[never_treated].to_numpy(dtype=float)
    if not (np.isfinite(Y_treated).all() and np.isfinite(Y_donors).all()):
        raise MlsynthDataError("missing outcomes")
    treatment_period = int(np.argmax(time_index >= intervention_time)) + 1
    T = int(time_index.size)
    T0 = treatment_period - 1
    cov_treated_panel = cov_donors_panel = None
    if covariates:
        treated_stacks, donor_stacks = [], []
        for c in covariates:
            cwide = (df.pivot(index=time, columns=unitid, values=c)
                     .reindex(time_index).ffill().bfill())
            if cwide[[treated_label, *never_treated]].isna().any().any():
                raise MlsynthDataError("covariate missing")
            treated_stacks.append(cwide[treated_label].to_numpy(dtype=float))
            donor_stacks.append(cwide[list(never_treated)].to_numpy(dtype=float))
        cov_treated_panel = np.column_stack(treated_stacks)
        cov_donors_panel = np.stack(donor_stacks, axis=-1)
    return MASCInputs(
        Y_treated=Y_treated, Y_donors=Y_donors, treated_label=treated_label,
        donor_labels=tuple(never_treated), time_index=time_index,
        intervention_time=intervention_time, treatment_period=treatment_period,
        T=T, T0=T0, T1=T - T0, J=Y_donors.shape[1],
        cov_treated_panel=cov_treated_panel, cov_donors_panel=cov_donors_panel,
        covariate_names=tuple(covariates or ()),
        covariate_windows=dict(covariate_windows or {}),
    )


def _name(u, names):
    return u if names == "int" else f"s{u:02d}"


def _panel(n_units=8, n_periods=12, treated=0, adopt=6, seed=0, names="int",
           shuffle=True, t0=2000, cov_gaps=0):
    """Balanced long panel with one treated unit and two covariates.

    ``cov_gaps`` blanks that many cells of the covariate ``x2``, which MASC
    fills forward and back within each unit.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            d = int(u == treated and t >= adopt)
            rows.append(dict(unit=_name(u, names), time=t0 + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d,
                             x1=rng.normal(), x2=rng.normal()))
    df = pd.DataFrame(rows)
    if cov_gaps:
        idx = rng.choice(len(df), size=cov_gaps, replace=False)
        df.loc[idx, "x2"] = np.nan
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _assert_same(got, ref):
    assert isinstance(got, MASCInputs)
    for f in ("Y_treated", "Y_donors", "time_index", "cov_treated_panel",
              "cov_donors_panel"):
        a, b = getattr(got, f), getattr(ref, f)
        if b is None:
            assert a is None, f
            continue
        assert a.dtype == b.dtype, f"{f}: dtype {a.dtype} vs {b.dtype}"
        assert a.shape == b.shape, f
        # The solver's sums depend on the layout, so it is pinned too.
        assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"], f
        assert a.flags["F_CONTIGUOUS"] == b.flags["F_CONTIGUOUS"], f
        np.testing.assert_array_equal(a, b, err_msg=f)
    for f in ("treated_label", "intervention_time"):
        a, b = getattr(got, f), getattr(ref, f)
        assert a == b and type(a) is type(b), f"{f}: {a!r} vs {b!r}"
    assert got.donor_labels == ref.donor_labels
    assert [type(u) for u in got.donor_labels] == [type(u) for u in ref.donor_labels]
    for f in ("treatment_period", "T", "T0", "T1", "J", "covariate_names",
              "covariate_windows"):
        assert getattr(got, f) == getattr(ref, f), f


def _both(df, **kw):
    args = {**COLS, **kw}
    return prepare_masc_inputs(df, **args), _reference(df, **args)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_setup_calls_dataprep_once_per_column(self, monkeypatch):
        calls = []
        real = SETUP.dataprep

        def spy(*a, **k):
            calls.append((a[3], k.get("keys"), k.get("missing_outcomes", "refuse")))
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "dataprep", spy)
        prepare_masc_inputs(_panel(cov_gaps=5), **COLS, covariates=["x1", "x2"])
        assert [(c, m) for c, _, m in calls] == [
            ("y", "refuse"), ("x1", "keep"), ("x2", "keep")]
        # balance()'s key scan is taken once and reused by every call.
        assert calls[0][1] is not None
        assert all(k is calls[0][1] for _, k, _ in calls)

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
        dict(names="str", treated=3, adopt=8),
        dict(shuffle=False, t0=-5, treated=7, adopt=11),     # last-period adoption
        dict(treated=5, adopt=2),                            # two pre-periods
        dict(n_units=40, n_periods=20, treated=17, adopt=12, seed=3),
        dict(n_units=40, n_periods=20, names="str", treated=0, adopt=15, seed=4),
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        _assert_same(*_both(_panel(**kw)))

    @pytest.mark.parametrize("kw", [
        dict(),
        dict(names="str", treated=4, adopt=7, cov_gaps=20),
        dict(n_units=40, n_periods=20, treated=9, adopt=12, seed=5, cov_gaps=150),
    ])
    def test_covariate_panels_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        windows = {"x1": (2001, 2004)}
        _assert_same(*_both(df, covariates=["x1", "x2"], covariate_windows=windows))

    def test_donors_keep_their_first_appearance_order(self):
        # The old setup listed donors in the order units first appear in the
        # frame, and the weights are reported in that order.
        df = _panel(shuffle=False)
        order = [5, 2, 7, 0, 1, 3, 6, 4]
        df = pd.concat([df[df.unit == u] for u in order])
        inputs = prepare_masc_inputs(df, **COLS)
        assert list(inputs.donor_labels) == [5, 2, 7, 1, 3, 6, 4]
        _assert_same(inputs, _reference(df, **COLS))

    @pytest.mark.parametrize("label", ["datetime", "str", "float"])
    def test_period_labels_keep_their_type(self, label):
        df = _panel(treated=1, adopt=7)
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0}[label]
        _assert_same(*_both(df, covariates=["x1"]))

    def test_a_boolean_treatment_is_read_as_before(self):
        df = _panel()
        df["d"] = df["d"].astype(bool)
        _assert_same(*_both(df))

    def test_extra_columns_are_ignored(self):
        df = _panel()
        df["junk"] = "z"
        _assert_same(*_both(df))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel(cov_gaps=10)
        before = df.copy()
        prepare_masc_inputs(df, **COLS, covariates=["x1", "x2"])
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        df = _panel(n_periods=20, adopt=14)
        res = MASC(dict(df=df, **COLS, display_graphs=False)).fit()
        assert np.isfinite(res.att)

    def test_fit_smoke_with_covariate_gaps(self):
        df = _panel(n_periods=20, adopt=14, cov_gaps=30)
        res = MASC(dict(df=df, **COLS, covariates=["x1", "x2"],
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
            prepare_masc_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2009), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_masc_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 6) & (df.time == 2001))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_masc_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_masc_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup took the first treated period and accepted this.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_masc_inputs(df, **COLS)

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, n_periods=20, adopt=14)
        df.loc[(df.unit == 0) & (df.time == 2016), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            MASC(dict(df=df, **COLS, display_graphs=False)).fit()

    def test_a_missing_treatment_after_adoption_is_refused(self):
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 0) & (df.time == 2008), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_masc_inputs(df, **COLS)

    @pytest.mark.parametrize("unit", [0, 4])
    def test_a_non_binary_treatment_is_refused(self, unit):
        # A 2 on the treated unit was kept as treated; on another unit it
        # dropped that unit from the donor pool. Both are refused now.
        df = _panel(shuffle=False)
        df.loc[(df.unit == unit) & (df.time == 2011), "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_masc_inputs(df, **COLS)

    def test_no_treated_cell_is_refused(self):
        df = _panel()
        df["d"] = 0
        with pytest.raises(MlsynthDataError, match="No treated"):
            prepare_masc_inputs(df, **COLS)

    def test_treatment_in_the_first_period_is_refused(self):
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_masc_inputs(_panel(adopt=0), **COLS)


# --------------------------------------------------------------------------- #
# MASC's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["y", "d", "unit", "time"])
    def test_missing_column(self, col):
        df = _panel().drop(columns=col)
        with pytest.raises(MlsynthDataError, match=f"Required column '{col}' missing"):
            prepare_masc_inputs(df, **COLS)

    def test_two_treated_units(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time >= 2006), "d"] = 1      # same cohort
        with pytest.raises(MlsynthDataError, match="single treated unit; found 2"):
            prepare_masc_inputs(df, **COLS)

    def test_one_pre_period_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="at least 2 pre-treatment periods"):
            prepare_masc_inputs(_panel(adopt=1), **COLS)

    def test_one_donor_is_enough(self):
        _assert_same(*_both(_panel(n_units=2)))

    def test_an_absent_covariate_is_refused(self):
        with pytest.raises(MlsynthDataError, match="Covariate 'zz' not found"):
            prepare_masc_inputs(_panel(), **COLS, covariates=["x1", "zz"])

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_an_infinite_covariate_is_refused_under_its_name(self, value):
        # The old setup passed an infinite covariate value to the fit.
        df = _panel(shuffle=False, cov_gaps=4)
        df.loc[(df.unit == 2) & (df.time == 2003), "x2"] = value
        with pytest.raises(MlsynthDataError, match="Covariate 'x2' cannot be read") as info:
            prepare_masc_inputs(df, **COLS, covariates=["x1", "x2"])
        assert "infinite" in str(info.value)          # dataprep's cause is kept
        assert isinstance(info.value.__cause__, MlsynthDataError)

    def test_the_estimator_reports_an_infinite_covariate(self):
        df = _panel(n_periods=20, adopt=14)
        df.loc[df.index[5], "x1"] = np.inf
        with pytest.raises(MlsynthDataError, match="Covariate 'x1' cannot be read"):
            MASC(dict(df=df, **COLS, covariates=["x1"], display_graphs=False)).fit()

    def test_a_covariate_missing_for_a_whole_unit_is_refused(self):
        df = _panel()
        df.loc[df.unit == 3, "x1"] = np.nan
        with pytest.raises(MlsynthDataError, match="Covariate 'x1' is missing"):
            prepare_masc_inputs(df, **COLS, covariates=["x1"])


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(2, 9))
    T = draw(st.integers(4, 10))
    treated = draw(st.integers(0, n_units - 1))
    adopt = draw(st.integers(2, T - 1))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    gaps = draw(st.integers(0, 3))
    df = _panel(n_units=n_units, n_periods=T, treated=treated, adopt=adopt,
                seed=seed, names=names, cov_gaps=gaps)
    defect = draw(st.sampled_from(
        ["none", "none", "none", "nan", "inf", "gap", "dup", "off", "dose", "cinf"]))
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
        if adopt >= T - 1:
            defect = "none"
        else:
            df.loc[(df.unit == _name(treated, names)) & (df.time == 2000 + T - 1), "d"] = 0
    elif defect == "dose":
        df.iloc[i, df.columns.get_loc("d")] = 2
    elif defect == "cinf":
        df.iloc[i, df.columns.get_loc("x2")] = np.inf
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    kw = dict(covariates=["x1", "x2"])
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_masc_inputs(df, **COLS, **kw)
        return
    got = prepare_masc_inputs(df, **COLS, **kw)
    _assert_same(got, _reference(df, **COLS, **kw))
    # The treated unit is the only unit with a treated cell, the donors are
    # every other unit, and the adoption period is its first treated period.
    treated_rows = df[df.d == 1]
    assert set(treated_rows.unit) == {got.treated_label}
    assert set(got.donor_labels) == set(df.unit) - {got.treated_label}
    assert got.intervention_time == treated_rows.time.min()
    assert got.T0 + got.T1 == got.T == df.time.nunique()
