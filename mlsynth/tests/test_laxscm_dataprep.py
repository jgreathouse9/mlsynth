"""RESCM (LAXSCM) ingestion through ``datautils.dataprep`` (#1138, part 1).

RESCM used to pivot the outcome itself and check the pivot for NaN cells only.
It now takes the outcome matrix, the time labels, the donor order and the
pre-period count from ``balance`` and ``dataprep``, so it gets the same
refusals as every other estimator: a missing or non-finite outcome anywhere in
the panel, an absent or duplicate unit-period, a non-binary treatment, and a
treatment that switches off after it starts.

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

from mlsynth import RESCM
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import PanelKeys
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.laxscm_helpers import setup as SETUP
from mlsynth.utils.laxscm_helpers.setup import prepare_rescm_inputs
from mlsynth.utils.laxscm_helpers.structures import RESCMInputs

COLS = dict(unitid="unit", time="time", outcome="y", treat="d")
LAYOUT = {"y": "C_CONTIGUOUS", "X": "F_CONTIGUOUS"}
PANDAS_2 = int(pd.__version__.split(".")[0]) < 3


def _reference(df, unitid, time, outcome, treat):
    """The pre-#1138 setup's arrays and labels."""
    treated_rows = df[df[treat] == 1]
    treated = pd.unique(treated_rows[unitid])[0]
    t_int = treated_rows.loc[treated_rows[unitid] == treated, time].min()
    times = np.sort(pd.unique(df[time]))
    wide = df.pivot(index=time, columns=unitid, values=outcome).reindex(times)
    donors = [u for u in wide.columns if u != treated]
    return dict(y=wide[treated].to_numpy(dtype=float),
                X=wide[donors].to_numpy(dtype=float),
                T0=int(np.sum(times < t_int)), treated_label=treated,
                donors=IndexSet.from_labels(donors).labels,
                times=IndexSet.from_labels(times).labels, intervention_time=t_int)


def _panel(n_donors=6, T=10, T0=6, treated=0, seed=0, names="int",
           shuffle=True, t0=2000):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_donors + 1):
        for t in range(T):
            rows.append(dict(unit=u if names == "int" else f"s{u:02d}",
                             time=t0 + t, y=rng.normal() + u + 0.1 * t,
                             d=int(u == treated and t >= T0), x=rng.normal()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _name(u, names="int"):
    return u if names == "int" else f"s{u:02d}"


def _assert_same(inputs, ref):
    assert isinstance(inputs, RESCMInputs)
    for field in ("y", "X"):
        got = getattr(inputs, field)
        assert got.dtype == np.float64, field
        assert got.shape == ref[field].shape, field
        np.testing.assert_array_equal(got, ref[field], err_msg=field)
        # The solvers' sums depend on the layout, so it is pinned too. The old
        # pivot's layout depends on the pandas version; the arrays are pinned
        # to what it gave under pandas 2.x, the version CI runs.
        assert got.flags[LAYOUT[field]], f"{field} is not {LAYOUT[field]}"
        if PANDAS_2:
            assert got.flags["C_CONTIGUOUS"] == ref[field].flags["C_CONTIGUOUS"], field
            assert got.flags["F_CONTIGUOUS"] == ref[field].flags["F_CONTIGUOUS"], field
    assert inputs.T0 == ref["T0"]
    assert inputs.treated_label == ref["treated_label"]
    assert type(inputs.treated_label) is type(ref["treated_label"])
    for got, want in ((inputs.unit_index.labels, ref["donors"]),
                      (inputs.time_index.labels, ref["times"])):
        np.testing.assert_array_equal(got, want)
        assert got.dtype == want.dtype
    assert inputs.metadata["intervention_time"] == ref["intervention_time"]


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_the_outcome_is_read_through_dataprep_with_balances_keys(self, monkeypatch):
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
        prepare_rescm_inputs(_panel(), **COLS)
        assert len(scans) == 1 and isinstance(scans[0], PanelKeys)
        assert len(calls) == 1 and calls[0][0] == "y"
        assert calls[0][1] is scans[0]

    def test_setup_does_not_pivot(self):
        # A runtime patch of DataFrame.pivot cannot test this: dataprep's
        # _wide_pivot falls back to df.pivot on some pandas versions.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called

    def test_the_estimator_scans_the_panel_once(self, monkeypatch):
        import mlsynth.estimators.laxscm as EST
        from mlsynth.utils import datautils

        scans = []
        real = PanelKeys.__init__

        def counting(self, *a, **k):
            scans.append(1)
            real(self, *a, **k)

        monkeypatch.setattr(datautils.PanelKeys, "__init__", counting)
        res = RESCM(dict(df=_panel(T=24, T0=16), **COLS, methods=["SC"],
                         display_graphs=False)).fit()
        assert np.isfinite(res.att)
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
        dict(t0=-5),                                     # negative times
        dict(n_donors=39, T=30, T0=20, treated=17, seed=5),          # 40 units
        dict(n_donors=39, T=30, T0=24, names="str", seed=6),         # 40, str ids
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        _assert_same(prepare_rescm_inputs(df, **COLS), _reference(df, **COLS))

    @pytest.mark.parametrize("label", ["datetime", "str", "float"])
    def test_period_labels_keep_their_type(self, label):
        df = _panel()
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(91 * step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0}[label]
        _assert_same(prepare_rescm_inputs(df, **COLS), _reference(df, **COLS))

    def test_a_boolean_treatment_matches_the_old_pivot(self):
        df = _panel()
        df["d"] = df["d"].astype(bool)
        _assert_same(prepare_rescm_inputs(df, **COLS), _reference(df, **COLS))

    def test_an_integer_outcome_matches_the_old_pivot(self):
        df = _panel()
        df["y"] = (10 * df["y"]).round().astype(int)
        _assert_same(prepare_rescm_inputs(df, **COLS), _reference(df, **COLS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_rescm_inputs(df, **COLS)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        res = RESCM(dict(df=_panel(T=24, T0=16), **COLS, methods=["SC", "RELAX_L2"],
                         display_graphs=False)).fit()
        assert np.isfinite(res.att)
        assert set(res.fits) == {"SC", "RELAX_L2"}


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2008), "y"] = np.nan    # post-period
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_rescm_inputs(df, **COLS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old NaN check let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_rescm_inputs(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 5) & (df.time == 2004))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_rescm_inputs(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old setup's pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_rescm_inputs(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_non_binary_treatment_is_refused(self):
        # The old setup read only the cells equal to 1.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2009), "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            prepare_rescm_inputs(df, **COLS)

    def test_a_treatment_that_switches_off_is_refused(self):
        # The old setup read it from its first treated period.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2008), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_rescm_inputs(df, **COLS)

    def test_a_missing_treatment_after_adoption_is_refused(self):
        # The old setup read only the cells equal to 1 and accepted it.
        df = _panel(shuffle=False)
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 0) & (df.time == 2008), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="not sustained"):
            prepare_rescm_inputs(df, **COLS)

    def test_a_missing_donor_treatment_is_read_as_untreated_as_before(self):
        # dataprep's binary check skips NaN, and the old setup ignored it.
        df = _panel()
        df["d"] = df["d"].astype(float)
        df.loc[(df.unit == 3) & (df.time == 2004), "d"] = np.nan
        _assert_same(prepare_rescm_inputs(df, **COLS), _reference(df, **COLS))

    def test_the_estimator_reports_the_switch_off(self):
        df = _panel(shuffle=False, T=24, T0=16)
        df.loc[(df.unit == 0) & (df.time == 2020), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            RESCM(dict(df=df, **COLS, methods=["SC"], display_graphs=False)).fit()


# --------------------------------------------------------------------------- #
# RESCM's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_no_treated_unit(self):
        df = _panel()
        df["d"] = 0
        with pytest.raises(MlsynthDataError, match="No treated rows"):
            prepare_rescm_inputs(df, **COLS)

    def test_two_treated_units_in_one_cohort(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time >= 2006), "d"] = 1
        with pytest.raises(MlsynthDataError, match="exactly one treated unit"):
            prepare_rescm_inputs(df, **COLS)

    def test_two_treated_units_in_two_cohorts(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time >= 2004), "d"] = 1
        with pytest.raises(MlsynthDataError, match="exactly one treated unit"):
            prepare_rescm_inputs(df, **COLS)

    def test_one_pre_period_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="at least two pre-treatment periods"):
            prepare_rescm_inputs(_panel(T0=1), **COLS)

    def test_treatment_in_the_first_period(self):
        # dataprep's own refusal, ahead of RESCM's two-period minimum.
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_rescm_inputs(_panel(T0=0), **COLS)

    def test_no_donor_is_passed_on_as_before(self):
        # The old setup returned an empty donor matrix for a lone treated
        # unit; dataprep is asked not to refuse it either.
        df = _panel()
        df = df[df.unit == 0]
        inputs = prepare_rescm_inputs(df, **COLS)
        assert inputs.X.shape == (10, 0) and len(inputs.unit_index) == 0
        _assert_same(inputs, _reference(df, **COLS))


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
DEFECTS = ["none", "none", "none", "nan_y", "inf_y", "gap", "dup", "off",
           "nonbinary"]


@st.composite
def cases(draw):
    n_donors = draw(st.integers(1, 6))
    T = draw(st.integers(3, 9))
    T0 = draw(st.integers(2, T - 1))
    treated = draw(st.integers(0, n_donors))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_donors=n_donors, T=T, T0=T0, treated=treated, seed=seed,
                names=names)
    defect = draw(st.sampled_from(DEFECTS))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan_y":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf_y":
        df.iloc[i, df.columns.get_loc("y")] = draw(st.sampled_from([np.inf, -np.inf]))
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
        u = draw(st.integers(0, n_donors))
        t = draw(st.integers(0, T - 1))
        df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + t), "d"] = 2
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            prepare_rescm_inputs(df, **COLS)
        return
    inputs = prepare_rescm_inputs(df, **COLS)
    _assert_same(inputs, _reference(df, **COLS))
    # The pre-period is the treated unit's untreated run, and the donors and
    # the treated unit partition the panel's units.
    assert inputs.T0 == int((df.loc[df.unit == inputs.treated_label, "d"] == 0).sum())
    assert len(inputs.unit_index) + 1 == df["unit"].nunique()
    assert inputs.X.shape == (inputs.T, len(inputs.unit_index))
