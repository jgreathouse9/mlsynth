"""MUSC ingestion through ``datautils.dataprep`` (#1138, part 1).

MUSC used to pivot the long panel in ``prepare_musc_inputs``, with units and
periods in the order their rows first appear, and to collapse a multi-unit
cohort into a synthetic unit by building a second long panel and pivoting that.
It now reads the panel once through ``balance`` and ``dataprep``: the outcome
matrix comes from ``dataprep``, units and periods in sorted order, and a cohort
is collapsed on that matrix. It gets the same refusals as every other estimator:
a missing or non-finite outcome anywhere in the panel (a NaN in a collapsed
cohort's unit used to be averaged away), an absent or duplicate unit-period, a
non-binary treatment, and a treatment that switches off after it starts.
``prepare_musc_inputs``, called with an explicit treated unit and no treatment
column, reads its panel through ``geoex_dataprep``.

``_reference`` is the old chain (``prepare_musc_inputs`` and the cohort
collapse), kept as the oracle. The old chain read units and periods in row
order, so it is applied to the panel sorted by unit and period; the new path
must match it on every row order.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import MUSC
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import PanelKeys
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.musc_helpers import orchestration as ORCH
from mlsynth.utils.musc_helpers import setup as SETUP
from mlsynth.utils.musc_helpers.orchestration import (
    _synthetic_cohort_label,
    derive_treatment_cohorts,
)
from mlsynth.utils.musc_helpers.setup import (
    cohort_inputs,
    prepare_musc_inputs,
    read_musc_panel,
)
from mlsynth.utils.musc_helpers.structures import MUSCInputs

COLS = dict(unitid="unit", time="time", outcome="y", treat="d")


# --------------------------------------------------------------------------- #
# the old chain, frozen
# --------------------------------------------------------------------------- #
def _old_prepare(df, unitid, time, outcome, treated_unit, intervention_time):
    units = list(pd.unique(df[unitid]))
    unit_index = IndexSet.from_labels(units)
    treated_idx = int(unit_index.get_index([treated_unit])[0])
    donor_idx = np.asarray([i for i in range(len(units)) if i != treated_idx], dtype=int)
    times = list(pd.unique(df[time]))
    time_index = IndexSet.from_labels(times)
    wide = df.pivot(index=time, columns=unitid, values=outcome)
    wide = wide.reindex(index=times, columns=units)
    Y = wide.to_numpy(dtype=float).T
    T0 = int(time_index.get_index([intervention_time])[0])
    return dict(unit_labels=unit_index.labels, time_labels=time_index.labels,
                treated_idx=treated_idx, donor_idx=donor_idx, Y=Y, T0=T0,
                treated_unit=treated_unit, intervention_time=intervention_time)


def _old_collapse(df, unitid, time, outcome, treat, treated_units,
                  intervention_time, synthetic_label, other_treated_units=()):
    treated_set, other_set = set(treated_units), set(other_treated_units)
    cohort_rows = df.loc[df[unitid].isin(treated_set), [time, outcome]].copy()
    averaged = cohort_rows.groupby(time, as_index=False, observed=True)[outcome].mean()
    averaged[unitid] = synthetic_label
    averaged[treat] = (averaged[time] >= intervention_time).astype(int)
    donor_rows = df.loc[~df[unitid].isin(treated_set | other_set)].copy()
    return pd.concat([donor_rows, averaged], ignore_index=True)


def _reference(df, unitid, time, outcome, treat):
    """The old inputs per cohort, read from the panel sorted by unit and period."""
    df = df.sort_values([unitid, time], kind="stable")
    cohorts = derive_treatment_cohorts(df, unitid, time, treat)
    every = [u for units, _ in cohorts for u in units]
    out = []
    for units, t in cohorts:
        if len(cohorts) == 1 and len(units) == 1:
            out.append(_old_prepare(df, unitid, time, outcome, units[0], t))
            continue
        label = _synthetic_cohort_label(units)
        others = [u for u in every if u not in set(units)] if len(cohorts) > 1 else ()
        collapsed = _old_collapse(df, unitid, time, outcome, treat, units, t, label, others)
        out.append(_old_prepare(collapsed, unitid, time, outcome, label, t))
    return out


def _new(df, unitid, time, outcome, treat):
    wide = read_musc_panel(df, unitid=unitid, time=time, outcome=outcome, treat=treat)
    cohorts = derive_treatment_cohorts(df, unitid, time, treat)
    every = [u for units, _ in cohorts for u in units]
    out = []
    for units, t in cohorts:
        others = [u for u in every if u not in set(units)] if len(cohorts) > 1 else ()
        out.append(cohort_inputs(wide, outcome=outcome, treated_units=units,
                                 intervention_time=t, other_treated_units=others,
                                 collapse=len(cohorts) > 1 or len(units) > 1))
    return out


def _collapse_tolerance(Y):
    """Rounding bound for a cohort mean: n terms, each off by at most eps * sum|y|."""
    return 4 * np.finfo(float).eps * Y.shape[0] * (np.abs(Y).max() + 1.0)


def _assert_same(inputs, ref, n_collapsed=0):
    assert isinstance(inputs, MUSCInputs)
    for got, want in ((inputs.unit_index.labels, ref["unit_labels"]),
                      (inputs.time_index.labels, ref["time_labels"])):
        np.testing.assert_array_equal(got, want)
        assert got.dtype == want.dtype
    assert inputs.treated_idx == ref["treated_idx"]
    np.testing.assert_array_equal(inputs.donor_idx, ref["donor_idx"])
    assert inputs.donor_idx.dtype == ref["donor_idx"].dtype
    assert inputs.T0 == ref["T0"]
    assert inputs.Y.dtype == np.float64 and inputs.Y.shape == ref["Y"].shape
    assert inputs.Y.flags["C_CONTIGUOUS"] == ref["Y"].flags["C_CONTIGUOUS"]
    assert inputs.Y.flags["F_CONTIGUOUS"] == ref["Y"].flags["F_CONTIGUOUS"]
    donors = ref["donor_idx"]
    np.testing.assert_array_equal(inputs.Y[donors], ref["Y"][donors])
    i = ref["treated_idx"]
    if n_collapsed > 2:
        # A mean of three or more units: the old groupby summed with Kahan
        # compensation, the new path sums the matrix row.
        np.testing.assert_allclose(inputs.Y[i], ref["Y"][i], rtol=0,
                                   atol=_collapse_tolerance(ref["Y"]))
    else:
        np.testing.assert_array_equal(inputs.Y[i], ref["Y"][i])
    assert inputs.metadata == {"outcome": "y", "treated_unit": ref["treated_unit"],
                               "intervention_time": ref["intervention_time"]}


def _panel(n_units=8, T=12, adopt=None, seed=0, names="int", shuffle=True,
           t0=2000):
    """Balanced long panel; ``adopt`` maps unit index -> adoption period index."""
    adopt = {0: 8} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            d = int(u in adopt and t >= adopt[u])
            rows.append(dict(unit=u if names == "int" else f"s{u:02d}", time=t0 + t,
                             y=rng.normal() + 0.3 * u + 0.2 * t + 1.5 * d, d=d))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _name(u, names="int"):
    return u if names == "int" else f"s{u:02d}"


def _check(df):
    cohorts = derive_treatment_cohorts(df, "unit", "time", "d")
    for inputs, ref, (units, _) in zip(_new(df, **COLS), _reference(df, **COLS), cohorts):
        _assert_same(inputs, ref, n_collapsed=len(units))


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_the_panel_is_read_through_dataprep_with_balances_keys(self, monkeypatch):
        scans, calls = [], []
        real_balance, real_dataprep = SETUP.balance, SETUP.dataprep

        def balance_spy(*a, **k):
            scans.append(real_balance(*a, **k))
            return scans[-1]

        def dataprep_spy(*a, **k):
            calls.append((a[3], a[4], k.get("keys")))
            return real_dataprep(*a, **k)

        monkeypatch.setattr(SETUP, "balance", balance_spy)
        monkeypatch.setattr(SETUP, "dataprep", dataprep_spy)
        read_musc_panel(_panel(), **COLS)
        assert len(scans) == 1 and isinstance(scans[0], PanelKeys)
        assert len(calls) == 1 and calls[0][:2] == ("y", "d")
        assert calls[0][2] is scans[0]

    def test_prepare_musc_inputs_reads_through_geoex_dataprep(self, monkeypatch):
        calls = []
        real = SETUP.geoex_dataprep

        def spy(*a, **k):
            calls.append(a[1:4])
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "geoex_dataprep", spy)
        prepare_musc_inputs(_panel(), unitid="unit", time="time", outcome="y",
                            treated_unit=0, intervention_time=2008)
        assert calls == [("unit", "time", "y")]

    @pytest.mark.parametrize("module", [SETUP, ORCH])
    def test_no_pivot(self, module):
        # A runtime patch of DataFrame.pivot cannot test this: dataprep's
        # _wide_pivot falls back to df.pivot on some pandas versions.
        tree = ast.parse(inspect.getsource(module))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called

    @pytest.mark.parametrize("adopt", [{0: 8}, {0: 8, 1: 8}, {0: 7, 1: 9}])
    def test_the_estimator_scans_the_panel_once(self, monkeypatch, adopt):
        import mlsynth.estimators.musc as EST
        from mlsynth.utils import datautils

        scans = []
        real = PanelKeys.__init__

        def counting(self, *a, **k):
            scans.append(1)
            real(self, *a, **k)

        monkeypatch.setattr(datautils.PanelKeys, "__init__", counting)
        res = MUSC(dict(df=_panel(adopt=adopt), **COLS, display_graphs=False,
                        run_inference=False)).fit()
        assert np.isfinite(res.att)
        assert len(scans) == 1, "the panel's keys were scanned more than once"
        assert not hasattr(EST, "balance")


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),                                              # one treated unit
        dict(names="str", adopt={3: 9}),
        dict(shuffle=False, t0=-5, adopt={2: 11}),           # last-period adoption
        dict(adopt={0: 6, 1: 6}),                            # one cohort of two
        dict(adopt={0: 6, 1: 6, 5: 6}),                      # one cohort of three
        dict(adopt={0: 4, 1: 6, 4: 9}),                      # staggered, three cohorts
        dict(names="str", adopt={0: 5, 2: 5, 3: 5, 6: 8}),   # staggered, a cohort of three
        dict(n_units=40, T=20, adopt={0: 12}, seed=3),                   # 40 units
        dict(n_units=40, T=20, names="str", seed=4,
             adopt={0: 10, 5: 10, 9: 14, 33: 17}),                       # 40, staggered
    ])
    def test_inputs_match_the_old_chain(self, kw):
        _check(_panel(**kw))

    @pytest.mark.parametrize("label", ["datetime", "str", "float"])
    def test_period_labels_keep_their_type(self, label):
        df = _panel(adopt={1: 7, 2: 7})
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(31 * step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0}[label]
        _check(df)

    def test_integer_ids_collapse_with_a_string_label(self):
        # The collapsed cohort's label is a string beside integer donors; the
        # old pivot accepted the mixed labels and the new collapse does too.
        df = _panel(adopt={0: 6, 1: 6})
        inputs = _new(df, **COLS)[0]
        assert list(inputs.unit_index.labels[-1:]) == ["_musc_cohort__0__1"]
        _check(df)

    def test_row_order_no_longer_moves_the_pre_period(self):
        # The old setup read periods in row order, so a panel whose rows run
        # backwards in time put the post-period first. dataprep sorts them.
        df = _panel(shuffle=False).sort_values(["unit", "time"], ascending=[True, False])
        inputs = prepare_musc_inputs(df, unitid="unit", time="time", outcome="y",
                                     treated_unit=0, intervention_time=2008)
        assert list(inputs.time_index.labels) == list(range(2000, 2012))
        assert inputs.T0 == 8
        old = _old_prepare(df, "unit", "time", "y", 0, 2008)
        assert old["T0"] == 3 and old["time_labels"][0] == 2011

    def test_the_caller_frame_is_not_changed(self):
        df = _panel(adopt={0: 6, 1: 6})
        before = df.copy()
        _new(df, **COLS)
        prepare_musc_inputs(df, unitid="unit", time="time", outcome="y",
                            treated_unit=0, intervention_time=2006)
        pd.testing.assert_frame_equal(df, before)

    def test_prepare_musc_inputs_matches_the_old_setup(self):
        df = _panel(names="str", adopt={3: 9})
        ref = _old_prepare(df.sort_values(["unit", "time"]), "unit", "time", "y", "s03", 2009)
        inputs = prepare_musc_inputs(df, unitid="unit", time="time", outcome="y",
                                     treated_unit="s03", intervention_time=2009)
        _assert_same(inputs, ref)

    @pytest.mark.parametrize("adopt", [{0: 8}, {0: 8, 1: 8}, {0: 7, 1: 9}])
    def test_fit_smoke(self, adopt):
        res = MUSC(dict(df=_panel(adopt=adopt), **COLS, display_graphs=False,
                        run_inference=False)).fit()
        assert np.isfinite(res.att)


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2008), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            read_musc_panel(df, **COLS)

    def test_a_missing_outcome_in_a_collapsed_cohort_is_refused(self):
        # The old groupby mean skipped the NaN and averaged the rest.
        df = _panel(shuffle=False, adopt={0: 6, 1: 6})
        df.loc[(df.unit == 1) & (df.time == 2003), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            MUSC(dict(df=df, **COLS, display_graphs=False, run_inference=False)).fit()

    def test_a_non_finite_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            read_musc_panel(df, **COLS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 5) & (df.time == 2004))]
        with pytest.raises(MlsynthDataError, match="not strongly balanced"):
            read_musc_panel(df, **COLS)

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            read_musc_panel(pd.concat([df, df.iloc[[7]]]), **COLS)

    def test_a_non_binary_treatment_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 2
        with pytest.raises(MlsynthDataError, match="binary"):
            MUSC(dict(df=df, **COLS, display_graphs=False, run_inference=False)).fit()

    @pytest.mark.parametrize("adopt", [{0: 8}, {0: 7, 1: 9}])
    def test_a_treatment_that_switches_off_is_refused(self, adopt):
        # The old setup read the cohort from the first treated period and
        # treated every later period as post.
        df = _panel(shuffle=False, adopt=adopt)
        df.loc[(df.unit == 0) & (df.time == 2010), "d"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            MUSC(dict(df=df, **COLS, display_graphs=False, run_inference=False)).fit()

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_prepare_musc_inputs_refuses_a_bad_outcome(self, value):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = value
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_musc_inputs(df, unitid="unit", time="time", outcome="y",
                                treated_unit=0, intervention_time=2008)


# --------------------------------------------------------------------------- #
# MUSC's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def _wide(self, **kw):
        return read_musc_panel(_panel(**kw), **COLS)

    def test_an_unknown_treated_unit(self):
        with pytest.raises(MlsynthDataError, match="treated_unit 99 not found"):
            cohort_inputs(self._wide(), outcome="y", treated_units=(99,),
                          intervention_time=2008)

    def test_three_donors_are_needed(self):
        with pytest.raises(MlsynthDataError, match="at least 3 donor units"):
            cohort_inputs(self._wide(n_units=3), outcome="y", treated_units=(0,),
                          intervention_time=2008)

    def test_a_lone_unit_is_refused_by_musc(self):
        # dataprep is asked not to refuse a panel with no donor, so the
        # refusal is MUSC's own.
        with pytest.raises(MlsynthDataError, match="at least 3 donor units"):
            MUSC(dict(df=_panel(n_units=1), **COLS, display_graphs=False,
                      run_inference=False)).fit()

    def test_three_donors_are_enough(self):
        _check(_panel(n_units=4))

    def test_an_intervention_outside_the_panel(self):
        with pytest.raises(MlsynthDataError, match="not in time column"):
            cohort_inputs(self._wide(), outcome="y", treated_units=(0,),
                          intervention_time=9999)

    @pytest.mark.parametrize("adopt", [{0: 1}, {0: 1, 1: 1}])
    def test_one_pre_period_is_too_few(self, adopt):
        with pytest.raises(MlsynthDataError, match="at least 2 pre-treatment periods"):
            MUSC(dict(df=_panel(adopt=adopt), **COLS, display_graphs=False,
                      run_inference=False)).fit()

    def test_adoption_in_the_last_period_is_accepted(self):
        inputs = cohort_inputs(self._wide(), outcome="y", treated_units=(0,),
                               intervention_time=2011)
        assert inputs.T0 == inputs.T - 1

    def test_a_synthetic_label_that_collides_is_refused(self):
        df = _panel(names="str", adopt={0: 6, 1: 6})
        df.loc[df.unit == "s07", "unit"] = "_musc_cohort__s00__s01"
        wide = read_musc_panel(df, **COLS)
        with pytest.raises(MlsynthDataError, match="collides"):
            cohort_inputs(wide, outcome="y", treated_units=("s00", "s01"),
                          intervention_time=2006, collapse=True)


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
DEFECTS = ["none", "none", "none", "nan_y", "inf_y", "gap", "dup", "off",
           "nonbinary"]


@st.composite
def cases(draw):
    n_treated = draw(st.integers(1, 3))
    n_units = draw(st.integers(n_treated + 3, 9))
    T = draw(st.integers(4, 10))
    adopt = {u: draw(st.integers(2, T - 1)) for u in range(n_treated)}
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, adopt=adopt, seed=seed, names=names)
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
        u = draw(st.sampled_from(sorted(adopt)))
        if adopt[u] >= T - 1:
            defect = "none"
        else:
            t = draw(st.integers(adopt[u] + 1, T - 1))
            df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + t), "d"] = 0
    elif defect == "nonbinary":
        u = draw(st.integers(0, n_units - 1))
        t = draw(st.integers(0, T - 1))
        df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + t), "d"] = 2
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_chain_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            _new(df, **COLS)
        return
    _check(df)
    # Every cohort's inputs put the treated (or collapsed) unit outside the
    # donors, and the donors are never-treated units only.
    treated = set(df.loc[df.d == 1, "unit"])
    for inputs in _new(df, **COLS):
        labels = list(inputs.unit_index.labels)
        assert inputs.treated_idx not in set(inputs.donor_idx)
        assert not treated & {labels[j] for j in inputs.donor_idx}
