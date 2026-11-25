"""ROLLDID ingestion through ``datautils.dataprep`` (#1138, part 1).

ROLLDID used to pivot the outcome itself, read each unit's cohort from its
rows, and check the pivot for NaN cells only. It now takes the outcome matrix,
the time labels and each treated unit's adoption period from ``balance`` and
``dataprep``, so it gets the same refusals as every other estimator: a missing
or non-finite outcome anywhere in the panel, an absent or duplicate
unit-period, a non-binary treatment, and a treatment that switches off after
it starts. ROLLDID's own checks (named columns, a 0/1 treatment, a treated
unit, a never-treated unit) stay on top.

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

from mlsynth import ROLLDID
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.datautils import PanelKeys
from mlsynth.utils.rolldid_helpers import setup as SETUP
from mlsynth.utils.rolldid_helpers.setup import rolldid_setup

ARGS = ("unit", "time", "y", "w")


def _reference(df, unit_id, time_id, outcome, treat):
    """The pre-#1138 setup: pivot, then each unit's first treated row."""
    work = df[[unit_id, time_id, outcome, treat]].copy()
    Ywide = work.pivot(index=time_id, columns=unit_id, values=outcome).sort_index()
    cohort_of = {}
    for unit, sub in work.groupby(unit_id, observed=True):
        sub = sub.sort_values(time_id)
        on = sub[sub[treat] == 1]
        if not on.empty:
            cohort_of[unit] = on[time_id].iloc[0]
    treated = sorted(cohort_of, key=lambda u: str(u))
    never = [u for u in Ywide.columns if u not in cohort_of]
    cohorts = sorted(set(cohort_of.values()))
    return {"Ywide": Ywide, "cohort_of": cohort_of, "never": never,
            "treated": treated, "times": Ywide.index.to_numpy(),
            "cohorts": cohorts,
            "design": "common" if len(cohorts) == 1 else "staggered"}


def _panel(n_units=8, T=10, adopt=None, seed=0, names="int", shuffle=True,
           t0=2000):
    """Balanced long panel; ``adopt`` maps unit index -> adoption period index."""
    adopt = {0: 6} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            rows.append(dict(unit=u if names == "int" else f"s{u:02d}", time=t0 + t,
                             y=rng.normal() + 0.3 * u + 0.1 * t,
                             w=int(u in adopt and t >= adopt[u]), x=rng.normal()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _name(u, names="int"):
    return u if names == "int" else f"s{u:02d}"


def _assert_same(got, ref):
    assert list(got) == list(ref)
    pd.testing.assert_frame_equal(got["Ywide"], ref["Ywide"], check_exact=True)
    assert (got["Ywide"].to_numpy().flags["C_CONTIGUOUS"]
            == ref["Ywide"].to_numpy().flags["C_CONTIGUOUS"])
    np.testing.assert_array_equal(got["times"], ref["times"])
    assert got["times"].dtype == ref["times"].dtype
    # Cohort labels keep their type (NumPy integers, Timestamps, strings).
    assert list(got["cohort_of"].items()) == list(ref["cohort_of"].items())
    assert [type(g) for g in got["cohort_of"].values()] == \
        [type(g) for g in ref["cohort_of"].values()]
    assert got["cohorts"] == ref["cohorts"]
    assert [type(g) for g in got["cohorts"]] == [type(g) for g in ref["cohorts"]]
    for key in ("never", "treated"):
        assert got[key] == ref[key], key
        assert [type(u) for u in got[key]] == [type(u) for u in ref[key]], key
    assert got["design"] == ref["design"]


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
            calls.append((a[3], a[4], k.get("keys")))
            return real_dataprep(*a, **k)

        monkeypatch.setattr(SETUP, "balance", balance_spy)
        monkeypatch.setattr(SETUP, "dataprep", dataprep_spy)
        rolldid_setup(_panel(), *ARGS)
        assert len(scans) == 1 and isinstance(scans[0], PanelKeys)
        assert len(calls) == 1 and calls[0][:2] == ("y", "w")
        assert calls[0][2] is scans[0]

    def test_setup_does_not_pivot(self):
        # A runtime patch of DataFrame.pivot cannot test this: dataprep's
        # _wide_pivot falls back to df.pivot on some pandas versions.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called

    def test_the_estimator_scans_the_panel_once(self, monkeypatch):
        from mlsynth.utils import datautils

        scans = []
        real = PanelKeys.__init__

        def counting(self, *a, **k):
            scans.append(1)
            real(self, *a, **k)

        monkeypatch.setattr(datautils.PanelKeys, "__init__", counting)
        res = ROLLDID(dict(df=_panel(adopt={0: 6, 1: 6}), outcome="y", treat="w",
                           unitid="unit", time="time", display_graphs=False)).fit()
        assert np.isfinite(res.effects.att)
        assert len(scans) == 1, "the panel's keys were scanned more than once"


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw", [
        dict(),                                              # one treated unit
        dict(names="str", adopt={3: 8}),
        dict(shuffle=False, t0=-5, adopt={2: 9}),            # last-period adoption
        dict(adopt={0: 5, 1: 5}),                            # one cohort of two
        dict(adopt={0: 4, 1: 6, 4: 9}),                      # staggered, three cohorts
        dict(names="str", adopt={u: 3 + (u % 4) for u in range(6)}),
        dict(adopt={0: 0, 1: 5}),                            # a cohort in the first period
        dict(n_units=40, T=20, adopt={0: 12}, seed=3),                   # 40 units
        dict(n_units=40, T=20, names="str", seed=4,
             adopt={0: 10, 5: 10, 9: 14, 33: 17}),                       # 40, staggered
    ])
    def test_inputs_match_the_old_pivot(self, kw):
        df = _panel(**kw)
        _assert_same(rolldid_setup(df, *ARGS), _reference(df, *ARGS))

    @pytest.mark.parametrize("label", ["datetime", "str", "float"])
    def test_period_labels_keep_their_type(self, label):
        df = _panel(adopt={1: 7, 2: 5})
        step = df["time"] - 2000
        df["time"] = {"datetime": pd.to_datetime("2001-01-01") + pd.to_timedelta(31 * step, unit="D"),
                      "str": "p" + step.map("{:02d}".format),
                      "float": step / 4.0}[label]
        _assert_same(rolldid_setup(df, *ARGS), _reference(df, *ARGS))

    def test_a_boolean_treatment_matches_the_old_pivot(self):
        df = _panel(adopt={0: 4, 1: 6})
        df["w"] = df["w"].astype(bool)
        _assert_same(rolldid_setup(df, *ARGS), _reference(df, *ARGS))

    def test_a_missing_treatment_before_adoption_is_read_as_untreated_as_before(self):
        df = _panel(adopt={0: 6, 1: 4})
        df["w"] = df["w"].astype(float)
        df.loc[(df.unit == 3) & (df.time == 2002), "w"] = np.nan
        df.loc[(df.unit == 0) & (df.time == 2001), "w"] = np.nan
        _assert_same(rolldid_setup(df, *ARGS), _reference(df, *ARGS))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        rolldid_setup(df, *ARGS)
        pd.testing.assert_frame_equal(df, before)

    @pytest.mark.parametrize("adopt", [{0: 6}, {0: 5, 1: 7}])
    def test_fit_smoke(self, adopt):
        res = ROLLDID(dict(df=_panel(adopt=adopt), outcome="y", treat="w",
                           unitid="unit", time="time", display_graphs=False)).fit()
        assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# refusals now come from balance and dataprep
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.time == 2008), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            rolldid_setup(df, *ARGS)

    def test_a_non_finite_outcome_is_refused(self):
        # The old NaN check let an infinite outcome through.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            rolldid_setup(df, *ARGS)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 5) & (df.time == 2004))]
        with pytest.raises(MlsynthDataError, match="not strongly balanced"):
            rolldid_setup(df, *ARGS)

    def test_a_duplicate_unit_period_is_refused(self):
        # The old setup's pivot raised pandas' ValueError here.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            rolldid_setup(pd.concat([df, df.iloc[[7]]]), *ARGS)

    def test_the_estimator_reports_a_duplicate(self):
        # The old ValueError surfaced from fit() as MlsynthEstimationError.
        df = _panel(shuffle=False)
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            ROLLDID(dict(df=pd.concat([df, df.iloc[[7]]]), outcome="y", treat="w",
                         unitid="unit", time="time", display_graphs=False)).fit()

    @pytest.mark.parametrize("adopt", [{0: 6}, {0: 5, 1: 6}])
    def test_a_treatment_that_switches_off_is_refused(self, adopt):
        df = _panel(shuffle=False, adopt=adopt)
        df.loc[(df.unit == 0) & (df.time == 2008), "w"] = 0
        with pytest.raises(MlsynthDataError, match="not sustained"):
            rolldid_setup(df, *ARGS)

    def test_a_missing_treatment_after_adoption_is_refused(self):
        df = _panel(shuffle=False)
        df["w"] = df["w"].astype(float)
        df.loc[(df.unit == 0) & (df.time == 2008), "w"] = np.nan
        with pytest.raises(MlsynthDataError, match="not sustained"):
            rolldid_setup(df, *ARGS)

    def test_one_treated_unit_in_the_first_period_is_refused(self):
        # dataprep's refusal; the old setup passed a cohort with no pre-period on.
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            rolldid_setup(_panel(adopt={0: 0}), *ARGS)


# --------------------------------------------------------------------------- #
# ROLLDID's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    @pytest.mark.parametrize("col", ["unit", "time", "y", "w"])
    def test_a_missing_column_is_named(self, col):
        with pytest.raises(MlsynthConfigError, match=f"column '{col}' not found"):
            rolldid_setup(_panel().drop(columns=col), *ARGS)

    def test_a_non_binary_treatment_keeps_its_message(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 0) & (df.time == 2009), "w"] = 2
        with pytest.raises(MlsynthDataError, match="must be 0/1"):
            rolldid_setup(df, *ARGS)

    def test_no_treated_unit(self):
        with pytest.raises(MlsynthDataError, match="no treated units found"):
            rolldid_setup(_panel(adopt={}), *ARGS)

    @pytest.mark.parametrize("adopt", [{u: 5 for u in range(4)},
                                       {0: 3, 1: 5, 2: 5, 3: 7}])
    def test_no_never_treated_unit(self, adopt):
        with pytest.raises(MlsynthDataError, match="never-treated"):
            rolldid_setup(_panel(n_units=4, adopt=adopt), *ARGS)

    def test_a_lone_treated_unit_has_no_never_treated_comparison(self):
        df = _panel(n_units=1)
        with pytest.raises(MlsynthDataError, match="never-treated"):
            rolldid_setup(df, *ARGS)


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
DEFECTS = ["none", "none", "none", "nan_y", "inf_y", "gap", "dup", "off",
           "nonbinary"]


@st.composite
def cases(draw):
    n_units = draw(st.integers(2, 8))
    T = draw(st.integers(3, 9))
    n_treated = draw(st.integers(1, n_units - 1))
    adopt = {u: draw(st.integers(1, T - 1)) for u in range(n_treated)}
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
            df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + t), "w"] = 0
    elif defect == "nonbinary":
        u = draw(st.integers(0, n_units - 1))
        t = draw(st.integers(0, T - 1))
        df.loc[(df.unit == _name(u, names)) & (df.time == 2000 + t), "w"] = 2
    return df, defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_dataprep_path_matches_the_old_pivot_or_refuses(case):
    df, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            rolldid_setup(df, *ARGS)
        return
    got = rolldid_setup(df, *ARGS)
    _assert_same(got, _reference(df, *ARGS))
    # Each treated unit's cohort is its first treated period, and the treated
    # and never-treated units partition the panel's units.
    for unit, g in got["cohort_of"].items():
        rows = df[df.unit == unit]
        assert g == rows.loc[rows.w == 1, "time"].min()
    assert sorted(got["never"] + got["treated"], key=str) == \
        sorted(df["unit"].unique().tolist(), key=str)
