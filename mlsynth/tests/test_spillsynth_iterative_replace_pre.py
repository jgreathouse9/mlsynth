"""The iterative SCM's pre-period replacement switch (Melnychuk 2024).

``runIterativeSCM(dataprepDf, replacePreTreatData, waterfallMode)`` has two
switches. The signature defaults ``replacePreTreatData`` to ``FALSE``, but every
call site in the reference -- ``runAllMethods`` and ``runAllMethodsWithCov``,
which are what produce the paper's tables -- passes ``TRUE``. The two branches
are different estimators:

``False``
    A cleaned donor keeps its observed pre-period, so the treated unit's refit
    sees bit-identical fitting data and returns the naive weights. The
    correction reaches the estimate only through the cleaned post-period
    counterfactual, and the "refit" is a no-op on the weights.

``True``
    The cleaned donor's whole series is replaced by its spillover-free
    synthetic. Its pre-period is then an exact convex combination of donors
    already in the pool, so the refit has no reason to load on it and the
    weights move.

These tests pin the distinction. Before ``iterative_replace_pre`` existed the
suite could not tell the branches apart: substituting ``Y[i, :] = cf`` for
``Y[i, T0:] = cf[T0:]`` left all 252 spillsynth tests green.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.config_models import SPILLSYNTHConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "repgermany.dta"


@pytest.fixture(scope="module")
def german_panel():
    if not _DATA.exists():
        pytest.skip("repgermany.dta not available")
    d = pd.read_stata(_DATA)[["country", "year", "gdp"]].copy()
    d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    return d


def _cfg(d, **kw):
    base = dict(df=d, outcome="gdp", treat="treat", unitid="country",
                time="year", method="iterative", affected_units=["Austria"],
                iscm_intercept=True, display_graphs=False)
    base.update(kw)
    return base


def _synthetic_panel(seed=0, T=20, T0=14, spillover=6.0, effect=-5.0):
    """Treated + one affected donor + two clean donors on a common factor."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 1, T)) + 50.0
    series = {
        "T": f + rng.normal(0, 0.1, T),
        "A": f + rng.normal(0, 0.1, T),
        "c1": f + rng.normal(0, 0.1, T),
        "c2": 0.5 * f + rng.normal(0, 0.1, T) + 10,
    }
    series["T"][T0:] += effect
    series["A"][T0:] += spillover
    rows = [{"unit": u, "time": t, "y": s[t], "d": int(u == "T" and t >= T0)}
            for u, s in series.items() for t in range(T)]
    return pd.DataFrame(rows)


def _syn_cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
                method="iterative", affected_units=["A"], iscm_intercept=True,
                display_graphs=False)
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def test_replace_pre_runs_end_to_end():
    res = SPILLSYNTH(_syn_cfg(_synthetic_panel(), iterative_replace_pre=True)).fit()
    assert np.isfinite(res.att)
    assert res.iterative.replace_pre is True
    assert res.iterative.cleaned_units == ["A"]


# --------------------------------------------------------------------------
# the invariant that separates the branches
# --------------------------------------------------------------------------

def test_default_is_post_only_and_leaves_the_refit_weights_alone():
    """Under the default the refit is a no-op on the weights.

    The treated unit's fitting data in the pre-period is untouched, so its
    pre-period synthetic on the cleaned pool must equal the one the naive fit
    produces -- exactly, not approximately.
    """
    df = _synthetic_panel()
    res = SPILLSYNTH(_syn_cfg(df)).fit()
    f = res.iterative
    assert f.replace_pre is False
    assert np.allclose(f.treated_synthetic_pre, f.naive_synthetic_pre, atol=1e-10)


def test_replace_pre_moves_the_refit_weights():
    """Under the reference's branch the refit is a real refit.

    Replacing the affected donor's pre-period changes the treated unit's
    fitting data, so the pre-period synthetic separates from the naive one and
    the donor weights are not the naive weights.
    """
    df = _synthetic_panel()
    post_only = SPILLSYNTH(_syn_cfg(df)).fit().iterative
    whole = SPILLSYNTH(_syn_cfg(df, iterative_replace_pre=True)).fit().iterative
    assert not np.allclose(whole.treated_synthetic_pre, whole.naive_synthetic_pre)
    assert whole.donor_weights != post_only.donor_weights


def test_replace_pre_makes_the_cleaned_donor_redundant():
    """The cleaned donor becomes a convex combination of the pool it was built
    from, so the refit stops loading on it."""
    df = _synthetic_panel()
    post_only = SPILLSYNTH(_syn_cfg(df)).fit().iterative
    whole = SPILLSYNTH(_syn_cfg(df, iterative_replace_pre=True)).fit().iterative
    assert whole.donor_weights.get("A", 0.0) < post_only.donor_weights.get("A", 0.0)


def test_replace_pre_rewrites_the_affected_donor_pre_period(german_panel):
    """The cleaned pre-period is the affected unit's own synthetic, so it lies
    in the convex hull of the clean donors and differs from what was observed."""
    res = SPILLSYNTH(_cfg(german_panel, iterative_replace_pre=True)).fit()
    f = res.iterative
    cleaned_pre = f.spillover_panel_pre["Austria"]
    observed_pre = res.inputs.Y[1, : res.inputs.T0]
    assert cleaned_pre.shape == observed_pre.shape
    assert not np.allclose(cleaned_pre, observed_pre)


def test_post_only_leaves_no_cleaned_pre_period():
    res = SPILLSYNTH(_syn_cfg(_synthetic_panel())).fit()
    assert res.iterative.spillover_panel_pre == {}


# --------------------------------------------------------------------------
# the two branches disagree on the estimate, and by how much
# --------------------------------------------------------------------------

def test_the_two_branches_give_different_estimates(german_panel):
    post_only = SPILLSYNTH(_cfg(german_panel)).fit()
    whole = SPILLSYNTH(_cfg(german_panel, iterative_replace_pre=True)).fit()
    assert abs(whole.att - post_only.att) > 100.0


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------

def test_replace_pre_with_a_single_clean_donor():
    """One clean donor: the cleaned series is that donor's level-shifted path,
    and the estimate stays finite."""
    rng = np.random.default_rng(3)
    T, T0 = 16, 12
    f = np.cumsum(rng.normal(0, 1, T)) + 40.0
    series = {"T": f + rng.normal(0, 0.1, T),
              "A": f + rng.normal(0, 0.1, T),
              "c1": f + rng.normal(0, 0.1, T)}
    series["T"][T0:] += -3.0
    series["A"][T0:] += 4.0
    rows = [{"unit": u, "time": t, "y": s[t], "d": int(u == "T" and t >= T0)}
            for u, s in series.items() for t in range(T)]
    res = SPILLSYNTH(_syn_cfg(pd.DataFrame(rows), iterative_replace_pre=True)).fit()
    assert np.isfinite(res.att)
    assert res.iterative.n_clean == 1


def test_replace_pre_is_inert_without_affected_units():
    """With nothing to clean the switch cannot change the answer."""
    df = _synthetic_panel()
    cfg = dict(_syn_cfg(df))
    cfg["affected_units"] = []
    a = SPILLSYNTH(dict(cfg)).fit()
    b = SPILLSYNTH(dict(cfg, iterative_replace_pre=True)).fit()
    assert a.att == pytest.approx(b.att)


def test_replace_pre_survives_a_covariate_backend(german_panel):
    """The switch replaces outcomes; the predictor block is untouched, so the
    covariate backend still runs and the two branches still differ."""
    d = pd.read_stata(_DATA)[["country", "year", "gdp", "trade", "infrate"]].copy()
    d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    cfg = dict(df=d, outcome="gdp", treat="treat", unitid="country", time="year",
               method="iterative", affected_units=["Austria"],
               covariates=["trade", "infrate"], bilevel_solver="malo",
               display_graphs=False)
    a = SPILLSYNTH(dict(cfg)).fit()
    b = SPILLSYNTH(dict(cfg, iterative_replace_pre=True)).fit()
    assert np.isfinite(a.att) and np.isfinite(b.att)
    assert a.att != b.att


# --------------------------------------------------------------------------
# the cleaning pool must itself be clean
# --------------------------------------------------------------------------

def _all_controls_affected_panel(seed=1, T=16, T0=12):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 1, T)) + 40.0
    series = {u: f + rng.normal(0, 0.1, T) for u in ("T", "A1", "A2")}
    series["T"][T0:] += -3.0
    series["A1"][T0:] += 4.0
    series["A2"][T0:] += 4.0
    rows = [{"unit": u, "time": t, "y": s[t], "d": int(u == "T" and t >= T0)}
            for u, s in series.items() for t in range(T)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("replace_pre", [False, True])
def test_no_clean_controls_is_rejected(replace_pre):
    """Every control declared affected leaves nothing spillover-free to learn
    from, so the waterfall has no first step and the fit must fail."""
    df = _all_controls_affected_panel()
    est = SPILLSYNTH(_syn_cfg(df, affected_units=["A1", "A2"],
                              iterative_replace_pre=replace_pre))
    with pytest.raises(MlsynthDataError, match="clean control"):
        est.fit()


def test_every_reported_cleaned_unit_was_actually_cleaned():
    """``cleaned_units`` and ``spillover_att`` describe the same set.

    A unit listed as cleaned but absent from the spillover ledger was passed
    over, and any later unit that borrowed from it was fit against contaminated
    outcomes.
    """
    rng = np.random.default_rng(7)
    T, T0 = 18, 13
    f = np.cumsum(rng.normal(0, 1, T)) + 45.0
    series = {u: f + rng.normal(0, 0.1, T)
              for u in ("T", "A1", "A2", "c1", "c2")}
    series["T"][T0:] += -4.0
    series["A1"][T0:] += 5.0
    series["A2"][T0:] += 3.0
    rows = [{"unit": u, "time": t, "y": s[t], "d": int(u == "T" and t >= T0)}
            for u, s in series.items() for t in range(T)]
    res = SPILLSYNTH(_syn_cfg(pd.DataFrame(rows),
                              affected_units=["A1", "A2"])).fit()
    f_ = res.iterative
    assert set(f_.cleaned_units) == set(f_.spillover_att)
    assert f_.n_clean == 2


# --------------------------------------------------------------------------
# failure / config contract
# --------------------------------------------------------------------------

def test_default_is_false():
    assert SPILLSYNTHConfig.model_fields["iterative_replace_pre"].default is False


def test_non_boolean_replace_pre_is_rejected():
    df = _synthetic_panel()
    with pytest.raises(MlsynthConfigError):
        SPILLSYNTH(_syn_cfg(df, iterative_replace_pre="waterfall"))


def test_dict_and_config_agree():
    df = _synthetic_panel()
    r1 = SPILLSYNTH(_syn_cfg(df, iterative_replace_pre=True)).fit()
    r2 = SPILLSYNTH(SPILLSYNTHConfig(**_syn_cfg(df, iterative_replace_pre=True))).fit()
    assert r1.att == pytest.approx(r2.att)
