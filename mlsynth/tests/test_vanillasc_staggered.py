"""Staggered-adoption VanillaSC, validated against the official ``scpi`` package.

Reference: Cattaneo, Feng, Palomba and Titiunik's multiple-treated-unit
illustration (``scpi_illustration-multi.py``) on the Germany reunification panel
with two staggered treated units -- West Germany (treated 1991) and Italy (treated
1992) -- and the 15 never-treated countries as donors. Run outcome-only with a
simplex constraint, ``scpi``'s ``scest`` reports the per-unit and unit-time
average treatment effects pinned below. mlsynth's staggered VanillaSC must
reproduce them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "scpi_germany.csv")

# scpi reference (outcome-only, simplex), from scest on the two-treated panel.
_SCPI = {
    "West Germany": -1.8476,
    "Italy": -1.1211,
    "overall_unit_time": -1.4989,
}


@pytest.fixture
def germany_staggered():
    df = pd.read_csv(os.path.abspath(_DATA))
    df["status"] = 0
    df.loc[(df["country"] == "West Germany") & (df["year"] >= 1991), "status"] = 1
    df.loc[(df["country"] == "Italy") & (df["year"] >= 1992), "status"] = 1
    return df[["country", "year", "gdp", "status"]]


def _fit(df):
    return VanillaSC({
        "df": df, "outcome": "gdp", "treat": "status",
        "unitid": "country", "time": "year", "display_graphs": False,
    }).fit()


def test_overall_att_matches_scpi(germany_staggered):
    res = _fit(germany_staggered)
    assert res.effects.att == pytest.approx(_SCPI["overall_unit_time"], abs=0.05)


def test_per_unit_att_matches_scpi(germany_staggered):
    res = _fit(germany_staggered)
    by_unit = {c.treated_unit_name: float(c.att)
               for c in res.sub_method_results.values()}
    assert by_unit["West Germany"] == pytest.approx(_SCPI["West Germany"], abs=0.05)
    assert by_unit["Italy"] == pytest.approx(_SCPI["Italy"], abs=0.05)


def test_donor_pool_is_never_treated(germany_staggered):
    """Each treated unit's synthetic control draws on the 15 never-treated
    countries, never on the other treated unit."""
    res = _fit(germany_staggered)
    for cohort in res.sub_method_results.values():
        donors = set(map(str, cohort.donor_names))
        assert "West Germany" not in donors
        assert "Italy" not in donors
        assert len(donors) == 15


def _synthetic_staggered(adopt, N=16, T=20, r=3, tau=3.0, seed=0):
    """Factor-model panel: ``adopt`` maps unit index -> adoption period; the rest
    are never-treated donors."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, r))
    Lam = rng.standard_normal((N, r))
    eps = rng.standard_normal((N, T)) * 0.3
    Y = Lam @ F.T + eps
    rows = []
    for i in range(N):
        Ti = adopt.get(i)
        for t in range(T):
            treated = int(Ti is not None and t >= Ti)
            rows.append({"id": f"u{i}", "time": t,
                         "y": float(Y[i, t] + (tau if treated else 0.0)),
                         "D": treated})
    return pd.DataFrame(rows)


def _fit_synth(df):
    return VanillaSC({"df": df, "outcome": "y", "treat": "D",
                      "unitid": "id", "time": "time",
                      "display_graphs": False}).fit()


def test_two_treated_same_cohort_smoke():
    """Two units adopting at the same time form one cohort; both are fit."""
    res = _fit_synth(_synthetic_staggered({0: 12, 1: 12}))
    assert len(res.sub_method_results) == 2
    assert set(res.sub_method_results) == {"u0", "u1"}
    # planted effect recovered to the ballpark
    assert res.effects.att == pytest.approx(3.0, abs=0.6)


def test_overall_att_is_postweighted_mean_of_units():
    res = _fit_synth(_synthetic_staggered({0: 10, 1: 13, 2: 16}))
    fits = list(res.sub_method_results.values())
    num = sum(f.att * f.post_periods for f in fits)
    den = sum(f.post_periods for f in fits)
    assert res.effects.att == pytest.approx(num / den, abs=1e-9)


def test_per_unit_weights_on_simplex():
    res = _fit_synth(_synthetic_staggered({0: 12, 1: 14}))
    for fit in res.sub_method_results.values():
        w = np.array(list(fit.donor_weights.values()), dtype=float)
        assert (w >= -1e-8).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Phase 2: per-unit CFT prediction intervals + aggregated ATT interval
# ---------------------------------------------------------------------------

def _fit_scpi(df, **kw):
    cfg = {"df": df, "outcome": "gdp", "treat": "status", "unitid": "country",
           "time": "year", "display_graphs": False, "inference": "scpi",
           "alpha": 0.1}
    cfg.update(kw)
    return VanillaSC(cfg).fit()


def test_staggered_scpi_per_unit_bands(germany_staggered):
    res = _fit_scpi(germany_staggered)
    for fit in res.sub_method_results.values():
        n = fit.post_periods
        assert fit.tau_lower is not None and len(fit.tau_lower) == n
        assert len(fit.tau_upper) == n
        assert np.all(np.asarray(fit.tau_lower) <= np.asarray(fit.tau_upper) + 1e-9)
        # the unit's average-effect interval brackets its point ATT
        assert fit.att_ci_lower <= fit.att <= fit.att_ci_upper


def test_staggered_scpi_counterfactual_bracket(germany_staggered):
    res = _fit_scpi(germany_staggered)
    for fit in res.sub_method_results.values():
        cf = np.asarray(fit.counterfactual)[fit.pre_periods:]
        lo = np.asarray(fit.cf_lower)
        hi = np.asarray(fit.cf_upper)
        assert np.all(lo <= cf + 1e-9) and np.all(cf <= hi + 1e-9)


def test_staggered_scpi_consistency_with_single(germany_staggered):
    """A unit's staggered per-period band equals the band obtained by fitting
    that unit alone on the same never-treated donors with the same seed."""
    res = _fit_scpi(germany_staggered)
    wg = res.sub_method_results["West Germany"]
    never = [c for c in germany_staggered.country.unique()
             if c not in ("West Germany", "Italy")]
    sub = germany_staggered[
        germany_staggered.country.isin(["West Germany"] + never)].copy()
    sub["status"] = ((sub.country == "West Germany") & (sub.year >= 1991)).astype(int)
    single = VanillaSC({"df": sub, "outcome": "gdp", "treat": "status",
                        "unitid": "country", "time": "year",
                        "display_graphs": False, "inference": "scpi",
                        "alpha": 0.1}).fit()
    sd = single.inference.details
    np.testing.assert_allclose(wg.tau_lower, sd["pi_lower"], atol=1e-9)
    np.testing.assert_allclose(wg.tau_upper, sd["pi_upper"], atol=1e-9)


def test_overall_att_pi_brackets_point(germany_staggered):
    res = _fit_scpi(germany_staggered)
    assert res.inference is not None
    assert res.inference.ci_lower <= res.effects.att <= res.inference.ci_upper


# ---------------------------------------------------------------------------
# Phase 3: aggregated predictands (event study), validated against scpi
# ---------------------------------------------------------------------------

# scpi effect="time" on the Germany two-treated panel (outcome-only simplex),
# balanced over event times 1..12 where both treated units are observed.
_SCPI_EVENT_TIME = [0.131, 0.0316, -0.4255, -0.67, -0.9013, -1.2304,
                    -1.2529, -1.6015, -2.2758, -2.6734, -2.8457, -3.0832]


def test_event_study_matches_scpi(germany_staggered):
    res = _fit(germany_staggered)
    es = res.additional_outputs["event_study"]
    vals = [es[k] for k in sorted(es)]
    np.testing.assert_allclose(vals, _SCPI_EVENT_TIME, atol=2e-3)


def test_event_study_is_balanced_event_times(germany_staggered):
    res = _fit(germany_staggered)
    es = res.additional_outputs["event_study"]
    min_post = min(f.post_periods for f in res.sub_method_results.values())
    assert sorted(es) == list(range(1, min_post + 1))   # 1..min post


def test_event_study_synthetic_recovers_planted_path():
    """On a clean factor panel the event-study effect tracks the planted +3."""
    res = _fit_synth(_synthetic_staggered({0: 10, 1: 13, 2: 16}))
    es = res.additional_outputs["event_study"]
    assert np.mean([es[k] for k in es]) == pytest.approx(3.0, abs=0.6)


# ---------------------------------------------------------------------------
# Section 4: cross-unit (TSUA) prediction intervals via the clean-room engine
# ---------------------------------------------------------------------------
def test_event_study_intervals_populated(germany_staggered):
    """The TSUA prediction intervals are exposed per event time, bracket the
    effect, and the full band contains the in-sample-only band."""
    res = _fit_scpi(germany_staggered, scpi_sims=120)
    esi = res.additional_outputs["event_study_intervals"]
    es = res.additional_outputs["event_study"]
    assert esi is not None and set(esi) == set(es)
    for ell, d in esi.items():
        lo, hi = d["effect_ci"]
        assert lo <= d["effect"] <= hi
        s_lo, s_hi = d["synthetic_ci"]
        i_lo, i_hi = d["insample_synthetic_ci"]
        # the full band adds the out-of-sample term, so it is at least as wide
        # as the in-sample-only band (it may shift via the out-of-sample mean).
        assert (s_hi - s_lo) >= (i_hi - i_lo) - 1e-9


def test_single_treated_unit_has_no_event_study_intervals(germany_staggered):
    """With one treated unit the cross-unit predictand is undefined; the scalar
    path runs and no ``event_study_intervals`` entry is produced."""
    df = germany_staggered.copy()
    df.loc[df["country"] == "Italy", "status"] = 0    # only West Germany treated
    res = _fit_scpi(df, scpi_sims=60)
    assert res.additional_outputs.get("event_study_intervals") is None


# ---------------------------------------------------------------------------
# Covariate (multi-feature) staggered matching -- scpi's multi illustration
# ---------------------------------------------------------------------------
def test_covariate_staggered_matches_scpi_per_unit():
    """With a shared multi-feature spec (gdp+trade, constant+trend, cointegrated)
    the per-unit average effects reproduce scpi's scest to the published digits.
    The spec never names treated units -- they come from the treatment column."""
    df = pd.read_csv(os.path.abspath(_DATA))     # full panel (trade, infrate, ...)
    df["status"] = 0
    df.loc[(df["country"] == "West Germany") & (df["year"] >= 1991), "status"] = 1
    df.loc[(df["country"] == "Italy") & (df["year"] >= 1992), "status"] = 1
    res = VanillaSC({
        "df": df, "outcome": "gdp", "treat": "status",
        "unitid": "country", "time": "year", "display_graphs": False,
        "inference": "scpi", "scpi_sims": 80, "seed": 8894, "scpi_compat": True,
        "staggered_spec": {"features": ["gdp", "trade"],
                           "cov_adj": ["constant", "trend"],
                           "constant": True, "cointegrated": True},
    }).fit()
    pu = res.additional_outputs["per_unit_att"]
    assert pu["Italy"] == pytest.approx(-0.8902, abs=2e-3)        # scpi scest
    assert pu["West Germany"] == pytest.approx(-1.7467, abs=2e-3)
    # full predictand set is populated, with intervals
    esi = res.additional_outputs["event_study_intervals"]
    assert esi and all("synthetic_ci" in d and "effect_ci" in d for d in esi.values())
    assert res.inference is not None
    assert np.isfinite(res.effects.att)
    assert len(res.additional_outputs["per_cell_effects"]) == 25
