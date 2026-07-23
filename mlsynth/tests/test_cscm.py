"""Tests for CSCM -- flexible synthetic control for count/non-negative outcomes.

Written test-first. Invariants are pinned against the Bonander (2021) Vision
Zero application, cross-validated against the authors' R implementation
(ground truth staged in the benchmark): classic SCM concentrates on Germany,
CSCM relaxes below the simplex (sum of weights < 1) while keeping the
counterfactual non-negative, and the reported effect is a rate ratio with a
cross-fitted t-interval. Floats are asserted as invariants/bands, not brittle
equalities (glmnet-vs-sklearn V differs at the ~1% level).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import CSCM
from mlsynth.config_models import BaseEstimatorResults, CSCMConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _viszero(ids=(25, 2, 3, 5, 7, 9, 10, 13, 14, 16), treat_year=1996):
    """Vision Zero panel filtered to the treated (Sweden=25) + 9 donors."""
    df = pd.read_csv("basedata/viszero.csv")
    df = df[df["ID"].isin(ids)].copy()
    df["treated"] = ((df["ID"] == 25) & (df["TIME"] >= treat_year)).astype(int)
    return df


def _cfg(**over):
    # uniform V by default: fast, deterministic, and on Vision Zero glmnet's
    # Poisson-ridge V collapses to ~uniform anyway (see test_uniform_V_*).
    base = dict(df=_viszero(), outcome="deathrate_mln", treat="treated",
                unitid="ID", time="TIME", display_graphs=False, K=2,
                v_method="uniform")
    base.update(over)
    return base


def _tiny_count_panel(n_units=6, T=12, t0=8, seed=0):
    """Small non-negative (count-like) panel for smoke/edge tests."""
    rng = np.random.default_rng(seed)
    rows = []
    base = rng.uniform(2, 8, n_units)
    for u in range(n_units):
        for t in range(T):
            y = max(0.0, base[u] + 0.1 * t + rng.normal(0, 0.3))
            treated = int(u == 0 and t >= t0)
            rows.append((f"u{u}", t, y + (1.5 if (u == 0 and t >= t0) else 0.0), treated))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "d"])


# --------------------------------------------------------------------------- #
# Reproduction: Vision Zero invariants vs the R ground truth
# --------------------------------------------------------------------------- #

def test_returns_effect_result_surface():
    res = CSCM(_cfg()).fit()
    assert isinstance(res, BaseEstimatorResults)
    assert res.att is not None
    assert res.counterfactual is not None
    assert res.weights is not None and res.weights.donor_weights is not None
    # rate ratio is the CSCM native estimand
    ae = res.effects.additional_effects
    assert ae is not None and "rate_ratio" in ae


def test_scm_warmstart_concentrates_on_finland():
    # classic SCM (the warm-start) puts ~all weight on Finland (ID 9) here,
    # matching the R ground truth w_scm = [0,0,0,0,1,0,0,0,0]
    res = CSCM(_cfg()).fit()
    scm = res.additional_outputs["scm_weights"]        # {donor id (str): w}
    assert scm["9"] > 0.95
    assert abs(sum(scm.values()) - 1.0) < 1e-6         # simplex


def test_cscm_relaxes_below_simplex_but_nonnegative():
    res = CSCM(_cfg()).fit()
    w = res.weights.weight_vector
    assert np.all(w >= -1e-9)                           # non-negativity kept
    assert w.sum() < 0.95                               # adding-up dropped (extrapolates)
    # counterfactual stays non-negative (the whole point for counts)
    assert np.all(res.counterfactual >= 0)


def test_rate_ratio_and_crossfit_ci():
    res = CSCM(_cfg()).fit()
    rr = res.effects.additional_effects["rate_ratio"]
    assert 0.85 < rr < 1.25                             # no large effect on Sweden
    lo, hi = res.inference.ci_lower, res.inference.ci_upper
    assert lo < rr < hi                                 # CI brackets the point
    assert lo < 1.0 < hi                                # spans the null (RR=1)
    assert res.inference.method == "crossfit_rate_ratio"


def test_uniform_V_close_to_poisson_on_viszero():
    # on Vision Zero glmnet's V collapses to ~uniform, so the two agree closely
    rr_p = CSCM(_cfg(v_method="poisson_ridge")).fit().effects.additional_effects["rate_ratio"]
    rr_u = CSCM(_cfg(v_method="uniform")).fit().effects.additional_effects["rate_ratio"]
    assert abs(rr_p - rr_u) < 0.05


# --------------------------------------------------------------------------- #
# Config validation (failure paths, reported not swallowed)
# --------------------------------------------------------------------------- #

def test_negative_outcome_rejected():
    df = _viszero()
    df.loc[df.index[0], "deathrate_mln"] = -1.0
    with pytest.raises(MlsynthDataError):
        CSCM(_cfg(df=df))


def test_K_below_two_rejected():
    with pytest.raises((MlsynthConfigError, ValueError, Exception)):
        CSCM(_cfg(K=1))


def test_extra_field_forbidden():
    with pytest.raises((MlsynthConfigError, Exception)):
        CSCM(_cfg(nonsense=3))


def test_missing_column_rejected():
    with pytest.raises((MlsynthDataError, Exception)):
        CSCM(_cfg(outcome="not_a_column"))


# --------------------------------------------------------------------------- #
# Edge cases + smoke
# --------------------------------------------------------------------------- #

def test_smoke_tiny_panel():
    cfg = dict(df=_tiny_count_panel(), outcome="y", treat="d",
               unitid="unit", time="time", display_graphs=False, K=2)
    res = CSCM(cfg).fit()
    assert np.isfinite(res.att)
    assert res.counterfactual.shape[0] == 12


def test_single_donor():
    df = _tiny_count_panel(n_units=2)               # treated + 1 donor
    cfg = dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
               display_graphs=False, K=2)
    res = CSCM(cfg).fit()
    assert np.isfinite(res.att)


def test_plot_smoke():
    import matplotlib
    matplotlib.use("Agg")
    res = CSCM(_cfg(display_graphs=True)).fit()       # exercises the plotter path
    assert res is not None


def test_insufficient_pre_periods_reported():
    from mlsynth.exceptions import MlsynthEstimationError
    # treat unit 0 from t=1 -> a single pre-period, below CSCM's minimum
    df = _tiny_count_panel(n_units=5, T=10, t0=1)
    cfg = dict(df=df, outcome="y", treat="d", unitid="unit", time="time",
               display_graphs=False, K=2, v_method="uniform")
    with pytest.raises(MlsynthEstimationError):
        CSCM(cfg).fit()
