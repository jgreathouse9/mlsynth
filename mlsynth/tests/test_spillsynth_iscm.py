"""Tests for the inclusive SCM method of SPILLSYNTH (Di Stefano & Mellace 2024).

* Layer 1 (numerical helpers): the cross-weight system ``Omega`` builder and
  its de-contaminating inverse.
* Layer 3 (estimator integration): end-to-end on the German reunification
  panel (West Germany treated, Austria the affected neighbour), outcome-only
  and covariate matching with both bilevel backends.
* Layer 4 (public API): the ``method='iscm'`` dispatcher, dict-vs-config
  equivalence, and config validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.config_models import SPILLSYNTHConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.spillsynth_helpers import (
    ISCMFit, build_omega, run_iscm, solve_inclusive,
)
from mlsynth.utils.spillsynth_helpers.setup import prepare_spillsynth_inputs


# --------------------------------------------------------------------------- #
# Layer 1: the Omega cross-weight system
# --------------------------------------------------------------------------- #
def test_build_omega_unit_diagonal_negative_off():
    cross = np.array([[0.0, 0.4], [0.3, 0.0]])
    omega = build_omega(cross)
    np.testing.assert_allclose(omega, [[1.0, -0.4], [-0.3, 1.0]])


def test_solve_inclusive_inverts_the_system():
    omega = np.array([[1.0, -0.42], [-0.33, 1.0]])
    theta_true = np.array([[-1000.0, -1200.0], [200.0, 150.0]])  # (m, T1)
    gaps = omega @ theta_true
    theta = solve_inclusive(omega, gaps)
    np.testing.assert_allclose(theta, theta_true, atol=1e-9)


def test_solve_inclusive_matches_m1_closed_form():
    # m=2 closed form: theta = (gap_1 + w*gap_2) / (1 - w*l), etc.
    w, l = 0.42, 0.33
    omega = build_omega(np.array([[0.0, w], [l, 0.0]]))
    gap = np.array([[-1000.0], [200.0]])
    theta = solve_inclusive(omega, gap)
    denom = 1 - w * l
    np.testing.assert_allclose(theta[0, 0], (gap[0, 0] + w * gap[1, 0]) / denom)
    np.testing.assert_allclose(theta[1, 0], (gap[1, 0] + l * gap[0, 0]) / denom)


# --------------------------------------------------------------------------- #
# Fixtures: German reunification panel
# --------------------------------------------------------------------------- #
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "repgermany.dta"


@pytest.fixture(scope="module")
def german_panel():
    if not _DATA.exists():
        pytest.skip("repgermany.dta not available")
    d = pd.read_stata(_DATA)
    cols = ["country", "year", "gdp", "trade", "infrate",
            "industry", "schooling", "invest80"]
    d = d[cols].copy()
    d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    return d


def _cfg(d, **kw):
    base = dict(df=d, outcome="gdp", treat="treat", unitid="country",
                time="year", method="iscm", affected_units=["Austria"],
                display_graphs=False)
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Layer 3: end-to-end on West Germany
# --------------------------------------------------------------------------- #
def test_iscm_outcome_only_west_germany(german_panel):
    res = SPILLSYNTH(_cfg(german_panel)).fit()
    f = res.iscm
    assert isinstance(f, ISCMFit)
    assert res.method == "iscm"
    # Reunification depressed West German GDP -> negative ATT.
    assert res.att < 0
    # Cross-weights reproduce the paper's neighbourhood (~0.33 each).
    assert 0.25 < f.cross_weights["Austria in West Germany"] < 0.45
    assert 0.0 < f.omega_det <= 1.0
    # Keeping Austria in the pool improves the treated unit's pre-fit.
    assert f.pre_rmspe <= f.pre_rmspe_restricted + 1e-6
    # The inclusive correction moves the estimate away from the naive gap.
    assert abs(res.att - res.att_scm) > 1e-6
    # Shapes line up with the post-period.
    T1 = german_panel.year.ge(1990).groupby(german_panel.country).sum().max()
    assert f.gap.shape == (int(T1),)
    assert f.theta.shape == (2, int(T1))
    assert f.bilevel_solver == "outcome-only"


def test_iscm_covariates_malo_returns_predictor_weights(german_panel):
    cov = ["trade", "infrate", "industry", "schooling", "invest80"]
    res = SPILLSYNTH(_cfg(german_panel, covariates=cov, bilevel_solver="malo")).fit()
    f = res.iscm
    assert f.bilevel_solver == "malo"
    assert f.predictor_weights is not None
    assert set(f.predictor_weights) == set(cov)
    assert sum(f.predictor_weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert res.att < 0


@pytest.mark.slow
def test_iscm_covariates_mscmt_runs(german_panel):
    cov = ["trade", "infrate", "industry", "schooling", "invest80"]
    res = SPILLSYNTH(_cfg(german_panel, covariates=cov, bilevel_solver="mscmt")).fit()
    assert res.iscm.bilevel_solver == "mscmt"
    assert np.isfinite(res.att)


# --------------------------------------------------------------------------- #
# Layer 4: public API contracts
# --------------------------------------------------------------------------- #
def test_dict_and_config_equivalent(german_panel):
    cfg = _cfg(german_panel)
    r1 = SPILLSYNTH(cfg).fit()
    r2 = SPILLSYNTH(SPILLSYNTHConfig(**cfg)).fit()
    assert r1.att == pytest.approx(r2.att)


def test_run_iscm_direct_matches_estimator(german_panel):
    inputs = prepare_spillsynth_inputs(
        df=german_panel, outcome="gdp", treat="treat", unitid="country",
        time="year", affected_units=["Austria"],
    )
    fit = run_iscm(inputs, bilevel_solver="malo")
    res = SPILLSYNTH(_cfg(german_panel)).fit()
    assert fit.att == pytest.approx(res.att)


def test_covariate_windows_applied(german_panel):
    cov = ["trade", "infrate", "industry"]
    # A window restricting to a single early year changes the predictor block,
    # hence (in general) the predictor weights, vs the full-pre-period default.
    full = SPILLSYNTH(_cfg(german_panel, covariates=cov, bilevel_solver="malo")).fit()
    win = SPILLSYNTH(_cfg(german_panel, covariates=cov, bilevel_solver="malo",
                          covariate_windows={"trade": (1960, 1965)})).fit()
    assert full.iscm.predictor_weights is not None
    assert win.iscm.predictor_weights is not None
    # Both are valid simplex predictor-weight vectors.
    assert sum(win.iscm.predictor_weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_covariate_windows_bad_key_rejected(german_panel):
    from mlsynth.exceptions import MlsynthDataError
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        SPILLSYNTH(_cfg(german_panel, covariates=["trade"],
                        covariate_windows={"not_a_cov": (1960, 1965)})).fit()


def test_unknown_bilevel_solver_rejected_by_config(german_panel):
    with pytest.raises((MlsynthConfigError, ValueError)):
        SPILLSYNTHConfig(**_cfg(german_panel, bilevel_solver="nope"))


def test_treated_cannot_be_affected(german_panel):
    with pytest.raises(MlsynthConfigError):
        SPILLSYNTHConfig(**_cfg(german_panel, affected_units=["West Germany"]))
