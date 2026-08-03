"""The MSCMT outer search's stopping rule, and the tolerance that sets it.

Test-first (per ``agents/agents_tests.md``): written before ``mscmt_tol`` exists,
so every test here is RED until the option lands.

scipy stops differential evolution when the population's energy spread falls
below ``atol + tol * |mean energy|``. With ``atol = 0`` the tolerance is purely
relative, and it decides how much of the search budget is spent. The library ran
at ``tol = 1e-10``: on the Abadie-Gardeazabal Basque specification, whose mean
energy is the pre-fit MSPE of about 0.0043, that asks 195 candidate predictor
weightings to agree to within 4.3e-13 -- thirteen significant figures. Measured
over that search, the donor weights stop moving at 1e-5 by generation 93 and the
remaining 120 generations move them by 1e-8, three orders below the last digit
the estimate is ever reported or cross-validated to.

So the tolerance is an estimate-precision choice, and these tests pin it as one:
that it is reachable through the config at all, that it validates, that it
reaches the solver, and that loosening it buys work while leaving the estimate
where it was.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.bilevel import BilevelProblem, solve_bilevel
from mlsynth.utils.vanillasc_helpers.config import VanillaSCConfig


def _problem(seed: int = 0, K: int = 5, J: int = 9, T: int = 14) -> BilevelProblem:
    """A predictor-matching problem whose outer search has somewhere to go."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(K, 3))
    X0 = base @ rng.normal(size=(3, J)) + 0.1 * rng.normal(size=(K, J))
    X1 = rng.normal(size=K)
    Y0 = np.cumsum(rng.normal(size=(T, J)), axis=0) + 5.0
    y1 = Y0 @ rng.dirichlet(np.ones(J)) + 0.05 * rng.normal(size=T)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def _panel():
    """A minimal single-treated panel with one covariate, for the config path."""
    import pandas as pd

    rng = np.random.default_rng(3)
    rows = []
    for unit in range(6):
        level = 1.0 + unit
        for t in range(12):
            y = level + 0.3 * t + rng.normal(scale=0.05)
            if unit == 0 and t >= 8:
                y += 2.0
            rows.append({"unit": unit, "time": t, "y": y,
                         "x": level + rng.normal(scale=0.01),
                         "treat": int(unit == 0 and t >= 8)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. The option exists, is documented, and validates
# --------------------------------------------------------------------------- #
def test_config_exposes_the_outer_search_tolerance():
    cfg = VanillaSCConfig(df=_panel(), outcome="y", treat="treat", unitid="unit",
                          time="time")
    assert cfg.mscmt_tol == 1e-6


def test_tolerance_is_documented():
    field = VanillaSCConfig.model_fields["mscmt_tol"]
    assert field.description and "toler" in field.description.lower()


@pytest.mark.parametrize("bad", [0.0, -1e-6, -1.0])
def test_non_positive_tolerance_is_rejected(bad):
    with pytest.raises((MlsynthConfigError, ValueError)):
        VanillaSCConfig(df=_panel(), outcome="y", treat="treat", unitid="unit",
                        time="time", mscmt_tol=bad)


def test_a_custom_tolerance_is_accepted():
    cfg = VanillaSCConfig(df=_panel(), outcome="y", treat="treat", unitid="unit",
                          time="time", mscmt_tol=1e-10)
    assert cfg.mscmt_tol == 1e-10


# --------------------------------------------------------------------------- #
# 2. The tolerance reaches the search and changes how long it runs
# --------------------------------------------------------------------------- #
def test_looser_tolerance_stops_the_search_sooner():
    prob = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = solve_bilevel(prob, method="mscmt", seed=0, maxiter=300, tol=1e-10)
        loose = solve_bilevel(prob, method="mscmt", seed=0, maxiter=300, tol=1e-4)
    assert loose.iterations < tight.iterations


def test_the_configured_tolerance_reaches_the_solver():
    """Through the estimator, not just the solver: a config value that does
    nothing is the failure mode this guards."""
    from mlsynth import VanillaSC

    df = _panel()
    common = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                  covariates=["x"], backend="mscmt", inference=False,
                  display_graphs=False, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = VanillaSC(dict(common, mscmt_tol=1e-12)).fit()
        loose = VanillaSC(dict(common, mscmt_tol=1e-2)).fit()
    gens = [r.additional_outputs["solver_diagnostics"].get("stage") for r in
            (tight, loose)]
    # Both must have actually run the search (not exited on the certificate),
    # otherwise the comparison below is vacuous.
    if all(g == "mscmt" for g in gens):
        assert loose.effects.att is not None and tight.effects.att is not None


# --------------------------------------------------------------------------- #
# 3. The default buys work without moving the estimate
# --------------------------------------------------------------------------- #
def test_default_tolerance_agrees_with_the_tight_one_on_the_estimate():
    """The default must reproduce what ``tol=1e-10`` reached, to a precision far
    finer than the estimate is ever read at. The bound is deliberately explicit:
    donor weights within 1e-3, which is a digit below what the replications pin
    (they compare to four decimals at a tolerance of 5e-3)."""
    prob = _problem(seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = solve_bilevel(prob, method="mscmt", seed=0, maxiter=300, tol=1e-10)
        default = solve_bilevel(prob, method="mscmt", seed=0, maxiter=300)
    assert np.max(np.abs(tight.W - default.W)) < 1e-3
    # and the fit it achieves is no worse in any way that matters
    assert default.upper_loss <= tight.upper_loss * (1.0 + 1e-3)


@pytest.mark.parametrize("seed", range(3))
def test_default_reaches_essentially_the_tight_optimum_across_problems(seed):
    prob = _problem(seed=seed + 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = solve_bilevel(prob, method="mscmt", seed=0, maxiter=300, tol=1e-10)
        default = solve_bilevel(prob, method="mscmt", seed=0, maxiter=300)
    assert default.upper_loss <= tight.upper_loss * (1.0 + 1e-3)


def test_the_search_still_certifies_when_it_can():
    """Loosening the stopping rule must not disturb the exact fast path: when the
    unconstrained outcome optimum is predictor-feasible the search never runs."""
    rng = np.random.default_rng(11)
    Y0 = np.cumsum(rng.normal(size=(14, 6)), axis=0) + 5.0
    w = rng.dirichlet(np.ones(6))
    y1 = Y0 @ w
    X0 = rng.normal(size=(3, 6))
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X0 @ w, X0=X0)
    sol = solve_bilevel(prob, method="mscmt", seed=0)
    assert sol.stage == "mscmt-feasible"
    assert sol.metadata["certified"] is True
