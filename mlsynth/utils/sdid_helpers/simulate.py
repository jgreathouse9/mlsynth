"""Data-generating process for the de Brabander et al. (2025) Path B benchmark.

Ports the DGP in the authors' ``Simulations/simulations_no covariates.R`` --
the factor model of Abadie, Diamond & Hainmueller (2010) and Kaul, Klossner,
Pfeifer & Schieler (2022), which they calibrate to their own Brexit panel:

.. math::

   y_{jt} \\;=\\; \\delta_t \\;+\\; \\theta_t' c_j \\;+\\; \\gamma_t \\mu_j
   \\;+\\; \\varepsilon_{jt}.

:math:`c_j` holds unit :math:`j`'s six covariate means over 2010:Q2-2016:Q1;
:math:`\\delta_t` and :math:`\\theta_t` are calibrated from the real panel;
:math:`\\gamma_t \\mu_j` is an interactive fixed effect whose strength the
benchmark sweeps; and :math:`\\varepsilon` is idiosyncratic noise. The true
treatment effect is zero, so an estimate at the final period *is* the
estimation error -- which is what the paper's Tables 9-12 summarise.

Calibration is a stack of unit-covariate by period-identity interactions plus
period dummies in the authors' code. That design is block diagonal by period,
so it collapses to one least-squares fit per period of the units' outcomes on
their covariate means with an intercept -- the same estimator, without building
a 175-column design matrix. ``test_sdid_simulation.py`` asserts the equivalence
through the residual orthogonality it implies rather than taking it on trust.

Where the authors' code and their prose disagree, the code is followed here,
because the code is what produced the published tables:

* the common factor is :math:`\\gamma(1 - 1/t)`, where the paper writes
  :math:`t\\gamma/T_0`;
* the noise is drawn with standard deviation :math:`\\sigma^2`, where the paper
  describes variance :math:`\\sigma^2`. This one is checkable rather than a
  judgment call -- the published :math:`\\sigma = 0.25` RMSEs are 0.0625 times
  the :math:`\\sigma = 1` ones, which a standard deviation of 0.25 cannot
  produce;
* the treated unit's loading :math:`\\mu_1` is the second largest of the draws
  rather than an independent uniform. That is a design choice, not a slip: it
  places the treated unit inside the donors' convex hull, so the simplex weight
  problem is feasible without extrapolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError

#: Born et al. (2019) covariates, as the authors' simulation script uses them.
BRABANDER_COVARIATES = ("real_con", "real_inv", "real_exp", "real_imp",
                        "lab_prod", "tot_emp_pc")
#: Calibration window: 2010:Q2-2016:Q1 outcomes plus the following quarter.
_FIRST_T, _LAST_COV_T, _LAST_T = 202, 225, 226
_TREATED = "United Kingdom"


@dataclass(frozen=True)
class BrabanderCalibration:
    """The structural truth, drawn once from the real panel and then held fixed.

    Attributes
    ----------
    delta : np.ndarray
        Common time effects, shape ``(T,)``.
    beta : np.ndarray
        Time-varying covariate coefficients, shape ``(K, T)``.
    covariates : np.ndarray
        Per-unit covariate means, shape ``(K, N)``, treated unit first.
    systematic : np.ndarray
        ``delta + beta' covariates``, shape ``(T, N)`` -- the non-stochastic
        part of every replication.
    unit_names : list of str
        Unit labels in column order, treated unit first.
    """

    delta: np.ndarray
    beta: np.ndarray
    covariates: np.ndarray
    systematic: np.ndarray
    unit_names: List[str]


@dataclass(frozen=True)
class BrabanderReplication:
    """One draw. ``outcome`` is the panel; the rest is what produced it.

    The components are exposed because the paper does not only report total
    error -- Tables 10 to 12 split it into covariate, common-factor and
    idiosyncratic parts, and that split needs the pieces, not just the sum.
    """

    outcome: np.ndarray
    loadings: np.ndarray
    factor_path: np.ndarray
    noise: np.ndarray


def calibrate_brabander_dgp(panel, covariates: Sequence[str] = BRABANDER_COVARIATES,
                            treated: str = _TREATED) -> BrabanderCalibration:
    """Calibrate the factor model to a Brexit panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel with ``country`` / ``t`` / ``y`` and the covariate columns,
        covering at least ``t = 202..226``.
    covariates : sequence of str
        Covariate columns; defaults to the paper's six.
    treated : str
        Unit placed first, matching the authors' reordering.

    Returns
    -------
    BrabanderCalibration

    Raises
    ------
    MlsynthDataError
        If a covariate column is absent, the window is not covered, or the
        calibration window is not a balanced panel.
    """
    cols = list(covariates)
    missing = [c for c in cols + ["country", "t", "y"]
               if c not in panel.columns]
    if missing:
        raise MlsynthDataError(
            f"column(s) {missing} are not in the panel; the de Brabander DGP "
            f"calibrates on {cols} plus country/t/y.")

    window = panel[(panel.t >= _FIRST_T) & (panel.t <= _LAST_T)]
    n_periods = int(window.t.nunique())
    expected = _LAST_T - _FIRST_T + 1
    if n_periods != expected:
        raise MlsynthDataError(
            f"the calibration window t={_FIRST_T}..{_LAST_T} needs {expected} "
            f"periods; the panel supplies {n_periods}.")

    names = list(dict.fromkeys(window.country))
    if treated not in names:
        raise MlsynthDataError(
            f"treated unit {treated!r} is not in the panel; units are {names}.")
    order = [treated] + [c for c in names if c != treated]

    wide = window.pivot(index="t", columns="country", values="y")
    outcomes = wide.loc[:, order].to_numpy(dtype=float)          # (T, N)
    if not np.isfinite(outcomes).all():
        raise MlsynthDataError(
            "the calibration window has missing outcomes; the DGP needs a "
            "balanced panel over t=%d..%d." % (_FIRST_T, _LAST_T))

    cov_window = panel[(panel.t >= _FIRST_T) & (panel.t <= _LAST_COV_T)]
    means = cov_window.groupby("country", observed=True)[cols].mean().reindex(order)
    C = means.to_numpy(dtype=float).T                            # (K, N)
    if not np.isfinite(C).all():
        raise MlsynthDataError(
            "covariate means contain NaN/inf over the calibration window.")

    # One least-squares fit per period (the collapsed block-diagonal design).
    design = np.vstack([np.ones(C.shape[1]), C]).T               # (N, K+1)
    coef = np.linalg.lstsq(design, outcomes.T, rcond=None)[0]    # (K+1, T)
    delta, beta = coef[0], coef[1:]
    return BrabanderCalibration(
        delta=delta, beta=beta, covariates=C,
        systematic=delta[:, None] + beta.T @ C, unit_names=order)


def _check(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise MlsynthConfigError(f"{name} must be finite; got {value}.")
    if v < 0:
        raise MlsynthConfigError(f"{name} must be non-negative; got {value}.")
    return v


def simulate_brabander(calibration: BrabanderCalibration, gamma: float,
                       sigma: float, rng) -> BrabanderReplication:
    """Draw one replication from the calibrated DGP.

    Parameters
    ----------
    calibration : BrabanderCalibration
        Structural truth from :func:`calibrate_brabander_dgp`, held fixed
        across replications so only the stochastic parts vary.
    gamma : float
        Common-factor strength; the paper sweeps ``{0, 0.25, 1, 1.5}``. Larger
        values make the interactive fixed effect dominate the covariates.
    sigma : float
        Noise scale; the errors are drawn with standard deviation ``sigma**2``
        (see the module docstring). The paper sweeps ``{0.25, 1}``.
    rng : np.random.Generator
        Seeded generator. Passing the same seed across ``gamma`` values makes
        the comparison paired -- the loadings and noise are identical and only
        the factor term moves.

    Returns
    -------
    BrabanderReplication
    """
    gamma = _check("gamma", gamma)
    sigma = _check("sigma", sigma)

    n_periods, n_units = calibration.systematic.shape
    t = np.arange(1, n_periods + 1, dtype=float)
    factor_path = gamma * (1.0 - 1.0 / t)

    others = rng.uniform(0.0, 1.0, n_units - 1)
    # Treated loading = second largest draw, so it sits inside the donor hull.
    loadings = np.concatenate([[np.sort(others)[-2]], others])
    noise = rng.normal(0.0, sigma ** 2, size=(n_periods, n_units))

    outcome = (calibration.systematic
               + np.outer(factor_path, loadings) + noise)
    return BrabanderReplication(outcome=outcome, loadings=loadings,
                                factor_path=factor_path, noise=noise)
