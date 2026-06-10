"""Data-generating process for the NSC Path-B benchmark (Tian 2023, Section 4).

Re-implements the Monte-Carlo design of Tian (2023), *"The Synthetic Control
Method with Nonlinear Outcomes,"* Section 4 (the source of the estimator's
Table 1). The paper provides the full DGP, so this is a faithful port of the
**model**:

* Each unit ``i`` carries two observed predictors ``X_i`` and four unobserved
  predictors ``mu_i``, both drawn ``U(0, sqrt(12))`` (unit variance); the
  time-varying coefficients ``beta_t`` (2-vector) and ``lambda_t`` (4-vector)
  are drawn ``N(10, 1)``; the idiosyncratic shock is ``eps_it ~ N(0, 1)``.
* The latent outcome ``Y*_it = X_i' beta_t + mu_i' lambda_t + eps_it`` is
  rescaled to ``[0, 1]`` and raised to the power ``r`` (eq. 9): ``r = 1`` is the
  linear case, ``r = 2`` the nonlinear case where the standard SC estimator is
  biased.
* The treated unit (``i = 0``) receives the ramped effect
  ``tau_t = 0.02, 0.04, ..., 0.20`` over the ten post-treatment periods (eq. 10);
  all other cells are untreated.

The paper draws the structure ``(X, mu, beta, lambda)`` 20 times and the shocks
250 times per structure (5000 samples/setting) and fixes the CV-tuned penalty
per structure to save compute. The benchmark instead draws independent full
samples (structure + shocks together) and lets NSC's own cross-validation pick
the penalty each time -- a valid Monte Carlo of the same DGP, just without the
variance-reduction trick, which is why the benchmark uses a smaller draw count.

Inferred / not-pinned-by-the-paper detail (scenario-1 disclosure): the rescaling
in eq. 9 is sample-dependent (min/max over the realized panel), reproduced here
exactly; nothing else is free.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

_N_POST = 10                       # ten post-treatment periods (eq. 10)
_TAU_STEP = 0.02                   # tau_t = 0.02, 0.04, ..., 0.20


def nsc_true_effects() -> np.ndarray:
    """The ramped post-treatment effects ``[0.02, 0.04, ..., 0.20]`` (eq. 10)."""
    return _TAU_STEP * np.arange(1, _N_POST + 1)


def simulate_nsc_panel(
    *,
    J: int = 50,
    T0: int = 30,
    r: int = 2,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """One Monte-Carlo draw of the Tian (2023) nonlinear-outcome DGP.

    Parameters
    ----------
    J : int
        Number of donor (control) units; the panel has ``J + 1`` units with
        unit ``0`` treated.
    T0 : int
        Pre-treatment periods; the panel has ``T0 + 10`` periods total.
    r : int
        Nonlinearity degree: ``1`` linear, ``2`` nonlinear.
    seed : int
        RNG seed for this draw.

    Returns
    -------
    (df, tau_path)
        ``df`` is the long panel (``unit, time, y, D``); ``tau_path`` is the
        length-10 vector of true post-treatment effects.
    """
    rng = np.random.default_rng(seed)
    n_units = J + 1
    T = T0 + _N_POST

    hi = np.sqrt(12.0)                                   # U(0, sqrt(12)) => var 1
    X = rng.uniform(0.0, hi, size=(n_units, 2))          # observed predictors
    mu = rng.uniform(0.0, hi, size=(n_units, 4))         # unobserved predictors
    beta_t = rng.normal(10.0, 1.0, size=(T, 2))
    lam_t = rng.normal(10.0, 1.0, size=(T, 4))
    eps = rng.normal(0.0, 1.0, size=(T, n_units))

    Y_star = (X @ beta_t.T).T + (mu @ lam_t.T).T + eps   # (T, n_units)
    Yn = (Y_star - Y_star.min()) / (Y_star.max() - Y_star.min())
    Y0 = Yn ** r                                         # eq. 9

    tau_path = nsc_true_effects()
    Y = Y0.copy()
    Y[T0:, 0] += tau_path                                # treat unit 0, post only

    rows = [{"unit": j, "time": t, "y": float(Y[t, j]),
             "D": int(j == 0 and t >= T0)}
            for j in range(n_units) for t in range(T)]
    return pd.DataFrame(rows), tau_path
