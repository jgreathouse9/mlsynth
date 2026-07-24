"""Calibrate the Ferman & Pinto (2021, QE) Monte-Carlo DGP from CPS employment.

This reproduces the *fixed environment* of the authors' ``monte_carlo.R``: a
linear factor model fit to monthly state employment rates (50 states + DC,
1982-2019), whose estimated loadings, unit fixed effects and idiosyncratic
dynamics parametrise the simulation. Running this script writes ``fixed_env.npz``
(the small, durable artifact the benchmark reads).

What the authors do, and what this port does:

* Authors: ``gsynth::interFE(y ~ state_dummies, force="time", normalize=F)`` --
  Bai (2009) interactive fixed effects (unit FE via the dummies, time FE via
  ``force``, plus ``r`` common factors), with ``r`` chosen by the ``IC_p1``
  criterion (Bai & Ng 2002). Then ``forecast::auto.arima`` (BIC) fits a Gaussian
  ARMA to each of the four factors and each of the 51 idiosyncratic residual
  series.
* Here: the same two-way-plus-``r``-factor interactive-FE estimator, ported to
  NumPy (``interFE_twoway``); ``r`` fixed to the paper's 4 (``gsynth`` is not
  installable in CI -- its ``IC_p1`` path selects 4, our port's lands at 5, a
  known fine-detail gap between the two iterated-FE implementations); factors put
  in the canonical Bai identification (``Lambda'Lambda`` diagonal, descending) so
  "factor 1" is well-defined; and a Gaussian AR(1) fit (OLS) to each factor and
  residual as a compact stand-in for ``auto.arima``.

The consequence -- validated in ``benchmarks/cases/ferman_pinto_mc.py`` -- is that
the fixed-effect-selected panels (Table 1 Panels A & B) reproduce the published
bias closely, because they turn on the unit FE, which is rotation-invariant; the
factor-loading-selected panels (C & D) reproduce the *structure* of Table 1 (all
three estimators biased, SC and demeaned-SC biases far below DID, bias shrinking
in ``T0``) but not the exact magnitudes, which depend on ``gsynth``'s precise
factor rotation. The estimator identity itself -- mlsynth's SC / MSCa versus the
authors' ``quadprog`` QP -- is checked value-for-value against live R and is
independent of any of this calibration detail.

Reference: Ferman, B. and Pinto, C. (2021), "Synthetic controls with imperfect
pretreatment fit", Quantitative Economics 12:1197-1221; supplementary
``monte_carlo.R`` / ``_aux.R``. Input ``cps_employment_1982_2019.csv`` is the
authors' ``base_state_final.csv`` restricted to the columns and window used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_N_FACTORS = 4                      # paper's IC_p1 selection


def build_panel(csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """(T x N) employment-rate matrix and the state labels (columns = states)."""
    d = pd.read_csv(csv)
    d = d[(d.year >= 1982) & (d.year <= 2019)].copy()
    d["ym"] = d.year.astype(str) + "-" + d.month.map(lambda x: f"{x:02d}")
    d = d.sort_values(["state", "ym"])
    states = d.state.unique()
    N, T = len(states), d.ym.nunique()
    Y = d["employment_per"].to_numpy().reshape(N, T).T
    return Y, states


def interFE_twoway(Y: np.ndarray, r: int, tol: float = 1e-9, maxit: int = 5000):
    """Bai (2009) interactive FE with unit + time fixed effects and ``r`` factors.

    Iterates: (1) two-way demean ``Y - F Lambda'`` to refresh the unit/time FE;
    (2) extract the top-``r`` principal components of the doubly-demeaned residual
    as factors in the Bai normalization ``F'F/T = I``. Returns
    ``(unit_fe, F, Lambda, resid)``.
    """
    T, N = Y.shape
    F = np.zeros((T, r))
    Lam = np.zeros((N, r))
    prev = None
    for _ in range(maxit):
        W = Y - F @ Lam.T
        g = W.mean()
        unit = W.mean(0) - g
        time = W.mean(1) - g
        U = Y - g - unit[None, :] - time[:, None]
        ev, evec = np.linalg.eigh(U @ U.T)
        idx = np.argsort(ev)[::-1][:r]
        F = evec[:, idx] * np.sqrt(T)
        Lam = U.T @ F / T
        cur = g + unit[None, :] + time[:, None] + F @ Lam.T
        if prev is not None and np.max(np.abs(cur - prev)) < tol:
            break
        prev = cur
    resid = Y - g - unit[None, :] - time[:, None] - F @ Lam.T
    return unit, F, Lam, resid


def canonical_rotate(F: np.ndarray, Lam: np.ndarray):
    """Rotate to the Bai identification: ``Lambda'Lambda`` diagonal, descending,
    with each factor signed so its largest-magnitude loading is positive."""
    ev, V = np.linalg.eigh(Lam.T @ Lam)
    V = V[:, np.argsort(ev)[::-1]]
    F, Lam = F @ V, Lam @ V
    for k in range(Lam.shape[1]):
        if Lam[np.argmax(np.abs(Lam[:, k])), k] < 0:
            F[:, k] *= -1
            Lam[:, k] *= -1
    return F, Lam


def ar1_fit(x: np.ndarray) -> tuple[float, float, float]:
    """OLS AR(1): return (rho, marginal mean, innovation SD)."""
    x0, x1 = x[:-1], x[1:]
    X = np.column_stack([np.ones_like(x0), x0])
    b = np.linalg.lstsq(X, x1, rcond=None)[0]
    return float(b[1]), float(x.mean()), float((x1 - X @ b).std(ddof=2))


def main() -> None:
    Y, states = build_panel(_HERE / "cps_employment_1982_2019.csv")
    T, N = Y.shape
    unit, F, Lam, resid = interFE_twoway(Y, _N_FACTORS)
    F, Lam = canonical_rotate(F, Lam)

    fac_ar = np.array([ar1_fit(F[:, k]) for k in range(_N_FACTORS)])   # (r, 3)
    res_ar = np.array([ar1_fit(resid[:, j]) for j in range(N)])        # (N, 3)
    sd_factors = F.std(0)

    np.savez(
        _HERE / "fixed_env.npz",
        unit=unit, Lam=Lam, fac_ar=fac_ar, res_ar=res_ar,
        sd_factors=sd_factors, states=states,
        n_factors=_N_FACTORS, T=T, N=N,
    )

    of, o1 = np.argsort(unit), np.argsort(Lam[:, 0])
    facts = {
        "n_states": int(N), "n_periods": int(T), "n_factors": _N_FACTORS,
        "panelA_state_2nd_largest_fe": int(states[of[-2]]),
        "panelB_state_2nd_smallest_fe": int(states[of[1]]),
        "panelC_state_2nd_largest_mu1": int(states[o1[-2]]),
        "panelD_state_2nd_smallest_mu1": int(states[o1[1]]),
        "fe_range": [round(float(unit.min()), 3), round(float(unit.max()), 3)],
        "resid_sd_mean": round(float(resid.std(0).mean()), 3),
        "sd_factor1": round(float(sd_factors[0]), 3),
    }
    (_HERE / "calibration_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print("wrote fixed_env.npz and calibration_facts.json")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
