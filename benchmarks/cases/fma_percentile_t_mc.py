"""FMA Path-B: Wang, Racine & Wang (2025) percentile-t coverage (Tables 1-2).

Validates the ``"percentile_t"`` inference option against the Monte Carlo in
Lixiong Wang, Jeffrey S. Racine & Qiying Wang, "Bootstrap inference on a factor
model based average treatment effects estimator", Econometric Reviews
45(1):78-95.

The paper's finding is about short pre-periods. Li & Sonnier's (2023) normal
interval for the ATT is correctly sized asymptotically, but at
:math:`T_1 = 10` their Table 1 measures it covering a nominal 95% only
80.0-80.7% of the time. Taking the critical values from the bootstrap
distribution of the studentized statistic instead of from the normal table
restores coverage to 93.0-95.6% without assuming the treated and control units
share an idiosyncratic variance.

The DGP is Equation 12, the same Hsiao-Ching-Wan factor processes Li & Sonnier
use, so :func:`mlsynth.utils.fma_helpers.simulation._factors_dgp1` supplies the
factors. What differs is the variance grid: the paper fixes
:math:`\\sigma^2_{tr} = 1` and varies :math:`\\sigma^2_{co} \\in \\{1, 1/4,
1/10\\}`, giving variance ratios 1, 4 and 10. The true ATT is zero, so coverage
is whether the interval contains 0.

Two cells are reproduced, both at :math:`(N, T_2) = (15, 10)` and
:math:`\\sigma^2_{co} = 1`, with the asymptotic interval computed on the same
draws for contrast:

  ==========  ===============  =============  =============
  cell        paper asymptotic  paper bootstrap  what this asserts
  ==========  ===============  =============  =============
  T1 = 10     0.800            0.930          bootstrap near nominal,
                                              asymptotic well below
  T1 = 30     0.914            0.961          both near nominal
  ==========  ===============  =============  =============

The asymptotic column here uses Appendix A.1's :math:`\\hat\\Omega` with normal
critical values, which is the interval the paper's own column reports. It is
not the interval :func:`~mlsynth.utils.fma_helpers.inference.asymptotic_inference`
returns: that one applies a :math:`T_0 - (r + 1)` degrees-of-freedom correction
to the residual variance, which widens it at short pre-periods and lifts its
coverage in this cell from about 0.79 to about 0.89.

Path B (the paper's simulation). Tolerances absorb the Monte Carlo noise at
``M`` draws (the paper uses 2,000 simulations with 1,000 bootstrap resamples;
at the full configuration every one of the 18 cells in Tables 1-2 reproduces
within 0.017). Deterministic (seeded).
"""
from __future__ import annotations

import warnings

import numpy as np

M = 300              # simulations per cell
B = 400              # bootstrap draws per simulation
N_CO = 15
T2 = 10


def _draw(T1: int, N_co: int, sigma_co: float, rng):
    """One draw from Equation 12 plus the outcome equation of Section 4."""
    from mlsynth.utils.fma_helpers.simulation import _factors_dgp1

    T, N = T1 + T2, N_co + 1
    F = _factors_dgp1(T, rng)
    lam = rng.normal(1.0, 1.0, size=(N, 3))
    u = np.empty((N, T))
    u[0] = rng.normal(0.0, 1.0, T)                  # sigma_tr = 1
    u[1:] = rng.normal(0.0, sigma_co, (N_co, T))
    Y = 1.0 + lam @ F.T + u                         # alpha = 1, ATT = 0
    return Y[0], Y[1:].T


def _cell(T1: int, sigma_co: float = 1.0) -> tuple:
    from scipy.stats import norm

    from mlsynth.utils.fma_helpers.factors import extract_factors
    from mlsynth.utils.fma_helpers.fit import (
        estimate_loading_and_counterfactual,
    )
    from mlsynth.utils.fma_helpers.inference import (
        percentile_t_inference, robust_omega,
    )

    z = float(norm.ppf(0.975))
    hits_norm = hits_boot = seen = 0
    for j in range(M):
        rng = np.random.default_rng(j)
        y, Yco = _draw(T1, N_CO, sigma_co, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, F, _ = extract_factors(
                Yco, stationarity="stationary", preprocessing="demean",
                n_factors=None, max_factors=10,
            )
            _, cf, F_aug, _ = estimate_loading_and_counterfactual(y, F, T1)
            if T1 <= F_aug.shape[1]:
                continue
            gap = y - cf
            att = float(gap[T1:].mean())
            omega, _, _ = robust_omega(
                factors_with_const=F_aug, T0=T1, T2=T2,
                residuals_pre=gap[:T1],
            )
            pt = percentile_t_inference(
                treated_outcome=y, counterfactual=cf,
                factors_with_const=F_aug, T0=T1, n_replicates=B,
                seed=10_000 + j,
            )
        if not np.isfinite(pt["lower"]) or not np.isfinite(omega):
            continue
        seen += 1
        hits_norm += abs(att) <= z * float(np.sqrt(omega / T2))
        hits_boot += pt["lower"] <= 0.0 <= pt["upper"]
    return hits_norm / seen, hits_boot / seen


def run() -> dict:
    norm_10, boot_10 = _cell(T1=10)
    norm_30, boot_30 = _cell(T1=30)
    return {
        "asymptotic_T1_10": norm_10,
        "bootstrap_T1_10": boot_10,
        "asymptotic_T1_30": norm_30,
        "bootstrap_T1_30": boot_30,
        # The headline: at T1 = 10 the bootstrap buys back the coverage the
        # normal table loses. The paper's own gap is 0.930 - 0.800 = 0.130.
        "coverage_gain_T1_10": boot_10 - norm_10,
        # And the gap closes as the pre-period lengthens: 0.961 - 0.914.
        "coverage_gain_T1_30": boot_30 - norm_30,
    }


# Deterministic (seeded). Paper cells: asymptotic 0.800 / 0.914, bootstrap
# 0.930 / 0.961 at T1 = 10 / 30. Tolerances are about 5 Monte Carlo standard
# errors at M = 300.
EXPECTED = {
    "asymptotic_T1_10": (0.800, 0.07),
    "bootstrap_T1_10": (0.930, 0.06),
    "asymptotic_T1_30": (0.914, 0.06),
    "bootstrap_T1_30": (0.961, 0.05),
    "coverage_gain_T1_10": (0.130, 0.09),
    "coverage_gain_T1_30": (0.047, 0.07),
}
