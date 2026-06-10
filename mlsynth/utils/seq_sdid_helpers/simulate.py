"""Data-generating process for the Sequential SDiD Path-B benchmark.

Re-implements the *design* of Arkhangelsky & Samkov (2025), Section 5.2.2
("Experiment 2: Calibrated State-Level Panel") from the paper's description --
we have the paper only (no replication code, and the authors' CPS women's
log-wage panel is not public), so this is a faithful reconstruction of the
**model**, not a port of their exact calibrated panel.

The paper's recipe, and how each piece maps here:

* **Structural truth treated as fixed.** The authors decompose a real
  state-by-year panel into a two-way fixed-effects part (~94% of variation), a
  low-rank interactive-fixed-effects (IFE) part (~5%), and an AR idiosyncratic
  remainder, then *freeze* the structural components and redraw only the
  shocks. We mirror that: :func:`calibrate_staggered_ife` draws the structural
  truth **once** (unit/time FE + a rank-one interactive fixed effect), and
  :func:`simulate_replication` redraws only the AR(2) idiosyncratic noise per
  replication. Fixing the structure is what makes the bootstrap -- which only
  resamples units within one panel -- a valid measure of the sampling
  variability the Monte Carlo averages over.

* **A parallel-trends violation correlated with treatment timing.** We use the
  canonical rank-one IFE: a *differential linear trend* (loading ``lambda_i``
  on the rising factor ``f_t = t / T``). Adoption is tilted toward high-loading
  units (steeper trends), so treated and comparison units have systematically
  different unobserved trends -- exactly the IFE confounding the paper induces
  by tying adoption to the leading factor loading. Standard DiD assumes a
  common trend and is therefore biased; Sequential SDiD balances the loading
  using later-adopting / never-treated donors and is not.

* **Large cohorts.** The paper replicates each unit four times (Section 5.2.1)
  so cohort aggregates concentrate and the low-noise asymptotics apply. We
  expose ``n_copies`` (default 4) and do the same.

* **Donor balance.** A cohort needs at least two later-adopting / never-treated
  donor cohorts to balance even a rank-one loading; the latest cohorts are
  donor-starved. :func:`calibrate_staggered_ife` reports ``a_max`` -- the
  largest cohort that keeps a healthy donor pool -- so the benchmark estimates
  only well-balanced cohorts (the paper's many-cohort regime avoids starvation
  by construction).

Inferred / not-pinned-by-the-paper details (scenario-1 disclosure): the exact
panel dimensions, the IFE rank (we use the minimal rank one), the
adoption-probability link, the trend magnitude, and the replication count's
interaction with the cohort spread are calibrated here to land in the paper's
qualitative regime, not read from their data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

_NEVER = 10**9  # adoption sentinel for never-treated units


@dataclass(frozen=True)
class CalibratedDesign:
    """Fixed structural truth for the calibrated state-panel design.

    Parameters
    ----------
    base : np.ndarray
        ``(T, S)`` noise-free structural mean ``alpha_i + beta_t +
        trend_strength * (t / T) * lambda_i`` -- unit/time FE plus the
        rank-one differential-trend interactive fixed effect.
    adopt : np.ndarray
        Length-``S`` adoption period (0-based time index) per unit;
        never-treated units carry the sentinel ``_NEVER``.
    a_max : int
        Largest treated cohort (1-based time index, matching the estimator's
        ``a_max`` config) that still has a healthy donor pool. Estimating up to
        here keeps every cohort donor-balanced.
    T, S, T0 : int
        Periods, units, and the first possible adoption period.
    n_treated, n_never, n_cohorts : int
        Counts describing the realized staggered design.
    """

    base: np.ndarray
    adopt: np.ndarray
    a_max: int
    T: int
    S: int
    T0: int
    n_treated: int
    n_never: int
    n_cohorts: int


def calibrate_staggered_ife(
    *,
    seed: int = 2024,
    S: int = 90,
    T: int = 48,
    T0: int = 18,
    n_cohorts_span: int = 16,
    trend_strength: float = 2.8,
    p_slope: float = 3.4,
    adopt_slope: float = 4.5,
    adopt_sd: float = 2.5,
    donor_tail: int = 6,
) -> CalibratedDesign:
    """Draw the fixed structural truth once (FE + rank-one differential trend).

    The leading loading ``lambda_i`` both sets each unit's trend slope and tilts
    its adoption (probability rises with ``lambda_i`` via a logistic link, and
    adoption *time* rises with it too), so treatment timing is correlated with
    the unobserved trend -- the parallel-trends violation. ``a_max`` is capped to
    the ``donor_tail``-th-latest cohort so the estimated cohorts keep enough
    later / never-treated donors to balance the loading.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.0, 1.0, S)             # unit FE
    beta = np.linspace(0.0, 1.0, T)             # common time trend (FE)
    lam = rng.normal(0.0, 1.0, S)               # trend loading
    f = np.arange(T) / T                        # rising factor, f_t = t / T
    base = alpha[None, :] + beta[:, None] + trend_strength * np.outer(f, lam)

    z = (lam - lam.mean()) / lam.std()
    p = 1.0 / (1.0 + np.exp(-(p_slope * z)))    # ever-treated probability
    ever = rng.random(S) < p

    adopt = np.full(S, _NEVER, dtype=int)
    lo, hi = T0, T0 + n_cohorts_span - 1
    mid = (lo + hi) // 2
    for j in range(S):
        if ever[j]:
            a = int(round(mid + adopt_slope * z[j] + rng.normal(0.0, adopt_sd)))
            adopt[j] = min(max(a, lo), hi)

    treated_dates = sorted({int(a) for a in adopt if a != _NEVER})
    a_max = (treated_dates[-donor_tail]
             if len(treated_dates) > donor_tail else treated_dates[0])

    return CalibratedDesign(
        base=base,
        adopt=adopt,
        a_max=int(a_max),
        T=T,
        S=S,
        T0=T0,
        n_treated=int((adopt != _NEVER).sum()),
        n_never=int((adopt == _NEVER).sum()),
        n_cohorts=len(treated_dates),
    )


def simulate_replication(
    design: CalibratedDesign,
    rng: np.random.Generator,
    *,
    tau: float = 1.0,
    n_copies: int = 4,
    ar: Tuple[float, float] = (0.5, -0.1),
    sigma: float = 1.0,
) -> pd.DataFrame:
    """One Monte-Carlo draw: fixed structure + fresh AR(2) shocks + effect.

    Each structural unit is replicated ``n_copies`` times (same FE, loading and
    adoption, independent noise) to enlarge cohorts, and a constant effect
    ``tau`` is added to every treated post-period cell. Returns the long panel
    (``unit, year, y, treat``) the estimator consumes.
    """
    base, adopt, T, S = design.base, design.adopt, design.T, design.S
    n_total = S * n_copies
    y = np.repeat(base, n_copies, axis=1) + _ar2_noise(T, n_total, ar, sigma, rng)
    treat = np.zeros((T, n_total), dtype=int)
    adv = np.repeat(adopt, n_copies)
    for u in range(n_total):
        if adv[u] != _NEVER:
            y[adv[u]:, u] += tau
            treat[adv[u]:, u] = 1
    return pd.DataFrame({
        "unit": np.repeat(np.arange(n_total), T),
        "year": np.tile(np.arange(T), n_total),
        "y": y.T.ravel(),
        "treat": treat.T.ravel(),
    })


def _ar2_noise(
    T: int, n: int, ar: Tuple[float, float], sigma: float,
    rng: np.random.Generator, burn: int = 50,
) -> np.ndarray:
    """Stationary AR(2) idiosyncratic noise, shape ``(T, n)``, with burn-in."""
    e = np.zeros((T + burn, n))
    innov = rng.normal(0.0, sigma, size=(T + burn, n))
    for t in range(2, T + burn):
        e[t] = ar[0] * e[t - 1] + ar[1] * e[t - 2] + innov[t]
    return e[burn:]
