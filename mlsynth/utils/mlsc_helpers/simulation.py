r"""README data-generating process for Bottmer (2025) mlSC.

Implements the simulation distributed with the author's reference
package (`multi-level-sc-estimator/README.md`): a hierarchical linear
factor model with state-level loadings and within-state county
deviations.

.. math::

   y_{s, c, t} = (\alpha_s + \eta_{sc}) \cdot f_t + \varepsilon_{sct},

with

* :math:`f_t \sim \mathcal{N}(0, \sigma_{\text{time}}^2)`         — one shared time factor;
* :math:`\alpha_s \sim \mathcal{N}(0, \sigma_{\text{state}}^2)`    — state-level loading;
* :math:`\eta_{sc} \sim \mathcal{N}(0, \sigma_{\text{county}}^2)`  — within-state county deviation;
* :math:`\varepsilon_{sct} \sim \mathcal{N}(0, \sigma_{\text{eps}}^2)` — idiosyncratic shock.

Aggregation to the state level uses equal within-state weights
(:math:`v_{sc} = 1 / C_s`), so :math:`y_{st} = (1 / C_s) \sum_c y_{sct}`.
The default README parameters are
:math:`N_s = 10`, :math:`C_s = 10`, :math:`T = 20`,
:math:`(\sigma_{\text{time}}, \sigma_{\text{state}}, \sigma_{\text{county}},
\sigma_{\text{eps}}) = (1.0, 0.8, 0.5, 0.3)`.

The simulation never adds a treatment effect — the true ATT is exactly
zero, so the Monte Carlo target is **unbiasedness of the estimated
ATT** and value-for-value agreement with the reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MLSCSample:
    """One draw from the README's hierarchical-factor DGP.

    Attributes
    ----------
    df_agg, df_disagg : pd.DataFrame
        Long panels ready for :class:`mlsynth.MLSC`. Columns:
        ``state`` / ``county`` (disagg only) / ``time`` / ``y`` /
        ``treated``.
    data_s : np.ndarray
        Aggregate-level outcome matrix, shape ``(N_states, T)``.
    data_c : np.ndarray
        Disaggregate-level outcome matrix, shape ``(N_states * C_s, T)``.
    n_c : np.ndarray
        Counties per state, shape ``(N_states,)``.
    w_c : list of np.ndarray
        Within-state population weights (each array sums to one). One
        entry per state, in the order the reference impl expects.
    idx : int
        Treated state index.
    t : int
        Treated period index (0-based; the README uses ``t = T - 1``).
    """

    df_agg: pd.DataFrame
    df_disagg: pd.DataFrame
    data_s: np.ndarray
    data_c: np.ndarray
    n_c: np.ndarray
    w_c: List[np.ndarray]
    idx: int
    t: int


def simulate_mlsc_sample(
    N_states: int = 10,
    counties_per_state: int = 10,
    T: int = 20,
    sigma_time:   float = 1.0,
    sigma_state:  float = 0.8,
    sigma_county: float = 0.5,
    sigma_eps:    float = 0.3,
    treated_idx: int = 0,
    treated_t: int | None = None,
    rng: np.random.Generator | None = None,
) -> MLSCSample:
    r"""Draw one sample from the Bottmer (2025) README DGP.

    Parameters
    ----------
    N_states : int, default 10
        Number of aggregate units (e.g. states).
    counties_per_state : int, default 10
        Number of disaggregate units inside each aggregate unit. Constant
        across states in the README; pass a per-state count via the
        reference implementation if you need imbalance.
    T : int, default 20
        Total number of periods.
    sigma_time, sigma_state, sigma_county, sigma_eps : float
        Standard deviations for the time factor, state-level loading,
        within-state county deviation, and idiosyncratic shock.
    treated_idx : int, default 0
        Aggregate unit assigned to treatment.
    treated_t : int or None
        First treated period. Defaults to ``T - 1`` (the README's choice).
    rng : np.random.Generator, optional
        NumPy RNG. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    MLSCSample
    """
    rng = rng or np.random.default_rng()
    if treated_t is None:
        treated_t = T - 1
    n_c = np.full(N_states, counties_per_state, dtype=int)
    n_counties = int(n_c.sum())

    # Latent factor structure (README "Generate data" block).
    time_factor    = rng.normal(0, sigma_time,   size=(T, 1))
    state_loadings = rng.normal(0, sigma_state,  size=(N_states, 1))
    county_comp    = rng.normal(0, sigma_county, size=(n_counties, 1))

    state_idx = np.repeat(np.arange(N_states), counties_per_state)
    county_loadings = state_loadings[state_idx] + county_comp     # (n_counties, 1)
    eps   = rng.normal(0, sigma_eps, size=(n_counties, T))
    data_c = county_loadings @ time_factor.T + eps                 # (n_counties, T)

    # Within-state equal weights (one array per state, the order the
    # reference impl expects).
    w_c = [np.full(n_c[s], 1.0 / n_c[s]) for s in range(N_states)]

    # Aggregate by within-state weighted average.
    data_s = np.zeros((N_states, T))
    for s in range(N_states):
        in_s = np.where(state_idx == s)[0]
        data_s[s, :] = np.average(data_c[in_s, :], axis=0, weights=w_c[s])

    # Long-form panels for the two-DataFrame API. Treatment is assigned
    # at the aggregate (state) level; the reference convention is to
    # mirror the same 0/1 indicator on every county inside the treated
    # state so the disaggregate panel encodes the same identification.
    df_agg = pd.DataFrame([
        {"state": f"s{s}", "time": tt, "y": float(data_s[s, tt]),
         "treated": int(s == treated_idx and tt >= treated_t)}
        for s in range(N_states) for tt in range(T)
    ])
    df_disagg = pd.DataFrame([
        {"county": f"c{c:03d}", "state": f"s{state_idx[c]}",
         "time": tt, "y": float(data_c[c, tt]),
         "treated": int(state_idx[c] == treated_idx and tt >= treated_t)}
        for c in range(n_counties) for tt in range(T)
    ])

    return MLSCSample(df_agg=df_agg, df_disagg=df_disagg,
                       data_s=data_s, data_c=data_c, n_c=n_c, w_c=w_c,
                       idx=treated_idx, t=treated_t)
