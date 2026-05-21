"""Bayesian-bootstrap inference for Sequential SDiD (paper Section 2.3).

For each bootstrap iteration we:

1. Draw ``xi_i ~ Exp(1)`` for every underlying unit (not cohort).
2. Re-weight the cohort-level outcomes:
   ``Y_{a, t}(xi) = sum_{i: A_i = a} Y_{i, t} xi_i / sum_{i: A_i = a} xi_i``.
3. Re-run Algorithm 1 on the perturbed panel.
4. Record the pooled event-study vector.

Wald-type SE/CI come from the sample standard deviation of the bootstrap
replicate matrix. Bootstrap draws are also retained on the result object
in case the user wants quantile-based intervals later.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .algorithm import pooled_event_study, run_sequential_sdid


def _unit_membership(
    df: pd.DataFrame,
    treat: str,
    unitid: str,
    time: str,
    cohort_periods: np.ndarray,
    cohort_labels: list,
    time_labels: np.ndarray,
) -> Tuple[List[np.ndarray], List[int]]:
    """For each cohort column, return the integer row indices (in df) of its
    members and the integer index of the cohort in the Y_agg column order.
    """

    treat_wide = (
        df.pivot(index=time, columns=unitid, values=treat)
        .reindex(index=time_labels)
        .astype(int)
    )
    unit_ids = list(treat_wide.columns)
    unit_to_adoption = {}
    for u in unit_ids:
        col = treat_wide[u].to_numpy()
        positions = np.where(col == 1)[0]
        if positions.size == 0:
            unit_to_adoption[u] = "never_treated"
        else:
            unit_to_adoption[u] = time_labels[int(positions[0])]

    members_per_cohort: List[np.ndarray] = []
    for label in cohort_labels:
        cohort_units = [
            u for u in unit_ids if unit_to_adoption[u] == label
        ]
        member_idx = np.array(
            [unit_ids.index(u) for u in cohort_units], dtype=int
        )
        members_per_cohort.append(member_idx)
    return members_per_cohort, unit_ids


def _Y_units_matrix(
    df: pd.DataFrame, outcome: str, unitid: str, time: str,
    unit_ids: List, time_labels: np.ndarray,
) -> np.ndarray:
    """Pivoted (T, N_units) outcome matrix aligned to ``unit_ids``."""
    Ywide = (
        df.pivot(index=time, columns=unitid, values=outcome)
        .reindex(index=time_labels, columns=unit_ids)
    )
    return np.asarray(Ywide.to_numpy(), dtype=float)


def bayesian_bootstrap_event_study(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    inputs,
    eta: float,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Return ``(n_bootstrap, K + 1)`` matrix of bootstrap-replicate event-study vectors.

    The bootstrap reweighting follows Section 2.3 of Arkhangelsky & Samkov
    (2025): independent ``xi_i ~ Exp(1)`` are drawn for every unit, and the
    cohort-level outcomes are reconstructed as weighted means with those
    weights.
    """

    rng = np.random.default_rng(seed)
    members_per_cohort, unit_ids = _unit_membership(
        df=df, treat=treat, unitid=unitid, time=time,
        cohort_periods=inputs.cohort_periods,
        cohort_labels=list(inputs.cohort_labels),
        time_labels=inputs.time_labels,
    )
    Y_units = _Y_units_matrix(
        df=df, outcome=outcome, unitid=unitid, time=time,
        unit_ids=unit_ids, time_labels=inputs.time_labels,
    )
    T, N_units = Y_units.shape
    A = inputs.Y_agg.shape[1]

    replicates = np.zeros((n_bootstrap, inputs.K + 1))
    for b in range(n_bootstrap):
        xi = rng.exponential(scale=1.0, size=N_units)
        Y_boot = np.zeros((T, A))
        for col, member_idx in enumerate(members_per_cohort):
            if member_idx.size == 0:
                Y_boot[:, col] = inputs.Y_agg[:, col]
                continue
            w = xi[member_idx]
            w_sum = w.sum()
            if w_sum <= 0:
                Y_boot[:, col] = inputs.Y_agg[:, col]
                continue
            Y_boot[:, col] = (Y_units[:, member_idx] * w[None, :]).sum(axis=1) / w_sum

        _, cohort_effects = run_sequential_sdid(
            Y_agg=Y_boot,
            pi=inputs.pi,
            cohort_periods=inputs.cohort_periods,
            treated_cohort_indices=inputs.treated_cohort_indices,
            a_min=inputs.a_min,
            a_max=inputs.a_max,
            K=inputs.K,
            eta=eta,
        )
        replicates[b, :] = pooled_event_study(
            cohort_effects=cohort_effects,
            pi=inputs.pi,
            cohort_periods=inputs.cohort_periods,
            a_min=inputs.a_min,
            a_max=inputs.a_max,
            K=inputs.K,
        )
    return replicates


def wald_intervals(
    tau_hat: np.ndarray,
    bootstrap_draws: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(se, ci)`` Wald-type SE and confidence intervals.

    ``se`` is the sample standard deviation of the bootstrap replicates;
    ``ci`` is the standard normal Wald interval centered at ``tau_hat``.
    """
    from scipy.stats import norm

    se = bootstrap_draws.std(axis=0, ddof=1)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower = tau_hat - z * se
    upper = tau_hat + z * se
    ci = np.column_stack([lower, upper])
    return se, ci
