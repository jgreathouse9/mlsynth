"""Placebo permutation inference for Distributional Synthetic Controls.

Implements the placebo-permutation test of Gunsilius (2023, Algorithm 1),
the distributional analogue of the Abadie-Diamond-Hainmueller (2010)
placebo test and the procedure in the reference DiSCo R package
(``DiSCo_per`` / ``DiSCo_per_iter``).

For every unit :math:`\\iota \\in \\{1, \\dots, J + 1\\}` we pretend it is
the treated unit, fit DSC weights on the *other* units over the
pre-period, aggregate them, build the post-period barycenter, and record
the per-period squared 2-Wasserstein distance between that unit's
observed quantile function and its barycenter,

.. math::

   d_{\\iota t} = \\int_0^1
       \\bigl| F^{-1}_{Y_{\\iota t, N}}(q) - F^{-1}_{Y_{\\iota t}}(q) \\bigr|^2 dq .

If there is a genuine treatment effect on the real treated unit and the
model fits the placebos well pre-treatment, the real unit's post-period
distance should sit in the extreme tail of the placebo distances. The
permutation p-value at post-period :math:`t` is the rank of the treated
unit's distance among all :math:`J + 1` distances,
:math:`p_t = r(d_{1t}) / (J + 1)`, with rank 1 the largest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .aggregation import aggregate_period_weights
from .quantiles import empirical_quantile, sample_quantile_grid
from .weights import solve_simplex_weights


@dataclass(frozen=True)
class DSCInference:
    """Placebo-permutation inference output for DSC.

    Attributes
    ----------
    time_labels : np.ndarray
        All ``T`` period labels (pre and post), the x-axis of the
        distance paths.
    post_time_labels : np.ndarray
        The ``T - T0`` post-treatment labels the p-values refer to.
    treated_distance : np.ndarray
        Squared 2-Wasserstein distance per period for the real treated
        unit, shape ``(T,)``.
    placebo_distances : np.ndarray
        Squared 2-Wasserstein distance per period for each placebo unit
        (each donor treated as pseudo-treated in turn), shape
        ``(J, T)``.
    placebo_units : list
        Donor labels aligned with the rows of ``placebo_distances``.
    p_values : dict
        Mapping ``{post_time_label: p_t}`` with
        :math:`p_t = r(d_{1t}) / (J + 1)` (rank 1 = largest distance).
    n_permutations : int
        Number of placebo units used (``J``).
    """

    time_labels: np.ndarray
    post_time_labels: np.ndarray
    treated_distance: np.ndarray
    placebo_distances: np.ndarray
    placebo_units: List[Any]
    p_values: Dict[Any, float]
    n_permutations: int


def _wasserstein_distance_path(
    inputs,
    target_unit: Any,
    donor_units: List[Any],
    grid: np.ndarray,
    lam: np.ndarray,
    eval_grid: np.ndarray,
) -> np.ndarray:
    """Per-period squared 2-Wasserstein distance for one (target, donors) split.

    Fits per-pre-period simplex weights of ``target_unit`` on
    ``donor_units``, aggregates them with ``lam``, and returns the
    squared 2-Wasserstein distance between the target's observed quantile
    function and the barycenter at every period.
    """
    T0 = inputs.T0
    pre_labels = inputs.time_labels[:T0]
    J = len(donor_units)
    period_w = np.zeros((T0, J))
    for i, t in enumerate(pre_labels):
        treated_vec = empirical_quantile(inputs.cell_samples[(target_unit, t)], grid)
        donor_mat = np.column_stack([
            empirical_quantile(inputs.cell_samples[(u, t)], grid) for u in donor_units
        ])
        period_w[i] = solve_simplex_weights(donor_mat, treated_vec)
    w_hat = aggregate_period_weights(period_w, lam)

    dists = np.empty(inputs.time_labels.size)
    for k, t in enumerate(inputs.time_labels):
        observed = empirical_quantile(inputs.cell_samples[(target_unit, t)], eval_grid)
        barycenter = np.column_stack([
            empirical_quantile(inputs.cell_samples[(u, t)], eval_grid)
            for u in donor_units
        ]) @ w_hat
        dists[k] = float(np.mean((observed - barycenter) ** 2))
    return dists


def placebo_permutation_test(
    inputs,
    *,
    M: int,
    grid_method: str,
    lam: np.ndarray,
    n_eval: int = 200,
    random_state: int = 0,
) -> DSCInference:
    """Run the Gunsilius (2023) Algorithm 1 placebo permutation test.

    Parameters
    ----------
    inputs : DSCInputs
        Preprocessed micro-panel.
    M : int
        Quantile-grid size for the weight-fitting Wasserstein loss.
    grid_method : {"halton", "sobol", "uniform"}
        Quantile-grid sampling rule (matches the main fit).
    lam : np.ndarray
        Length-``T0`` pre-period aggregation weights (the same vector the
        point estimate used).
    n_eval : int
        Number of evenly spaced quantiles used to evaluate the squared
        2-Wasserstein distances.
    random_state : int
        Seed for the quantile grid.
    """
    grid = sample_quantile_grid(M=M, method=grid_method, random_state=random_state)
    eval_grid = np.arange(1, n_eval + 1, dtype=float) / (n_eval + 1)

    treated = inputs.treated_unit_name
    donors = list(inputs.unit_names[1:])

    treated_distance = _wasserstein_distance_path(
        inputs, treated, donors, grid, lam, eval_grid,
    )

    placebo_rows = []
    for iota in donors:
        # The placebo's donor pool is the real treated unit plus every
        # other donor (DiSCo_per_iter): swap iota into the target slot.
        pool = [treated] + [u for u in donors if u != iota]
        placebo_rows.append(
            _wasserstein_distance_path(inputs, iota, pool, grid, lam, eval_grid)
        )
    placebo_distances = np.vstack(placebo_rows) if placebo_rows else np.empty((0, inputs.T))

    J1 = inputs.J + 1
    post_labels = inputs.time_labels[inputs.T0:]
    p_values: Dict[Any, float] = {}
    for k_global in range(inputs.T0, inputs.T):
        d_treated = treated_distance[k_global]
        d_placebo = placebo_distances[:, k_global]
        # Rank 1 = largest distance (most extreme deviation).
        rank = 1 + int(np.sum(d_placebo > d_treated))
        p_values[inputs.time_labels[k_global]] = rank / J1

    return DSCInference(
        time_labels=inputs.time_labels.copy(),
        post_time_labels=post_labels.copy(),
        treated_distance=treated_distance,
        placebo_distances=placebo_distances,
        placebo_units=donors,
        p_values=p_values,
        n_permutations=inputs.J,
    )
