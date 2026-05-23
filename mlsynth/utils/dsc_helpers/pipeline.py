"""Orchestration pipeline for Distributional Synthetic Controls.

Implements the four-step procedure of Gunsilius (2023) as formalised
in Algorithm 1 of Zhang, Zhang & Zhang (2026):

1. Estimate the empirical quantile function
   :math:`\\widehat F^{-1}_{Y_{jt, n_j}}` for every ``(unit, time)``
   cell.
2. For each pre-period :math:`t \\in \\mathcal T_0`, draw an
   :math:`M`-point quantile grid :math:`\\{V_m\\}`, build the
   pseudo-sample matrix :math:`\\widetilde Y_t`, and solve the
   simplex-constrained Wasserstein regression
   :math:`\\widehat w_t = \\arg\\min_{w \\in \\mathcal H}
   \\| \\widetilde Y_t w - \\widehat Y_{1t} \\|_2^2`.
3. Aggregate across the pre-periods:
   :math:`\\widehat w = \\sum_t \\lambda_t \\widehat w_t`.
4. For each post-period :math:`t \\in \\mathcal T_1`, predict the
   counterfactual quantile function
   :math:`\\widehat F^{-1}_{Y_{1t, N}}(q) = \\sum_{j=2}^{J+1}
   \\widehat w_j\\, \\widehat F^{-1}_{Y_{jt, n_j}}(q)` and form the
   quantile treatment effect
   :math:`\\widehat \\alpha_{1t, q} = \\widehat F^{-1}_{Y_{1t, I}}(q)
   - \\widehat F^{-1}_{Y_{1t, N}}(q)`.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..results_helpers import make_weights_results
from .aggregation import aggregate_period_weights, build_lambda_weights
from .quantiles import (
    build_pseudo_sample_matrix,
    empirical_quantile,
    sample_quantile_grid,
)
from .structures import DSCInputs, DSCResults, QTECurve
from .weights import solve_simplex_weights, wasserstein_loss_at_weights


def run_dsc(
    inputs: DSCInputs,
    *,
    M: Optional[int] = None,
    grid_method: Literal["halton", "sobol", "uniform"] = "halton",
    lambda_method: Literal["uniform", "recency"] = "uniform",
    lambda_decay: float = 0.9,
    lambda_weights: Optional[Sequence[float]] = None,
    qte_quantiles: Optional[Sequence[float]] = None,
    n_qte_points: int = 99,
    random_state: int = 0,
) -> DSCResults:
    """Run Algorithm 1 of Zhang, Zhang & Zhang (2026) and assemble :class:`DSCResults`.

    Parameters
    ----------
    inputs : DSCInputs
        Preprocessed micro-panel; build via
        :func:`mlsynth.utils.dsc_helpers.setup.prepare_dsc_inputs`.
    M : int, optional
        Number of quantile-grid points :math:`V_m`. If ``None``,
        defaults to ``max(200, min_cell_size)`` where ``min_cell_size``
        is the smallest within-cell sample size in the panel
        (Zhang et al. 2026 suggest ``M = C * n`` for some constant C).
    grid_method : {"halton", "sobol", "uniform"}
        Sampling rule for the quantile grid. ``"halton"`` (default)
        and ``"sobol"`` are deterministic QMC sequences with
        Koksma-Hlawka error :math:`O(\\log M / M)`; ``"uniform"``
        gives i.i.d. :math:`U[0, 1]` with :math:`O(M^{-1/2})` error.
    lambda_method : {"uniform", "recency"}
        Default aggregation rule for the pre-period weights. Ignored
        if ``lambda_weights`` is provided explicitly.
    lambda_decay : float
        Geometric decay for ``lambda_method="recency"``.
    lambda_weights : sequence of float, optional
        Caller-supplied length-``T0`` aggregation weights. Useful for
        Arkhangelsky-et-al-style SDiD time weights computed outside
        this estimator.
    qte_quantiles : sequence of float, optional
        Quantile grid in ``(0, 1)`` at which to report the QTE. If
        ``None``, an evenly spaced grid of ``n_qte_points`` quantiles
        between ``1/(n_qte_points + 1)`` and ``n_qte_points/(n_qte_points + 1)``
        is used.
    n_qte_points : int
        Length of the default QTE grid.
    random_state : int
        Seed forwarded to the QMC sampler.

    Returns
    -------
    DSCResults
        Frozen container with donor weights, per-pre-period weights,
        QTE curves per post-period, the averaged QTE, and a single
        scalar ATT summary.
    """
    if inputs.J < 2:
        raise MlsynthEstimationError("DSC needs at least 2 donor units.")

    # Default M: large enough for QMC error to be small relative to the
    # within-cell empirical-quantile noise.
    if M is None:
        min_n = min(arr.size for arr in inputs.cell_samples.values())
        M = max(200, min_n)
    if M < 50:
        raise MlsynthEstimationError("M < 50 is too small for the Wasserstein loss.")

    V = sample_quantile_grid(M=M, method=grid_method, random_state=random_state)

    # ------------------------------------------------------------------
    # Steps 1 & 2: per-pre-period simplex regression on pseudo-samples.
    # ------------------------------------------------------------------
    T0 = inputs.T0
    J = inputs.J
    period_weight_matrix = np.zeros((T0, J))
    period_loss = np.zeros(T0)
    pre_period_labels = inputs.time_labels[:T0]
    for i, t in enumerate(pre_period_labels):
        donor_mat, treated_vec = build_pseudo_sample_matrix(inputs, t, V)
        w_t = solve_simplex_weights(donor_mat, treated_vec)
        period_weight_matrix[i] = w_t
        period_loss[i] = wasserstein_loss_at_weights(donor_mat, treated_vec, w_t)

    # ------------------------------------------------------------------
    # Step 3: aggregate weights across pre-periods.
    # ------------------------------------------------------------------
    if lambda_weights is not None:
        lam = np.asarray(lambda_weights, dtype=float)
        if lam.shape != (T0,):
            raise MlsynthEstimationError(
                f"lambda_weights has length {lam.size}; expected T0={T0}."
            )
        if not np.all(lam >= -1e-12):
            raise MlsynthEstimationError("lambda_weights must be non-negative.")
        if abs(float(lam.sum()) - 1.0) > 1e-6:
            raise MlsynthEstimationError("lambda_weights must sum to 1.")
    else:
        lam = build_lambda_weights(
            T0=T0, method=lambda_method, decay=lambda_decay,
        )
    w_hat = aggregate_period_weights(period_weight_matrix, lam)

    donor_names = inputs.unit_names[1:]
    donor_weights = {name: float(w) for name, w in zip(donor_names, w_hat)}
    period_weights = {
        t: period_weight_matrix[i].copy()
        for i, t in enumerate(pre_period_labels)
    }

    # ------------------------------------------------------------------
    # Step 4: per-post-period QTE.
    # ------------------------------------------------------------------
    if qte_quantiles is None:
        if n_qte_points < 1:
            raise MlsynthEstimationError("n_qte_points must be >= 1.")
        q_grid = np.arange(1, n_qte_points + 1, dtype=float) / (n_qte_points + 1)
    else:
        q_grid = np.asarray(qte_quantiles, dtype=float)
        if np.any((q_grid <= 0.0) | (q_grid >= 1.0)):
            raise MlsynthEstimationError(
                "qte_quantiles must lie strictly in (0, 1)."
            )

    post_labels = inputs.time_labels[T0:]
    qte_curves: list[QTECurve] = []
    avg_qte = np.zeros_like(q_grid)
    for t in post_labels:
        treated_sample = inputs.cell_samples[(inputs.treated_unit_name, t)]
        observed_qf = empirical_quantile(treated_sample, q_grid)
        # Counterfactual quantile function = donor-weight-weighted average
        # of the donor quantile functions evaluated at the same grid.
        donor_qfs = np.column_stack([
            empirical_quantile(inputs.cell_samples[(unit, t)], q_grid)
            for unit in donor_names
        ])
        counterfactual_qf = donor_qfs @ w_hat
        qte = observed_qf - counterfactual_qf
        qte_curves.append(QTECurve(
            time_label=t,
            quantiles=q_grid.copy(),
            observed=observed_qf,
            counterfactual=counterfactual_qf,
            qte=qte,
        ))
        avg_qte += qte
    avg_qte /= max(post_labels.size, 1)

    # Single scalar ATT: average over quantiles AND post-periods. Same as
    # the L^1-aggregation of QTE -- a natural mean-of-distribution shift.
    att = float(avg_qte.mean())

    metadata = {
        "M": int(M),
        "grid_method": grid_method,
        "lambda_method": (
            "custom" if lambda_weights is not None else lambda_method
        ),
        "lambda_decay": float(lambda_decay) if lambda_method == "recency" else None,
        "random_state": int(random_state),
        "n_qte_points": int(q_grid.size),
    }

    weights_res = make_weights_results(
        donor_weights, constraint="simplex (non-negative, sum to 1)",
        extra={"aggregation": "w_hat = sum_t lambda_t w_t over pre-periods"},
    )

    return DSCResults(
        inputs=inputs,
        donor_weights=donor_weights,
        period_weights=period_weights,
        lambda_weights=lam,
        qte_curves=qte_curves,
        average_qte=avg_qte,
        att=att,
        pre_period_wasserstein=period_loss,
        weights=weights_res,
        metadata=metadata,
    )
