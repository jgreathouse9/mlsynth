"""Variance decomposition for the mlSC heuristic penalty.

Implements Appendix G of Bottmer (2025): under the simplified hierarchical
random-effects model

    Y_sct = alpha_s + eta_sc + eps_sct,

the unit-level pre-treatment means yield consistent estimates of
``sigma_eps^2`` and ``sigma_y^2``. The mlSC heuristic
``lambda = 2 * sigma_eps^2 / sigma_y^2`` (Section 5.2) is built from these.

The decomposition is computed *only* over the control aggregates — the
treated aggregate is excluded, matching the upstream package and the paper's
description ("taking the average estimated variance across all other
aggregated units except for the treated unit").
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .structures import MLSCInputs


def estimate_variance_components(inputs: MLSCInputs) -> Tuple[float, float]:
    """Return ``(sigma_eps2, sigma_y2)`` per Appendix G of Bottmer (2025).

    Parameters
    ----------
    inputs : MLSCInputs
        Pre-processed mlSC inputs. Uses ``Y_disagg_pre_full``,
        ``disagg_to_agg_full``, and ``treated_agg_idx_full`` to run the
        decomposition over the full disaggregate pre-treatment panel,
        excluding the treated aggregate.

    Returns
    -------
    sigma_eps2 : float
        Estimated within-disaggregate-unit noise variance, averaged across
        non-treated aggregates.
    sigma_y2 : float
        Estimated total disaggregate-outcome variance, averaged across
        non-treated aggregates.
    """

    Y_pre = inputs.Y_disagg_pre_full  # (T0, M_full), rows = time, cols = disagg unit
    disagg_to_agg = inputs.disagg_to_agg_full
    treated = inputs.treated_agg_idx_full

    # Unit pre-treatment means mu_sc, shape (M_full,).
    mu_sc = Y_pre.mean(axis=0)

    sigma_eps_list = []
    sigma_y_list = []
    for s in np.unique(disagg_to_agg):
        if int(s) == int(treated):
            continue
        block_mask = disagg_to_agg == s
        Y_block = Y_pre[:, block_mask]                  # (T0, C_s)
        mu_block = mu_sc[block_mask]                    # (C_s,)

        # Var(eps)_s: mean over c, t of (Y_sct - mu_sc)^2  (Appendix G).
        eps = Y_block - mu_block[np.newaxis, :]
        sigma_eps_list.append(float(np.mean(eps ** 2)))

        # Var(y)_s: empirical variance of Y_sct over (c, t) within block.
        sigma_y_list.append(float(np.var(Y_block)))

    if not sigma_eps_list:
        raise ValueError(
            "Variance decomposition needs at least one control aggregate."
        )

    sigma_eps2 = float(np.mean(sigma_eps_list))
    sigma_y2 = float(np.mean(sigma_y_list))
    # Guard against degenerate panels where outcomes are constant.
    sigma_y2 = max(sigma_y2, 1e-12)
    return sigma_eps2, sigma_y2


def heuristic_lambda(sigma_eps2: float, sigma_y2: float) -> float:
    """Closed-form heuristic ``lambda = 2 * sigma_eps^2 / sigma_y^2``."""
    return 2.0 * sigma_eps2 / max(sigma_y2, 1e-12)
