"""
Post-inference update module for synthetic experimental design.
"""

import numpy as np
from typing import List, Optional

from .structure import SEDCandidate
from .inference import compute_post_inference, compute_moving_block_conformal_ci   # Adjust path if needed


def update_post_inference(
    candidate_results: List[SEDCandidate],
    Y_full: np.ndarray,
    post_idx: np.ndarray,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = 42
) -> List[SEDCandidate]:
    """
    Update all candidates with:
      - Full-timeline synthetic treated, synthetic control, and effects
      - Average Treatment Effect (ATE) over post periods
      - Permutation p-value
      - Block conformal confidence intervals

    Parameters
    ----------
    candidate_results : list of SEDCandidate
        List of evaluated candidates (must have .weights and .identification)
    Y_full : np.ndarray
        Full timeline matrix (pre + post), shape (T_total, N)
    post_idx : np.ndarray
        Indices corresponding to the post-treatment period in Y_full
    n_sims : int
        Number of simulations for permutation test
    alpha : float
        Significance level for conformal CI
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    candidate_results : list of SEDCandidate (updated in place)
    """
    if len(post_idx) == 0:
        return candidate_results

    for cand in candidate_results:
        # Get column indices of the treated units for this candidate
        treated_col_idx = np.asarray(cand.identification.treated_idx, dtype=int)

        # Extract weights
        treated_weights = cand.weights.treated      # shape (m,)
        control_weights = cand.weights.control      # shape (N,)

        # Compute predictions on full timeline
        synth_treated_full = Y_full[:, treated_col_idx] @ treated_weights
        synth_control_full = Y_full @ control_weights

        # Store results
        cand.predictions.synthetic_treated = synth_treated_full
        cand.predictions.synthetic_control = synth_control_full
        cand.predictions.effects = synth_treated_full - synth_control_full

        # Point estimate (ATE)
        post_gap = cand.predictions.effects[post_idx]
        cand.inference.ate = float(np.mean(post_gap))

        # Store metadata
        cand.inference.treated_col_idx = treated_col_idx.tolist()

        # === Inference ===
        # Permutation p-value
        inference_result = compute_post_inference(
            candidate=cand,
            post_idx=post_idx,
            n_perms=n_sims,
            seed=seed
        )
        cand.inference.p_value = inference_result.inference.p_value

        # Block conformal confidence intervals
        cand = compute_moving_block_conformal_ci(
            candidate=cand,
            post_idx=post_idx,
            alpha=alpha,
            seed=seed
        )

    return candidate_results
