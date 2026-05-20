"""Inference helpers for SCDI."""

from __future__ import annotations

from typing import Union

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from .relaxed_structures import RelaxedDesign, RelaxedInference
from .structures import SCDIDesign, SCDIInference


def permutation_test_global(
    Y_pre: np.ndarray,
    Y_post: np.ndarray,
    design: SCDIDesign,
    alpha: float = 0.10,
    include_null_stats: bool = True,
) -> SCDIInference:
    """Run a moving-block permutation test for a global two-way SCDI design."""

    if design.mode != "global_2way":
        raise MlsynthConfigError(
            "Permutation inference is currently implemented only for global_2way."
        )
    if design.w is None or design.q is None:
        raise MlsynthEstimationError("Global SCDI inference requires w and q weights.")
    if Y_post is None or Y_post.size == 0:
        raise MlsynthDataError("Y_post is required for SCDI permutation inference.")

    Y_full = np.vstack([Y_pre, Y_post])
    n_post = Y_post.shape[0]
    total_periods = Y_full.shape[0]

    contrast = 2 * np.asarray(design.q) - np.asarray(design.w)
    observed = float(np.mean(Y_post @ contrast))
    u_obs = abs(observed)

    null_stats = []
    for shift in range(total_periods):
        Y_perm = np.roll(Y_full, shift, axis=0)[-n_post:, :]
        null_stats.append(abs(float(np.mean(Y_perm @ contrast))))

    null_stats_arr = np.asarray(null_stats)
    p_value = float(np.mean(null_stats_arr >= u_obs))

    return SCDIInference(
        atet=observed,
        p_value=p_value,
        reject=p_value <= alpha,
        alpha=alpha,
        method="moving_block_permutation_global",
        null_stats=null_stats_arr if include_null_stats else None,
    )


def permutation_test_relaxed_global(
    Y_pre: np.ndarray,
    Y_post: np.ndarray,
    design: RelaxedDesign,
    alpha: float = 0.10,
    include_null_stats: bool = True,
) -> RelaxedInference:
    """Moving-block permutation test for a relaxed two-way SCDI design.

    Mirrors :func:`permutation_test_global` but consumes the relaxed
    solver's ``contrast_weights`` directly rather than reconstructing
    them from ``q`` and ``w``.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(T_pre, N)``.
    Y_post : np.ndarray
        Post-treatment outcome matrix of shape ``(T_post, N)``.
    design : RelaxedDesign
        Best-state design from :func:`solve_two_way_relaxed`.
    alpha : float, optional
        Significance level for the test.
    include_null_stats : bool, optional
        Whether to attach the empirical null distribution to the result.

    Returns
    -------
    RelaxedInference
        Permutation-inference output.

    Raises
    ------
    MlsynthDataError
        If ``Y_post`` is missing or empty.
    """

    if Y_post is None or Y_post.size == 0:
        raise MlsynthDataError(
            "Y_post is required for relaxed SCDI permutation inference."
        )

    Y_full = np.vstack([Y_pre, Y_post])
    n_post = Y_post.shape[0]
    total_periods = Y_full.shape[0]

    contrast = np.asarray(design.contrast_weights)
    observed = float(np.mean(Y_post @ contrast))
    u_obs = abs(observed)

    null_stats = []
    for shift in range(total_periods):
        Y_perm = np.roll(Y_full, shift, axis=0)[-n_post:, :]
        null_stats.append(abs(float(np.mean(Y_perm @ contrast))))

    null_stats_arr = np.asarray(null_stats)
    p_value = float(np.mean(null_stats_arr >= u_obs))

    return RelaxedInference(
        atet=observed,
        p_value=p_value,
        reject=p_value <= alpha,
        alpha=alpha,
        method="moving_block_permutation_relaxed_global",
        null_stats=null_stats_arr if include_null_stats else None,
    )
