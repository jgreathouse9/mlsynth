"""Inference helpers for SYNDES."""

from __future__ import annotations

from typing import Union

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from .relaxed_structures import RelaxedDesign, RelaxedInference
from .structures import SYNDESDesign, SYNDESInference


def _build_contrast_vector(design: SYNDESDesign, n_units: int) -> np.ndarray:
    """Return the unit-level contrast that produces the ATET when applied to Y.

    For all three Doudchenko et al. (2021) MIP formulations the
    estimated ATET at period ``t`` is ``Y_t @ c`` for some unit-level
    contrast vector ``c``:

    * ``global_2way`` / ``global_equal_weights``  --  ``c = 2 q - w``,
      i.e. ``(treated_weight - control_weight)`` per unit. The
      contrast was already cached on the design as
      :py:attr:`SYNDESDesign.contrast_weights`.
    * ``per_unit``  --  for each treated ``i`` the per-unit estimator
      is ``Y_{i,t} - sum_j q_{ij} Y_{j,t}``; averaging across ``K``
      treated units gives ``c_j = (1/K) (D_j - sum_i q_{ij})``.
    """

    # Accept both legacy SYNDES mode names and the paper-aligned SYNDES
    # names so the inference helper is callable from either orchestrator.
    _ALIAS = {
        "two_way_global": "global_2way",
        "one_way_global": "global_equal_weights",
    }
    mode = _ALIAS.get(design.mode, design.mode)
    if mode in {"global_2way", "global_equal_weights"}:
        if design.contrast_weights is not None:
            return np.asarray(design.contrast_weights, dtype=float)
        if design.w is None or design.q is None:
            raise MlsynthEstimationError(
                f"{mode} inference requires w and q weights on the design."
            )
        return 2.0 * np.asarray(design.q) - np.asarray(design.w)

    if mode == "per_unit":
        if design.q is None or design.assignment is None:
            raise MlsynthEstimationError(
                "per_unit inference requires q weights and the assignment "
                "vector on the design."
            )
        q = np.asarray(design.q, dtype=float)
        D = np.asarray(design.assignment, dtype=float)
        K = float(D.sum())
        if K <= 0:
            raise MlsynthEstimationError(
                "per_unit inference requires at least one treated unit."
            )
        # c_j = (1/K) (D_j - sum_i q[i, j]) where q[i, j] is unit i's
        # weight on donor j (per_unit q has shape (N, N), with row i
        # holding the SC weights for treated unit i over the donor
        # pool). The contrast aggregates the K per-unit estimators
        # into an average ATET.
        return (D - q.sum(axis=0)) / K

    raise MlsynthConfigError(
        f"Unknown SYNDES mode {mode!r}; expected one of "
        "{'global_2way', 'global_equal_weights', 'per_unit'}."
    )


def permutation_test_global(
    Y_pre: np.ndarray,
    Y_post: np.ndarray,
    design: SYNDESDesign,
    alpha: float = 0.10,
    include_null_stats: bool = True,
) -> SYNDESInference:
    """Moving-block permutation test for any SYNDES / Synthetic-Design mode.

    Generalises the original ``global_2way``-only implementation to the
    full set of MIP formulations from Doudchenko et al. (2021):
    ``global_2way``, ``global_equal_weights`` (paper's "one-way global")
    and ``per_unit``. The test follows the Chernozhukov, Wuethrich, and
    Zhu (2021) permutation-across-time logic: we treat each period's
    cross-unit contrast as exchangeable under the no-effect null and
    compare the post-period mean to the null distribution obtained by
    cyclically shifting the stacked panel.
    """

    if Y_post is None or Y_post.size == 0:
        raise MlsynthDataError("Y_post is required for SYNDES permutation inference.")

    contrast = _build_contrast_vector(design, n_units=Y_pre.shape[1])

    Y_full = np.vstack([Y_pre, Y_post])
    n_post = Y_post.shape[0]
    total_periods = Y_full.shape[0]

    observed = float(np.mean(Y_post @ contrast))
    u_obs = abs(observed)

    null_stats = []
    for shift in range(total_periods):
        Y_perm = np.roll(Y_full, shift, axis=0)[-n_post:, :]
        null_stats.append(abs(float(np.mean(Y_perm @ contrast))))

    null_stats_arr = np.asarray(null_stats)
    p_value = float(np.mean(null_stats_arr >= u_obs))

    return SYNDESInference(
        atet=observed,
        p_value=p_value,
        reject=p_value <= alpha,
        alpha=alpha,
        method=f"moving_block_permutation_{design.mode}",
        null_stats=null_stats_arr if include_null_stats else None,
    )


def permutation_test_relaxed_global(
    Y_pre: np.ndarray,
    Y_post: np.ndarray,
    design: RelaxedDesign,
    alpha: float = 0.10,
    include_null_stats: bool = True,
) -> RelaxedInference:
    """Moving-block permutation test for a relaxed two-way SYNDES design.

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
            "Y_post is required for relaxed SYNDES permutation inference."
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
