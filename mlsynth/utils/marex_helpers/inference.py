"""Placebo/permutation inference for a MAREX synthetic treated-vs-control pair.

The held-out *blank* pre-periods act as placebos: the synthetic treated minus
synthetic control there should be noise, so its distribution calibrates a
permutation p-value and a split-conformal confidence band for the post-period
effect (Abadie & Zhao 2026, OA; Chernozhukov-Wuthrich-Zhu 2021).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .structures import MAREXInference


def compute_inference(
    Y_treated: np.ndarray,
    Y_control: np.ndarray,
    T0: int,
    TcE: int,
    Tb: int,
    alpha: float = 0.05,
    max_combinations: int = 1000,
    random_state: Optional[int] = None,
) -> MAREXInference:
    """Compute permutation inference for one synthetic contrast.

    Parameters
    ----------
    Y_treated, Y_control : np.ndarray
        Synthetic treated / control outcomes over the full timeline, shape ``(T,)``.
    T0 : int
        Number of pre-treatment periods.
    TcE : int
        Start index of the blank (held-out) window.
    Tb : int
        Number of blank periods.
    alpha : float
        Two-sided significance level.
    max_combinations : int
        Number of permutation draws for the global test.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    MAREXInference
    """
    rng = np.random.default_rng(random_state)
    Y_treated = np.asarray(Y_treated, dtype=float)
    Y_control = np.asarray(Y_control, dtype=float)
    T_post = len(Y_treated) - T0

    blank_idx = np.arange(TcE, TcE + Tb)
    post_idx = np.arange(T0, T0 + T_post)

    placebo_effects = Y_treated[blank_idx] - Y_control[blank_idx]
    treated_effects = Y_treated[post_idx] - Y_control[post_idx]
    full_effects = Y_treated - Y_control

    all_effects = np.concatenate([placebo_effects, treated_effects])
    n_total = len(all_effects)
    s_obs = float(np.mean(np.abs(treated_effects))) if T_post > 0 else 0.0
    if T_post > 0 and n_total >= T_post:
        pi = np.array([rng.choice(n_total, size=T_post, replace=False)
                       for _ in range(max_combinations)])
        s_perm = np.mean(np.abs(all_effects[pi]), axis=1)
        global_p = float(np.mean(s_perm >= s_obs))
    else:
        global_p = float("nan")

    if placebo_effects.size and treated_effects.size:
        per_period = np.mean(
            np.abs(placebo_effects)[:, None] >= np.abs(treated_effects)[None, :],
            axis=0,
        )
        q = np.quantile(np.abs(placebo_effects), 1 - alpha)
    else:
        per_period = np.full(T_post, np.nan)
        q = np.nan
    interval = np.column_stack([treated_effects - q, treated_effects + q])
    ci = np.vstack([np.full((T0, 2), np.nan), interval])

    return MAREXInference(
        treated_effects=treated_effects,
        placebo_effects=placebo_effects,
        fulltreated_effects=full_effects,
        s_obs=s_obs,
        global_p_value=global_p,
        per_period_pvals=per_period,
        ci=ci,
        alpha=alpha,
    )
