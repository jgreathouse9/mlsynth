"""Vectorized treatment-*effect* primitives.

Bite-sized, pure functions for the *effect* side of estimator reporting --
ATT, percent ATT, total and standardized effects, and the per-period gap --
kept separate from the goodness-of-fit / loss primitives in
:mod:`mlsynth.utils.fitutils`. Each is a vectorized operation (dot products
over outcome paths) returning *raw* (unrounded) values; callers round only for
display. Higher layers compose effects + fit primitives together:

* :func:`mlsynth.utils.results_helpers.build_effect_submodels` -- the
  standardized ``EffectResult`` sub-models;
* :meth:`mlsynth.utils.resultutils.effects.calculate` -- the legacy display
  dictionaries.

Notation (Shi--Huang): ``T0`` pre-treatment periods, ``T1`` post; ``r`` the
gap vector ``y - y_hat``; ``s^2 = r_pre . r_pre / T0`` the pre-period
residual mean square.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _ravel(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(x, dtype=float).ravel()


def gap(observed: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """Per-period treatment effect ``observed - counterfactual``."""
    return _ravel(observed) - _ravel(counterfactual)


def split_pre_post(
    arr: np.ndarray, n_pre: int, n_post: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice an array into its pre- and post-treatment segments."""
    a = _ravel(arr)
    return a[:n_pre], a[n_pre : n_pre + n_post]


def att(post_gap: np.ndarray) -> float:
    """Average treatment effect on the treated: mean post-period gap."""
    g = _ravel(post_gap)
    return float(g.mean()) if g.size else float("nan")


def total_effect(post_gap: np.ndarray) -> float:
    """Total treatment effect: summed post-period gap (``1 . r_post``)."""
    g = _ravel(post_gap)
    return float(g.sum()) if g.size else float("nan")


def percent_att(att_value: float, counterfactual_post: np.ndarray) -> float:
    """ATT as a percent of the mean post-period counterfactual."""
    cf = _ravel(counterfactual_post)
    mean_cf = float(cf.mean()) if cf.size else 0.0
    return float(100.0 * att_value / mean_cf) if mean_cf != 0 else float("nan")


def standardized_att(pre_gap: np.ndarray, post_gap: np.ndarray) -> float:
    """Standardized ATT ``sqrt(T1) * att / sqrt((T1/T0) * s^2 + s^2)``."""
    r_pre = _ravel(pre_gap)
    r_post = _ravel(post_gap)
    t0, t1 = r_pre.size, r_post.size
    if t0 == 0 or t1 == 0:
        return float("nan")
    mean_sq_resid = float(r_pre @ r_pre / t0)
    denom = np.sqrt((t1 / t0) * mean_sq_resid + mean_sq_resid)
    return float(np.sqrt(t1) * r_post.mean() / denom) if denom != 0 else float("nan")


def percent_gap(post_gap: np.ndarray, counterfactual_post: np.ndarray) -> np.ndarray:
    """Per-period percent effect; ``nan`` where the counterfactual is zero."""
    g = _ravel(post_gap)
    cf = _ravel(counterfactual_post)
    out = np.full_like(g, np.nan)
    nonzero = cf != 0
    out[nonzero] = 100.0 * g[nonzero] / cf[nonzero]
    return out
