"""Inference for SCMO.

* **Permutation / placebo** (default; Abadie, used by Tian-Lee-Panchenko):
  reassign treatment to each donor in turn, refit the *same* scheme, and rank
  the treated unit's post/pre RMSPE ratio against the placebo distribution.
* **Conformal** (optional): agnostic conformal prediction intervals via
  :func:`mlsynth.utils.inferutils.ag_conformal`, retained from the legacy SCMO.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .estimation import _fit_core
from .structures import SCMOInputs


def rmspe_ratio(y: np.ndarray, cf: np.ndarray, T0: int) -> float:
    """Post/pre root-mean-squared prediction-error ratio (Abadie test statistic)."""
    pre = np.sqrt(np.mean((y[:T0] - cf[:T0]) ** 2))
    post = np.sqrt(np.mean((y[T0:] - cf[T0:]) ** 2))
    return float(post / pre) if pre > 0 else float("inf")


def placebo_test(
    inputs: SCMOInputs, scheme: str, demean: bool = False
) -> Tuple[float, Dict[Any, float]]:
    """Abadie permutation test: p-value + each unit's RMSPE ratio.

    Each donor is treated as a placebo (with the remaining donors as its
    pool; the real treated unit is excluded from placebo pools). The p-value
    is the share of units (treated included) whose RMSPE ratio is at least the
    treated unit's.
    """
    _, cf_t, _, _, _ = _fit_core(
        inputs.Z, inputs.Y, inputs.treated_idx, inputs.donor_idx,
        inputs.T0, scheme, inputs.col_period, demean,
    )
    r_treated = rmspe_ratio(inputs.y_treated, cf_t, inputs.T0)
    ratios: Dict[Any, float] = {inputs.treated_label: r_treated}

    for j in inputs.donor_idx:
        placebo_pool = inputs.donor_idx[inputs.donor_idx != j]
        _, cf_j, _, _, _ = _fit_core(
            inputs.Z, inputs.Y, int(j), placebo_pool,
            inputs.T0, scheme, inputs.col_period, demean,
        )
        label = inputs.unit_index.get_labels([j])[0]
        ratios[label] = rmspe_ratio(inputs.Y[j], cf_j, inputs.T0)

    r_arr = np.array(list(ratios.values()))
    p_value = float(np.mean(r_arr >= r_treated))
    return p_value, ratios


def conformal_intervals(
    inputs: SCMOInputs, counterfactual: np.ndarray, miscoverage_rate: float = 0.1
) -> Optional[np.ndarray]:
    """Post-treatment agnostic conformal intervals, shape ``(T - T0, 2)``."""
    try:
        from ..inferutils import ag_conformal
    except Exception:  # pragma: no cover - optional dependency path
        return None
    T0 = inputs.T0
    lo, hi = ag_conformal(
        inputs.y_treated[:T0], counterfactual[:T0], counterfactual[T0:],
        miscoverage_rate=miscoverage_rate,
    )
    return np.vstack([lo[T0:], hi[T0:]]).T
