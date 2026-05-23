"""Shared helper for building standardized :class:`WeightsResults`.

Lets each estimator expose the donor weights underlying its counterfactual
through the same pydantic model (no black boxes), with consistent summary
statistics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..config_models import WeightsResults


def make_weights_results(
    donor_weights: Dict[Any, float],
    constraint: str,
    extra: Optional[Dict[str, Any]] = None,
) -> WeightsResults:
    """Wrap a ``{donor: weight}`` mapping in a standardized WeightsResults.

    Parameters
    ----------
    donor_weights : dict
        Mapping of donor unit name -> weight.
    constraint : str
        Human-readable description of the weight constraint (e.g.
        ``"simplex (non-negative, sum to 1)"`` or
        ``"unconstrained regression weights"``).
    extra : dict, optional
        Additional fields merged into ``summary_stats`` (e.g. time weights,
        intercepts, per-unit weight maps).
    """
    # WeightsResults.donor_weights requires string keys; coerce so unit ids
    # that are integers (a common panel convention) validate cleanly.
    donor_weights = {str(k): float(v) for k, v in (donor_weights or {}).items()}
    w = np.array(list(donor_weights.values()), dtype=float) if donor_weights \
        else np.array([])
    summary: Dict[str, Any] = {
        "sum_of_weights": float(w.sum()) if w.size else 0.0,
        "n_donors": int(w.size),
        "n_nonzero": int((np.abs(w) > 1e-6).sum()),
        "n_negative": int((w < 0).sum()),
        "max_abs_weight": float(np.abs(w).max()) if w.size else 0.0,
        "constraint": constraint,
    }
    if extra:
        summary.update(extra)
    return WeightsResults(donor_weights=donor_weights, summary_stats=summary)
