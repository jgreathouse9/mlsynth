"""Run the requested PDA variant(s) and assemble typed per-method fits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .structures import FS, L2, LASSO, PDAInputs, PDAMethodFit
from .l2 import fit_l2, l2_ate_inference
from .lasso import fit_lasso, lasso_ate_inference
from .fs import forward_select, fs_ate_inference

# Map config method strings to internal keys.
_NORMALIZE = {"l2": L2, "L2": L2, "LASSO": LASSO, "lasso": LASSO, "fs": FS, "FS": FS}


def resolve_methods(method: str, methods: Optional[List[str]]) -> List[str]:
    """Internal method keys to run (explicit ``methods`` win over ``method``)."""
    chosen = methods if methods else [method]
    return [_NORMALIZE.get(m, m) for m in chosen]


def _weights_dict(beta: np.ndarray, labels: np.ndarray) -> Dict[Any, float]:
    return {labels[i]: float(round(beta[i], 4)) for i in range(len(beta)) if abs(beta[i]) > 1e-8}


def run_pda(
    inputs: PDAInputs, methods: List[str], tau: Optional[float], alpha: float,
) -> Dict[str, PDAMethodFit]:
    """Fit each requested PDA variant with its own paper's inference."""
    y, X, T0 = inputs.y, inputs.X, inputs.T0
    labels = inputs.donor_labels
    fits: Dict[str, PDAMethodFit] = {}

    for m in methods:
        meta: Dict[str, Any] = {}
        selected = None
        if m == L2:
            beta, intercept, cf, tau_used = fit_l2(y, X, T0, tau=tau)
            att, se, ci, p = l2_ate_inference(y, cf, T0, alpha=alpha)
            meta["tau"] = tau_used
        elif m == LASSO:
            beta, intercept, cf, support = fit_lasso(y, X, T0)
            att, se, ci, p = lasso_ate_inference(y, X, cf, support, T0, alpha=alpha)
            selected = [labels[i] for i in np.where(support)[0]]
        elif m == FS:
            sel_idx, beta, intercept, cf = forward_select(y, X, T0)
            att, se, ci, p = fs_ate_inference(y, cf, T0, alpha=alpha)
            selected = [labels[i] for i in sel_idx]
        else:
            raise ValueError(f"Unknown PDA method: {m!r}")

        fits[m] = PDAMethodFit(
            name=m, beta=beta, intercept=intercept, counterfactual=cf,
            gap=y - cf, att=att, att_se=se, ci=ci, p_value=p,
            donor_weights=_weights_dict(beta, labels),
            selected_donors=selected, metadata=meta,
        )
    return fits
