"""Run the requested RESCM corner case(s) and assemble typed per-method fits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .crossval import fit_en_scm, fit_relaxed_scm
from .inference import ate_inference
from .specs import MethodSpec, resolve_specs
from .structures import ELASTIC, RELAXED, RESCMInputs, RESCMMethodFit


def _weights_vector(donor_weights: Dict[Any, float], labels: np.ndarray) -> np.ndarray:
    """Align an engine ``{label: weight}`` dict back to a column vector over labels."""
    return np.array([float(donor_weights.get(lbl, 0.0)) for lbl in labels], dtype=float)


def _run_one(
    spec: MethodSpec,
    inputs: RESCMInputs,
    *,
    tau: Optional[float],
    n_splits: Optional[int],
    n_taus: Optional[int],
    solver: str,
) -> Dict[str, Any]:
    """Dispatch a single spec to its convex-engine entry point."""
    y, X, T0 = inputs.y, inputs.X, inputs.T0
    X_pre, X_post = X[:T0], X[T0:]
    y_pre = y[:T0]
    donor_names = list(inputs.donor_labels)

    common = dict(X_pre=X_pre, y_pre=y_pre, X_post=X_post, y=y, donor_names=donor_names)

    if spec.branch == RELAXED:
        kw = dict(spec.kwargs)
        if tau is not None:
            kw.setdefault("tau", tau)
        if n_splits is not None:
            kw.setdefault("n_splits", n_splits)
        if n_taus is not None:
            kw.setdefault("n_taus", n_taus)
        return fit_relaxed_scm(**common, solver=solver, **kw)

    if spec.branch == ELASTIC:
        kw = dict(spec.kwargs)
        if n_splits is not None:
            kw.setdefault("n_splits", n_splits)
        return fit_en_scm(**common, solver=solver, **kw)

    raise ValueError(f"Unknown RESCM branch: {spec.branch!r}")


def run_rescm(
    inputs: RESCMInputs,
    methods: List[str],
    *,
    tau: Optional[float] = None,
    n_splits: Optional[int] = None,
    n_taus: Optional[int] = None,
    solver: str = "CLARABEL",
    alpha: float = 0.05,
) -> Dict[str, RESCMMethodFit]:
    """Fit each requested RESCM corner case and attach weak-dependence ATE inference."""
    specs = resolve_specs(methods)
    labels = inputs.donor_labels
    fits: Dict[str, RESCMMethodFit] = {}

    for spec in specs:
        raw = _run_one(
            spec, inputs, tau=tau, n_splits=n_splits, n_taus=n_taus, solver=solver,
        )
        cf = np.asarray(raw["predictions"], dtype=float).flatten()
        gap = inputs.y - cf
        weights = _weights_vector(raw["donor_weights"], labels)
        nonzero = {k: v for k, v in raw["donor_weights"].items() if v}
        att, se, ci, p = ate_inference(gap, inputs.T0, alpha=alpha)

        fits[spec.name] = RESCMMethodFit(
            name=spec.name,
            branch=spec.branch,
            display_name=raw.get("Model", spec.name),
            weights=weights,
            intercept=float(raw.get("intercept", 0.0)),
            counterfactual=cf,
            gap=gap,
            att=att,
            att_se=se,
            ci=ci,
            p_value=p,
            donor_weights=nonzero,
            fit_diagnostics=raw.get("Results", {}).get("Fit", {}),
            hyperparameters=raw.get("hyperparameters", {}),
            metadata={"description": spec.description},
        )
    return fits
