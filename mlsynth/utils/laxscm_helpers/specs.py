"""Named corner-case estimators of the RESCM convex program.

The legacy API exposed a nested ``models_to_run`` dict where the caller had to
know which ``second_norm`` / ``relaxation`` / ``constraint_type`` / ``alpha``
combination realised which estimator. This module replaces that with a flat
registry of *named* estimators: the user picks methods by name and each name
resolves to the exact engine call.

Every spec dispatches to one of two engine entry points
(:func:`mlsynth.utils.crossval.fit_relaxed_scm` or ``fit_en_scm``); ``kwargs``
are forwarded verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .structures import ELASTIC, RELAXED


@dataclass(frozen=True)
class MethodSpec:
    """How a named RESCM estimator maps onto the convex engine."""

    name: str
    branch: str                                   # RELAXED or ELASTIC
    description: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


# Registry of named corner cases. Each is a special case of the single
# penalized/relaxed convex SCM program.
METHOD_SPECS: Dict[str, MethodSpec] = {
    # --- classic / penalized branch (Opt2.SCopt objective_type="penalized") ---
    "SC": MethodSpec(
        "SC", ELASTIC,
        "Classic Abadie simplex SCM (no penalty: lambda = 0).",
        {"second_norm": "L1_L2", "alpha": 0.0, "lam": 0.0,
         "constraint_type": "simplex", "fit_intercept": False},
    ),
    "LASSO": MethodSpec(
        "LASSO", ELASTIC,
        "L1 (LASSO) penalized SCM; sparse donor weights.",
        {"second_norm": "L1_L2", "alpha": 1.0, "constraint_type": "simplex"},
    ),
    "RIDGE": MethodSpec(
        "RIDGE", ELASTIC,
        "L2 (ridge) penalized SCM; dense shrunken weights.",
        {"second_norm": "L1_L2", "alpha": 0.0, "constraint_type": "simplex"},
    ),
    "ENET": MethodSpec(
        "ENET", ELASTIC,
        "Elastic-net (L1 + L2) penalized SCM; alpha chosen by CV.",
        {"second_norm": "L1_L2", "alpha": None, "constraint_type": "simplex"},
    ),
    "LINF": MethodSpec(
        "LINF", ELASTIC,
        "L-infinity-norm SCM (Wang, Xing & Ye 2025); spreads weight, "
        "nesting equal-weights/DiD as lambda grows.",
        {"second_norm": "L1_INF", "alpha": 0.0, "constraint_type": "simplex"},
    ),
    "L1LINF": MethodSpec(
        "L1LINF", ELASTIC,
        "Mixed L1 + L-infinity penalized SCM.",
        {"second_norm": "L1_INF", "alpha": 0.5, "constraint_type": "simplex"},
    ),
    # --- relaxation branch (Opt2.SCopt objective_type="relaxed") ---
    "RELAX_L2": MethodSpec(
        "RELAX_L2", RELAXED,
        "SCM-relaxation with L2 divergence (Liao, Shi & Zheng 2026).",
        {"relaxation_type": "l2"},
    ),
    "RELAX_ENTROPY": MethodSpec(
        "RELAX_ENTROPY", RELAXED,
        "SCM-relaxation with entropy divergence.",
        {"relaxation_type": "entropy"},
    ),
    "RELAX_EL": MethodSpec(
        "RELAX_EL", RELAXED,
        "SCM-relaxation with empirical-likelihood divergence.",
        {"relaxation_type": "el"},
    ),
}

# Aliases accepted from config for convenience.
_ALIASES = {
    "l2": "RELAX_L2", "relax": "RELAX_L2", "relaxed": "RELAX_L2",
    "entropy": "RELAX_ENTROPY", "el": "RELAX_EL",
    "lasso": "LASSO", "ridge": "RIDGE", "enet": "ENET",
    "linf": "LINF", "l_inf": "LINF", "l1linf": "L1LINF",
    "sc": "SC", "scm": "SC",
}


def normalize_method(name: str) -> str:
    """Map a user-supplied method name to a registry key (case-insensitive)."""
    if name in METHOD_SPECS:
        return name
    key = _ALIASES.get(name.lower())
    if key is None:
        raise ValueError(
            f"Unknown RESCM method {name!r}. "
            f"Choose from {sorted(METHOD_SPECS)} (aliases allowed)."
        )
    return key


def resolve_specs(methods) -> "list[MethodSpec]":
    """Return the ordered list of :class:`MethodSpec` for the requested names."""
    return [METHOD_SPECS[normalize_method(m)] for m in methods]
