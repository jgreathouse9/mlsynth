"""Assemble the typed RESCMResults container from per-method fits."""

from __future__ import annotations

from typing import Dict

from .structures import RESCMInputs, RESCMMethodFit, RESCMResults


def assemble_rescm_results(
    inputs: RESCMInputs, fits: Dict[str, RESCMMethodFit], selected_variant: str,
) -> RESCMResults:
    """Wrap the per-method fits into an RESCMResults (selected_variant drives aliases)."""
    if selected_variant not in fits and fits:
        selected_variant = next(iter(fits))
    return RESCMResults(inputs=inputs, fits=fits, selected_variant=selected_variant)
