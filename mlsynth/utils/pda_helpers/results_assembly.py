"""Assemble the typed PDAResults container from per-method fits."""

from __future__ import annotations

from typing import Dict

from .structures import PDAInputs, PDAMethodFit, PDAResults


def assemble_pda_results(
    inputs: PDAInputs, fits: Dict[str, PDAMethodFit], selected_variant: str,
) -> PDAResults:
    """Wrap the per-method fits into a PDAResults (selected_variant drives aliases)."""
    if selected_variant not in fits and fits:
        selected_variant = next(iter(fits))
    return PDAResults(inputs=inputs, fits=fits, selected_variant=selected_variant)
