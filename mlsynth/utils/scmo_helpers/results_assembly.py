"""Assemble the typed :class:`SCMOResults` container from per-scheme fits."""

from __future__ import annotations

from typing import Dict

from .structures import CONCATENATED, SCMOInputs, SCMOMethodFit, SCMOResults


def assemble_scmo_results(
    inputs: SCMOInputs,
    fits: Dict[str, SCMOMethodFit],
    selected_variant: str = CONCATENATED,
) -> SCMOResults:
    """Wrap the per-scheme fits into an :class:`SCMOResults`.

    Parameters
    ----------
    inputs : SCMOInputs
        Preprocessed panel.
    fits : dict
        ``{scheme_name: SCMOMethodFit}`` for the schemes that were run.
    selected_variant : str
        Which scheme drives the convenience aliases (``att``, ``counterfactual``,
        ...). Falls back to the first available fit if absent.

    Returns
    -------
    SCMOResults
    """
    if selected_variant not in fits and fits:
        selected_variant = next(iter(fits))
    return SCMOResults(inputs=inputs, fits=fits, selected_variant=selected_variant)
