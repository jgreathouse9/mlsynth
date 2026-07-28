"""ESC pipeline: run the ensemble engine and assemble :class:`ESCResults`."""

from __future__ import annotations

from .config import ESCConfig
from .estimate import fit_esc
from .structures import ESCFit, ESCInputs, ESCResults


def run_esc(inputs: ESCInputs, config: ESCConfig) -> ESCResults:
    """Fit ESC on preprocessed inputs and return the standardized result."""
    fit: ESCFit = fit_esc(inputs, config)
    return ESCResults(inputs=inputs, fit=fit,
                      metadata={"estimator": "ESC", **inputs.metadata})
