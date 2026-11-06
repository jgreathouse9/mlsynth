"""Scale-safe descriptive summaries for finite sensitivity trajectories."""

from statistics import mean, median_high, median_low

import numpy as np


def rms(values):
    """Compute RMS without squaring values in their original units."""
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values))
    return float(scale * np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0


def median(values):
    """Average the central observations without overflowing their sum."""
    return (
        mean([float(median_low(values)), float(median_high(values))])
        if len(values)
        else np.nan
    )


def std(values):
    """Return sample SD, or NaN if undefined or outside floating-point range."""
    if len(values) < 2:
        return np.nan
    values = np.asarray(values, dtype=float)
    # Remove the common level before scaling to preserve small differences.
    with np.errstate(over="ignore"):
        shifted = values - values[0]
    if np.isfinite(shifted).all():
        values = shifted
    scale = float(np.max(np.abs(values)))
    with np.errstate(over="ignore"):
        value = float(np.std(values / scale, ddof=1) * scale) if scale else 0.0
    return value if np.isfinite(value) else np.nan
