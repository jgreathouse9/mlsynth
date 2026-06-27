"""MAREX power analysis: the minimum detectable effect (MDE) of a design.

Mirrors the SYNDES power analysis (``utils/syndes_helpers/power.py``): from a
design's unit-level contrast ``c = w_treated - w_control``, form the pre-period
contrast series ``g_t = (Y_pre^T c)``, estimate its long-run (serial-correlation
-robust, Newey-West) standard deviation, and report the MDE at each post-period
horizon ``h`` as ``(z_{1-alpha/2} + z_power) * sigma / sqrt(h)``.

Reusing SYNDES's ``_newey_west_sigma`` keeps the power math identical across the
two design methods, so MAREX and SYNDES designs are scored the same way.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
from scipy.stats import norm

from ..syndes_helpers.power import _newey_west_sigma


def marex_mde_curve(Y_fit: np.ndarray, contrast: np.ndarray, *,
                    horizons: Iterable[int] = range(1, 13),
                    alpha: float = 0.05, power: float = 0.80,
                    baseline: float | None = None) -> Dict[str, Any]:
    """MDE curve for one design's contrast.

    Parameters
    ----------
    Y_fit : np.ndarray
        Pre-period outcomes, shape ``(N, T_fit)`` (units x time).
    contrast : np.ndarray
        Unit-level contrast ``w_treated - w_control``, shape ``(N,)``.
    horizons : iterable of int
        Post-period horizons to evaluate (default 1..12).
    alpha, power : float
        Two-sided significance level and target power.
    baseline : float, optional
        Treated baseline level for the percent MDE; ``None``/0 -> percent is inf.
    """
    Y_fit = np.asarray(Y_fit, dtype=float)
    c = np.asarray(contrast, dtype=float)
    per_period = Y_fit.T @ c                       # (T_fit,)
    sigma = _newey_west_sigma(per_period)
    mult = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    h = np.asarray(list(horizons), dtype=float)
    mde_abs = mult * sigma / np.sqrt(h)
    if baseline is None or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        mde_pct = np.full_like(h, np.inf)
    else:
        mde_pct = 100.0 * mde_abs / abs(float(baseline))
    return {"horizons": h.astype(int), "mde_abs": mde_abs, "mde_pct": mde_pct,
            "long_run_sigma": float(sigma), "baseline": baseline}
