"""Minimum-detectable-effect (MDE) power analysis for SYNDES designs.

The Doudchenko et al. (2021) Synthetic Design paper computes power
curves by Monte Carlo simulation (Appendix A.4, Figure 2): seed a
known true ATET, run the permutation test, repeat, and tally the
rejection rate. Repeating that procedure across a grid of effect
sizes traces out the rejection probability as a function of the
true effect.

For a single fitted design we can short-circuit that loop by
appealing to the asymptotic normality of the permutation test
statistic under the null. With

    sigma_perm = std_t (Y_t @ c)

— the std of the per-period contrast applied to the **pre-period**
outcome panel (the empirical null distribution of the test
statistic, the same one the moving-block permutation test
samples) — the variance of the ATT estimator over ``n_post``
post-treatment periods is ``sigma_perm^2 / n_post``. The MDE at
significance level ``alpha`` (two-sided) and power ``1 - beta`` is

    MDE_abs(n_post) = (z_{1 - alpha/2} + z_{1 - beta})
                       * sigma_perm / sqrt(n_post),

which we report alongside its percentage version

    MDE_pct(n_post) = 100 * MDE_abs(n_post) / baseline,

where ``baseline`` defaults to the mean pre-period outcome on the
SYNDES-selected treated units (so MDE_pct reads as a percentage of
treated-unit baseline). Other baselines available: ``"overall"``
(full panel mean), ``"control"`` (the SC-weighted control mean
under the design's contrast), or a user-supplied scalar.

Use :func:`power_analysis` as the public entry point; pass it the
:class:`mlsynth.utils.syndes_helpers.structures.SYNDESResults` returned
by :meth:`mlsynth.SYNDES.fit` (or any of the legacy SYNDES modes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthEstimationError
from .inference import _build_contrast_vector
from .structures import SYNDESResults


@dataclass(frozen=True)
class SYNDESPower:
    """Per-horizon MDE table for a fitted SYNDES design.

    Parameters
    ----------
    n_post_periods : np.ndarray
        Horizons evaluated, shape ``(H,)``.
    mde_absolute : np.ndarray
        MDE in the same units as the outcome, shape ``(H,)``.
    mde_percent : np.ndarray
        ``100 * MDE_abs / baseline``, shape ``(H,)``.
    sigma_perm : float
        Std of the per-period contrast applied to the pre-period
        outcomes -- the permutation-null std the MDE rests on.
    baseline : float
        Baseline outcome level used to convert ``mde_absolute`` into a
        percentage.
    baseline_kind : str
        Tag identifying which baseline was used (``"treated"``,
        ``"overall"``, ``"control"``, or ``"custom"``).
    alpha : float
        Two-sided significance level used to build the CI.
    power : float
        Target power ``1 - beta`` used to compute the MDE.
    contrast : np.ndarray
        The unit-level contrast vector that maps outcomes to the ATT
        estimator. Stored for downstream inspection.
    """

    n_post_periods: np.ndarray
    mde_absolute: np.ndarray
    mde_percent: np.ndarray
    sigma_perm: float
    baseline: float
    baseline_kind: str
    alpha: float
    power: float
    contrast: np.ndarray = field(repr=False)

    def to_dataframe(self):
        """Return a tidy ``(n_post, mde_abs, mde_pct)`` DataFrame."""
        import pandas as pd  # local import keeps the helper light

        return pd.DataFrame({
            "n_post": self.n_post_periods,
            "mde_absolute": self.mde_absolute,
            "mde_percent": self.mde_percent,
        })


def power_analysis(
    results: SYNDESResults,
    n_post_periods: Iterable[int] = range(1, 13),
    alpha: float = 0.05,
    power: float = 0.80,
    baseline: Union[str, float] = "treated",
) -> SYNDESPower:
    """Compute the per-horizon minimum detectable effect for a SYNDES design.

    Parameters
    ----------
    results : SYNDESResults
        Output of :meth:`mlsynth.SYNDES.fit` or :meth:`mlsynth.SYNDES.fit`.
        Only the ``design`` and ``inputs`` fields are read.
    n_post_periods : iterable of int, default ``range(1, 13)``
        Horizons (in post-treatment periods) at which to report the
        MDE.
    alpha : float, default 0.05
        Two-sided significance level.
    power : float, default 0.80
        Target power for the MDE (``1 - beta``).
    baseline : str or float, default ``"treated"``
        Denominator for the percentage MDE. Choices:

        * ``"treated"`` (default) -- mean pre-period outcome over the
          SYNDES-selected treated units.
        * ``"overall"``         -- mean pre-period outcome over every unit.
        * ``"control"``         -- SC-weighted mean pre-period control
          outcome implied by the design's contrast.
        * float                   -- user-supplied baseline value.

    Returns
    -------
    SYNDESPower
        Frozen container with per-horizon MDE in absolute and
        percentage units.
    """

    inputs = results.inputs
    design = results.design

    Y_pre = np.asarray(inputs.Y_pre, dtype=float)
    if Y_pre.ndim != 2:
        raise MlsynthEstimationError(
            "power_analysis expects a 2-D pre-period outcome matrix."
        )

    n_units = Y_pre.shape[1]
    contrast = _build_contrast_vector(design, n_units=n_units)
    if contrast.size != n_units:
        raise MlsynthEstimationError(
            f"Contrast length ({contrast.size}) does not match the panel "
            f"width ({n_units})."
        )

    # Permutation-null std: under the sharp null, the per-period
    # contrast Y_t @ c is exchangeable across periods. Its empirical
    # std on the pre-period gives the std of the post-period mean
    # divided by sqrt(n_post).
    per_period = Y_pre @ contrast
    if per_period.size < 2:
        raise MlsynthEstimationError(
            "Need >= 2 pre-treatment periods to estimate the permutation "
            "null variance."
        )
    sigma_perm = float(np.std(per_period, ddof=1))

    # Baseline for the percentage conversion.
    treated_idx = np.asarray(design.selected_unit_indices, dtype=int)
    baseline_kind = baseline if isinstance(baseline, str) else "custom"
    if isinstance(baseline, str):
        if baseline == "treated":
            if treated_idx.size == 0:
                raise MlsynthEstimationError(
                    "baseline='treated' requires at least one treated unit."
                )
            baseline_val = float(np.mean(Y_pre[:, treated_idx]))
        elif baseline == "overall":
            baseline_val = float(np.mean(Y_pre))
        elif baseline == "control":
            if design.control_weights is None:
                # Two-way / per-unit fall back to overall in that case.
                baseline_val = float(np.mean(Y_pre))
                baseline_kind = "overall_fallback"
            else:
                control_w = np.asarray(design.control_weights, dtype=float)
                if control_w.ndim != 1 or control_w.shape[0] != n_units:
                    baseline_val = float(np.mean(Y_pre))
                    baseline_kind = "overall_fallback"
                else:
                    baseline_val = float(np.mean(Y_pre @ control_w))
        else:
            raise MlsynthEstimationError(
                f"Unknown baseline {baseline!r}; expected one of "
                "'treated', 'overall', 'control', or a float."
            )
    else:
        baseline_val = float(baseline)

    if not np.isfinite(baseline_val) or abs(baseline_val) < 1e-12:
        raise MlsynthEstimationError(
            "baseline is zero or non-finite; pass a non-zero float baseline "
            f"explicitly (got baseline_val={baseline_val})."
        )

    if not 0.0 < alpha < 1.0:
        raise MlsynthEstimationError("alpha must lie in (0, 1).")
    if not 0.0 < power < 1.0:
        raise MlsynthEstimationError("power must lie in (0, 1).")

    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    multiplier = z_alpha + z_beta

    horizons = np.asarray(list(n_post_periods), dtype=int)
    if horizons.size == 0:
        raise MlsynthEstimationError("n_post_periods is empty.")
    if np.any(horizons <= 0):
        raise MlsynthEstimationError("n_post_periods entries must be >= 1.")

    mde_abs = multiplier * sigma_perm / np.sqrt(horizons.astype(float))
    mde_pct = 100.0 * mde_abs / baseline_val

    return SYNDESPower(
        n_post_periods=horizons,
        mde_absolute=mde_abs,
        mde_percent=mde_pct,
        sigma_perm=sigma_perm,
        baseline=baseline_val,
        baseline_kind=baseline_kind,
        alpha=float(alpha),
        power=float(power),
        contrast=contrast,
    )
