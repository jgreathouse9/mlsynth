"""Inference for FSC: conformal prediction bands and the placebo test.

Both are section 4.3, and both are cheap because the weights are held fixed.

The band inverts the sharp null :math:`Y^N_{1t}(x) = y_0` pointwise in
:math:`x`, comparing the post-treatment residual with the :math:`T_0`
pre-treatment ones. The inversion has a closed form: the accepted set is the
interval centred on the estimate with half-width the appropriate quantile of the
absolute pre-treatment residuals. Note the departure from Chernozhukov, Wüthrich
& Zhu (2021) that the paper makes deliberately -- the weights are *not* refit
under each candidate null, which is what guarantees the band contains the point
estimate but costs the exchangeability argument. The paper conjectures
asymptotic validity rather than proving it.

The placebo test re-runs the whole estimator with each donor cast as the treated
unit and ranks the treated unit's effect magnitude among them. With :math:`N`
units the smallest attainable p-value is :math:`1/N`, so the test's resolution
is set by the donor pool, not the data.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError


def conformal_halfwidth(values: np.ndarray, n_pre: int, synthetic: np.ndarray,
                        alpha: float) -> np.ndarray:
    """Pointwise half-width ``q_alpha(x)`` of the prediction band.

    Parameters
    ----------
    values : numpy.ndarray
        The ``(N, T, M)`` cube of embedded objects.
    n_pre : int
        Number of pre-treatment periods.
    synthetic : numpy.ndarray
        The ``(T, M)`` counterfactual objects.
    alpha : float
        Nominal level.

    Returns
    -------
    numpy.ndarray
        Half-width at each grid point, shape ``(M,)``.

    Raises
    ------
    MlsynthEstimationError
        ``alpha <= 1 / (n_pre + 1)``, where the inverted test excludes nothing
        and the band is unbounded.
    """
    if alpha <= 1.0 / (n_pre + 1.0):
        raise MlsynthEstimationError(
            f"conformal_alpha={alpha} is not above 1 / (T0 + 1) = "
            f"{1.0 / (n_pre + 1.0):.4f} with {n_pre} pre-treatment period(s). "
            "Below that threshold the inverted test never excludes a value and "
            "the band is unbounded. Raise conformal_alpha, lengthen the "
            "pre-period, or use inference='placebo'.")
    residuals = np.abs(values[0, :n_pre] - synthetic[:n_pre])   # (T0, M)
    level = 1.0 - ((n_pre + 1.0) * alpha - 1.0) / n_pre
    return np.quantile(residuals, level, axis=0, method="linear")


def placebo_p_values(values: np.ndarray, n_pre: int,
                     estimate: Callable[[np.ndarray], np.ndarray]
                     ) -> np.ndarray:
    """Per-post-period p-values by casting each donor as treated.

    ``estimate`` maps a cube whose unit 0 is the candidate treated unit to that
    unit's ``(T, M)`` counterfactual, so the placebo runs are the same
    computation as the real one -- including the penalty selection.
    """
    n_units, n_times, _ = values.shape
    n_post = n_times - n_pre
    magnitudes = np.empty((n_units, n_post))
    for i in range(n_units):
        order = [i] + [j for j in range(n_units) if j != i]
        rotated = values[order]
        gap = rotated[0, n_pre:] - estimate(rotated)[n_pre:]
        magnitudes[i] = np.sqrt(np.sum(gap ** 2, axis=1))
    return (magnitudes[0] <= magnitudes).mean(axis=0)


def band_from_halfwidth(synthetic: np.ndarray, halfwidth: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` band on the counterfactual objects."""
    return synthetic - halfwidth, synthetic + halfwidth
