"""What the expert library is actually doing, which the paper does not report.

Ensembling reduces error when the members err independently. These three
measurements say whether that holds, and on two panels it does not: the error
participation ratio of a four-expert library is 1.08 of 4 on Viviano and Bradic's
six southern states and 1.03 of 4 on Prop 99, with error correlations of 0.85 to
0.99 and 0.49 to 0.98. An equal-weight ensemble of those four is 1.17 and 2.92
times the best single expert's in-window RMSE -- averaging correlated biases
makes the fit worse, not better.

So the library's composition is an empirical question per fit, and these go on
the result, not into a docstring.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

#: A loss this many times the best expert's counts as a failure to fit.
SSR_RATIO = 10.0
#: ...and only matters if the expert still carries at least this much weight.
WEIGHT_FLOOR = 0.05


def error_matrix(predictions: np.ndarray, y: np.ndarray,
                 window: np.ndarray) -> np.ndarray:
    """Per-expert prediction error on ``window``, shape ``(len(window), K)``."""
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    if P.shape[0] == 1 and P.shape[1] != 1:  # pragma: no cover - defensive
        P = P.T
    y = np.asarray(y, dtype=float).ravel()
    return P[window] - y[window, None]


def participation_ratio(errors: np.ndarray) -> float:
    """Effective number of independent error directions, in ``[1, K]``.

    ``(sum s_i^2)^2 / sum s_i^4`` on the singular values. One means every expert
    errs in the same direction, so averaging cannot cancel anything; ``K`` means
    the errors are orthogonal and averaging is worth the full variance
    reduction.

    The matrix is not centred. Its column means are the experts' biases, and
    correlated bias is exactly what this is meant to detect.
    """
    R = np.asarray(errors, dtype=float)
    if R.size == 0:  # pragma: no cover - defensive
        return 1.0
    sv2 = np.linalg.svd(R, compute_uv=False) ** 2
    total = sv2.sum()
    if not np.isfinite(total) or total <= 0.0:
        # An exact library has no error structure. One direction, not a nan.
        return 1.0
    return float(total ** 2 / np.sum(sv2 ** 2))


def error_correlation(errors: np.ndarray) -> np.ndarray:
    """Correlation between the experts' errors, ``(K, K)``.

    ``np.corrcoef`` returns nan for a column with no variance, which a constant
    expert produces, so the standardisation guards the divisor and the diagonal
    is set explicitly.
    """
    R = np.asarray(errors, dtype=float)
    K = R.shape[1]
    sd = R.std(axis=0)
    safe = np.where(sd > 0.0, sd, 1.0)
    Z = (R - R.mean(axis=0)) / safe
    C = Z.T @ Z / max(R.shape[0], 1)
    np.fill_diagonal(C, 1.0)
    return C


def flag_degenerate(names: Sequence[str], errors: np.ndarray, *,
                    ssr: np.ndarray, weights: np.ndarray,
                    ssr_ratio: float = SSR_RATIO,
                    weight_floor: float = WEIGHT_FLOOR) -> Dict[str, str]:
    """Experts that are not doing the job the library assumes they do.

    Two failures, both observed in the wild:

    * constant on the weighting window. Their ``did`` expert is constant on one
      of the paper's own two windows, which leaves the library with a single
      effective member.
    * fitting an order of magnitude worse than the best expert while still
      carrying weight. On Prop 99 the l2 expert fits 51 times worse than the
      best and keeps 14.3 percent of the weight, because ``eta`` is far too
      small to discriminate; that is what drags the ensemble's effect away from
      every well-fitting member's.

    Returns
    -------
    dict
        Expert name to the reason, for the experts that fired. Empty when the
        library is healthy.
    """
    R = np.asarray(errors, dtype=float)
    ssr = np.asarray(ssr, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    best = float(ssr.min()) if ssr.size else 0.0
    out: Dict[str, str] = {}
    # A column built by subtracting a constant is constant only to rounding, so
    # the comparison is relative to the column's own level, not to exact zero.
    tol = 1e-12 * max(float(np.max(np.abs(R))), 1.0)
    for j, name in enumerate(names):
        if R.shape[0] > 1 and float(R[:, j].std()) <= tol:
            out[str(name)] = (
                "constant error on the weighting window, so it contributes no "
                "shape to the ensemble and only a level")
            continue
        if best > 0.0 and ssr[j] >= ssr_ratio * best and weights[j] >= weight_floor:
            out[str(name)] = (
                f"in-window loss {ssr[j] / best:.0f}x the best expert's while "
                f"carrying {weights[j]:.1%} of the weight; eta is too small to "
                f"discriminate and this expert is moving the estimate")
    return out
