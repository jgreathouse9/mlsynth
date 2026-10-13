"""Exponential weights over the expert library: Viviano and Bradic Equations 11-12.

The weights are a softmax of the negative cumulative squared loss on the
weighting window. Two properties of that map decide how the estimator behaves,
and both are asserted in the tests:

* it is invariant to a common shift in the losses, which is what makes a
  numerically safe implementation possible at all;
* ``eta`` interpolates between the simple average (``eta = 0``) and the single
  best expert (``eta`` large).

Where on that interpolation a fit sits is not a detail. Measured on the paper's
own application at its own ``eta = 1/(sqrt(T) var(y)) = 51.43``, the effective
number of experts is 3.69 of 4 -- the weighting averages, it does not select,
and concentrating on the best expert would need ``eta`` 17 to 60 times larger.
:func:`effective_k` is on the result so a caller can see which regime they are in.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthDataError


def exponential_weights(ssr: np.ndarray, eta: float) -> np.ndarray:
    """Equation 12: ``w_k proportional to exp(-eta * cumulative loss_k)``.

    Parameters
    ----------
    ssr : np.ndarray
        Cumulative squared loss per expert on the weighting window, shape ``(K,)``.
    eta : float
        Learning rate. ``0`` gives the simple average; larger concentrates.

    Returns
    -------
    np.ndarray
        Weights on the simplex, shape ``(K,)``.
    """
    ssr = np.asarray(ssr, dtype=float).ravel()
    K = ssr.size
    if K == 0:  # pragma: no cover - build_experts refuses an empty library
        return np.zeros(0)
    uniform = np.full(K, 1.0 / K)
    if not np.isfinite(ssr).all() or eta <= 0.0:
        return uniform
    # Subtracting the minimum is exact, not an approximation: the softmax is
    # invariant to a common shift. Without it exp(-eta * ssr) underflows to zero
    # for every expert whenever eta * ssr is large, and the ratio is lost.
    w = np.exp(-eta * (ssr - ssr.min()))
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - shift prevents this
        return uniform
    return w / total


def effective_k(weights: np.ndarray) -> float:
    """Perplexity of the weights: 1 when one expert carries everything, ``K``
    when the weighting is the simple average.

    This is the number that says whether the exponential weighting selected or
    averaged, which the paper does not report and which decides how much the
    ensembling bought.
    """
    w = np.asarray(weights, dtype=float).ravel()
    w = w[w > 0.0]
    if w.size == 0:  # pragma: no cover - weights are a simplex point
        return 1.0
    return float(np.exp(-np.sum(w * np.log(w))))


def paper_eta(y: np.ndarray, horizon: int) -> float:
    """The learning rate the authors' scripts use: ``1 / (sqrt(horizon) var(y))``.

    Their ``analyze_main_text.R`` writes ``1/(sqrt(88) * var(med_ts))``, where 88
    is the length of the usable series. It evaluates to 51.43 on their panel,
    against the ``best_eta <- 50`` hard-coded in the package's ``.Rhistory`` --
    so the formula is their formalisation of that constant.

    Raises
    ------
    MlsynthDataError
        If the outcome is constant, which makes the formula a division by zero.
    """
    y = np.asarray(y, dtype=float).ravel()
    var = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    if not np.isfinite(var) or var <= 0.0:
        raise MlsynthDataError(
            "The treated outcome is constant on the fitting window, so the "
            "paper's learning rate 1/(sqrt(T) var(y)) divides by zero. Pass an "
            "explicit `eta`, or check the panel: a constant treated series has "
            "no counterfactual to estimate.")
    return 1.0 / (np.sqrt(float(horizon)) * var)
