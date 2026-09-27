"""Step 3 of CPDA: which donors enter Equation 13.

Hsiao and Zhou (2019) say the subset "can be chosen using a model selection
criterion as in Hsiao, Ching, and Wan (2012), or the least absolute shrinkage
and selection operator (LASSO) method (Tibshirani, 1996), as suggested by Li
and Bell (2017)", and stop there.

That sentence does not pin the answer. Measured on their own Table 9 panel,
19 pre-periods against 38 donors, the selectors below span a mean absolute
effect of 3.79 to 14.04 around a published 9.56 -- a factor of 3.7. Under
leave-one-pre-period-out error with the selection repeated inside every fold,
the lowest error belongs to ``lasso_cv``, and the two selectors landing nearest
the published value score worst on it, so nothing measurable from the
pre-period picks the published number out.

``lasso_cv`` is the default for that reason and not because it agrees with
anything. The rest are here so a caller can see the spread, which
:func:`sweep_selectors` reports.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LassoCV, lars_path

#: Every selector this module offers, in the order a sweep reports them.
SELECTORS: Tuple[str, ...] = ("lasso_cv", "lasso_bic", "aicc", "all")


def _standardized(Z: np.ndarray) -> np.ndarray:
    sd = Z.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (Z - Z.mean(axis=0)) / sd


def _folds(T0: int) -> int:
    return int(min(10, max(2, T0 // 3)))


def _ic_path(design: np.ndarray, target: np.ndarray, criterion: str) -> np.ndarray:
    """Walk the LASSO path and keep the knot minimising AICc or BIC.

    The criterion is evaluated on the least-squares refit at each knot, which
    is what Equation 13 fits, so the penalty only ever proposes a subset.
    """
    n = design.shape[0]
    scale = float(target @ target) or 1.0
    _, _, coefs = lars_path(design, target - target.mean(), method="lasso")
    best, best_ic = np.array([], dtype=int), np.inf
    for j in range(coefs.shape[1]):
        keep = np.flatnonzero(np.abs(coefs[:, j]) > 0)
        if keep.size > n - 3:
            continue
        A = np.column_stack([np.ones(n), design[:, keep]]) if keep.size \
            else np.ones((n, 1))
        resid = target - A @ np.linalg.lstsq(A, target, rcond=None)[0]
        ssr = float(resid @ resid)
        # A knot that interpolates drives log(ssr/n) to minus infinity and wins
        # every comparison, so the criterion would always name the widest
        # subset it can reach. Floating point makes ssr small but rarely zero,
        # so the floor is relative to the target's own scale.
        if ssr <= 1e-12 * scale:
            continue
        p = keep.size + 2
        if criterion == "aicc":
            if n - p - 1 <= 0:
                continue
            ic = n * np.log(ssr / n) + 2 * p + 2 * p * (p + 1) / (n - p - 1)
        else:
            ic = n * np.log(ssr / n) + p * np.log(n)
        if ic < best_ic:
            best, best_ic = keep, ic
    return best


def select_donors(design: np.ndarray, target: np.ndarray, *, selector: str,
                  standardize: bool = False, seed: int = 0) -> np.ndarray:
    """Indices of the donors a rule keeps, from the pre-period alone.

    Selection only. Equation 13 then estimates ``mu`` and ``w`` by least
    squares on the raw columns, so a penalty never sets a coefficient.

    Parameters
    ----------
    design : np.ndarray
        Pre-period donor residuals, shape ``(T0, N)``.
    target : np.ndarray
        Pre-period treated residual, shape ``(T0,)``.
    selector : str
        One of :data:`SELECTORS`.
    standardize : bool
        Rescale the columns before the penalty. Off by default: this design is
        donor residuals, one measurement, where a donor's own spread carries
        information and rescaling would assert that a quiet donor should be as
        easy to select as a volatile one. Turn it on for a design whose columns
        are different measurements, where an unstandardized penalty cannot
        reach the small-scale ones at any weight.
    seed : int
        Seed for the cross-validation split, where the rule uses one.

    Returns
    -------
    np.ndarray
        Column indices into ``design``, sorted.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    n, N = design.shape
    if selector == "all" or N == 0:
        return np.arange(N)

    Z = _standardized(design) if standardize else design
    if selector == "lasso_cv":
        if n < 4:                     # too short to split; keep everything
            return np.arange(N)
        fit = LassoCV(cv=_folds(n), max_iter=100000, random_state=seed).fit(Z, target)
        keep = np.flatnonzero(np.abs(fit.coef_) > 0)
    elif selector in ("lasso_bic", "aicc"):
        keep = _ic_path(Z, target, "bic" if selector == "lasso_bic" else "aicc")
    else:  # pragma: no cover - the config's pattern rejects anything else
        raise ValueError(f"unknown selector {selector!r}")

    # An empty subset leaves Equation 11 with only its intercept, which is a
    # valid fit but a silent one, so fall back to the full pool and let the
    # least-squares step decide.
    return keep if keep.size else np.arange(N)
