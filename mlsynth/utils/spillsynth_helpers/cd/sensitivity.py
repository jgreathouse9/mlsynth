"""Cao-Dowd v3 Section 5.2 pure-donor sensitivity analysis.

Given the same panel and a declared spillover-structure matrix :math:`A`,
this module quantifies the worst-case misspecification bias of two
estimators -- the spillover-aware (SP) estimator and the pure-donor (PD)
SCM -- as a function of the unknown magnitude :math:`\\bar \\alpha` of
*missed* spillovers (i.e.\\ spillovers on units the researcher
incorrectly assumed to be clean).

Section 5.2 of the paper shows that both biases are linear in
:math:`\\bar \\alpha` with coefficients

* SP:  :math:`c_p^{SP} = \\sum_{j=1}^{p} |\\widetilde w_{SP, j}|`,
  where :math:`\\widetilde w_{SP}` is the first row of
  :math:`A (A' (I - B)' (I - B) A)^{-1} A' (I - B)' (I - B) - I_N`,
  restricted to columns corresponding to **clean** controls (the units
  the researcher assumed to be unaffected) and sorted by absolute value
  in descending order;

* Pure-donor: :math:`c_p^{PD} = \\sum_{j=1}^{p} |\\widetilde w_{PD, j}|`,
  where :math:`\\widetilde w_{PD}` is the treated unit's SCM weight
  vector from the pure-donor fit (which drops every assumed-affected
  unit from the donor pool), sorted by absolute value in descending
  order.

For each :math:`p` (number of missed spillovers), the identified bias
sets are :math:`[-c_p^M \\bar\\alpha, +c_p^M \\bar\\alpha]` for
:math:`M \\in \\{SP, PD\\}`. A smaller :math:`c_p^M` means greater
robustness to misspecification.

Figure 3 of v3 plots these bounds against :math:`\\bar\\alpha` for
:math:`p = 1, 2` on the Proposition-99 panel; this module exposes the
raw weight vectors so users can reproduce it or compute the smallest
:math:`\\bar\\alpha` capable of invalidating their estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scm_core import fit_demeaned_sc


@dataclass(frozen=True)
class PureDonorSensitivity:
    """Misspecification-bias weights for SP and pure-donor SCM.

    Parameters
    ----------
    w_sp : np.ndarray
        Length-``(N - 1 - p_declared)`` vector of SP misspecification
        weights on the *clean* control rows of the panel, sorted by
        absolute value in descending order. ``c_p_sp[i] = sum(|w_sp[:i+1]|)``
        is the SP bias bound coefficient when ``i + 1`` spillovers are
        missed.
    w_pd : np.ndarray
        Length-``(N - 1 - p_declared)`` vector of pure-donor SCM weights
        on the clean controls (the treated unit's own SCM weight vector
        after dropping every assumed-affected unit from the donor pool),
        sorted by absolute value in descending order.
    a_pd : float
        Pure-donor SCM intercept (treated unit).
    n_clean : int
        Number of clean controls.

    Notes
    -----
    For a missed spillover of magnitude at most :math:`\\bar\\alpha`
    on ``p`` of the assumed-clean units, the worst-case bias bounds are

        SP:  :math:`\\pm \\sum_{j=1}^{p} \\text{w\\_sp}[j-1] \\cdot \\bar\\alpha`,
        PD:  :math:`\\pm \\sum_{j=1}^{p} \\text{w\\_pd}[j-1] \\cdot \\bar\\alpha`.

    Compare the two to assess which method is more robust to
    misspecification on this particular panel + A choice.
    """

    w_sp: np.ndarray
    w_pd: np.ndarray
    a_pd: float
    n_clean: int

    def bias_bounds(
        self, p: int, alpha_bar_grid: np.ndarray,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Linear bias-bound curves for SP and PD over a grid of α̅.

        Returns a pair of length-``len(alpha_bar_grid)`` arrays:
        ``(c_sp * grid, c_pd * grid)`` where
        ``c_sp = sum(|w_sp[:p]|)`` and likewise for PD.
        """
        if p < 1 or p > self.n_clean:
            raise ValueError(
                f"p must satisfy 1 <= p <= n_clean ({self.n_clean}); got {p}."
            )
        c_sp = float(np.sum(self.w_sp[:p]))
        c_pd = float(np.sum(self.w_pd[:p]))
        grid = np.asarray(alpha_bar_grid, dtype=float)
        return c_sp * grid, c_pd * grid


def pure_donor_sensitivity(
    Y_pre: np.ndarray, B: np.ndarray, A: np.ndarray,
) -> PureDonorSensitivity:
    """Compute the SP-vs-PD misspecification-bias weight vectors.

    Parameters
    ----------
    Y_pre : np.ndarray
        Shape ``(N, T0)`` pre-treatment outcome panel.
    B : np.ndarray
        Shape ``(N, N)`` leave-one-out SCM weight matrix from the
        full-panel fit.
    A : np.ndarray
        Shape ``(N, k)`` declared spillover-structure matrix.

    Returns
    -------
    PureDonorSensitivity
    """
    N = Y_pre.shape[0]
    I_B = np.eye(N) - B
    M = I_B.T @ I_B + 1e-12 * np.eye(N)

    # SP misspecification weights: row 0 of (A(A'MA)^{-1}A'(I-B)'(I-B) - I).
    AMA_inv = np.linalg.inv(A.T @ M @ A)
    Q = A @ AMA_inv @ A.T @ I_B.T @ I_B - np.eye(N)
    w_sp_row = Q[0]                                      # (N,)

    # Restrict to clean control columns: rows of A that are entirely zero
    # (the units the researcher assumed unaffected).
    clean_mask = (np.abs(A).sum(axis=1) == 0)            # (N,)
    w_sp_clean = np.sort(np.abs(w_sp_row[clean_mask]))[::-1]

    # Pure-donor SCM: drop assumed-affected rows from the panel, fit the
    # treated unit (row 0) against the remaining controls.
    keep_mask = clean_mask.copy()
    keep_mask[0] = True                                  # always keep treated
    Y_pre_PD = Y_pre[keep_mask]                          # (n_PD, T0)
    a_pd, b_full_PD = fit_demeaned_sc(Y_pre_PD)
    # b_full_PD has length n_PD; first entry is 0 (treated unit), rest are
    # the donor weights on clean controls in row order.
    w_pd_clean = np.sort(np.abs(b_full_PD[1:]))[::-1]

    return PureDonorSensitivity(
        w_sp=w_sp_clean,
        w_pd=w_pd_clean,
        a_pd=float(a_pd),
        n_clean=int(clean_mask.sum()),
    )
