"""Stepwise historical-donor selection for SHC.

Relocated from the shared ``selector_helpers`` grab-bag: this routine is
SHC-specific (it greedily builds the synthetic historical control from the
treated unit's own latent-trend segments, with a BIC-style stop). Its only
dependency is the SHC quadratic program in :mod:`.kernels`, so it now lives
with the rest of the SHC pieces.
"""

from __future__ import annotations

import numpy as np

from .kernels import solve_shc_qp


def stepwise_donor_selection(L_full, L_post, ell_eval, m, varsigma=1e-6, tol=1e-8):
    """Greedy BIC-stopped donor selection for SHC.

    Iteratively adds the historical-donor segment that most reduces the MSE
    between ``ell_eval`` and the simplex-weighted donor combination, stopping
    via a BIC-style penalty.

    Parameters
    ----------
    L_full : np.ndarray, shape (m, N)
        Pre-treatment donor (latent-trend) segments, one per column.
    L_post : np.ndarray, shape (n, N)
        Post-treatment donor segments.
    ell_eval : np.ndarray, shape (m,)
        Target latent-trend segment to match.
    m : int
        Donor-window length (also the BIC sample size).
    varsigma, tol : float
        Low-variance-direction penalty weight and eigenvalue threshold passed
        to :func:`~mlsynth.utils.shc_helpers.kernels.solve_shc_qp`.

    Returns
    -------
    dict
        ``best_donors`` (indices), ``best_weights``, ``best_mse``,
        ``mse_path``, ``bic_path``.
    """
    T0, N = L_full.shape

    varsigma = 1e-6
    tol = 1e-8

    mse_list = []
    weight_list = []
    bic_list = []
    donor_indices = []

    remaining = list(range(N))
    current_donors = []
    lambda_penalty = np.log(m)  # BIC-style penalty

    for j in range(1, N + 1):
        best_mse = np.inf
        best_idx = None
        best_w = None

        for idx in remaining:
            candidate = current_donors + [idx]
            L_j = L_full[:, candidate]
            w, _ = solve_shc_qp(L_j, ell_eval, use_augmented=False,
                                varsigma=varsigma, tol=tol)
            if w is not None:
                mse = np.mean((ell_eval - L_j @ w) ** 2)
                if mse < best_mse:
                    best_mse, best_idx, best_w = mse, idx, w

        if best_idx is None:
            break

        current_donors.append(best_idx)
        remaining.remove(best_idx)

        bic_j = m * np.log(best_mse) + lambda_penalty * j
        bic_list.append(bic_j)
        if j > 3 and bic_list[-1] > bic_list[-2] and bic_list[-2] > bic_list[-3]:
            break

        donor_indices.append(current_donors.copy())
        mse_list.append(best_mse)
        weight_list.append(best_w)

    best_j = int(np.argmin(mse_list))
    return {
        "best_donors": donor_indices[best_j],
        "best_weights": weight_list[best_j],
        "best_mse": mse_list[best_j],
        "mse_path": mse_list,
        "bic_path": bic_list,
    }
