"""Top-level SPCD solver — glues the four algorithm steps together.

The flow follows Algorithm 1 (page 7) and Algorithm 2 (page 13) of the
paper, both of which share the same skeleton:

    1. Formulation (Eq. 2)              build_iteration_matrix()
    2. Spectral Initialization          spectral_initialization()
    3. while not Converged do           run_spcd_iteration()
           Pick ONE update box:           or
           - SPCD     (Eq. 4 / 7)       run_norm_spcd_iteration()
           - NormSPCD (Eq. 5 / 8)
    4. Final weight step                empirical_weights() OR exact_weights()
           Algorithm 1 uses Eq. (6).
           Algorithm 2 uses Eq. (9).
    5. Minority-group flip + tau-hat    apply_minority_flip(),
                                        build_synthetic_paths()

The two iteration-box choices and the two weight-step choices are
exposed as independent options here. The combination
``variant="norm_spcd"`` + ``weights="empirical"`` matches the algorithm
used in the paper's Section 4 experiments ("In all the experiment in
this paper, we use this simplified implementation", page 9).

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...exceptions import MlsynthConfigError
from .formulation import build_iteration_matrix
from .iteration_norm_spcd import run_norm_spcd_iteration
from .iteration_spcd import run_spcd_iteration
from .spectral_init import spectral_initialization
from .structures import SPCDDesign, SPCDInputs
from .treatment_effect import (
    apply_minority_flip,
    build_synthetic_paths,
    build_weight_groups,
)
from .weights_empirical import empirical_weights
from .weights_exact import exact_weights


def solve_spcd(
    inputs: SPCDInputs,
    *,
    variant: str = "norm_spcd",
    weights: str = "empirical",
    alpha: Optional[float] = None,
    lam: Optional[float] = None,
    beta: Optional[float] = None,
    max_iter: int = 200,
    solver: Optional[Any] = None,
    verbose: bool = False,
) -> SPCDDesign:
    """Run SPCD end-to-end and return the resulting design.

    Parameters
    ----------
    inputs : SPCDInputs
        Pre-treatment (and optional post-treatment) outcome matrices and
        index metadata.
    variant : {"spcd", "norm_spcd"}
        Iteration-box choice. ``"spcd"`` selects Eq. (4)/(7),
        ``"norm_spcd"`` selects Eq. (5)/(8).
    weights : {"empirical", "exact"}
        Final-weight-step choice. ``"empirical"`` uses Eq. (9) (the
        closed-form approximation used in the paper's experiments),
        ``"exact"`` solves Eq. (6) via cvxpy.
    alpha, lam, beta : float, optional
        Hyperparameters for Eq. (2) and Eqs. (4)/(5)/(7)/(8). If any is
        ``None``, it is auto-estimated from the spectrum of
        ``Y_pre.T @ Y_pre``; see ``formulation.build_iteration_matrix``.
    max_iter : int
        Maximum iterations for the SPCD/NormSPCD while loop.
    solver : optional
        Solver passed to cvxpy when ``weights="exact"``. Ignored
        otherwise.
    verbose : bool
        Whether to print solver progress for ``weights="exact"``.

    Returns
    -------
    SPCDDesign
        Container with the final sign vector, weight components,
        synthetic paths, and diagnostics.
    """

    if variant not in {"spcd", "norm_spcd"}:
        raise MlsynthConfigError(
            f"Unknown SPCD variant '{variant}'. Use 'spcd' or 'norm_spcd'."
        )
    if weights not in {"empirical", "exact"}:
        raise MlsynthConfigError(
            f"Unknown SPCD weights mode '{weights}'. Use 'empirical' or 'exact'."
        )

    # Step 1: Formulation, Eq. (2).
    _M, M_inv, alpha_used, lam_used, beta_used = build_iteration_matrix(
        Y_pre=inputs.Y_pre, alpha=alpha, lam=lam, beta=beta
    )

    # Step 2: Spectral Initialization (shared by Algorithms 1 & 2).
    y0 = spectral_initialization(_M)

    # Step 3: SPCD or NormSPCD iteration box, Eqs. (4)/(7) or (5)/(8).
    if variant == "spcd":
        y_star, n_iter, converged = run_spcd_iteration(
            M_inv=M_inv, y0=y0, beta=beta_used, max_iter=max_iter
        )
    else:
        y_star, n_iter, converged = run_norm_spcd_iteration(
            M_inv=M_inv, y0=y0, beta=beta_used, max_iter=max_iter
        )

    # Step 4: Final weight step. Algorithm 1 -> Eq. (6); Algorithm 2 -> Eq. (9).
    if weights == "empirical":
        raw_weights = empirical_weights(M_inv=M_inv, y_star=y_star)
    else:
        raw_weights = exact_weights(
            Y_pre=inputs.Y_pre,
            y_star=y_star,
            sigma=alpha_used,
            solver=solver,
            verbose=verbose,
        )

    # Step 5: Minority-group convention + synthetic-path assembly.
    assignment_pm1, raw_weights = apply_minority_flip(y_star, raw_weights)
    selected_mask, treated_weights, control_weights, contrast_weights = (
        build_weight_groups(assignment_pm1, raw_weights)
    )
    synthetic_treated, synthetic_control, synthetic_gap = build_synthetic_paths(
        Y_pre=inputs.Y_pre,
        Y_post=inputs.Y_post,
        treated_weights=treated_weights,
        control_weights=control_weights,
    )

    selected_unit_indices = np.where(selected_mask == 1)[0]
    selected_unit_labels = inputs.unit_index.get_labels(selected_unit_indices)

    return SPCDDesign(
        variant=variant,
        weights_mode=weights,
        assignment_pm1=assignment_pm1,
        selected_mask=selected_mask,
        raw_weights=raw_weights,
        treated_weights=treated_weights,
        control_weights=control_weights,
        contrast_weights=contrast_weights,
        synthetic_treated=synthetic_treated,
        synthetic_control=synthetic_control,
        synthetic_gap=synthetic_gap,
        selected_unit_indices=selected_unit_indices,
        selected_unit_labels=selected_unit_labels,
        n_treated=int(selected_mask.sum()),
        n_iterations=n_iter,
        converged=converged,
        alpha_ridge=alpha_used,
        lam_balance=lam_used,
        beta=beta_used,
    )
