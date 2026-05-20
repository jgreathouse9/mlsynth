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

import warnings
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from ...exceptions import MlsynthConfigError
from .formulation import build_iteration_matrix
from .holdout import compute_holdout_residuals, split_pre_window
from .inference import SPCDConformalResult, compute_conformal_ci
from .iteration_norm_spcd import run_norm_spcd_iteration
from .iteration_spcd import run_spcd_iteration
from .power import (
    SPCDPowerAnalysis,
    compute_detectability_curve,
    compute_mde,
)
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


def solve_spcd_with_holdout(
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
    enable_inference: bool = True,
    holdout_frac_E: float = 0.7,
    inference_alpha: float = 0.05,
    power_target: float = 0.8,
    mde_n_sims: int = 5000,
    mde_n_trials: int = 400,
    mde_horizon_grid: Optional[list] = None,
    inference_seed: int = 1400,
    min_blank_size: int = 5,
):
    """Run SPCD with the train-on-E / calibrate-on-B inference flow.

    When ``enable_inference=False`` this falls back to
    :func:`solve_spcd` on the full pretreatment matrix and returns
    ``(design, None, None)`` for backwards compatibility with the
    pre-inference SPCD release.

    When ``enable_inference=True``:

    1. Split ``inputs.Y_pre`` into ``Y_E`` (first ``frac_E``) and
       ``Y_B`` (remainder).
    2. Fit the SPCD design on ``Y_E`` only.
    3. Re-render the synthetic paths over the *full* timeline
       (``Y_E`` + ``Y_B`` + ``Y_post``) so plots show the entire window.
    4. Compute holdout residuals ``r_B = Y_B @ contrast_weights``.
    5. Run the Monte Carlo MDE analysis on ``r_B`` (always, since this
       is pre-experiment planning).
    6. If ``Y_post`` is present, also compute the moving-block
       conformal CI for the ATT.

    Returns
    -------
    design : SPCDDesign
        The SPCD design, with synthetic paths spanning the full
        timeline. Fit on ``Y_E`` only when ``enable_inference=True``.
    conformal : SPCDConformalResult or None
        Conformal CI for the post-period ATT. ``None`` if
        ``Y_post is None`` or ``enable_inference=False`` or the holdout
        window is too small.
    power : SPCDPowerAnalysis or None
        MDE / detectability output. ``None`` if
        ``enable_inference=False`` or the holdout window is too small.
    """

    # ------------------------------------------------------------------
    # Fast path: inference disabled -> exactly the legacy behavior.
    # ------------------------------------------------------------------
    if not enable_inference:
        design = solve_spcd(
            inputs=inputs,
            variant=variant,
            weights=weights,
            alpha=alpha,
            lam=lam,
            beta=beta,
            max_iter=max_iter,
            solver=solver,
            verbose=verbose,
        )
        return design, None, None

    # ------------------------------------------------------------------
    # Holdout split. May raise if the estimation window is too small.
    # ------------------------------------------------------------------
    Y_E, Y_B, n_E, n_B, can_infer = split_pre_window(
        inputs.Y_pre, frac_E=holdout_frac_E, min_blank_size=min_blank_size
    )

    if not can_infer:
        warnings.warn(
            f"SPCD: holdout window of {n_B} period(s) is below "
            f"min_blank_size={min_blank_size}. Fitting the design on "
            f"the estimation window but skipping inference and power.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Fit the SPCD design on Y_E only. Y_post is set to None during this
    # call so the inner solve_spcd doesn't try to render paths beyond E.
    # ------------------------------------------------------------------
    training_inputs = replace(inputs, Y_pre=Y_E, Y_post=None)
    design_E = solve_spcd(
        inputs=training_inputs,
        variant=variant,
        weights=weights,
        alpha=alpha,
        lam=lam,
        beta=beta,
        max_iter=max_iter,
        solver=solver,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Re-render synthetic paths over the full timeline.
    # ------------------------------------------------------------------
    Y_pre_full = inputs.Y_pre  # already Y_E vstack Y_B by construction
    synthetic_treated, synthetic_control, synthetic_gap = build_synthetic_paths(
        Y_pre=Y_pre_full,
        Y_post=inputs.Y_post,
        treated_weights=design_E.treated_weights,
        control_weights=design_E.control_weights,
    )
    design = replace(
        design_E,
        synthetic_treated=synthetic_treated,
        synthetic_control=synthetic_control,
        synthetic_gap=synthetic_gap,
    )

    # ------------------------------------------------------------------
    # If the holdout is too small, return early with the design only.
    # ------------------------------------------------------------------
    if not can_infer:
        return design, None, None

    # ------------------------------------------------------------------
    # Holdout residuals are the calibration set for both inference and
    # power. Baseline is the holdout-window mean of synthetic_treated.
    # ------------------------------------------------------------------
    r_B = compute_holdout_residuals(Y_B, design.contrast_weights)

    holdout_treated = Y_B @ design.treated_weights
    baseline = float(np.mean(holdout_treated))

    # ------------------------------------------------------------------
    # MDE always runs (its purpose is *pre-experiment* planning).
    # Default horizon = len(Y_post) if available, otherwise len(Y_B).
    # ------------------------------------------------------------------
    n_post_for_power = (
        int(inputs.Y_post.shape[0]) if inputs.Y_post is not None else int(n_B)
    )
    n_post_for_power = max(n_post_for_power, 1)
    power = compute_mde(
        residuals_B=r_B,
        baseline=baseline,
        n_post=n_post_for_power,
        alpha=inference_alpha,
        power_target=power_target,
        n_sims=mde_n_sims,
        n_trials=mde_n_trials,
        seed=inference_seed,
    )

    if mde_horizon_grid:
        detectability = compute_detectability_curve(
            residuals_B=r_B,
            baseline=baseline,
            horizon_grid=list(mde_horizon_grid),
            alpha=inference_alpha,
            power_target=power_target,
            n_sims=mde_n_sims,
            n_trials=mde_n_trials,
            seed=inference_seed,
        )
        power = replace(power, detectability=detectability)

    # ------------------------------------------------------------------
    # Conformal CI only when post data exists.
    # ------------------------------------------------------------------
    conformal: Optional[SPCDConformalResult]
    if inputs.Y_post is not None:
        post_gap = inputs.Y_post @ design.contrast_weights
        conformal = compute_conformal_ci(
            residuals_B=r_B,
            post_gap=post_gap,
            alpha=inference_alpha,
        )
    else:
        conformal = None

    return design, conformal, power
