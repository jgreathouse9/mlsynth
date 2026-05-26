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
from .formulation import build_iteration_matrix, estimate_noise_variance
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


def _contrast_weights_for_alpha(
    Y_fit: np.ndarray,
    *,
    variant: str,
    weights: str,
    alpha: float,
    lam: Optional[float],
    beta: Optional[float],
    max_iter: int,
    solver: Optional[Any],
) -> np.ndarray:
    """Run the SPCD design pipeline on a raw array and return its signed
    contrast weights (``treated - control``).

    This is the array-level core of :func:`solve_spcd` (Steps 1-5 minus
    label/path assembly), used by :func:`select_alpha_by_holdout` to score
    candidate ``alpha`` values out-of-sample.
    """

    _M, M_inv, _a, _l, beta_used = build_iteration_matrix(
        Y_pre=Y_fit, alpha=alpha, lam=lam, beta=beta
    )
    y0 = spectral_initialization(_M)
    if variant == "spcd":
        y_star, _, _ = run_spcd_iteration(
            M_inv=M_inv, y0=y0, beta=beta_used, max_iter=max_iter
        )
    else:
        y_star, _, _ = run_norm_spcd_iteration(
            M_inv=M_inv, y0=y0, beta=beta_used, max_iter=max_iter
        )
    if weights == "empirical":
        raw_weights = empirical_weights(M_inv=M_inv, y_star=y_star)
    else:
        raw_weights = exact_weights(
            Y_pre=Y_fit, y_star=y_star, sigma=_a, solver=solver, verbose=False
        )
    assignment_pm1, raw_weights = apply_minority_flip(y_star, raw_weights)
    _mask, _tw, _cw, contrast_weights = build_weight_groups(
        assignment_pm1, raw_weights
    )
    return contrast_weights


def select_alpha_by_holdout(
    Y_pre: np.ndarray,
    *,
    variant: str = "norm_spcd",
    weights: str = "empirical",
    lam: Optional[float] = None,
    beta: Optional[float] = None,
    max_iter: int = 200,
    solver: Optional[Any] = None,
    val_frac: float = 0.3,
) -> float:
    """Choose the ridge ``alpha`` by out-of-sample pre-period balance.

    The post-period RMSE of an SPCD design is a **non-monotone, jumpy
    function of** ``alpha`` when ``N > T_pre`` (the assignment is a discrete
    sign vector that flips as ``alpha`` moves), so no single closed-form
    estimate of the noise variance is robust. Instead, this fits the design
    on the first ``1 - val_frac`` of the pre-period and scores each
    candidate ``alpha`` by the RMS of the resulting contrast on the held-out
    pre-period tail, returning the best. Candidates are a multiplicative
    grid around the Gavish-Donoho noise-variance estimate
    (:func:`~mlsynth.utils.spcd_helpers.formulation.estimate_noise_variance`),
    which sets the correct *scale*; the holdout picks the robust point on
    the jumpy curve.

    Falls back to the bare noise-variance estimate when the pre-period is
    too short to spare a validation tail.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment matrix of shape ``(T_pre, N)``.
    variant, weights, lam, beta, max_iter, solver
        Passed through to the design pipeline.
    val_frac : float
        Fraction of the pre-period held out (from the end) for scoring.

    Returns
    -------
    float
        Selected ``alpha``.
    """

    T_pre = Y_pre.shape[0]
    gd = max(estimate_noise_variance(Y_pre), 1e-8)
    n_val = max(3, int(round(val_frac * T_pre)))
    if T_pre - n_val < 2:  # too short to hold out; use the scale estimate
        return gd

    Y_fit, Y_val = Y_pre[:-n_val], Y_pre[-n_val:]
    grid = sorted({gd * f for f in (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)})

    best_alpha, best_score = gd, np.inf
    for cand in grid:
        try:
            contrast = _contrast_weights_for_alpha(
                Y_fit,
                variant=variant,
                weights=weights,
                alpha=float(cand),
                lam=lam,
                beta=beta,
                max_iter=max_iter,
                solver=solver,
            )
        except Exception:  # pragma: no cover - skip infeasible candidates
            continue
        score = float(np.sqrt(np.mean((Y_val @ contrast) ** 2)))
        if score < best_score:
            best_alpha, best_score = float(cand), score
    return best_alpha


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
        Hyperparameters for Eq. (2) and Eqs. (4)/(5)/(7)/(8). The paper
        treats all three as *pre-defined*. When ``alpha`` is ``None`` it is
        chosen by :func:`select_alpha_by_holdout` (out-of-sample pre-period
        balance over a noise-scale grid); ``lam`` and ``beta`` default from
        the spectrum via ``formulation.build_iteration_matrix``. Pass an
        explicit ``alpha`` (e.g. a known noise variance) to bypass
        selection.
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

    # Step 0: Choose alpha by out-of-sample pre-period balance when the user
    # has not fixed it. alpha is the noise-scale ridge of Eq. (2); its
    # post-period RMSE surface is jumpy when N > T_pre, so we select it by
    # holdout balance rather than a single closed-form estimate.
    if alpha is None:
        alpha = select_alpha_by_holdout(
            inputs.Y_pre,
            variant=variant,
            weights=weights,
            lam=lam,
            beta=beta,
            max_iter=max_iter,
            solver=solver,
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

    # ------------------------------------------------------------------
    # Guard: the SPCD design should balance the held-out block at least as
    # well as a naive random equal-split design. When N > T the assignment
    # can overfit the estimation window and generalize *worse* than random;
    # surface that loudly rather than shipping a bad design silently.
    # ------------------------------------------------------------------
    spcd_oos = float(np.sqrt(np.mean(r_B ** 2)))
    rng_guard = np.random.default_rng(inference_seed)
    n_units = Y_B.shape[1]
    rand_scores = []
    for _ in range(50):
        signs = rng_guard.choice([-1.0, 1.0], size=n_units)
        if signs.min() == signs.max():
            signs[rng_guard.integers(n_units)] *= -1.0
        treated = signs > 0
        w = np.where(treated, 1.0 / max(treated.sum(), 1), 0.0) - np.where(
            ~treated, 1.0 / max((~treated).sum(), 1), 0.0
        )
        rand_scores.append(float(np.sqrt(np.mean((Y_B @ w) ** 2))))
    random_oos = float(np.median(rand_scores))
    if spcd_oos > random_oos:
        warnings.warn(
            f"SPCD: the fitted design balances the holdout block worse than "
            f"a random equal-split baseline (out-of-sample RMS {spcd_oos:.3g} "
            f"vs random {random_oos:.3g}). The design may be overfitting the "
            f"estimation window (common when N > T_pre). Consider supplying "
            f"'alpha' explicitly or lengthening the pre-period.",
            UserWarning,
            stacklevel=2,
        )

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

    # Detectability ("MDE at time point t"): when no grid is supplied,
    # default to every horizon from 1 to the planned post length so the
    # curve is always available per design.
    horizon_grid = (
        list(mde_horizon_grid)
        if mde_horizon_grid
        else list(range(1, n_post_for_power + 1))
    )
    detectability = compute_detectability_curve(
        residuals_B=r_B,
        baseline=baseline,
        horizon_grid=horizon_grid,
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
