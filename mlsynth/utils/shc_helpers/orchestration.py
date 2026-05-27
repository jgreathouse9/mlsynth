"""End-to-end Synthetic Historical Control procedure (Chen-Yang-Yang 2024).

Composes the estimator's stages (Section 2.3) using the shared numerical
kernels in :mod:`mlsynth.utils.shc_helpers.kernels`, :mod:`mlsynth.utils.datautils`,
and :mod:`mlsynth.utils.shc_helpers.selection`:

    1. LOOCV-select a kernel bandwidth and smooth the pre-period into the
       latent trend ``ell_hat`` (Eq. 21).
    2. Build the treated block and the overlapping historical blocks
       (Eq. 7).
    3. Stepwise-match the treated pre-segment with a simplex combination
       of historical pre-segments (Eqs. 32-34).
    4. (Optional) augment with a ridge refinement (ASHC).
    5. Apply the weights to the historical forward segments to obtain the
       post-intervention counterfactual.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..datautils import build_donor_segments
from .kernels import (
    _solve_SHC_QP,
    loocv_bandwidth,
    smooth,
    tune_lambda_ashc,
)
from ..resultutils import effects
from .selection import stepwise_donor_selection
from .structures import SHCDesign, SHCInputs

_DEFAULT_BANDWIDTH_GRID = np.linspace(0.05, 1.0, 50)


def _lambda_grid(L_pre: np.ndarray, lambda_min_ratio: float = 1e-8,
                 n_lambda: int = 100) -> np.ndarray:
    """Geometric ridge-penalty grid spanning ``[lambda_min_ratio, 1] * lambda_max``."""
    _, sing, _ = np.linalg.svd(L_pre.T, full_matrices=False)
    lambda_max = sing[0] ** 2.0
    scaler = lambda_min_ratio ** (1.0 / n_lambda)
    return lambda_max * (scaler ** np.arange(n_lambda))


def solve_shc(
    inputs: SHCInputs,
    *,
    use_augmented: bool = False,
    bandwidth_grid: Optional[Sequence[float]] = None,
) -> SHCDesign:
    """Run the SHC pipeline and return the fitted design.

    Parameters
    ----------
    inputs : SHCInputs
        From :func:`prepare_shc_inputs`.
    use_augmented : bool
        Apply the augmented (ASHC) ridge refinement to the simplex weights.
    bandwidth_grid : sequence of float, optional
        Candidate bandwidths for LOOCV. Defaults to 50 points on
        ``[0.05, 1.0]``.
    """
    m, T0, n = inputs.m, inputs.T0, inputs.n
    y_pre = inputs.y[:T0]

    grid = np.asarray(bandwidth_grid if bandwidth_grid is not None
                      else _DEFAULT_BANDWIDTH_GRID, dtype=float)

    # Step 1: latent trend.
    bandwidth, _ = loocv_bandwidth(y_pre, grid)
    ell_hat = smooth(y_pre, bandwidth)

    # Step 2: treated + historical blocks.
    L_full, L_post, ell_eval = build_donor_segments(ell_hat, m, T0, n)

    # Step 3: stepwise simplex matching.
    selection = stepwise_donor_selection(L_full, L_post, ell_eval, m)
    selected = selection["best_donors"]
    weights = np.zeros(L_full.shape[1])
    weights[selected] = selection["best_weights"]

    # Step 4: optional ridge refinement.
    best_lambda: Optional[float] = None
    if use_augmented:
        lam_grid = _lambda_grid(L_full[:m])
        best_lambda, _ = tune_lambda_ashc(
            L=L_full, ell_eval=ell_eval, w_shc=weights, lambda_grid=lam_grid,
        )
        weights, _ = _solve_SHC_QP(
            L=L_full, ell_eval=ell_eval, use_augmented=True,
            w_shc=weights, lam=best_lambda,
        )

    # Step 5: counterfactual over the m + n window.
    counterfactual_window = np.vstack([L_full, L_post]) @ weights

    block_weights = {
        f"block@{int(i)}": float(weights[i])
        for i in np.nonzero(weights)[0]
    }

    return SHCDesign(
        bandwidth=float(bandwidth),
        latent_pre=ell_hat,
        weights=weights,
        selected_blocks=[int(i) for i in np.nonzero(weights)[0]],
        block_weights=block_weights,
        counterfactual_window=counterfactual_window,
        use_augmented=use_augmented,
        best_lambda=best_lambda,
    )


def summarize_effects(
    inputs: SHCInputs, design: SHCDesign,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute ATT, gap, and fit diagnostics over the treated-block window.

    Returns
    -------
    att, att_percent : float
    observed, counterfactual, gap : np.ndarray
        Series over the ``m + n`` block window.
    window_time : np.ndarray
        Period labels for the window.
    fit_diagnostics : dict
        ``{"rmse_pre", "rmse_post", "r_squared_pre"}``.
    """
    m, T0, n = inputs.m, inputs.T0, inputs.n
    cf = design.counterfactual_window
    observed = inputs.y[T0 - m:T0 + n]
    if observed.shape[0] != cf.shape[0]:
        raise MlsynthEstimationError(
            f"Observed window ({observed.shape[0]}) and counterfactual "
            f"({cf.shape[0]}) lengths differ; check m and the pre-period."
        )

    attdict, fitdict, _ = effects.calculate(observed, cf, m, n)
    gap = observed - cf
    window_time = inputs.time_index.labels[T0 - m:T0 + n]

    fit_diagnostics = {
        "rmse_pre": fitdict.get("T0 RMSE"),
        "rmse_post": fitdict.get("T1 RMSE"),
        "r_squared_pre": fitdict.get("R-Squared"),
    }
    return (
        float(attdict.get("ATT")),
        float(attdict.get("Percent ATT")),
        observed,
        cf,
        gap,
        window_time,
        fit_diagnostics,
    )
