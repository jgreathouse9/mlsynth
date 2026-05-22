"""Inference procedures for SparseSC.

Two methods are implemented:

* ``run_placebo`` -- the Abadie-style placebo permutation. For each
  donor we treat that donor as the placebo treated unit, refit
  SparseSC at the already-selected lambda, and compare the observed
  ATT against the distribution of placebo ATTs.

* ``conformal_inference`` -- a moving-block conformal CI in the
  spirit of Chernozhukov, Wuethrich and Zhu (2021), adapted to the
  SparseSC pre / validation / post panel layout. Calibration
  residuals come from either the validation block (default --
  smallest sample but truly out-of-sample under V) or the entire
  pre-treatment block (larger sample, but training residuals are
  in-sample under V). The ATT CI is obtained by inverting a moving-
  block test of the form ``mean(|e_post - theta|) <= q_{1-alpha}``
  of the calibration distribution; pointwise per-period bands use
  the same ``q_{1-alpha}`` quantile directly.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .objective import outer_loss
from .optimization import default_v20, recover_w, sweep_lambda
from scipy.optimize import minimize


def _moving_block_means(
    residuals: np.ndarray,
    block_size: int,
    include_circular: bool = True,
) -> np.ndarray:
    """Return mean-absolute-residual over every moving block of ``block_size``.

    With ``include_circular``, also produces wrap-around blocks so the
    score distribution stays informative when ``len(residuals)`` is
    small (the SparseSC validation block is typically only a handful
    of periods). Block size is clipped to at most ``len(residuals)``.
    """
    e = np.abs(np.asarray(residuals, dtype=float).flatten())
    n = e.size
    if n == 0:
        return np.asarray([], dtype=float)
    bs = max(1, min(int(block_size), n))

    scores = []
    for i in range(n - bs + 1):
        scores.append(float(np.mean(e[i:i + bs])))

    if include_circular and n >= bs and bs > 1:
        for i in range(1, bs):
            tail = e[n - i:]
            head = e[: bs - i]
            scores.append(float(np.mean(np.concatenate([tail, head]))))

    return np.asarray(scores, dtype=float)


def conformal_inference(
    gap: np.ndarray,
    T0_train: int,
    T0_total: int,
    T: int,
    conformal_window: str = "validation",
    alpha: float = 0.05,
    block_size: int | None = None,
    grid_size: int = 401,
    grid_half_width_se: float = 6.0,
) -> dict:
    """Conformal ATT confidence interval from in-sample residuals.

    Parameters
    ----------
    gap : np.ndarray
        Full-period residual ``Y1 - Y0 @ w``, shape ``(T,)``. The pre-
        treatment portion (``gap[:T0_total]``) is interpreted as noise
        under the no-treatment null; ``gap[T0_total:]`` is the post-
        treatment effect-plus-noise sequence.
    T0_train, T0_total, T : int
        Training-block end / pre-block end / full length. Pre = ``[0,
        T0_total)``, validation = ``[T0_train, T0_total)``, post =
        ``[T0_total, T)``.
    conformal_window : {"validation", "pre"}
        Which residual block to use for calibration. ``"validation"``
        uses only ``gap[T0_train:T0_total]`` (truly out-of-sample under
        the chosen V); ``"pre"`` uses the entire ``gap[:T0_total]``.
    alpha : float
        Two-sided significance level.
    block_size : int, optional
        Moving-block size for the conformity score. Defaults to
        ``max(3, sqrt(n_post))``, matching LEXSCM.
    grid_size : int
        Number of theta candidates in the grid search for the ATT CI.
    grid_half_width_se : float
        The grid spans ``[ATT_hat +/- grid_half_width_se * SE]`` where
        SE is a plug-in standard error from the calibration residuals.

    Returns
    -------
    dict
        Keys: ``method``, ``att_observed``, ``ci_lower``, ``ci_upper``,
        ``p_value``, ``calibration_residuals``, ``pointwise_lower``,
        ``pointwise_upper``, ``alpha``.
    """
    gap = np.asarray(gap, dtype=float).flatten()

    if conformal_window not in {"validation", "pre"}:
        raise ValueError(
            "conformal_window must be 'validation' or 'pre', "
            f"got {conformal_window!r}."
        )

    if conformal_window == "validation":
        e_calib = gap[T0_train:T0_total]
        method_tag = "conformal_validation"
    else:
        e_calib = gap[:T0_total]
        method_tag = "conformal_pre"

    e_post = gap[T0_total:T]
    n_post = int(e_post.size)
    n_calib = int(e_calib.size)

    if n_post == 0:
        return {
            "method": method_tag,
            "att_observed": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "calibration_residuals": e_calib,
            "pointwise_lower": np.asarray([], dtype=float),
            "pointwise_upper": np.asarray([], dtype=float),
            "alpha": float(alpha),
        }

    if n_calib < 2:
        # Not enough calibration data to produce a non-degenerate CI.
        att = float(np.mean(e_post))
        return {
            "method": method_tag,
            "att_observed": att,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "calibration_residuals": e_calib,
            "pointwise_lower": np.full(n_post, np.nan),
            "pointwise_upper": np.full(n_post, np.nan),
            "alpha": float(alpha),
        }

    if block_size is None:
        block_size = max(3, int(np.sqrt(n_post)))
    block_size = min(block_size, n_calib)

    att = float(np.mean(e_post))
    conformity = _moving_block_means(e_calib, block_size, include_circular=True)
    if conformity.size == 0:
        conformity = np.abs(e_calib)

    # P-value for H0: ATT = 0
    observed_score = float(np.mean(np.abs(e_post)))
    p_value = float(np.mean(conformity >= observed_score))

    # Grid search for the ATT CI: invert the conformal test.
    se_proxy = float(np.std(e_calib, ddof=1)) / np.sqrt(max(n_post, 1)) if n_calib > 1 else 1.0
    half_width = max(grid_half_width_se * se_proxy, 1e-6)
    grid = np.linspace(att - half_width, att + half_width, int(grid_size))

    accepted = []
    for theta in grid:
        score = float(np.mean(np.abs(e_post - theta)))
        if np.mean(conformity >= score) > alpha:
            accepted.append(float(theta))

    if accepted:
        ci_lower = min(accepted)
        ci_upper = max(accepted)
    else:
        # No grid point accepted: fall back to a SE-scaled interval
        # to avoid returning NaN bounds for a non-degenerate fit.
        fallback = 4.0 * se_proxy
        ci_lower = att - fallback
        ci_upper = att + fallback

    # Pointwise per-period bands from the (1 - alpha) quantile of
    # the calibration conformity scores.
    q = float(np.quantile(conformity, 1.0 - alpha))
    pointwise_lower = e_post - q
    pointwise_upper = e_post + q

    return {
        "method": method_tag,
        "att_observed": att,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "calibration_residuals": e_calib,
        "pointwise_lower": pointwise_lower,
        "pointwise_upper": pointwise_upper,
        "alpha": float(alpha),
    }


def _refit_at_lambda(
    X1: np.ndarray, X0: np.ndarray,
    Z1_outer: np.ndarray, Z0_outer: np.ndarray,
    lam: float, solver: Any,
) -> np.ndarray:
    """Refit V-weights at a fixed lambda; return full v including v[0] = 1.

    ``Z1_outer``, ``Z0_outer`` are the outcome block the outer V loss
    is evaluated on -- validation block by default, training block in
    MATLAB-driver mode. The caller picks the window.
    """
    P = X0.shape[0]
    v20 = default_v20(X0)
    bounds = [(0.0, None)] * (P - 1)
    res = minimize(
        outer_loss, x0=v20,
        args=(X1, X0, Z1_outer, Z0_outer, float(lam), solver),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    return np.concatenate([[1.0], np.clip(res.x, 0.0, None)])


def run_placebo(
    Y0: np.ndarray, Y1: np.ndarray,
    X0: np.ndarray, X1: np.ndarray,
    T0_total: int, T0_train: int,
    selected_lambda: float,
    observed_att: float,
    solver: Any = None,
    resweep: bool = False,
    lambda_grid: np.ndarray | None = None,
    n_placebo: int | None = None,
    seed: int = 1400,
    outer_loss_window: str = "validation",
) -> Tuple[np.ndarray, float, int]:
    """Return ``(placebo_atts, p_value, n_completed)``.

    Parameters
    ----------
    Y0, Y1, X0, X1 : np.ndarray
        Full pre-standardized panel + predictor matrices.
    T0_total, T0_train : int
        Pre-treatment window bounds.
    selected_lambda : float
        Lambda chosen on the actual treated unit. Reused for each
        placebo when ``resweep=False`` (default).
    observed_att : float
        ATT of the actual treated unit, used to construct the p-value.
    resweep : bool
        If True, re-run the full lambda grid for each placebo. Slow.
    lambda_grid : np.ndarray, optional
        Grid for the resweep case.
    n_placebo : int, optional
        Subsample of donors to use as placebos. ``None`` uses every donor.
    seed : int
        Seed for the subsample when ``n_placebo < N``.
    """
    rng = np.random.default_rng(seed)
    N = X0.shape[1]
    donor_indices = np.arange(N)
    if n_placebo is not None and n_placebo < N:
        donor_indices = rng.choice(donor_indices, size=int(n_placebo),
                                   replace=False)

    placebo_list = []
    for j in donor_indices:
        # Swap donor j into the treated slot.
        X0_loo = np.delete(X0, j, axis=1)
        Y0_loo = np.delete(Y0, j, axis=1)
        X1_placebo = X0[:, j].copy()
        Y1_placebo = Y0[:, j].copy()

        if outer_loss_window == "validation":
            Z1_outer = Y1_placebo[T0_train:T0_total]
            Z0_outer = Y0_loo[T0_train:T0_total, :]
        else:
            Z1_outer = Y1_placebo[:T0_train]
            Z0_outer = Y0_loo[:T0_train, :]

        if resweep:
            best_v, _, _, _, _, _ = sweep_lambda(
                X1=X1_placebo, X0=X0_loo,
                Y1=Y1_placebo, Y0=Y0_loo,
                T0_total=T0_total, T0_train=T0_train,
                lambda_grid=lambda_grid, solver=solver,
                outer_loss_window=outer_loss_window,
            )
        else:
            try:
                best_v = _refit_at_lambda(
                    X1=X1_placebo, X0=X0_loo,
                    Z1_outer=Z1_outer, Z0_outer=Z0_outer,
                    lam=float(selected_lambda), solver=solver,
                )
            except Exception:
                continue

        try:
            w = recover_w(best_v, X1_placebo, X0_loo, solver=solver)
        except Exception:
            continue
        cf = Y0_loo @ w
        placebo_att = float(np.mean((Y1_placebo - cf)[T0_total:]))
        if np.isfinite(placebo_att):
            placebo_list.append(placebo_att)

    placebo_atts = np.asarray(placebo_list, dtype=float)
    if placebo_atts.size == 0 or not np.isfinite(observed_att):
        return placebo_atts, float("nan"), 0
    p_value = float(
        (np.sum(np.abs(placebo_atts) >= abs(observed_att)) + 1)
        / (placebo_atts.size + 1)
    )
    return placebo_atts, p_value, int(placebo_atts.size)
