"""Orchestration pipeline for RPCA-SC (Bayani 2021, Algorithm 4).

Composes the five steps of *Robust PCA Synthetic Control*:

1. :mod:`.fpca` -- standardised FPC scores from pre-period trajectories.
2. :mod:`.clustering` -- silhouette-driven :math:`k`-means and donor
   selection via the treated unit's cluster membership.
3. :mod:`.pcp` / :mod:`.hqf` -- robust :math:`Y = L + S` decomposition
   of the selected donor pool.
4. :mod:`.weights` -- non-negative least squares against the denoised
   donor matrix :math:`L^-`.
5. *Project*: :math:`\\widehat{Y}_i^+ = (L^+)^\\top \\widehat{\\beta}`
   using the same denoised donor matrix in the post-period.

The dispatcher :func:`run_rpca` takes the same numpy-array signature
as :func:`run_pcr` so the CLUSTERSC orchestrator can call both behind
a uniform interface.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError
from ..structures import MethodFit
from .clustering import FPCACluster, assign_clusters
from .fpca import FPCAFeatures, compute_fpca_features
from .hqf import HQFResult, hqf_decompose
from .pcp import PCPResult, pcp_decompose
from .inference import cft_prediction_intervals
from .tuning import cv_hqf_rank as _cv_hqf_rank
from .tuning import cv_pcp_lambda
from .weights import solve_nnls

_MIN_PRE = 2
_MIN_DONORS = 2


def run_rpca(
    treated_outcome: np.ndarray,
    donor_outcomes: np.ndarray,
    donor_names: Sequence[str],
    T0: int,
    *,
    rpca_method: str = "PCP",
    # FPCA / clustering knobs
    fpca_cumvar: float = 0.95,
    k_clusters: Optional[int] = None,
    k_max: int = 8,
    # PCP knobs
    pcp_lambda: Optional[float] = None,
    pcp_mu: Optional[float] = None,
    pcp_max_iter: int = 1000,
    pcp_tol: float = 1e-9,
    # HQF knobs
    hqf_rank: Optional[int] = None,
    hqf_cumvar: float = 0.999,
    hqf_lambda: Optional[float] = None,
    hqf_ip: float = 1.0,
    hqf_max_iter: int = 1000,
    # CV knobs
    cv_lambda: bool = False,
    cv_hqf_rank: bool = False,
    cv_lambda_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    cv_hqf_rank_grid: Optional[Sequence[int]] = None,
    # CFT prediction-interval knobs
    compute_cft_pi: bool = False,
    cft_alpha: float = 0.05,
    cft_sims: int = 200,
    cft_e_method: str = "gaussian",
    random_state: int = 0,
) -> MethodFit:
    """Run the five-step RPCA-SC pipeline and assemble a :class:`MethodFit`.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes (columns = donors), shape ``(T, J)``.
    donor_names : sequence of str
        Length-``J`` donor labels.
    T0 : int
        Number of pre-treatment periods.
    rpca_method : {"PCP", "HQF"}
        Robust PCA decomposition.
    fpca_cumvar : float
        Cumulative-variance target for FPCA truncation (Step 1).
        Paper default ``0.95``.
    k_clusters, k_max
        Cluster-count controls; see :func:`.clustering.assign_clusters`.
    pcp_lambda, pcp_mu, pcp_max_iter, pcp_tol
        PCP solver knobs (Candes et al. 2011 / Bayani 2021).
    hqf_rank, hqf_cumvar, hqf_lambda, hqf_ip, hqf_max_iter
        HQF solver knobs (Wang et al. 2023).
    random_state : int
        Seed for k-means and HQF.

    Returns
    -------
    MethodFit
        Frozen container with the RPCA-SC fit (counterfactual projected
        through the denoised donor matrix in both pre and post).
    """
    if rpca_method not in {"PCP", "HQF"}:
        raise MlsynthEstimationError(
            f"rpca_method must be 'PCP' or 'HQF'; got {rpca_method!r}."
        )
    if T0 < _MIN_PRE:
        raise MlsynthEstimationError(
            f"RPCA-SC requires T0 >= {_MIN_PRE}; got {T0}."
        )

    treated_outcome = np.asarray(treated_outcome, dtype=float).flatten()
    donor_outcomes = np.asarray(donor_outcomes, dtype=float)
    donor_names = [str(n) for n in donor_names]
    T = treated_outcome.shape[0]
    if donor_outcomes.shape != (T, len(donor_names)):
        raise MlsynthEstimationError(
            f"donor_outcomes has shape {donor_outcomes.shape}; expected "
            f"(T, J)=({T}, {len(donor_names)})."
        )

    # ------------------------------------------------------------------
    # Step 1: FPCA on the full panel (treated unit prepended as row 0).
    # ------------------------------------------------------------------
    full_pre_panel = np.vstack([treated_outcome[:T0], donor_outcomes[:T0].T])
    features: FPCAFeatures = compute_fpca_features(
        pre_outcomes=full_pre_panel, cumvar_threshold=fpca_cumvar,
    )

    # ------------------------------------------------------------------
    # Step 2: k-means and donor selection.
    # ------------------------------------------------------------------
    cluster: FPCACluster = assign_clusters(
        scores=features.scores,
        treated_row=0,
        k_clusters=k_clusters,
        k_max=k_max,
        random_state=random_state,
    )
    # donor_indices is in *full panel* coordinates; subtract 1 to get
    # back to the donor_outcomes column indices.
    donor_col_idx = cluster.donor_indices - 1
    donor_col_idx = donor_col_idx[donor_col_idx >= 0]
    if donor_col_idx.size < _MIN_DONORS:
        raise MlsynthEstimationError(
            f"Treated cluster has {donor_col_idx.size} donor(s); "
            f"need at least {_MIN_DONORS}."
        )
    selected_donor_full = donor_outcomes[:, donor_col_idx]
    selected_names = [donor_names[i] for i in donor_col_idx]

    # ------------------------------------------------------------------
    # Optional: leave-one-time-out CV for the dominant solver knob
    # (PCP lambda or HQF rank). Tunes the prediction-oriented value
    # rather than the L/S identifiability default from Candes 2011.
    # See `tuning.py` for the algorithm.
    # ------------------------------------------------------------------
    cv_metadata: dict = {}
    if cv_lambda and rpca_method == "PCP":
        cv_res = cv_pcp_lambda(
            donor_pre=selected_donor_full[:T0],
            treated_pre=treated_outcome[:T0],
            multipliers=cv_lambda_multipliers,
            pcp_mu=pcp_mu,
            pcp_max_iter=pcp_max_iter,
            pcp_tol=pcp_tol,
        )
        pcp_lambda = cv_res.best
        cv_metadata = {
            "cv_lambda_grid": cv_res.grid.tolist(),
            "cv_lambda_mse": cv_res.cv_mse.tolist(),
            "cv_lambda_best": cv_res.best,
        }
    if cv_hqf_rank and rpca_method == "HQF":
        cv_res = _cv_hqf_rank(
            donor_pre=selected_donor_full[:T0],
            treated_pre=treated_outcome[:T0],
            grid=cv_hqf_rank_grid,
            hqf_lambda=hqf_lambda,
            hqf_ip=hqf_ip,
            hqf_max_iter=hqf_max_iter,
            random_state=random_state,
        )
        hqf_rank = int(cv_res.best)
        cv_metadata = {
            "cv_hqf_rank_grid": cv_res.grid.tolist(),
            "cv_hqf_rank_mse": cv_res.cv_mse.tolist(),
            "cv_hqf_rank_best": int(cv_res.best),
        }

    # ------------------------------------------------------------------
    # Step 3: Robust PCA on the selected donor matrix (rows = donors).
    # ------------------------------------------------------------------
    donor_matrix = selected_donor_full.T  # shape (n_donors, T)
    if rpca_method == "PCP":
        result = pcp_decompose(
            Y=donor_matrix,
            lam=pcp_lambda,
            mu=pcp_mu,
            max_iter=pcp_max_iter,
            tol=pcp_tol,
        )
        solver_metadata = {
            "pcp_lambda": result.lambda_used,
            "pcp_mu": result.mu_used,
            "pcp_iterations": result.iterations,
            "pcp_converged": result.converged,
        }
    else:
        result = hqf_decompose(
            Y=donor_matrix,
            rank=hqf_rank,
            cumvar_threshold=hqf_cumvar,
            lam=hqf_lambda,
            ip=hqf_ip,
            max_iter=hqf_max_iter,
            random_state=random_state,
        )
        solver_metadata = {
            "hqf_rank": result.rank_used,
            "hqf_lambda": result.lambda_used,
            "hqf_ip": result.ip_used,
            "hqf_iterations": result.iterations,
        }

    L_full = result.low_rank.T          # shape (T, n_donors), donors as columns
    L_pre = L_full[:T0]

    # ------------------------------------------------------------------
    # Step 4: non-negative LS against the denoised pre-period donors.
    # ------------------------------------------------------------------
    beta = solve_nnls(denoised_donor_pre=L_pre, target_pre=treated_outcome[:T0])

    # ------------------------------------------------------------------
    # Step 5: project through the denoised donor matrix in both periods.
    # ------------------------------------------------------------------
    counterfactual = L_full @ beta
    gap = treated_outcome - counterfactual
    att = float(np.mean(gap[T0:])) if T > T0 else float("nan")
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))

    donor_weights = {name: float(w) for name, w in zip(selected_names, beta)}

    metadata = {
        "rpca_method": rpca_method,
        "fpca_cumvar": float(fpca_cumvar),
        "fpca_rank": int(features.rank),
        "fpca_smoothing": features.smoothing,
        "k_clusters": int(cluster.k),
        "treated_cluster": int(cluster.treated_cluster),
        "cluster_labels": cluster.labels.tolist(),
        **solver_metadata,
        **cv_metadata,
    }

    # Optional CFT (Cattaneo-Feng-Titiunik) prediction intervals.
    # Build a closure that refits the full pipeline at the same
    # hyperparameters but with a perturbed treated outcome.
    if compute_cft_pi and T > T0:
        def _refit(y_star: np.ndarray) -> np.ndarray:
            star_fit = run_rpca(
                treated_outcome=y_star,
                donor_outcomes=donor_outcomes,
                donor_names=donor_names,
                T0=T0,
                rpca_method=rpca_method,
                fpca_cumvar=fpca_cumvar,
                k_clusters=k_clusters,
                k_max=k_max,
                pcp_lambda=pcp_lambda,
                pcp_mu=pcp_mu,
                pcp_max_iter=pcp_max_iter,
                pcp_tol=pcp_tol,
                hqf_rank=hqf_rank,
                hqf_cumvar=hqf_cumvar,
                hqf_lambda=hqf_lambda,
                hqf_ip=hqf_ip,
                hqf_max_iter=hqf_max_iter,
                # No CV inside the bootstrap (CV picks the
                # hyperparameter once on the actual data; bootstrap
                # refits use that fixed value).
                cv_lambda=False,
                cv_hqf_rank=False,
                compute_cft_pi=False,
                random_state=random_state,
            )
            return star_fit.counterfactual

        cft_obj = cft_prediction_intervals(
            treated_outcome=treated_outcome,
            counterfactual=counterfactual,
            T0=T0,
            refit_fn=_refit,
            e_method=cft_e_method,
            alpha=cft_alpha,
            sims=cft_sims,
            random_state=random_state,
        )
        metadata["cft_inference"] = cft_obj

    return MethodFit(
        name=f"rpca_{rpca_method.lower()}",
        counterfactual=np.asarray(counterfactual, dtype=float),
        gap=gap,
        att=att,
        pre_rmse=pre_rmse,
        donor_weights=donor_weights,
        selected_donors=np.asarray(selected_names),
        metadata=metadata,
    )
