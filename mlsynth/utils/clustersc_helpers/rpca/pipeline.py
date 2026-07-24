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

from ....exceptions import MlsynthConfigError, MlsynthEstimationError
from ...pcr.core import hsvt as _hsvt
from ...pcr.rank import select_rank as _select_rank
from ..pcr.convex import solve_simplex as _solve_simplex
from ..structures import MethodFit
from .clustering import FPCACluster, assign_clusters
from .fgrc import fgrc_cluster as _fgrc_cluster
from .fpca import FPCAFeatures, compute_fpca_features
from .hqf import HQFResult, hqf_decompose
from .pcp import PCPResult, pcp_decompose
from .inference import cft_prediction_intervals
from .tuning import cv_hqf_rank as _cv_hqf_rank
from .tuning import cv_pcp_lambda
from .weights import solve_nnls

_RPCA_METHODS = {"PCP", "HQF", "HSVT"}
_CLUSTER_METHODS = {"fpca", "fgrc"}
_WEIGHT_OBJECTIVES = {"nnls", "simplex"}

_MIN_PRE = 2
_MIN_DONORS = 2


def run_rpca(
    treated_outcome: np.ndarray,
    donor_outcomes: np.ndarray,
    donor_names: Sequence[str],
    T0: int,
    *,
    rpca_method: str = "PCP",
    cluster_method: str = "fpca",
    weight_objective: str = "nnls",
    # FPCA / clustering knobs
    fpca_cumvar: float = 0.95,
    k_clusters: Optional[int] = None,
    k_max: int = 8,
    # fGRC clustering knobs (cluster_method="fgrc")
    fgrc_c1: int = 2,
    fgrc_c2: int = 1,
    fgrc_k: Optional[int] = None,
    fgrc_knots: Optional[int] = None,
    fgrc_order: int = 4,
    # HSVT denoiser knobs (rpca_method="HSVT")
    hsvt_rank_method: str = "usvt",
    hsvt_rank: Optional[int] = None,
    hsvt_cumvar: float = 0.95,
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
    # scpi prediction-interval knobs (Cattaneo-Feng-Palomba-Titiunik 2025)
    compute_scpi_pi: bool = False,
    scpi_constraint: str = "ridge",
    scpi_sims: int = 200,
    scpi_e_method: str = "gaussian",
    scpi_alpha: float = 0.05,
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
    if rpca_method not in _RPCA_METHODS:
        raise MlsynthEstimationError(
            f"rpca_method must be one of {sorted(_RPCA_METHODS)}; got {rpca_method!r}."
        )
    if cluster_method not in _CLUSTER_METHODS:
        raise MlsynthConfigError(
            f"cluster_method must be one of {sorted(_CLUSTER_METHODS)}; got {cluster_method!r}."
        )
    if weight_objective not in _WEIGHT_OBJECTIVES:
        raise MlsynthConfigError(
            f"weight_objective must be one of {sorted(_WEIGHT_OBJECTIVES)}; got {weight_objective!r}."
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
    # Steps 1-2: cluster the pre-period trajectories and keep the treated
    # unit's cluster as the donor pool. Default: FPCA + silhouette k-means
    # (Bayani 2021). Alternative: fGRC subspace-separated clustering, which
    # projects out a shared disturbing trend before grouping.
    # ------------------------------------------------------------------
    full_pre_panel = np.vstack([treated_outcome[:T0], donor_outcomes[:T0].T])
    if cluster_method == "fpca":
        features: FPCAFeatures = compute_fpca_features(
            pre_outcomes=full_pre_panel, cumvar_threshold=fpca_cumvar,
        )
        cluster: FPCACluster = assign_clusters(
            scores=features.scores, treated_row=0,
            k_clusters=k_clusters, k_max=k_max, random_state=random_state,
        )
        # donor_indices is in *full panel* coordinates; subtract 1 to get
        # back to the donor_outcomes column indices.
        donor_col_idx = cluster.donor_indices - 1
        donor_col_idx = donor_col_idx[donor_col_idx >= 0]
        cluster_meta = {
            "cluster_method": "fpca",
            "fpca_cumvar": float(fpca_cumvar),
            "fpca_rank": int(features.rank),
            "fpca_smoothing": features.smoothing,
            "k_clusters": int(cluster.k),
            "treated_cluster": int(cluster.treated_cluster),
            "cluster_labels": cluster.labels.tolist(),
        }
    else:  # cluster_method == "fgrc"
        n_units = full_pre_panel.shape[0]
        k = int(fgrc_k) if fgrc_k is not None else 2
        knots = int(fgrc_knots) if fgrc_knots is not None else max(4, T0 // 2 - 2)
        labels, fgrc_loss = _fgrc_cluster(
            full_pre_panel, c1=fgrc_c1, c2=fgrc_c2, k=k,
            n_knots=knots, order=fgrc_order, seed=random_state,
        )
        treated_cluster = int(labels[0])
        donor_col_idx = np.where(labels[1:] == treated_cluster)[0]
        cluster_meta = {
            "cluster_method": "fgrc",
            "fgrc_c1": int(fgrc_c1), "fgrc_c2": int(fgrc_c2), "fgrc_k": int(k),
            "fgrc_knots": int(knots), "fgrc_order": int(fgrc_order),
            "fgrc_loss": float(fgrc_loss),
            "treated_cluster": treated_cluster,
            "cluster_labels": labels.tolist(),
        }
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
        L_full = result.low_rank.T
        solver_metadata = {
            "pcp_lambda": result.lambda_used,
            "pcp_mu": result.mu_used,
            "pcp_iterations": result.iterations,
            "pcp_converged": result.converged,
        }
    elif rpca_method == "HQF":
        result = hqf_decompose(
            Y=donor_matrix,
            rank=hqf_rank,
            cumvar_threshold=hqf_cumvar,
            lam=hqf_lambda,
            ip=hqf_ip,
            max_iter=hqf_max_iter,
            random_state=random_state,
        )
        L_full = result.low_rank.T
        solver_metadata = {
            "hqf_rank": result.rank_used,
            "hqf_lambda": result.lambda_used,
            "hqf_ip": result.ip_used,
            "hqf_iterations": result.iterations,
        }
    else:  # rpca_method == "HSVT" -- hard singular-value truncation (RSC/PCR-native)
        if hsvt_rank_method == "fixed":
            r_hsvt = _select_rank(donor_matrix, method="fixed", r=hsvt_rank)
        elif hsvt_rank_method == "cumvar":
            r_hsvt = _select_rank(donor_matrix, method="cumvar", cumvar_threshold=hsvt_cumvar)
        else:  # "usvt" -- Donoho-Gavish optimal hard threshold
            r_hsvt = _select_rank(donor_matrix, method="usvt")
        L_donor, _u, _s, _vt = _hsvt(donor_matrix, r_hsvt)
        L_full = L_donor.T
        solver_metadata = {"hsvt_rank": int(r_hsvt), "hsvt_rank_method": hsvt_rank_method}

    # shape (T, n_donors), donors as columns
    L_pre = L_full[:T0]

    # ------------------------------------------------------------------
    # Step 4: fit weights against the denoised pre-period donors. Default is
    # non-negative LS (Bayani 2021); "simplex" adds the Abadie-Diamond-
    # Hainmueller sum-to-one convex-hull constraint on the denoised donors.
    # ------------------------------------------------------------------
    if weight_objective == "simplex":
        beta = _solve_simplex(L_pre, treated_outcome[:T0])
    else:
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
        "weight_objective": weight_objective,
        **cluster_meta,
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
                cluster_method=cluster_method,
                weight_objective=weight_objective,
                fgrc_c1=fgrc_c1,
                fgrc_c2=fgrc_c2,
                fgrc_k=fgrc_k,
                fgrc_knots=fgrc_knots,
                fgrc_order=fgrc_order,
                hsvt_rank_method=hsvt_rank_method,
                hsvt_rank=hsvt_rank,
                hsvt_cumvar=hsvt_cumvar,
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
                compute_scpi_pi=False,
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

    # scpi prediction intervals: run on the denoised donor design and the NNLS
    # weights (counterfactual = L_full @ beta). Constraint chosen by the caller
    # (simplex matches the RPCA non-negative weights).
    if compute_scpi_pi and T > T0:
        from ..scpi_pi import scpi_pi_inference
        try:
            metadata["scpi_inference"] = scpi_pi_inference(
                treated_outcome, L_full, T0, beta,
                constraint=scpi_constraint, sims=scpi_sims, alpha=scpi_alpha,
                e_method=scpi_e_method, seed=random_state,
                periods=list(range(T0, T)),
            )
        except (MlsynthEstimationError, ValueError, ImportError):
            metadata["scpi_inference"] = None

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
