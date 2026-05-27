"""PCR-SC orchestration pipeline.

Implements Rho, Tang, Bergam, Cummings & Misra (2025), *ClusterSC:
Advancing Synthetic Control with Donor Selection*, by composing:

* :mod:`.hsvt` -- rank selection + HSVT denoising (Algorithm 2 Step 2).
* :mod:`.clustering` -- donor clustering and target-cluster assignment
  (Algorithms 3 and 4 Steps 1-3).
* :mod:`.frequentist`, :mod:`.bayesian`, :mod:`.convex` -- the three
  weight solvers.

The public entry point :func:`run_pcr` accepts the same call signature
the legacy ``estutils.pcr`` exposed (so the orchestrator and SI estimator
work unchanged) plus a handful of paper-aligned knobs:

* ``rank`` and ``rank_method`` (``"cumvar"`` / ``"fixed"`` / ``"usvt"``);
* ``k_clusters`` and ``k_max`` for the silhouette-driven clustering step;
* ``alpha`` and ``n_bayes_samples`` for the Bayesian credible band.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError
from ..structures import MethodFit
from .bayesian import solve_bayesian
from .clustering import ClusterPartition, assign_target, cluster_donors
from .convex import solve_simplex
from .frequentist import solve_ols
from .hsvt import hsvt, select_rank
from .inference import shen_inference

_MIN_PRE_PERIODS = 2
_MIN_CLUSTER_DONORS = 2


def run_pcr(
    treated_outcome: np.ndarray,
    donor_outcomes: np.ndarray,
    donor_names: Sequence[str],
    T0: int,
    *,
    objective: str = "OLS",
    estimator: str = "frequentist",
    clustering: bool = False,
    # rank knobs
    rank: Optional[int] = None,
    rank_method: str = "cumvar",
    cumvar_threshold: float = 0.95,
    standardize_for_rank: bool = True,
    # projection knob
    project_denoised: bool = False,
    # clustering knobs
    k_clusters: Optional[int] = None,
    k_max: int = 8,
    # bayesian knobs
    alpha: float = 0.05,
    n_bayes_samples: int = 1000,
    alpha_prior: float = 1.0,
    # frequentist regularisation knobs
    lambda_penalty: Optional[float] = None,
    p: Optional[float] = None,
    q: Optional[float] = None,
    # frequentist OLS inference (Shen et al. 2023)
    shen_variance: str = "homoskedastic",
    compute_shen_ci: bool = True,
    random_state: int = 0,
) -> Tuple[MethodFit, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Run the paper-aligned PCR-SC pipeline.

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
    objective : {"OLS", "SIMPLEX"}
        Which weight solver to use for ``estimator="frequentist"``.
        ``"OLS"`` is the paper's Algorithm 2; ``"SIMPLEX"`` is the
        Abadie-style mlsynth extension.
    estimator : {"frequentist", "bayesian"}
        Frequentist Algorithm 2 vs. the Bayani 2022 posterior.
    clustering : bool
        If True, run Algorithm 4 (donor clustering + target match).
    rank, rank_method, cumvar_threshold
        Truncation-rank controls; see :func:`.hsvt.select_rank`.
    k_clusters, k_max
        Cluster-count controls; see :func:`.clustering.cluster_donors`.
    alpha, n_bayes_samples, alpha_prior
        Bayesian-path controls; see :func:`.bayesian.solve_bayesian`.
    lambda_penalty, p, q
        Optional elastic-net knobs for the frequentist OLS path.
    random_state : int
        Seed forwarded to k-means and the Bayesian sampler.

    Returns
    -------
    fit : MethodFit
        Frozen container with the PCR fit (counterfactual projected
        through the *denoised* donor matrix, Algorithm 4 Step 5).
    credible_band : tuple of np.ndarray, optional
        ``(lower, upper)`` per-period credible bounds when
        ``estimator="bayesian"``; ``None`` otherwise.
    """
    if T0 < _MIN_PRE_PERIODS:
        raise MlsynthEstimationError(
            f"PCR requires T0 >= {_MIN_PRE_PERIODS}; got {T0}."
        )
    if objective not in {"OLS", "SIMPLEX"}:
        raise MlsynthEstimationError(
            f"objective must be 'OLS' or 'SIMPLEX'; got {objective!r}."
        )
    if estimator not in {"frequentist", "bayesian"}:
        raise MlsynthEstimationError(
            f"estimator must be 'frequentist' or 'bayesian'; got {estimator!r}."
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

    pre_donors = donor_outcomes[:T0]
    pre_target = treated_outcome[:T0]

    # ------------------------------------------------------------------
    # (Optional) Algorithm 4 Steps 1-3: cluster and select donor subset.
    # ------------------------------------------------------------------
    selected_full = donor_outcomes
    selected_names = donor_names
    cluster_info: Optional[ClusterPartition] = None
    target_cluster: Optional[int] = None

    if clustering:
        # Paper convention: rows = donors. Transpose so cluster_donors
        # sees X in (J, T0).
        partition = cluster_donors(
            donor_outcomes_pre=pre_donors.T,
            rank=_rank_for_clustering(
                X=pre_donors.T,
                rank=rank,
                rank_method=rank_method,
                cumvar_threshold=cumvar_threshold,
            ),
            k_clusters=k_clusters,
            k_max=k_max,
            random_state=random_state,
        )
        cluster_id, _ = assign_target(pre_target, partition)
        donor_idx = np.where(partition.labels == cluster_id)[0]
        if donor_idx.size < _MIN_CLUSTER_DONORS:
            raise MlsynthEstimationError(
                f"Cluster {cluster_id} has {donor_idx.size} donor(s); "
                f"need at least {_MIN_CLUSTER_DONORS}."
            )
        selected_full = donor_outcomes[:, donor_idx]
        selected_names = [donor_names[i] for i in donor_idx]
        cluster_info = partition
        target_cluster = cluster_id

    # ------------------------------------------------------------------
    # Algorithm 2 Step 1-2 (Amjad, Shah, Shen 2018 convention):
    # HSVT-denoise the *pre-period* donor matrix only. Using the full
    # (T, J) matrix for the denoising step leaks post-period donor
    # information into the rank-r reconstruction, which can wash out
    # the very post-period deviations the synthetic control is meant
    # to detect (we observed this on the California Proposition 99
    # panel: HSVT-on-full collapsed the ATT from ~-19 to ~-7 at r=4).
    # If the caller passes an explicit `rank`, promote to fixed-rank
    # truncation regardless of the default `rank_method`.
    # ------------------------------------------------------------------
    pre_donors = selected_full[:T0]
    effective_rank_method = "fixed" if rank is not None else rank_method
    r = select_rank(
        pre_donors,
        method=effective_rank_method,
        cumvar_threshold=cumvar_threshold,
        r=rank,
        standardize=standardize_for_rank,
    )
    denoised_pre, _, _, _ = hsvt(pre_donors, rank=r)
    # Optional post-period denoising for the projection step. Default
    # `project_denoised=False` keeps the post-period donor outcomes raw
    # (Amjad-Shah-Shen 2018 / canonical SCM). When True, we apply HSVT
    # to the full (T, J) matrix at the same rank so the counterfactual
    # is projected through M̂ in both periods (Rho et al. 2025
    # Algorithm 4 Step 5, paper-strict).
    if project_denoised:
        denoised_full, _, _, _ = hsvt(selected_full, rank=r)
        projection_full = denoised_full
    else:
        projection_full = selected_full

    # ------------------------------------------------------------------
    # Algorithm 2 Step 3 (paper) / mlsynth extensions: solve for f̂.
    # ------------------------------------------------------------------
    credible_band: Optional[Tuple[np.ndarray, np.ndarray]] = None

    if estimator == "frequentist":
        if objective == "OLS":
            f_hat = solve_ols(
                denoised_donor_pre=denoised_pre,
                target_pre=pre_target,
                donor_names=selected_names,
                lambda_penalty=lambda_penalty,
                p=p,
                q=q,
            )
        else:
            f_hat = solve_simplex(
                denoised_donor_pre=denoised_pre,
                target_pre=pre_target,
                donor_names=selected_names,
            )
        counterfactual = projection_full @ f_hat
        method_tag = f"pcr_{'simplex' if objective == 'SIMPLEX' else 'ols'}"
    else:
        # Bayesian: replace inner OLS with a Gaussian posterior over the
        # weights. The Bayesian predictive counterfactual (posterior mean) and
        # its credible band are projected through the *denoised* rank-r full
        # donor matrix M-hat -- the Amjad-Shah-Shen (2018) Bayesian model. This
        # keeps the band on the rank-r signal subspace so it is not inflated by
        # raw donor noise in the weight null space (independent of the
        # frequentist `project_denoised` flag).
        denoised_full, _, _, _ = hsvt(selected_full, rank=r)
        rng = np.random.default_rng(random_state)
        f_hat, counterfactual, cf_lo, cf_hi = solve_bayesian(
            denoised_donor_pre=denoised_pre,
            target_pre=pre_target,
            denoised_donor_full=denoised_full,
            alpha=alpha,
            n_samples=n_bayes_samples,
            alpha_prior=alpha_prior,
            rng=rng,
        )
        credible_band = (cf_lo, cf_hi)
        method_tag = "pcr_bayesian"

    # ------------------------------------------------------------------
    # Algorithm 4 Step 6: inferred ATT = y_post - m̂_0^+ averaged in post.
    # ------------------------------------------------------------------
    gap = treated_outcome - counterfactual
    att = float(np.mean(gap[T0:])) if T > T0 else float("nan")
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))

    donor_weights = {name: float(w) for name, w in zip(selected_names, f_hat)}

    # Shen et al. (2023) frequentist CIs for the OLS path.
    shen_obj = None
    if (
        compute_shen_ci
        and estimator == "frequentist"
        and objective == "OLS"
        and T > T0
    ):
        try:
            shen_obj = shen_inference(
                treated_outcome=treated_outcome,
                donor_outcomes=selected_full,
                T0=T0,
                rank=r,
                variance=shen_variance,
                alpha=alpha,
            )
        except MlsynthEstimationError:
            # Best-effort: e.g. HRK validity check fails on a panel.
            # Leave shen_obj = None and let the orchestrator fall back.
            shen_obj = None

    metadata: dict = {
        "objective": objective,
        "estimator": estimator,
        "clustering": bool(clustering),
        "rank": int(r),
        "rank_method": rank_method,
        "cumvar_threshold": float(cumvar_threshold),
        "standardize_for_rank": bool(standardize_for_rank),
        "project_denoised": bool(project_denoised),
        "lambda_penalty": lambda_penalty,
        "p": p,
        "q": q,
    }
    if shen_obj is not None:
        metadata["shen_inference"] = shen_obj
    if cluster_info is not None:
        metadata.update({
            "k_clusters": int(cluster_info.k),
            "target_cluster": int(target_cluster) if target_cluster is not None else None,
            "cluster_labels": cluster_info.labels.tolist(),
        })

    fit = MethodFit(
        name=method_tag,
        counterfactual=np.asarray(counterfactual, dtype=float),
        gap=gap,
        att=att,
        pre_rmse=pre_rmse,
        donor_weights=donor_weights,
        selected_donors=np.asarray(selected_names),
        metadata=metadata,
    )
    return fit, credible_band


def _rank_for_clustering(
    X: np.ndarray,
    rank: Optional[int],
    rank_method: str,
    cumvar_threshold: float,
) -> int:
    """Rank passed to Algorithm 3.

    The clustering step needs an explicit truncation rank for the SVD
    feature map. We reuse :func:`select_rank` on the same matrix the
    weight step will see (transposed if needed), so clustering and HSVT
    share their rank by default.
    """
    effective = "fixed" if rank is not None else rank_method
    return select_rank(
        X, method=effective, cumvar_threshold=cumvar_threshold, r=rank,
    )
