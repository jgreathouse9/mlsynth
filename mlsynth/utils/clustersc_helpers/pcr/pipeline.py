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

Missing donor cells (NaN in ``donor_outcomes``) take the two-step PCR of
Agarwal, Shah and Shen (JMLR 2025), Section 2.2, with a donor-mean fill, in
:mod:`.missing`; with every cell observed that path is the complete-data one
bit for bit, so the complete-data code below runs unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthDataError, MlsynthEstimationError
from ..structures import MethodFit
from .bayesian import solve_bayesian
from .clustering import ClusterPartition, assign_target, cluster_donors
from ..spannability import assess_spannability, warn_if_poorly_spanned
from .convex import solve_simplex
from .frequentist import solve_ols
from .hsvt import hsvt, select_rank
from .inference import shen_inference
from .missing import (
    check_missing_supported,
    denoise_observed,
    pre_period_means,
    refuse_empty_periods,
    rescale_observed,
    two_block_counterfactual,
)

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
    # scpi prediction intervals (Cattaneo-Feng-Palomba-Titiunik 2025)
    compute_scpi_pi: bool = False,
    scpi_constraint: str = "ridge",
    scpi_sims: int = 200,
    scpi_e_method: str = "gaussian",
    random_state: int = 0,
    time_labels: Optional[Sequence] = None,
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
    time_labels : sequence, optional
        Length-``T`` period labels, used to name a period in an error.

    Missing donor cells (NaN) are handled by the two-step PCR of Agarwal,
    Shah and Shen (JMLR 2025, Section 2.2), with each donor centred at its
    observed mean before the rescaling (:mod:`.missing`), on the
    frequentist OLS path without clustering, projection through the
    de-noised matrix, or Shen / scpi inference; other options raise
    :class:`~mlsynth.exceptions.MlsynthDataError`. A missing treated
    cell, a period with no observed donor and a donor with no observed
    pre-period cell are refused.

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
    if np.isnan(treated_outcome).any():
        raise MlsynthDataError(
            "The treated outcome has missing cells; PCR regresses the treated "
            "unit's pre-period series on the donors (Agarwal, Shah and Shen "
            "2025, Section 2.2, Step I) and defines no missing response."
        )
    observed = ~np.isnan(donor_outcomes)
    has_missing = not bool(observed.all())
    if has_missing:
        check_missing_supported(
            method="pcr", clustering=clustering, estimator=estimator,
            objective=objective, lambda_penalty=lambda_penalty,
            project_denoised=project_denoised, compute_shen_ci=compute_shen_ci,
            compute_scpi_pi=compute_scpi_pi,
        )
        refuse_empty_periods(observed, time_labels)

    pre_donors = donor_outcomes[:T0]
    pre_target = treated_outcome[:T0]

    # ------------------------------------------------------------------
    # (Optional) Algorithm 4 Steps 1-3: cluster and select donor subset.
    # ------------------------------------------------------------------
    selected_full = donor_outcomes
    selected_names = donor_names
    cluster_info: Optional[ClusterPartition] = None
    target_cluster: Optional[int] = None
    spannability_meta: dict = {}

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

        # Clustering optimises trajectory similarity and carries no
        # spannability term, so the surviving donors need not still reach the
        # treated unit. Measured and reported; see `..spannability`.
        if donor_idx.size < donor_outcomes.shape[1]:
            report = assess_spannability(
                donor_pre_pool=donor_outcomes[:T0],
                treated_pre=pre_target,
                cluster_index=donor_idx,
            )
            warn_if_poorly_spanned(report)
            spannability_meta = {
                "spannability_ratio": report.ratio,
                "spannability_excluded_mass": report.excluded_mass,
                "spannability_cluster_rmse": report.cluster_rmse,
                "spannability_pool_rmse": report.pool_rmse,
            }

    # ------------------------------------------------------------------
    # HSVT-denoise the *pre-period* donor matrix only: Step I of Agarwal,
    # Shah and Shen (JMLR 2025, Section 2.2), the forecast of Agarwal,
    # Amjad, Shah and Shen (2018, Section 3.3; the authors' tslib), and the
    # PCR of Shen, Ding, Sekhon and Yu (arXiv:2207.14481). The 2018 RSC
    # paper's Algorithm 1 (eq. 8) writes the truncation over the full
    # (T, J) matrix instead; on the California Proposition 99 panel that
    # moves the ATT from about -19 to about -7 at r=4.
    # If the caller passes an explicit `rank`, promote to fixed-rank
    # truncation regardless of the default `rank_method`.
    # ------------------------------------------------------------------
    pre_donors = selected_full[:T0]
    effective_rank_method = "fixed" if rank is not None else rank_method
    if has_missing:
        # Agarwal, Shah and Shen (2025), Step I: the rank rule and the
        # truncation both read Z-tilde, the block filled at the donor means
        # and rescaled by rho-hat. Clustering is refused with missing cells,
        # so selected_full is the whole pool.
        pre_means = pre_period_means(pre_donors, observed[:T0], selected_names)
        z_tilde, _ = rescale_observed(pre_donors, observed[:T0], pre_means)
        r = select_rank(
            z_tilde,
            method=effective_rank_method,
            cumvar_threshold=cumvar_threshold,
            r=rank,
            standardize=standardize_for_rank,
        )
        observed_fit = denoise_observed(pre_donors, observed[:T0], r,
                                        selected_names)
        denoised_pre = observed_fit.M_hat
    else:
        r = select_rank(
            pre_donors,
            method=effective_rank_method,
            cumvar_threshold=cumvar_threshold,
            r=rank,
            standardize=standardize_for_rank,
        )
        denoised_pre, _, _, _ = hsvt(pre_donors, rank=r)
    # Optional post-period denoising for the projection step. Default
    # `project_denoised=False` keeps the post-period donor outcomes raw,
    # which is Step II of Agarwal, Shah and Shen (2025) at full rank. When
    # True, we apply HSVT to the full (T, J) matrix at the same rank so the
    # counterfactual is projected through M̂ in both periods (Rho et al.
    # 2025 Algorithm 4 Step 5); that is not Step II, which truncates the
    # post block on its own.
    if project_denoised:
        denoised_full, _, _, _ = hsvt(selected_full, rank=r)
        projection_full = denoised_full
    else:
        projection_full = selected_full

    # ------------------------------------------------------------------
    # Algorithm 2 Step 3 (paper) / mlsynth extensions: solve for f̂.
    # The HSVT rank goes with the design, so the OLS weights are truncated
    # at r and the rank is not decided again on the reconstruction.
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
                rank=r,
            )
        else:
            f_hat = solve_simplex(
                denoised_donor_pre=denoised_pre,
                target_pre=pre_target,
                donor_names=selected_names,
            )
        if has_missing:
            # Step II at full rank on the post block, with its own rho-hat'
            # and post-period means; the in-sample fit Z-tilde @ f_hat on the
            # pre block.
            counterfactual, p_hat_post, n_fallback = two_block_counterfactual(
                selected_full, observed, T0, f_hat, observed_fit.means,
            )
        else:
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
        **spannability_meta,
    }
    if has_missing:
        metadata["missing_p_hat"] = float(observed_fit.p_hat)
        metadata["missing_p_hat_post"] = float(p_hat_post)
        metadata["missing_fill"] = "donor_mean"
        metadata["missing_post_mean_fallbacks"] = int(n_fallback)
        metadata["missing_donor_cells"] = int((~observed).sum())
    if shen_obj is not None:
        metadata["shen_inference"] = shen_obj

    # scpi prediction intervals under the chosen constraint (ridge = Amjad et al.
    # 2018 Robust SC per scpi Table 3). Runs on the fitted weights and the donor
    # design the counterfactual projects through (counterfactual = projection_full
    # @ f_hat), so scpi's synthetic-prediction model is exactly the PCR fit.
    if compute_scpi_pi and T > T0 and estimator == "frequentist":
        from ..scpi_pi import scpi_pi_inference
        try:
            metadata["scpi_inference"] = scpi_pi_inference(
                treated_outcome, projection_full, T0, f_hat,
                constraint=scpi_constraint, sims=scpi_sims, alpha=alpha,
                e_method=scpi_e_method, seed=random_state,
                periods=list(range(T0, T)),
            )
        except (MlsynthEstimationError, ValueError, ImportError):
            metadata["scpi_inference"] = None

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
