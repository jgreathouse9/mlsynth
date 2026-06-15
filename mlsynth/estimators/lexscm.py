import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List, Literal

# Configuration and Exceptions
from ..config_models import BaseMAREXConfig, LEXSCMConfig, WeightsResults
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.post_fit import to_effect_result as post_fit_to_effect_result
from ..utils.helperutils import lexplot
# Utilities - Data Handling
from ..utils.datautils import balance

# Utilities - Fast SCM Core Setup
from ..utils.fast_scm_helpers.fast_scm_setup import (
    _prepare_working_df,
    build_candidate_mask,
    build_f_vector,
    build_X_tilde,
    build_Y_matrix,
    build_Z_matrix,
    prepare_experiment_inputs,
    split_periods, IndexSet, _run_post_intervention_updates
)

# Utilities - Search and Evaluation
from ..utils.fast_scm_helpers.lexsearch import select_treated_designs
from ..utils.fast_scm_helpers.fast_scm_control import evaluate_candidates
from ..utils.fast_scm_helpers.conflict import build_conflict_matrix
# Utilities - Power and Ranking
from ..utils.fast_scm_helpers.lexpower import detectability_curve
from ..utils.fast_scm_helpers.lexselect import DesignMetrics, select_design

from ..utils.fast_scm_helpers.inference import compute_moving_block_conformal_ci

from dataclasses import dataclass, field

from ..utils.fast_scm_helpers.structure import (
    SEDCandidate,
    LEXSCMResults,
    LEXSCMSearch,
    LEXSCMPanel,
    UnitInfo,
    TimeInfo
)


class LEXSCM:
    """
    Lexicographic Synthetic Control (LEXSCM) estimator.

    This estimator automatically designs synthetic control experiments by jointly
    optimizing:

        (i) Pre-treatment fit (validity)
        (ii) Statistical power (detectability of effects)

    The user interacts with the estimator through a single entry point:
    `fit()`, while all modeling choices are controlled via the configuration object.

    Parameters
    ----------
    config : dict or LEXSCMConfig
        Configuration object specifying data inputs, identification strategy,
        synthetic control settings, search budget, and inference parameters.

        The following fields are supported:

        **Required**

        - ``df`` (pd.DataFrame): Panel dataset containing unit-level
          observations over time.
        - ``outcome`` (str): Name of the outcome variable.
        - ``unitid`` (str): Column identifying observational units.
        - ``time`` (str): Time index column.
        - ``candidate_col`` (str): Boolean (0/1 or True/False) column
          indicating units eligible for treatment assignment.
        - ``m`` (int): Number of units selected per treated tuple.

        **Identification / Design**

        - ``post_col`` (str, optional): Indicator for post-treatment
          period (0/1).
        - ``frac_E`` (float): Fraction of pre-treatment period used for
          estimation window E.
        - ``unit_cost_col`` (str, optional): Per-unit treatment cost column
          (must be constant within unit).
        - ``budget`` (float, optional): Total budget constraint on selected
          treated units.
        - ``seed`` (int): Random seed for reproducibility.

        **Synthetic Control Specification**

        - ``weight_col`` (str, optional): Unit-level weights (e.g.,
          population, revenue).
        - ``covariates`` (list of str, optional): Covariates included in
          synthetic control construction.
        - ``lambda_penalty`` (float): Regularization penalty for control
          mismatch in quadratic program.

        **Search / Optimization Budget**

        - ``top_K`` (int): Number of top candidate tuples returned by the
          treated-design selection search.
        - ``top_P`` (int): Deprecated/unused (the multi-start search seeds
          itself); retained for backward-compatible configs.

        **Inference / Power Analysis**

        - ``alpha`` (float): Significance level for statistical testing.
        - ``n_post_grid`` (list of int): Post-treatment horizons used for
          MDE detectability curves.
        - ``n_sims`` (int): Number of Monte Carlo simulations for null
          distribution.
        - ``post_imputation`` ({"mean", "max", "double_max"}): Method for
          imputing post-treatment outcomes.
        - ``test_statistic`` ({"mean_abs", "mean", "rms"}): Test statistic
          used for treatment effect evaluation.
        - ``delta`` (float): Absolute minimum detectable effect threshold.
        - ``relative_delta`` (float, optional): Relative MDE threshold
          (> 1.0).
        - ``target_mde_horizon`` (str): Target horizon for MDE evaluation
          (e.g., early_mde_avg).
        - ``max_shortlist`` (int): Maximum number of candidate designs
          retained after filtering.

        **System / Logging**

        - ``verbose`` (bool): If True, enables progress logging.

    Returns
    -------
    LEXSCM
        Initialized estimator ready for `.fit()` execution.

    See Also
    --------
    LEXSCM.fit : Executes the full optimization and estimation pipeline

    Notes
    -----
    - The pipeline integrates: search, estimation, evaluation, power analysis, selection.
    - Designed for fully automated experimental design under constraints.

    References
    ----------
    This implementation is based on the synthetic experimental design framework:

    - https://economics.mit.edu/sites/default/files/2026-02/Synthetic%20Controls%20for%20Experimental%20Design%20Feb%202026.pdf
      Develops the formal framework for selecting treated units using synthetic controls
      to reduce bias and improve experimental design in aggregate settings.

    - https://ivalua.cat/sites/default/files/2023-03/Vives-i-Bastida_2022_anon.pdf
      Provides a practical guide to synthetic experimental design in policy contexts,
      including inference and design trade-offs.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            try:
                config = LEXSCMConfig(**config)
            except Exception as e:  # pydantic ValidationError
                raise MlsynthDataError(f"Invalid LEXSCM configuration: {e}") from e

        # =========================================================
        # CORE DATA
        # =========================================================
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time

        # =========================================================
        # IDENTIFICATION
        # =========================================================
        self.candidate_col: str = config.candidate_col
        self.post_col: Optional[str] = config.post_col
        self.m: int = config.m
        self.weight_col: Optional[str] = config.weight_col
        self.unit_cost_col: Optional[str] = config.unit_cost_col
        self.budget: Optional[float] = config.budget
        self.cluster_col: Optional[str] = config.cluster_col
        self.adjacency = config.adjacency
        self.spillover_threshold: float = config.spillover_threshold
        self.stratum_col: Optional[str] = config.stratum_col
        self.min_per_stratum: Optional[int] = config.min_per_stratum
        self.max_per_stratum: Optional[int] = config.max_per_stratum
        self.size_col: Optional[str] = config.size_col
        self.min_size: Optional[float] = config.min_size
        self.max_size: Optional[float] = config.max_size
        self.frac_E: float = config.frac_E

        # =========================================================
        # SYNTHETIC CONTROL
        # =========================================================
        self.covariates: Optional[list] = config.covariates
        self.lambda_penalty: float = config.lambda_penalty

        # =========================================================
        # SEARCH / BNB
        # =========================================================
        self.top_K: int = config.top_K
        self.top_P: int = config.top_P
        self.targeting_penalty: float = config.targeting_penalty

        # =========================================================
        # INFERENCE / POWER
        # =========================================================
        self.alpha: float = config.alpha
        self.n_post_grid: list = config.n_post_grid
        self.n_sims: int = config.n_sims
        self.post_imputation: str = config.post_imputation
        self.test_statistic: str = config.test_statistic
        self.target_mde_horizon: Literal["early_mean", "early_min", "late"] = config.mde_horizon
        self.max_shortlist: int = config.max_shortlist
        self.power_target: float = config.power_target
        self.imbalance_tol: float = config.imbalance_tol

        self.display_graph: bool = config.display_graph

        # =========================================================
        # SYSTEM
        # =========================================================
        self.seed: int = config.seed
        self.verbose: bool = config.verbose

    def _representative_mde(self, dc: dict):
        """Collapse a curve to one (mde_sd, mde_abs, mde_pct) per `mde_horizon`.

        - "late"       : MDE at the longest horizon.
        - "early_min"  : smallest MDE across horizons (most optimistic).
        - "early_mean" : mean MDE across feasible horizons.

        ``mde_pct`` is the percentage relative to the counterfactual level and may
        be ``NaN`` when that level is not a trustworthy magnitude (see
        :func:`~mlsynth.utils.fast_scm_helpers.lexpower.compute_mde`).
        """
        det = dc["details"]
        horizons = sorted(det.keys())
        rows = [(w, det[w]["mde_sd"], det[w]["mde_abs"], det[w].get("mde_pct", np.nan))
                for w in horizons]
        finite = [(w, s, a, p) for w, s, a, p in rows if np.isfinite(s)]
        if self.target_mde_horizon == "late":
            _, s, a, p = rows[-1]
            return s, a, p
        if not finite:
            return np.inf, np.inf, np.nan
        if self.target_mde_horizon == "early_min":
            _, s, a, p = min(finite, key=lambda x: x[1])
            return s, a, p
        # early_mean (average over feasible horizons; percentage averaged only
        # over horizons where it is defined)
        pcts = [p for _, _, _, p in finite if np.isfinite(p)]
        return (float(np.mean([s for _, s, _, _ in finite])),
                float(np.mean([a for _, _, a, _ in finite])),
                float(np.mean(pcts)) if pcts else np.nan)

    def fit(self, **kwargs) -> "LEXSCMResults":
        """
        Run the full Synthetic Experiment Design pipeline.

        This method executes the end-to-end workflow:
            - prepares panel data
            - searches over candidate treated unit sets
            - fits synthetic controls
            - evaluates pre-treatment fit
            - estimates statistical power (MDE)
            - selects the optimal design

        Parameters
        ----------
        **kwargs : dict, optional
            Reserved for future extensions (currently unused).

        Returns
        -------
        LEXSCMResults
            Object containing:
            - summary : ranked table of candidate designs
            - best_candidate : selected experimental design
            - all_candidates : full evaluated set
            - selection_metadata : search diagnostics
            - additional dataset and inference diagnostics
        """

        balance(self.df, self.unitid, self.time)

        # Step 1: Prepare working DataFrame using the helper
        working_df, self.post_df = _prepare_working_df(
            self.df,
            self.post_col
        )
        self.pre_df = working_df  # store for convenience if desired

        unit_index = IndexSet.from_labels(
            sorted(working_df[self.unitid].unique())
        )

        time_index = IndexSet.from_labels(
            sorted(working_df[self.time].unique())
        )

        full_time_index = None

        if self.post_col is not None and not self.post_df.empty:
            full_time_index = IndexSet.from_labels(
                sorted(self.df[self.time].unique())
            )

        final_time_index = full_time_index if full_time_index is not None else time_index

        # Step 2: Build candidate mask (aligned with future Y columns)
        candidate_mask = build_candidate_mask(
            working_df=working_df,
            candidate_col=self.candidate_col,
            unit_index=unit_index,
            unitid=self.unitid
        )

        # --- Treated-unit size band: drop candidates outside [min_size, max_size]
        # from the TREATABLE pool (they remain eligible as donors). ---
        if self.size_col is not None:
            size_map = (self.df.groupby(self.unitid)[self.size_col]
                        .first().to_dict())
            sizes = np.array([size_map.get(lab, np.nan)
                              for lab in unit_index.labels], dtype=float)
            elig = ~np.isnan(sizes)
            if self.min_size is not None:
                elig &= sizes >= self.min_size
            if self.max_size is not None:
                elig &= sizes <= self.max_size
            candidate_mask = np.asarray(candidate_mask, dtype=bool) & elig
        size_band = (None if self.size_col is None
                     else (self.min_size, self.max_size))

        # The candidate pool is needed before the matrices are built, so report a
        # too-small pool here in the SAME shape the Stage-1 audit uses for every
        # other binding constraint (e.g. when a size band leaves fewer than m).
        n_elig = int(np.asarray(candidate_mask, dtype=bool).sum())
        if n_elig < self.m:
            within = "" if size_band is None else \
                f" within the size band [{self.min_size}, {self.max_size}]"
            raise MlsynthConfigError(
                "LEXSCM design is infeasible -- the binding constraint(s):\n  - "
                f"candidate pool: only {n_elig} eligible market(s){within}, but "
                f"m={self.m}. Widen eligibility / the size band, or reduce m."
            )

        # Step 3: Build Y matrix
        self.Y = build_Y_matrix(
            working_df=working_df,
            outcome=self.outcome,
            time=self.time,
            unitid=self.unitid, unit_index=unit_index
        )

        # Step 4: Build Z matrix (covariates)
        self.Z = build_Z_matrix(
            working_df=working_df,
            covariates=self.covariates,
            time=self.time,
            unitid=self.unitid, unit_index=unit_index
        )

        # Step 5: Build f weighting vector
        self.f = build_f_vector(
            working_df=working_df,
            weight_col=self.weight_col,
            unitid=self.unitid, unit_index=unit_index
        )

        X, f, candidate_idx, T0_pre, N = prepare_experiment_inputs(
            self.Y, self.Z, self.f, candidate_mask, self.m
        )

        n_covs = self.Z.shape[0] if self.Z is not None else 0

        # 2. Split the periods and capture the new time-count metadata
        E_idx, B_idx, post_idx, n_fit_time, n_blank_time = split_periods(
            T0=self.Y.shape[0],
            n_covariates=n_covs,
            frac_E=self.frac_E,
            post_df=self.post_df,  # Ensure this is passed if you have post-treatment data
            time_col=self.time
        )

        # Store as a simple dictionary for the results object
        time_metadata = {
            "n_fit_time": n_fit_time,
            "n_blank_time": n_blank_time,
            "n_post": len(post_idx),
            "index_labels": working_df[self.time].unique()  # Actual X-axis values
        }

        # Standardize over estimation period
        X_E, G = build_X_tilde(X, f, E_idx)

        # --- Cost Alignment Logic ---
        if self.unit_cost_col and self.unit_cost_col in self.df.columns:
            # Map costs to Unit IDs
            unit_to_cost_map = self.df.groupby(self.unitid)[self.unit_cost_col].first().to_dict()

            # Align to the IndexSet
            costs_aligned = np.array([
                unit_to_cost_map.get(label, 0.0) for label in unit_index.labels
            ])
        else:
            # If no cost column provided, every unit is free
            costs_aligned = np.zeros(len(unit_index))

        # --- Budget Sanitization ---
        # If no budget is provided, set to infinity so no pruning occurs
        effective_budget = self.budget if self.budget is not None else np.inf

        # --- Spillover conflict graph (Vives-i-Bastida interference exclusions) ---
        # The IndexSet is the source of truth, so build the (J, J) conflict matrix
        # aligned to it; one graph drives both Stage-1 (no co-treating conflicting
        # units) and Stage-2 (no conflicting unit as a treated unit's donor).
        cluster_of = None
        if self.cluster_col is not None:
            cluster_of = (self.df.groupby(self.unitid)[self.cluster_col]
                          .first().to_dict())
        conflict = build_conflict_matrix(
            unit_index,
            cluster_of=cluster_of,
            adjacency=self.adjacency,
            spillover_threshold=self.spillover_threshold,
        )

        # --- Coverage / stratification quotas (treated-set per-stratum min/max) ---
        from ..utils.fast_scm_helpers.strata import build_strata
        stratum_of = None
        if self.stratum_col is not None:
            stratum_of = (self.df.groupby(self.unitid)[self.stratum_col]
                          .first().to_dict())
        strata = build_strata(unit_index, stratum_of)

        # ---------- Stage 1: treated-tuple selection (lexsearch) ----------
        search = select_treated_designs(
            G=G,
            candidate_idx=candidate_idx,
            m=self.m,
            top_K=self.top_K,
            unit_costs=costs_aligned,
            budget=(None if np.isinf(effective_budget) else effective_budget),
            unit_index=unit_index,
            method="auto",
            random_state=self.seed,
            conflict=conflict,
            strata=strata,
            min_per_stratum=self.min_per_stratum,
            max_per_stratum=self.max_per_stratum,
            size_band=size_band,
            targeting_penalty=self.targeting_penalty,
        )
        selection_results = {"top_tuples": search["top_designs"], "stats": search["stats"]}

        # --------------- Stage 2: control fit (unchanged) -----------------
        candidate_results = evaluate_candidates(
            candidates=search["top_designs"],
            X=X,
            X_E=X_E,
            Y=self.Y,
            f=self.f,
            E_idx=E_idx,
            B_idx=B_idx,
            lambda_penalty=self.lambda_penalty, index_set=unit_index,
            conflict=conflict,
        )

        # ----- Stage 3: power / MDE (moving-block placebo null) -----------
        design_metrics = []
        for c in candidate_results:
            dc = detectability_curve(
                np.asarray(c.predictions.residuals_B),
                self.n_post_grid,
                baseline_series=np.asarray(c.predictions.synthetic_treated),
                alpha=self.alpha,
                power_target=self.power_target,
                random_state=self.seed,
            )
            c.mde_results = dc
            mde_sd, mde_abs, mde_pct = self._representative_mde(dc)
            sol = c.identification.solution
            design_metrics.append(DesignMetrics(
                design_id=c.identification.tuple_id,
                indices=list(c.identification.treated_idx),
                labels=getattr(sol, "labels", []),
                # Outcome-only pre-fit imbalance: the QP weight-solve matches on
                # outcomes + covariates, but the reported/selection imbalance is
                # the synthetic-control RMSE over the OUTCOME estimation rows only
                # (covariates never enter an RMSE; their balance is the SMD).
                imbalance=float(c.losses.rmse_sc_E),
                mde_sd=mde_sd,
                mde_abs=mde_abs,
                mde_pct=mde_pct,
                mde_feasible=bool(np.isfinite(mde_sd)),
                stability=float(c.losses.nmse_B),
                total_cost=float(getattr(sol, "total_cost", 0.0)),
            ))

        y_pop_mean_t = self.Y.mean(axis=1)

        # --------- Stage 4: final recommendation (lexicographic) ----------
        recommendation = select_design(
            design_metrics,
            imbalance_tol=self.imbalance_tol,
            max_shortlist=self.max_shortlist,
        )
        shortlist = pd.DataFrame(recommendation.table)
        selection_results["recommendation"] = {
            "status": recommendation.status,
            "winner": recommendation.winner.design_id if recommendation.winner else None,
            "pareto_ids": recommendation.pareto_ids,
            "explanation": recommendation.explanation,
        }

        if self.post_col is not None and not self.post_df.empty:
            y_pop_mean_t, candidate_results = _run_post_intervention_updates(
                candidate_results=candidate_results,
                Y_pre=self.Y,
                post_df=self.post_df,
                post_idx=post_idx,
                unit_index=unit_index,
                unitid=self.unitid,
                time=self.time,
                outcome=self.outcome,
                n_sims=self.n_sims,
                alpha=self.alpha,
                seed=self.seed
            )

        # ==============================================================
        # 6. Final Assembly
        # ==============================================================
        # Re-fetch the winner (now with updated post-intervention results)
        winner_id = recommendation.winner.design_id if recommendation.winner else None
        best_candidate = next(
            (c for c in candidate_results
             if c.identification.tuple_id == winner_id),
            candidate_results[0]  # fallback
        )

        # =========================================================
        # BUILD TIME METADATA
        # =========================================================
        time_info = TimeInfo(
            n_total=len(final_time_index),
            n_pre=n_fit_time + n_blank_time,
            n_fit_time=n_fit_time,  # Pure time count for plotting
            n_blank_time=n_blank_time,
            n_fit=len(E_idx),  # Math count (includes covariates) for diagnostics
            n_blank=len(B_idx),
            n_post=len(post_idx),
            index=final_time_index
        )

        # =========================================================
        # BUILD UNIT METADATA
        # =========================================================
        best = best_candidate

        # The IndexSet is the single source of truth for unit identity: the
        # weight dicts are already keyed by its labels (fast_scm_control /
        # lexsearch). The result contract serializes weight-dict keys as ``str``
        # (and ``UnitInfo.treated_labels`` is typed ``List[str]``), so canonicalize
        # the returned labels the SAME way -- otherwise selected_units /
        # assignment keep the raw label type (e.g. ``np.int64``) and fall out of
        # lock-step with the str-keyed ``treated_weights`` dict below.
        treated_labels = [str(k) for k in best.treated_weight_dict.keys()]
        control_labels = [str(k) for k in best.control_weight_dict.keys()]

        unit_info = UnitInfo(
            n_units_total=self.Y.shape[1],
            treated_labels=treated_labels,
            control_labels=control_labels
        )

        # =========================================================
        # Standardized post-fit + power analysis from the chosen design's gap
        # series. Mirrors the MAREX / SYNDES wiring so downstream consumers see
        # the same SyntheticControlPostFit surface across the entire family.
        # Computed before assembling the (frozen) result so it can feed both the
        # ``post_fit`` field and the contract-standard ``report``.
        # =========================================================
        pf = None
        try:
            from ..utils.post_fit import compute_post_fit, compute_power_analysis
            from dataclasses import replace as _dc_replace
            preds = best_candidate.predictions
            inf_obj = getattr(best_candidate, "inference", None)
            pf = compute_post_fit(
                treated_series=preds.synthetic_treated,
                control_series=preds.synthetic_control,
                n_fit=int(time_info.n_fit_time),
                n_blank=int(time_info.n_blank_time),
                n_post=int(time_info.n_post),
                treated_weights=best_candidate.weights.treated,
                control_weights=best_candidate.weights.control,
                inference=inf_obj,
                n_treated_units=int(np.sum(best_candidate.weights.treated > 1e-8)),
            )
            try:
                power = compute_power_analysis(pf, alpha=self.alpha)
                pf = _dc_replace(pf, power=power)
            except Exception:        # never let power analysis break a fit
                pass
        except Exception:            # never let post_fit assembly break a fit
            pf = None

        # =========================================================
        # Standardized two-family (design) result contract.
        # =========================================================
        design_donor_weights = {
            str(k): float(v) for k, v in best.control_weight_dict.items()
        }
        design_weights = WeightsResults(
            donor_weights=design_donor_weights,
            summary_stats={
                "n_treated": int(unit_info.treated_size),
                "n_control": int(unit_info.control_size),
                "treated_weights": {
                    str(k): float(v) for k, v in best.treated_weight_dict.items()
                },
            },
        )

        # The realized effect: one standardized EffectResult built by the shared
        # family adapter (no per-estimator field-copying). `report` is the single
        # source for ATT / CI / pre-fit; the rich SyntheticControlPostFit (with
        # per-period effects and covariate SMDs) rides along in
        # `report.additional_outputs['post_fit']`. `power` is the single source
        # for the MDE / power analysis.
        times = np.asarray(time_info.index.labels)
        intervention = (times[time_info.n_pre]
                        if time_info.n_pre < times.shape[0] else None)
        report = (
            post_fit_to_effect_result(
                pf, time_periods=times, intervention_time=intervention,
                method_name="LEXSCM", donor_weights=design_donor_weights)
            if pf is not None else None
        )
        rec = selection_results.get("recommendation", {})

        # =========================================================
        # FINAL RESULTS OBJECT -- a small, grouped surface.
        # =========================================================
        results = LEXSCMResults(
            # --- standardized contract front door ---
            report=report,
            power=(pf.power if pf is not None else None),
            selected_units=list(treated_labels),
            assignment={"treated": list(treated_labels),
                        "control": list(control_labels)},
            design_weights=design_weights,
            metadata={
                "status": rec.get("status"),
                "winner": rec.get("winner"),
                "pareto_ids": rec.get("pareto_ids"),
                "n_candidates": len(candidate_results),
                "outcome": self.outcome,
            },
            # --- grouped detail ---
            search=LEXSCMSearch(
                shortlist=shortlist,
                candidates=candidate_results,
                winner=best_candidate,
                selection=selection_results,
            ),
            panel=LEXSCMPanel(
                time=time_info,
                units=unit_info,
                outcome=self.outcome,
                population_mean=y_pop_mean_t,
            ),
        )

        if self.display_graph:
            lexplot(results)

        return results
