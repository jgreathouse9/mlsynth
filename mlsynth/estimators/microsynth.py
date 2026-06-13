"""MicroSynth estimator (Robbins-Davenport 2021).

User-level balancing synthetic control. Solves a constrained QP for
non-negative simplex weights on the control population that exactly
balance covariate moments against the treated group's moments, then
reads off the ATT as the weighted-mean outcome difference. Scales
to ``N_C`` in the millions on a single machine because the dual
optimization is in ``R^{d+1}`` regardless of ``N_C``.

See :class:`mlsynth.config_models.MicroSynthConfig` for the public
configuration. Helpers live in
:mod:`mlsynth.utils.microsynth_helpers`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import MicroSynthConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.microsynth_helpers.diagnostics import (
    effective_sample_size,
    feasibility_check,
    max_weight,
    standardized_mean_difference,
)
from ..utils.microsynth_helpers.dual_solver import (
    DualSolverResult,
    solve_microsynth_dual,
)
from ..utils.microsynth_helpers.inference import paired_bootstrap_ci
from ..utils.microsynth_helpers.panel_inference import panel_permutation_test
from ..utils.microsynth_helpers.panel_qp import solve_panel_qp
from ..utils.microsynth_helpers.plotter import plot_microsynth
from ..utils.microsynth_helpers.setup import prepare_microsynth_inputs
from ..utils.microsynth_helpers.structures import (
    MicroSynthDesign,
    MicroSynthInference,
    MicroSynthResults,
)


class MicroSynth:
    """User-level balancing synthetic control estimator.

    Parameters
    ----------
    config : MicroSynthConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.MicroSynthConfig`.

    Returns
    -------
    MicroSynthResults
        Typed container with the dual-ascent weights, balance
        diagnostics, counterfactual trajectory, ATT, and (optionally)
        a paired stratified bootstrap CI.

    Notes
    -----
    Unlike aggregate-unit estimators in :mod:`mlsynth` (FDID, SDID,
    PPSCM, SparseSC, etc.), MicroSynth treats individual users as
    units. There can be thousands of treated users; the control
    "donor pool" is the entire untreated population. Covariate
    moments listed in ``covariates`` -- and optionally
    pre-treatment outcome values listed in ``outcome_lag_periods`` --
    are exactly balanced between treated and weighted controls by
    a quadratic program.

    The identifying assumption is selection-on-observables: given
    the covariate set, treatment exposure is independent of
    potential outcomes. Marketing applications typically need
    covariates that include audience-segment / persona membership,
    device, geo, prior engagement, and frequency exposure to
    parallel campaigns; missing any of those that influence both
    exposure and the outcome introduces residual bias.

    Examples
    --------
    >>> import pandas as pd                                                  # doctest: +SKIP
    >>> from mlsynth import MicroSynth                                       # doctest: +SKIP
    >>> df = pd.read_csv("user_panel.csv")                                   # doctest: +SKIP
    >>> res = MicroSynth({                                                   # doctest: +SKIP
    ...     "df": df, "outcome": "converted",
    ...     "treat": "saw_ad", "unitid": "user_id", "time": "week",
    ...     "covariates": ["age", "device", "prior_engagement",
    ...                    "country_tier", "gender"],
    ...     "display_graphs": False,
    ... }).fit()
    >>> res.att                                                              # doctest: +SKIP
    0.052
    """

    def __init__(self, config: Union[MicroSynthConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MicroSynthConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid MicroSynth configuration: {exc}"
                ) from exc

        self.config: MicroSynthConfig = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.covariates: List[str] = config.covariates
        self.outcome_lag_periods = config.outcome_lag_periods
        self.match_outcomes = config.match_outcomes
        self.standardize_covariates: bool = config.standardize_covariates
        self.weight_method: str = config.weight_method
        self.panel_ridge: float = config.panel_ridge
        self.propensity_mode: bool = config.propensity_mode
        self.balance_tol: float = config.balance_tol
        self.max_iter: int = config.max_iter
        self.gtol: float = config.gtol
        self.run_inference: bool = config.run_inference
        self.n_bootstrap: int = config.n_bootstrap
        self.n_permutations: int = config.n_permutations
        self.permutation_test: str = config.permutation_test
        self.seed: int = config.seed

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> MicroSynthResults:
        """Run the dual-ascent fit and (optionally) bootstrap CI."""
        try:
            inputs = prepare_microsynth_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                covariates=self.covariates,
                outcome_lag_periods=self.outcome_lag_periods,
                standardize=self.standardize_covariates,
                match_outcomes=self.match_outcomes,
            )

            # Propensity-score mode (microsynth match.out=FALSE) is a panel QP
            # on covariates only; it implies the panel weighting.
            is_panel = self.weight_method == "panel" or self.propensity_mode
            if is_panel:
                # microsynth panel method (Robbins et al.): a non-negative QP on
                # TOTALS. Hard block = intercept + raw covariates (exact balance,
                # weights sum to the treated count); soft block = each pre-period
                # outcome (least-squares fit). A ridge pins the unique optimum.
                n_T = inputs.n_T
                n_C = inputs.n_C
                hard_C = np.column_stack([np.ones(n_C), inputs.cov_C_raw])
                hard_targets = np.concatenate(
                    [[float(n_T)], inputs.cov_T_raw.sum(axis=0)]
                )
                # Propensity mode ignores lagged outcomes (match.out=FALSE).
                use_lags = inputs.lag_C_raw.shape[1] and not self.propensity_mode
                soft_C = inputs.lag_C_raw if use_lags else None
                soft_targets = (
                    inputs.lag_T_raw.sum(axis=0) if soft_C is not None else None
                )
                panel = solve_panel_qp(
                    hard_C, hard_targets, soft_C, soft_targets,
                    ridge=self.panel_ridge,
                )
                # Reuse the dual-result container fields the design expects.
                dual = DualSolverResult(
                    w=panel.w, dual_lambda=np.zeros(hard_C.shape[1]), dual_nu=0.0,
                    n_iterations=0, converged=panel.converged,
                )
            else:
                xbar_T = inputs.X_T.mean(axis=0)
                dual = solve_microsynth_dual(
                    inputs.X_C, xbar_T,
                    max_iter=self.max_iter, gtol=self.gtol,
                )

            # Panel weights sum to the treated count; the balance diagnostics
            # (mean comparison / ESS) are defined for unit-sum weights, so feed
            # them the normalized weights while keeping the raw QP weights for
            # the total-scale per-period effects below.
            w_diag = dual.w / dual.w.sum() if is_panel else dual.w
            smd_before = standardized_mean_difference(inputs.X_T, inputs.X_C)
            smd_after = standardized_mean_difference(
                inputs.X_T, inputs.X_C, w=w_diag
            )
            ess = effective_sample_size(w_diag)
            mw = max_weight(w_diag)
            feasible, feasibility_msg = feasibility_check(
                smd_after, self.balance_tol
            )

            design = MicroSynthDesign(
                w=dual.w,
                dual_lambda=dual.dual_lambda,
                dual_nu=dual.dual_nu,
                smd_before=smd_before,
                smd_after=smd_after,
                ess=ess,
                max_weight=mw,
                feasible=feasible,
                feasibility_message=feasibility_msg,
                n_iterations=dual.n_iterations,
                converged=dual.converged,
            )

            # Counterfactual + gap per post-period. For the panel method the
            # contrast is on TOTALS: treated-area total minus the weighted
            # control total (weights sum to the treated count). For simplex it
            # is the per-treated-unit weighted-mean contrast.
            treated_agg = inputs.Y_T.sum if is_panel else inputs.Y_T.mean
            if inputs.Y_T.ndim == 1:
                cf = float(dual.w @ inputs.Y_C)
                cf_arr = np.asarray(cf)
                gap = float(treated_agg() - cf)
                gap_arr = np.asarray(gap)
                gap_traj = np.asarray([gap], dtype=float)
                att = float(gap)
            else:
                cf_arr = dual.w @ inputs.Y_C  # shape (T_post,)
                gap_arr = treated_agg(axis=0) - cf_arr
                gap_traj = gap_arr.copy()
                att = float(gap_traj.mean())

            if self.run_inference and is_panel and self.n_permutations > 0:
                # Panel method: placebo-permutation inference (microsynth perm).
                Y_C_post = (
                    inputs.Y_C.reshape(-1, 1)
                    if inputs.Y_C.ndim == 1 else inputs.Y_C
                )
                perm = panel_permutation_test(
                    cov_C=inputs.cov_C_raw,
                    lag_C=soft_C,
                    Y_C_post=Y_C_post,
                    n_T=inputs.n_T,
                    obs_gap_trajectory=gap_traj,
                    obs_att=att,
                    ridge=self.panel_ridge,
                    n_perm=self.n_permutations,
                    test=self.permutation_test,
                    seed=self.seed,
                )
                inference = MicroSynthInference(
                    method="permutation",
                    att=att,
                    se=perm.se,
                    ci=perm.ci,
                    n_bootstrap=perm.n_perm,
                    bootstrap_atts=perm.placebo_atts,
                    p_value=perm.p_value,
                    p_values_by_period=perm.p_values_by_period,
                    test=perm.test,
                )
            elif self.run_inference and not is_panel:
                se, ci, boot_atts, n_complete = paired_bootstrap_ci(
                    X_T=inputs.X_T, X_C=inputs.X_C,
                    Y_T=inputs.Y_T, Y_C=inputs.Y_C,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                    max_iter=self.max_iter, gtol=self.gtol,
                )
                inference = MicroSynthInference(
                    method="paired_bootstrap",
                    att=att,
                    se=se,
                    ci=ci,
                    n_bootstrap=n_complete,
                    bootstrap_atts=boot_atts,
                )
            else:
                inference = MicroSynthInference(
                    method="none",
                    att=att,
                    se=float("nan"),
                    ci=np.asarray([float("nan"), float("nan")]),
                    n_bootstrap=0,
                    bootstrap_atts=np.asarray([]),
                )

            donor_weights: Dict[Any, float] = {
                str(name): float(w_i)
                for name, w_i in zip(inputs.control_unit_names, dual.w)
                if w_i > 0
            }

            results = MicroSynthResults(
                inputs=inputs,
                design=design,
                inference=inference,
                counterfactual=cf_arr,
                gap=gap_arr,
                gap_trajectory=gap_traj,
                att=att,
                donor_weights=donor_weights,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"MicroSynth estimation failed: {exc}"
            ) from exc

        if self.display_graphs:
            try:
                plot_microsynth(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=self.counterfactual_color,
                    save=self.save,
                )
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"MicroSynth plotting failed: {exc}"
                ) from exc

        return results
