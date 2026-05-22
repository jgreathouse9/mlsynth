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
from ..utils.microsynth_helpers.dual_solver import solve_microsynth_dual
from ..utils.microsynth_helpers.inference import paired_bootstrap_ci
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
        self.standardize_covariates: bool = config.standardize_covariates
        self.balance_tol: float = config.balance_tol
        self.max_iter: int = config.max_iter
        self.gtol: float = config.gtol
        self.run_inference: bool = config.run_inference
        self.n_bootstrap: int = config.n_bootstrap
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
            )

            xbar_T = inputs.X_T.mean(axis=0)
            dual = solve_microsynth_dual(
                inputs.X_C, xbar_T,
                max_iter=self.max_iter, gtol=self.gtol,
            )

            smd_before = standardized_mean_difference(inputs.X_T, inputs.X_C)
            smd_after = standardized_mean_difference(
                inputs.X_T, inputs.X_C, w=dual.w
            )
            ess = effective_sample_size(dual.w)
            mw = max_weight(dual.w)
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

            # Counterfactual + gap per post-period.
            if inputs.Y_T.ndim == 1:
                cf = float(dual.w @ inputs.Y_C)
                cf_arr = np.asarray(cf)
                gap = float(inputs.Y_T.mean() - cf)
                gap_arr = np.asarray(gap)
                gap_traj = np.asarray([gap], dtype=float)
                att = float(gap)
            else:
                cf_arr = dual.w @ inputs.Y_C  # shape (T_post,)
                gap_arr = inputs.Y_T.mean(axis=0) - cf_arr
                gap_traj = gap_arr.copy()
                att = float(gap_traj.mean())

            if self.run_inference:
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
