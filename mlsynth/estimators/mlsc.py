"""Multi-Level Synthetic Control (mlSC) estimator.

Implements:

    Bottmer, L. (2025). "Synthetic Control with Disaggregated Data."
    Stanford University job-market paper.

Reference Python package: ``multi-levelSC`` (Apache-2.0), reimplemented here
under mlsynth's MIT license following the SCDI / SPCD / LEXSCM / TASC
architectural conventions.

The mlSC estimator solves a hierarchical-aggregation regularized SC problem
on a *two-level* panel (e.g. state-level treated unit + county-level
controls). Treatment is assigned at the aggregate level; the disaggregate
data enters as an enlarged donor pool, with a ridge-type penalty that
shrinks each disaggregate weight ``omega_sc`` toward ``v_sc * w_s``
(population-share times implied aggregate weight). At lambda -> infinity
the estimator recovers classical SC; at lambda = 0 it recovers the fully-
disaggregated control SC (dGSC-AD in the paper's notation). v1 implements
the Section 5.2 heuristic ``lambda = 2 * sigma_eps^2 / sigma_y^2`` and a
user-supplied fixed lambda.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import MLSCConfig, PlotConfig, WeightsResults
from ..utils.results_helpers import build_effect_submodels
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.mlsc_helpers.crossval import select_lambda_cv
from ..utils.mlsc_helpers.inference import counterfactual_path, summarize_effects
from ..utils.mlsc_helpers.optimization import solve_mlsc
from ..utils.mlsc_helpers.penalty import build_penalty_matrix
from ..utils.mlsc_helpers.setup import prepare_mlsc_inputs
from ..utils.mlsc_helpers.structures import MLSCDesign, MLSCResults
from ..utils.mlsc_helpers.variance import (
    estimate_variance_components,
    heuristic_lambda,
)


class MLSC:
    """Multi-Level Synthetic Control (mlSC) estimator.

    Operates on two long-form DataFrames: one aggregate-level
    (``df_agg``, e.g. state by time) and one disaggregate-level
    (``df_disagg``, e.g. county by time with an ``agg_id`` column mapping
    each county to its state). Treatment is assigned at the aggregate level.
    The optimization weights all disaggregate control units to minimize
    pre-treatment fit on the aggregate treated series, with a hierarchical
    penalty that shrinks toward classical SC.

    Parameters
    ----------
    config : MLSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.MLSCConfig`.

    Returns
    -------
    MLSCResults
        Container with the optimal disaggregate weights, implied aggregate
        weights, counterfactual path, ATT, and pre-period RMSE.

    Notes
    -----
    Both DataFrames must share the same ``outcome``, ``time``, and ``treat``
    column names and must agree on treatment timing. The treated row in
    ``df_disagg`` must map (via ``agg_id``) to the treated aggregate unit
    in ``df_agg``.

    References
    ----------
    Bottmer, L. (2025). *Synthetic Control with Disaggregated Data.*
    Stanford University job-market paper.
    """

    def __init__(self, config: Union[MLSCConfig, dict]) -> None:
        """Initialize MLSC from an MLSCConfig instance or compatible dict."""

        if isinstance(config, dict):
            try:
                config = MLSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid MLSC configuration: {exc}"
                ) from exc

        self.config: MLSCConfig = config

        self.df_agg: pd.DataFrame = config.df_agg
        self.df_disagg: pd.DataFrame = config.df_disagg
        self.outcome: str = config.outcome
        self.time: str = config.time
        self.treat: str = config.treat
        self.unitid_agg: str = config.unitid_agg
        self.unitid_disagg: str = config.unitid_disagg
        self.agg_id: str = config.agg_id
        self.weight_col: Optional[str] = config.weight_col

        self.lambda_est: str = config.lambda_est
        self.lambda_val: float = config.lambda_val
        self.lambda_grid = config.lambda_grid
        self.cv_holdout_periods: int = config.cv_holdout_periods
        self.solver: Any = config.solver

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> MLSCResults:
        """Run the mlSC pipeline and return the learned design."""

        try:
            inputs = prepare_mlsc_inputs(
                df_agg=self.df_agg,
                df_disagg=self.df_disagg,
                outcome=self.outcome,
                time=self.time,
                treat=self.treat,
                unitid_agg=self.unitid_agg,
                unitid_disagg=self.unitid_disagg,
                agg_id=self.agg_id,
                weight_col=self.weight_col,
            )
        except (MlsynthConfigError, MlsynthDataError):
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"Error preparing mlSC inputs: {exc}"
            ) from exc

        try:
            sigma_eps2, sigma_y2 = estimate_variance_components(inputs)

            Q = build_penalty_matrix(
                v_population=inputs.v_population,
                disagg_to_agg=inputs.disagg_to_agg,
            )

            if self.lambda_est == "heuristic":
                lambda_used = heuristic_lambda(sigma_eps2, sigma_y2)
            elif self.lambda_est == "fixed":
                lambda_used = float(self.lambda_val)
            elif self.lambda_est == "cross-validation":
                lambda_used = select_lambda_cv(
                    inputs=inputs,
                    Q=Q,
                    sigma_y2=sigma_y2,
                    lambda_grid=self.lambda_grid,
                    cv_holdout_periods=self.cv_holdout_periods,
                    solver=self.solver,
                )
            else:  # defensive; pydantic Literal should have caught this
                raise MlsynthConfigError(
                    f"Unsupported lambda_est value: {self.lambda_est!r}."
                )

            omega, aggregate_weights, status = solve_mlsc(
                inputs=inputs,
                Q=Q,
                lambda_val=lambda_used,
                sigma_y2=sigma_y2,
                solver=self.solver,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"mlSC estimation failed: {exc}"
            ) from exc

        design = MLSCDesign(
            omega=omega,
            aggregate_weights=aggregate_weights,
            lambda_used=lambda_used,
            sigma_eps2=sigma_eps2,
            sigma_y2=sigma_y2,
            lambda_est=self.lambda_est,
            solver_status=status,
        )

        paths = counterfactual_path(inputs, omega)
        att, pre_rmse = summarize_effects(inputs, paths)

        donor_weights = {
            str(label): float(w)
            for label, w in zip(inputs.disagg_labels, omega)
        }
        aggregate_donor_weights = {
            str(label): float(w)
            for label, w in zip(inputs.agg_labels, aggregate_weights)
        }

        # Standardized sub-models built from the aggregate treated series:
        # observed = counterfactual + gap. mlSC has no statistical inference.
        observed = np.asarray(paths.counterfactual) + np.asarray(paths.gap)
        weights = WeightsResults(
            donor_weights=donor_weights,
            summary_stats={"aggregate_donor_weights": aggregate_donor_weights},
        )
        submodels = build_effect_submodels(
            observed_outcome=observed,
            counterfactual_outcome=np.asarray(paths.counterfactual),
            n_pre_periods=int(inputs.T0),
            n_post_periods=int(len(inputs.time_labels) - inputs.T0),
            time_periods=np.asarray(inputs.time_labels),
            weights=weights,
            method_name="MLSC",
            effects_overrides={"att": float(att)},
            fit_overrides={"rmse_pre": float(pre_rmse)},
            intervention_time=(inputs.time_labels[inputs.T0]
                               if inputs.T0 < len(inputs.time_labels)
                               else inputs.time_labels[-1]),
        )

        results = MLSCResults(
            **submodels,
            inputs=inputs,
            design=design,
            paths=paths,
            aggregate_donor_weights=aggregate_donor_weights,
        )

        # Standardized plotting (MLSCConfig is a plain BaseModel without
        # resolved_plot(), so build the PlotConfig from its legacy fields).
        cf_colors = ([self.counterfactual_color]
                     if isinstance(self.counterfactual_color, str)
                     else list(self.counterfactual_color))
        pc = PlotConfig(
            observed_color=self.treated_color,
            counterfactual_colors=cf_colors,
            xlabel=self.time,
            ylabel=self.outcome,
            display=self.display_graphs,
            save=self.save,
        )
        object.__setattr__(results, "plot_config", pc)
        if self.display_graphs:
            try:
                results.plot()
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"mlSC plotting failed: {exc}"
                ) from exc

        return results
