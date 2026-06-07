"""Sparse Synthetic Control (SparseSC) estimator.

Implements the L1-penalized predictor-weighting SCM of
Vives-i-Bastida (2023, *Predictor Selection for Synthetic Controls*),
applied to the canonical Abadie, Diamond, and Hainmueller (2010)
framework.

The estimator has a two-level structure. The *inner* problem is the
standard SCM simplex QP that picks donor weights ``w`` given a fixed
diagonal predictor-importance matrix ``diag(v)``. The *outer* problem
picks the V-weights themselves by minimizing the validation-block
pre-treatment outcome MSE plus an L1 penalty on ``|v|`` (matching
Algorithm 1 of the paper). The penalty parameter is selected by the
unpenalized validation MSE. The first V-weight is pinned to 1 to
anchor the scale; the others are bound-constrained non-negative.

Compared with canonical SCM, the L1 penalty yields interpretable
*predictor selection*: as ``lambda`` increases, V-weights collapse to
zero on uninformative predictors, leaving a sparse explanation of the
fit.

The unpublished MATLAB driver ``sparse_synth.m`` minimizes the
outcome MSE on the *training* block (not the validation block) in the
outer V step. That behavior is available via
``outer_loss_window="training"``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import InferenceResults, SparseSCConfig, WeightsResults
from ..utils.results_helpers import build_effect_submodels
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.sparse_sc_helpers.inference import conformal_inference, run_placebo
from ..utils.sparse_sc_helpers.optimization import recover_w, sweep_lambda
from ..utils.sparse_sc_helpers.setup import prepare_sparse_sc_inputs
from ..utils.sparse_sc_helpers.structures import (
    SparseSCDesign,
    SparseSCInference,
    SparseSCResults,
)


class SparseSC:
    """L1-penalized Sparse Synthetic Control estimator.

    Parameters
    ----------
    config : SparseSCConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.SparseSCConfig`.

    Returns
    -------
    SparseSCResults
        Typed container with the selected V- and W-weights, the
        validation-MSE curve over the lambda grid, the counterfactual,
        and (optionally) Abadie placebo inference.

    Notes
    -----
    Predictors are supplied through ``covariates`` (columns in ``df``
    whose per-unit pre-treatment mean becomes one predictor row) and/or
    ``outcome_lag_periods`` (specific pre-treatment time labels whose
    outcome values become predictor rows -- the canonical ADH lagged-
    outcome predictors). The first predictor is the "anchor" whose
    V-weight is fixed at 1.

    Examples
    --------
    >>> import pandas as pd                                                  # doctest: +SKIP
    >>> from mlsynth import SparseSC                                         # doctest: +SKIP
    >>> df = pd.read_csv("smoking_long.csv")                                 # doctest: +SKIP
    >>> res = SparseSC({                                                     # doctest: +SKIP
    ...     "df": df, "outcome": "cigsale",
    ...     "treat": "Proposition 99", "unitid": "state", "time": "year",
    ...     "covariates": ["p_cig", "loginc", "pct15-24", "pc_beer"],
    ...     "outcome_lag_periods": [1975, 1980, 1988],
    ...     "display_graphs": False,
    ... }).fit()
    >>> res.att                                                              # doctest: +SKIP
    -19.5...
    """

    def __init__(self, config: Union[SparseSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SparseSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SparseSC configuration: {exc}"
                ) from exc

        self.config: SparseSCConfig = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.covariates = config.covariates
        self.outcome_lag_periods = config.outcome_lag_periods
        self.T0_train = config.T0_train
        self.lambda_grid = (
            np.asarray(config.lambda_grid, dtype=float)
            if config.lambda_grid is not None else None
        )
        self.standardize: bool = config.standardize
        self.outer_loss_window: str = config.outer_loss_window
        self.solver: Any = config.solver
        self.max_outer_iter: int = config.max_outer_iter
        self.use_analytical_grad: bool = config.use_analytical_grad
        self.warm_start: bool = config.warm_start
        self.run_inference: bool = config.run_inference
        self.inference_method: str = config.inference_method
        self.conformal_window: str = config.conformal_window
        self.alpha: float = config.alpha
        self.n_placebo = config.n_placebo
        self.placebo_resweep: bool = config.placebo_resweep
        self.seed: int = config.seed

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SparseSCResults:
        """Run the lambda sweep, recover W-weights, and return results."""
        try:
            inputs = prepare_sparse_sc_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
                covariates=self.covariates,
                outcome_lag_periods=self.outcome_lag_periods,
                T0_train=self.T0_train,
                standardize=self.standardize,
            )

            optv, opt_lambda, grid, train_curve, val_curve, v_path = sweep_lambda(
                X1=inputs.X1, X0=inputs.X0,
                Y1=inputs.Y1, Y0=inputs.Y0,
                T0_total=inputs.T0_total, T0_train=inputs.T0_train,
                lambda_grid=self.lambda_grid,
                solver=self.solver,
                max_outer_iter=self.max_outer_iter,
                outer_loss_window=self.outer_loss_window,
                use_analytical_grad=self.use_analytical_grad,
                warm_start=self.warm_start,
            )
            optw = recover_w(optv, inputs.X1, inputs.X0, solver=self.solver)

            cf = inputs.Y0 @ optw
            gap = inputs.Y1 - cf
            T0 = inputs.T0_total
            att = float(np.mean(gap[T0:])) if inputs.T > T0 else float("nan")
            pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))

            design = SparseSCDesign(
                v=optv, w=optw,
                opt_lambda=float(opt_lambda),
                lambda_grid=grid,
                train_loss_curve=train_curve,
                val_mse_curve=val_curve,
                v_path=v_path,
            )

            if not self.run_inference or self.inference_method == "none":
                inference = SparseSCInference(
                    method="none",
                    p_value=float("nan"),
                    att_observed=att,
                    alpha=float(self.alpha),
                )
            elif self.inference_method == "placebo":
                placebo_atts, p_val, n_done = run_placebo(
                    Y0=inputs.Y0, Y1=inputs.Y1,
                    X0=inputs.X0, X1=inputs.X1,
                    T0_total=inputs.T0_total, T0_train=inputs.T0_train,
                    selected_lambda=float(opt_lambda),
                    observed_att=att,
                    solver=self.solver,
                    resweep=self.placebo_resweep,
                    lambda_grid=self.lambda_grid,
                    n_placebo=self.n_placebo,
                    seed=self.seed,
                    outer_loss_window=self.outer_loss_window,
                )
                inference = SparseSCInference(
                    method="abadie_placebo_permutation",
                    p_value=p_val,
                    att_observed=att,
                    alpha=float(self.alpha),
                    placebo_atts=placebo_atts,
                    n_placebo=n_done,
                )
            elif self.inference_method == "conformal":
                conf = conformal_inference(
                    gap=gap,
                    T0_train=inputs.T0_train,
                    T0_total=inputs.T0_total,
                    T=inputs.T,
                    conformal_window=self.conformal_window,
                    alpha=self.alpha,
                )
                inference = SparseSCInference(
                    method=conf["method"],
                    p_value=conf["p_value"],
                    att_observed=conf["att_observed"],
                    ci_lower=conf["ci_lower"],
                    ci_upper=conf["ci_upper"],
                    alpha=conf["alpha"],
                    calibration_residuals=conf["calibration_residuals"],
                    pointwise_lower=conf["pointwise_lower"],
                    pointwise_upper=conf["pointwise_upper"],
                )
            else:
                raise MlsynthConfigError(
                    f"Unknown inference_method={self.inference_method!r}; "
                    "expected 'conformal', 'placebo', or 'none'."
                )

            donor_weights: Dict[Any, float] = {
                str(n): float(w) for n, w in zip(inputs.donor_names, optw)
            }
            predictor_weights: Dict[Any, float] = {
                str(p): float(v) for p, v in zip(inputs.predictor_names, optv)
            }

            # Standardized inference mirrored from the raw placebo/conformal
            # object (NaN -> None); the contract slot holds InferenceResults.
            def _f(x):
                return None if x is None or np.isnan(x) else float(x)

            std_inference = None
            if inference.method != "none":
                std_inference = InferenceResults(
                    method=inference.method,
                    p_value=_f(inference.p_value),
                    ci_lower=_f(inference.ci_lower),
                    ci_upper=_f(inference.ci_upper),
                    confidence_level=(None if np.isnan(inference.alpha)
                                      else float(1.0 - inference.alpha)),
                    details=inference,
                )
            weights = WeightsResults(
                donor_weights=donor_weights,
                summary_stats={"predictor_weights": predictor_weights},
            )
            submodels = build_effect_submodels(
                observed_outcome=np.asarray(inputs.Y1),
                counterfactual_outcome=np.asarray(cf),
                n_pre_periods=int(inputs.T0_total),
                n_post_periods=int(inputs.T - inputs.T0_total),
                time_periods=np.asarray(inputs.time_labels),
                weights=weights,
                inference=std_inference,
                method_name="SparseSC",
                effects_overrides={"att": float(att)},
                fit_overrides={"rmse_pre": float(pre_rmse)},
                intervention_time=(inputs.time_labels[inputs.T0_total]
                                   if inputs.T0_total < inputs.T
                                   else inputs.time_labels[-1]),
            )
            results = SparseSCResults(
                **submodels,
                inputs=inputs, design=design, inference_detail=inference,
                predictor_weights=predictor_weights,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SparseSC estimation failed: {exc}"
            ) from exc

        # Standardized plotting: attach the resolved PlotConfig and route
        # through result.plot().
        pc = self.config.resolved_plot()
        if pc.xlabel is None:
            pc.xlabel = self.time
        if pc.ylabel is None:
            pc.ylabel = self.outcome
        object.__setattr__(results, "plot_config", pc)
        if self.display_graphs:
            try:
                results.plot()
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"SparseSC plotting failed: {exc}"
                ) from exc

        return results
