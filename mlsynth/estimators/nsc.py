"""Nonlinear Synthetic Control (NSC) estimator.

Tian, W. (2023). *"The Synthetic Control Method with Nonlinear
Outcomes: Estimating the Impact of the 2019 Anti-Extradition Law
Amendments Bill Protests on Hong Kong's Economy."* arXiv:2306.01967.

NSC generalises the canonical Abadie-Diamond-Hainmueller (2010)
synthetic-control method to nonlinear outcomes by

* dropping the non-negativity restriction on donor weights (only the
  adding-up constraint :math:`\\sum_j w_j = 1` remains);
* augmenting the weight-fitting objective with a pairwise-distance-
  weighted L1 penalty (favours donors close to the treated unit in
  pretreatment matching variables) and an L2 penalty (spreads the
  weights);
* scaling the tuning parameters by the eigenvalues of
  :math:`Z_0 Z_0'` so the dimensionless tuning parameters
  :math:`(a^*, b^*) \\in [0, 1]` admit a coarse cross-validation grid.

Inference defaults to the Doudchenko-Imbens (2017) variance
estimator: for each post-treatment period the variance of the gap
is estimated by the MSE of predicting each donor's outcome from the
others under the same ``(a^*, b^*)`` regime.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import NSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.nsc_helpers.crossval import cv_select
from ..utils.nsc_helpers.inference import doudchenko_imbens_inference
from ..utils.nsc_helpers.optimization import (
    design_eigenvalues,
    fit_nsc,
)
from ..utils.nsc_helpers.plotter import plot_nsc
from ..utils.nsc_helpers.setup import prepare_nsc_inputs
from ..utils.nsc_helpers.structures import (
    NSCDesign,
    NSCInference,
    NSCResults,
)


class NSC:
    """Nonlinear Synthetic Control (Tian 2023) estimator.

    Parameters
    ----------
    config : NSCConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.NSCConfig`.

    Returns
    -------
    NSCResults
        Frozen container with the optimised weights, the coordinate-
        descent CV trace (if any), Doudchenko-Imbens inference, the
        counterfactual series, and the ATT.
    """

    def __init__(self, config: Union[NSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = NSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid NSC configuration: {exc}"
                ) from exc

        self.config: NSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.a = config.a
        self.b = config.b
        self.cv_grid_size: float = config.cv_grid_size
        self.cv_target: str = config.cv_target
        self.cv_max_iterations: int = config.cv_max_iterations
        self.covariates = config.covariates
        self.alpha: float = config.alpha
        self.run_inference: bool = config.run_inference
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> NSCResults:
        """Run the NSC pipeline end to end."""
        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_nsc_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                covariates=self.covariates,
            )

            # ----- (a*, b*) selection -----
            if self.a is None or self.b is None:
                a_star, b_star, trace = cv_select(
                    Z1=inputs.treated_matching_vector,
                    Z0=inputs.matching_matrix,
                    grid_size=self.cv_grid_size,
                    max_iterations=self.cv_max_iterations,
                    target=self.cv_target,
                )
            else:
                a_star = float(self.a)
                b_star = float(self.b)
                trace = None

            # ----- Final NSC fit at (a*, b*) -----
            eigvals = design_eigenvalues(inputs.matching_matrix)
            w, a_scaled, b_scaled = fit_nsc(
                Z1=inputs.treated_matching_vector,
                Z0=inputs.matching_matrix,
                a_star=a_star,
                b_star=b_star,
                eigvals=eigvals,
            )
            donor_weights = {
                str(inputs.donor_names[i]): float(round(w[i], 6))
                for i in range(inputs.J)
            }
            design = NSCDesign(
                w=w,
                donor_weights=donor_weights,
                a_star=float(a_star),
                b_star=float(b_star),
                a_scaled=float(a_scaled),
                b_scaled=float(b_scaled),
                eigvals=eigvals,
            )

            counterfactual = inputs.donor_outcomes @ w
            gap = inputs.treated_outcome - counterfactual
            pre_rmse = float(np.sqrt(np.mean(gap[:inputs.T0] ** 2)))
            att = (
                float(np.mean(gap[inputs.T0:]))
                if inputs.n_post > 0 else float("nan")
            )

            # ----- Inference -----
            if self.run_inference and inputs.n_post > 0:
                inference = doudchenko_imbens_inference(
                    treated_outcome=inputs.treated_outcome,
                    donor_outcomes=inputs.donor_outcomes,
                    counterfactual=counterfactual,
                    Z0=inputs.matching_matrix,
                    T0=inputs.T0,
                    a_star=a_star,
                    b_star=b_star,
                    alpha=self.alpha,
                )
            else:
                inference = NSCInference(
                    method="none",
                    alpha=float(self.alpha),
                    gap=gap,
                )

            results = NSCResults(
                inputs=inputs,
                design=design,
                cv_trace=trace,
                inference=inference,
                counterfactual=counterfactual,
                gap=gap,
                att=att,
                pre_rmse=pre_rmse,
                metadata={
                    "cv_converged": (trace.converged if trace else None),
                    "cv_iterations": (trace.iterations if trace else 0),
                    "cv_target": (trace.target if trace else "user"),
                    "matching_dim": int(inputs.matching_matrix.shape[1]),
                    "n_eigvals": int(eigvals.size),
                },
            )

            if self.display_graphs:
                try:
                    plot_nsc(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"NSC plotting failed: {exc}"
                    ) from exc

            return results

        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
            MlsynthPlottingError,
        ):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"NSC estimation failed: {exc}"
            ) from exc
