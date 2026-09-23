"""Generalized synthetic control (GSYNTH) estimator.

Xu, Y. (2017). *"Generalized Synthetic Control Method: Causal Inference with
Interactive Fixed Effects Models."* Political Analysis 25(1):57-76.

GSYNTH imputes what treated units would have done by fitting a factor model to
the units that are never treated, then placing each treated unit in that
estimated factor space using its own pre-treatment history:

1. Fit an interactive fixed effects model on the never-treated units alone,
   obtaining the coefficient vector, the common factors and the additive
   effects.
2. Recover each treated unit's factor loadings by regressing its pre-treatment
   residuals on those factors.
3. Impute the untreated outcome from the two and difference against what was
   observed.

The factor count comes from the paper's Algorithm 1, a leave-one-period-out
cross-validation over the treated units' pre-treatment periods, and the
uncertainty from its Algorithm 2 parametric bootstrap.

The estimator handles staggered, absorbing adoption: each treated unit's
loadings are fit on its own pre-period block, so units adopting at different
dates contribute the histories they have.

This is not the estimator :class:`~mlsynth.CTSC` implements. That one is Cao,
Lu and Wu's generalized synthetic control, differently constructed; the name
collision is why this class is exported as ``GSYNTH``.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import GSYNTHConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.gsynth_helpers.inference import parametric_bootstrap
from ..utils.gsynth_helpers.pipeline import cross_validate_rank, gsc_fit
from ..utils.gsynth_helpers.plotter import plot_gsynth
from ..utils.gsynth_helpers.setup import prepare_gsynth_inputs
from ..utils.gsynth_helpers.structures import (
    GSYNTHAggregate,
    GSYNTHDesign,
    GSYNTHInference,
    GSYNTHResults,
    GSYNTHUnitFit,
    event_study_from_effect,
)


class GSYNTH:
    """Generalized synthetic control (Xu 2017) estimator.

    Parameters
    ----------
    config : GSYNTHConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.GSYNTHConfig`.

    Returns
    -------
    GSYNTHResults
        Frozen container with the fitted factor structure, the effect by period
        since adoption, the per-unit breakdown, and the parametric bootstrap.
    """

    def __init__(self, config: Union[GSYNTHConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = GSYNTHConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid GSYNTH configuration: {exc}"
                ) from exc

        self.config: GSYNTHConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.covariates = config.covariates
        self.r = config.r
        self.r_max: int = config.r_max
        self.force: str = config.force
        self.run_inference: bool = config.inference
        self.n_bootstrap: int = config.n_bootstrap
        self.alpha: float = config.alpha
        self.seed: int = config.seed
        self.tol: float = config.tol
        self.max_iter: int = config.max_iter
        self.display_graphs: bool = getattr(config, "display_graphs", False)

    def fit(self) -> GSYNTHResults:
        """Run the GSYNTH pipeline end to end."""
        try:
            inputs = prepare_gsynth_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time, covariates=self.covariates,
            )

            # ----- Rank -----
            if self.r is None:
                cv = cross_validate_rank(
                    inputs, r_max=self.r_max, force=self.force,
                    tol=self.tol, max_iter=self.max_iter)
                r, source = cv.r_selected, "cv"
            else:
                cv, r, source = None, int(self.r), "user"

            # ----- Steps 1-3 -----
            fit = gsc_fit(inputs, r, force=self.force, tol=self.tol,
                          max_iter=self.max_iter)
            control = fit.control_fit
            es = event_study_from_effect(fit.effect, inputs.adoption_index)

            # ----- Algorithm 2 -----
            if self.run_inference:
                inference = parametric_bootstrap(
                    inputs, fit, r, n_bootstrap=self.n_bootstrap,
                    alpha=self.alpha, seed=self.seed, force=self.force,
                    tol=self.tol, max_iter=self.max_iter)
            else:
                inference = GSYNTHInference(
                    n_bootstrap=0, alpha=float(self.alpha), att=float(fit.att),
                    tau=np.asarray(es.tau, dtype=float),
                    horizons=np.asarray(es.horizons, dtype=int))

            names = inputs.covariate_names
            design = GSYNTHDesign(
                r_selected=int(r),
                r_source=source,
                force=self.force,
                beta=np.asarray(control.beta, dtype=float),
                covariate_names=names,
                dropped_covariates=tuple(names[k] for k in control.dropped),
                factors=np.asarray(control.factor, dtype=float),
                control_loadings=np.asarray(control.lam, dtype=float),
                treated_loadings=fit.loadings[:, :r],
                unit_effects=np.asarray(fit.unit_effects, dtype=float),
                period_effects=np.asarray(control.xi, dtype=float),
                grand_mean=float(control.mu),
                n_iterations=int(control.niter),
                cv=cv,
            )

            # ----- Per-unit breakdown -----
            per_unit = {}
            pre_sq, pre_n = 0.0, 0
            for j, col in enumerate(inputs.treated_index):
                t0 = int(inputs.adoption_index[j])
                label = inputs.unit_labels[col]
                pre = fit.effect[:t0, j]
                pre_sq += float(np.sum(pre ** 2))
                pre_n += int(pre.size)
                per_unit[label] = GSYNTHUnitFit(
                    label=label,
                    adoption_time=inputs.time_labels[t0],
                    n_pre=t0,
                    n_post=inputs.T - t0,
                    att=float(fit.effect[t0:, j].mean()),
                    prefit_rmse=float(np.sqrt(np.mean(pre ** 2))),
                    tau=np.asarray(fit.effect[:, j], dtype=float),
                    loading=np.asarray(fit.loadings[j, :r], dtype=float),
                    unit_effect=float(fit.unit_effects[j]),
                )

            aggregate = GSYNTHAggregate(
                att=float(fit.att),
                n_post_cells=int(fit.n_post_cells),
                rmse_pre=float(np.sqrt(pre_sq / pre_n)),
                observed=inputs.Y[:, inputs.treated_index].mean(axis=1),
                counterfactual=fit.counterfactual.mean(axis=1),
                intervention_time=inputs.time_labels[
                    int(inputs.adoption_index.min())],
            )

            results = GSYNTHResults(
                inputs=inputs,
                design=design,
                aggregate=aggregate,
                event_study=es,
                inference_detail=inference,
                per_unit=per_unit,
                metadata={
                    "r": int(r),
                    "r_source": source,
                    "n_iterations": int(control.niter),
                    "dropped_covariates": list(design.dropped_covariates),
                    "n_bootstrap_failed": int(inference.n_failed),
                },
            )

            if self.display_graphs:
                try:
                    plot_gsynth(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"GSYNTH plotting failed: {exc}"
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
                f"GSYNTH estimation failed: {exc}") from exc
