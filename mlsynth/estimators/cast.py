"""CAST: confidence intervals for treatment effects under staggered adoption.

Xia, E., Yan, Y., & Wainwright, M. J. (2025). *Inference under Staggered
Adoption: Case Study of the Affordable Care Act* (arXiv:2412.09482).

Most panel estimators report uncertainty for a *pooled* effect -- an ATT, or an
event-study path. CAST reports it for every treated unit-period. Sorting units
by how long they are treated turns a staggered design into a staircase, which
decomposes into four-block sub-problems; each is solved by a spectral routine
that estimates the factor subspaces from the observed blocks and regresses one
onto the other to impute the counterfactual. The same subspaces give a
closed-form entrywise variance, so each imputed cell carries a confidence
interval with asymptotic coverage, and any weighted aggregate of them (the
average effect on the treated, or a population-weighted total) inherits a
standard error.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import CASTConfig, InferenceResults
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.results_helpers import build_effect_submodels, make_weights_results
from ..utils.cast_helpers.pipeline import run_cast
from ..utils.cast_helpers.plotter import plot_cast
from ..utils.cast_helpers.setup import prepare_cast_inputs
from ..utils.cast_helpers.structures import CASTResults


class CAST:
    """Confidence intervals under staggered adoption (Xia et al. 2025).

    Parameters
    ----------
    config : CASTConfig or dict
        Configuration. See :class:`mlsynth.config_models.CASTConfig`.

    Returns
    -------
    CASTResults
        An ``EffectResult`` whose flat accessors summarize the treated average,
        plus the entrywise effect / standard-error / interval panels.
    """

    def __init__(self, config: Union[CASTConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = CASTConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid CAST configuration: {exc}") from exc
        self.config: CASTConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome, self.treat = config.outcome, config.treat
        self.unitid, self.time = config.unitid, config.time
        self.display_graphs: bool = getattr(config, "display_graphs", False)

    def fit(self) -> CASTResults:
        """Run the CAST pipeline end to end."""
        cfg = self.config
        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_cast_inputs(
                self.df, self.outcome, self.treat, self.unitid, self.time
            )
            fit = run_cast(inputs, cfg)

            # ----- treated-average surface for the standardized sub-models -----
            periods = list(fit.period_effects)
            labels = np.asarray(inputs.time_labels)
            treated = inputs.treated_mask
            ever = treated.any(axis=1)
            observed = inputs.Y[ever].mean(axis=0)
            counterfactual = fit.counterfactual[ever].mean(axis=0)
            n_post = len(periods)
            T0 = inputs.T - n_post

            att = float(np.mean([fit.period_effects[p][0] for p in periods]))
            att_se = float(np.sqrt(np.mean(
                [fit.period_effects[p][1] ** 2 for p in periods]))) / np.sqrt(max(n_post, 1))
            from scipy.stats import norm
            z = norm.ppf(1 - cfg.alpha / 2)
            inference = InferenceResults(
                standard_error=att_se,
                ci_lower=att - z * att_se,
                ci_upper=att + z * att_se,
                p_value=float(2 * norm.cdf(-abs(att) / att_se)) if att_se > 0 else None,
                confidence_level=1 - cfg.alpha,
                method="cast:entrywise",
            )

            # entrywise band on the treated-average counterfactual path
            band = np.full(inputs.T, np.nan)
            for j in range(inputs.T):
                rows = np.where(treated[:, j])[0]
                if rows.size:
                    band[j] = np.sqrt(np.mean(np.square(fit.std_errors[rows, j])))
            lower = np.where(np.isfinite(band), counterfactual - z * band, np.nan)
            upper = np.where(np.isfinite(band), counterfactual + z * band, np.nan)

            subs = build_effect_submodels(
                observed_outcome=observed,
                counterfactual_outcome=counterfactual,
                n_pre_periods=T0,
                n_post_periods=n_post,
                time_periods=labels,
                weights=make_weights_results(
                    {}, constraint="none (factor-model imputation, no donor weights)",
                    extra={"weights_are": "not_applicable",
                           "note": ("CAST imputes the counterfactual from estimated "
                                    "factor subspaces rather than a weighted average "
                                    "of donors, so it has no donor weights. The "
                                    "per-unit weights in the config apply to the "
                                    "aggregate effect, not to the imputation.")},
                ),
                inference=inference,
                method_name="CAST",
                effects_overrides={"att": att, "att_std_err": att_se},
                intervention_time=(labels[T0] if T0 < inputs.T else None),
                prediction_interval={"lower": lower, "upper": upper,
                                     "level": 1 - cfg.alpha,
                                     "kind": "cast:entrywise"},
            )

            results = CASTResults(
                inputs=inputs,
                rank=fit.rank,
                effects_panel=fit.effects,
                std_errors=fit.std_errors,
                ci_lower_panel=fit.ci_lower,
                ci_upper_panel=fit.ci_upper,
                counterfactual_panel=fit.counterfactual,
                period_effects=fit.period_effects,
                significance=fit.significance,
                metadata={
                    "rank": fit.rank, "alpha": cfg.alpha,
                    "n_treated_units": int(ever.sum()),
                    "n_treated_cells": int(treated.sum()),
                    **fit.extras,
                },
                **subs,
            )

            if self.display_graphs:
                try:
                    plot_cast(results)
                except Exception as exc:           # pragma: no cover
                    raise MlsynthPlottingError(f"CAST plotting failed: {exc}") from exc
            return results

        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError,
                MlsynthPlottingError):
            raise
        except Exception as exc:                   # pragma: no cover
            raise MlsynthEstimationError(f"CAST estimation failed: {exc}") from exc
