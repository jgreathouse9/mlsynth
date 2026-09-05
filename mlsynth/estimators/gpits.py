"""Gaussian-process interrupted time series (GPITS).

Implements:

    Cho, S. (2026). "Let Time Tell: Identification and Gaussian Process
    Estimation for Interrupted Time Series." arXiv:2608.20610.

Some interventions reach every unit at once -- a Supreme Court ruling, a
national policy, a pandemic. Difference-in-differences and synthetic control
both need somebody who stays untreated, and here nobody does, so the
counterfactual has to come from the treated unit's own past. GPITS learns the
pre-treatment relationship between the outcome and observable inputs (calendar
time and any covariates) as a Gaussian process, then evaluates it after the
intervention. The gap between what happened and that prediction is the effect.

What the Gaussian process adds over the usual segmented regression is honest
uncertainty about the extrapolation. Many trends fit a pre-period equally well
and disagree afterwards, by more the further out you look. A kernel carries all
of them at once instead of committing to one, so the posterior band widens with
distance from the data (Section 4). The cost is that the bands are conservative
by design: they are calibrated to the worst case the kernel admits, so a small
effect can sit inside them.

Identification rests on Mean Sufficiency (Assumption 1): once the observables
are known, unobservables do not shift the expected untreated outcome. That is
the temporal counterpart of selection-on-observables, and it fails when
something other than the intervention changes the trajectory -- so shorter
post-treatment windows are more credible, and the placebo checks of Section 3.3
are there to falsify it.

The estimator is a thin orchestration layer over
:mod:`mlsynth.utils.gpits_helpers`:

    setup.py      : long DataFrame -> GPITSInputs (via dataprep)
    pipeline.py   : the GP, the effects, the placebo checks
    plotter.py    : observed vs counterfactual
    structures.py : frozen result containers
"""

from __future__ import annotations

import warnings
from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import GPITSConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.gpits_helpers.pipeline import fit_gpits, run_placebo, summarize_effects
from ..utils.gpits_helpers.plotter import plot_gpits
from ..utils.gpits_helpers.setup import prepare_gpits_inputs
from ..utils.gpits_helpers.structures import GPITSResults


class GPITS:
    """Gaussian-process interrupted time series (Cho 2026).

    Estimates a treated unit's untreated counterfactual from its own
    pre-treatment history, with no donor pool, and reports a band that widens
    as the forecast extends past the data. Use it when the treatment reached
    everyone at once so no comparison unit survives.

    Parameters
    ----------
    config : GPITSConfig or dict
        Configuration object. See
        :class:`mlsynth.utils.gpits_helpers.config.GPITSConfig`. Key fields:
        ``kernel`` and ``period`` (the covariance structure), ``covariates`` /
        ``categorical_covariates`` (extra design columns, typically calendar
        indicators), and ``placebo_periods`` (the Section 3.3 diagnostic).

    Returns
    -------
    GPITSResults
        Counterfactual and pointwise band over every period, the ATT with its
        interval, the running cumulative effect with intervals from the full
        post-period covariance, fit diagnostics, and the placebo checks.

    Notes
    -----
    Any untreated units in the frame are ignored: the counterfactual is built
    from the treated unit's own series alone. The hyperparameters are chosen in
    two stages, as the paper specifies -- the length-scale by a rule that reads
    only the covariates, then the noise variance by marginal likelihood with
    the length-scale fixed. Pin either through ``length_scale`` or
    ``noise_variance`` and the result records that you did.

    Examples
    --------
    >>> from mlsynth import GPITS
    >>> res = GPITS({"df": panel, "outcome": "handgun_rate", "treat": "treated",
    ...              "unitid": "unit", "time": "date",
    ...              "covariates": ["month"],
    ...              "categorical_covariates": ["month"],
    ...              "period": 12}).fit()          # doctest: +SKIP
    >>> res.effects.att                            # doctest: +SKIP
    """

    def __init__(self, config: Union[GPITSConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = GPITSConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid GPITS configuration: {exc}"
                ) from exc

        self.config: GPITSConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.kernel: str = config.kernel
        self.period = config.period
        self.covariates = config.covariates
        self.categorical_covariates = config.categorical_covariates
        self.length_scale = config.length_scale
        self.noise_variance = config.noise_variance
        self.interval_type: str = config.interval_type
        self.alpha: float = config.alpha
        self.placebo_periods = config.placebo_periods

        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color = config.treated_color

    def fit(self) -> GPITSResults:
        """Run the GPITS pipeline and return structured results."""
        try:
            inputs = prepare_gpits_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                covariates=self.covariates,
                categorical_covariates=self.categorical_covariates,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - setup raises only
            # translated errors; this catches an unforeseen third-party failure
            raise MlsynthDataError(f"Error preparing GPITS inputs: {exc}") from exc

        try:
            design = fit_gpits(
                inputs, kernel=self.kernel, period=self.period,
                length_scale=self.length_scale,
                noise_variance=self.noise_variance,
                interval_type=self.interval_type, alpha=self.alpha,
            )
            (att, att_ci, observed, counterfactual, gap, lower, upper,
             cumulative, cumulative_ci, diagnostics) = summarize_effects(
                inputs, design, self.alpha)
            placebo = None
            if self.placebo_periods is not None:
                placebo = run_placebo(
                    inputs, placebo_periods=self.placebo_periods,
                    kernel=self.kernel, period=self.period,
                    length_scale=self.length_scale,
                    noise_variance=self.noise_variance,
                    interval_type=self.interval_type, alpha=self.alpha,
                )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - the pipeline raises only
            # translated errors; this catches an unforeseen numerical failure
            raise MlsynthEstimationError(f"GPITS estimation failed: {exc}") from exc

        results = GPITSResults(
            inputs=inputs,
            design=design,
            att_value=att,
            att_interval=att_ci,
            observed=observed,
            cf_series=counterfactual,
            gap_series=gap,
            counterfactual_lower=lower,
            counterfactual_upper=upper,
            cumulative_effect=cumulative,
            cumulative_ci=cumulative_ci,
            time_labels=inputs.time_index.labels,
            fit_diagnostics_detail=diagnostics,
            placebo=placebo,
            metadata={
                "T0": inputs.T0,
                "n_post": inputs.n_post,
                "kernel": design.kernel,
                "period": self.period,
                "interval_type": self.interval_type,
                "confidence_level": 1.0 - self.alpha,
                "design_columns": inputs.column_names,
            },
        )

        if self.display_graphs:
            try:
                plot_gpits(
                    results,
                    counterfactual_color=self.counterfactual_color,
                    treated_color=self.treated_color,
                )
            except MlsynthPlottingError:  # pragma: no cover - the plotter does
                raise                     # not raise this; re-raised unwrapped
            except Exception as exc:
                warnings.warn(
                    f"GPITS plotting failed: {type(exc).__name__}: {exc}",
                    UserWarning,
                )

        return results
