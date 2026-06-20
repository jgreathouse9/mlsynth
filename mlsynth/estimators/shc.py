"""Synthetic Historical Control (SHC) estimator.

Implements:

    Chen, Y.-T., Yang, J.-C., & Yang, T.-T. (2024). "Synthetic Historical
    Control for Policy Evaluation." SSRN 4995085.

SHC estimates the time-varying intervention effect on a single treated
unit *without cross-sectional controls*. It is built on a semi-parametric
time-series regression whose latent, smooth trend ``ell_t`` is the
time-varying confounder. The trend is estimated by kernel regression on
the pre-period, the series is partitioned into a "treated block" and a set
of overlapping "historical blocks", and the treated block's pre-segment is
matched by a simplex combination of its historical counterparts. Applying
that combination to the historical forward segments identifies the
post-intervention counterfactual -- the spirit of the synthetic control
method carried over to a single time series.

The estimator is a thin orchestration layer over
:mod:`mlsynth.utils.shc_helpers`:

    setup.py         : long DataFrame -> SHCInputs (time IndexSet)
    orchestration.py : solve_shc + summarize_effects (Section 2.3)
    inference.py     : conformal permutation test (footnote 21)
    plotter.py       : observed vs counterfactual
    structures.py    : frozen result containers
"""

from __future__ import annotations

import warnings
from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SHCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.shc_helpers.inference import run_conformal_inference
from ..utils.shc_helpers.orchestration import solve_shc, summarize_effects
from ..utils.shc_helpers.plotter import plot_shc
from ..utils.shc_helpers.setup import prepare_shc_inputs
from ..utils.shc_helpers.structures import SHCResults


class SHC:
    """Synthetic Historical Control (SHC) estimator.

    Estimates a single treated unit's untreated counterfactual from its own
    time series alone, by matching the latent pre-intervention trend with a
    simplex combination of overlapping historical blocks (Chen, Yang &
    Yang 2024). The augmented variant (ASHC) adds a ridge refinement.

    Parameters
    ----------
    config : SHCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SHCConfig`.
        Key fields: ``m`` (pre-intervention block length),
        ``use_augmented`` (ASHC), ``bandwidth_grid`` (LOOCV candidates).

    Returns
    -------
    SHCResults
        Bandwidth, latent trend, historical-block weights, the
        post-intervention counterfactual, the ATT, fit diagnostics, and the
        conformal permutation inference.

    References
    ----------
    Chen, Yi-Ting, Jui-Chung Yang, and Tzu-Ting Yang (2024). "Synthetic
    Historical Control for Policy Evaluation." SSRN 4995085.
    """

    def __init__(self, config: Union[SHCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SHCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SHC configuration: {exc}"
                ) from exc

        self.config: SHCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.m: int = config.m
        self.use_augmented: bool = config.use_augmented
        self.bandwidth_grid = config.bandwidth_grid
        self.inference_method: str = config.inference_method
        self.permutation_scheme: str = config.permutation_scheme
        self.num_permutations = config.num_permutations

        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SHCResults:
        """Run the SHC pipeline and return structured results."""
        try:
            inputs = prepare_shc_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                m=self.m,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:
            raise MlsynthDataError(f"Error preparing SHC inputs: {exc}") from exc

        try:
            design = solve_shc(
                inputs,
                use_augmented=self.use_augmented,
                bandwidth_grid=self.bandwidth_grid,
            )
            (att, att_percent, observed, counterfactual, gap,
             window_time, fit_diagnostics) = summarize_effects(inputs, design)
            inference = run_conformal_inference(
                inputs, design, observed, counterfactual,
                method=self.inference_method,
                permutation_scheme=self.permutation_scheme,
                num_permutations=self.num_permutations,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SHC estimation failed: {exc}"
            ) from exc

        results = SHCResults(
            inputs=inputs,
            design=design,
            att_value=att,
            att_percent=att_percent,
            observed=observed,
            cf_window=counterfactual,
            gap_window=gap,
            time_labels=window_time,
            fit_diagnostics_detail=fit_diagnostics,
            inference_detail=inference,
            metadata={
                "m": inputs.m,
                "n": inputs.n,
                "n_historical_blocks": inputs.N,
                "bandwidth": design.bandwidth,
                "use_augmented": design.use_augmented,
                "best_lambda": design.best_lambda,
            },
        )

        if self.display_graphs:
            cf_color = self.counterfactual_color
            if isinstance(cf_color, (list, tuple)):
                cf_color = cf_color[0] if cf_color else "red"
            try:
                plot_shc(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=cf_color,
                )
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                warnings.warn(
                    f"SHC plotting failed: {type(exc).__name__}: {exc}",
                    UserWarning,
                )

        return results
