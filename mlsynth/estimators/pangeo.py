"""PANGEO: Parallel-trends supergeo experimental design.

PANGEO is a prospective **experimental-design** method for geographic
(geo) experiments, in the lineage of Supergeo Design (Chen, Doudchenko,
Jiang, Stein & Ying 2023). The Supergeo idea -- group geos into composite
"supergeos" and form balanced pairs, randomising treatment within each
pair, without trimming any geo -- is retained, including its set-
partitioning mixed-integer program.

The departure is the **matching objective**. Supergeo (and the scalable
OSD variant) match on a *scalar* aggregate (the summed response) or a few
summary covariate balances. PANGEO instead matches on the full
pre-treatment **trajectory**: it chooses the partition whose treatment and
control halves are as *parallel as possible* over the pre-period, scored
by the difference-in-differences pre-period residual sum of squares (the
level-removed gap variance; cf.
:func:`mlsynth.utils.selector_helpers._did_from_mean`). Because the DiD
level shift is absorbed, two supergeos can differ in level yet still match
perfectly on *shape* -- exactly what a downstream DiD / synthetic-control
analysis needs, and what scalar sum-matching throws away.

Multi-arm support: a single categorical column names each geo's eligible
treatment arm (e.g. ``A``/``B``/``C``); arms occupy non-overlapping geos
and PANGEO designs each arm independently. The output is a **design**
(supergeo pairs + treatment/control assignment + achieved parallelism),
not a treatment effect.
"""

from __future__ import annotations

import dataclasses
from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import PANGEOConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.pangeo_helpers.effects import compute_pangeo_effects
from ..utils.pangeo_helpers.pipeline import run_pangeo
from ..utils.pangeo_helpers.plotter import plot_pangeo
from ..utils.pangeo_helpers.setup import build_post_matrix, prepare_pangeo_inputs
from ..utils.pangeo_helpers.structures import PangeoResults


class PANGEO:
    """Parallel-trends supergeo experimental design.

    Parameters
    ----------
    config : PANGEOConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.PANGEOConfig`.
    """

    def __init__(self, config: Union[PANGEOConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = PANGEOConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid PANGEO configuration: {exc}"
                ) from exc
        self.config: PANGEOConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.arm: str = config.arm
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.post_col = config.post_col
        self.weight_col = config.weight_col
        self.max_supergeo_size = config.max_supergeo_size
        self.min_pairs: int = config.min_pairs
        self.objective: str = config.objective
        self.recency_decay: float = config.recency_decay
        self.frac_E: float = config.frac_E
        self.covariates = config.covariates
        self.covariate_weights = config.covariate_weights
        self.standardize_covariates: bool = config.standardize_covariates
        self.compute_power: bool = config.compute_power
        self.power_target: float = config.power_target
        self.power_alpha: float = config.power_alpha
        self.power_post_periods = config.power_post_periods
        self.att_augment: bool = config.att_augment
        self.att_trend: bool = config.att_trend
        self.display_graphs: bool = config.display_graphs
        self.save = config.save

    def fit(self) -> PangeoResults:
        """Design the parallel supergeo pairs and return :class:`PangeoResults`.

        With a ``post_col``, the design is built on the pre rows only (so it
        is identical to the design-only result) and the realized DiD ATT on
        the post rows is attached as ``results.effects``.
        """
        try:
            if self.post_col is not None:
                if self.post_col not in self.df.columns:
                    raise MlsynthDataError(
                        f"post_col {self.post_col!r} missing.")
                pre_df = self.df[self.df[self.post_col] == 0].copy()
                post_df = self.df[self.df[self.post_col] != 0].copy()
                if pre_df.empty:
                    raise MlsynthDataError("No pre-period rows (post_col).")
            else:
                pre_df, post_df = self.df, None

            inputs = prepare_pangeo_inputs(
                df=pre_df, outcome=self.outcome, arm=self.arm,
                unitid=self.unitid, time=self.time,
                covariates=self.covariates,
                standardize_covariates=self.standardize_covariates,
                weight_col=self.weight_col,
            )
            results = run_pangeo(
                inputs=inputs,
                max_supergeo_size=self.max_supergeo_size,
                min_pairs=self.min_pairs,
                objective=self.objective,
                recency_decay=self.recency_decay,
                frac_E=self.frac_E,
                covariate_weights=self.covariate_weights,
                compute_power=self.compute_power,
                power_target=self.power_target,
                power_alpha=self.power_alpha,
                power_post_periods=self.power_post_periods,
                att_augment=self.att_augment,
                att_trend=self.att_trend,
            )

            if post_df is not None and not post_df.empty:
                Y_post, _ = build_post_matrix(
                    post_df, inputs, self.outcome, self.unitid, self.time)
                effects = compute_pangeo_effects(
                    results, inputs, Y_post, alpha=self.power_alpha,
                    augment=self.att_augment, trend=self.att_trend)
                results = dataclasses.replace(results, effects=effects)

            if self.display_graphs:
                plot_pangeo(results, save=self.save, outcome_label=self.outcome)
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"PANGEO design failed: {exc}"
            ) from exc
