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

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import PANGEOConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.pangeo_helpers.pipeline import run_pangeo
from ..utils.pangeo_helpers.plotter import plot_pangeo
from ..utils.pangeo_helpers.setup import prepare_pangeo_inputs
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
        self.max_supergeo_size: int = config.max_supergeo_size
        self.min_pairs: int = config.min_pairs
        self.display_graphs: bool = config.display_graphs
        self.save = config.save

    def fit(self) -> PangeoResults:
        """Design the parallel supergeo pairs and return :class:`PangeoResults`."""
        try:
            inputs = prepare_pangeo_inputs(
                df=self.df, outcome=self.outcome, arm=self.arm,
                unitid=self.unitid, time=self.time,
            )
            results = run_pangeo(
                inputs=inputs,
                max_supergeo_size=self.max_supergeo_size,
                min_pairs=self.min_pairs,
            )
            if self.display_graphs:
                plot_pangeo(results, save=self.save, outcome_label=self.outcome)
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"PANGEO design failed: {exc}"
            ) from exc
