"""SNN: Synthetic Nearest Neighbors / Causal Matrix Completion (Agarwal et al. 2021).

Agarwal, A., Dahleh, M., Shah, D. & Shen, D. (2021). *"Causal Matrix
Completion."* arXiv:2109.15154.

SNN recovers missing entries of a partially observed matrix under
**missing not at random (MNAR)** patterns -- where the probability that an
entry is observed depends on the underlying values (selection bias), as in
recommender systems and panel data. It does so by combining nearest
neighbors (collaborative filtering) with synthetic controls: for a target
entry it finds a fully observed **anchor submatrix** and runs **principal
component regression** to impute the value. SNN generalises the Synthetic
Interventions estimator (which mlsynth exposes as :class:`mlsynth.SI`),
which in turn generalises classic synthetic control.

In the causal/panel setting handled by this estimator, the treated units'
post-treatment untreated potential outcomes :math:`Y(0)` are exactly the
missing entries; SNN imputes them and the treatment effect is the
observed outcome minus the imputed counterfactual. The underlying matrix
completion engine is also exposed directly via
:func:`mlsynth.utils.snn_helpers.snn_complete` for general MNAR matrix
completion (e.g. recommender systems).

The block-structured missingness of panel data -- a fully observed
control block -- naturally induces the anchor rows and columns SNN needs,
so the method is especially well suited to comparative case studies and
staggered-adoption designs.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SNNConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.snn_helpers.pipeline import run_snn
from ..utils.snn_helpers.setup import prepare_snn_inputs
from ..utils.snn_helpers.structures import SNNResults


class SNN:
    """Synthetic Nearest Neighbors (causal matrix completion) estimator.

    Parameters
    ----------
    config : SNNConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SNNConfig`.
    """

    def __init__(self, config: Union[SNNConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SNNConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SNN configuration: {exc}"
                ) from exc
        self.config: SNNConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.n_neighbors: int = config.n_neighbors
        self.max_rank = config.max_rank
        self.spectral_energy: float = config.spectral_energy
        self.universal_rank: bool = config.universal_rank
        self.clip: bool = config.clip
        self.inference: bool = config.inference
        self.alpha: float = config.alpha
        self.random_state: int = config.random_state
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SNNResults:
        """Run SNN and return :class:`SNNResults`."""
        try:
            inputs = prepare_snn_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
            )
            results = run_snn(
                inputs=inputs,
                n_neighbors=self.n_neighbors,
                max_rank=self.max_rank,
                spectral_energy=self.spectral_energy,
                universal=self.universal_rank,
                clip=self.clip,
                inference=self.inference,
                alpha_level=self.alpha,
                random_state=self.random_state,
            )
            # Attach plotting context so result.plot() is self-contained and
            # styled from the (possibly nested) config; labels default to the
            # column names when the user has not set them.
            pc = self.config.resolved_plot()
            if pc.xlabel is None:
                pc.xlabel = self.time
            if pc.ylabel is None:
                pc.ylabel = self.outcome
            object.__setattr__(results, "plot_config", pc)
            if self.display_graphs:
                results.plot()
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SNN estimation failed: {exc}"
            ) from exc
