"""SPOTSYNTH: spillover detection for donor selection (O'Riordan & Gilligan-Lee 2025).

O'Riordan, M. & Gilligan-Lee, C. M. (2025). *"Spillover detection for donor
selection in synthetic control models."* Journal of Causal Inference
13:20240036. doi:10.1515/jci-2024-0036.

To identify a causal effect, a synthetic control needs donors that are **not**
impacted by the intervention -- *valid* donors. Deciding which donors are valid
usually demands strong a-priori domain knowledge, which is infeasible with large
donor pools. SPOTSYNTH replaces that domain knowledge with a **forecast test**.

The paper's Theorem 3.1 shows that, under invariant causal mechanisms and the
proxy-completeness condition, a valid donor's post-intervention value is
forecastable from pre-intervention donor data. Algorithm 1 turns this into a
practical screen: for each candidate donor, fit a forecast on pre-intervention
data and predict the first post-intervention value. A donor whose realised value
departs from the forecast has either been hit by a spillover or seen its latent
distribution shift -- either way it is excluded. Two selection rules are offered:
``S1`` (keep the donors with the smallest forecast error; the analyst fixes the
number kept) and ``S2`` (keep donors whose realised value falls inside a
posterior predictive interval; controls the false-positive rate). The surviving
donors feed a canonical simplex synthetic control.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SPOTSYNTHConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.spotsynth_helpers.pipeline import run_spotsynth
from ..utils.spotsynth_helpers.setup import prepare_spotsynth_inputs
from ..utils.spotsynth_helpers.structures import SpotSynthResults


class SPOTSYNTH:
    """Spillover-detecting synthetic control.

    Parameters
    ----------
    config : SPOTSYNTHConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SPOTSYNTHConfig`.
    """

    def __init__(self, config: Union[SPOTSYNTHConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SPOTSYNTHConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SPOTSYNTH configuration: {exc}"
                ) from exc
        self.config: SPOTSYNTHConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.selection: str = config.selection
        self.forecast: str = config.forecast
        self.n_donors = config.n_donors
        self.ppi: float = config.ppi
        self.n_factors: int = config.n_factors
        self.time_average = config.time_average
        self.inference: str = config.inference
        self.dirichlet_alpha: float = config.dirichlet_alpha
        self.ci_level: float = config.ci_level
        self.n_samples: int = config.n_samples
        self.n_warmup: int = config.n_warmup
        self.debias: bool = config.debias
        self.seed: int = config.seed
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SpotSynthResults:
        """Screen donors for spillover, fit the synthetic control, return results."""
        try:
            inputs = prepare_spotsynth_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
            )
            results = run_spotsynth(
                inputs, selection=self.selection, forecast=self.forecast,
                n_donors=self.n_donors, ppi=self.ppi, n_factors=self.n_factors,
                time_average=self.time_average, inference=self.inference,
                dirichlet_alpha=self.dirichlet_alpha, ci_level=self.ci_level,
                n_samples=self.n_samples, n_warmup=self.n_warmup,
                debias=self.debias, seed=self.seed,
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
                f"SPOTSYNTH estimation failed: {exc}"
            ) from exc
