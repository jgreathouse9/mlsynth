"""Two-Step Synthetic Control (TSSC) estimator.

Implements:

    Li, K. T., & Shankar, V. (2023). "A Two-Step Synthetic Control Approach
    for Estimating Causal Effects of Marketing Events." Management Science.
    https://doi.org/10.1287/mnsc.2023.4878

TSSC addresses a gap in synthetic-control practice: the SC pretrends
assumption is usually checked only by visual inspection. TSSC instead

1. **Step 1 (model selection).** Formally tests the SC pretrends
   assumption -- equivalent to the joint restriction that the donor
   weights sum to one *and* the intercept is zero (Proposition 3.1) --
   using a subsampling procedure (Proposition 3.2), then walks a decision
   tree to recommend the SC-class variant that balances bias and
   efficiency: SC, MSCa, MSCb, or MSCc.
2. **Step 2 (estimation).** Fits the recommended variant and reports the
   ATT as the mean post-period gap.

See ``mlsynth.utils.tssc_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import TSSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.tssc_helpers.estimation import fit_variant
from ..utils.tssc_helpers.plotter import plot_tssc
from ..utils.tssc_helpers.results_assembly import build_summary
from ..utils.tssc_helpers.selection import select_method
from ..utils.tssc_helpers.setup import prepare_tssc_inputs
from ..utils.tssc_helpers.structures import METHODS, TSSCResults


class TSSC:
    """Two-Step Synthetic Control (TSSC) estimator.

    Parameters
    ----------
    config : TSSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.TSSCConfig`.

    Returns
    -------
    TSSCResults
        Container with all four SC-class variant fits, the Step-1
        selection record, and a standardized ``summary`` for the
        recommended variant.
    """

    def __init__(self, config: Union[TSSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = TSSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid TSSC configuration: {exc}"
                ) from exc

        self.config: TSSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.alpha: float = config.alpha
        self.subsample_size = config.subsample_size
        self.draws: int = config.draws
        self.ci: float = config.ci
        self.seed = config.seed
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> TSSCResults:
        """Run the two-step pipeline and return the design."""

        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_tssc_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat,
            )

            ci_rng = np.random.default_rng(self.seed)
            variants = {
                method: fit_variant(
                    inputs, method, n_bootstrap=self.draws,
                    confidence_level=self.ci, rng=ci_rng,
                )
                for method in METHODS
            }

            selection = select_method(
                inputs=inputs,
                alpha=self.alpha,
                subsample_size=self.subsample_size,
                n_subsamples=self.draws,
                seed=self.seed,
            )

            summary = build_summary(
                inputs=inputs,
                variant=variants[selection.recommended],
                selection=selection,
            )

            results = TSSCResults(
                inputs=inputs, variants=variants, selection=selection,
                summary=summary,
            )

        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"TSSC estimation failed: {exc}") from exc

        if self.display_graphs:
            try:
                plot_tssc(results)
            except Exception as exc:
                raise MlsynthPlottingError(f"TSSC plotting failed: {exc}") from exc

        return results
