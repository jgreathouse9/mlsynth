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

from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.tssc_helpers.config import TSSCConfig
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
        Configuration object. See
        :class:`mlsynth.utils.tssc_helpers.config.TSSCConfig`.

    Returns
    -------
    TSSCResults
        Container with the SC-class variant fits, the Step-1 selection
        record, and a standardized ``summary`` for the recommended variant.

    Notes
    -----
    By default all four variants are fit and Step 1 recommends one. Setting
    ``config.method`` fits that variant alone and skips Step 1 entirely, so
    ``results.selection`` is None and ``recommended_method`` names the forced
    variant; ``config.inference=False`` additionally drops the subsampling
    confidence intervals, which dominate the runtime. Neither option changes a
    point estimate. See :class:`~mlsynth.utils.tssc_helpers.config.TSSCConfig`.
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
        self.method = config.method
        self.inference: bool = config.inference
        self.ci: float = config.ci
        self.seed = config.seed
        self.display_graphs: bool = config.display_graphs
        self.compute_scpi_pi: bool = config.compute_scpi_pi
        self.scpi_sims: int = config.scpi_sims
        self.scpi_alpha: float = config.scpi_alpha
        self.scpi_e_method: str = config.scpi_e_method

    def fit(self) -> TSSCResults:
        """Run the two-step pipeline and return the design."""

        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_tssc_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat,
            )

            # Only the requested variant is fit. The point estimate is a
            # deterministic QP and does not depend on the RNG, so it matches the
            # full run exactly; the confidence interval is a fresh subsampling
            # draw rather than a replay of the full run's stream position, and
            # so may differ in its last digits. Pinned in
            # tests/test_tssc_variant_select.py.
            ci_rng = np.random.default_rng(self.seed)
            wanted = METHODS if self.method is None else (self.method,)
            variants = {
                method: fit_variant(
                    inputs, method, n_bootstrap=self.draws,
                    confidence_level=self.ci, rng=ci_rng,
                    compute_scpi_pi=self.compute_scpi_pi,
                    scpi_sims=self.scpi_sims, scpi_alpha=self.scpi_alpha,
                    scpi_e_method=self.scpi_e_method,
                    scpi_seed=int(self.seed) if self.seed is not None else 0,
                    compute_ci=self.inference,
                )
                for method in wanted
            }

            if self.method is None:
                selection = select_method(
                    inputs=inputs,
                    alpha=self.alpha,
                    subsample_size=self.subsample_size,
                    n_subsamples=self.draws,
                    seed=self.seed,
                )
                chosen = selection.recommended
            else:
                # Step 1 is skipped, not overridden: there is no recommendation
                # to report once the caller has made the choice, and reporting
                # one would misrepresent the test as having endorsed it.
                selection = None
                chosen = self.method

            summary = build_summary(
                inputs=inputs,
                variant=variants[chosen],
                selection=selection,
                method_name=chosen,
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
