"""Synthetic Business Cycle (SBC) estimator.

Implements:

    Shi, Z., Xi, J., & Xie, H. (2025). "A Synthetic Business Cycle Approach
    to Counterfactual Analysis with Nonstationary Macroeconomic Data."
    arXiv:2505.22388.

SBC tackles the spurious-regression risk of running standard SCM on
nonstationary macro series by splitting each outcome into a trend
(via the Hamilton (2018) filter) and a stationary cycle. The trend of
the treated unit is extrapolated from its own pre-treatment lags; the
cycle is imputed with a classic Abadie-Diamond-Hainmueller simplex SCM
fit on the donors' cycles. The post-treatment counterfactual is then

    Y_hat_{1, t}(0) = tau_hat_{1, t} + c_hat_{1, t}.

See ``mlsynth.utils.sbc_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SBCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.sbc_helpers.orchestration import solve_sbc, summarize_effects
from ..utils.sbc_helpers.plotter import plot_sbc
from ..utils.sbc_helpers.setup import prepare_sbc_inputs
from ..utils.sbc_helpers.structures import SBCResults


class SBC:
    """Synthetic Business Cycle (SBC) estimator.

    Parameters
    ----------
    config : SBCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SBCConfig`.

    Returns
    -------
    SBCResults
        Hamilton fits, donor weights, post-treatment counterfactual, ATT.
    """

    def __init__(self, config: Union[SBCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SBCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SBC configuration: {exc}"
                ) from exc

        self.config: SBCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat

        self.h: int = config.h
        self.p: int = config.p
        self.weights_mode: str = config.weights_mode
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> SBCResults:
        """Run the SBC pipeline."""

        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_sbc_inputs(
                df=self.df,
                outcome=self.outcome,
                unitid=self.unitid,
                time=self.time,
                treat=self.treat,
                h=self.h,
                p=self.p,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:
            raise MlsynthDataError(f"Error preparing SBC inputs: {exc}") from exc

        try:
            design = solve_sbc(
                inputs=inputs,
                h=self.h,
                p=self.p,
                weights_mode=self.weights_mode,
            )
            att, cf_full, te = summarize_effects(inputs=inputs, design=design)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SBC estimation failed: {exc}"
            ) from exc

        # Build per-donor weight dict (skip the trivially-zero entries in
        # the simplex case for legibility).
        threshold = 1e-8 if self.weights_mode == "simplex" else -1.0
        weights_by_donor = {
            str(name): float(w)
            for name, w in zip(inputs.donor_names, design.weights)
            if abs(w) > threshold
        }

        results = SBCResults(
            inputs=inputs,
            design=design,
            att=att,
            counterfactual_full=cf_full,
            treatment_effect=te,
            weights_by_donor=weights_by_donor,
        )

        if self.display_graphs:
            try:
                plot_sbc(results)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"SBC plotting failed: {exc}"
                ) from exc

        return results
