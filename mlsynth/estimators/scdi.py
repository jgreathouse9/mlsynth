"""Synthetic Control Design Intervention (SCDI) estimator."""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SCDIConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.scdi_helpers.inference import permutation_test_global
from ..utils.scdi_helpers.optimization import solve_synthetic_design
from ..utils.scdi_helpers.plotter import plot_scdi_design
from ..utils.scdi_helpers.setup import prepare_scdi_inputs
from ..utils.scdi_helpers.structures import SCDIResults


class SCDI:
    """Synthetic Control Design Intervention estimator.

    SCDI solves a mixed-integer synthetic design problem that selects ``K``
    treated units from a balanced panel and constructs either a global two-way
    synthetic treated/control contrast or per-unit synthetic twins. Unlike
    standard treatment-effect estimators, SCDI designs the assignment vector
    rather than consuming an observed treatment indicator.
    """

    def __init__(self, config: Union[SCDIConfig, dict]) -> None:
        """Initialize SCDI from a Pydantic config or compatible dictionary."""

        if isinstance(config, dict):
            try:
                config = SCDIConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid SCDI configuration: {exc}") from exc

        self.config = config

        # Core panel data
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time

        # Design specification
        self.K: int = config.K
        self.mode: str = config.mode
        self.lam: Optional[float] = config.lam

        # Pre/post split
        self.T0: Optional[int] = config.T0
        self.post_col: Optional[str] = config.post_col

        # Inference, solver, and display controls
        self.alpha: float = config.alpha
        self.run_inference: bool = config.run_inference
        self.solver: Any = config.solver
        self.display_graph: bool = config.display_graph
        self.verbose: bool = config.verbose

    def fit(self) -> SCDIResults:
        """Run the SCDI design, optional post-period inference, and plotting pipeline."""

        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_scdi_inputs(
                df=self.df,
                outcome=self.outcome,
                unitid=self.unitid,
                time=self.time,
                T0=self.T0,
                post_col=self.post_col,
            )

            design = solve_synthetic_design(
                Y=inputs.Y_pre,
                K=self.K,
                mode=self.mode,
                lam=self.lam,
                solver=self.solver,
                verbose=self.verbose,
                unit_index=inputs.unit_index,
            )

            inference = None
            if self.run_inference and inputs.Y_post is not None:
                if self.mode == "global_2way":
                    inference = permutation_test_global(
                        Y_pre=inputs.Y_pre,
                        Y_post=inputs.Y_post,
                        design=design,
                        alpha=self.alpha,
                    )
                else:
                    raise MlsynthConfigError(
                        "Post-period inference is currently implemented only for mode='global_2way'."
                    )

            results = SCDIResults(design=design, inputs=inputs, inference=inference)

            if self.display_graph:
                try:
                    plot_scdi_design(results)
                except Exception as exc:
                    raise MlsynthPlottingError(f"SCDI plotting failed: {exc}") from exc

            return results

        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, MlsynthPlottingError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"SCDI estimation failed: {exc}") from exc
