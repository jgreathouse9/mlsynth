"""Synthetic Design (SYNDES) estimator.

This is the paper-aligned interface to Doudchenko, Khosravi,
Pouget-Abadie, Lahaie, Lubin, Mirrokni, Spiess, and Imbens (2021),
*"Synthetic Design: An Optimization Approach to Experimental Design
with Synthetic Controls"* (arXiv:2112.00278). SYNDES jointly selects
treated units and synthetic-control weights via mixed-integer
programming.

This estimator is the public entry point. ``mlsynth.SCDI`` is kept
as a deprecated alias for backwards compatibility (same MIP
formulations, older mode names).

The three formulations exposed via the ``mode`` field correspond
directly to Section 3 of the paper:

* ``"per_unit"``       — separate SC weights ``w_{ji}`` per treated
                          unit ``i``. Trades a tighter per-unit fit
                          against a richer parameter space.
* ``"two_way_global"`` — single weight vector ``w_i`` applied
                          symmetrically to the treated and control
                          contrasts. Recommended when treatment
                          effects are homogeneous.
* ``"one_way_global"`` — ``two_way_global`` with the treated weights
                          pinned to ``1/K``; the SC step only adjusts
                          the *control* combination. Easiest to
                          interpret as a "weighted difference-in-means".

Inference defaults to the moving-block permutation test of
Chernozhukov, Wuethrich, and Zhu (2021), applied uniformly to all
three modes via the shared contrast-vector dispatch in
``mlsynth.utils.scdi_helpers.inference``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SYNDESConfig
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


# Paper-aligned -> internal SCDI mode mapping. The internal helpers
# pre-date the SYNDES public interface and use the older mlsynth
# naming; we translate at the orchestrator boundary so the design
# object the user sees carries the paper-aligned label.
_MODE_TO_INTERNAL = {
    "per_unit": "per_unit",
    "two_way_global": "global_2way",
    "one_way_global": "global_equal_weights",
}
_MODE_FROM_INTERNAL = {v: k for k, v in _MODE_TO_INTERNAL.items()}


class SYNDES:
    """Synthetic Design (Doudchenko et al. 2021) estimator.

    Parameters
    ----------
    config : SYNDESConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SYNDESConfig`.
    """

    def __init__(self, config: Union[SYNDESConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SYNDESConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SYNDES configuration: {exc}"
                ) from exc

        self.config: SYNDESConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.K: Optional[int] = config.K
        self.mode_public: str = config.mode
        self.mode_internal: str = _MODE_TO_INTERNAL[config.mode]
        self.lam: Optional[float] = config.lam
        self.T0: Optional[int] = config.T0
        self.post_col: Optional[str] = config.post_col
        self.alpha: float = config.alpha
        self.run_inference: bool = config.run_inference
        self.solver: Any = config.solver
        self.display_graph: bool = config.display_graph
        self.verbose: bool = config.verbose

    def fit(self) -> SCDIResults:
        """Solve the MIP, run optional inference, return :class:`SCDIResults`.

        The returned ``design.mode`` carries the paper-aligned name
        (``per_unit`` / ``two_way_global`` / ``one_way_global``).
        """

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
                mode=self.mode_internal,
                lam=self.lam,
                solver=self.solver,
                verbose=self.verbose,
                unit_index=inputs.unit_index,
            )

            # Re-tag the design with the paper-aligned mode name so
            # downstream consumers see the SYNDES vocabulary.
            design = replace(design, mode=self.mode_public)

            inference = None
            if self.run_inference and inputs.Y_post is not None:
                inference = permutation_test_global(
                    Y_pre=inputs.Y_pre,
                    Y_post=inputs.Y_post,
                    design=design,
                    alpha=self.alpha,
                )

            results = SCDIResults(
                design=design, inputs=inputs, inference=inference
            )

            if self.display_graph:
                try:
                    plot_scdi_design(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"SYNDES plotting failed: {exc}"
                    ) from exc

            return results

        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
            MlsynthPlottingError,
        ):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SYNDES estimation failed: {exc}"
            ) from exc
