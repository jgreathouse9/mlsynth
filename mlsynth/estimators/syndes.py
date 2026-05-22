"""Synthetic Design (SYNDES) estimator.

Implements the three mixed-integer programming formulations of
Doudchenko, Khosravi, Pouget-Abadie, Lahaie, Lubin, Mirrokni, Spiess,
and Imbens (2021), *"Synthetic Design: An Optimization Approach to
Experimental Design with Synthetic Controls"* (arXiv:2112.00278).
SYNDES jointly selects treated units and synthetic-control weights
by solving a single MIP that minimises the post-period mean squared
error of the resulting ATT estimator.

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
``mlsynth.utils.syndes_helpers.inference``.

Two additional MIP features round out the estimator:

* **Budget constraint** (paper section 1): supply ``costs`` (length
  ``N``) and ``budget`` to add ``sum_i c_i D_i <= B`` to the MIP.
* **Annealed relaxation** (``mode="two_way_global_annealed"``):
  simulated-annealing alternative to the MIP for the symmetric
  two-way formulation. Useful when a commercial MIP solver is
  unavailable or the problem size makes the MIP impractical.

Post-fit, see :func:`mlsynth.power_analysis` for per-horizon
minimum-detectable-effect tables.
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
from ..utils.syndes_helpers.inference import (
    permutation_test_global,
    permutation_test_relaxed_global,
)
from ..utils.syndes_helpers.optimization import solve_synthetic_design
from ..utils.syndes_helpers.plotter import plot_syndes_design
from ..utils.syndes_helpers.relaxed_solver import solve_two_way_relaxed
from ..utils.syndes_helpers.relaxed_structures import RelaxedSolverResults
from ..utils.syndes_helpers.setup import prepare_syndes_inputs
from ..utils.syndes_helpers.structures import SYNDESResults


# Paper-aligned -> internal optimization mode mapping. The optimization
# layer uses the paper's "global_2way / global_equal_weights / per_unit"
# vocabulary internally; the public SYNDES API exposes the paper-title
# names. Mapping happens at the orchestrator boundary so the design
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

    Returns
    -------
    SYNDESResults or RelaxedSolverResults
        For the three MIP modes, a :class:`SYNDESResults` container
        with the optimised design and optional permutation inference.
        For ``mode="two_way_global_annealed"`` the relaxed solver
        returns a :class:`RelaxedSolverResults` container with
        ``design``, ``trace``, ``inputs``, and optional ``inference``.
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
        self.lam: Optional[float] = config.lam
        self.T0: Optional[int] = config.T0
        self.post_col: Optional[str] = config.post_col
        self.alpha: float = config.alpha
        self.run_inference: bool = config.run_inference
        self.solver: Any = config.solver
        self.relaxed_max_iter: int = config.relaxed_max_iter
        self.relaxed_decay: float = config.relaxed_decay
        self.display_graph: bool = config.display_graph
        self.verbose: bool = config.verbose
        self.costs = config.costs
        self.budget = config.budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> Union[SYNDESResults, RelaxedSolverResults]:
        """Solve the MIP (or relaxation), run optional inference, return results."""

        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_syndes_inputs(
                df=self.df,
                outcome=self.outcome,
                unitid=self.unitid,
                time=self.time,
                T0=self.T0,
                post_col=self.post_col,
            )

            if self.mode_public == "two_way_global_annealed":
                return self._fit_relaxed(inputs)
            return self._fit_mip(inputs)

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

    # ------------------------------------------------------------------
    # MIP path
    # ------------------------------------------------------------------

    def _fit_mip(self, inputs) -> SYNDESResults:
        mode_internal = _MODE_TO_INTERNAL[self.mode_public]

        design = solve_synthetic_design(
            Y=inputs.Y_pre,
            K=self.K,
            mode=mode_internal,
            lam=self.lam,
            solver=self.solver,
            verbose=self.verbose,
            unit_index=inputs.unit_index,
            costs=self.costs,
            budget=self.budget,
        )

        # Re-tag with the paper-aligned mode label so the design surface
        # the user sees uses SYNDES vocabulary.
        design = replace(design, mode=self.mode_public)

        inference = None
        if self.run_inference and inputs.Y_post is not None:
            inference = permutation_test_global(
                Y_pre=inputs.Y_pre,
                Y_post=inputs.Y_post,
                design=design,
                alpha=self.alpha,
            )

        results = SYNDESResults(
            design=design, inputs=inputs, inference=inference
        )

        if self.display_graph:
            try:
                plot_syndes_design(results)
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"SYNDES plotting failed: {exc}"
                ) from exc

        return results

    # ------------------------------------------------------------------
    # Annealed relaxation path
    # ------------------------------------------------------------------

    def _fit_relaxed(self, inputs) -> RelaxedSolverResults:
        if self.K is None:
            raise MlsynthConfigError(
                "Annealed relaxation requires an explicit K."
            )

        relaxed = solve_two_way_relaxed(
            Y=inputs.Y_pre,
            K=self.K,
            lam=self.lam,
            max_iter=self.relaxed_max_iter,
            decay=self.relaxed_decay,
            verbose=self.verbose,
        )

        inference = None
        if self.run_inference and inputs.Y_post is not None:
            inference = permutation_test_relaxed_global(
                Y_pre=inputs.Y_pre,
                Y_post=inputs.Y_post,
                design=relaxed.design,
                alpha=self.alpha,
            )

        return replace(relaxed, inputs=inputs, inference=inference)
