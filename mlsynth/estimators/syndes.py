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

import numpy as np
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
from ..utils.post_fit import compute_post_fit, compute_power_analysis
from ..utils.syndes_helpers.inference import (
    permutation_test_global,
    permutation_test_relaxed_global,
)
from ..utils.syndes_helpers.optimization import solve_synthetic_design
from ..utils.syndes_helpers.plotter import plot_syndes_design
from ..utils.syndes_helpers.relaxed_solver import solve_two_way_relaxed
from ..utils.syndes_helpers.relaxed_structures import RelaxedSolverResults
from ..utils.syndes_helpers.setup import prepare_syndes_inputs
from ..utils.syndes_helpers.structures import (
    SYNDESMultiArmResults,
    SYNDESResults,
)


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


def _syndes_post_fit(inputs, design, inference, alpha):
    """Build a :class:`SyntheticControlPostFit` for any SYNDES design.

    SYNDES has no pre-period blank window (its inference is a moving-block
    permutation on the post-period), so ``n_blank = 0`` and the unified
    power-analysis module falls back to the pre-period gap as its placebo
    proxy for the noise scale.

    Three mode shapes are handled:

    * ``global_2way`` / ``global_equal_weights`` -- ``treated_weights`` and
      ``control_weights`` are both ``(N,)`` simplex vectors; the synthetic
      trajectories are :math:`Y \\cdot w_t` and :math:`Y \\cdot w_c`.
    * ``per_unit`` -- ``treated_weights`` is the per-treated-unit SC matrix
      ``q`` of shape ``(K, N)`` and ``control_weights`` is ``None``. The
      naturally-defined contrast for the unified post-fit is the *aggregate*
      (1/K)-averaged synthetic treated vs synthetic control derived from
      :math:`D` and :math:`q.sum(axis=0)`.
    * ``two_way_global_annealed`` -- same shape as ``global_2way`` (the
      annealed solver returns plain ``(N,)`` ``treated_weights`` /
      ``control_weights``).
    """
    Y_pre = np.asarray(inputs.Y_pre, dtype=float)
    if inputs.Y_post is not None:
        Y_full = np.vstack([Y_pre, np.asarray(inputs.Y_post, dtype=float)])
    else:
        Y_full = Y_pre
    n_fit = int(Y_pre.shape[0])
    n_post = int(Y_full.shape[0] - n_fit)

    N = Y_full.shape[1]
    mode = getattr(design, "mode", None)
    assignment = np.asarray(getattr(design, "assignment", np.zeros(N)),
                            dtype=float).flatten()
    K = int(assignment.sum()) or 1

    tw_raw = getattr(design, "treated_weights", None)
    cw_raw = getattr(design, "control_weights", None)

    if mode == "per_unit" and tw_raw is not None and np.asarray(tw_raw).ndim == 2:
        # per_unit: q is (K, N), control side derived from D - q.sum(axis=0)
        q = np.asarray(tw_raw, dtype=float)
        tw = assignment / K                       # uniform 1/K on treated set
        cw = q.sum(axis=0) / K                    # average synthetic control
    else:
        tw = (np.asarray(tw_raw, dtype=float).flatten()
              if tw_raw is not None else np.zeros(N))
        cw = (np.asarray(cw_raw, dtype=float).flatten()
              if cw_raw is not None else np.zeros(N))

    syn_t = Y_full @ tw
    syn_c = Y_full @ cw
    pf = compute_post_fit(
        treated_series=syn_t, control_series=syn_c,
        n_fit=n_fit, n_blank=0, n_post=n_post,
        treated_weights=tw, control_weights=cw,
        inference=inference,
        n_treated_units=int(np.sum(tw > 1e-8)),
    )
    try:
        from dataclasses import replace as _dc_replace
        power = compute_power_analysis(pf, alpha=alpha)
        pf = _dc_replace(pf, power=power)
    except Exception:                # never let power analysis break a fit
        pass
    return pf


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
        self.arm: Optional[str] = config.arm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
    ) -> Union[SYNDESResults, "RelaxedSolverResults", SYNDESMultiArmResults]:
        """Solve the MIP (or relaxation), run optional inference, return results.

        Returns a single result when no ``arm`` column is configured;
        otherwise solves the SYNDES design **independently within each arm's
        units** and returns a :class:`SYNDESMultiArmResults` keyed by arm
        label.
        """

        try:
            if self.arm is None:
                return self._fit_single(self.df)

            # Multi-arm: solve the SYNDES design independently within each arm.
            if self.arm not in self.df.columns:
                raise MlsynthDataError(
                    f"Arm column {self.arm!r} not found in the data."
                )
            if self.df.groupby(self.unitid)[self.arm].nunique().max() > 1:
                raise MlsynthDataError(
                    "The arm column varies within a unit over time."
                )

            arm_designs = {
                arm_label: self._fit_single(sub.copy())
                for arm_label, sub in self.df.groupby(self.arm, sort=True)
            }
            return SYNDESMultiArmResults(arm_designs=arm_designs, arm=self.arm)

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

    def _fit_single(
        self, df: pd.DataFrame
    ) -> Union[SYNDESResults, "RelaxedSolverResults"]:
        """Run the SYNDES pipeline on one (sub-)panel and return its result."""

        balance(df, self.unitid, self.time)
        inputs = prepare_syndes_inputs(
            df=df,
            outcome=self.outcome,
            unitid=self.unitid,
            time=self.time,
            T0=self.T0,
            post_col=self.post_col,
        )

        if self.mode_public == "two_way_global_annealed":
            return self._fit_relaxed(inputs)
        return self._fit_mip(inputs)

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

        post_fit = _syndes_post_fit(
            inputs=inputs, design=design,
            inference=inference, alpha=self.alpha,
        )

        results = SYNDESResults(
            design=design, inputs=inputs, inference=inference,
            post_fit=post_fit,
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

        post_fit = _syndes_post_fit(
            inputs=inputs, design=relaxed.design,
            inference=inference, alpha=self.alpha,
        )

        return replace(relaxed, inputs=inputs, inference=inference,
                        post_fit=post_fit)
