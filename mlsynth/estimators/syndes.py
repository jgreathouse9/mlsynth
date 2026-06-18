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
from ..utils.syndes_helpers.holdout import select_by_holdout
from ..utils.syndes_helpers.infocriterion import select_by_ic
from ..utils.syndes_helpers.optimization import (
    solve_synthetic_design,
    solve_synthetic_design_pool,
)
from ..utils.syndes_helpers.restrictions import build_restrictions
from ..utils.syndes_helpers.plotter import plot_syndes_design
from ..utils.syndes_helpers.power import power_analysis
from ..utils.syndes_helpers.select import recommend_syndes
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


def _design_control_weights(design, n_units):
    """Return the (N,) control-side weight vector for any SYNDES mode.

    Mirrors the contrast bookkeeping in :func:`_syndes_post_fit`: the two-way /
    equal-weight / annealed modes store the control simplex in
    ``control_weights`` directly, whereas ``per_unit`` leaves ``control_weights``
    ``None`` and keeps the per-treated-unit synthetic controls in the ``(K, N)``
    ``treated_weights`` matrix -- whose column sum is the aggregate control side.
    """
    tw_raw = getattr(design, "treated_weights", None)
    cw_raw = getattr(design, "control_weights", None)
    if tw_raw is not None and np.asarray(tw_raw).ndim == 2:
        return np.asarray(tw_raw, dtype=float).sum(axis=0)
    if cw_raw is not None:
        return np.asarray(cw_raw, dtype=float).reshape(-1)
    return np.zeros(n_units)


def _syndes_power_curve(results, alpha, horizons=range(1, 13)):
    """Per-horizon MDE table (horizons 1..12 by default) for a SYNDES design.

    Attached to every SYNDES fit so the minimum-detectable-effect curve comes
    back from :meth:`SYNDES.fit` without a separate :func:`mlsynth.power_analysis`
    call, and reused per pool entry so the whole menu is comparable on power.
    Custom horizon grids, significance levels, or baselines still go through
    :func:`mlsynth.power_analysis`. Never breaks a fit: returns ``None`` on any
    degeneracy (e.g. a too-short pre-period), mirroring ``post_fit.power``.
    """
    try:
        return power_analysis(results, n_post_periods=horizons, alpha=alpha)
    except Exception:                # never let power analysis break a fit
        return None


def _syndes_pool_menu(pool_designs, inputs, costs, alpha, power_target=0.8,
                      oos_errors=None, ic_values=None, df_values=None):
    """Re-score a SYNDES solution pool into a manager-facing menu.

    The MIP ranks designs by fit alone, so each pooled design is annotated with
    the dimensions the objective ignored: its full power curve, the headline MDE%
    at the realised post horizon, and its cost (when ``costs`` is supplied).
    Returned as a list of dicts ranked by MSE.

    Crucially, every entry is *actionable*, not just rankable: it carries the
    full :class:`SYNDESDesign` under ``"design"`` (treated/control weights and
    all), the names of the donor units that form its synthetic control under
    ``"control_group"``, and its own per-horizon power curve under
    ``"power_curve"`` -- so a manager can pick any entry, not only the rank-1
    winner kept on ``results.design``, and have everything needed to deploy it
    and to compare the whole pool on power. Each entry's keys are:

    * ``"markets"``       -- labels of the treated units (the design's arms).
    * ``"control_group"`` -- labels of the donor units carrying nonzero control
      weight (the synthetic-control pool backing the treated arms).
    * ``"objective"``     -- the MIP objective (fit) the design was ranked by.
    * ``"pre_fit_rmse"``  -- root-mean-square pre-period contrast.
    * ``"mde_pct"``       -- minimum detectable effect at the realised post
      horizon, % of treated baseline. This is the design's ``power_curve``
      value at that horizon (the Newey-West HAC MDE), so the headline number and
      the curve always agree.
    * ``"cost"``          -- summed cost of the treated units (``None`` if no
      ``costs`` supplied).
    * ``"design"``        -- the full :class:`SYNDESDesign` (treated/control/
      contrast weights) for this entry.
    * ``"power_curve"``   -- the entry's own :class:`SYNDESPower` over horizons
      ``1..12`` (``None`` if the computation is degenerate).
    """
    from scipy.stats import norm

    Y_pre = np.asarray(inputs.Y_pre, dtype=float)
    n_units = Y_pre.shape[1]
    unit_labels = np.asarray(inputs.unit_index.labels)
    n_post = int(inputs.Y_post.shape[0]) if inputs.Y_post is not None else 4
    z = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power_target))
    costs_arr = None if costs is None else np.asarray(costs, dtype=float).reshape(-1)
    menu = []
    for entry_i, d in enumerate(pool_designs):
        idx = np.asarray(d.selected_unit_indices, dtype=int)
        base = float(np.mean(Y_pre[:, idx])) if idx.size else float("nan")

        # Per-entry power curve (Newey-West, horizons 1..12) so the whole pool is
        # comparable on power, not just the rank-1 winner on results.power_curve.
        power_curve = _syndes_power_curve(SYNDESResults(design=d, inputs=inputs),
                                          alpha)
        # Headline MDE% is that curve's value at the realised post horizon, so
        # the menu's summary number and its curve always agree.
        if power_curve is not None and 1 <= n_post <= 12:
            mde_pct = float(power_curve.mde_percent[n_post - 1])
        else:
            pc_h = _syndes_power_curve(SYNDESResults(design=d, inputs=inputs),
                                       alpha, horizons=[n_post])
            mde_pct = (float(pc_h.mde_percent[0]) if pc_h is not None
                       else float("nan"))
        if not np.isfinite(mde_pct):
            # Degenerate curve -> fall back to the i.i.d. realised-window MDE.
            contrast = np.asarray(d.contrast_weights, dtype=float).reshape(-1)
            per_period = Y_pre @ contrast
            sigma = (float(np.std(per_period, ddof=1)) if per_period.size > 1
                     else float(np.std(per_period)))
            mde_abs = z * sigma / np.sqrt(n_post)
            mde_pct = (100.0 * mde_abs / abs(base)
                       if abs(base) > 1e-9 else float("nan"))

        # Donor units backing the synthetic control: nonzero control weight.
        control_w = _design_control_weights(d, n_units)
        control_idx = np.flatnonzero(np.abs(control_w) > 1e-8)
        control_group = [x for x in unit_labels[control_idx].tolist()]
        menu.append({
            "markets": [x for x in np.asarray(d.selected_unit_labels).tolist()],
            "control_group": control_group,
            "objective": float(d.objective_value),
            "pre_fit_rmse": (None if d.pre_fit_rmse is None else float(d.pre_fit_rmse)),
            "mde_pct": float(mde_pct),
            "cost": (None if costs_arr is None else float(costs_arr[idx].sum())),
            "oos_rmse": (None if oos_errors is None
                         else float(oos_errors[entry_i])),
            "ic": (None if ic_values is None else float(ic_values[entry_i])),
            "df": (None if df_values is None else int(df_values[entry_i])),
            "design": d,
            "power_curve": power_curve,
        })
    return menu


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
        self.gap_limit: Optional[float] = config.gap_limit
        self.time_limit: Optional[float] = config.time_limit
        self.relaxed_max_iter: int = config.relaxed_max_iter
        self.relaxed_decay: float = config.relaxed_decay
        self.display_graph: bool = config.display_graph
        self.verbose: bool = config.verbose
        self.costs = config.costs
        self.budget = config.budget
        self.top_K: int = config.top_K
        self.arm: Optional[str] = config.arm
        self.power_weight: float = config.power_weight
        self.fit_weight: float = config.fit_weight
        self.max_shortlist: int = config.max_shortlist
        self.holdout_frac: Optional[float] = config.holdout_frac
        # Resolve the design-selection rule (None infers from holdout_frac).
        self.selection: str = config.selection or (
            "holdout" if config.holdout_frac is not None else "in_sample"
        )
        # Design restrictions (geography / clustering / size / forcing).
        self.to_be_treated = config.to_be_treated
        self.not_to_be_treated = config.not_to_be_treated
        self.cluster_col = config.cluster_col
        self.adjacency = config.adjacency
        self.spillover_threshold = config.spillover_threshold
        self.stratum_col = config.stratum_col
        self.min_per_stratum = config.min_per_stratum
        self.max_per_stratum = config.max_per_stratum
        self.size_col = config.size_col
        self.min_size = config.min_size
        self.max_size = config.max_size
        self.donor_region_col = config.donor_region_col
        self.exclude_bordering_donors = config.exclude_bordering_donors
        self.donor_exclusion = config.donor_exclusion

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
        restrictions = self._build_restrictions(df, inputs)
        return self._fit_mip(inputs, restrictions)

    def _build_restrictions(self, df, inputs):
        """Translate restriction config into an index-level bundle (or None).

        Also runs the cheap feasibility checks that would otherwise surface as an
        opaque solver ``INFEASIBLE``: forced count and treatable-pool size vs K.
        """
        restrictions = build_restrictions(
            df, self.unitid, inputs.unit_index,
            to_be_treated=self.to_be_treated,
            not_to_be_treated=self.not_to_be_treated,
            cluster_col=self.cluster_col, adjacency=self.adjacency,
            spillover_threshold=self.spillover_threshold,
            stratum_col=self.stratum_col,
            min_per_stratum=self.min_per_stratum,
            max_per_stratum=self.max_per_stratum,
            size_col=self.size_col, min_size=self.min_size, max_size=self.max_size,
            donor_region_col=self.donor_region_col,
            exclude_bordering_donors=self.exclude_bordering_donors,
            donor_exclusion=self.donor_exclusion,
        )
        if restrictions.is_empty:
            return None
        if self.K is not None:
            n_units = int(inputs.Y_pre.shape[1])
            treatable = n_units - len(restrictions.forbidden)
            if treatable < self.K:
                raise MlsynthConfigError(
                    f"design restrictions leave only {treatable} treatable "
                    f"unit(s), fewer than K ({self.K}); relax not_to_be_treated "
                    "or the size band."
                )
        return restrictions

    # ------------------------------------------------------------------
    # MIP path
    # ------------------------------------------------------------------

    @staticmethod
    def _require_feasible_pool(designs):
        """Raise a translated error when the candidate pool is empty.

        An empty pool means every MIP solve was infeasible -- almost always
        because the active restrictions are jointly unsatisfiable for the
        requested ``K``. Without this guard the downstream ``designs[0]`` would
        leak a bare ``IndexError`` ("list index out of range").
        """
        if not designs:
            raise MlsynthEstimationError(
                "SYNDES found no feasible design: the active restrictions "
                "(forced/forbidden units, spillover/adjacency conflict, stratum "
                "quotas, donor-pool rules) are jointly unsatisfiable for the "
                "requested K. Relax the restrictions or reduce K."
            )

    def _fit_mip(self, inputs, restrictions=None) -> SYNDESResults:
        mode_internal = _MODE_TO_INTERNAL[self.mode_public]

        solve_kw = dict(
            Y=inputs.Y_pre, K=self.K, mode=mode_internal, lam=self.lam,
            solver=self.solver, verbose=self.verbose,
            unit_index=inputs.unit_index, costs=self.costs, budget=self.budget,
            gap_limit=self.gap_limit, time_limit=self.time_limit,
            restrictions=restrictions,
        )
        pool = None
        if self.selection == "holdout":
            # Holdout selection: learn the candidate pool on the leading
            # 1 - holdout_frac of the pre-period, then pick the design with the
            # smallest out-of-sample contrast error on the held-out tail. The
            # returned pool is ranked by OOS error (rank-1 is the winner).
            base_kw = {k: v for k, v in solve_kw.items() if k != "Y"}
            ranked, oos_errors = select_by_holdout(
                inputs.Y_pre, holdout_frac=self.holdout_frac,
                top_K=self.top_K, **base_kw,
            )
            self._require_feasible_pool(ranked)
            design = ranked[0]
            pool = _syndes_pool_menu(ranked, inputs, self.costs, self.alpha,
                                     oos_errors=oos_errors)
        elif self.selection == "ic":
            # Information-criterion selection: solve the pool on the whole
            # pre-period, then re-rank by IC = SSR_pre + 2 sigma^2 df (no data
            # split). The returned pool is ranked by IC (rank-1 is the winner).
            pool_designs = solve_synthetic_design_pool(top_K=self.top_K, **solve_kw)
            self._require_feasible_pool(pool_designs)
            ranked, ic_values, df_values, _sigma2 = select_by_ic(
                pool_designs, inputs.Y_pre,
            )
            design = ranked[0]
            pool = _syndes_pool_menu(ranked, inputs, self.costs, self.alpha,
                                     ic_values=ic_values, df_values=df_values)
        elif self.top_K and self.top_K > 1:
            # Solution pool: top-K distinct designs by no-good cuts. The rank-1
            # design IS the single-solve optimum, so reuse it (no double solve).
            pool_designs = solve_synthetic_design_pool(top_K=self.top_K, **solve_kw)
            self._require_feasible_pool(pool_designs)
            design = pool_designs[0]
            pool = _syndes_pool_menu(pool_designs, inputs, self.costs, self.alpha)
        else:
            design = solve_synthetic_design(**solve_kw)

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

        recommendation = (
            recommend_syndes(
                pool, power_weight=self.power_weight, fit_weight=self.fit_weight,
                max_shortlist=self.max_shortlist,
            )
            if pool else None
        )
        results = SYNDESResults(
            design=design, inputs=inputs, inference=inference,
            post_fit=post_fit, pool=pool, recommendation=recommendation,
        )
        results = replace(
            results, power_curve=_syndes_power_curve(results, self.alpha)
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

        results = replace(relaxed, inputs=inputs, inference=inference,
                          post_fit=post_fit)
        return replace(
            results, power_curve=_syndes_power_curve(results, self.alpha)
        )
