"""Typed result containers for SparseSC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class SparseSCInputs:
    """Pre-processed panel + predictor matrices for SparseSC.

    Parameters
    ----------
    Y0 : np.ndarray
        Donor outcome matrix, shape ``(T, N)`` (rows = time, columns =
        donors), aligned with ``donor_names``.
    Y1 : np.ndarray
        Treated outcome series, shape ``(T,)``.
    X0 : np.ndarray
        Donor predictor matrix, shape ``(P, N)`` (rows = predictors,
        columns = donors), already standardized.
    X1 : np.ndarray
        Treated predictor vector, shape ``(P,)``, already standardized.
    T : int
        Total number of time periods.
    T0_total : int
        End of the full pre-treatment window (exclusive).
    T0_train : int
        End of the training block within the pre-period (exclusive).
        Validation block is ``[T0_train, T0_total)``.
    treated_unit_name : Any
        Label of the treated unit.
    donor_names : Sequence
        Donor labels in column order of ``Y0`` / ``X0``.
    predictor_names : Sequence
        Predictor labels in row order of ``X0`` / ``X1``.
    time_labels : np.ndarray
        Time labels in row order of ``Y0``.
    Ywide : Any
        Wide outcome frame preserved for plotting.
    outcome : str
        Outcome variable name.
    """

    Y0: np.ndarray
    Y1: np.ndarray
    X0: np.ndarray
    X1: np.ndarray
    T: int
    T0_total: int
    T0_train: int
    treated_unit_name: Any
    donor_names: Sequence
    predictor_names: Sequence
    time_labels: np.ndarray
    Ywide: Any
    outcome: str

    @property
    def N(self) -> int:
        """Number of donor units."""
        return self.Y0.shape[1]

    @property
    def P(self) -> int:
        """Number of predictors."""
        return self.X0.shape[0]


@dataclass(frozen=True)
class SparseSCDesign:
    """Optimization outputs of the lambda sweep.

    Parameters
    ----------
    v : np.ndarray
        Final V-weights, shape ``(P,)``. First entry is 1 (the anchor).
    w : np.ndarray
        Final donor weights, shape ``(N,)``, on the simplex.
    opt_lambda : float
        Selected L1 penalty.
    lambda_grid : np.ndarray
        Full grid of lambdas swept.
    train_loss_curve : np.ndarray
        Training loss at each grid point, length equal to
        ``lambda_grid``.
    val_mse_curve : np.ndarray
        Validation MSE at each grid point.
    v_path : np.ndarray
        Per-grid-point V-weights, shape ``(len(grid), P)``.
    """

    v: np.ndarray
    w: np.ndarray
    opt_lambda: float
    lambda_grid: np.ndarray
    train_loss_curve: np.ndarray
    val_mse_curve: np.ndarray
    v_path: np.ndarray


@dataclass(frozen=True)
class SparseSCDegeneracy:
    """How degenerate the critical point a SparseSC solve returned was.

    The outer problem is neither convex nor smooth, so ``v`` is whichever
    critical point the solve reached, and critical points here differ
    enormously in how much of the problem they see. These integers say how
    much. Nothing here re-solves anything or changes an estimate.

    ``dim_u`` is the dimension of the U-space -- the directions in which the
    objective varies smoothly. Liu and Sagastizabal (Example 9.4, pp. 322-323,
    in Bagirov et al., *Numerical Nonsmooth Optimization*) treat exactly
    ``h(x) = q(x) + ||x||_1`` with ``q`` smooth and give the bases in closed
    form: "respective bases for V(x) and U(x) are ``{e_j : x_j = 0}`` and
    ``{e_j : x_j != 0}``". So the U-space is the support of ``v`` and no
    computation is needed. Their algorithms do not transfer -- Sect. 9.8,
    p. 327 restricts VU-methods to convex functions -- only this reading.

    ``n_active_donors`` is ``|A|``, the width of the window the outer solve
    looked through. ``w*(v)`` sits on a face of the donor simplex carrying
    ``|A|`` donors, where it has ``|A| - 1`` degrees of freedom, so the
    envelope gradient carries no information about donors that are out; at
    ``|A| = 1`` it vanishes identically.

    Every count is reported with its denominator and with the threshold that
    defines it, because a count without either is not a measurement. Two
    numbers for "predictors kept" computed at different cutoffs is a defect
    this library has already shipped once.

    Parameters
    ----------
    dim_u : int
        Predictors with ``|v_p| > support_tol``; the U-space dimension.
    n_predictors : int
        ``P``. Denominator for ``dim_u``.
    n_active_donors : int
        Donors with ``|w_j| > active_tol``; ``|A|``.
    n_donors : int
        ``N``. Denominator for ``n_active_donors``.
    n_anchor_only_grid : int
        Grid points whose ``v`` collapsed to the anchor-only corner, i.e.
        every free weight at zero. There the fit matches on the anchor
        predictor alone.
    n_distinct_supports : int
        Distinct predictor supports along ``v_path``. One means the penalty
        never changed its mind across the grid; many means "which predictors
        matter" is not settled by the data.
    lambda_selected : float
        The lambda the sweep chose, ``design.opt_lambda``.
    lambda_grid_max : float
        The largest lambda the grid offered. When the selection equals it, a
        heavier penalty was never tried, so the sweep ran to the edge of its
        own search and the choice is a boundary, not an interior optimum.
    n_grid : int
        Rows of ``v_path``. Denominator for the two path counts.
    support_tol : float
        Threshold defining ``dim_u`` and both path counts.
    active_tol : float
        Threshold defining ``n_active_donors``.
    """

    dim_u: int
    n_predictors: int
    n_active_donors: int
    n_donors: int
    n_anchor_only_grid: int
    n_distinct_supports: int
    lambda_selected: float
    lambda_grid_max: float
    n_grid: int
    support_tol: float
    active_tol: float

    @property
    def anchor_only(self) -> bool:
        """The returned fit matches on the anchor predictor alone.

        A property and not a field because it *is* ``dim_u == 1``: the anchor
        is pinned at 1, so a support of size one is the corner. Storing it
        separately would create two values that can disagree.
        """
        return self.dim_u == 1

    @property
    def nothing_pruned(self) -> bool:
        """Every predictor survived: the support is the whole list.

        A property for the same reason as ``anchor_only``: it *is*
        ``dim_u == n_predictors``. On its own this is a reading and not a
        fault -- at ``lambda* = 0`` there is no penalty, so pruning nothing is
        the correct answer. It becomes a statement about the method when a
        positive penalty was selected and still removed nothing, which is what
        ``warn_if_degenerate`` tests.
        """
        return self.dim_u == self.n_predictors

    @property
    def penalty_at_grid_edge(self) -> bool:
        """The sweep chose the largest lambda it was offered.

        A heavier penalty was never tried, so the choice is a boundary of the
        search and not an interior optimum. A grid of one point, or one whose
        largest entry is zero, offers no edge to run to and is excluded.
        """
        return (self.lambda_grid_max > 0.0
                and self.lambda_selected >= self.lambda_grid_max)


@dataclass(frozen=True)
class SparseSCInference:
    """Inference results for SparseSC.

    Either the Abadie-style placebo permutation or the validation-block
    conformal inference of Chernozhukov, Wuethrich and Zhu (2021)
    adapted to the SparseSC pre/post layout. The ``method`` tag
    identifies which fields are populated.

    Parameters
    ----------
    method : str
        ``"abadie_placebo_permutation"``, ``"conformal_validation"``,
        ``"conformal_pre"``, or ``"none"``.
    p_value : float
        Two-sided p-value for ``H_0: ATT = 0``. NaN when no inference
        was run.
    att_observed : float
        Point estimate of ATT, copied here for convenience.
    ci_lower, ci_upper : float
        Lower/upper bounds of the (1 - alpha) confidence interval for
        the ATT. NaN for ``method="none"``.
    alpha : float
        Two-sided significance level used to build ``ci_*``.
    placebo_atts : np.ndarray
        Placebo ATTs, populated only when ``method`` is the placebo
        permutation. Empty array otherwise.
    n_placebo : int
        Number of placebo runs (placebo method only; 0 otherwise).
    calibration_residuals : np.ndarray
        Residuals used to build the conformity scores (conformal
        method only). Empty for the placebo method.
    pointwise_lower, pointwise_upper : np.ndarray
        Per-period pointwise band around each post-period gap from
        the (1 - alpha)-quantile of the conformity scores. Empty for
        non-conformal methods.
    """

    method: str
    p_value: float
    att_observed: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    alpha: float = float("nan")
    placebo_atts: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    n_placebo: int = 0
    calibration_residuals: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    pointwise_lower: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    pointwise_upper: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )


class SparseSCResults(BaseEstimatorResults):
    """Public ``SparseSC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the SparseSC-specific fields below it exposes the
    standardized sub-models (``effects``, ``time_series``, ``weights``,
    ``inference``, ``fit_diagnostics``, ``method_details``) and the flat
    accessors ``att`` / ``counterfactual`` / ``gap`` / ``att_ci`` /
    ``pre_rmse`` / ``donor_weights``.

    Parameters
    ----------
    inputs : SparseSCInputs
        Pre-processed panel + predictors.
    design : SparseSCDesign
        Lambda-selection results, V and W weights.
    inference_detail : SparseSCInference
        The raw placebo / conformal inference object (``method`` / ``p_value``
        / ``placebo_atts`` / ``pointwise_*`` / ...) or ``method="none"``.
        (Renamed from ``inference``; the standardized
        :class:`~mlsynth.config_models.InferenceResults` is mirrored into the
        ``inference`` slot so ``res.att_ci`` resolves.)
    predictor_weights : Dict[Any, float]
        ``{predictor_name: v_p}``.

    Notes
    -----
    The donor weights (``{donor_name: w_j}``) live in the standardized
    ``weights`` slot and are served by ``res.donor_weights``; the predictor
    weights are also mirrored into ``weights.summary_stats``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SparseSCInputs
    design: SparseSCDesign
    inference_detail: SparseSCInference
    predictor_weights: Dict[Any, float]
    scpi: Optional[Any] = None        # ScpiPIInference (simplex PI), when computed
