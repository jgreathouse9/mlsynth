"""Structured containers for the mlSC pipeline.

All matrices follow mlsynth's standard ``(T, N)`` orientation (rows = time,
columns = unit), matching ``datautils.dataprep``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class MLSCInputs:
    """Pre-processed data fed into the mlSC optimization and inference loops.

    Parameters
    ----------
    Y_agg_treated : np.ndarray
        Observed aggregate-level treated outcome series, length ``T``.
    X_disagg : np.ndarray
        Disaggregate-control outcome matrix of shape ``(T, M)`` where ``M``
        is the total number of disaggregate control units (sum of ``C_s``
        across all non-treated aggregates). Columns are ordered first by
        aggregate label and then by disaggregate label, with this ordering
        preserved in ``disagg_to_agg`` and ``v_population``.
    v_population : np.ndarray
        Population aggregation weights ``v_sc`` for each disaggregate control
        column, length ``M``. Within each aggregate block the entries sum to
        1.
    disagg_to_agg : np.ndarray
        Length-``M`` integer array giving the aggregate-block index for each
        disaggregate column. Aggregate blocks are indexed 0..S-1.
    agg_labels : Sequence
        Labels of the ``S`` control aggregates in block order.
    disagg_labels : Sequence
        Labels of the ``M`` disaggregate-control columns in column order.
    Y_disagg_pre_full : np.ndarray
        Pre-treatment disaggregate outcome matrix for *all* aggregates
        (including the treated one), shape ``(T0, M_full)``. Used by the
        Appendix-G variance decomposition.
    disagg_to_agg_full : np.ndarray
        Length-``M_full`` aggregate-index assignment matching
        ``Y_disagg_pre_full`` columns.
    treated_agg_idx_full : int
        Index of the treated aggregate within ``Y_disagg_pre_full`` blocks
        (used to exclude it from the variance estimate).
    T : int
        Total number of time periods.
    T0 : int
        Number of pre-treatment periods.
    treated_unit_name : str
        Label of the treated aggregate.
    time_labels : np.ndarray
        Time labels in original order, length ``T``.
    Ywide_agg : object
        Wide aggregate-level outcome frame (rows = time, columns = aggregate
        unit) preserved from ``dataprep`` for downstream plotting.
    outcome : str
        Outcome variable name.
    """

    Y_agg_treated: np.ndarray
    X_disagg: np.ndarray
    v_population: np.ndarray
    disagg_to_agg: np.ndarray
    agg_labels: Sequence
    disagg_labels: Sequence
    Y_disagg_pre_full: np.ndarray
    disagg_to_agg_full: np.ndarray
    treated_agg_idx_full: int
    T: int
    T0: int
    treated_unit_name: str
    time_labels: np.ndarray
    Ywide_agg: object
    outcome: str

    @property
    def M(self) -> int:
        """Number of disaggregate control columns."""
        return self.X_disagg.shape[1]

    @property
    def S(self) -> int:
        """Number of control aggregate units."""
        return len(self.agg_labels)


@dataclass(frozen=True)
class MLSCDesign:
    """Optimization output and selected hyperparameters.

    Parameters
    ----------
    omega : np.ndarray
        Optimal disaggregate weights, length ``M``. Satisfies ``sum(omega) = 1``
        and ``omega >= 0``.
    aggregate_weights : np.ndarray
        Implied aggregate weights ``w_s = sum_c omega_sc`` for each control
        aggregate, length ``S``.
    lambda_used : float
        Penalty value actually applied.
    sigma_eps2 : float
        Estimated noise variance from Appendix G.
    sigma_y2 : float
        Estimated outcome variance from Appendix G.
    lambda_est : str
        Selection rule that was used (``'heuristic'`` or ``'fixed'``).
    solver_status : str
        cvxpy solver status string.
    """

    omega: np.ndarray
    aggregate_weights: np.ndarray
    lambda_used: float
    sigma_eps2: float
    sigma_y2: float
    lambda_est: str
    solver_status: str


@dataclass(frozen=True)
class MLSCInference:
    """Counterfactual path and post-period gap.

    Parameters
    ----------
    counterfactual : np.ndarray
        Estimated aggregate counterfactual ``X_disagg @ omega``, length ``T``.
    gap : np.ndarray
        Observed minus counterfactual, length ``T``.
    """

    counterfactual: np.ndarray
    gap: np.ndarray


class MLSCResults(BaseEstimatorResults):
    """Public ``MLSC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the mlSC-specific fields below it exposes the standardized
    sub-models (``effects``, ``time_series``, ``weights``, ``fit_diagnostics``,
    ``method_details``) and the flat accessors ``att`` / ``counterfactual`` /
    ``gap`` / ``pre_rmse`` / ``donor_weights``. mlSC has no statistical
    inference (no SE/CI), so the ``inference`` slot is ``None``.

    Parameters
    ----------
    inputs : MLSCInputs
        Pre-processed two-level panel.
    design : MLSCDesign
        Optimization outputs and variance-decomposition diagnostics.
    paths : MLSCInference
        Counterfactual path and gap. (Renamed from ``inference`` — it carries
        the fitted series, not statistical inference, and the contract reserves
        the ``inference`` slot for :class:`~mlsynth.config_models.InferenceResults`.
        The same series are exposed flat as ``res.counterfactual`` / ``res.gap``.)
    aggregate_donor_weights : dict
        Mapping from aggregate-unit label to its implied weight
        ``w_s = sum_c omega_sc``. Convenience view of
        ``design.aggregate_weights``. (The disaggregate ``donor_weights`` live
        in the standardized ``weights`` slot, served by ``res.donor_weights``.)
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MLSCInputs
    design: MLSCDesign
    paths: MLSCInference
    aggregate_donor_weights: Dict[Any, float]
