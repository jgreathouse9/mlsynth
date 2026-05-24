"""Frozen dataclasses for the Forward Difference-in-Differences estimator.

FDID (Li 2023, *Frontiers: A Simple Forward Difference-in-Differences
Method*, Marketing Science) builds the control group for a single treated
unit by **forward selection**: it greedily adds the donor that most
improves pre-treatment fit (R^2 between the treated unit and the running
donor average), tracks the R^2 path, and keeps the subset that maximises
it. The synthetic control is the simple average of the selected donors,
with a difference-in-differences intercept.

Two estimates are always returned side by side:

* **FDID** -- the forward-selected difference-in-differences (best donor
  subset).
* **DID** -- the textbook two-way difference-in-differences using *all*
  donors (the average of every control unit). This is the natural
  benchmark the forward search improves upon.

Both carry Li (2023) analytical standard errors. The three layers below
(inputs, per-method fit, top-level results) mirror the CLUSTERSC /
PROXIMAL container design used elsewhere in ``mlsynth``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Public method names.
FDID = "FDID"
DID = "DID"


@dataclass(frozen=True)
class FDIDInputs:
    """Preprocessed panel data for the FDID pipeline.

    Parameters
    ----------
    y : np.ndarray
        Treated-unit outcome over all ``T`` periods, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcomes, shape ``(T, n_donors)``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    post_periods : int
        Number of post-treatment periods ``T1 = T - T0``.
    T : int
        Total number of periods.
    donor_names : Sequence
        Length-``n_donors`` donor labels (column order of ``donor_matrix``).
    time_labels : np.ndarray
        Length-``T`` time labels.
    treated_unit_name : Any
        Identifier of the treated unit.
    verbose : bool
        Whether the forward-selection path is recorded step by step.
    prepped : dict
        The raw :func:`mlsynth.utils.datautils.dataprep` dictionary, kept
        so the plotter can reuse the prepared panel.
    """

    y: np.ndarray
    donor_matrix: np.ndarray
    pre_periods: int
    post_periods: int
    T: int
    donor_names: Sequence
    time_labels: np.ndarray
    treated_unit_name: Any
    verbose: bool = True
    prepped: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_donors(self) -> int:
        """Number of donor units."""
        return int(self.donor_matrix.shape[1])


@dataclass(frozen=True)
class FDIDMethodFit:
    """Single FDID/DID fit output.

    Parameters
    ----------
    name : str
        Method identifier (``"FDID"`` or ``"DID"``).
    counterfactual : np.ndarray
        Estimated counterfactual outcome path, shape ``(T,)``.
    gap : np.ndarray
        Observed treated minus counterfactual, shape ``(T,)``.
    att : float
        Mean post-treatment treatment effect.
    att_se : float
        Li (2023) analytical standard error of the ATT.
    att_percent : float
        ATT as a percentage of the post-period counterfactual mean.
    satt : float
        Standardised ATT (``att / se * sqrt(T1)``).
    pre_rmse : float
        Root-mean-squared pre-treatment fit error.
    r_squared : float
        Pre-treatment R^2 of the difference-in-differences fit.
    intercept : float
        Difference-in-differences intercept (treated minus donor
        pre-period mean).
    p_value : float
        Two-sided p-value for the ATT.
    ci : tuple of float
        ``(lower, upper)`` 95% confidence interval for the ATT.
    selected_indices : list of int
        Column indices of the donors retained (all donors for DID).
    selected_names : list
        Donor labels corresponding to ``selected_indices``.
    donor_weights : dict
        Mapping ``{donor_name: weight}`` (equal weights over the selected
        donors).
    r2_path : np.ndarray or None
        R^2 after each forward-selection step (FDID only; ``None`` for DID).
    intermediary : list or None
        Per-step diagnostics when ``verbose`` (FDID only).
    metadata : dict
        Free-form per-method diagnostics.
    """

    name: str
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    att_se: float
    att_percent: float
    satt: float
    pre_rmse: float
    r_squared: float
    intercept: float
    p_value: float
    ci: Tuple[float, float]
    selected_indices: List[int]
    selected_names: List[Any]
    donor_weights: Dict[Any, float]
    r2_path: Optional[np.ndarray] = None
    intermediary: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FDIDResults:
    """Top-level container returned by :meth:`mlsynth.FDID.fit`.

    Parameters
    ----------
    inputs : FDIDInputs
        Preprocessed panel.
    fdid : FDIDMethodFit
        Forward-selected difference-in-differences fit (primary).
    did : FDIDMethodFit
        Textbook difference-in-differences using all donors.
    selected_variant : str
        Which fit is exposed via the convenience aliases ``att``,
        ``att_se``, ``counterfactual``, ``gap``, ``donor_weights`` --
        ``"FDID"`` or ``"DID"``. Defaults to ``"FDID"``.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    inputs: FDIDInputs
    fdid: FDIDMethodFit
    did: FDIDMethodFit
    selected_variant: str = FDID
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def methods(self) -> Dict[str, FDIDMethodFit]:
        """``{method_name: fit}`` for both fits, FDID first."""
        return {FDID: self.fdid, DID: self.did}

    @property
    def _primary(self) -> FDIDMethodFit:
        return self.methods.get(self.selected_variant, self.fdid)

    @property
    def att(self) -> float:
        """ATT of the primary variant."""
        return self._primary.att

    @property
    def att_se(self) -> float:
        """ATT standard error of the primary variant."""
        return self._primary.att_se

    @property
    def counterfactual(self) -> np.ndarray:
        """Counterfactual of the primary variant."""
        return self._primary.counterfactual

    @property
    def gap(self) -> np.ndarray:
        """Gap of the primary variant."""
        return self._primary.gap

    @property
    def donor_weights(self) -> Dict[Any, float]:
        """Donor weights of the primary variant."""
        return self._primary.donor_weights

    @property
    def pre_rmse(self) -> float:
        """Pre-treatment RMSE of the primary variant."""
        return self._primary.pre_rmse

    def att_by_method(self) -> Dict[str, float]:
        """``{method: ATT}`` for both fits."""
        return {name: fit.att for name, fit in self.methods.items()}

    def se_by_method(self) -> Dict[str, float]:
        """``{method: ATT standard error}`` for both fits."""
        return {name: fit.att_se for name, fit in self.methods.items()}

    def ci_by_method(self) -> Dict[str, Tuple[float, float]]:
        """``{method: (lower, upper)}`` confidence intervals for both fits."""
        return {name: fit.ci for name, fit in self.methods.items()}
