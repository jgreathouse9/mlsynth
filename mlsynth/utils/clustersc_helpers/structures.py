"""Frozen dataclasses for the Cluster-based Synthetic Control (CLUSTERSC) estimator.

CLUSTERSC bundles two complementary robust synthetic-control families:

* **PCR-RSC** -- Robust Synthetic Control via Principal Component
  Regression. Combines SVD-based donor clustering (Amjad, Shah and
  Shen 2018) with PCR weight estimation (Agarwal, Shah, Shen and Song
  2021). Frequentist mode uses an SCM-style QP via :mod:`cvxpy`;
  Bayesian mode uses the conjugate-prior posterior of Bayani (2022,
  CUNY dissertation Chapter 1) with credible intervals.

* **RPCA-SC** -- Robust PCA Synthetic Control. Applies functional-PCA
  clustering on the pre-treatment outcomes, then a robust low-rank
  decomposition (PCP -- Candes, Li, Ma, Wright 2011; or HQF -- Wang,
  Li, So and Liu 2023) to denoise the donor panel before fitting
  simplex SCM weights.

Both methods can run in parallel via ``method = "both"`` so the user
can compare their estimates side by side.

The four layers below (inputs, per-method fit, optional Bayesian
inference, top-level results) keep the pipeline pluggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CLUSTERSCInputs:
    """Preprocessed panel data for CLUSTERSC.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcome matrix, shape ``(T, J)``.
    donor_names : np.ndarray
        Length-``J`` labels of the donor units.
    treated_unit_name : Any
        Label of the treated unit.
    T : int
        Total number of panel periods.
    T0 : int
        Number of pre-treatment periods.
    time_labels : np.ndarray
        Length-``T`` labels of the time periods.
    """

    treated_outcome: np.ndarray
    donor_outcomes: np.ndarray
    donor_names: np.ndarray
    treated_unit_name: Any
    T: int
    T0: int
    time_labels: np.ndarray

    @property
    def J(self) -> int:
        """Number of donor units."""
        return int(self.donor_outcomes.shape[1])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class MethodFit:
    """Single-method (PCR or RPCA) fit output.

    Parameters
    ----------
    name : str
        Method identifier (``"pcr_frequentist"``, ``"pcr_bayesian"``,
        or ``"rpca"``).
    counterfactual : np.ndarray
        Synthetic-control imputation of the treated outcome at every
        period, shape ``(T,)``.
    gap : np.ndarray
        Observed treated minus counterfactual, shape ``(T,)``.
    att : float
        Mean post-treatment gap.
    pre_rmse : float
        Root-mean-squared pre-treatment fit error.
    donor_weights : dict
        Mapping ``{donor_name: weight}`` with non-trivial weights only.
    selected_donors : np.ndarray
        Subset of ``donor_names`` retained after clustering (PCR) or
        cluster selection (RPCA). Equal to all donors when no
        clustering is applied.
    metadata : dict
        Free-form per-method diagnostics (e.g. cluster-id assignment,
        rank used by RPCA, posterior credible bounds for Bayesian PCR).
    """

    name: str
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    pre_rmse: float
    donor_weights: Dict[Any, float]
    selected_donors: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CLUSTERSCInference:
    """Optional inferential output.

    Two inference families are supported on the PCR-SC fit:

    * **Bayesian credible band** -- populated when ``estimator =
      "bayesian"``. Reports a posterior credible interval for the
      ATT (Bayani 2022 Ch. 1).
    * **Shen-Ding-Sekhon-Yu (2023) frequentist CIs** -- populated
      for the frequentist OLS PCR path. Reports per-period and ATT
      CIs under three sources of randomness (HZ / VT / DR) and one
      of three variance estimators (homoskedastic / jackknife /
      HRK).

    Parameters
    ----------
    method : str
        ``"bayesian_credible"`` for the Bayesian path,
        ``"shen_<variance>"`` (e.g. ``"shen_homoskedastic"``) for the
        Shen et al. path, or ``"none"`` if no inference was run.
    alpha : float
        Two-sided significance level (e.g. 0.05 -> 95% interval).
    att : float
        Mean post-treatment gap for the primary variant.
    credible_interval : tuple of float
        ``(lower, upper)`` posterior credible interval for the ATT
        (Bayesian path only; ``(nan, nan)`` otherwise).
    shen : object, optional
        Full :class:`mlsynth.utils.clustersc_helpers.pcr.inference.ShenInference`
        object when the Shen et al. CIs were computed (frequentist
        OLS PCR only). Carries per-period and ATT CIs under each
        source-of-randomness assumption.
    """

    method: str
    alpha: float
    att: float = float("nan")
    credible_interval: Tuple[float, float] = (float("nan"), float("nan"))
    shen: Optional[Any] = None


@dataclass(frozen=True)
class CLUSTERSCResults:
    """Top-level container returned by :meth:`mlsynth.CLUSTERSC.fit`.

    Parameters
    ----------
    inputs : CLUSTERSCInputs
        Preprocessed panel.
    pcr : MethodFit or None
        PCR-RSC fit (populated when ``method in {"pcr", "both"}``).
    rpca : MethodFit or None
        RPCA-SC fit (populated when ``method in {"rpca", "both"}``).
    inference : CLUSTERSCInference
        Optional inferential output (Bayesian credible interval for
        the PCR variant when ``estimator = "bayesian"``).
    selected_variant : str
        Which fit is exposed via the convenience aliases ``att``,
        ``counterfactual``, ``gap`` -- ``"pcr"`` or ``"rpca"``. When
        ``method = "both"`` the user picks via
        ``CLUSTERSCConfig.primary``; default ``"pcr"``.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    inputs: CLUSTERSCInputs
    pcr: Optional[MethodFit]
    rpca: Optional[MethodFit]
    inference: CLUSTERSCInference
    selected_variant: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def _primary(self) -> Optional[MethodFit]:
        return self.pcr if self.selected_variant == "pcr" else self.rpca

    @property
    def att(self) -> float:
        """ATT of the primary variant."""
        fit = self._primary
        return float("nan") if fit is None else fit.att

    @property
    def counterfactual(self) -> np.ndarray:
        """Counterfactual of the primary variant."""
        fit = self._primary
        return (
            np.full(self.inputs.T, np.nan)
            if fit is None else fit.counterfactual
        )

    @property
    def gap(self) -> np.ndarray:
        """Gap of the primary variant."""
        fit = self._primary
        return (
            np.full(self.inputs.T, np.nan)
            if fit is None else fit.gap
        )

    @property
    def donor_weights(self) -> Dict[Any, float]:
        """Donor weights of the primary variant."""
        fit = self._primary
        return {} if fit is None else fit.donor_weights

    @property
    def pre_rmse(self) -> float:
        """Pre-treatment RMSE of the primary variant."""
        fit = self._primary
        return float("nan") if fit is None else fit.pre_rmse
