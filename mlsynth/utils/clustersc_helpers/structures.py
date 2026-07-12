"""Result structures for the Cluster-based Synthetic Control (CLUSTERSC) estimator.

The inputs and per-method fits are frozen dataclasses; the top-level
``CLUSTERSCResults`` is a Pydantic :class:`~mlsynth.config_models.EffectResult`
(the two-family result contract), carrying the dataclass layers as typed fields.

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
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


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
    outcome_name : str
        Name of the outcome column, used for axis labelling.
    time_name : str
        Name of the time column, used for axis labelling.
    """

    treated_outcome: np.ndarray
    donor_outcomes: np.ndarray
    donor_names: np.ndarray
    treated_unit_name: Any
    T: int
    T0: int
    time_labels: np.ndarray
    outcome_name: str = ""
    time_name: str = ""

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
    cft : object, optional
        Full :class:`mlsynth.utils.clustersc_helpers.rpca.inference.CFTInference`
        object when Cattaneo-Feng-Titiunik (2021) prediction
        intervals were computed (RPCA-SC only, opt-in via
        ``CLUSTERSCConfig.compute_cft_pi``). Carries per-period and
        ATT prediction intervals.
    scpi : object, optional
        Full :class:`mlsynth.utils.clustersc_helpers.scpi_pi.ScpiPIInference`
        object when the generalized scpi (Cattaneo-Feng-Palomba-Titiunik 2025)
        prediction intervals were computed (opt-in via
        ``CLUSTERSCConfig.compute_scpi_pi``, under ``scpi_constraint``). Carries
        per-period and ATT pointwise and simultaneous prediction intervals.
    """

    method: str
    alpha: float
    att: float = float("nan")
    credible_interval: Tuple[float, float] = (float("nan"), float("nan"))
    shen: Optional[Any] = None
    cft: Optional[Any] = None
    scpi: Optional[Any] = None


class CLUSTERSCResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.CLUSTERSC.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    it populates the standardized sub-models (``effects``, ``time_series``,
    ``weights``, ``inference``, ``fit_diagnostics``, ``method_details``) from the
    *primary* variant, so the flat accessors ``att`` / ``counterfactual`` /
    ``gap`` / ``att_ci`` / ``donor_weights`` / ``pre_rmse`` resolve through the
    base contract. The CLUSTERSC-specific fields below carry the dispatcher's
    extra structure (both family fits side by side, and the rich inference).

    Parameters
    ----------
    inputs : CLUSTERSCInputs
        Preprocessed panel.
    pcr : MethodFit or None
        PCR-RSC fit (populated when ``method in {"pcr", "both"}``).
    rpca : MethodFit or None
        RPCA-SC fit (populated when ``method in {"rpca", "both"}``).
    selected_variant : str
        Which fit drives the standardized sub-models / flat accessors --
        ``"pcr"`` or ``"rpca"``. When ``method = "both"`` the user picks via
        ``CLUSTERSCConfig.primary``; default ``"pcr"``.
    cluster_inference : CLUSTERSCInference or None
        The rich inferential output (Bayesian credible interval, Shen et al.
        per-period/ATT CIs, or CFT prediction intervals). The scalar summary
        is mirrored into the standardized ``inference`` slot so ``res.att_ci``
        resolves.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: CLUSTERSCInputs
    pcr: Optional[MethodFit] = None
    rpca: Optional[MethodFit] = None
    selected_variant: str = "pcr"
    cluster_inference: Optional[CLUSTERSCInference] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)


# Resolve the string annotations (the module uses ``from __future__ import
# annotations``) now that all referenced dataclasses are defined.
CLUSTERSCResults.model_rebuild()
