"""Structured containers for the Two-Step Synthetic Control (TSSC) estimator.

Implements:

    Li, K. T., & Shankar, V. (2023). "A Two-Step Synthetic Control Approach
    for Estimating Causal Effects of Marketing Events." Management Science.
    https://doi.org/10.1287/mnsc.2023.4878

Notation (following Li & Shankar, 2023):

    Treated unit ``j = 1``; control units ``j = 2, ..., N``. The regression
    is ``y_{1t} = x_t' beta + e_{1t}`` with ``x_t = (1, y_{2t}, ..., y_{Nt})'``,
    so ``beta_1`` is the intercept and ``beta_2, ..., beta_N`` are the donor
    slope coefficients. Pre-period length ``T_1``, post-period length ``T_2``.

The class of SC methods (Table 1) is indexed by which restrictions are
imposed on ``beta``:

    SC    : (1) beta_1 = 0, (2) sum_{j>=2} beta_j = 1, (3) beta_j >= 0
    MSCa  : (2) and (3)                 -- weights sum to one, with intercept
    MSCb  : (1) and (3)                 -- no adding-up, zero intercept
    MSCc  : (3) only                    -- most flexible benchmark
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict, model_validator

from ...config_models import BaseEstimatorResults


# Public (paper) method names.
SC = "SC"
MSCA = "MSCa"
MSCB = "MSCb"
MSCC = "MSCc"
METHODS = (SC, MSCA, MSCB, MSCC)


@dataclass(frozen=True)
class TSSCInputs:
    """Preprocessed panel data for TSSC.

    Parameters
    ----------
    y : np.ndarray
        Treated-unit outcome over all ``T`` periods, shape ``(T,)``.
    donor_matrix : np.ndarray
        Control-unit outcomes, shape ``(T, N-1)`` (the ``N-1`` donors;
        the intercept column is added internally by the variants that use
        it).
    donor_names : Sequence
        Length ``N-1`` donor labels.
    T0 : int
        Number of pre-treatment periods (``T_1``).
    T2 : int
        Number of post-treatment periods.
    T : int
        Total number of periods (``T_1 + T_2``).
    time_labels : np.ndarray
        Length-``T`` time labels.
    treated_unit_name : str
    """

    y: np.ndarray
    donor_matrix: np.ndarray
    donor_names: Sequence
    T0: int
    T2: int
    T: int
    time_labels: np.ndarray
    treated_unit_name: str

    @property
    def n_donors(self) -> int:
        """Number of control units ``N - 1``."""
        return self.donor_matrix.shape[1]


@dataclass(frozen=True)
class TSSCRestrictionTest:
    """Outcome of one Step-1 subsampling restriction test.

    Parameters
    ----------
    name : str
        ``"joint"`` (H0: weights sum to one AND zero intercept),
        ``"sum_to_one"`` (H0a), or ``"zero_intercept"`` (H0b).
    statistic : float
        Feasible test statistic on the full pre-treatment sample
        (``S_hat_{T1}`` for the joint test; ``(sqrt(T1) d_hat_s)^2`` for a
        single restriction).
    ci_lower, ci_upper : float
        The ``alpha/2`` and ``1 - alpha/2`` quantiles of the subsampling
        distribution -- the estimated ``(1 - alpha)`` acceptance region
        under H0 (Proposition 3.2).
    rejected : bool
        True if ``statistic`` falls outside ``[ci_lower, ci_upper]``.
    """

    name: str
    statistic: float
    ci_lower: float
    ci_upper: float
    rejected: bool


@dataclass(frozen=True)
class TSSCSelection:
    """Record of the Step-1 model-selection procedure.

    Parameters
    ----------
    recommended : str
        Selected method: ``"SC"``, ``"MSCa"``, ``"MSCb"``, or ``"MSCc"``.
    tests : dict
        ``test_name -> TSSCRestrictionTest`` for each test actually run
        (the decision tree short-circuits, so fewer than three may appear).
    alpha : float
        Two-sided significance level used for the acceptance regions.
    subsample_size : int
        Subsample size ``m`` (Step i of the subsampling procedure).
    n_subsamples : int
        Number of subsampling replications ``B``.
    mscc_beta : np.ndarray
        Full-sample MSC(c) coefficient vector ``beta_hat_{MSC,T1}`` of
        length ``N`` (intercept first), the benchmark all tests build on.
    decision_path : tuple of str
        Human-readable trace of the decision tree.
    """

    recommended: str
    tests: Dict[str, TSSCRestrictionTest]
    alpha: float
    subsample_size: int
    n_subsamples: int
    mscc_beta: np.ndarray
    decision_path: Tuple[str, ...]


@dataclass(frozen=True)
class TSSCVariantFit:
    """Fitted result for one SC-class variant.

    Parameters
    ----------
    method : str
        ``"SC"``, ``"MSCa"``, ``"MSCb"``, or ``"MSCc"``.
    weights : np.ndarray
        Raw coefficient vector returned by the solver. Length ``N`` for
        the intercept variants (MSCa/MSCc; intercept first) and ``N-1``
        for the no-intercept variants (SC/MSCb).
    intercept : float or None
        The fitted intercept (``None`` for SC and MSCb, which impose
        ``beta_1 = 0``).
    donor_weights : dict
        ``donor_label -> weight`` for donors with non-negligible weight.
    counterfactual : np.ndarray
        Length-``T`` synthetic control path.
    gap : np.ndarray
        Length-``T`` gap ``y - counterfactual``.
    att : float
        Mean post-period gap.
    att_ci : tuple of float
        ``(lower, upper)`` bootstrap confidence interval for the ATT.
    rmse_pre, rmse_post : float
        Pre/post RMSE of the gap.
    r2_pre : float
        Pre-treatment R-squared of the fit.
    """

    method: str
    weights: np.ndarray
    intercept: Optional[float]
    donor_weights: dict
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    att_ci: Tuple[float, float]
    rmse_pre: float
    rmse_post: float
    r2_pre: float
    scpi: Optional[object] = None     # ScpiPIInference band, when computed


class TSSCResults(BaseEstimatorResults):
    """Public ``TSSC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational
    report): besides the TSSC-specific fields below, it exposes the
    standardized sub-models (``effects``, ``time_series``, ``weights``,
    ``inference``, ``fit_diagnostics``, ``method_details``) -- lifted from
    the recommended variant's ``summary`` -- and the flat accessors
    ``att``/``att_ci``/``counterfactual``/``gap``/``donor_weights``/
    ``pre_rmse``.

    Parameters
    ----------
    inputs : TSSCInputs
        Preprocessed panel data.
    variants : dict
        ``method_name -> TSSCVariantFit`` for all four SC-class methods.
    selection : TSSCSelection
        The Step-1 recommendation and its underlying tests.
    summary : BaseEstimatorResults, optional
        Standardized result bundle for the recommended variant.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: TSSCInputs
    variants: Dict[str, TSSCVariantFit]
    selection: TSSCSelection
    summary: Optional[BaseEstimatorResults] = None

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "TSSCResults":
        """Lift the recommended variant's standardized sub-models to the top
        level so a TSSC result exposes the same surface as every other
        EffectResult. Uses ``object.__setattr__`` because the model is
        frozen.
        """
        if self.effects is None and self.summary is not None:
            for key in (
                "effects",
                "time_series",
                "weights",
                "inference",
                "fit_diagnostics",
                "method_details",
            ):
                object.__setattr__(self, key, getattr(self.summary, key))
        return self

    @property
    def mode(self) -> str:
        return "tssc"

    @property
    def recommended_method(self) -> str:
        """Name of the method recommended by Step 1."""
        return self.selection.recommended

    @property
    def recommended(self) -> TSSCVariantFit:
        """The :class:`TSSCVariantFit` for the recommended method."""
        return self.variants[self.selection.recommended]

    @property
    def att(self) -> float:
        """ATT of the recommended method."""
        return self.recommended.att

    @property
    def att_ci(self) -> Tuple[float, float]:
        """ATT confidence interval of the recommended method."""
        return self.recommended.att_ci

    @property
    def donor_weights(self) -> dict:
        """Donor weights of the recommended method."""
        return self.recommended.donor_weights

    @property
    def scpi(self):
        """scpi prediction-interval band of the recommended method (``None``
        unless ``compute_scpi_pi`` was set)."""
        return self.recommended.scpi

    def att_by_method(self) -> Dict[str, float]:
        """``{method: ATT}`` across all four SC-class variants."""
        return {name: v.att for name, v in self.variants.items()}

    def att_ci_by_method(self) -> Dict[str, Tuple[float, float]]:
        """``{method: (ci_lower, ci_upper)}`` across all four variants."""
        return {name: v.att_ci for name, v in self.variants.items()}
