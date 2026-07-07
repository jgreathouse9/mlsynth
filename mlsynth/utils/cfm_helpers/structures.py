"""Frozen dataclasses and result container for the CFM estimator.

CFM implements Bai & Wang (2026), *"Causal Inference Using Factor Models"*.
Unlike a single-equation imputation estimator (which models only the
untreated potential outcome ``Y(0)`` and reports ``Y(1) - hat Y(0)``), CFM
models *both* potential outcomes within one factor structure and lets the
treated unit's factor loadings break at the intervention date. The target is
the systematic causal effect

.. math::

   \\tau^*_t = [\\lambda_1(1) - \\lambda_1(0)]' f_t + [a_1(1) - a_1(0)],

for a single treated unit over the post-period. The four layers below
(inputs, design, inference, results) keep that pipeline pluggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from pydantic import ConfigDict, Field

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class CFMInputs:
    """Preprocessed panel data for CFM estimation.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Outcome series for the treated unit, shape ``(T,)``.
    control_outcomes : np.ndarray
        Outcome matrix for the ``N_co`` control units, shape ``(T, N_co)``.
    donor_names : np.ndarray
        Labels of the control units, length ``N_co``.
    treated_unit_name : Any
        Label of the treated unit.
    T : int
        Total number of panel periods.
    T0 : int
        Number of pre-treatment periods.
    time_labels : np.ndarray
        Labels of the time periods, length ``T``.
    """

    treated_outcome: np.ndarray
    control_outcomes: np.ndarray
    donor_names: np.ndarray
    treated_unit_name: Any
    T: int
    T0: int
    time_labels: np.ndarray

    @property
    def N_co(self) -> int:
        """Number of control units."""
        return int(self.control_outcomes.shape[1])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class CFMDesign:
    """Systematic-effect design produced by the CFM fit.

    Parameters
    ----------
    n_factors : int
        Number of common factors used.
    n_factors_source : str
        ``"ER"`` / ``"GR"`` (Ahn-Horenstein 2013), ``"BaiNg"`` (Bai-Ng
        2002 information criterion), or ``"user"``.
    factors : np.ndarray
        Estimated factors ``F_hat``, shape ``(T, r)``, in the Bai
        normalization ``F'F / T = I``.
    a0, a1 : float
        Pre- and post-treatment intercepts for the treated unit.
    lambda0, lambda1 : np.ndarray
        Pre- and post-treatment factor loadings, shape ``(r,)``.
    kappa : float
        Intercept shift ``a1 - a0`` (the constant-shift / structural-break
        term; also absorbs any constant shift in the factor process).
    tau : np.ndarray
        Systematic causal effect over the post-period, shape ``(n_post,)``.
    tau_full : np.ndarray
        Systematic difference ``(a1 + lambda1'f_t) - (a0 + lambda0'f_t)``
        over every period, shape ``(T,)``. Pre-period values are a
        placebo/pre-trend diagnostic; post-period values equal ``tau``.
    counterfactual : np.ndarray
        Systematic untreated path ``a0 + lambda0'f_t``, shape ``(T,)``.
    att : float
        Mean post-treatment systematic effect.
    chow_fstat : float
        Chow F-statistic for a structural break at the treatment date.
    """

    n_factors: int
    n_factors_source: str
    factors: np.ndarray
    a0: float
    a1: float
    lambda0: np.ndarray
    lambda1: np.ndarray
    kappa: float
    tau: np.ndarray
    tau_full: np.ndarray
    counterfactual: np.ndarray
    att: float
    chow_fstat: float


@dataclass(frozen=True)
class CFMInference:
    """Asymptotic inference for the systematic causal effect.

    The variance has two asymptotically-uncorrelated components (Bai & Wang
    appendix A.2): the treated-regression component ``V_reg`` (a
    block-additive heteroskedasticity-robust sandwich) and the
    factor-estimation component ``V_f`` (from estimating the common factors
    off the control units). ``factor_variance=False`` reports ``V_reg`` only.

    Parameters
    ----------
    alpha : float
        Two-sided significance level.
    factor_variance : bool
        Whether ``V_f`` was added to the standard errors.
    att, att_se, att_lower, att_upper, att_p_value : float
        ATT point estimate, standard error, ``(1 - alpha)`` CI, and the
        two-sided z-test p-value of ``H_0: ATT = 0``.
    kappa, kappa_se, kappa_t : float
        Intercept-shift estimate, standard error, and t-statistic
        (block-additive HC1; the paper's post-treatment intercept-shift test).
    tau : np.ndarray
        Per-period systematic effect, shape ``(n_post,)``.
    se_t, ci_lower_t, ci_upper_t : np.ndarray
        Per-period standard errors and CI bounds, shape ``(n_post,)``.
    """

    alpha: float
    factor_variance: bool

    att: float = float("nan")
    att_se: float = float("nan")
    att_lower: float = float("nan")
    att_upper: float = float("nan")
    att_p_value: float = float("nan")

    kappa: float = float("nan")
    kappa_se: float = float("nan")
    kappa_t: float = float("nan")

    tau: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    se_t: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    ci_lower_t: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    ci_upper_t: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))


class CFMResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.CFM.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational
    report): it populates the standardized sub-models so the flat accessors
    (``att`` / ``att_ci`` / ``counterfactual`` / ``gap`` / ``pre_rmse``)
    resolve through the base contract. The reported ``gap`` is the
    *systematic* causal effect ``tau*`` -- not ``observed - counterfactual``
    -- which is the paper's estimand. CFM is a factor-model counterfactual,
    so it carries no donor weights; the full asymptotic inference lives on
    ``inference_detail`` (the standardized ``inference`` slot mirrors the
    ATT-level CI).

    Parameters
    ----------
    inputs : CFMInputs
        Preprocessed panel.
    design : CFMDesign
        Systematic-effect design.
    inference_detail : CFMInference
        Full asymptotic inference output.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: CFMInputs
    design: CFMDesign
    inference_detail: CFMInference
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        """Selected number of factors."""
        return self.design.n_factors


# Resolve forward references (module uses ``from __future__ import annotations``).
CFMResults.model_rebuild()
