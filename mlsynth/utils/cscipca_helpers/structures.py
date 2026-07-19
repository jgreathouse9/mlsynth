"""Frozen dataclasses and result container for the CSCIPCA estimator.

CSC-IPCA (Wang 2024) is a factor-model counterfactual imputer whose factor
loadings are a linear projection of observed covariates:

.. math::

   Y_{it}(0) = (X_{it}\\,\\Gamma)\\,F_t' + \\epsilon_{it},
   \\qquad \\Lambda_{it} = X_{it}\\,\\Gamma,

with an ``L \\times K`` mapping matrix ``\\Gamma`` and ``K`` latent factors
``F_t`` estimated by alternating least squares. The counterfactual for the
treated unit is ``\\hat Y_{it}(0) = (X_{it}\\hat\\Gamma)\\hat F_t'`` and the
effect is ``\\hat\\delta_{it} = Y_{it} - \\hat Y_{it}(0)``. The four layers
below (inputs, design, inference, results) keep that pipeline pluggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from pydantic import ConfigDict, Field

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class CSCIPCAInputs:
    """Preprocessed panel data for CSC-IPCA estimation.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Outcome series for the treated unit, shape ``(T,)``.
    control_outcomes : np.ndarray
        Outcome matrix for the ``N_co`` control units, shape ``(T, N_co)``.
    treated_covariates : np.ndarray
        Covariate cube for the treated unit, shape ``(T, L)``.
    control_covariates : np.ndarray
        Covariate cube for the control units, shape ``(N_co, T, L)``.
    covariate_names : tuple of str
        Column names of the covariates, length ``L``.
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
    treated_covariates: np.ndarray
    control_covariates: np.ndarray
    covariate_names: tuple
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
    def L(self) -> int:
        """Number of covariates."""
        return int(self.treated_covariates.shape[1])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class CSCIPCADesign:
    """Estimated CSC-IPCA design.

    Parameters
    ----------
    n_factors : int
        Number of latent common factors ``K``.
    gamma : np.ndarray
        Normalized mapping matrix ``hat Gamma`` for the treated unit, shape
        ``(L, K)`` (in the Connor-Korajczyk / Bai-Ng normalization
        ``Gamma'Gamma = I_K``, ``FF'/T`` diagonal).
    factors : np.ndarray
        Normalized latent factors ``hat F``, shape ``(K, T)``.
    counterfactual : np.ndarray
        Imputed untreated path ``hat Y_t(0)``, shape ``(T,)``.
    gap : np.ndarray
        Effect path ``Y_t - hat Y_t(0)`` over every period, shape ``(T,)``.
        Pre-period values are the in-sample fit residual (a placebo check).
    tau : np.ndarray
        Post-treatment effect path, shape ``(n_post,)``.
    att : float
        Mean post-treatment effect.
    n_iter : int
        Number of ALS iterations to convergence on the control fit.
    converged : bool
        Whether the control ALS met ``tol`` before ``max_iter``.
    pre_rmse : float
        Root-mean-square pre-treatment fit residual for the treated unit.
    """

    n_factors: int
    gamma: np.ndarray
    factors: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray
    tau: np.ndarray
    att: float
    n_iter: int
    converged: bool
    pre_rmse: float


@dataclass(frozen=True)
class CSCIPCAInference:
    """Moving-block conformal inference for the CSC-IPCA effect path.

    Follows the sharp-null / block-permutation conformal procedure of
    Chernozhukov, Wuthrich & Zhu (2021): for each post-period a grid of
    candidate effects is tested, the treated series is adjusted under each
    null, the model is re-estimated, and the null is retained in the band when
    its block-permutation p-value exceeds ``alpha``.

    Parameters
    ----------
    alpha : float
        Two-sided significance level.
    tau : np.ndarray
        Per-period point effect, shape ``(n_post,)``.
    ci_lower_t, ci_upper_t : np.ndarray
        Per-period conformal band bounds, shape ``(n_post,)``.
    p_value_zero_t : np.ndarray
        Per-period block-permutation p-value of the sharp null ``H_0: tau=0``,
        shape ``(n_post,)``.
    att : float
        Mean post-treatment effect.
    att_p_value : float
        Block-permutation p-value of ``H_0: ATT = 0`` (constant-effect null).
    att_lower, att_upper : float
        Conformal band for the ATT (constant-effect inversion).
    """

    alpha: float

    tau: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    ci_lower_t: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    ci_upper_t: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    p_value_zero_t: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))

    att: float = float("nan")
    att_p_value: float = float("nan")
    att_lower: float = float("nan")
    att_upper: float = float("nan")


class CSCIPCAResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.CSCIPCA.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational
    report): it populates the standardized sub-models so the flat accessors
    (``att`` / ``att_ci`` / ``counterfactual`` / ``gap`` / ``pre_rmse``)
    resolve through the base contract. CSC-IPCA is a factor-model
    counterfactual, so it carries no donor weights; the full conformal
    inference lives on ``inference_detail`` (the standardized ``inference``
    slot mirrors the ATT-level band).

    Parameters
    ----------
    inputs : CSCIPCAInputs
        Preprocessed panel.
    design : CSCIPCADesign
        Estimated CSC-IPCA design.
    inference_detail : CSCIPCAInference
        Full conformal inference output (empty when ``inference=False``).
    metadata : dict
        Free-form pipeline diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: CSCIPCAInputs
    design: CSCIPCADesign
    inference_detail: CSCIPCAInference
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        """Number of latent factors used."""
        return self.design.n_factors


# Resolve forward references (module uses ``from __future__ import annotations``).
CSCIPCAResults.model_rebuild()
