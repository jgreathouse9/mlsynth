"""MCNNM: Matrix Completion with Nuclear Norm Minimization (Athey et al. 2021).

Athey, S., Bayati, M., Doudchenko, N., Imbens, G. & Khosravi, K. (2021).
*"Matrix Completion Methods for Causal Panel Data Models."* Journal of
the American Statistical Association 116(536):1716-1730.

MC-NNM estimates causal effects in panel data by treating the treated
unit/period cells as *missing* entries of the outcome matrix and imputing
them via low-rank matrix completion. It models the untreated-outcome
matrix as a low-rank component plus two-way (unit and time) fixed effects,

.. math::

   (\\widehat L, \\widehat\\Gamma, \\widehat\\Delta)
     = \\arg\\min_{L, \\Gamma, \\Delta}
       \\tfrac{1}{|\\mathcal{O}|}
       \\| P_\\mathcal{O}(Y - L - \\Gamma 1_T^\\top - 1_N \\Delta^\\top)\\|_F^2
       + \\lambda \\|L\\|_*,

regularising only the low-rank part :math:`L` via its nuclear norm (the
sum of singular values). The fixed effects are estimated explicitly and
left unregularised, which substantially improves imputation. The problem
is solved by the SOFT-IMPUTE iteration (singular-value soft-thresholding)
with the regularisation strength chosen by cross-validation over the
observed cells.

MC-NNM *nests* the unconfoundedness, synthetic-control, and
difference-in-differences estimators (paper Theorem 1): all minimise the
same objective and differ only in the restrictions/regularisation they
impose. By regularising rather than imposing hard restrictions, MC-NNM
performs well whether ``N >> T``, ``T >> N``, or ``N ~ T`` -- regimes
where the unconfoundedness or synthetic-control approaches individually
break down.

This estimator targets the block / staggered-adoption causal setting:
control units and treated units' pre-treatment periods are the observed
entries; treated post-treatment cells are imputed, and the treatment
effect is the observed outcome minus the imputed counterfactual.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import MCNNMConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.mcnnm_helpers.pipeline import run_mcnnm
from ..utils.mcnnm_helpers.setup import prepare_mcnnm_inputs
from ..utils.mcnnm_helpers.structures import MCNNMResults


class MCNNM:
    """Matrix Completion with Nuclear Norm Minimization estimator.

    Parameters
    ----------
    config : MCNNMConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.MCNNMConfig`.
    """

    def __init__(self, config: Union[MCNNMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MCNNMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid MCNNM configuration: {exc}"
                ) from exc
        self.config: MCNNMConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.estimate_unit_fe: bool = config.estimate_unit_fe
        self.estimate_time_fe: bool = config.estimate_time_fe
        self.n_lambda: int = config.n_lambda
        self.n_folds: int = config.n_folds
        self.inference: bool = config.inference
        self.alpha: float = config.alpha
        self.random_state: int = config.random_state
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> MCNNMResults:
        """Run MC-NNM and return :class:`MCNNMResults`."""
        try:
            inputs = prepare_mcnnm_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
            )
            results = run_mcnnm(
                inputs=inputs,
                est_u=self.estimate_unit_fe,
                est_v=self.estimate_time_fe,
                n_lam=self.n_lambda,
                n_folds=self.n_folds,
                inference=self.inference,
                alpha_level=self.alpha,
                random_state=self.random_state,
            )
            # Attach plotting context so result.plot() is self-contained and
            # styled from the (possibly nested) config; labels default to the
            # column names when the user has not set them.
            pc = self.config.resolved_plot()
            if pc.xlabel is None:
                pc.xlabel = self.time
            if pc.ylabel is None:
                pc.ylabel = self.outcome
            object.__setattr__(results, "plot_config", pc)
            if self.display_graphs:
                results.plot()
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"MCNNM estimation failed: {exc}"
            ) from exc
