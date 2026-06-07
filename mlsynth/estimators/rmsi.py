"""RMSI: Robust Matrix estimation with Side Information (Agarwal, Choi & Yuan 2026).

Agarwal, A., Choi, J. & Yuan, M. (2026). *"Robust Matrix Estimation with Side
Information."* arXiv:2603.24833.

RMSI estimates a panel's untreated-outcome matrix while **exploiting covariates
on both margins** -- unit-level (row) characteristics :math:`X` and time-level
(column) characteristics :math:`Z`. It decomposes the target matrix into four
complementary pieces,

.. math::

   M = \\underbrace{G_1(X) Q_1(Z)^\\top}_{M_1\\text{: both}}
     + \\underbrace{G_2(X) V_1^\\top}_{M_2\\text{: row-driven}}
     + \\underbrace{W_1 Q_2(Z)^\\top}_{M_3\\text{: column-driven}}
     + \\underbrace{W_2 V_2^\\top}_{M_4\\text{: residual low-rank}},

and estimates each by **sieve projection plus nuclear-norm penalization**: with
projectors :math:`P_X, P_Z` onto polynomial sieve bases of the covariates, the
component explained by both margins is :math:`\\widehat M_1 = P_X Y P_Z`, and the
remaining three are singular-value soft-thresholds of the corresponding
projected residuals (the penalised least squares has that closed form). Because
the pieces are recovered separately and summed, the method is **robust**: it
accommodates nonlinear covariate effects, parts explained by only one margin,
and a part explained by neither -- and degrades gracefully when the covariates
are uninformative (it then reduces to a de-meaned low-rank completion).

For causal panel data the treated cells form a missing block. RMSI applies the
estimator to the fully observed "tall" submatrix (all units, pre-treatment
periods) and "wide" submatrix (control units, all periods) and recombines their
singular subspaces to impute the missing treated counterfactual; the ATT is the
observed minus the imputed outcome over the treated cells. This estimator
targets the **block** (common adoption time) causal setting.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import RMSIConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.rmsi_helpers.pipeline import run_rmsi
from ..utils.rmsi_helpers.setup import prepare_rmsi_inputs
from ..utils.rmsi_helpers.structures import RMSIResults


class RMSI:
    """Robust Matrix estimation with Side Information.

    Parameters
    ----------
    config : RMSIConfig or dict
        Configuration object. See :class:`mlsynth.config_models.RMSIConfig`.
    """

    def __init__(self, config: Union[RMSIConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = RMSIConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid RMSI configuration: {exc}"
                ) from exc
        self.config: RMSIConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.unit_covariates = config.unit_covariates
        self.time_covariates = config.time_covariates
        self.sieve_order: int = config.sieve_order
        self.rank = config.rank
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> RMSIResults:
        """Run RMSI and return :class:`RMSIResults`."""
        try:
            inputs = prepare_rmsi_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
                unit_covariates=self.unit_covariates,
                time_covariates=self.time_covariates,
            )
            results = run_rmsi(inputs, J=self.sieve_order, rank=self.rank)
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
                f"RMSI estimation failed: {exc}"
            ) from exc
