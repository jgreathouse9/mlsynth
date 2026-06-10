"""Matching and Synthetic Control (MASC) estimator.

A thin NumPy-first orchestration over :mod:`mlsynth.utils.masc_helpers`.
MASC of Kellogg, Mogstad, Pouliot & Torgovitsky (2021) combines a
nearest-neighbour matching weight vector with the standard SC simplex
weight vector,

.. math::

   \\boldsymbol{\\omega}_{\\mathrm{MASC}}
       = \\varphi\\,\\boldsymbol{\\omega}_{\\mathrm{match}}
       + (1-\\varphi)\\,\\boldsymbol{\\omega}_{\\mathrm{SC}},

with the number of neighbours :math:`m` and the model-averaging weight
:math:`\\varphi` chosen jointly by rolling-origin cross-validation. The
CV-optimal :math:`\\varphi` admits a closed-form solution at each
candidate :math:`m` (Kellogg et al. 2021, eq. 15), so the joint search
reduces to a one-dimensional sweep over :math:`m`.

When ``covariates`` are supplied the SC step runs the bilevel solver of
Malo, Eskelinen, Zhou & Kuosmanen (2024) jointly over predictor weights
:math:`\\mathbf{V}` and donor weights :math:`\\mathbf{W}`; without
covariates the SC step is the canonical outcome-paths simplex fit.

References
----------
Kellogg, M., Mogstad, M., Pouliot, G., & Torgovitsky, A. (2021).
*Combining Matching and Synthetic Control to Trade Off Biases from
Extrapolation and Interpolation.* Journal of the American Statistical
Association, 116(536), 1804-1816.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from ..config_models import MASCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.masc_helpers import (
    MASCResults,
    prepare_masc_inputs,
    run_masc,
)
from ..utils.masc_helpers.plotter import plot_masc


class MASC:
    """Matching and Synthetic Control estimator.

    Parameters
    ----------
    config : MASCConfig or dict
        Validated configuration. In addition to the common fields
        (``df``, ``outcome``, ``treat``, ``unitid``, ``time``,
        ``display_graphs``, ``save``, colours), MASC reads:

        * ``covariates`` / ``covariate_windows`` -- optional predictor
          columns and their aggregation windows (matches the Abadie
          ``synth()`` predictor specification, with per-fold aggregation
          inside CV).
        * ``m_grid`` -- candidate nearest-neighbour counts (defaults to
          ``1..J``).
        * ``min_preperiods`` and ``set_f`` -- mutually exclusive CV-fold
          specifications (defaults to
          ``ceil(treatment_period / 2)..(treatment_period - 2)`` per the
          R reference).
        * ``forecast_minlength`` and ``forecast_maxlength`` -- forecast
          horizon per fold.
        * ``solver`` -- cvxpy solver for the SC QP (CLARABEL by default).
    """

    def __init__(self, config: Union[MASCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MASCConfig(**config)
            except Exception as exc:
                raise MlsynthConfigError(
                    f"Invalid MASC configuration: {exc}"
                ) from exc
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color = config.treated_color

    def fit(self) -> MASCResults:
        """Run MASC end to end and return :class:`MASCResults`.

        Raises
        ------
        MlsynthDataError
            If the input panel violates MASC's identification
            requirements (single treated unit, balanced panel, at least
            two pre-treatment periods).
        MlsynthEstimationError
            If the SC or CV optimisation steps fail at runtime.
        MlsynthPlottingError
            If plotting raises when ``display_graphs=True``.
        """
        try:
            balance(self.df, self.unitid, self.time)
        except Exception as exc:
            raise MlsynthDataError(
                f"MASC: panel failed the balance / structure check: {exc}"
            ) from exc

        try:
            inputs = prepare_masc_inputs(
                self.df,
                outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
                covariates=self.config.covariates,
                covariate_windows=self.config.covariate_windows,
            )
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"MASC: failed to prepare inputs: {exc}"
            ) from exc

        try:
            fit = run_masc(
                inputs,
                m_grid=self.config.m_grid,
                min_preperiods=self.config.min_preperiods,
                set_f=self.config.set_f,
                fold_weights=self.config.fold_weights,
                forecast_minlength=self.config.forecast_minlength,
                forecast_maxlength=self.config.forecast_maxlength,
                solver=self.config.solver,
                sc_backend=self.config.sc_backend,
                match_on=self.config.match_on,
            )
        except (MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"MASC: estimation pipeline failed: {exc}"
            ) from exc

        results = MASCResults(inputs=inputs, fit=fit)
        if self.display_graphs:
            try:
                plot_masc(
                    results,
                    outcome=self.outcome,
                    time=self.time,
                    treated_color=self.treated_color,
                    counterfactual_color=(
                        self.counterfactual_color
                        if isinstance(self.counterfactual_color, str)
                        else self.counterfactual_color[0]
                    ),
                    save=self.save,
                )
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"MASC: plotting failed: {exc}"
                ) from exc
        return results
