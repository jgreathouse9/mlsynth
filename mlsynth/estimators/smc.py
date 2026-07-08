"""Synthetic Matching Control (SMC) estimator.

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.smc_helpers`.
SMC (Zhu 2023) addresses the imperfect-pre-fit problem of the simplex synthetic
control in two deterministic steps. First, *unit matching*: each donor is
rescaled to the treated unit by a univariate OLS, ``theta_j``, giving a matched
control ``theta_j x_j`` that can already extrapolate (``theta_j`` is
unconstrained). Second, *synthesis*: the matched controls are combined with
box-``[0, 1]`` weights ``w`` chosen to minimise a Mallows/Cp unbiased-risk
criterion; the combined donor coefficient is ``theta_j w_j``. The Cp penalty on
``sum(w)`` -- not an Abadie predictor-weight search -- identifies the weights, so
the estimator is reproducible.

The synthesis QP is solved exactly by an active-set box solver
(:func:`~mlsynth.utils.smc_helpers.solver.solve_box_qp`) that matches the
reference R ``solve.QP`` to machine precision. Optional ``covariates`` add
predictor rows (Algorithm 3) at equal predictor weight.

References
----------
Zhu, Rong J. B. (2023). *Synthetic Matching Control Method.* arXiv:2306.02584.
https://arxiv.org/abs/2306.02584
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from ..config_models import SMCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.smc_helpers import (
    SMCResults,
    plot_smc,
    prepare_smc_inputs,
    run_smc,
)


class SMC:
    """Synthetic Matching Control estimator.

    Parameters
    ----------
    config : SMCConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``outcome``,
        ``treat``, ``unitid``, ``time``, ``display_graphs``, ``save``, colours),
        SMC reads ``ridge`` (the Cp box-QP stabiliser) and ``covariates``
        (optional predictor columns for Algorithm 3).
    """

    def __init__(self, config: Union[SMCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SMCConfig(**config)
            except Exception as exc:
                raise MlsynthConfigError(f"Invalid SMC configuration: {exc}") from exc
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

    def fit(self) -> SMCResults:
        """Run SMC end to end and return :class:`SMCResults`.

        Returns
        -------
        SMCResults
            Container with the ATT, the counterfactual and gap paths, the
            combined donor coefficients (``theta * w``), the box weights, the
            pre-period RMSE, and the standardized sub-models.

        Raises
        ------
        MlsynthDataError
            If the panel violates SMC's requirements (single treated unit,
            balanced panel, at least two pre-treatment periods, a donor pool).
        MlsynthEstimationError
            If the weight computation fails at runtime.
        MlsynthPlottingError
            If plotting raises when ``display_graphs=True``.
        """
        try:
            balance(self.df, self.unitid, self.time)
        except Exception as exc:
            raise MlsynthDataError(
                f"SMC: panel failed the balance / structure check: {exc}"
            ) from exc

        try:
            inputs = prepare_smc_inputs(
                self.df,
                outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
                covariates=self.config.covariates,
                covariate_windows=self.config.covariate_windows,
                fit_window=self.config.fit_window,
            )
        except MlsynthDataError:
            raise
        except Exception as exc:  # pragma: no cover - defensive translation of an
            raise MlsynthDataError(  # unexpected prepare failure
                f"SMC: failed to prepare inputs: {exc}") from exc

        try:
            fit = run_smc(
                inputs, ridge=self.config.ridge,
                v_search=self.config.v_search, v_seed=self.config.v_seed,
                v_maxiter=self.config.v_maxiter, v_popsize=self.config.v_popsize,
            )
        except (MlsynthDataError, MlsynthEstimationError):  # pragma: no cover
            raise                                           # already translated
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(
                f"SMC: estimation pipeline failed: {exc}") from exc

        results = SMCResults(inputs=inputs, fit=fit)
        if self.display_graphs:
            try:
                plot_smc(
                    results,
                    outcome=self.outcome, time=self.time,
                    treated_color=self.treated_color,
                    counterfactual_color=(
                        self.counterfactual_color
                        if isinstance(self.counterfactual_color, str)
                        else self.counterfactual_color[0]
                    ),
                    save=self.save,
                )
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"SMC: plotting failed: {exc}") from exc
        return results
