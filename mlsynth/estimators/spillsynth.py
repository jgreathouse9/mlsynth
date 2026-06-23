"""SPILLSYNTH: spillover-aware synthetic-control estimator.

Public dispatcher that runs a spillover-adjusted SCM under the
``method`` selected in :class:`~mlsynth.config_models.SPILLSYNTHConfig`.

Currently implements ``method='cd'`` (Cao & Dowd 2023): a Ferman-Pinto
demeaned SCM is fit leave-one-out for every unit; the treatment effect
on the treated unit and the spillover effects on a user-supplied set
of *potentially-affected* control units are recovered jointly via the
closed-form formula

    gamma_hat(t) = (A' M A)^{-1} A' (I - B)' [(I - B) Y_t - a],

with ``B`` and ``a`` the leave-one-out SCM artifacts, ``A`` encoding
the spillover structure (Example 3 of the paper), and
``M = (I - B)' (I - B)``. The first entry of ``alpha = A gamma`` is the
spillover-adjusted ATT on the treated unit; the remaining entries are
the per-affected-unit spillover effects.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SPILLSYNTHConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.spillsynth_helpers.cd import run_cd
from ..utils.spillsynth_helpers.grossi import run_grossi
from ..utils.spillsynth_helpers.iscm import run_iscm
from ..utils.spillsynth_helpers.iterative import run_iterative
from ..utils.spillsynth_helpers.plotter import plot_spillsynth
from ..utils.spillsynth_helpers.sar.pipeline import run_sar
from ..utils.spillsynth_helpers.sar.setup import prepare_sar_inputs
from ..utils.spillsynth_helpers.setup import prepare_spillsynth_inputs
from ..utils.spillsynth_helpers.structures import SpillSynthResults


class SPILLSYNTH:
    """Spillover-aware synthetic control estimator.

    Parameters
    ----------
    config : SPILLSYNTHConfig or dict
        Typed configuration. See
        :class:`mlsynth.config_models.SPILLSYNTHConfig`.

    Returns
    -------
    SpillSynthResults
        A standardized ``EffectResult`` (subclass of
        :class:`~mlsynth.config_models.BaseEstimatorResults`) carrying the
        leave-one-out SCM artifacts, per-period spillover-adjusted treatment
        effect on the treated unit, per-affected-unit spillover trajectories,
        and the vanilla SCM comparison, alongside the shared sub-models
        (``effects``, ``time_series``, ``weights`` ...) every estimator exposes.
    """

    def __init__(self, config: Union[SPILLSYNTHConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SPILLSYNTHConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SPILLSYNTH configuration: {exc}"
                ) from exc

        self.config: SPILLSYNTHConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.method: str = config.method
        self.affected_units = list(config.affected_units or [])
        self.spillover_structure: str = getattr(
            config, "spillover_structure", "per_unit",
        )
        self.unit_distances = getattr(config, "unit_distances", None)
        self.weighting: str = getattr(config, "weighting", "identity")
        self.solver = config.solver
        self.covariates = list(config.covariates) if config.covariates else None
        self.covariate_windows = getattr(config, "covariate_windows", None)
        self.bilevel_solver: str = getattr(config, "bilevel_solver", "malo")
        self.bias_correct: bool = getattr(config, "bias_correct", False)
        self.iscm_intercept: bool = getattr(config, "iscm_intercept", False)
        self.n_boot: int = getattr(config, "n_boot", 0)
        self.ci_level: float = getattr(config, "ci_level", 0.90)
        self.seed: int = getattr(config, "seed", 0)
        # SAR (method='sar') fields
        self.spatial_W = getattr(config, "spatial_W", None)
        self.spatial_w = getattr(config, "spatial_w", None)
        self.p_factors: int = getattr(config, "p_factors", 1)
        self.mcmc_iter: int = getattr(config, "mcmc_iter", 6000)
        self.mcmc_burn: int = getattr(config, "mcmc_burn", 2000)
        self.step_rho: float = getattr(config, "step_rho", 0.02)
        self.mcmc_seed: int = getattr(config, "mcmc_seed", 0)
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color = config.treated_color

    def fit(self) -> SpillSynthResults:
        """Run the selected spillover-aware SCM and return a :class:`SpillSynthResults`."""

        try:
            balance(self.df, self.unitid, self.time)
            if self.method == "sar":
                inputs = prepare_sar_inputs(
                    df=self.df,
                    outcome=self.outcome,
                    treat=self.treat,
                    unitid=self.unitid,
                    time=self.time,
                    spatial_W=self.spatial_W,
                    spatial_w=self.spatial_w,
                    covariates=self.covariates,
                )
            else:
                inputs = prepare_spillsynth_inputs(
                    df=self.df,
                    outcome=self.outcome,
                    treat=self.treat,
                    unitid=self.unitid,
                    time=self.time,
                    affected_units=self.affected_units,
                    spillover_structure=self.spillover_structure,
                    unit_distances=self.unit_distances,
                    covariates=self.covariates,
                    covariate_windows=self.covariate_windows,
                )
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"SPILLSYNTH: failed to prepare panel ({exc})."
            ) from exc

        try:
            if self.method == "cd":
                fit = run_cd(
                    inputs, solver=self.solver, weighting=self.weighting,
                )
                results = SpillSynthResults(
                    inputs=inputs, method="cd", cd=fit,
                )
            elif self.method == "iscm":
                fit = run_iscm(inputs, bilevel_solver=self.bilevel_solver,
                               bias_correct=self.bias_correct,
                               intercept=self.iscm_intercept)
                results = SpillSynthResults(
                    inputs=inputs, method="iscm", iscm=fit,
                )
            elif self.method == "grossi":
                # The paper's estimator is the penalized SCG; use it unless the
                # user explicitly picked another backend.
                gsolver = "penalized" if self.bilevel_solver == "malo" else self.bilevel_solver
                fit = run_grossi(inputs, bilevel_solver=gsolver,
                                 bias_correct=self.bias_correct,
                                 n_boot=self.n_boot, ci_level=self.ci_level,
                                 seed=self.seed)
                results = SpillSynthResults(
                    inputs=inputs, method="grossi", grossi=fit,
                )
            elif self.method == "sar":
                fit = run_sar(
                    inputs, p_factors=self.p_factors, M=self.mcmc_iter,
                    burn=self.mcmc_burn, step_rho=self.step_rho,
                    seed=self.mcmc_seed, ci_level=self.ci_level,
                )
                results = SpillSynthResults(
                    inputs=inputs, method="sar", sar=fit,
                )
            elif self.method == "iterative":
                fit = run_iterative(inputs, bilevel_solver=self.bilevel_solver,
                                    bias_correct=self.bias_correct,
                                    intercept=self.iscm_intercept)
                results = SpillSynthResults(
                    inputs=inputs, method="iterative", iterative=fit,
                )
            else:                                            # pragma: no cover
                raise MlsynthConfigError(
                    f"SPILLSYNTH: unknown method {self.method!r}."
                )
        except (MlsynthConfigError, MlsynthDataError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SPILLSYNTH/{self.method}: estimation failed ({exc})."
            ) from exc

        if self.display_graphs:
            try:
                cf_color = (
                    self.counterfactual_color[0]
                    if isinstance(self.counterfactual_color, list)
                    and self.counterfactual_color
                    else "red"
                )
                save_path = self.save if isinstance(self.save, str) else None
                plot_spillsynth(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=cf_color,
                    save=save_path,
                )
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"SPILLSYNTH: plotting failed ({exc})."
                ) from exc

        return results
