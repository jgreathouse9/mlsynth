"""Partially Pooled Synthetic Control (PPSCM) estimator.

Implements:

    Ben-Michael, E., Feller, A., & Rothstein, J. (2022). "Synthetic
    Controls with Staggered Adoption." *JRSS-B* 84(2):351-381.

This is the *outcome-only* variant (Sections 3-4 of the paper); the
auxiliary-covariate extension of Section 5.2 is intentionally not
implemented.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import PPSCMConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.ppscm_helpers.imbalance import compute_q_pool, compute_q_sep
from ..utils.ppscm_helpers.inference import (
    event_study_taus,
    jackknife_inference,
)
from ..utils.ppscm_helpers.optimization import solve_ppscm
from ..utils.ppscm_helpers.plotter import plot_ppscm
from ..utils.ppscm_helpers.setup import prepare_ppscm_inputs
from ..utils.ppscm_helpers.structures import (
    PPSCMDesign,
    PPSCMEventStudy,
    PPSCMInference,
    PPSCMResults,
)


class PPSCM:
    """Partially Pooled SCM estimator.

    Parameters
    ----------
    config : PPSCMConfig or dict
        Configuration object. See :class:`mlsynth.config_models.PPSCMConfig`.

    Returns
    -------
    PPSCMResults
        Typed container with the weight matrix, per-horizon ATT
        trajectory, overall ATT with jackknife inference, and frontier
        diagnostics.

    References
    ----------
    Ben-Michael, E., Feller, A., & Rothstein, J. (2022). "Synthetic
    Controls with Staggered Adoption." *JRSS-B* 84(2):351-381.
    """

    def __init__(self, config: Union[PPSCMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = PPSCMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid PPSCM configuration: {exc}"
                ) from exc

        self.config: PPSCMConfig = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.L = config.L
        self.K = config.K
        self.nu = config.nu
        self.nu_grid_size: int = config.nu_grid_size
        self.lam: float = config.lam
        self.demean: bool = config.demean
        self.solver: Any = config.solver
        self.run_inference: bool = config.run_inference
        self.alpha: float = config.alpha

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> PPSCMResults:
        """Fit PPSCM and return the typed result container."""
        try:
            inputs = prepare_ppscm_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
                L=self.L, K=self.K, demean=self.demean,
            )

            Gamma, nu_used, frontier, status, q_sep_base, q_pool_base = solve_ppscm(
                Y_treated_pre=inputs.Y_treated_pre,
                Y_donors_pre=inputs.Y_donors_pre,
                nu=self.nu, lam=self.lam, solver=self.solver,
                nu_grid_size=self.nu_grid_size,
            )

            q_sep_val = compute_q_sep(inputs.Y_treated_pre,
                                      inputs.Y_donors_pre, Gamma)
            q_pool_val = compute_q_pool(inputs.Y_treated_pre,
                                        inputs.Y_donors_pre, Gamma)

            design = PPSCMDesign(
                Gamma=Gamma, nu_used=float(nu_used), lam=self.lam,
                q_sep=q_sep_val, q_pool=q_pool_val,
                q_sep_baseline=q_sep_base, q_pool_baseline=q_pool_base,
                frontier=frontier, solver_status=status,
            )

            # Per-horizon ATTs on the full panel.
            tau_per_horizon = event_study_taus(
                inputs.Y_treated_post, inputs.Y_donors_post, Gamma
            )

            if self.run_inference and inputs.J >= 2:
                att, se, ci, loo_means, se_per_h = jackknife_inference(
                    Y_treated_pre=inputs.Y_treated_pre,
                    Y_donors_pre=inputs.Y_donors_pre,
                    Y_treated_post=inputs.Y_treated_post,
                    Y_donors_post=inputs.Y_donors_post,
                    nu=self.nu, lam=self.lam, solver=self.solver,
                    nu_grid_size=self.nu_grid_size, alpha=self.alpha,
                )
                inf_method = "jackknife"
            else:
                att = float(tau_per_horizon.mean())
                se = float("nan")
                ci = (float("nan"), float("nan"))
                se_per_h = np.full(inputs.K + 1, float("nan"))
                inf_method = "none"

            from scipy.stats import norm
            z = float(norm.ppf(1.0 - self.alpha / 2.0))
            ci_per_horizon = np.column_stack([
                tau_per_horizon - z * se_per_h,
                tau_per_horizon + z * se_per_h,
            ])
            event_study = PPSCMEventStudy(
                horizons=np.arange(inputs.K + 1),
                tau=tau_per_horizon,
                se=se_per_h,
                ci=ci_per_horizon,
            )
            inference = PPSCMInference(
                att=float(att), se=float(se), ci=tuple(ci), method=inf_method
            )

            # Pre-period RMSE on the raw outcome scale.
            pre_rmse = q_sep_val

            donor_weights: Dict[Any, Dict[Any, float]] = {
                str(inputs.treated_unit_names[j]): {
                    str(inputs.donor_names[i]): float(Gamma[i, j])
                    for i in range(inputs.N)
                }
                for j in range(inputs.J)
            }

            results = PPSCMResults(
                inputs=inputs, design=design, event_study=event_study,
                inference=inference, pre_rmse=pre_rmse,
                donor_weights=donor_weights, demean=self.demean,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"PPSCM estimation failed: {exc}"
            ) from exc

        if self.display_graphs:
            try:
                plot_ppscm(results, save=self.save)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"PPSCM plotting failed: {exc}"
                ) from exc

        return results
