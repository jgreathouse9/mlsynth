"""Partially Pooled Synthetic Control (PPSCM) estimator.

A thin orchestration over :mod:`mlsynth.utils.ppscm_helpers`, faithfully
porting augsynth::multisynth:

    Ben-Michael, E., Feller, A., & Rothstein, J. (2022). "Synthetic Controls
    with Staggered Adoption." *JRSS-B* 84(2):351-381.

PPSCM removes two-way fixed effects, balances the residuals with a
partially-pooled QP (``nu`` interpolating between separate and fully pooled
SCM), and reports a relative-time event study and overall ATT with the paper's
delete-one jackknife. ``time_cohort=True`` collapses units sharing an adoption
time into one fully-pooled cohort.
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
from ..utils.ppscm_helpers.engine import run_multisynth
from ..utils.ppscm_helpers.inference import jackknife_inference, bootstrap_inference
from ..utils.ppscm_helpers.plotter import plot_ppscm
from ..utils.ppscm_helpers.setup import prepare_ppscm_inputs
from ..utils.ppscm_helpers.structures import (
    PPSCMDesign,
    PPSCMEventStudy,
    PPSCMInference,
    PPSCMResults,
)


class PPSCM:
    """Partially Pooled SCM estimator (augsynth::multisynth port).

    Parameters
    ----------
    config : PPSCMConfig or dict
        Validated configuration. Reads ``nu`` (pooling, or ``"auto"``),
        ``fixedeff``, ``n_leads``, ``n_lags``, ``time_cohort``, ``lam``,
        ``run_inference`` and ``alpha`` beyond the common panel fields.

    Returns
    -------
    PPSCMResults
        Design (pooling level + balance diagnostics), relative-time event
        study, overall ATT with jackknife inference, and donor weights.
    """

    def __init__(self, config: Union[PPSCMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = PPSCMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid PPSCM configuration: {exc}") from exc

        self.config: PPSCMConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.nu = config.nu
        self.fixedeff: bool = config.fixedeff
        self.n_leads = config.n_leads
        self.n_lags = config.n_lags
        self.time_cohort: bool = config.time_cohort
        self.lam: float = config.lam
        self.solver: Any = config.solver
        self.run_inference: bool = config.run_inference
        self.inference_method: str = config.inference_method
        self.n_boot: int = config.n_boot
        self.seed: int = config.seed
        self.alpha: float = config.alpha

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> PPSCMResults:
        """Fit PPSCM and return the typed result container."""
        try:
            inputs = prepare_ppscm_inputs(
                self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
            )
            Xy, trt, d = inputs.Xy, inputs.trt, inputs.n_pre
            T = Xy.shape[1]

            # augsynth defaults: n_leads = post periods of last-treated unit,
            # n_lags = all pre-treatment periods.
            n_leads = self.n_leads if self.n_leads is not None else (T - d)
            n_leads = min(n_leads, T - d)
            n_lags = self.n_lags if self.n_lags is not None else d
            n_lags = min(n_lags, d)

            nu_arg = None if (isinstance(self.nu, str) and self.nu == "auto") else float(self.nu)

            fit = run_multisynth(
                Xy, trt, d, n_leads, n_lags,
                fixedeff=self.fixedeff, time_cohort=self.time_cohort,
                nu=nu_arg, lam=self.lam, solver=self.solver,
            )

            design = PPSCMDesign(
                nu_used=fit["nu_used"], lam=self.lam, fixedeff=self.fixedeff,
                time_cohort=self.time_cohort, n_leads=n_leads, n_lags=n_lags,
                global_l2=fit["global_l2"], ind_l2=fit["ind_l2"],
                scaled_global_l2=fit["scaled_global_l2"],
                scaled_ind_l2=fit["scaled_ind_l2"],
            )

            per_time = fit["per_time"]
            att = fit["att"]
            if self.run_inference and self.inference_method == "bootstrap":
                att, se, ci, pt_se, pt_ci = bootstrap_inference(
                    fit, alpha=self.alpha, n_boot=self.n_boot, seed=self.seed,
                    per_time_full=per_time, att_full=att,
                )
                method = "bootstrap"
            elif self.run_inference:
                att, se, ci, pt_se, pt_ci = jackknife_inference(
                    Xy, trt, d, n_leads, n_lags,
                    fixedeff=self.fixedeff, time_cohort=self.time_cohort,
                    nu_used=fit["nu_used"], lam=self.lam, solver=self.solver,
                    alpha=self.alpha, per_time_full=per_time, att_full=att,
                )
                method = "jackknife"
            else:
                se = float("nan")
                ci = (float("nan"), float("nan"))
                pt_se = np.full_like(per_time, np.nan)
                pt_ci = np.column_stack([per_time, per_time])
                method = "none"

            event_study = PPSCMEventStudy(
                horizons=np.arange(n_leads), tau=per_time, se=pt_se, ci=pt_ci,
            )
            inference = PPSCMInference(att=float(att), se=float(se),
                                      ci=tuple(ci), method=method)

            # donor weights per treated cohort (label -> {donor: weight})
            donor_weights: Dict[Any, Dict[Any, float]] = {}
            for g, w in fit["weights"].items():
                key = (str(inputs.time_labels[fit["adopt_of"][g]]) if self.time_cohort
                       else str(inputs.units[fit["members"][g][0]]))
                donor_weights[key] = {
                    str(inputs.units[i]): float(w[i]) for i in np.nonzero(w > 1e-8)[0]
                }

            results = PPSCMResults(
                inputs=inputs, design=design, event_study=event_study,
                inference=inference, donor_weights=donor_weights,
                metadata={"n_treated": int(np.isfinite(trt).sum()),
                          "n_control": int((~np.isfinite(trt)).sum())},
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"PPSCM estimation failed: {exc}") from exc

        if self.display_graphs:
            try:
                plot_ppscm(results, save=self.save)
            except Exception as exc:
                raise MlsynthPlottingError(f"PPSCM plotting failed: {exc}") from exc

        return results
