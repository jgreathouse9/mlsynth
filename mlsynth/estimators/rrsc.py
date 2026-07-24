"""RRSC: robust-regression synthetic control with interference.

He, P., Li, Y., Shi, X., & Miao, W. (2026). *A robust regression approach to
synthetic control with interference* (arXiv:2411.01249).

Under an untreated linear factor model ``Y_it(0) = lambda_i' f_t + eps_it``,
RRSC recovers the average direct effect on the treated unit *and* the average
interference effect on every control unit -- without prespecifying which
controls are interfered. Stage one estimates the time-invariant loadings from
the pre-period; stage two writes the average effects as a sparse-outlier
component of a robust regression against those loadings:

* fixed-N (Section 3): ``factanal`` loadings + high-breakdown LTS under the
  majority-valid-controls condition, then a Donoho-Johnstone selection of valid
  controls and an OLS refit.
* large-N (Section 4): EM factor analysis + Huber-IRLS factor regression, valid
  even with a short post-period.

The distinctive output is a per-unit *interference map* -- an average effect, a
confidence interval and a p-value for every unit (the treated unit's entry is
the direct effect). Inference is the circular block bootstrap (default) or the
Chernozhukov et al. (2021) conformal permutation test.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import InferenceResults, RRSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.results_helpers import build_effect_submodels
from ..utils.rrsc_helpers.inference import run_cbb, run_conformal
from ..utils.rrsc_helpers.pipeline import implied_donor_weights, run_rrsc
from ..utils.rrsc_helpers.plotter import plot_rrsc
from ..utils.rrsc_helpers.setup import prepare_rrsc_inputs
from ..utils.rrsc_helpers.structures import RRSCResults


class RRSC:
    """Robust-regression synthetic control with interference (He et al. 2026).

    Parameters
    ----------
    config : RRSCConfig or dict
        Configuration. See :class:`mlsynth.config_models.RRSCConfig`.

    Returns
    -------
    RRSCResults
        An ``EffectResult`` whose flat accessors report the treated unit's
        direct effect, plus the per-unit interference map.
    """

    def __init__(self, config: Union[RRSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = RRSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid RRSC configuration: {exc}") from exc
        self.config: RRSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome, self.treat = config.outcome, config.treat
        self.unitid, self.time = config.unitid, config.time
        self.display_graphs: bool = getattr(config, "display_graphs", False)

    def fit(self) -> RRSCResults:
        """Run the RRSC pipeline end to end."""
        cfg = self.config
        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_rrsc_inputs(
                self.df, self.outcome, self.treat, self.unitid, self.time,
                regime=cfg.regime,
            )
            fit = run_rrsc(inputs, cfg)

            # ----- inference (per unit) -----
            block = cfg.cbb_block_length or max(2, inputs.T0 // 10)
            if cfg.inference == "cbb":
                inf = run_cbb(
                    inputs, fit.r, block_length=block, reps=cfg.cbb_reps,
                    separation=cfg.cbb_separation, update_dif=cfg.update_dif,
                    lts_alpha=cfg.lts_alpha, delta=cfg.huber_delta, alpha=cfg.alpha,
                    rng=np.random.default_rng(cfg.random_state),
                )
            elif cfg.inference == "conformal":
                inf = run_conformal(inputs, fit.r, update_dif=cfg.update_dif,
                                    lts_alpha=cfg.lts_alpha, delta=cfg.huber_delta)
            else:                                            # "none"
                nan = np.full(inputs.N, np.nan)
                inf = dict(est=fit.effects, var=nan, ci_lower=nan, ci_upper=nan,
                           pvalue=nan)

            names = [str(u) for u in inputs.unit_names]
            interference_effects = {n: float(e) for n, e in zip(names, fit.effects)}
            interference_ci = {
                n: (None if np.isnan(lo) else float(lo),
                    None if np.isnan(hi) else float(hi))
                for n, lo, hi in zip(names, inf["ci_lower"], inf["ci_upper"])
            }
            interference_pvalue = {
                n: (float("nan") if np.isnan(p) else float(p))
                for n, p in zip(names, inf["pvalue"])
            }
            communality = {n: float(c) for n, c in zip(names, fit.communality)}

            # ----- treated (index 0) standardized surface -----
            observed = np.asarray(inputs.treated_outcome, dtype=float)
            counterfactual = np.asarray(fit.counterfactual_panel[0], dtype=float)
            labels = np.asarray(inputs.time_labels)
            T0, T = inputs.T0, inputs.T
            treated_lo, treated_hi = inf["ci_lower"][0], inf["ci_upper"][0]
            treated_sd = float(np.sqrt(inf["var"][0])) if not np.isnan(inf["var"][0]) else None
            treated_inf = InferenceResults(
                standard_error=treated_sd,
                ci_lower=None if np.isnan(treated_lo) else float(treated_lo),
                ci_upper=None if np.isnan(treated_hi) else float(treated_hi),
                p_value=None if np.isnan(inf["pvalue"][0]) else float(inf["pvalue"][0]),
                confidence_level=1 - cfg.alpha,
                method=cfg.inference,
            )
            subs = build_effect_submodels(
                observed_outcome=observed,
                counterfactual_outcome=counterfactual,
                n_pre_periods=T0,
                n_post_periods=T - T0,
                time_periods=labels,
                weights=implied_donor_weights(fit.Lambda, inputs.unit_names),
                inference=treated_inf,
                method_name=f"RRSC ({fit.regime})",
                effects_overrides={"att": float(fit.effects[0]),
                                   "att_std_err": treated_sd},
                intervention_time=(labels[T0] if T0 < T else None),
            )

            results = RRSCResults(
                inputs=inputs,
                regime=fit.regime,
                n_factors=fit.r,
                interference_effects=interference_effects,
                interference_ci=interference_ci,
                interference_pvalue=interference_pvalue,
                counterfactual_panel=np.asarray(fit.counterfactual_panel, dtype=float),
                communality=communality,
                metadata={
                    "regime": fit.regime, "n_factors": fit.r,
                    "inference": cfg.inference, "block_length": int(block),
                    "loss": cfg.loss, "update_dif": cfg.update_dif,
                    "n_selected": (None if fit.selection is None
                                   else int(np.sum(fit.selection))),
                    "alpha": cfg.alpha,
                },
                **subs,
            )

            if self.display_graphs:
                try:
                    plot_rrsc(results)
                except Exception as exc:                     # pragma: no cover
                    raise MlsynthPlottingError(f"RRSC plotting failed: {exc}") from exc
            return results

        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError,
                MlsynthPlottingError):
            raise
        except Exception as exc:                             # pragma: no cover
            raise MlsynthEstimationError(f"RRSC estimation failed: {exc}") from exc
