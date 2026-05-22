"""Synthetic IV (SIV) estimator.

Gulek, A., and Vives-i-Bastida, J. (2024). "Synthetic IV Estimation
in Panels." Job Market Paper.

The estimator addresses unmeasured confounding in panel data by
chaining two ideas:

  1. **Synthetic control** per unit on the pre-period to remove the
     contribution of an unobserved factor structure ``mu_i' F_t``.
  2. **2SLS** on the debiased post-period series ``(\\tilde Y,
     \\tilde R, \\tilde Z)`` to handle simultaneity / measurement-error
     correlation between treatment ``R`` and outcome shock ``epsilon``.

Three variants are exposed via the ``mode`` config field:

* ``siv``        - debias Y, R, Z and run 2SLS (the canonical paper
                   estimator).
* ``projected``  - project Y_pre into the instrument space before
                   fitting the SC, then debias and run 2SLS. More
                   robust when the idiosyncratic noise dominates the
                   factor structure (Section 5.1).
* ``ensemble``   - convex combination of ``siv`` and ``projected``
                   with the blend weight picked on a held-out
                   validation block (Section 5.1).
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import SIVConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.siv_helpers.ensemble import select_alpha
from ..utils.siv_helpers.inference import (
    asymptotic_ci,
    split_conformal_inference,
)
from ..utils.siv_helpers.plotter import plot_event_study
from ..utils.siv_helpers.projection import project_outcome_pre
from ..utils.siv_helpers.setup import build_design_matrix, prepare_siv_inputs
from ..utils.siv_helpers.structures import (
    SIVEstimate,
    SIVInference,
    SIVResults,
)
from ..utils.siv_helpers.twosls import two_sls_just_identified
from ..utils.siv_helpers.weights import (
    assemble_weights,
    fit_synthetic_controls,
    fit_synthetic_controls_asymmetric,
)


class SIV:
    """Synthetic Instrumental Variables estimator.

    Parameters
    ----------
    config : SIVConfig or dict
        Estimator configuration. See :class:`mlsynth.config_models.SIVConfig`.
    """

    def __init__(self, config: Union[SIVConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SIVConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SIV configuration: {exc}"
                ) from exc

        self.config: SIVConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.instrument: str = config.instrument
        self.unitid: str = config.unitid
        self.time: str = config.time

    def fit(self) -> SIVResults:
        """Run the SIV pipeline end-to-end and return a :class:`SIVResults`."""

        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_siv_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                instrument=self.instrument,
                unitid=self.unitid,
                time=self.time,
                T0=self.config.T0,
                post_col=self.config.post_col,
                T0_train=self.config.T0_train,
            )

            # ---------- SIV pipeline ----------
            design_siv = build_design_matrix(inputs, series="default")
            W_siv = fit_synthetic_controls(
                design_siv,
                constraint=self.config.weight_constraint,
                l1_C=self.config.l1_C,
            )
            weights_siv = assemble_weights(inputs, W_siv, self.config.weight_constraint)

            est_siv = two_sls_just_identified(
                Y=weights_siv.Y_tilde,
                R=weights_siv.R_tilde,
                Z=weights_siv.Z_tilde,
                T0=inputs.T0,
                variant="siv",
            )
            est_siv_z = two_sls_just_identified(
                Y=inputs.Y,
                R=inputs.R,
                Z=weights_siv.Z_tilde,
                T0=inputs.T0,
                variant="siv_z",
            )
            est_siv_yr = two_sls_just_identified(
                Y=weights_siv.Y_tilde,
                R=weights_siv.R_tilde,
                Z=inputs.Z,
                T0=inputs.T0,
                variant="siv_yr",
            )

            estimates: Dict[str, SIVEstimate] = {
                "siv": est_siv,
                "siv_z": est_siv_z,
                "siv_yr": est_siv_yr,
            }

            # ---------- Projected pipeline (always computed for diagnostics) ----------
            weights_proj = None
            est_projected = None
            try:
                Y_pre_proj = project_outcome_pre(inputs)
                # Projected SC fit (eq. 5.1.2): focal unit matches its
                # *raw* pre-period series; donors contribute their
                # *projected* (instrument-space) series.
                target_blocks = [inputs.Y[:, : inputs.T0]]
                donor_blocks = [Y_pre_proj]
                if inputs.has_pre_treatment:
                    target_blocks.append(inputs.R[:, : inputs.T0])
                    donor_blocks.append(inputs.R[:, : inputs.T0])
                if inputs.has_pre_instrument:
                    target_blocks.append(inputs.Z[:, : inputs.T0])
                    donor_blocks.append(inputs.Z[:, : inputs.T0])
                target_design = np.concatenate(target_blocks, axis=1)
                donor_design = np.concatenate(donor_blocks, axis=1)
                W_proj = fit_synthetic_controls_asymmetric(
                    target_design=target_design,
                    donor_design=donor_design,
                    constraint=self.config.weight_constraint,
                    l1_C=self.config.l1_C,
                )
                weights_proj = assemble_weights(
                    inputs, W_proj, self.config.weight_constraint
                )
                est_projected = two_sls_just_identified(
                    Y=weights_proj.Y_tilde,
                    R=weights_proj.R_tilde,
                    Z=weights_proj.Z_tilde,
                    T0=inputs.T0,
                    variant="projected",
                )
                estimates["projected"] = est_projected
            except Exception as exc:  # noqa: BLE001
                # Projection requires a non-degenerate post-period
                # instrument; if that fails (e.g. all zeros) we skip
                # the projected variant but still return SIV results.
                if self.config.mode in {"projected", "ensemble"}:
                    raise MlsynthEstimationError(
                        f"Projected SIV failed: {exc}"
                    ) from exc

            # ---------- Ensemble (if requested) ----------
            ensemble_alpha = None
            if self.config.mode == "ensemble" and est_projected is not None:
                if self.config.ensemble_alpha is not None:
                    ensemble_alpha = float(self.config.ensemble_alpha)
                else:
                    T_v = inputs.T0_train or max(2, int(0.75 * inputs.T0))
                    val_slice = slice(T_v, inputs.T0)
                    # select_alpha returns the weight on the PROJECTED
                    # residuals that minimises the blended validation
                    # MSE. We apply the same weight on theta_projected
                    # so a smaller PROJ residual maps to more weight on
                    # theta_projected (the paper's eq. 5.1 step 4 has
                    # an inconsistent label; we use the consistent
                    # interpretation: alpha = weight on PROJ in both
                    # the residual blend and the theta blend).
                    ensemble_alpha = select_alpha(
                        Y_tilde_siv_val=weights_siv.Y_tilde[:, val_slice],
                        Y_tilde_proj_val=weights_proj.Y_tilde[:, val_slice],
                    )
                # Blend at the estimate level: alpha is the weight on
                # the projected estimator.
                theta_E = (
                    ensemble_alpha * est_projected.theta_hat
                    + (1.0 - ensemble_alpha) * est_siv.theta_hat
                )
                se_E = (
                    ensemble_alpha * est_projected.se
                    + (1.0 - ensemble_alpha) * est_siv.se
                    if np.isfinite(est_siv.se) and np.isfinite(est_projected.se)
                    else float("nan")
                )
                est_ensemble = SIVEstimate(
                    variant="ensemble",
                    theta_hat=float(theta_E),
                    se=float(se_E),
                    pi_hat=ensemble_alpha * est_projected.pi_hat
                        + (1.0 - ensemble_alpha) * est_siv.pi_hat,
                    beta_first_stage=ensemble_alpha * est_projected.beta_first_stage
                        + (1.0 - ensemble_alpha) * est_siv.beta_first_stage,
                    f_stat=ensemble_alpha * est_projected.f_stat
                        + (1.0 - ensemble_alpha) * est_siv.f_stat,
                    n_post_obs=est_siv.n_post_obs,
                )
                estimates["ensemble"] = est_ensemble

            selected = self.config.mode
            if selected not in estimates:
                raise MlsynthEstimationError(
                    f"Mode {selected!r} not available: estimates="
                    f"{list(estimates)}."
                )
            selected_estimate = estimates[selected]

            # ---------- Inference ----------
            if self.config.inference_method == "none":
                inference = SIVInference(
                    method="none",
                    alpha=float(self.config.alpha),
                    theta_hat=float(selected_estimate.theta_hat),
                )
            elif self.config.inference_method == "asymptotic":
                inference = asymptotic_ci(selected_estimate, alpha=self.config.alpha)
            elif self.config.inference_method == "conformal":
                inference = split_conformal_inference(
                    inputs=inputs,
                    weights=weights_siv if selected != "projected" else weights_proj,
                    estimate=selected_estimate,
                    alpha=self.config.alpha,
                    max_permutations=self.config.n_permutations,
                    seed=self.config.seed,
                )
            else:
                raise MlsynthConfigError(
                    f"Unknown inference_method={self.config.inference_method!r}."
                )

            results = SIVResults(
                inputs=inputs,
                weights=weights_siv,
                weights_projected=weights_proj,
                estimates=estimates,
                selected_variant=selected,
                inference=inference,
                metadata={
                    "ensemble_alpha": ensemble_alpha,
                    "has_pre_treatment": inputs.has_pre_treatment,
                    "has_pre_instrument": inputs.has_pre_instrument,
                },
            )

            if self.config.display_graph:
                try:
                    plot_event_study(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"SIV plotting failed: {exc}"
                    ) from exc

            return results

        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
            MlsynthPlottingError,
        ):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SIV estimation failed: {exc}"
            ) from exc
