"""Inference for the Synthetic Historical Control estimator.

Primary inference is the conformal permutation test of Chen, Yang & Yang
(2024, footnote 21) -- their application of Chernozhukov, Wuthrich & Zhu
(2021) to SHC -- computed by :func:`mlsynth.utils.inferutils.shc_conformal_test`.
Andrews-Genton conformal prediction bands are computed alongside for the
plot.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..inferutils import ag_conformal, shc_conformal_test
from .structures import SHCDesign, SHCInference, SHCInputs


def run_conformal_inference(
    inputs: SHCInputs,
    design: SHCDesign,
    observed: np.ndarray,
    counterfactual: np.ndarray,
    *,
    miscoverage_rate: float = 0.10,
    num_resamples: int = 1000,
    levels: Sequence[float] = (0.01, 0.05, 0.10),
    random_state: int = 0,
) -> SHCInference:
    """Assemble the SHC conformal permutation test and conformal bands.

    Parameters
    ----------
    inputs : SHCInputs
        Preprocessed series (supplies the pre-period and latent trend pool).
    design : SHCDesign
        Fitted design (supplies ``latent_pre`` for the pre-period residuals).
    observed, counterfactual : np.ndarray
        Observed and SHC series over the ``m + n`` block window.
    miscoverage_rate : float
        ``1 - coverage`` for the Andrews-Genton bands (0.10 -> 90%).
    num_resamples, levels, random_state
        Forwarded to :func:`shc_conformal_test`.
    """
    m = inputs.m
    T0 = inputs.T0

    # Paper's residuals: pre-period eps_t^0 = y_t - ell_hat_t over t = 1..T0
    # (the kernel-smoother residuals); post-period eps_t^0 = the gap.
    pre_residuals = inputs.y[:T0] - np.asarray(design.latent_pre).ravel()
    post_residuals = observed[m:] - counterfactual[m:]

    test = shc_conformal_test(
        pre_intervention_residuals=pre_residuals,
        post_intervention_residuals=post_residuals,
        num_resamples=num_resamples,
        levels=tuple(levels),
        random_state=random_state,
    )

    lower, upper = ag_conformal(
        actual_outcomes_pre_treatment=observed[:m],
        predicted_outcomes_pre_treatment=counterfactual[:m],
        predicted_outcomes_post_treatment=counterfactual[m:],
        miscoverage_rate=miscoverage_rate,
        pad_value=np.nan,
    )

    return SHCInference(
        method="conformal_permutation",
        test_statistic=test["test_statistic"],
        p_value=test["p_value"],
        critical_values=test["critical_values"],
        reject=test["reject"],
        num_resamples=test["num_resamples"],
        null_distribution=test["null_distribution"],
        conformal_lower=lower[m:],
        conformal_upper=upper[m:],
        confidence_level=1.0 - miscoverage_rate,
    )
