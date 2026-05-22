"""PCR-based Robust Synthetic Control pipeline.

Wraps the existing :func:`mlsynth.utils.estutils.pcr` routine
(Agarwal, Shah, Shen and Song 2021 robust PCR-SC; Amjad, Shah and
Shen 2018 RSC clustering; Bayani 2022 Bayesian RSC) and packages
its output into a clean :class:`MethodFit` for downstream consumers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..estutils import pcr
from .structures import MethodFit


def run_pcr(
    treated_outcome: np.ndarray,
    donor_outcomes: np.ndarray,
    donor_names: np.ndarray,
    T0: int,
    objective: str = "OLS",
    clustering: bool = True,
    estimator: str = "frequentist",
    lambda_penalty: Optional[float] = None,
    p: Optional[float] = None,
    q: Optional[float] = None,
) -> Tuple[MethodFit, Optional[Tuple[float, float]]]:
    """Run the PCR-RSC pipeline and assemble a :class:`MethodFit`.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcome matrix, shape ``(T, J)``.
    donor_names : np.ndarray
        Length-``J`` labels of the donor units.
    T0 : int
        Number of pre-treatment periods.
    objective : {"OLS", "SIMPLEX"}
        Inner SCM weight objective.
    clustering : bool
        Whether to apply SVD-based donor clustering before fitting.
    estimator : {"frequentist", "bayesian"}
        Selects between the frequentist QP (Opt.SCopt) and the
        Bayesian-RSC posterior (Bayani 2022 dissertation, Ch. 1).
    lambda_penalty, p, q : float or None
        Optional elastic-net-style regularisation knobs forwarded to
        ``pcr``.

    Returns
    -------
    MethodFit
        Frozen container with the PCR fit.
    credible_interval : tuple of float or None
        Bayesian PCR credible interval when ``estimator == "bayesian"``;
        ``None`` otherwise.
    """
    if objective not in {"OLS", "SIMPLEX"}:
        raise MlsynthEstimationError(
            f"PCR objective must be 'OLS' or 'SIMPLEX'; got {objective!r}."
        )
    if estimator not in {"frequentist", "bayesian"}:
        raise MlsynthEstimationError(
            f"PCR estimator must be 'frequentist' or 'bayesian'; "
            f"got {estimator!r}."
        )

    raw = pcr(
        donor_outcomes_matrix=donor_outcomes,
        treated_unit_outcome_vector=treated_outcome,
        scm_objective_model_type=objective,
        all_donor_names=list(donor_names),
        num_pre_treatment_periods=T0,
        enable_clustering=clustering,
        use_frequentist_scm=(estimator == "frequentist"),
        lambda_penalty=lambda_penalty,
        p=p,
        q=q,
    )

    # ``pcr`` returns either a "weights" + "cf_mean" pair (Bayesian)
    # or a full {"Effects", "Fit", "Vectors", "Weights"} dict
    # (frequentist via _pcr_results_frequentist). Normalise here.
    counterfactual, donor_weights_dict, selected_donors, ci_band = (
        _unpack_pcr_raw(raw, donor_names, treated_outcome.shape[0])
    )

    gap = treated_outcome - counterfactual
    att = (
        float(np.mean(gap[T0:])) if treated_outcome.shape[0] > T0
        else float("nan")
    )

    # Bayesian credible interval: pcr() returns per-period (T,) bounds
    # for the counterfactual. Convert to an ATT-level credible interval
    # by averaging the per-period bound across the post-period, then
    # mapping back to the gap: tau_t = y_t - cf_t, so a 95% CI for tau
    # is (y_post_mean - cf_upper_post_mean, y_post_mean - cf_lower_post_mean).
    credible: Optional[Tuple[float, float]] = None
    if ci_band is not None and treated_outcome.shape[0] > T0:
        cf_low = np.asarray(ci_band[0], dtype=float).flatten()
        cf_high = np.asarray(ci_band[1], dtype=float).flatten()
        if cf_low.size == treated_outcome.shape[0] and cf_high.size == treated_outcome.shape[0]:
            y_post_mean = float(np.mean(treated_outcome[T0:]))
            credible = (
                y_post_mean - float(np.mean(cf_high[T0:])),
                y_post_mean - float(np.mean(cf_low[T0:])),
            )
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))

    metadata = {
        "objective": objective,
        "clustering": bool(clustering),
        "estimator": estimator,
        "lambda_penalty": lambda_penalty,
        "p": p,
        "q": q,
    }
    if credible is not None:
        metadata["bayesian_credible_interval"] = credible

    fit = MethodFit(
        name=f"pcr_{estimator}",
        counterfactual=counterfactual,
        gap=gap,
        att=att,
        pre_rmse=pre_rmse,
        donor_weights=donor_weights_dict,
        selected_donors=np.asarray(selected_donors),
        metadata=metadata,
    )
    return fit, credible


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unpack_pcr_raw(raw: dict, donor_names: np.ndarray, T: int):
    """Tease the components apart from the legacy ``pcr`` output shape.

    Returns ``(counterfactual, donor_weights_dict, selected_donors,
    ci_band)``. ``ci_band`` is the raw per-period (low, high) pair from
    the Bayesian path or ``None`` otherwise; the caller converts it to
    an ATT-level credible interval.
    """

    counterfactual: Optional[np.ndarray] = None
    donor_weights_dict: dict = {}
    selected = list(donor_names)
    ci_band: Optional[Tuple] = None

    # Frequentist path: estutils.pcr returns the full results dict.
    if "Vectors" in raw and "Weights" in raw:
        counterfactual = np.asarray(
            raw["Vectors"].get("Counterfactual"), dtype=float
        ).flatten()
        weights_payload = raw.get("Weights")
        if isinstance(weights_payload, list) and weights_payload:
            donor_weights_dict = {
                str(k): float(v)
                for k, v in dict(weights_payload[0]).items()
            }
        selected = list(donor_weights_dict.keys()) or list(donor_names)

    # Bayesian path: pcr returns {"weights", "cf_mean", "credible_interval"}.
    elif "cf_mean" in raw:
        counterfactual = np.asarray(raw["cf_mean"], dtype=float).flatten()
        donor_weights_dict = {
            str(k): float(v) for k, v in dict(raw.get("weights", {})).items()
        }
        ci_band = raw.get("credible_interval")
        selected = list(donor_weights_dict.keys()) or list(donor_names)
    else:
        raise MlsynthEstimationError(
            f"Unrecognised pcr() output shape: keys={sorted(raw.keys())}."
        )

    if counterfactual is None or counterfactual.size != T:
        raise MlsynthEstimationError(
            f"pcr() counterfactual has wrong length: "
            f"{None if counterfactual is None else counterfactual.size} vs {T}."
        )
    return counterfactual, donor_weights_dict, selected, ci_band
