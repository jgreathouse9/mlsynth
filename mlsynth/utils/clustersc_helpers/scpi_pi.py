"""scpi prediction intervals for the CLUSTERSC RSC / PCR (and RPCA) fits.

Cattaneo, Feng, Palomba & Titiunik (2025, JSS ``scpi``) Table 3 pairs each
weight-constraint family with a synthetic-control method: the ridge constraint
is the inference setting for Amjad, Kim, Shah & Shen (2018) Robust Synthetic
Control, which CLUSTERSC's PCR (RSC) path implements. This module runs
VanillaSC's generalized :func:`mlsynth.utils.vanillasc_helpers.scpi.scpi_intervals`
on a CLUSTERSC fit -- the denoised donor matrix and the fitted weights -- under
a chosen constraint, and packages the pointwise and simultaneous prediction
intervals into a small frozen container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ScpiPIInference:
    """scpi prediction intervals for a CLUSTERSC fit (per-period + ATT).

    Parameters
    ----------
    method : str
        Human-readable label, e.g. ``"scpi prediction intervals (ridge)"``.
    constraint : str
        The scpi weight constraint used (``ols`` / ``simplex`` / ``lasso`` /
        ``ridge`` / ``L1-L2``).
    att_pi : tuple of float
        ``(lower, upper)`` prediction interval for the ATT.
    pi_lower, pi_upper : np.ndarray
        Pointwise per-post-period prediction intervals for the treatment effect.
    cf_lower, cf_upper : np.ndarray
        Pointwise per-post-period bands for the counterfactual.
    pi_lower_simul, pi_upper_simul : np.ndarray
        Simultaneous (joint-coverage) per-period effect prediction intervals.
    cf_lower_simul, cf_upper_simul : np.ndarray
        Simultaneous per-period counterfactual bands.
    df : float
        Effective degrees of freedom scpi used for the constraint.
    Q, lambda_ : float or None
        The constraint budget and (ridge / L1-L2) penalty scpi estimated.
    periods : list
        Post-period labels aligned with the per-period arrays.
    metadata : dict
        Passthrough of the scpi metadata.
    """

    method: str
    constraint: str
    att_pi: Tuple[float, float]
    pi_lower: np.ndarray
    pi_upper: np.ndarray
    cf_lower: np.ndarray
    cf_upper: np.ndarray
    pi_lower_simul: np.ndarray
    pi_upper_simul: np.ndarray
    cf_lower_simul: np.ndarray
    cf_upper_simul: np.ndarray
    df: float
    Q: Any = None
    lambda_: Any = None
    periods: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prediction_interval_spec(self) -> Dict[str, Any]:
        """Normalized spec for the canonical ``TimeSeriesResults`` band.

        Maps this scpi object's per-post-period counterfactual bands (pointwise
        and simultaneous) onto the spec ``build_effect_submodels`` consumes, so
        every scpi-based estimator populates the one contract field the same way.
        """
        u_alpha = float(self.metadata.get("u_alpha", 0.05))
        return {
            "lower": self.cf_lower, "upper": self.cf_upper,
            "lower_simultaneous": self.cf_lower_simul,
            "upper_simultaneous": self.cf_upper_simul,
            "periods": list(self.periods),
            "level": 1.0 - 2.0 * u_alpha,
            "kind": f"scpi:{self.constraint}",
        }


def scpi_pi_inference(
    treated_outcome: np.ndarray,
    donor_full: np.ndarray,
    T0: int,
    weights: np.ndarray,
    *,
    constraint: Any = "ridge",
    constant: bool = False,
    sims: int = 200,
    alpha: float = 0.05,
    e_method: str = "gaussian",
    seed: int = 0,
    periods: Any = None,
) -> ScpiPIInference:
    """Run scpi prediction intervals on a fit under ``constraint``.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    donor_full : np.ndarray
        Donor design the counterfactual projects through (the denoised donor
        matrix for PCR / RPCA, or the donor pool for SCUL), shape ``(T, J)``,
        columns aligned with the donor block of ``weights``.
    T0 : int
        Number of pre-treatment periods.
    weights : np.ndarray
        Fitted weights: the donor weights ``(J,)``, or ``(J + 1,)`` with a
        trailing intercept coefficient when ``constant=True``.
    constraint : str or dict
        scpi weight-constraint family (default ``"ridge"``, scpi's Table-3
        setting for Robust SC), or an explicit ``{"name": ..., "Q": ...}`` dict.
    constant : bool
        If True, the design gains an unconstrained intercept (scpi's ``KM``
        block) and ``weights`` carries its coefficient last.
    sims, alpha, e_method, seed
        Passed through to ``scpi_intervals`` (``u_alpha = e_alpha = alpha``).
    periods : sequence, optional
        Post-period labels for reporting.
    """
    from ..vanillasc_helpers.scpi import scpi_intervals

    y = np.asarray(treated_outcome, float).ravel()
    Y0 = np.asarray(donor_full, float)
    W = np.asarray(weights, float).ravel()

    sc = scpi_intervals(
        y, Y0, int(T0), W, w_constr=constraint, constant=bool(constant),
        sims=int(sims), u_alpha=float(alpha), e_alpha=float(alpha),
        e_method=e_method, cointegrated=False, seed=int(seed),
    )
    post = list(periods) if periods is not None else list(range(len(sc.tau)))
    return ScpiPIInference(
        method=f"scpi prediction intervals ({sc.metadata['w_constr']})",
        constraint=str(sc.metadata["w_constr"]),
        att_pi=(float(sc.metadata["att_lower"]), float(sc.metadata["att_upper"])),
        pi_lower=np.asarray(sc.lower, float),
        pi_upper=np.asarray(sc.upper, float),
        cf_lower=np.asarray(sc.cf_lower, float),
        cf_upper=np.asarray(sc.cf_upper, float),
        pi_lower_simul=np.asarray(sc.lower_simul, float),
        pi_upper_simul=np.asarray(sc.upper_simul, float),
        cf_lower_simul=np.asarray(sc.cf_lower_simul, float),
        cf_upper_simul=np.asarray(sc.cf_upper_simul, float),
        df=float(sc.metadata["df"]),
        Q=sc.metadata.get("Q"),
        lambda_=sc.metadata.get("lambda"),
        periods=post,
        metadata=dict(sc.metadata),
    )
