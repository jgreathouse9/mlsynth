"""The EASC pipeline: diagnostics, the four fits, and bias-aware inference.

Orchestrates the pure kernels in ``diagnostics``, ``correction``, ``frontier``
and ``inference`` on an :class:`EASCInputs`. The reference is the author's
``ebsc.py`` with ``app.py`` and ``app_inference.py``; the replication spike in
``benchmarks/studies/easc_replicate`` cross-validates every quantity here
against it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthDataError
from . import correction as C
from . import diagnostics as D
from . import frontier as Fr
from . import inference as I
from .config import OVERALL_PHASE
from .structures import (
    EASCDiagnostics,
    EASCFit,
    EASCFrontier,
    EASCInference,
    EASCInputs,
    EASCPhaseInference,
)


@dataclass(frozen=True)
class EASCEstimates:
    """Everything the pipeline computes, for :class:`EASCResults` to hold."""

    fits: Dict[str, EASCFit]
    diagnostics: EASCDiagnostics
    frontier: EASCFrontier
    inference: Optional[EASCInference]


def _phase_masks(inputs: EASCInputs, phases: Optional[Dict[str, Tuple[Any, Any]]]) -> Dict[str, np.ndarray]:
    labels = list(inputs.time_labels)
    T, T0 = inputs.T, inputs.T0
    masks = {OVERALL_PHASE: np.arange(T) >= T0}
    for name, (start, end) in (phases or {}).items():
        idx = []
        for label in (start, end):
            if label not in labels:
                raise MlsynthDataError(f"EASC: phase {name!r} bound {label!r} is not a period of the panel.")
            idx.append(labels.index(label))
        if idx[0] < T0:
            raise MlsynthDataError(f"EASC: phase {name!r} must lie in the post-treatment period.")
        mask = np.zeros(T, dtype=bool)
        mask[idx[0]:idx[1] + 1] = True
        masks[name] = mask
    return masks


def _check(inputs: EASCInputs, n_factors: int, n_subperiods: int, inference: bool) -> None:
    J = inputs.n_donors
    p = 2 + n_subperiods
    if J < p + 1:
        raise MlsynthDataError(
            f"EASC: the exposure regression has {p} coefficients and needs at least {p + 1} donors; the pool "
            f"has {J}. Add donors or use fewer growth-control sub-periods (n_subperiods).")
    if inference and J < p + 2:
        raise MlsynthDataError(
            f"EASC: the placebos refit the exposure regression without each donor, which needs at least "
            f"{p + 2} donors; the pool has {J}. Set inference=False or use fewer sub-periods.")
    if n_factors > min(inputs.T0, J) - 2:
        raise MlsynthDataError(
            f"EASC: n_factors={n_factors} needs at least {n_factors + 2} donors and {n_factors + 2} "
            f"pre-treatment periods; there are {J} and {inputs.T0}.")
    if inputs.base_index < 2 * n_subperiods:
        raise MlsynthDataError(
            f"EASC: each of the {n_subperiods} growth-control sub-periods needs at least two pre-treatment "
            f"periods before the base period, which has {inputs.base_index} before it.")
    if not C.exposure_is_identified(inputs.donor_matrix, inputs.exposure_donors, inputs.pre,
                                    inputs.base_index, n_subperiods):
        raise MlsynthDataError(
            "EASC: the exposure index does not vary across donors beyond what the growth controls span, "
            "so the exposure effect is not identified.")


def _fit(name: str, inputs: EASCInputs, w: np.ndarray, phi: Optional[np.ndarray]) -> EASCFit:
    y1, Y0, pre = inputs.y, inputs.donor_matrix, inputs.pre
    sc_gap = y1 - Y0 @ w
    if phi is None:
        gap = sc_gap
        ei = D.exposure_imbalance(inputs.exposure_treated, inputs.exposure_donors, w)
    else:
        gap, ei = C.easc_gap(y1, Y0, w, inputs.exposure_treated, inputs.exposure_donors, phi)
    return EASCFit(name=name, weights=np.asarray(w, dtype=float), counterfactual=y1 - gap, gap=gap,
                   att=float(gap[~pre].mean()), pre_rmse=float(np.sqrt(np.mean(sc_gap[pre] ** 2))),
                   exposure_imbalance=float(ei), corrected=phi is not None)


def _unit(y1: np.ndarray, Y0: np.ndarray, s1: float, s0: np.ndarray, pre: np.ndarray, base: int,
          n_subperiods: int) -> Dict[str, Any]:
    """SC and EASC gaps for one unit treated as treated (``run_unit`` in ``app_inference.py``)."""
    w = Fr.sc_weights(y1[pre], Y0[pre])
    reg = C.exposure_effects(Y0, s0, pre, base, n_subperiods)
    ge, ei = C.easc_gap(y1, Y0, w, s1, s0, reg.phi)
    g = y1 - Y0 @ w
    return dict(SC=g, EASC=ge, ei=ei, reg=reg, rmspe=float(np.sqrt(np.mean(g[pre] ** 2))))


def _inference(inputs: EASCInputs, treated: Dict[str, Any], masks: Dict[str, np.ndarray],
               leverage: Dict[str, Tuple[float, float]], *, n_subperiods: int, alpha: float,
               m_grid: Sequence[float], headline_m: float, correction_se: str) -> EASCInference:
    Y0, s0, pre, base = inputs.donor_matrix, inputs.exposure_donors, inputs.pre, inputs.base_index
    J = inputs.n_donors
    placebos = []
    for j in range(J):
        keep = np.delete(np.arange(J), j)
        placebos.append(_unit(Y0[:, j], Y0[:, keep], float(s0[j]), s0[keep], pre, base, n_subperiods))
    labels = list(inputs.time_labels)
    rows = []
    headline = (np.nan, np.nan)
    for est, which in (("SC", 0), ("EASC", 1)):
        for name, mask in masks.items():
            tau = float(treated[est][mask].mean())
            means = np.array([p[est][mask].mean() for p in placebos])
            rmspe = np.array([p["rmspe"] for p in placebos])
            sd, sdm = I.placebo_noise_scale(means, rmspe, treated["rmspe"])
            if est == "EASC":
                reg = treated["reg"]
                se = reg.se[mask].mean() if correction_se == "period_mean" else C.jackknife_phase_se(reg.replicates, mask)
                extra = abs(treated["ei"]) * se
                sd, sdm = float(np.hypot(sd, extra)), float(np.hypot(sdm, extra))
            kappa = leverage[name][which]
            r = treated["rmspe"]

            def interval(M: float) -> Tuple[float, float]:
                hw = I.bias_aware_halfwidth(M, kappa, r, sd, alpha)
                return (tau - hw, tau + hw)

            idx = np.flatnonzero(mask)
            rows.append(EASCPhaseInference(
                estimator=est, phase=name, first=labels[idx[0]], last=labels[idx[-1]], estimate=tau,
                noise_sd=sd, noise_sd_robust=sdm, leverage=kappa,
                placebo_p_value=I.placebo_p_value(tau, r, means, rmspe),
                intervals={float(M): interval(M) for M in m_grid},
                breakdown=I.breakdown(tau, kappa, r, sd, alpha),
                breakdown_robust=I.breakdown(tau, kappa, r, sdm, alpha)))
            if est == "EASC" and name == OVERALL_PHASE:
                headline = interval(headline_m)
    return EASCInference(alpha=alpha, m_grid=tuple(float(m) for m in m_grid), headline_m=float(headline_m),
                         correction_se=correction_se, rows=tuple(rows), headline_interval=headline)


def run_easc(inputs: EASCInputs, *, n_factors: int = 2, n_subperiods: int = 3, frontier_points: int = 41,
             phases: Optional[Dict[str, Tuple[Any, Any]]] = None, inference: bool = True, alpha: float = 0.05,
             m_grid: Sequence[float] = (0.0, 0.5, 1.0), headline_m: float = 0.0,
             correction_se: str = "period_mean") -> EASCEstimates:
    """Diagnostics, SC / EASC / EBSC-min / EASC*, and (optionally) bias-aware inference.

    Raises
    ------
    MlsynthDataError
        If the donor pool is too small for the exposure regression (or for the
        placebos when ``inference`` is on), there are too few pre-treatment
        periods for the factors or the growth controls, the exposure index is
        not identified across donors, or a phase bound is not a post-treatment
        period of the panel.
    MlsynthEstimationError
        If the frontier's quadratic programme cannot be solved.
    """
    _check(inputs, n_factors, n_subperiods, inference)
    masks = _phase_masks(inputs, phases)
    y1, Y0, pre, x = inputs.y, inputs.donor_matrix, inputs.pre, inputs.shock
    s1, s0, base = inputs.exposure_treated, inputs.exposure_donors, inputs.base_index
    post = ~pre

    Fl = D.donor_factors(Y0, n_factors, s0)
    ell, r2 = D.partial_leverage(x, Fl, pre)
    G_lat, G_full = D.with_constant(Fl), D.with_constant(Fl, x)
    kappa_lat = D.leverage(G_lat, pre)
    # With the shock in the factors' span before treatment (S_r = 0) its base-period
    # residual is zero, and Proposition 4(d)'s bound reduces to the latent leverage.
    ell_b = float(ell[base]) if np.isfinite(ell[base]) else 0.0
    phase_lev = {name: (D.phase_leverage(G_full, pre, m),
                        float(np.sqrt(D.phase_leverage(G_lat, pre, m) ** 2 + ell_b ** 2)))
                 for name, m in masks.items()}

    reg = C.exposure_effects(Y0, s0, pre, base, n_subperiods)
    w_sc = Fr.sc_weights(y1[pre], Y0[pre])
    fr = Fr.exposure_fit_frontier(y1[pre], Y0[pre], s1, s0, frontier_points, w_sc)
    crit = Fr.selection_criterion(fr.rmspe, fr.exposure_imbalance, kappa_lat, reg.se, post)
    selected = Fr.first_minimiser(crit)
    fits = {
        "sc": _fit("sc", inputs, w_sc, None),
        "easc": _fit("easc", inputs, w_sc, reg.phi),
        "ebsc_min": _fit("ebsc_min", inputs, fr.weights[0], None),
        "easc_tuned": _fit("easc_tuned", inputs, fr.weights[selected], reg.phi),
    }

    dmin = D.min_imbalance(s1, s0)
    if dmin > 0:
        warnings.warn(
            f"EASC: the treated unit's exposure ({s1:.4g}) lies outside the range of the donors' exposure "
            f"[{s0.min():.4g}, {s0.max():.4g}]. Simplex weights cannot balance it (Proposition 3) and the "
            "correction extrapolates the exposure regression (Remark 1); add donors whose exposure brackets "
            "the treated unit's, even if they receive no weight.", UserWarning, stacklevel=2)
    diagnostics = EASCDiagnostics(
        n_factors=n_factors, leverage=D.leverage(G_full, pre), latent_leverage=kappa_lat,
        partial_leverage=ell, shock_r2=r2, max_partial_leverage_pre=float(np.max(np.abs(ell[pre]))),
        max_partial_leverage_post=float(np.max(np.abs(ell[post]))), d_min=dmin, inside_hull=dmin == 0.0,
        exposure_effect=reg.phi, exposure_effect_se=reg.se,
        hat_values=C.hat_values(Y0, s0, pre, base, n_subperiods), phase_leverage=phase_lev)
    frontier = EASCFrontier(tolerance=fr.tolerance, exposure_imbalance=fr.exposure_imbalance, rmspe=fr.rmspe,
                            weights=fr.weights, criterion=crit, selected=selected)

    inf = None
    if inference:
        treated = dict(SC=fits["sc"].gap, EASC=fits["easc"].gap, ei=fits["easc"].exposure_imbalance, reg=reg,
                       rmspe=fits["sc"].pre_rmse)
        inf = _inference(inputs, treated, masks, phase_lev, n_subperiods=n_subperiods, alpha=alpha,
                         m_grid=m_grid, headline_m=headline_m, correction_se=correction_se)
    return EASCEstimates(fits=fits, diagnostics=diagnostics, frontier=frontier, inference=inf)
