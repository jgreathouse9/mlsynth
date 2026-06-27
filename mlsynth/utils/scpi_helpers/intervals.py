"""CFPT/scpi prediction-interval construction.

Out-of-sample bands (Section 4.2 of Cattaneo, Feng, Palomba & Titiunik 2025)
for the four causal predictands and their simultaneous (uniform-over-periods)
versions, plus the in-sample simulation bound (Section 4.1) as a reusable
helper for *quadratic-loss* SC estimators.

All bands are returned as :class:`SCPIBand` objects and assembled into a
:class:`SCPIResults`. The interval for a predictand ``tau`` is

    [ tau_hat - Mbar_in - Mbar_out ,  tau_hat - M_in - M_out ].

When the in-sample error is not modelled (``in_sample_included=False``, e.g.
MSQRT), the in-sample contribution is ``(0, 0)`` and the full miscoverage
budget ``alpha`` is spent on the out-of-sample band.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .moments import unit_moments
from .structures import SCPIBand, SCPIResults


def _half(sigma: float, alpha: float) -> float:
    """Sub-Gaussian half-width ``sqrt(2 sigma^2 log(2/alpha))``."""
    if sigma <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * sigma ** 2 * np.log(2.0 / alpha)))


def _band(predictand, label, point, mu, sigma, alpha) -> SCPIBand:
    """One out-of-sample band: recentre by ``mu`` and add the concentration
    half-width. ``M_out = mu - h``, ``Mbar_out = mu + h``; interval is
    ``[point - Mbar_out, point - M_out]``.
    """
    h = _half(sigma, alpha)
    m_out, mbar_out = mu - h, mu + h
    return SCPIBand(
        predictand=predictand, label=label, point=float(point),
        lower=float(point - mbar_out), upper=float(point - m_out),
        out_sample=(float(m_out), float(mbar_out)),
    )


def _resolve_unit_weights(unit_weights, units) -> np.ndarray:
    """Normalised cross-section weights aligned to ``units`` order.

    Accepts a ``{unit: weight}`` mapping. Raises ``ValueError`` on a missing
    unit, any negative weight, or a non-positive total. Returns ``omega`` with
    ``sum(omega) == 1`` -- the convex weights for the size-weighted predictand.
    """
    try:
        w = np.array([float(unit_weights[u]) for u in units], dtype=float)
    except KeyError as e:
        raise ValueError(f"unit_weights is missing a weight for unit {e}.") from e
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("unit_weights must be finite and non-negative.")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("unit_weights must sum to a positive value.")
    return w / total


def out_of_sample_intervals(
    effects: np.ndarray,
    pre_residuals: np.ndarray,
    unit_names: Sequence,
    period_labels: Sequence,
    *,
    alpha: float = 0.1,
    time_dependence: str = "iid",
    assume_zero_mean: bool = False,
    unit_weights: Optional[Dict] = None,
) -> SCPIResults:
    """Build the full CFPT out-of-sample interval family for a block design.

    Parameters
    ----------
    effects : np.ndarray
        ``(L, m)`` post-period predicted effects ``tau_hat_ik`` (rows =
        post-periods, columns = treated units).
    pre_residuals : np.ndarray
        ``(T0, m)`` pre-period SC residuals (one column per treated unit).
    unit_names : sequence
        Length-``m`` treated-unit identifiers.
    period_labels : sequence
        Length-``L`` post-period labels.
    alpha : float
        Total (and, here, out-of-sample) miscoverage level.
    time_dependence : {"iid", "general"}
        Time-averaging bound for the time-averaged predictands (TAUS, TAUA).
        ``"iid"`` shrinks the band by ``sqrt(L)`` (residuals independent over
        time); ``"general"`` makes no dependence assumption (no shrinkage).
    assume_zero_mean : bool
        Pass-through to the conditional-moment estimator.
    """
    effects = np.asarray(effects, dtype=float)
    L, m = effects.shape
    units = list(unit_names)
    periods = list(period_labels)
    mu, sigma = unit_moments(pre_residuals, units, assume_zero_mean=assume_zero_mean)

    # ---- TSUS: per unit, per period -------------------------------------
    tsus: Dict[Tuple, SCPIBand] = {}
    for j, u in enumerate(units):
        for k, p in enumerate(periods):
            tsus[(u, p)] = _band("TSUS", (u, p), effects[k, j], mu[u], sigma[u], alpha)

    # ---- TAUS: per unit, averaged over time -----------------------------
    # general: sigma_i. = mean_k sigma  (= sigma_u, constant)  -> no shrink
    # iid    : sigma_i. = sqrt(sum sigma^2)/L = sigma_u / sqrt(L)
    taus: Dict = {}
    for j, u in enumerate(units):
        point = float(effects[:, j].mean())
        s = sigma[u] / np.sqrt(L) if (time_dependence == "iid" and L > 0) else sigma[u]
        taus[u] = _band("TAUS", u, point, mu[u], s, alpha)

    mu_bar = float(np.mean(list(mu.values()))) if mu else 0.0
    sigma_bar = float(np.mean(list(sigma.values()))) if sigma else 0.0  # cross-section: general

    # ---- TSUA: per period, averaged over units (cross-section general) --
    tsua: Dict = {}
    for k, p in enumerate(periods):
        point = float(effects[k, :].mean())
        tsua[p] = _band("TSUA", p, point, mu_bar, sigma_bar, alpha)

    # ---- TAUA: overall average ------------------------------------------
    # cross-section general (mean sigma); time iid shrinks by sqrt(L).
    point_all = float(effects.mean())
    s_taua = sigma_bar / np.sqrt(L) if (time_dependence == "iid" and L > 0) else sigma_bar
    taua = _band("TAUA", None, point_all, mu_bar, s_taua, alpha)

    # ---- Size-weighted unit aggregation (optional) ----------------------
    # Same TSUA/TAUA predictands but with a convex combination of treated units
    # by user weights omega (e.g. market size) instead of 1/m. Per CFPT's
    # decomposition the aggregate error is sum_i omega_i (in_sample_i +
    # out_sample_i), so point, mu and sigma all combine with the same omega.
    taua_weighted: Optional[SCPIBand] = None
    tsua_weighted: Optional[Dict] = None
    if unit_weights is not None:
        omega = _resolve_unit_weights(unit_weights, units)
        mu_w = float(sum(omega[j] * mu[u] for j, u in enumerate(units)))
        sigma_w = float(sum(omega[j] * sigma[u] for j, u in enumerate(units)))
        tsua_weighted = {}
        for k, p in enumerate(periods):
            point = float(effects[k, :] @ omega)
            tsua_weighted[p] = _band("TSUA", p, point, mu_w, sigma_w, alpha)
        point_all_w = float((effects @ omega).mean())
        s_taua_w = sigma_w / np.sqrt(L) if (time_dependence == "iid" and L > 0) else sigma_w
        taua_weighted = _band("TAUA", None, point_all_w, mu_w, s_taua_w, alpha)

    # ---- Simultaneous: TSUS widened for joint coverage over k -----------
    # Bonferroni over the L periods: spend alpha/L per period.
    simultaneous: Dict = {}
    alpha_simul = alpha / max(L, 1)
    for j, u in enumerate(units):
        bands = [_band("TSUS", (u, p), effects[k, j], mu[u], sigma[u], alpha_simul)
                 for k, p in enumerate(periods)]
        simultaneous[u] = bands

    return SCPIResults(
        method="cfpt_scpi", alpha=float(alpha), alpha_in=0.0, alpha_out=float(alpha),
        in_sample_included=False, taua=taua, tsua=tsua, taus=taus, tsus=tsus,
        simultaneous=simultaneous, sigma=sigma, time_dependence=time_dependence,
        metadata={"L": int(L), "n_treated": int(m),
                  "alpha_simultaneous_per_period": float(alpha_simul)},
        taua_weighted=taua_weighted, tsua_weighted=tsua_weighted,
    )


# ---------------------------------------------------------------------------
# In-sample simulation bound (reusable; for quadratic-loss SC estimators).
# Not used by MSQRT, whose square-root-lasso loss does not satisfy the
# quadratic optimality condition the bound is derived from.
# ---------------------------------------------------------------------------

def in_sample_band_gaussian(
    donor_post: np.ndarray,
    Q_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    *,
    alpha_in: float = 0.05,
    n_sim: int = 1000,
    random_state: int = 0,
) -> Tuple[float, float]:
    """In-sample error band for a TSUS counterfactual, **unconstrained** weights.

    Implements Cattaneo et al.'s simulation (their eq. 4.4) for the special case
    of an unconstrained donor regression, where the feasible set ``Delta*`` is
    all of ``R^J0`` and the inf/sup of the donor projection over the quadratic
    constraint ``delta' Q_hat delta - 2 G*' delta <= 0`` has the closed form

        a' Q_hat^{-1} G*  ±  sqrt( (G*' Q_hat^{-1} G*) (a' Q_hat^{-1} a) ),

    with ``a = donor_post``. Draws ``G* ~ N(0, Sigma_hat)`` and returns the
    ``(alpha_in/2, 1 - alpha_in/2)`` quantiles ``(M_in, Mbar_in)``.

    Parameters
    ----------
    donor_post : np.ndarray
        ``(J0,)`` donor outcomes at the post-treatment period.
    Q_hat : np.ndarray
        ``(J0, J0)`` donor Gram matrix ``Y_N' Y_N`` over the pre-period.
    Sigma_hat : np.ndarray
        ``(J0, J0)`` estimate of ``V[gamma_hat | H]``.
    """
    a = np.asarray(donor_post, dtype=float).ravel()
    Q = np.asarray(Q_hat, dtype=float)
    Qinv = np.linalg.pinv(Q)
    aQinv_a = float(a @ Qinv @ a)
    rng = np.random.default_rng(random_state)
    G = rng.multivariate_normal(np.zeros(a.size), np.asarray(Sigma_hat, float), size=n_sim)
    centre = G @ Qinv @ a                          # a' Qinv G* for each draw
    radius = np.sqrt(np.maximum(np.einsum("sj,jk,sk->s", G, Qinv, G), 0.0) * max(aQinv_a, 0.0))
    lo = centre - radius
    hi = centre + radius
    m_in = float(np.quantile(lo, alpha_in / 2.0))
    mbar_in = float(np.quantile(hi, 1.0 - alpha_in / 2.0))
    return m_in, mbar_in
