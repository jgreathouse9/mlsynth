"""Error-in-variables prediction intervals for the synthetic control.

Hirshberg (2021), *"Least Squares with Error in Variables"* (arXiv 2104.08931).
The synthetic control is a least-squares fit on donor pre-outcomes that are
themselves noisy (``X = A + eps``: a low-rank signal plus idiosyncratic noise),
i.e. an *error-in-variables* regression. Hirshberg's Corollary 3 gives conditions
-- a low-rank panel, mild Tikhonov regularisation, and enough pre-treatment
periods (``T0 -> inf``) -- under which the estimator

    tau_hat = y_e - x_e' theta_hat

is asymptotically normal around the treatment effect with an *estimable*
variance, and *without* assuming time-stationarity, unit-exchangeability, or the
absence of weak factors. The z-statistic ``(tau_hat - tau) / sigma_tau`` is then
standard normal with ``sigma_tau = sigma_e * p_eff**(-1/2)``, where ``p_eff`` is
the weights' participation ratio (``1 / ||theta||^2``; for equal weights on
``k`` donors it is ``k``).

The estimator itself (Hirshberg eq. 1) is Tikhonov-regularised least squares; for
the standard factor-model case (roughly i.i.d. donor noise) the penalty reduces
to ridge with scale ``eta`` and ``eta = 1`` recovers ordinary synthetic control.
This module therefore reads the *fitted* simplex weights (``eta = 1``) and forms
the inference; a ridge backend supplies ``eta > 1``.

Estimating ``sigma_tau``. Hirshberg's ``sigma_e * ||theta||`` captures only the
donor-noise term ``eps_e' theta`` and is valid in his scaling where the treated
unit's own last-period noise ``nu_e`` is ``p**(-1/2)``-negligible. For a single
treated unit with ``O(1)`` noise that term matters, so we estimate the full
per-period prediction-error scale directly from the pre-treatment residuals
``u_t = y_t - x_t' theta_hat`` (which are distributed like the post-period error
``nu_e - eps_e' theta``): ``sigma_hat = sqrt( sum u_t^2 / (T0 - df) )`` with
``df = (#nonzero weights) - 1`` the simplex degrees of freedom. In Monte Carlo on
the paper's low-rank DGP this recovers coverage that rises toward the nominal
level as ``T0`` grows (Hirshberg's asymptotic regime), whereas the donor-only
``sigma_e ||theta||`` badly under-covers a real single unit. Intervals use a
``t(T0 - df)`` reference; ``sigma_e_donor = sigma_hat * ||theta||`` and ``p_eff``
are reported as diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_NZ = 1e-6


@dataclass
class EIVResult:
    tau: np.ndarray                 # per-period effect (post)
    lower: np.ndarray               # effect PI lower (post)
    upper: np.ndarray               # effect PI upper (post)
    cf_lower: np.ndarray            # counterfactual band lower (post)
    cf_upper: np.ndarray            # counterfactual band upper (post)
    att: float
    att_lower: float
    att_upper: float
    metadata: dict = field(default_factory=dict)


def eiv_intervals(y, Y0, pre, W, *, alpha: float = 0.05) -> EIVResult:
    """Hirshberg (2021) error-in-variables prediction intervals for a simplex SC.

    Parameters
    ----------
    y : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    Y0 : np.ndarray
        Donor outcomes, shape ``(T, J)`` (columns match ``W``).
    pre : int
        Number of pre-treatment periods ``T0``.
    W : np.ndarray
        Fitted simplex donor weights, shape ``(J,)`` (Hirshberg eq. 1, eta=1).
    alpha : float
        Two-sided level (``1 - alpha`` intervals).
    """
    from scipy import stats

    y = np.asarray(y, float).ravel()
    Y0 = np.asarray(Y0, float)
    W = np.clip(np.asarray(W, float).ravel(), 0.0, None)

    cf = Y0 @ W                                     # synthetic counterfactual, all periods
    gap = y - cf                                    # observed - synthetic
    u = gap[:pre]                                   # pre-treatment residuals

    n_active = int(np.sum(W > _NZ))
    df = max(n_active - 1, 0)                        # simplex degrees of freedom
    dof = max(pre - df, 1)
    sigma = float(np.sqrt(np.sum(u ** 2) / dof))    # per-period prediction-error scale

    l2 = float(np.sqrt(np.sum(W ** 2)))
    p_eff = float(1.0 / np.sum(W ** 2)) if np.any(W > 0) else float(n_active)
    tq = float(stats.t.ppf(1.0 - alpha / 2.0, dof))

    tau = gap[pre:]
    half = tq * sigma
    lower = tau - half
    upper = tau + half
    cf_lower = y[pre:] - upper                       # cf band = y - effect band
    cf_upper = y[pre:] - lower

    tau_post = tau
    att = float(np.mean(tau_post))
    T1 = int(tau_post.size)
    att_se = sigma / np.sqrt(max(T1, 1))             # independence across post periods
    att_lower = att - tq * att_se
    att_upper = att + tq * att_se

    return EIVResult(
        tau=tau, lower=lower, upper=upper, cf_lower=cf_lower, cf_upper=cf_upper,
        att=att, att_lower=att_lower, att_upper=att_upper,
        metadata={
            "sigma_tau": sigma, "dof": dof, "t_quantile": tq,
            "p_eff": p_eff, "theta_l2": l2,
            "n_active": n_active, "att_se": float(att_se),
        },
    )
