"""Cattaneo-Feng-Titiunik prediction intervals for RPCA-SC.

Implements a focused port of the prediction-interval machinery from
*Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021), "Prediction
Intervals for Synthetic Control Methods"* (JASA 116(536):1865-1880)
and its multi-period extension in Cattaneo, Feng, Palomba, Titiunik
(2025). The reference implementation lives at
https://github.com/nppackages/scpi (Python package ``scpi_pkg``).

The full ``scpi_pkg`` ships an ECOS-based constrained optimisation
solver for the in-sample component and three out-of-sample methods
(``gaussian`` / ``ls`` / ``qreg``) plus dask-parallel simulation.
Vendoring that for one inference method on one estimator family
would pull in dask, plotnine, mizani, fsspec, cloudpickle, ecos,
luddite, and partd as hard dependencies. Instead this module
implements the same two-component structure with two lightweight
choices that capture the spirit of the paper:

**Prediction interval at post-period** :math:`t > T_0`:

.. math::

   \\mathrm{PI}_t = \\widehat y^0_t \\;\\pm\\;
                   \\bigl[ M_w(t, \\alpha/2) + M_e(t, \\alpha/2) \\bigr],

where the two components are quantified as follows.

* **In-sample uncertainty** :math:`M_w(t, \\alpha/2)` -- HC1-scaled
  parametric bootstrap of the pre-period residuals. At each of
  ``sims`` draws we perturb the treated pre-period outcome with a
  resampled-residual error and refit the full RPCA-SC pipeline via
  the user-supplied ``refit_fn``. The :math:`(\\alpha/2, 1-\\alpha/2)`
  empirical quantiles of the resulting counterfactual draws give an
  asymmetric in-sample band per post-period. This replaces the
  paper's ECOS solver, which solves a constrained
  :math:`\\sup / \\inf` over the "compatible set" of weights -- the
  bootstrap is equivalent under regularity conditions and avoids
  pulling in ``ecos`` as a dependency.

* **Out-of-sample uncertainty** :math:`M_e(t, \\alpha/2)` -- the
  ``gaussian`` variant of the paper's ``e_method``. Under
  sub-Gaussian post-period shocks the Hoeffding bound gives

  .. math::

     M_e(t, \\alpha/2) = \\sqrt{-2 \\log \\alpha} \\; \\widehat\\sigma_e,

  where :math:`\\widehat\\sigma_e = \\mathrm{sd}(\\widehat u^-)` is
  estimated from the pre-period residuals. The two more elaborate
  variants in the paper (``ls`` -- location-scale model;
  ``qreg`` -- quantile regression on residuals) are deferred for
  a future revision.

For the **average treatment effect on the treated**, the in-sample
component aggregates by simulating the *post-period mean of the
counterfactual* across bootstrap draws and taking quantiles. The
out-of-sample component aggregates as
:math:`\\widehat\\sigma_e \\sqrt{-2 \\log \\alpha} / \\sqrt{T_1}` under
post-period shock independence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError


@dataclass(frozen=True)
class CFTInference:
    """Output of :func:`cft_prediction_intervals`.

    Attributes
    ----------
    method : str
        Tag identifying the e_method used (currently ``"cft_gaussian"``).
    alpha : float
        Nominal level used to form the intervals.
    att : float
        Mean post-period gap.
    per_period_gap : np.ndarray
        Per-post-period treatment effect estimates, shape ``(T1,)``.
    per_period_pi : np.ndarray
        Per-post-period :math:`(1-\\alpha)` PI for the treatment
        effect :math:`\\tau_t`, shape ``(T1, 2)``.
    in_sample_band : np.ndarray
        Per-post-period in-sample bootstrap band on the counterfactual,
        shape ``(T1, 2)``. ``in_sample_band[t, 0]`` is the bootstrap
        :math:`\\alpha/2` quantile minus the point estimate;
        ``in_sample_band[t, 1]`` is the :math:`(1 - \\alpha/2)` quantile
        minus the point estimate.
    out_of_sample_eps : float
        Out-of-sample Hoeffding bound -- the same scalar at every
        post-period under the homoskedastic Gaussian default.
    att_pi : tuple of float
        Aggregated PI on the ATT.
    sims : int
        Number of bootstrap draws used.
    sigma_e : float
        Estimated pre-period residual standard deviation.
    """

    method: str
    alpha: float
    att: float
    per_period_gap: np.ndarray
    per_period_pi: np.ndarray
    in_sample_band: np.ndarray
    out_of_sample_eps: float
    att_pi: Tuple[float, float]
    sims: int
    sigma_e: float


def cft_prediction_intervals(
    treated_outcome: np.ndarray,
    counterfactual: np.ndarray,
    T0: int,
    refit_fn: Callable[[np.ndarray], np.ndarray],
    *,
    e_method: str = "gaussian",
    alpha: float = 0.05,
    sims: int = 200,
    random_state: int = 0,
) -> CFTInference:
    """Two-component prediction intervals for an RPCA-SC fit.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    counterfactual : np.ndarray
        Counterfactual from the actual fit, shape ``(T,)``.
    T0 : int
        Number of pre-treatment periods.
    refit_fn : callable
        ``refit_fn(y_perturbed) -> counterfactual`` where the
        argument is a length-``T`` modified treated outcome whose
        pre-period entries have been perturbed by a resampled residual
        draw, and the return is the resulting full-period
        counterfactual. The pipeline supplies this thunk so the
        inference module stays decoupled from the orchestration.
    e_method : {"gaussian"}
        Out-of-sample method. Currently only ``"gaussian"`` (Hoeffding
        bound) is supported.
    alpha : float
        Two-sided level. Default 0.05 -> 95% intervals.
    sims : int
        Number of bootstrap draws for the in-sample component.
    random_state : int
        Seed for the resampling RNG.
    """
    if e_method != "gaussian":
        raise MlsynthEstimationError(
            f"Unknown e_method {e_method!r}; expected 'gaussian'."
        )
    if not (0.0 < alpha < 1.0):
        raise MlsynthEstimationError("alpha must lie in (0, 1).")
    if sims < 10:
        raise MlsynthEstimationError("sims must be >= 10.")

    y = np.asarray(treated_outcome, dtype=float).flatten()
    cf = np.asarray(counterfactual, dtype=float).flatten()
    T = y.shape[0]
    T1 = T - T0
    if T1 < 1:
        raise MlsynthEstimationError("Need at least one post-period.")

    # Pre-period residuals + HC1 scaling so bootstrap noise has the
    # right variance under standard regression assumptions.
    u_pre = y[:T0] - cf[:T0]
    # Effective d.o.f.: deduct the number of fit parameters we don't
    # know exactly; conservative choice is to count the donor weights.
    # Use len(u_pre) - 1 as a robust fall-back when k is unknown.
    hc1_scale = np.sqrt(max(T0 - 1, 1) / T0)
    u_pre_scaled = u_pre / hc1_scale
    sigma_e = float(np.std(u_pre, ddof=1)) if T0 > 1 else float("nan")

    # In-sample bootstrap.
    rng = np.random.default_rng(random_state)
    cf_draws = np.zeros((sims, T))
    for s in range(sims):
        eps_star = rng.choice(u_pre_scaled, size=T0, replace=True)
        y_star = y.copy()
        y_star[:T0] = cf[:T0] + eps_star
        cf_draws[s] = np.asarray(refit_fn(y_star), dtype=float).flatten()

    # Per-period in-sample band on the counterfactual.
    cf_lo = np.quantile(cf_draws[:, T0:], alpha / 2.0, axis=0) - cf[T0:]
    cf_hi = np.quantile(cf_draws[:, T0:], 1.0 - alpha / 2.0, axis=0) - cf[T0:]
    in_sample = np.column_stack([cf_lo, cf_hi])

    # Out-of-sample Hoeffding bound (sub-Gaussian with sigma_e).
    # Paper uses sqrt(-log(alpha) * 2); we follow the same form but
    # interpret alpha as the one-sided tail mass.
    eps = float(np.sqrt(-2.0 * np.log(alpha)) * sigma_e) if T0 > 1 else 0.0

    # Per-period PI on the gap tau_t = y_t - cf_t. Adding/removing the
    # in-sample component is asymmetric, so:
    #   cf_t in [cf_t + cf_lo - eps, cf_t + cf_hi + eps]
    # => tau_t in [y_t - (cf_t + cf_hi + eps), y_t - (cf_t + cf_lo - eps)]
    gaps = y[T0:] - cf[T0:]
    tau_lo = y[T0:] - (cf[T0:] + cf_hi + eps)
    tau_hi = y[T0:] - (cf[T0:] + cf_lo - eps)
    per_period_pi = np.column_stack([tau_lo, tau_hi])

    # ATT-level PI:
    #   In-sample: bootstrap distribution of the average counterfactual.
    #   Out-of-sample: average of T1 independent sub-Gaussian shocks
    #     -> sigma_e / sqrt(T1) under independence.
    cf_post_means = cf_draws[:, T0:].mean(axis=1)
    cf_avg_lo = float(np.quantile(cf_post_means, alpha / 2.0)) - cf[T0:].mean()
    cf_avg_hi = float(np.quantile(cf_post_means, 1.0 - alpha / 2.0)) - cf[T0:].mean()
    eps_att = float(np.sqrt(-2.0 * np.log(alpha)) * sigma_e / np.sqrt(T1)) \
        if T0 > 1 else 0.0
    att = float(gaps.mean())
    att_lo = att - cf_avg_hi - eps_att
    att_hi = att - cf_avg_lo + eps_att

    return CFTInference(
        method=f"cft_{e_method}",
        alpha=float(alpha),
        att=att,
        per_period_gap=gaps,
        per_period_pi=per_period_pi,
        in_sample_band=in_sample,
        out_of_sample_eps=eps,
        att_pi=(att_lo, att_hi),
        sims=int(sims),
        sigma_e=sigma_e,
    )
