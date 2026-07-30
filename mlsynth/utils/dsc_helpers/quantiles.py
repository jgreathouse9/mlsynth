"""Empirical quantile functions and Wasserstein-aligned pseudo-samples for DSC.

DSC fits weights on the *quantile functions* of donor and treated
outcomes. Given the empirical CDF :math:`\\widehat F_{Y_{jt, n_j}}`
constructed from the within-cell sample :math:`\\{Y_{l, jt}\\}_{l=1}^{n_j}`,
its quantile function is

.. math::

   \\widehat F^{-1}_{Y_{jt, n_j}}(q)
       = Y_{t, n_j(k)}, \\quad
         \\frac{k - 1}{n_j} < q \\le \\frac{k}{n_j},

where :math:`Y_{t, n_j(k)}` is the :math:`k`-th order statistic of the
sample. This module exposes:

* :func:`empirical_quantile` -- evaluate
  :math:`\\widehat F^{-1}_{Y_{jt, n_j}}` at a vector of quantile points
  via NumPy's ``inverted_cdf`` rule (matches the paper's order-statistic
  estimator).
* :func:`sample_quantile_grid` -- draw the :math:`M` quantile points
  :math:`\\{V_m\\}_{m=1}^{M}` used to build pseudo-samples. Supports
  uniform i.i.d. draws and the Halton / Sobol low-discrepancy
  sequences (Koksma-Hlawka error :math:`O(\\log M / M)` vs.
  :math:`O(M^{-1/2})` for i.i.d.).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import qmc

from ...exceptions import MlsynthEstimationError


def empirical_quantile(
    sample: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Evaluate the order-statistic empirical quantile function at ``quantiles``.

    Parameters
    ----------
    sample : np.ndarray
        Length-``n`` 1-D sample of observed outcomes for a single
        ``(unit, time)`` cell.
    quantiles : np.ndarray
        Quantile probabilities in ``[0, 1]``, shape ``(M,)``. The endpoints
        are allowed and evaluate the sample minimum and maximum; the
        reference implementations both include them in the grid.

    Returns
    -------
    np.ndarray
        Shape ``(M,)`` evaluations of
        :math:`\\widehat F^{-1}_{Y_{jt, n_j}}(q)`.
    """
    if sample.ndim != 1 or sample.size == 0:
        raise MlsynthEstimationError(
            "empirical_quantile expects a non-empty 1-D sample."
        )
    if np.any((quantiles < 0.0) | (quantiles > 1.0)) or np.any(~np.isfinite(quantiles)):
        raise MlsynthEstimationError(
            "quantiles must be finite and lie in the closed interval [0, 1]."
        )
    # method="linear" is R's quantile type 7. Both papers define the
    # generalized inverse (type 1, numpy "inverted_cdf"), but BOTH official
    # implementations use type 7 -- DiSCos passes qtype = 7 and the Stata
    # command's disco_quantile_sorted interpolates as (N-1)p + 1. Matching a
    # shipped reference beats matching prose that neither author implemented;
    # test_dsc_quantile_convention.py pins the distinction so the choice stays
    # visible. Using type 1 instead moves the fitted weights by ~0.003 on the
    # Stata Journal tenure panel.
    return np.quantile(sample, quantiles, method="linear")


def sample_quantile_grid(
    M: int,
    method: Literal["equidistant", "halton", "sobol", "uniform"] = "equidistant",
    random_state: int = 0,
) -> np.ndarray:
    """Build the quantile grid :math:`\\{V_m\\}_{m=1}^{M} \\subset [0, 1]`.

    Parameters
    ----------
    M : int
        Number of quantile points.
    method : {"equidistant", "halton", "sobol", "uniform"}
        Grid rule. ``"equidistant"`` (default) is
        ``V_m = (m - 1) / (M - 1)``, the closed grid used by the authors'
        Stata implementation (``disco_prob_grid`` in ``disco_utils.mata``)
        and the only rule that reproduces its published weights. The
        endpoints are *kept*: :math:`V = 0` and :math:`V = 1` evaluate the
        sample minimum and maximum, and dropping them moves the fitted
        weights by ~0.09 on the Stata Journal tenure panel -- the single
        largest source of disagreement with the reference.
        ``"halton"`` and ``"sobol"`` give deterministic low-discrepancy
        sequences with Koksma-Hlawka error :math:`O(\\log M / M)`;
        ``"uniform"`` draws i.i.d. samples with :math:`O(M^{-1/2})` error,
        which is what Gunsilius (2023, eq. 3) writes and what the R package
        does -- and is why that package's weights vary with its seed.
    random_state : int
        Seed for the QMC scrambling / i.i.d. RNG. Unused by
        ``"equidistant"``, which is deterministic.

    Returns
    -------
    np.ndarray
        Length-``M`` quantile points. Closed ``[0, 1]`` for
        ``"equidistant"``; open ``(0, 1)`` for the sampling rules, whose
        draws are clipped away from the endpoints.
    """
    if M < 2:
        raise MlsynthEstimationError("M must be >= 2.")
    if method == "equidistant":
        # Matches disco_prob_grid: p[j] = q_min + (q_max - q_min)*(j-1)/(N-1).
        return np.linspace(0.0, 1.0, M)
    if method == "halton":
        sampler = qmc.Halton(d=1, seed=random_state)
        V = sampler.random(n=M).flatten()
    elif method == "sobol":
        sampler = qmc.Sobol(d=1, seed=random_state)
        V = sampler.random(n=M).flatten()
    elif method == "uniform":
        rng = np.random.default_rng(random_state)
        V = rng.uniform(low=0.0, high=1.0, size=M)
    else:
        raise MlsynthEstimationError(
            f"Unknown quantile-grid method {method!r}; expected one of "
            "'equidistant', 'halton', 'sobol', 'uniform'."
        )
    # Map any exact 0 / 1 draws into the open interval so that
    # empirical_quantile does not refuse them.
    eps = 1.0 / (2.0 * max(M, 2))
    V = np.clip(V, eps, 1.0 - eps)
    return V


def build_pseudo_sample_matrix(
    inputs,
    time_label,
    quantile_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct :math:`(\\widetilde{Y}_t, \\widehat{Y}_{1t})` for one period.

    Returns
    -------
    donor_matrix : np.ndarray
        Shape ``(M, J)`` -- column ``j`` is the donor's quantile
        function evaluated at the grid.
    treated_vec : np.ndarray
        Shape ``(M,)`` -- the treated unit's quantile function
        evaluated at the grid.
    """
    treated_sample = inputs.cell_samples[(inputs.unit_names[0], time_label)]
    treated_vec = empirical_quantile(treated_sample, quantile_grid)
    donor_matrix = np.column_stack([
        empirical_quantile(
            inputs.cell_samples[(unit, time_label)], quantile_grid,
        )
        for unit in inputs.unit_names[1:]
    ])
    return donor_matrix, treated_vec
