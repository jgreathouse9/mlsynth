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
        Quantile probabilities in ``(0, 1)``, shape ``(M,)``.

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
    if np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
        raise MlsynthEstimationError(
            "quantiles must lie in the open interval (0, 1)."
        )
    return np.quantile(sample, quantiles, method="inverted_cdf")


def sample_quantile_grid(
    M: int,
    method: Literal["halton", "sobol", "uniform"] = "halton",
    random_state: int = 0,
) -> np.ndarray:
    """Draw the quantile-grid :math:`\\{V_m\\}_{m=1}^{M} \\subset (0, 1)`.

    Parameters
    ----------
    M : int
        Number of quantile points.
    method : {"halton", "sobol", "uniform"}
        Sampling rule. ``"halton"`` (default) and ``"sobol"`` give
        deterministic low-discrepancy sequences with Koksma-Hlawka
        error :math:`O(\\log M / M)`; ``"uniform"`` draws i.i.d.
        samples with :math:`O(M^{-1/2})` error.
    random_state : int
        Seed for the QMC scrambling / i.i.d. RNG.

    Returns
    -------
    np.ndarray
        Length-``M`` quantile points in ``(0, 1)``.
    """
    if M < 2:
        raise MlsynthEstimationError("M must be >= 2.")
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
            "'halton', 'sobol', 'uniform'."
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
