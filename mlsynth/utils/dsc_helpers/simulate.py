"""Data-generating process for the DSC Path-B benchmark (Zhang et al. 2026, Sec. 5.1).

Re-implements the model-free Monte Carlo of Zhang, L., Zhang, X. & Zhang, X.
(2026), *"Asymptotic Properties of the Distributional Synthetic Controls"*
(arXiv:2405.00953v3), the paper whose Algorithm 1 mlsynth's :class:`~mlsynth.DSC`
follows. The design exists to illustrate the paper's two theorems, and both are
measured against a population quantity this module supplies in closed form:

* Theorem 1 (asymptotic optimality) -- the averaged post-treatment risk at the
  fitted weight, :math:`\\bar R_{T_1}(\\widehat w)`, approaches the smallest
  risk any simplex weight attains, :math:`\\inf_{w \\in \\mathcal H} \\bar
  R_{T_1}(w)`, as the number of draws :math:`M` grows.
* Theorem 2 (weight convergence) -- :math:`\\widehat w` approaches the
  risk-minimising :math:`w^{\\mathrm{opt}}_{T_1}`, and more slowly as the donor
  pool :math:`J` grows.

The design
----------

For every period :math:`t` and draw :math:`m`, the treated unit's pseudo-sample
is :math:`\\widetilde Y_{1tm} \\sim \\chi^2(2)` and each donor's is
:math:`\\widetilde Y_{jtm} \\sim \\mathcal N(\\mu_j, \\sigma_j^2)`, with
:math:`\\mu_j \\sim U(3, 10)` and :math:`\\sigma_j = 3` for odd unit labels
:math:`j`, 2.5 for even ones. The draws are taken at ranks :math:`\\{V_m\\}`
generated in dependent pairs: :math:`M/2` ranks :math:`V^{(1)}_k \\sim U[0, 1]`
and their partners :math:`V^{(2)}_k = V^{(1)}_k \\pm \\delta` with
:math:`\\delta = 0.01`, shifted toward the centre of the unit interval. The
dependence across :math:`m` is the point of the construction: the paper's
Assumption 4 relaxes Gunsilius (2023)'s independence requirement to a mixing
condition, and the paired draws are its illustration.

The paper reports :math:`T_0 = 10` pre-treatment periods, :math:`T_1 = 5` post
periods, equal aggregation weights :math:`\\lambda_t = 1/T_0`, and
:math:`R = 1000` replications over :math:`J \\in \\{20, 50\\}` and
:math:`M \\in \\{50, 100, 200, 400\\}`.

Two details are inferred (scenario-1 disclosure)
------------------------------------------------

1. Units are drawn independently -- each unit carries its own rank sequence.
   The paper is not consistent on this. Section 2 defines the risk as the
   comonotonic integral :math:`\\int_0^1 (\\sum_j w_j F^{-1}_{Y_{jt}}(q) -
   F^{-1}_{Y_{1t}}(q))^2 dq`, which is one rank sequence shared by every unit.
   The moment matrix the paper writes for that same risk has off-diagonal
   entries :math:`\\mu_i \\mu_j`, which is independence across units, and the
   companion quantile-factor design generates its pseudo-sample with no ranks
   at all. Only the independent reading reproduces the published figures; under
   the shared-rank reading the estimator converges far faster than they show
   (a risk ratio of 1.002 against their 1.024 at :math:`J = 20`, :math:`M = 50`).
   :func:`model_free_risk_matrices` is the moment matrix implied by that same
   reading, so the estimator and the target it is measured against agree.
2. The rank sequence is redrawn every period, and :math:`\\mu_j` is redrawn
   every replication. Neither is stated.

Because the design is time-invariant, the averaged post-treatment risk does not
depend on :math:`t` and reduces to a single quadratic form, which is why the
oracle here is exact algebra and not a nested simulation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import ndtri

from ...exceptions import MlsynthEstimationError
from .weights import solve_simplex_weights

_TREATED_DF = 2                 # the treated unit's chi2 degrees of freedom
_DELTA = 0.01                   # rank-pair offset
_MU_LOW, _MU_HIGH = 3.0, 10.0   # donor location support
_SIGMA_ODD, _SIGMA_EVEN = 3.0, 2.5


def donor_scales(J: int) -> np.ndarray:
    """Donor standard deviations: 3 for odd unit labels, 2.5 for even ones.

    The donors carry labels :math:`j = 2, \\ldots, J + 1`, so the first donor
    is even and the sequence starts at 2.5.
    """
    if J < 1:
        raise MlsynthEstimationError("J must be at least 1.")
    labels = np.arange(2, J + 2)
    return np.where(labels % 2 == 1, _SIGMA_ODD, _SIGMA_EVEN).astype(float)


def draw_model_free_design(
    J: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one design: donor locations ``mu ~ U(3, 10)`` and their scales."""
    if J < 1:
        raise MlsynthEstimationError("J must be at least 1.")
    return rng.uniform(_MU_LOW, _MU_HIGH, size=J), donor_scales(J)


def paired_uniform_ranks(
    M: int, rng: np.random.Generator, delta: float = _DELTA
) -> np.ndarray:
    """Draw ``M`` ranks in dependent pairs :math:`\\delta` apart.

    ``M / 2`` ranks are drawn uniformly and each is given a partner offset by
    ``delta`` toward the centre of the unit interval, so every value stays in
    ``(0, 1)``. Returns the first-of-pair draws followed by their partners.
    """
    if M < 2:
        raise MlsynthEstimationError("M must be at least 2.")
    if M % 2:
        raise MlsynthEstimationError("M must be even: the ranks come in pairs.")
    if not 0.0 < delta < 0.5:
        raise MlsynthEstimationError("delta must lie strictly between 0 and 0.5.")
    first = rng.uniform(size=M // 2)
    second = np.where(first < 0.5, first + delta, first - delta)
    return np.concatenate([first, second])


def simulate_model_free_period(
    mu: np.ndarray,
    sigma: np.ndarray,
    M: int,
    rng: np.random.Generator,
    delta: float = _DELTA,
) -> Tuple[np.ndarray, np.ndarray]:
    """One period's pseudo-samples: an ``(M, J)`` donor matrix and treated vector.

    Every unit -- each donor and the treated unit -- gets its own paired rank
    sequence, which is the independent-units reading documented at module level.
    The treated unit's quantile function is that of :math:`\\chi^2(2)`, which is
    exponential with mean 2, so :math:`F^{-1}(q) = -2 \\log(1 - q)`.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.shape != sigma.shape:
        raise MlsynthEstimationError("mu and sigma must have the same length.")
    J = mu.size
    ranks = np.column_stack([paired_uniform_ranks(M, rng, delta)
                             for _ in range(J + 1)])
    donors = mu[None, :] + sigma[None, :] * ndtri(ranks[:, :J])
    treated = -float(_TREATED_DF) * np.log1p(-ranks[:, J])
    return donors, treated


def model_free_risk_matrices(
    mu: np.ndarray, sigma: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Closed-form pieces of the population risk :math:`\\bar R_{T_1}(w)`.

    With the units drawn independently, the risk
    :math:`\\mathbb E[(\\sum_j w_j \\widetilde Y_j - \\widetilde Y_1)^2]`
    is the quadratic ``w' G w - 2 b' w + const`` with

    * ``G = mu mu' + diag(sigma^2)`` -- the donor second-moment matrix,
    * ``b = mu_1 * mu = 2 * mu`` -- the treated-donor cross moments,
    * ``const = E[Y_1^2] = k^2 + 2k = 8`` for :math:`\\chi^2(k)`, ``k = 2``.

    These are the entries the paper lists for its moment matrix. The design is
    time-invariant, so the average over post-treatment periods is this same
    quadratic.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.shape != sigma.shape:
        raise MlsynthEstimationError("mu and sigma must have the same length.")
    G = np.outer(mu, mu) + np.diag(sigma ** 2)
    b = float(_TREATED_DF) * mu
    const = float(_TREATED_DF ** 2 + 2 * _TREATED_DF)
    return G, b, const


def population_risk(
    w: np.ndarray, G: np.ndarray, b: np.ndarray, const: float
) -> float:
    """Evaluate ``w' G w - 2 b' w + const``, the population risk at ``w``."""
    w = np.asarray(w, dtype=float)
    return float(w @ G @ w - 2.0 * w @ b + const)


def oracle_simplex_weights(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The infeasible optimum :math:`w^{\\mathrm{opt}} = \\arg\\min_{w \\in \\mathcal H} \\bar R_{T_1}(w)`.

    Solved through the estimator's own solver so the oracle and the estimate
    come from one code path. ``G`` is positive definite here, so its Cholesky
    factor :math:`G = L L^{\\top}` turns the quadratic into the least-squares
    problem :math:`\\| L^{\\top} w - L^{-1} b \\|^2` up to an additive constant.
    """
    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float)
    if G.shape[0] == 1:
        return np.ones(1)
    L = np.linalg.cholesky(G)
    return solve_simplex_weights(L.T, np.linalg.solve(L, b))


def model_free_replication(
    *, J: int, M: int, T0: int = 10, seed: int = 0, delta: float = _DELTA
) -> Tuple[float, float]:
    """One Monte-Carlo replication of the Section 5.1 design.

    Draws the design, fits :math:`\\widehat w_t` on each of ``T0`` pre-treatment
    periods with :func:`~mlsynth.utils.dsc_helpers.weights.solve_simplex_weights`,
    averages them with equal weights :math:`\\lambda_t = 1/T_0`, and scores the
    result against the population optimum.

    Returns
    -------
    tuple of float
        ``(risk_ratio, weight_error)`` -- the Theorem 1 quantity
        :math:`\\bar R_{T_1}(\\widehat w) / \\inf_w \\bar R_{T_1}(w)` and the
        Theorem 2 quantity :math:`\\| \\widehat w - w^{\\mathrm{opt}} \\|_2`.
    """
    if J < 1:
        raise MlsynthEstimationError("J must be at least 1.")
    if T0 < 1:
        raise MlsynthEstimationError("T0 must be at least 1.")
    if M < 2:
        raise MlsynthEstimationError("M must be at least 2.")
    if M % 2:
        raise MlsynthEstimationError("M must be even: the ranks come in pairs.")

    rng = np.random.default_rng(seed)
    mu, sigma = draw_model_free_design(J, rng)
    G, b, const = model_free_risk_matrices(mu, sigma)
    w_opt = oracle_simplex_weights(G, b)
    best = population_risk(w_opt, G, b, const)

    fitted = np.empty((T0, J))
    for t in range(T0):
        donors, treated = simulate_model_free_period(mu, sigma, M, rng, delta)
        fitted[t] = solve_simplex_weights(donors, treated)
    w_hat = fitted.mean(axis=0)

    return (population_risk(w_hat, G, b, const) / best,
            float(np.linalg.norm(w_hat - w_opt)))
