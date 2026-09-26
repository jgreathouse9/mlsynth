"""SL's inference: the test of Equations 7-8 and the bootstrap of Algorithm 2.

What this supplies is a hypothesis test with size control (Theorem 3.1), and a
bias-adjusted point estimate (Equation 10). It does not supply a standard error
or a confidence interval, and neither does the paper: the only interval in its
figures is labelled ``(GSC)``, belonging to the generalised synthetic control
comparator, and the ``SE_TT <- sd(bootsr$t)`` in their ``placebo_test.R`` is the
spread of the test statistic's null distribution, assigned twice and never used.

The statistic is a non-negative quadratic, so a percentile of its null
distribution is a critical value. Reading one as an endpoint for a signed mean
is not available, whatever the variable it is stored in is called. An interval
for the effect would come from inverting this test over a grid of candidate
constant effects, collecting the ones it does not reject. That is not
implemented here, and mlsynth's ``conformal_att_interval`` does not substitute
for it: that function refits a ridge on a donor design, so its interval belongs
to a different estimator's point estimate.

One divergence from their code, and it is a correction. Their
``function4boot_TE`` refits the ensemble with ``eta = 1`` hard-coded
(``library.R:214``) while the observed statistic uses ``1/(sqrt(88) var(y))``,
which is 51.43 on their panel. The critical values then describe a
near-equal-weighted ensemble while the statistic they gate describes an
exponentially-weighted one. Measured on the paper's own two raw blocks, that
inflates the 10 percent critical value by 19 and 61 percent, so the test is
conservative: one block's non-rejection goes from comfortable (0.689 against
1.266) to 3 percent inside the 20 percent critical value (0.689 against 0.710).
Here the bootstrap refits with the same ``eta`` as the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from .weights import exponential_weights

#: Levels a fit reports critical values at, unless the caller asks otherwise.
LEVELS: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class BootstrapResult:
    """The test: the observed statistic, its null quantiles, and a p-value."""

    statistic: float
    p_value: float
    critical_values: Dict[float, float]
    n_boot: int
    block: int
    eta: float
    metadata: Dict[str, object] = field(default_factory=dict)


def test_statistic(prediction: np.ndarray, y: np.ndarray,
                   window: np.ndarray) -> float:
    """Equations 7-8: the squared prediction error on ``window``, over ``sqrt(n)``.

    Non-negative, and zero only where the prediction is exact.
    """
    pred = np.asarray(prediction, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    idx = np.asarray(window)
    resid = pred[idx] - y[idx]
    n = resid.size
    if n == 0:  # pragma: no cover - the pipeline refuses an empty post window
        return float("nan")
    return float(np.sum(resid ** 2) / np.sqrt(n))


def bias_adjusted_att(prediction: np.ndarray, y: np.ndarray,
                      weight: np.ndarray, post: np.ndarray
                      ) -> Tuple[float, float]:
    """Equation 10: the post-window gap, less the gap the fit leaves in sample.

    The adjustment makes the estimate a difference of residual means, so a level
    the ensemble misses by on the weighting window does not read as an effect.

    Returns
    -------
    (att, bias)
    """
    pred = np.asarray(prediction, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    bias = -float(np.mean(pred[np.asarray(weight)] - y[np.asarray(weight)]))
    raw = -float(np.mean(pred[np.asarray(post)] - y[np.asarray(post)]))
    return raw - bias, bias


def _blocks(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    """Fixed-length moving-block resample of ``range(n)``, wrapping at the end.

    ``tsboot(..., sim="fixed", l=block)`` in their code. Wrapping keeps every
    block the same length, so no period is systematically under-sampled.
    """
    block = int(min(max(block, 1), n))
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])
    return idx[:n]


def block_bootstrap_test(predictions: np.ndarray, y: np.ndarray, train: slice,
                         weight: np.ndarray, post: np.ndarray, *, eta: float,
                         n_boot: int, block: int, seed: int = 0,
                         levels: Sequence[float] = LEVELS) -> BootstrapResult:
    """Algorithm 2: the statistic's null distribution by resampling blocks.

    The resampling pool is every period outside the expert-training split, which
    is the weighting window plus the post window -- exchangeable under the null
    of no effect, which is the null being tested. Each replicate resamples that
    pool in blocks, refits the ensemble weights on the first ``len(weight)``
    draws, and evaluates the statistic on the rest.

    Parameters
    ----------
    predictions : np.ndarray
        Expert predictions, shape ``(T, K)``. Held fixed across replicates: the
        bootstrap resamples periods and refits the weights, not the experts.
    y : np.ndarray
        Treated outcome, shape ``(T,)``.
    train : slice
        The expert-training split, excluded from the pool.
    weight, post : np.ndarray
        Period indices for the weighting and evaluation windows.
    eta : float
        The same learning rate the point estimate used.
    n_boot : int
        Replicates.
    block : int
        Block length; clamped to the pool size.
    seed : int
        Seed for the resampling.
    levels : sequence of float
        Levels to report critical values at.

    Returns
    -------
    BootstrapResult
    """
    P = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    weight = np.asarray(weight)
    post = np.asarray(post)

    ssr = ((P[weight] - y[weight, None]) ** 2).sum(axis=0)
    w_obs = exponential_weights(ssr, eta)
    observed = test_statistic(P @ w_obs, y, post)

    pool = np.concatenate([weight, post])
    n_w = weight.size
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = pool[_blocks(rng, pool.size, block)]
        fit_idx, eval_idx = idx[:n_w], idx[n_w:]
        if eval_idx.size == 0:  # pragma: no cover - pool is always longer
            draws[b] = np.nan
            continue
        s = ((P[fit_idx] - y[fit_idx, None]) ** 2).sum(axis=0)
        yhat = P @ exponential_weights(s, eta)
        draws[b] = test_statistic(yhat, y, eval_idx)

    good = draws[np.isfinite(draws)]
    if good.size == 0:  # pragma: no cover - defensive
        crit = {float(a): float("nan") for a in levels}
        return BootstrapResult(observed, float("nan"), crit, int(n_boot),
                               int(block), float(eta))
    # Upper tail, add-one corrected so the p-value is never exactly zero. Their
    # code reports the statistic's rank in the sorted draws, which is the null
    # CDF at the statistic -- one minus this.
    p = float((1 + np.count_nonzero(good >= observed)) / (1 + good.size))
    crit = {float(a): float(np.quantile(good, 1.0 - a)) for a in levels}
    return BootstrapResult(statistic=observed, p_value=p, critical_values=crit,
                           n_boot=int(n_boot), block=int(block), eta=float(eta),
                           metadata={"pool_periods": int(pool.size),
                                     "weight_periods": int(n_w)})
