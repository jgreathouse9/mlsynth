"""SL end to end: Algorithm 1's split, Equations 11-12, Equation 10, Algorithm 2.

Four steps, in the paper's order:

1. split the pre-period into an expert-training window and a weighting window;
2. fit every expert on the first, and predict over all periods;
3. weight them by exponential weights on the second (out of sample for them);
4. difference against the observed path, bias-adjust, and test.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ...exceptions import MlsynthConfigError
from .diagnostics import (
    error_correlation,
    error_matrix,
    flag_degenerate,
    participation_ratio,
)
from .experts import EXPERTS, build_experts
from .inference import bias_adjusted_att, block_bootstrap_test, test_statistic
from .structures import SLFit, SLInputs
from .weights import effective_k, exponential_weights, paper_eta


def resolve_split(T0: int, train_periods: Optional[int]) -> int:
    """Algorithm 1's split point, in periods.

    ``None`` takes 60 percent of the pre-window, which is the 30-of-50 the
    paper's own scripts use. Both sides must be non-empty: with no weighting
    window there is nothing to weight on that the experts have not already seen.
    """
    if train_periods is None:
        train = int(max(1, min(T0 - 1, round(0.6 * T0))))
    else:
        train = int(train_periods)
    if train < 1 or train >= T0:
        raise MlsynthConfigError(
            f"train_periods={train} leaves no weighting window: the panel has "
            f"{T0} pre-treatment periods and Algorithm 1 needs at least one on "
            f"each side of the split. Pass 1 <= train_periods <= {T0 - 1}, or "
            f"None for the paper's 60 percent.")
    return train


def run_sl(inputs: SLInputs, *, experts: Sequence[str] = EXPERTS,
           train_periods: Optional[int] = None, eta: Optional[float] = None,
           post_skip: int = 0, n_boot: int = 10000, block: int = 3,
           alpha: float = 0.05, seed: int = 0,
           lasso_folds: int = 5, factor_rank: int = 1, forest_trees: int = 500,
           forest_max_leaf_nodes: int = 20) -> SLFit:
    """Run the construction and package the fit.

    Parameters
    ----------
    inputs : SLInputs
        Preprocessed panel.
    experts : sequence of str
        The library, in column order.
    train_periods : int, optional
        Algorithm 1's split; ``None`` takes 60 percent of the pre-window.
    eta : float, optional
        Learning rate; ``None`` takes the paper's ``1/(sqrt(T) var(y))``.
    post_skip : int
        Periods to drop from the start of the post window before measuring.
    n_boot, block, seed
        Algorithm 2's replicates, block length and seed.
    alpha : float
        Recorded on the fit; the test reports critical values at fixed levels.
    lasso_folds, factor_rank, forest_trees, forest_max_leaf_nodes
        Pinned expert hyperparameters.

    Returns
    -------
    SLFit
    """
    T, T0 = inputs.T, inputs.T0
    train_n = resolve_split(T0, train_periods)
    train = slice(0, train_n)
    weight = np.arange(train_n, T0)
    post = np.arange(T0 + int(post_skip), T)
    if post.size == 0:
        raise MlsynthConfigError(
            f"post_skip={post_skip} consumes the whole post-treatment window: "
            f"the panel has {T - T0} post periods, so there is nothing left to "
            f"measure. Pass post_skip < {T - T0}.")

    lib = build_experts(
        inputs.Yco, inputs.y, train, experts, covariates=inputs.covariates,
        seed=seed, lasso_folds=lasso_folds, factor_rank=factor_rank,
        forest_trees=forest_trees, forest_max_leaf_nodes=forest_max_leaf_nodes)
    P, names = lib.predictions, lib.names

    resolved_eta = (paper_eta(inputs.y, T) if eta is None else float(eta))

    R = error_matrix(P, inputs.y, weight)
    ssr = (R ** 2).sum(axis=0)
    w = exponential_weights(ssr, resolved_eta)
    counterfactual = P @ w
    gap = inputs.y - counterfactual
    att, bias = bias_adjusted_att(counterfactual, inputs.y, weight, post)

    boot = block_bootstrap_test(P, inputs.y, train, weight, post,
                                eta=resolved_eta, n_boot=n_boot, block=block,
                                seed=seed)

    return SLFit(
        experts=names,
        predictions=P,
        weights={n: float(v) for n, v in zip(names, w)},
        eta=resolved_eta,
        effective_k=effective_k(w),
        counterfactual=counterfactual,
        gap=gap,
        att=att,
        bias=bias,
        test_statistic=test_statistic(counterfactual, inputs.y, post),
        critical_values=boot.critical_values,
        p_value=boot.p_value,
        expert_ssr={n: float(v) for n, v in zip(names, ssr)},
        error_correlation=error_correlation(R),
        error_participation_ratio=participation_ratio(R),
        degenerate_experts=flag_degenerate(names, R, ssr=ssr, weights=w),
        dropped_experts=lib.dropped,
        train_periods=train_n,
        weight_periods=int(weight.size),
        post_periods=int(post.size),
        details=lib.details,
        metadata={"n_boot": int(n_boot), "block": int(block),
                  "alpha": float(alpha), "seed": int(seed),
                  "post_skip": int(post_skip)},
    )
