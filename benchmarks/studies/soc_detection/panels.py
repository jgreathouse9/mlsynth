"""The panel and the MAREX program the detection study is run against.

One cluster, the standard design, no covariates or restrictions: the smallest
configuration that still produces the two second-order cones cvxpy emits for a
MAREX objective. ``program`` returns the cvxpy problem built from the library's
own formulation helpers, so what the study measures is the program MAREX
actually solves and not a restatement of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp

from mlsynth.utils.marex_helpers.formulation import (
    build_constraints,
    build_objective,
    build_membership_mask,
    compute_cluster_means_members,
    init_cvxpy_variables,
    precompute_distances,
    prepare_clusters,
    prepare_fit_slices,
)


def panel(J: int = 12, T: int = 24, T0: int = 18, seed: int = 3,
          n_good: int = 5, hi: float = 4.0):
    """A factor panel on which the design concentrates.

    ``n_good`` markets track the cluster's mean loading with almost no
    idiosyncratic noise and the rest load differently and are noisy, so the
    optimizer has a reason to prefer a few markets. The detection question does
    not depend on that, but keeping the panel identical to the one in
    ``mlsynth/tests/test_marex_weight_cap.py`` means a result here can be
    compared against the cap tests without re-deriving the fixture.
    """
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    load = rng.uniform(0.4, 1.6, size=(2, J))
    load[:, :n_good] = load.mean(axis=1, keepdims=True)
    scale = np.full(J, hi)
    scale[:n_good] = 0.05
    Y = f @ load + rng.normal(size=(T, J)) * scale + 20.0
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), T),
                       "time": np.tile(np.arange(T), J),
                       "y": Y.T.reshape(-1)})
    return df, T0


def program(J: int = 12, m_eq: int = 3, cap: float | None = None,
            boolean: bool = True, seed: int = 3) -> cp.Problem:
    """MAREX's design program, built through the library's own helpers."""
    df, T0 = panel(J=J, seed=seed)
    Y = df.pivot(index="unit", columns="time", values="y").to_numpy()
    clusters = np.zeros(J, dtype=int)
    Yn, clusters, N, labels, K, l2k = prepare_clusters(Y, clusters)
    Y_fit, _, _ = prepare_fit_slices(Yn, T0, 0)
    M = build_membership_mask(clusters, l2k, N, K)
    Xbar, members = compute_cluster_means_members(Y_fit, M, labels)
    D1, D2 = precompute_distances(Y_fit, Xbar, members)
    w, v, z = init_cvxpy_variables(N, K, boolean=boolean)
    cons = build_constraints(w, v, z, M, members, labels, m_eq, None, None,
                             None, None, True, max_control_weight=cap) \
        if _accepts_cap() else \
        build_constraints(w, v, z, M, members, labels, m_eq, None, None,
                          None, None, True)
    if not boolean:
        cons += [z <= 1]
    obj = build_objective(Y_fit, Xbar, members, w, v, z, "standard",
                          1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, D1, D2)
    return cp.Problem(obj, cons)


def _accepts_cap() -> bool:
    """``max_control_weight`` lands on a separate branch; stay usable without it."""
    import inspect
    return "max_control_weight" in inspect.signature(build_constraints).parameters
