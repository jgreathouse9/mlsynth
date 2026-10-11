"""The panels the search is measured on, and MAREX's program for SCIP.

``panel`` is the fixture of ``mlsynth/tests/test_marex_weight_cap.py``, copied
so the study does not import from a test file: five markets track the
cluster's mean loading with almost no idiosyncratic noise and the rest load
differently and are noisy, so the design has a reason to prefer a few markets.

``fit_matrices`` returns what the search needs, ``B`` (fit periods by markets)
and the target ``A`` (the cluster mean over the fit window), computed through
the library's own MAREX helpers so the search and SCIP see the same numbers.
``program`` builds the cvxpy MIQP MAREX hands to SCIP, from the same helpers.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from mlsynth.utils.marex_helpers.formulation import (
    build_constraints,
    build_membership_mask,
    build_objective,
    compute_cluster_means_members,
    init_cvxpy_variables,
    precompute_distances,
    prepare_clusters,
    prepare_fit_slices,
)


def panel(J: int = 12, T: int = 24, T0: int = 18, seed: int = 3,
          n_good: int = 5, hi: float = 4.0):
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


def _prepared(J: int, seed: int):
    df, T0 = panel(J=J, seed=seed)
    Y = df.pivot(index="unit", columns="time", values="y").to_numpy()
    clusters = np.zeros(J, dtype=int)
    Yn, clusters, N, labels, K, l2k = prepare_clusters(Y, clusters)
    Y_fit, _, _ = prepare_fit_slices(Yn, T0, 0)
    M = build_membership_mask(clusters, l2k, N, K)
    Xbar, members = compute_cluster_means_members(Y_fit, M, labels)
    return Y_fit, Xbar, members, labels, M, N, K


def fit_matrices(J: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """``B`` of shape (T_fit, J) and the target ``A`` of shape (T_fit,)."""
    Y_fit, Xbar, *_ = _prepared(J, seed)
    return np.ascontiguousarray(Y_fit.T), np.asarray(Xbar[0], dtype=float)


def program(J: int, m_eq: int, seed: int = 3) -> cp.Problem:
    """MAREX's standard-design MIQP, as ``solve_design`` builds it."""
    Y_fit, Xbar, members, labels, M, N, K = _prepared(J, seed)
    D1, D2 = precompute_distances(Y_fit, Xbar, members)
    w, v, z = init_cvxpy_variables(N, K, boolean=True)
    cons = build_constraints(w, v, z, M, members, labels, m_eq, None, None,
                             None, None, True)
    obj = build_objective(Y_fit, Xbar, members, w, v, z, "standard",
                          1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, D1, D2)
    return cp.Problem(obj, cons)
