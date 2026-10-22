"""The semi-synthetic study of Section 4.1, Appendix B.2 and Appendix C, and the gate.

Stated by the paper: 200 donors and one treated unit over ``T = 70`` periods,
``T0 = 50``; 10 latent clusters; each cluster's trajectory "a low-frequency
Fourier series (sum of 6 sine and cosine terms with random coefficients)
superimposed on a linear trend"; donors are their cluster's trajectory plus
Gaussian noise with ``sigma = 1``; the treated unit is a convex combination of
donors from one cluster and receives ``tau = 2`` after ``T0``.

Not stated, and fixed here: the frequencies (1, 2 and 3 cycles over the 70
periods, a sine and a cosine each, so 6 terms), the coefficients (standard
normal), the trend (slope standard normal per 70 periods, intercept standard
normal), equal clusters of 20, the combination weights (Dirichlet(1) over all
20 donors of cluster 0) and noise on the treated unit (``sigma = 1``,
Assumption 3.6's ``x1 = X0 w* + eps``).

The gate asks whether the graph's content matters. Beside the paper's arms it
runs Graph-SC on a uniformly random graph with as many edges as the learned
one (Appendix C's 100%-rewired graph) and on no graph at all, with ``L``
replaced by the identity: ridge at the same ``lambda_g``. The normalized
Laplacian's eigenvalues average one, so both are the same amount of shrinkage
with the donor structure removed.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import oracle as G

N_DONORS, K, T, T0, SIGMA, TAU = 200, 10, 70, 50, 1.0, 2.0
LAM1, LAMG, LAMSUM, KNN = 0.05, 10.0, 50.0, 5
SEED = 20261022

#: Every arm, in both readings of eq. (2) where it has one.
ARM_NAMES = ("SC", "Sparse-only", "Graph-only", "Graph-SC", "Graph-SC oracle graph",
             "Graph-SC random graph", "Ridge (same lambda, no graph)")


def draw(rng) -> Dict[str, np.ndarray]:
    """One panel: ``y`` (treated, with the effect), ``y0``, donors ``X`` (T x 200), clusters, ``w_true``."""
    t = np.arange(T) / T
    basis = np.column_stack([f(2 * np.pi * c * t) for c in (1, 2, 3) for f in (np.sin, np.cos)])
    centroids = (basis @ rng.normal(size=(6, K)) + np.outer(t, rng.normal(size=K))
                 + rng.normal(size=K)[None, :])
    cluster = np.repeat(np.arange(K), N_DONORS // K)
    X = centroids[:, cluster] + SIGMA * rng.normal(size=(T, N_DONORS))
    w_true = np.zeros(N_DONORS)
    members = np.flatnonzero(cluster == 0)
    w_true[members] = rng.dirichlet(np.ones(members.size))
    noiseless_cf = X @ w_true
    y0 = noiseless_cf + SIGMA * rng.normal(size=T)
    y = y0 + TAU * (np.arange(T) >= T0)
    return dict(y=y, y0=y0, X=X, cluster=cluster, w_true=w_true, noiseless_cf=noiseless_cf)


def oracle_graph(cluster):
    """Appendix B.2's ``L_true``: an edge between every two donors of one cluster."""
    A = (cluster[:, None] == cluster[None, :]).astype(float)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    inv = 1.0 / np.sqrt(d)
    return np.eye(len(cluster)) - inv[:, None] * A * inv[None, :], A


def one_rep(rep: int, signed: bool = True) -> Dict:
    """Every arm on one panel. Returns ATT estimates and pre/post RMSPE against the noiseless counterfactual."""
    rng = np.random.default_rng([SEED, rep])
    d = draw(rng)
    y, X = d["y"], d["X"]
    x1, X0 = y[:T0], X[:T0]
    L, A = G.knn_laplacian(X0.T, KNN)
    graphs = {"Graph-only": L, "Graph-SC": L, "Graph-SC oracle graph": oracle_graph(d["cluster"])[0],
              "Graph-SC random graph": G.random_graph_laplacian(int(A.sum() // 2), N_DONORS, rng)[0],
              "Ridge (same lambda, no graph)": np.eye(N_DONORS), "Sparse-only": L}
    lam = {"Sparse-only": (LAM1, 0.0, LAMSUM), "Graph-only": (0.0, LAMG, LAMSUM)}
    solve = G.solve_signed if signed else G.solve_exact
    W = {"SC": G.simplex_sc(x1, X0)}
    for arm, Lg in graphs.items():
        W[arm] = solve(x1, X0, Lg, *lam.get(arm, (LAM1, LAMG, LAMSUM)))
    out = dict(rep=rep, att={}, pre_rmspe={}, post_rmspe_true={}, active={})
    for arm, w in W.items():
        cf = X @ w
        out["att"][arm] = float(np.mean(y[T0:] - cf[T0:]))
        out["pre_rmspe"][arm] = float(np.sqrt(np.mean((x1 - cf[:T0]) ** 2)))
        out["post_rmspe_true"][arm] = float(np.sqrt(np.mean((d["noiseless_cf"][T0:] - cf[T0:]) ** 2)))
        out["active"][arm] = G.active_donors(w)
    return out


#: The shrinkage grid of the gate: the paper's lambda_g = 10 and two decades either side.
GATE_LAMG = (0.1, 1.0, 10.0, 100.0, 1000.0)


def gate_sweep(rep: int) -> Dict:
    """ATT of Graph-SC (signed) on the learned, random and identity penalty over ``GATE_LAMG``."""
    rng = np.random.default_rng([SEED, rep])
    d = draw(rng)
    y, X = d["y"], d["X"]
    x1, X0 = y[:T0], X[:T0]
    L, A = G.knn_laplacian(X0.T, KNN)
    pens = {"learned": L, "random": G.random_graph_laplacian(int(A.sum() // 2), N_DONORS, rng)[0],
            "ridge": np.eye(N_DONORS)}
    out = {}
    for name, P in pens.items():
        for lg in GATE_LAMG:
            w = G.solve_signed(x1, X0, P, LAM1, lg, LAMSUM)
            out[f"{name}@{lg:g}"] = float(np.mean(y[T0:] - X[T0:] @ w))
    return out
