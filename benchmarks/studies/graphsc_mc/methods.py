"""The arms: classic SC and the Graph-SC variants, as weight functions of the rows they are given.

Each arm is a closure ``fn(idx) -> w`` over one panel, the shape
``mlsynth.utils.inferutils.debiased_sc_ttest`` takes as ``weight_fn``, so the
same function gives the plain estimate (``idx`` = every pre-period) and the
CWZ cross-fitted folds (``idx`` = the pre-period without one block). The donor
graph is learned from the rows in ``idx`` and nothing else.

Graph-SC's lambdas are the paper's fixed values. They are not scale-free: the
fit term grows with the square of the data's units and the penalties do not.
The scaled arms divide the rows by their root-mean-square donor level before
solving, which puts every panel on the order-one scale of the paper's own
simulation; the raw arm solves on the data as given, as the paper does on
Prop 99.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..graphsc_replicate import oracle as G
from . import solver as S

#: What the replication oracle says when its own certificate fails.
_ORACLE_FAILURES = ("failed its KKT test", "did not converge", "CLARABEL returned")

LAM = (0.05, 10.0, 50.0)
KNN = 5

#: Each reading of eq. (2) with its two controls: the same penalty strength on a
#: random graph with as many edges, and on no graph (``L`` replaced by ``I``).
METHODS = ("SC", "Graph-SC signed", "Random graph signed", "Ridge signed",
           "Graph-SC w>=0", "Random graph w>=0", "Ridge w>=0", "Graph-SC signed, raw units")
SCALED = METHODS[1:-1]


def unit_scale(X) -> float:
    """Root-mean-square level of the donor rows; refuses a panel with none."""
    s = float(np.sqrt(np.mean(np.square(X))))
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"cannot scale the donor rows: their root-mean-square level is {s!r}")
    return s


def reported(solve, *args):
    """``solve(*args)``, with the oracle's certificate failures re-issued as :class:`solver.CertificateWarning`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = solve(*args)
    for c in caught:
        if any(s in str(c.message) for s in _ORACLE_FAILURES):
            warnings.warn(S.CertificateWarning(f"{solve.__name__}: {c.message}"), stacklevel=2)
        else:
            warnings.warn_explicit(c.message, c.category, c.filename, c.lineno)
    return w


def penalty(method: str, X, seed: int):
    """``(P, n_edges)``: the arm's quadratic penalty on rows ``X`` and the learned graph's edge count."""
    L, A = G.knn_laplacian(X.T, KNN)
    m = int(A.sum() // 2)
    if method.startswith("Random graph"):
        return G.random_graph_laplacian(m, X.shape[1], np.random.default_rng(seed))[0], m
    if method.startswith("Ridge"):
        return np.eye(X.shape[1]), m
    return L, m


def weight_fn(method: str, y, Y0, T0: int, seed: int):
    """The arm's weights as a function of the pre-period rows they are fit on.

    The signed arms warm-start each solve from the previous one; the minimiser
    is unique when the penalized Gram matrix is positive definite, so the order
    of calls moves only the work. ``fn.stats`` counts the search's fallbacks to
    the interior-point solve.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    y, Y0 = np.asarray(y, dtype=float), np.asarray(Y0, dtype=float)
    state = {"w": None}

    def fn(idx):
        idx = np.asarray(idx)
        if idx.size and idx.max() >= T0:
            raise ValueError("weights are fit on pre-period rows only")
        x1, X0 = y[idx], Y0[idx]
        if method == "SC":
            return reported(G.simplex_sc, x1, X0)
        if method != "Graph-SC signed, raw units":
            s = unit_scale(X0)
            x1, X0 = x1 / s, X0 / s
        P, _ = penalty(method, X0, seed)
        if method.endswith("w>=0"):
            return reported(G.solve_exact, x1, X0, P, *LAM)
        w, info = S.solve_signed(x1, X0, P, *LAM, w0=state["w"])
        state["w"] = w
        fn.stats["fallback"] += int(info["fallback"])
        return w

    fn.stats = {"fallback": 0}
    return fn
