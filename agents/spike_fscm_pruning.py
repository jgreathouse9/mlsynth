"""Measurement behind agents/agents_fscm_complexity.md.

How much of FSCM's forward scan can be skipped without solving, using a bound
rather than an approximation. Run: python agents/spike_fscm_pruning.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.datautils import dataprep


def load(name):
    if name == "prop99":
        df = pd.read_csv("basedata/P99data.csv")
        u, t, o, tr, yr = "state", "year", "cigsale", "California", 1989
    else:
        df = pd.read_csv("basedata/basque_data.csv")
        df = df[df["regionname"] != "Spain (Espana)"]
        u, t, o = "regionname", "year", "gdpcap"
        tr, yr = "Basque Country (Pais Vasco)", 1975
    df["treat"] = ((df[u] == tr) & (df[t] >= yr)).astype(int)
    p = dataprep(df, u, t, o, "treat")
    T0 = int(p["pre_periods"])
    return (np.asarray(p["donor_matrix"], float)[:T0],
            np.asarray(p["y"], float).ravel()[:T0])


def sweep(X, y, relaxation):
    """One full greedy sweep, counting solves against candidates skipped."""
    J = X.shape[1]
    G, h, yy = X.T @ X, X.T @ y, float(y @ y)

    def lower_bound(idx):
        Gs, hs, k = G[np.ix_(idx, idx)], h[idx], len(idx)
        if relaxation == "unconstrained":
            b = np.linalg.lstsq(Gs, hs, rcond=None)[0]
        else:                                     # affine: keep 1'b = 1
            K = np.zeros((k + 1, k + 1))
            K[:k, :k], K[:k, k], K[k, :k] = 2 * Gs, -1.0, 1.0
            r = np.zeros(k + 1); r[:k], r[k] = 2 * hs, 1.0
            b = np.linalg.lstsq(K, r, rcond=None)[0][:k]
        return max(yy - 2.0 * hs @ b + b @ Gs @ b, 0.0)

    selected, remaining, solves, pruned, saturated_at = [], list(range(J)), 0, 0, None
    for step in range(J):
        best, best_j = np.inf, None
        for lb, j in sorted((lower_bound(selected + [j]), j) for j in remaining):
            if lb > best:                         # cannot beat the incumbent
                pruned += 1
                continue
            idx = selected + [j]
            w = solve_simplex_qp(X[:, idx], y)
            solves += 1
            sse = float(np.sum((y - X[:, idx] @ w) ** 2))
            if sse < best:
                best, best_j = sse, j
        if saturated_at is None and step > 0 and abs(best - prev) <= 1e-12:
            saturated_at = step               # every further donor is a tie
        prev = best
        selected.append(best_j)
        remaining.remove(best_j)
    return solves, pruned, selected, saturated_at


if __name__ == "__main__":
    for name in ("prop99", "basque"):
        X, y = load(name)
        base = None
        for relaxation in ("unconstrained", "affine"):
            solves, pruned, order, sat = sweep(X, y, relaxation)
            total = solves + pruned
            if base is None:
                base = (order, sat)
            # The bound cannot change which donor wins where one strictly wins.
            # Past saturation every remaining candidate scores identically, so
            # the order there is a tie-break and not a selection; comparing it
            # would be comparing evaluation order.
            assert sat == base[1], "saturation moved"
            assert order[:sat] == base[0][:sat], "pruning changed the selection"
            print(f"{name:<7} {relaxation:<14} solved {solves:>4} / {total:<4} "
                  f"pruned {100 * pruned / total:>5.1f}%   "
                  f"order determined through step {sat}")
