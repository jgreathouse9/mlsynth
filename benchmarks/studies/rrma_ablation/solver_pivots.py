"""Pivot counts of a bounded dual simplex on RRMA's cross-validation windows.

Every RRMA fit is a weighted pairwise LAD. Its dual is the bounded-variable
programme of Pan (2023, Linear Programming Computation, 2nd ed., problem 8.3):
``min -r'd`` subject to ``D'd = 0`` and ``-w <= d <= w``, with ``p`` equality
rows and one boxed column per pair of periods. Three facts from the LP texts
decide how it should be solved:

* any vertex ``beta`` (``p`` pairs with zero residual) is a dual-feasible basis
  once each nonbasic ``d_k`` sits at ``w_k sign(e_k)``: the reduced costs are the
  negated residuals, and with every variable boxed any basis is dual feasible
  once the nonbasic variables sit at the bound their reduced cost picks (Pan
  Section 8.2, condition (8.19)). Optimality is the remaining test,
  ``|d_B| <= w_B``;
* a pivot is the dual simplex with bound flipping (Pan Algorithm 8.4.1). In
  ``beta``-space it releases the basic pair whose multiplier most exceeds its
  bound and crosses residual breakpoints until the slope turns non-negative:
  a weighted median, the Barrodale-Roberts long step;
* the cross-validation windows are nested, and a new period only adds columns.
  Pan Section 7.3.2 and Bazaraa, Jarvis and Sherali (2010, Section 6.7, "Adding
  a New Activity") restart the simplex from the old basis when a variable is
  added; here the added pair columns sit at ``w_k sign(e_k)``, so the previous
  window's optimal basis stays dual feasible by the first point. That is the
  warm start measured here.

A pairwise LAD optimum is degenerate by construction: when ``p + 1`` rows tie,
all their pairs have zero residual. The pair responses are perturbed by a
deterministic ``1e-10``, the perturbation argument of Luenberger and Ye (2021,
Exercises 4.15-4.17) applied to the costs of this programme, and Bland's rule
(Cottle and Thapa 2017, Section 4.2) takes over if a basis repeats. None of
the LP texts read for the study gives an anti-cycling rule for the bounded dual
simplex: Schrijver (1986, Theorem 11.1) proves Bland's smallest-index rule
terminates for the vertex walk on ``Ax <= b``, and Vanderbei (2020, p. 53)
reports that rounding small right-hand sides to zero makes about five in 1,000
random problems cycle, so tolerances classify here and never alter the data.
Every solution is scored against HiGHS on the unperturbed programme.

Run::

    python -m benchmarks.studies.rrma_ablation.solver_pivots --reps 1

Results are written to ``results/solver_pivots.json``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np

from . import oracle as O
from . import simulate as S
from .highs_reference import wlad

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "solver_pivots.json"
EPS = 1e-10
P_BANDS = ((3, 4), (5, 6), (7, 9), (10, 19))


def pair_index(i: int, j: int, n: int) -> int:
    """Position of pair ``(i, j)``, ``i < j``, in ``np.triu_indices(n, 1)``."""
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def problem(y, X, h, n):
    """The pairwise programme over the first ``n`` rows, and its perturbed responses."""
    i, j = O.pairs(n)
    r = y[i] - y[j]
    xi = np.modf(np.sin(i * 12.9898 + j * 78.233) * 43758.5453)[0]    # fixed per pair
    return i, j, r, r + EPS * np.abs(r).max() * xi, X[i] - X[j], h[i] * h[j]


def crash(r, D, w) -> np.ndarray:
    """A starting vertex: weighted least squares, then the p independent pairs nearest zero."""
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(D * sw[:, None], r * sw, rcond=None)[0]
    basis, Q = [], np.zeros((0, D.shape[1]))
    for k in np.argsort(np.abs(r - D @ beta)):
        v = D[k] - Q.T @ (Q @ D[k])
        if np.linalg.norm(v) > 1e-9 * np.linalg.norm(D[k]):
            basis.append(k)
            Q = np.vstack([Q, v / np.linalg.norm(v)])
            if len(basis) == D.shape[1]:
                break
    return np.array(basis)


def solve(r, D, w, basis, max_pivots: int = 500):
    """Bounded dual simplex from a vertex. Returns ``(beta, basis, pivots, bland_pivots)``."""
    p = D.shape[1]
    basis = basis.copy()
    seen, bland, bland_pivots = set(), False, 0
    for piv in range(max_pivots + 1):
        key = tuple(sorted(basis.tolist()))
        bland |= key in seen
        seen.add(key)
        DB = D[basis]
        beta = np.linalg.solve(DB, r[basis])
        e = r - D @ beta
        nb = np.ones(len(r), bool)
        nb[basis] = False
        dB = np.linalg.solve(DB.T, -(D.T @ (w * np.sign(e) * nb)))
        viol = np.abs(dB) - w[basis]
        tol = 1e-9 * np.maximum(1.0, w[basis])
        if np.all(viol <= tol):
            return beta, basis, piv, bland_pivots
        bad = np.flatnonzero(viol > tol)
        k = int(bad[np.argmin(basis[bad])]) if bland else int(np.argmax(viol))
        ek = np.zeros(p)
        ek[k] = -np.sign(dB[k])
        a = D @ np.linalg.solve(DB, ek)
        cand = nb & (np.sign(e) == np.sign(a)) & (a != 0)
        order = np.argsort(e[cand] / a[cand])
        slope = (w[basis][k] - np.abs(dB[k])) + np.cumsum((2 * w[cand] * np.abs(a[cand]))[order])
        basis[k] = np.flatnonzero(cand)[order[np.argmax(slope >= 0)]]
        bland_pivots += bland
    raise RuntimeError("pivot limit")                    # pragma: no cover - never reached here


def windows(case: str, rep: int, T1: int = 60, T2: int = 30, N: int = 25):
    """Each candidate model's training windows, ``T1 // 2`` to ``T1 - 1`` rows, then all ``T1``."""
    rng = np.random.default_rng(S.seed_for(case, T1, N, rep))
    y0, X = S.draw(case, T1, T2, N, rng)
    order, _, _ = O.screen(y0[:T1], X[:T1])
    for groups in O.candidate_sets(order, X.shape[1], S.M_RANGE).values():
        for g in groups:
            h, _ = O.mallows_or_ones(X[:T1][:, g], rng)
            yield [(y0[:n], X[:n][:, g], h[:n]) for n in range(T1 // 2, T1 + 1)]


def one_rep(case: str, rep: int):
    rows, t = [], dict(warm=0.0, cold=0.0, highs=0.0)
    for wins in windows(case, rep):
        prev = None
        for yy, XX, hh in wins:
            n = len(yy)
            i, j, r, rp, D, w = problem(yy, XX, hh, n)
            t0 = time.perf_counter()
            b0 = crash(rp, D, w) if prev is None else np.array([pair_index(a, b, n) for a, b in prev])
            beta, basis, pw, bw = solve(rp, D, w, b0)
            t["warm"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            _, _, pc, bc = solve(rp, D, w, crash(rp, D, w))
            t["cold"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            ref = wlad(r, D, w)
            t["highs"] += time.perf_counter() - t0
            f = lambda b: float(np.sum(w * np.abs(r - D @ b)))
            rows.append((D.shape[1], pw, pc, (f(beta) - f(ref)) / f(ref), bw + bc))
            prev = [(int(i[k]), int(j[k])) for k in basis]
    return np.array(rows), t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    out = dict(python=platform.python_version(), reps=args.reps, cases={})
    allrows = []
    for case in S.CASES:
        rows, times = [], dict(warm=0.0, cold=0.0, highs=0.0)
        for rep in range(args.reps):
            r_, t_ = one_rep(case, rep)
            rows.append(r_)
            for k in times:
                times[k] += t_[k]
        R = np.vstack(rows)
        allrows.append(R)
        W, C, G = R[:, 1], R[:, 2], R[:, 3]
        out["cases"][case] = dict(
            solves=len(R), warm_mean=float(W.mean()), warm_median=float(np.median(W)),
            warm_p95=float(np.quantile(W, .95)), warm_max=int(W.max()),
            cold_mean=float(C.mean()), cold_median=float(np.median(C)), cold_max=int(C.max()),
            warm_zero_pivots=float(np.mean(W == 0)), gap_max=float(G.max()), gap_min=float(G.min()),
            bland_pivots=int(R[:, 4].sum()), seconds=times)
        print(f"case {case}: {len(R)} solves | warm mean {W.mean():.1f} median {np.median(W):.0f} "
              f"p95 {np.quantile(W, .95):.0f} max {W.max():.0f} | cold mean {C.mean():.1f} | "
              f"gap vs HiGHS {G.min():+.1e} .. {G.max():+.1e} | seconds warm {times['warm']:.1f} "
              f"cold {times['cold']:.1f} HiGHS {times['highs']:.1f}")
    R = np.vstack(allrows)
    out["by_p"] = {}
    for lo, hi in P_BANDS:
        m = (R[:, 0] >= lo) & (R[:, 0] <= hi)
        if m.any():
            out["by_p"][f"{lo}-{hi}"] = dict(solves=int(m.sum()), warm_mean=float(R[m, 1].mean()),
                                             cold_mean=float(R[m, 2].mean()),
                                             warm_per_p=float(np.mean(R[m, 1] / R[m, 0])),
                                             cold_per_p=float(np.mean(R[m, 2] / R[m, 0])))
            v = out["by_p"][f"{lo}-{hi}"]
            print(f"  p {lo}-{hi}: {v['solves']} solves  warm {v['warm_mean']:.1f} ({v['warm_per_p']:.2f} p)  "
                  f"cold {v['cold_mean']:.1f} ({v['cold_per_p']:.2f} p)")
    out["warm_mean"], out["cold_mean"] = float(R[:, 1].mean()), float(R[:, 2].mean())
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    main()
