"""Durable speed record for the active-set simplex QP solver (PR #60 follow-up).

Generates a **Dolan-More performance profile** (Dolan & More 2002) comparing the
pure-NumPy active-set solver against the cvxpy/CLARABEL reference across a grid
of problem shapes, plus a **warm-start sweep** measuring the conformal /
market-selection pattern (a chain of nearby problems). Correctness parity
(objective agreement) is asserted while timing, so a regression in either speed
*or* correctness shows up here.

Run::

    python benchmarks/active_set_profile.py            # writes the profile PNG + table

This is a *profiling artifact*, not a pass/fail benchmark case: wall-clock is
machine-dependent, so the committed PNG is the record and the script re-creates
it on demand.
"""
from __future__ import annotations

import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.bilevel.ridge_augment import _simplex_qp_cvxpy

# Representative SCM shapes: (matching rows m == T0, donors J), few to many.
_SHAPES = [(20, 5), (40, 10), (50, 20), (89, 49), (30, 30), (15, 40), (120, 80)]
_SEEDS = range(8)


def _objective(B, A, w):
    r = B @ np.asarray(w, float).ravel() - A
    return float(r @ r)


def _median_time(fn, B, A, repeats=25):
    fn(B, A)                                    # warm the caches
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(B, A)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def collect():
    """Time both solvers on every (shape, seed); assert objective parity."""
    rows = []
    for (m, J) in _SHAPES:
        for s in _SEEDS:
            rng = np.random.default_rng(1000 * m + 10 * J + s)
            B = rng.normal(size=(m, J))
            A = rng.normal(size=m)
            t_as = _median_time(solve_simplex_qp, B, A)
            t_cx = _median_time(_simplex_qp_cvxpy, B, A)
            # correctness: active-set must be no worse than cvxpy (which, being
            # interior-point, may sit marginally below the true optimum).
            o_as = _objective(B, A, solve_simplex_qp(B, A))
            o_cx = _objective(B, A, _simplex_qp_cvxpy(B, A))
            assert o_as <= o_cx + 1e-6 * (1 + abs(o_cx)), f"parity fail @ {(m, J, s)}"
            rows.append((m, J, t_as, t_cx))
    return np.array([(r[2], r[3]) for r in rows])      # (n_problems, 2): [active-set, cvxpy]


def warm_start_sweep(m=50, J=20, n=40):
    """Conformal/market-selection pattern: a chain of nearby problems."""
    rng = np.random.default_rng(0)
    B = rng.normal(size=(m, J)); A = rng.normal(size=m)
    prev = solve_simplex_qp(B, A)
    cold = warm = 0.0
    for _ in range(n):
        B = B.copy(); B[:, rng.integers(J)] += 0.02 * rng.normal(size=m)
        t = time.perf_counter(); solve_simplex_qp(B, A); cold += time.perf_counter() - t
        t = time.perf_counter(); w = solve_simplex_qp(B, A, warm_start=prev); warm += time.perf_counter() - t
        prev = w
    return cold, warm


def performance_profile(times):
    """Dolan-More profile: rho_s(tau) = fraction of problems within tau x best."""
    best = times.min(axis=1, keepdims=True)
    ratios = times / best                              # >= 1
    taus = np.linspace(1.0, ratios.max() * 1.05, 400)
    rho = np.array([[np.mean(ratios[:, s] <= t) for t in taus]
                    for s in range(times.shape[1])])
    return taus, rho


def main():
    times = collect()
    taus, rho = performance_profile(times)
    speedup = times[:, 1] / times[:, 0]                 # cvxpy / active-set, per problem
    cold, warm = warm_start_sweep()

    print("=== active-set vs cvxpy (Dolan-More) ===")
    print(f"problems: {times.shape[0]}  "
          f"active-set faster on {int(np.mean(speedup > 1) * 100)}% of them")
    print(f"speedup (cvxpy/active-set): median {np.median(speedup):.1f}x  "
          f"min {speedup.min():.1f}x  max {speedup.max():.1f}x")
    print(f"warm-start sweep (40 nearby problems): cold {cold * 1e3:.0f}ms  "
          f"warm {warm * 1e3:.0f}ms  ({cold / warm:.0f}x)")

    fig, ax = plt.subplots(figsize=(7, 5))
    for s, label in enumerate(("active-set", "cvxpy/CLARABEL")):
        ax.step(taus, rho[s], where="post", linewidth=1.8, label=label)
    ax.set_xlabel(r"$\tau$  (within $\tau\times$ the faster solver)")
    ax.set_ylabel(r"fraction of problems  $\rho_s(\tau)$")
    ax.set_title("Active-set vs cvxpy — Dolan–Moré performance profile")
    ax.set_xlim(left=1.0); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    out = "benchmarks/active_set_profile.png"
    fig.savefig(out, bbox_inches="tight", dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
