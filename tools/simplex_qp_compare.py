"""All four simplex-QP paths on the same problems: do they agree, and which is fastest?

The headline is not the speed ranking. Where the donor pool is larger than the
pre-period the fit is exact and the optimal face is flat, so the minimiser is
not unique and each solver lands on a different point of it. The objectives
agree to 1e-18; the weights differ by up to 1.9e-2. Weights are what a
synthetic control reports, and `docs/choose.rst` sends exactly this regime --
"Donor pool N >~ T0" -- to CLUSTERSC, SparseSC, PDA, RESCM, FSCM and BVSS. So
the choice of solver decides which synthetic control those estimators return.

min_w ||A - B w||^2  s.t.  w >= 0, sum(w) = 1

  active set   bilevel/active_set.py::solve_simplex_qp   exact, warm-startable
  FISTA        bilevel/simplex.py::simplex_lstsq         projected gradient
  cvxpy        cp.Problem(...).solve()                   interior point + canonicalisation
  Clarabel     sparse_sc_helpers/inner.py::solve_w       interior point, direct

All four solve a convex problem to optimality, so a disagreement in weights is
a bug in one of them and is the first thing to look for. Sizes span the donor
pools this library sees, both well-conditioned and rank-deficient (J > T0).
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
import cvxpy as cp

from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.bilevel.simplex import simplex_lstsq
from mlsynth.utils.sparse_sc_helpers.inner import solve_w

def cvxpy_solve(B, a):
    w = cp.Variable(B.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(a - B @ w)), [cp.sum(w) == 1]).solve()
    return np.asarray(w.value).ravel()

def clarabel_solve(B, a):
    # solve_w takes (v, X1, X0) with X0 of shape (P, N); v = 1 recovers
    # the unweighted least-squares objective this benchmark uses.
    return np.asarray(solve_w(np.ones(B.shape[0]), a, B)).ravel()

def obj(B, a, w):
    return float(np.sum((a - B @ w) ** 2))

CASES = [(20, 40, "J=20  T0=40  well-conditioned"),
         (40, 40, "J=40  T0=40  square"),
         (60, 30, "J=60  T0=30  rank-deficient"),
         (120, 40, "J=120 T0=40  rank-deficient, large pool")]
REPS = 12

print(f"{'case':<34} {'solver':<11} {'ms/solve':>9} {'objective':>13} {'max|dw| vs AS':>14}")
for J, T0, label in CASES:
    rng = np.random.default_rng(J)
    B = rng.standard_normal((T0, J))
    a = B @ rng.dirichlet(np.ones(J)) + 0.01 * rng.standard_normal(T0)
    ref = None
    for name, fn in (("active-set", lambda: solve_simplex_qp(B, a)),
                     ("FISTA", lambda: simplex_lstsq(B, a)),
                     ("cvxpy", lambda: cvxpy_solve(B, a)),
                     ("Clarabel", lambda: clarabel_solve(B, a))):
        try:
            w = np.asarray(fn()).ravel()
            t0 = time.perf_counter()
            for _ in range(REPS):
                w = np.asarray(fn()).ravel()
            ms = 1e3 * (time.perf_counter() - t0) / REPS
        except Exception as e:
            print(f"{label:<34} {name:<11}   FAILED {type(e).__name__}: {str(e)[:40]}")
            continue
        if ref is None:
            ref = w; d = 0.0
        else:
            d = float(np.max(np.abs(w - ref)))
        print(f"{label:<34} {name:<11} {ms:>9.3f} {obj(B,a,w):>13.6g} {d:>14.2e}")
    print()
