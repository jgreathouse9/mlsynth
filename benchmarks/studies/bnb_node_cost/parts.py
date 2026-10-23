"""Candidates for a cheaper node (#844), each beside the library's step.

- :func:`cut_matrix`: the lift's ``sum_k w_k S_k`` by one ``bincount`` over
  the upper triangle, where the library makes six ``np.add.at`` calls. Each
  cell receives the same terms in the same order, starting from zero, so the
  sums are the library's bit for bit, and the lower triangle is the upper's
  transpose because every term lands in both.
- :func:`to_float`: a dyadic matrix rounded to doubles by integer true
  division, ``x / 2^e``, where the library builds a ``Fraction`` per entry.
  Python rounds an integer quotient correctly, and so does ``Fraction``'s
  conversion, so the doubles are the same; a quotient past the largest double
  raises ``OverflowError`` in both and becomes a signed infinity.
- :func:`rump_lower`: Rump's floating-point test of positive definiteness
  (Rump, "Verification methods", Acta Numerica 2010, and "Mathematically
  rigorous global optimization in floating-point arithmetic", p. 12, Theorem
  4.1, after Demmel): for a symmetric floating-point ``A``, ``B = A - D``
  with ``D`` diagonal, ``D >= alpha I`` and ``alpha >= gamma_{n+1} trace(A)
  > 0``, a floating-point Cholesky factorization of ``B`` that completes
  proves ``A`` positive definite, barring overflow and underflow. In place of
  the library's exact residual ``D - sigma I - L L'``, a cubic count of
  big-integer products, it costs one factorization and a few scalar
  rationals. Its bound differs from the library's at rounding level, and the
  library's exact form was chosen on purpose, so it is measured here and left
  to the author.

:func:`lbfgsb_overhead` measures SciPy's L-BFGS-B per evaluation on an
objective that costs nothing, and :func:`candidates` times each candidate
against the library's step at nodes of a panel.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Optional

import numpy as np

from mlsynth.utils.marex_helpers import branch as marex_branch
from mlsynth.utils.marex_helpers.certify import Dyadic
from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.branch import KAPPA

_U = Fraction(1, 2 ** 53)          # the unit roundoff of a double
_TINY = Fraction(1, 2 ** 1074)     # the smallest subnormal


def _gamma(k: int) -> Fraction:
    return k * _U / (1 - k * _U)


def cut_matrix(size: int, cuts: branch.Cuts, w: np.ndarray) -> np.ndarray:
    """The library's ``_cut_matrix(size, cuts, w)``, bit for bit."""
    if not len(cuts):
        return np.zeros((size, size))
    sg = branch._SIGNS[cuts.t - 1] * (0.5 * w)[:, None]
    cells = np.concatenate([cuts.i * size + cuts.j, cuts.i * size + cuts.k,
                            cuts.j * size + cuts.k])
    upper = np.bincount(cells, weights=np.concatenate([sg[:, 0], sg[:, 1], sg[:, 2]]),
                        minlength=size * size).reshape(size, size)
    return upper + upper.T


def _values(X: np.ndarray, cuts: branch.Cuts) -> np.ndarray:
    """``cuts.values(X)``, bit for bit, by one-dimensional takes."""
    size = X.shape[0]
    flat = X.ravel()
    s = branch._SIGNS[cuts.t - 1]
    return (1.0 + s[:, 0] * flat[cuts.i * size + cuts.j] + s[:, 1] * flat[cuts.i * size + cuts.k]
            + s[:, 2] * flat[cuts.j * size + cuts.k])


def theta(c: float, b: np.ndarray, A: np.ndarray, s: int, y: np.ndarray,
          cuts: branch.Cuts, v: np.ndarray, alpha: float):
    """The library's ``_theta``, bit for bit, with fewer temporaries: the
    diagonal through a strided view, the rows of ``X r = 0`` as one outer
    product added to its transpose (``u_i r_j = r_j u_i`` exactly), the cut
    matrix by :func:`cut_matrix` and the cut values by one-dimensional takes.
    Every entry is the same expression in the same order."""
    n = len(b)
    size = n + 1
    C = np.empty((size, size))
    C[0, 0] = 0.0
    C[0, 1:] = b
    C[1:, 0] = b
    C[1:, 1:] = A
    yd, ys, yr = y[:size], y[size], y[size + 1:]
    C.reshape(-1)[::size + 1] -= yd
    half = 0.5 * ys
    C[0, 1:] -= half
    C[1:, 0] -= half
    u = np.ones(size)
    u[0] = -float(s)
    rows = np.empty(size)
    rows[0] = 0.0
    rows[1:] = yr
    T = np.multiply.outer(rows, u)
    T += T.T
    T *= 0.5
    C -= T
    w = KAPPA * v
    C += cut_matrix(size, cuts, w)
    const = c + yd.sum() + ys * s + w.sum()
    lam, U = np.linalg.eigh(C)
    pos = np.maximum(lam, 0.0)
    X = (U * (pos / alpha)) @ U.T
    exact = const + size * lam[-1]
    smooth = const + float(pos @ pos) / (2.0 * alpha) + 0.5 * alpha * size * size
    gy = np.empty(len(y))
    gy[:size] = 1.0 - np.diag(X)
    gy[size] = s - X[0, 1:].sum()
    gy[size + 1:] = -(X[1:, :] @ u)
    gv = KAPPA * _values(X, cuts) if len(cuts) else np.zeros(0)
    return exact, smooth, gy, gv, X


def to_float(D: Dyadic) -> np.ndarray:
    """The library's ``_float(D)``, bit for bit."""
    scale = 1 << D.e

    def one(x: int) -> float:
        try:
            return x / scale
        except OverflowError:
            return math.inf if x > 0 else -math.inf
    return np.array([one(int(x)) for x in D.N.ravel()], dtype=float).reshape(D.N.shape)


def rump_lower(D: Dyadic, tries: int = 40) -> Optional[Fraction]:
    """A lower bound on the smallest eigenvalue of the exact symmetric ``D``
    by Rump's floating-point test, or ``None`` when no shift in ``tries``
    passes it.

    ``S`` is ``D`` rounded to nearest, so ``|D - S| <= u / (1 - u) |S|``
    entrywise, plus a subnormal each, and ``|D - S|_2`` is at most that over
    the Frobenius norm. ``A`` is ``S - sigma I`` in floating point, off by the
    rounding of its diagonal, ``delta``, computed exactly. If the test proves
    ``A`` positive definite, ``lambda_min(D) >= sigma - delta - |D - S|_2``
    by Weyl's inequality. The shift ``sigma`` starts where the library's
    starts and moves down fourfold on each failure."""
    S = to_float(D)
    n = S.shape[0]
    if not np.all(np.isfinite(S)):
        return None
    lam = float(np.linalg.eigvalsh(S)[0])
    size = float(np.abs(S).sum(axis=1).max()) or 1.0
    step = max(abs(lam) * 1e-12, 64 * n * np.finfo(float).eps * size)
    root = float(np.sqrt((S * S).sum()))
    if not math.isfinite(root):
        return None
    fro = Fraction(root) * (1 + _gamma(n * n + 2))
    spread = _U / (1 - _U) * fro + n * (_TINY + Fraction(1, 2 ** 537))
    d = np.diag(S).copy()
    for _ in range(tries):
        sigma = lam - step
        a = d - sigma
        A = S.copy()
        A[np.diag_indices(n)] = a
        exact_a = [Fraction(float(x)) for x in a]
        delta = max(abs(ea - (Fraction(float(x)) - Fraction(sigma))) for ea, x in zip(exact_a, d))
        trace = sum(exact_a)
        alpha = _gamma(n + 1) * trace
        # B = A - alpha I, its diagonal rounded down until A - B >= alpha exactly
        b = a - float(alpha)
        for i in range(n):
            while exact_a[i] - Fraction(float(b[i])) < alpha:
                b[i] = math.nextafter(b[i], -math.inf)
        B = A.copy()
        B[np.diag_indices(n)] = b
        try:
            if trace <= 0:  # pragma: no cover - sigma under every Rayleigh quotient leaves each a_i positive
                raise np.linalg.LinAlgError("no positive trace")
            np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            step *= 4.0
            continue
        return Fraction(sigma) - delta - spread
    return None


def lbfgsb_overhead(variables: int, iterations: int = 200) -> dict:
    """SciPy's L-BFGS-B seconds per evaluation beyond its objective: the
    lift's options, bounds on half the unknowns, and Rosenbrock's function in
    ``variables`` unknowns, which runs the whole ``iterations``; the
    objective's own seconds are clocked and taken out."""
    from scipy.optimize import minimize, rosen, rosen_der

    spent = [0.0, 0]

    def fun(x):
        t0 = time.perf_counter()
        out = rosen(x), rosen_der(x)
        spent[0] += time.perf_counter() - t0
        spent[1] += 1
        return out

    x0 = np.linspace(-1.2, 1.2, variables)
    bounds = [(None, None)] * (variables - variables // 2) + [(-1.5, None)] * (variables // 2)
    t0 = time.perf_counter()
    minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
             options={"maxcor": branch.MAXCOR, "maxfun": iterations, "maxiter": iterations,
                      "ftol": 0.0, "gtol": 0.0})
    seconds = time.perf_counter() - t0
    return {"variables": variables, "iterations": spent[1],
            "seconds_per_evaluation": (seconds - spent[0]) / spent[1]}


def _clock(fn, repeats: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def candidates(Y: np.ndarray, K: int, *, depths=(0, 2, 4, 6, 8), seed: int = 0,
               repeats: int = 50) -> list:
    """At ``duals``'s nodes of ``Y``: the lift's evaluations; at the state
    three quarters through its run, the cut matrix by the library and by
    :func:`cut_matrix`; at the certificate of its multipliers, the rounding by
    the library and by :func:`to_float`, the library's ``lambda_min_lower``
    and :func:`rump_lower` with their bounds, and the whole certificate. Each
    candidate says whether it reproduced the library's numbers."""
    from threadpoolctl import threadpool_limits

    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    from benchmarks.studies.syndes_global_bnb.duals import _node

    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
    xr = branch.ExactR(eng)
    N = Y.shape[1]
    rng = np.random.default_rng(seed)
    rows = []
    theta, lml = branch._theta, branch.lambda_min_lower
    for depth in depths:
        I, O, F = _node(rng, N, K, depth)
        c, b, A = branch.node_parts(eng.R, I, O, F)
        s = 2 * (K - len(I)) - len(F)
        states, matrices = [], []

        def spy_theta(*args):
            states.append(args)
            return theta(*args)

        def spy_lml(M, *a, **k):
            matrices.append(M)
            return lml(M, *a, **k)
        branch._theta, branch.lambda_min_lower = spy_theta, spy_lml
        try:
            with threadpool_limits(1):
                _, m, evaluations = branch.lift(c, b, A, s)
                xr.certify(I, O, F, K, m)
        finally:
            branch._theta, branch.lambda_min_lower = theta, lml
        cuts, v, size = states[len(states) * 3 // 4][5], states[len(states) * 3 // 4][6], len(F) + 1
        w = branch.KAPPA * v
        M = matrices[0]
        with threadpool_limits(1):
            seconds = {
                "cut_matrix": _clock(lambda: branch._cut_matrix(size, cuts, w), repeats),
                "cut_matrix_bincount": _clock(lambda: cut_matrix(size, cuts, w), repeats),
                "to_float": _clock(lambda: marex_branch._float(M), repeats),
                "to_float_division": _clock(lambda: to_float(M), repeats),
                "lambda_min_lower": _clock(lambda: lml(M), max(repeats // 10, 1)),
                "rump_lower": _clock(lambda: rump_lower(M), max(repeats // 10, 1)),
                "certify": _clock(lambda: xr.certify(I, O, F, K, m), max(repeats // 10, 1)),
            }
        library, rump = lml(M), rump_lower(M)
        rows.append({
            "N": N, "K": K, "fixed": len(I) + len(O), "free": len(F),
            "evaluations": evaluations, "cuts": len(cuts),
            "cut_matrix_identical": bool(np.array_equal(cut_matrix(size, cuts, w),
                                                        branch._cut_matrix(size, cuts, w))),
            "to_float_identical": bool(np.array_equal(to_float(M), marex_branch._float(M))),
            "library": None if library is None else float(library),
            "rump": None if rump is None else float(rump),
            "seconds": seconds})
    return rows


def _same(a, b) -> bool:
    """Two lifts' outputs equal bit for bit: bound, multipliers, evaluations."""
    (Ua, ma, ea), (Ub, mb, eb) = a, b
    return (Ua == Ub and ea == eb and np.array_equal(ma.diag, mb.diag)
            and np.array_equal(ma.rows, mb.rows) and np.array_equal(ma.weights, mb.weights)
            and all(np.array_equal(getattr(ma.cuts, f), getattr(mb.cuts, f)) for f in "ijkt"))


def lean_lift(Y: np.ndarray, K: int, *, depths=(0, 2, 4, 6, 8), seed: int = 0,
              repeats: int = 7) -> list:
    """At ``duals``'s nodes of ``Y``: the library's lift, and the lift with
    :func:`theta` in place of its evaluation; whether the two agree bit for
    bit, and the fastest of ``repeats`` runs of each on one BLAS thread."""
    from threadpoolctl import threadpool_limits

    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    from benchmarks.studies.syndes_global_bnb.duals import _node

    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
    N = Y.shape[1]
    rng = np.random.default_rng(seed)
    shipped = branch._theta
    rows = []
    for depth in depths:
        I, O, F = _node(rng, N, K, depth)
        c, b, A = branch.node_parts(eng.R, I, O, F)
        s = 2 * (K - len(I)) - len(F)
        runs, seconds = {}, {}
        for name, fn in (("library", shipped), ("lean", theta)):
            branch._theta = fn
            try:
                with threadpool_limits(1):
                    best = np.inf
                    for _ in range(repeats):
                        t0 = time.perf_counter()
                        runs[name] = branch.lift(c, b, A, s)
                        best = min(best, time.perf_counter() - t0)
            finally:
                branch._theta = shipped
            seconds[name] = best
        rows.append({"N": N, "K": K, "fixed": len(I) + len(O), "free": len(F),
                     "evaluations": runs["library"][2],
                     "identical": _same(runs["library"], runs["lean"]),
                     "seconds": seconds})
    return rows


#: Where the runner appends its rows.
RESULTS = __import__("pathlib").Path(__file__).resolve().parent / "results"


def main(argv=None) -> int:
    """``candidates PANEL N K``, ``lean PANEL N K`` or ``overhead VARIABLES``:
    JSON lines appended to ``results/parts_<mode>.jsonl``.

        python -m benchmarks.studies.bnb_node_cost.parts candidates factor 40 10
        python -m benchmarks.studies.bnb_node_cost.parts lean geolift 40 10
        python -m benchmarks.studies.bnb_node_cost.parts overhead 380
    """
    import json
    import sys

    from .stages import _panel

    args = list(sys.argv[1:] if argv is None else argv)
    mode = args[0]
    if mode == "candidates":
        rows = [dict(r, panel=args[1]) for r in
                candidates(_panel(args[1], int(args[2])), int(args[3]))]
    elif mode == "lean":
        rows = [dict(r, panel=args[1]) for r in
                lean_lift(_panel(args[1], int(args[2])), int(args[3]))]
    elif mode == "overhead":
        rows = [lbfgsb_overhead(int(args[1]))]
    else:
        raise ValueError(f"no mode named {mode!r}")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"parts_{mode}.jsonl", "a") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=float) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())
