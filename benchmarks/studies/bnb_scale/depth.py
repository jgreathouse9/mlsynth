"""How deep a node must be before its lift passes the trust region (#858).

From 60 units, on the #842 panels, the root's relaxation collapses onto the
trust-region bound: every relaxation tested equals it to 2e-9. Fixing units
adds a linear term, which might break that degeneracy before the free count
nears ``T``. This module measures what happens at the nodes the search visits.

The structure. The two-way matrix is ``R = (alpha/lam) I - W`` with ``W``
semidefinite of rank at most ``T + 1``. At a node with free set ``F``, the
quadratic part ``A = R_FF`` keeps the top eigenvalue ``alpha/lam`` on the null
space of ``W_FF`` within the plane ``1'x = s``, and the linear term the fixed
units add, ``b = R_FD sigma_D = -W_FD sigma_D``, lies in the range of ``W``'s
rows. So the linear term cannot reach that eigenspace, whose dimension is the
node's excess ``n_F - T - 1``. :func:`spectrum` measures the top eigenvalue,
its multiplicity and the linear term's share in its eigenspace.

The certificate. Where the linear term misses the top eigenspace ``E`` and the
trust region is in its hard case, its maximiser is ``xw + Q e`` for any ``e``
of the right length, ``Q`` a basis of ``E`` on the plane. Every pseudo-moment
matrix ``G`` of ``(1, e)``, semidefinite with ``G_00 = 1``, then gives a lifted
``Y`` with the trust region's value, the cardinality and its products. If ``G``
also puts ones on ``Y``'s diagonal, and ``Y`` violates no triangle, then ``Y``
is feasible for the lift's relaxation (unit diagonal, ``X r = 0``, the RLT
products, the triangles) at the trust region's value, and no bound from that
relaxation can pass the trust region. Finding ``G`` is fitting an ellipsoid
through the rows ``p_i = (xw_i, q_i)``, the problem of Saunderson,
Chandrasekaran, Parrilo and Willsky (2012). :func:`certificate` tries the
least-norm fit, one linear solve of ``n + 1`` equations, and reports why it
fails when it does. A failed least-norm fit proves nothing, since another
``G`` may exist; the projected fit (``fit="projection"``) searches for one by
alternating projections, at the cost of an eigendecomposition per iteration.
The fit is checked in floating point, so a certificate is a diagnosis for the
study, not a proof the search could prune on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from mlsynth.utils.syndes_helpers import branch

#: Why :func:`certificate` stopped, in the order its checks run.
REASONS = ("small", "simple", "reached", "easy", "not_psd", "fit", "triangle",
           "certified")


def nodes(design: Sequence[int], N: int, fixed: int, *, reps: int = 1,
          seed: int = 0) -> list:
    """``reps`` nodes with ``fixed`` units decided, as ``(I, O, F)``: half of
    them (rounded down) drawn from ``design`` into the treated set, the rest
    from its complement out of it, so each node lies on the path to ``design``.
    The root (``fixed = 0``) is returned once."""
    design = sorted(int(u) for u in design)
    others = [u for u in range(N) if u not in set(design)]
    into, out = fixed // 2, fixed - fixed // 2
    if fixed < 0 or into > len(design) or out > len(others):
        raise ValueError(f"cannot fix {fixed} units on the path to a design of "
                         f"{len(design)} among {N}")
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(1 if fixed == 0 else reps):
        I = tuple(sorted(int(u) for u in rng.permutation(design)[:into]))
        O = tuple(sorted(int(u) for u in rng.permutation(others)[:out]))
        F = tuple(u for u in range(N) if u not in I and u not in O)
        found.append((I, O, F))
    return found


def _plane(c: float, b: np.ndarray, A: np.ndarray, s: int):
    """The node on the plane ``1'x = s``: ``x = x0 + P z`` with ``P`` an
    orthonormal basis of ``1``'s complement, as :func:`branch.trust_region`
    writes it. Returns ``x0``, ``P``, ``B = P'AP``, ``g = P'(b + A x0)``."""
    n = len(b)
    x0 = np.full(n, s / n)
    P = np.linalg.svd(np.ones((1, n)))[2][1:].T
    B = P.T @ A @ P
    return x0, P, 0.5 * (B + B.T), P.T @ (b + A @ x0)


def _top(lam: np.ndarray, rel: float) -> np.ndarray:
    """The eigenvalues within ``rel`` of the largest, relative to the
    spectrum's largest magnitude."""
    scale = max(float(np.abs(lam).max()), 1e-300)
    return lam >= lam[-1] - rel * scale


def spectrum(c: float, b: np.ndarray, A: np.ndarray, s: int, *,
             rel: float = 1e-9) -> dict:
    """The node's top eigenvalue on the plane, its multiplicity, and the share
    ``|E'g| / |g|`` of the linear term in its eigenspace ``E`` (zero when the
    node has no linear term)."""
    if len(b) <= 1:
        return {"top": float("nan"), "multiplicity": 0, "reach": 0.0}
    _, _, B, g = _plane(c, b, A, s)
    lam, V = np.linalg.eigh(B)
    inE = _top(lam, rel)
    norm = float(np.linalg.norm(g))
    reach = float(np.linalg.norm(V[:, inE].T @ g)) / norm if norm > 0 else 0.0
    return {"top": float(lam[-1]), "multiplicity": int(inE.sum()), "reach": reach}


@dataclass(frozen=True)
class Certificate:
    """:func:`certificate`'s outcome: ``reason`` (one of :data:`REASONS`), and
    when it is ``certified``, the lifted ``Y`` and its value ``<Q, Y>``."""

    reason: str
    value: Optional[float] = None
    Y: Optional[np.ndarray] = None


def certificate(c: float, b: np.ndarray, A: np.ndarray, s: int, *,
                rel: float = 1e-9, fit: str = "least_norm",
                iterations: int = 3_000) -> Certificate:
    """A point of the lift's relaxation at the trust region's value, by an
    ellipsoid fit (see the module docstring), or the check it failed:
    ``small`` (fewer than three free units), ``simple`` (the top eigenvalue
    has multiplicity one), ``reached`` (the linear term reaches its
    eigenspace), ``easy`` (the trust region is not in its hard case),
    ``not_psd`` (the fit is not semidefinite), ``fit`` (it misses an
    equation), ``triangle`` (the point violates a triangle).

    ``fit``: ``least_norm``, one linear solve; or ``projection``, Dykstra's
    alternating projections between the fit's equations and the semidefinite
    cone from the least-norm point, at most ``iterations`` of them. The second
    finds a semidefinite fit wherever one exists, given the iterations, at the
    price of an eigendecomposition per iteration."""
    if fit not in ("least_norm", "projection"):
        raise ValueError(f"no fit named {fit!r}")
    n = len(b)
    if n < 3:
        return Certificate("small")
    x0, P, B, g = _plane(c, b, A, s)
    lam, V = np.linalg.eigh(B)
    inE = _top(lam, rel)
    if inE.sum() < 2:
        return Certificate("simple")
    E, Vr, lr = V[:, inE], V[:, ~inE], lam[~inE]
    if np.linalg.norm(E.T @ g) > rel * max(float(np.linalg.norm(g)), 1.0):
        return Certificate("reached")
    w = Vr @ ((Vr.T @ g) / (lam[-1] - lr))
    if w @ w >= n - s * s / n:
        return Certificate("easy")
    xw = x0 + P @ w
    Pp = np.hstack([xw[:, None], P @ E])          # the rows p_i = (xw_i, q_i)
    e0 = np.zeros(Pp.shape[1])
    e0[0] = 1.0
    Pa = np.vstack([Pp, e0])                      # and G_00 = 1
    K = (Pa @ Pa.T) ** 2
    a = np.linalg.lstsq(K, np.ones(n + 1), rcond=None)[0]
    G = Pa.T @ (a[:, None] * Pa)
    if fit == "projection":
        G = _project(G, Pa, np.linalg.pinv(K), iterations)
    if np.linalg.eigvalsh(G)[0] < -1e-10 * max(1.0, float(np.abs(G).max())):
        return Certificate("not_psd")
    x, X = Pp @ G[:, 0], Pp @ G @ Pp.T
    if abs(G[0, 0] - 1.0) > 1e-8 or np.abs(np.diag(X) - 1.0).max() > 1e-8:
        return Certificate("fit")
    Y = np.empty((n + 1, n + 1))
    Y[0, 0], Y[0, 1:], Y[1:, 0], Y[1:, 1:] = 1.0, x, x, X
    if len(branch.separate(Y, gap=1e-9, limit=1)):
        return Certificate("triangle")
    return Certificate("certified", c + 2.0 * float(b @ x) + float(np.sum(A * X)), Y)


def _project(G: np.ndarray, Pa: np.ndarray, Kp: np.ndarray, iterations: int) -> np.ndarray:
    """Dykstra's projections from ``G``, which meets the equations
    ``p_k' G p_k = 1``, onto their intersection with the semidefinite cone,
    checked every twenty iterations and stopped once ``G`` is semidefinite to
    1e-10 of its scale. The last iterate meets the equations."""
    def onto_equations(M):
        r = 1.0 - np.einsum("ki,ij,kj->k", Pa, M, Pa)
        return M + Pa.T @ ((Kp @ r)[:, None] * Pa)

    def onto_cone(M):
        lam, U = np.linalg.eigh(0.5 * (M + M.T))
        return (U * np.maximum(lam, 0.0)) @ U.T

    Q = np.zeros_like(G)
    for k in range(iterations):
        if k % 20 == 0 and np.linalg.eigvalsh(G)[0] >= -1e-10 * max(1.0, float(np.abs(G).max())):
            break
        Z = onto_cone(G + Q)
        Q = G + Q - Z
        G = onto_equations(Z)
    return G


def depth_row(R: np.ndarray, T: int, K: int, I, O, F, *,
              closure: bool = False, projection: bool = False) -> dict:
    """The node's trust-region bound and the lift's, their gap as the lift's
    gain, the lift's evaluations and seconds, :func:`spectrum` and
    :func:`certificate` with its seconds; with ``closure``, the triangle
    closure by interior point at the node (``root_gap.closure``); with
    ``projection``, where the least-norm fit is not semidefinite, the projected
    fit's outcome and seconds."""
    c, b, A = branch.node_parts(R, I, O, F)
    n = len(F)
    s = 2 * (K - len(I)) - n
    U_tr = branch.trust_region(c, b, A, s)[0]
    t0 = time.process_time()
    U, _, evaluations = branch.lift(c, b, A, s)
    lift_seconds = time.process_time() - t0
    t0 = time.process_time()
    cert = certificate(c, b, A, s)
    row = {"fixed": len(I) + len(O), "free": n, "excess": n - T - 1,
           "trust_region": U_tr, "lift": U, "gain": (U_tr - U) / U_tr,
           "evaluations": evaluations, "lift_seconds": lift_seconds,
           "certificate": cert.reason,
           "certificate_seconds": time.process_time() - t0,
           **spectrum(c, b, A, s)}
    if projection and cert.reason == "not_psd":
        t0 = time.process_time()
        row["projection"] = certificate(c, b, A, s, fit="projection").reason
        row["projection_seconds"] = time.process_time() - t0
    if closure:
        from benchmarks.studies.bnb_scale.root_gap import closure as solve
        U_cl = solve(c, b, A, s)["U"]
        row.update(closure=U_cl, closure_gain=(U_tr - U_cl) / U_tr)
    return row


def depth_table(Y: np.ndarray, K: int, design: Sequence[int], *,
                fixed: Sequence[int] = (0, 5, 10, 20, 30, 40, 50, 60, 70),
                reps: int = 3, seed: int = 0,
                closure_at: Sequence[int] = (), projection: bool = False) -> list:
    """:func:`depth_row` at :func:`nodes` of each ``fixed`` count that leaves
    at least four units free, on one BLAS thread; the triangle closure at the
    first node of each count in ``closure_at``; the projected fit where asked."""
    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    T, N = Y.shape
    rows = []
    with threadpool_limits(1):
        R = BoundEngine.from_problem(TwoWayProblem.from_panel(Y)).R
        for d in fixed:
            if N - d < 4:
                continue
            for rep, (I, O, F) in enumerate(nodes(design, N, d, reps=reps, seed=seed)):
                row = depth_row(R, T, K, I, O, F, closure=rep == 0 and d in closure_at,
                                projection=projection)
                rows.append({"N": N, "T": T, "K": K, "rep": rep, **row})
    return rows
