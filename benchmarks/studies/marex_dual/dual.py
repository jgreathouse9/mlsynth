"""MAREX's persp+N node bound by a regularised dual (#833).

The library solves persp+N with Clarabel and certifies the multipliers it
returns (``fullform.certificate``: the full-form certificate the library had
when this study ran, kept beside it since #900). The certificate is a function
of the multipliers alone: per side, ``y0`` on ``M_00 = 1``, ``alpha`` on
``1'x = 1``, ``beta`` on ``X 1 = x``, ``H >= 0`` on ``X >= 0``, ``mu >= 0``
on ``ind_j - x_j >= 0`` and a semidefinite 2x2 ``Lam_j`` on each perspective
block. For any such values the Lagrangian is ``const + <S, M>``
per side plus ``c'z``, and since a design has ``tr M <= 2`` per side and ``z``
in the capped simplex ``{0 <= z <= 1, 1'z = r}``,

    bound(theta) = const + sum_sides 2 min(0, lambda_min(S)) + (the r smallest c_j).

This module maximises that bound over the multipliers with L-BFGS-B, as
BiqCrunch maximises its bound (Krislock, Malick and Roupin 2014), on a smooth
lower bound of it: per side ``<S, M> >= -2 |S_-|_F >= -|S_-|_F^2 / (2 a) - 2 a``,
and for the indicators ``min c'z >= min (c'z + a |z|^2 / 2) - a r / 2``, since
``|z|^2 <= r`` on the capped simplex. Both hold for every ``a > 0``, so every
iterate is a valid bound; the smoothing ``a`` is halved round by round. The
multipliers with the best exact bound go to the library's certificate, so the
bound reported is certified in exact arithmetic exactly as Clarabel's is.

Parametrisation: ``H`` by its upper triangle (diagonal included) with ``H >=
0``, ``mu >= 0`` as bounds, and each ``Lam_j`` by a Cholesky factor
``[[p, 0], [q, s]]``, so it is semidefinite for every value. Units are the
library's: the Gram matrix divided by ``scale``.

From the repository root::

    python -m benchmarks.studies.marex_dual.dual geolift
    python -m benchmarks.studies.marex_dual.dual factor12
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mlsynth.utils.marex_helpers import branch
from mlsynth.utils.marex_helpers.certify import ExactData

from . import fullform

#: Where the runner appends its rows.
RESULTS = Path(__file__).resolve().parent / "results"


@dataclass
class SidePart:
    """One side of a node: the markets ``U`` it can use, its Gram block and
    penalty in the library's units, the free pairs ``(i, f)`` (position in
    ``U``, position among the free markets) and whether its indicator is
    ``1 - z``."""

    U: List[int]
    G: np.ndarray
    lin: Optional[np.ndarray]
    free: List[Tuple[int, int]]
    control: bool

    @property
    def k(self) -> int:
        return len(self.U)


@dataclass
class Node:
    data: ExactData
    sides: List[SidePart]
    r: int
    n_free: int
    scale: Fraction


def node_parts(data: ExactData, m: int, I: Sequence[int], O: Sequence[int],
               F: Sequence[int], linear_treated: Optional[np.ndarray] = None,
               linear_control: Optional[np.ndarray] = None) -> Node:
    """The node ``(I, O, F)`` as ``branch.node_bound`` sets it up: the same
    scale, the same sides and the same free positions."""
    I, O, F = list(I), list(O), list(F)
    G = data.gram
    gmax = max(G.frac(i, i) for i in range(data.n))
    scale = Fraction(2) ** round(math.log2(gmax))
    Gs = data.Gf / float(scale)
    pos = {j: i for i, j in enumerate(F)}
    sides = []
    for U, lin, control in ((sorted(I + F), linear_treated, False),
                            (sorted(O + F), linear_control, True)):
        free = [(i, pos[j]) for i, j in enumerate(U) if j in pos]
        L = None if lin is None else np.asarray(lin, float)[U] / float(scale)
        sides.append(SidePart(U=U, G=Gs[np.ix_(U, U)], lin=L, free=free, control=control))
    return Node(data=data, sides=sides, r=m - len(I), n_free=len(F), scale=scale)


class Layout:
    """Where each side's multipliers sit in the vector ``theta``."""

    def __init__(self, node: Node):
        self.blocks = []
        at = 0
        for s in node.sides:
            k, nf = s.k, len(s.free)
            b = {"y0": at, "alpha": at + 1, "beta": (at + 2, at + 2 + k)}
            at += 2 + k
            b["H"] = (at, at + k * (k + 1) // 2)
            at = b["H"][1]
            b["mu"] = (at, at + nf)
            at += nf
            b["lam"] = (at, at + 3 * nf)
            at += 3 * nf
            b["iu"] = np.triu_indices(k)
            self.blocks.append(b)
        self.size = at

    def start(self, lam0: float) -> np.ndarray:
        """Zero multipliers, except each perspective factor at ``p = s =
        lam0``: at ``p = q = s = 0`` the factor's gradient is zero, so a start
        there never moves the perspective multipliers."""
        theta = np.zeros(self.size)
        for b in self.blocks:
            lam = theta[b["lam"][0]:b["lam"][1]].reshape(-1, 3)
            lam[:, 0] = lam[:, 2] = lam0
            theta[b["lam"][0]:b["lam"][1]] = lam.ravel()
        return theta

    def bounds(self) -> list:
        out = [(None, None)] * self.size
        for b in self.blocks:
            for key in ("H", "mu"):
                lo, hi = b[key]
                out[lo:hi] = [(0.0, None)] * (hi - lo)
        return out


def _side_matrix(theta, s: SidePart, b) -> tuple:
    """``S``, the side's constant and its indicators' coefficients, with the
    pieces the gradient needs."""
    k = s.k
    y0, alpha = theta[b["y0"]], theta[b["alpha"]]
    beta = theta[b["beta"][0]:b["beta"][1]]
    h = theta[b["H"][0]:b["H"][1]]
    mu = theta[b["mu"][0]:b["mu"][1]]
    lam = theta[b["lam"][0]:b["lam"][1]].reshape(-1, 3)
    H = np.zeros((k, k))
    H[b["iu"]] = h
    H = H + np.triu(H, 1).T
    S = np.empty((k + 1, k + 1))
    S[0, 0] = -y0
    S[1:, 1:] = s.G - H - 0.5 * (beta[:, None] + beta[None, :])
    xc = -alpha + beta
    if s.lin is not None:
        xc = xc + s.lin
    const = y0 + alpha
    cz = {}
    p, q, sv = lam[:, 0], lam[:, 1], lam[:, 2]
    a_, b_, c_ = p * p, p * q, q * q + sv * sv
    for t, (i, f) in enumerate(s.free):
        xc[i] += mu[t] - 2.0 * b_[t]
        S[i + 1, i + 1] -= c_[t]
        if s.control:
            const -= mu[t] + a_[t]
            cz[f] = cz.get(f, 0.0) + mu[t] + a_[t]
        else:
            cz[f] = cz.get(f, 0.0) - (mu[t] + a_[t])
    S[0, 1:] = S[1:, 0] = 0.5 * xc
    return S, const, cz, (p, q, sv)


def capped_simplex(c: np.ndarray, r: int, a: float) -> Tuple[float, np.ndarray]:
    """``min c'z + a |z|^2 / 2 - a r / 2`` over ``{0 <= z <= 1, 1'z = r}`` and
    its minimiser, which is the smooth part's gradient in ``c``."""
    n = len(c)
    if r <= 0 or r >= n:
        z = np.full(n, 1.0 if r >= n else 0.0)
        return float(c @ z + 0.5 * a * (z @ z) - 0.5 * a * r), z
    lo, hi = -c.max() - a, -c.min() + a
    for _ in range(200):
        nu = 0.5 * (lo + hi)
        if np.clip(-(c + nu) / a, 0.0, 1.0).sum() > r:     # the sum falls as nu rises
            lo = nu
        else:
            hi = nu
    z = np.clip(-(c + 0.5 * (lo + hi)) / a, 0.0, 1.0)
    return float(c @ z + 0.5 * a * (z @ z) - 0.5 * a * r), z


def evaluate(theta: np.ndarray, node: Node, layout: Layout, a: float):
    """The exact bound, the smooth bound and the smooth bound's gradient at
    ``theta``, in the library's units."""
    grad = np.zeros(layout.size)
    exact = smooth = 0.0
    cz = np.zeros(node.n_free)
    keep = []
    for s, b in zip(node.sides, layout.blocks):
        S, const, czs, pqs = _side_matrix(theta, s, b)
        for f, v in czs.items():
            cz[f] += v
        lam, V = np.linalg.eigh(S)
        neg = np.minimum(lam, 0.0)
        exact += const + 2.0 * neg[0]
        smooth += const - (neg @ neg) / (2.0 * a) - 2.0 * a
        Mh = (V * (-neg / a)) @ V.T              # the smooth part's gradient in S
        keep.append((s, b, Mh, pqs))
    if node.n_free:
        exact += float(np.sort(cz)[:node.r].sum())
        val, z = capped_simplex(cz, node.r, a)
        smooth += val
    else:
        z = np.zeros(0)
    for s, b, Mh, (p, q, sv) in keep:
        x, X = Mh[0, 1:], Mh[1:, 1:]
        grad[b["y0"]] = 1.0 - Mh[0, 0]
        grad[b["alpha"]] = 1.0 - x.sum()
        grad[b["beta"][0]:b["beta"][1]] = x - X.sum(axis=1)
        gH = -2.0 * X
        gH[np.diag_indices(s.k)] = -np.diag(X)
        grad[b["H"][0]:b["H"][1]] = gH[b["iu"]]
        gmu = np.empty(len(s.free))
        glam = np.empty((len(s.free), 3))
        for t, (i, f) in enumerate(s.free):
            ind = 1.0 - z[f] if s.control else z[f]
            gmu[t] = x[i] - ind
            da, db, dc = -ind, -2.0 * x[i], -X[i, i]
            glam[t] = (2 * p[t] * da + q[t] * db, p[t] * db + 2 * q[t] * dc, 2 * sv[t] * dc)
        grad[b["mu"][0]:b["mu"][1]] = gmu
        grad[b["lam"][0]:b["lam"][1]] = glam.ravel()
    return exact, smooth, grad


def to_duals(theta: np.ndarray, node: Node, layout: Layout) -> list:
    """The multipliers as ``fullform.certificate`` reads them."""
    out = []
    for s, b in zip(node.sides, layout.blocks):
        k = s.k
        H = np.zeros((k, k))
        H[b["iu"]] = theta[b["H"][0]:b["H"][1]]
        H = H + np.triu(H, 1).T
        lam = theta[b["lam"][0]:b["lam"][1]].reshape(-1, 3)
        Lam = [np.array([[p * p, p * q], [p * q, q * q + sv * sv]]) for p, q, sv in lam]
        lin = None if s.lin is None else np.zeros(node.data.n)
        if lin is not None:
            lin[s.U] = s.lin * float(node.scale)
        out.append(fullform.SideDuals(
            U=list(s.U), free=list(s.free), control=s.control, lin=lin,
            y0=float(theta[b["y0"]]), alpha=float(theta[b["alpha"]]),
            beta=np.array(theta[b["beta"][0]:b["beta"][1]]), H=H,
            mu=list(theta[b["mu"][0]:b["mu"][1]]), Lam=Lam))
    return out


def dual_bound(data: ExactData, m: int, I, O, F, linear_treated=None, linear_control=None,
               *, a0: float = 0.1, a_min: float = 1e-10, shrink: float = 0.5,
               max_rounds: int = 60, per_round: int = 1500, budget: int = 40_000,
               lam0: float = 0.1, target: Optional[float] = None) -> dict:
    """The node's bound by the regularised dual: its certified value, the best
    exact value the iterates reached, evaluations, seconds and whether
    ``target`` (in the objective's units) was reached. The run stops at
    ``budget`` evaluations. The defaults are BiqCrunch's schedule (``a`` from
    0.1, halved), with rounds long enough for GeoLift's 40 markets."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    node = node_parts(data, m, I, O, F, linear_treated, linear_control)
    layout = Layout(node)
    scale = float(node.scale)
    goal = None if target is None else target / scale
    theta = layout.start(lam0)
    best = {"exact": -np.inf, "theta": theta.copy()}
    count = [0]

    class _Reached(Exception):
        pass

    class _Spent(Exception):
        pass

    a = a0
    reached = False
    for _ in range(max_rounds):
        def fun(th, a=a):
            ex, sm, g = evaluate(th, node, layout, a)
            count[0] += 1
            if ex > best["exact"]:
                best.update(exact=ex, theta=th.copy())
            if goal is not None and best["exact"] >= goal:
                raise _Reached
            if count[0] >= budget:
                raise _Spent
            return -sm, -g
        try:
            res = minimize(fun, theta, jac=True, method="L-BFGS-B", bounds=layout.bounds(),
                           options={"maxiter": per_round, "maxfun": per_round,
                                    "maxcor": 20, "ftol": 1e-15, "gtol": 1e-12})
            theta = res.x
        except _Reached:
            reached = True
            break
        except _Spent:
            break
        if a <= a_min:
            break
        a = max(a * shrink, a_min)
    lower = fullform.certificate(data.gram, node.scale, node.r, node.n_free,
                                to_duals(best["theta"], node, layout))
    return {"lower": lower, "exact": best["exact"] * scale, "evaluations": count[0],
            "seconds": time.perf_counter() - t0, "reached": reached}


def compare(name: str) -> list:  # pragma: no cover - the long runs; their parts are tested
    """The dual against the library's interior-point bound at each node of
    the panel ``name``: certified bounds over the subtree minimum, and seconds."""
    from benchmarks.studies.marex_bnb.panels import factor, geolift, leaves
    from benchmarks.studies.marex_bnb.persp_n_geolift import nodes

    if name == "geolift":
        B, A = geolift()
        m = 4
        sets, g, h = leaves("geolift_m4")
        f = g + h
        todo = nodes(B, A, sets, f)
    else:
        B, A = factor(12, 40, seed=1)
        m = 3
        n = B.shape[1]
        sets = np.array(list(__import__("itertools").combinations(range(n), m)))
        from mlsynth.utils.marex_helpers.exact import side_fit
        f = np.array([side_fit(B, A, list(S))[0]
                      + side_fit(B, A, [j for j in range(n) if j not in S])[0] for S in sets])
        todo = [("root", [], [], list(range(n)))]
    data = ExactData(B, A)
    n = B.shape[1]
    member = np.zeros((len(sets), n), bool)
    member[np.arange(len(sets))[:, None], sets] = True
    fstar = float(f.min())
    rows = []
    for lab, I, O, F in todo:
        mask = member[:, I].all(1) if I else np.ones(len(sets), bool)
        if O:
            mask &= ~member[:, O].any(1)
        sub = float(f[mask].min())
        t0 = time.perf_counter()
        ipm = branch.node_bound(data, m, I, O, F)
        t_ipm = time.perf_counter() - t0
        d = dual_bound(data, m, I, O, F)
        dt = dual_bound(data, m, I, O, F, target=fstar)
        rows.append({"node": lab, "sets": int(mask.sum()), "need": fstar / sub,
                     "ipm": None if ipm.lower is None else float(ipm.lower) / sub,
                     "ipm_seconds": t_ipm,
                     "dual": None if d["lower"] is None else float(d["lower"]) / sub,
                     "dual_exact": d["exact"] / sub, "dual_seconds": d["seconds"],
                     "dual_evaluations": d["evaluations"],
                     "targeted_reached": dt["reached"], "targeted_seconds": dt["seconds"],
                     "targeted_evaluations": dt["evaluations"]})
    return rows


def main(argv=None) -> int:
    """``PANEL``: one JSON line per node appended to ``results/compare.jsonl``."""
    import json
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "compare.jsonl", "a") as fh:
        for row in compare(args[0]):
            fh.write(json.dumps(dict(row, panel=args[0]), default=float) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())
