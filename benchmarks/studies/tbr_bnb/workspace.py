"""The slope-free full relaxation on one SCS workspace per size (#965).

At 96 candidate controls an interior-point solve of a size's full relaxation (a
97 x 97 cone and about 18,400 rows) takes Clarabel 100-145 s, each iteration
factoring the cone's block. SCS (O'Donoghue, Chu, Parikh and Boyd 2016; O'Donoghue
2021 for version 3) is ADMM on the homogeneous self-dual embedding: an iteration
projects onto the cone and solves one linear system whose matrix,
[[rho I, A'], [A, -I/sigma]], depends only on the rows. Called through cvxpy, SCS
factors that matrix again at every node and starts from zero, and on the test beds
it took more nodes than Clarabel and left 72 groups uncertified against 11
(README, "SCS, an option, not the default"). Here it is called directly, and three
things change:

1. One workspace for a size, its cuts and its triangle rows. A node's fixing
   u_j = v is two box rows, u_j <= hi_j and -u_j <= -lo_j, with lo_j = hi_j = v; a
   free geo has lo_j = 0, hi_j = 1, which every design satisfies. Fixings change
   only the right-hand side, so every node of the size reuses the factorization
   (``scs.SCS.update``), and each node starts from the last node's iterate.
2. Two stages. The first runs ``STAGES[0]`` (a small dual scale) for a few chunks
   and the second the default scale; each node runs them in order, both from the
   node's warm start.
3. A certificate after every chunk of ``CHUNK`` iterations. The multipliers go to
   fbound.certificate on the node's own rows, the rigorous float bounds first and
   the exact certificate (slopefree.bound) only between them; the first chunk the
   certificate closes ends the solve. SCS's stopping test decides nothing that is
   proved.

The certificate holds for any multipliers (slopefree's module docstring), so the
bound is valid at every outcome; how tight it is depends on how far SCS got. Each
solve ends in one of ``OUTCOMES`` (agents_solvers.md, rule 1): "certified" (a
chunk's certificate is at most zero), "converged" (SCS's own test passed at the
last stage, the certificate above zero), "cap" (every stage spent its chunks,
rule 7) or "infeasible" (SCS reports no point: the node holds no design). Each is
reported as the solve event's status. An SCS error is reported and the node is
solved again by Clarabel.

``Workspaces`` holds one search's workspaces and its triangle pool: the triangle
rows active at the last root solved (a multiplier above ``ACTIVE`` times the
largest triangle multiplier). size_certificate starts every node's full relaxation
from the pool, so a size begins with the rows the size before it needed. The pool
belongs to one certify_slope_free call: carried across groups it reads triangles
of another group's controls."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scs

from mlsynth.exceptions import MlsynthConfigError

from . import fbound, slopefree as sf
from .progress import emit

SQ2 = math.sqrt(2.0)

#: The ends a workspace solve reports (module docstring).
OUTCOMES = ("certified", "converged", "cap", "infeasible")

#: Each stage's SCS settings and its chunks. alpha is the relaxation parameter and
#: scale the dual scale factor (SCS 3 settings). On the 100-geo group (4, 35, 58,
#: 60) a node at the default scale alone took about twice the iterations of one
#: started at 0.01 and finished at the default (scratch measurements of the spike).
STAGES: Tuple[Tuple[Dict, int], ...] = ((dict(alpha=1.8, scale=0.01), 2), (dict(alpha=1.8), 100))

#: SCS iterations between certificates.
CHUNK = 200

#: SCS's absolute and relative tolerances, which decide only when a stage ends.
EPS = 1e-7

#: A triangle row stays in the pool when its multiplier is above this fraction of
#: the largest triangle multiplier at the root that set the pool.
ACTIVE = 1e-7


def _svec_index(n):
    """SCS's vectorisation of a symmetric matrix: the lower triangle by columns,
    the off-diagonal entries scaled by sqrt(2). (i, j), i >= j -> position."""
    idx, k = {}, 0
    for j in range(n):
        for i in range(j, n):
            idx[(i, j)] = k
            k += 1
    return idx


@dataclass
class Outcome:
    kind: str                            # one of OUTCOMES
    scs_status: str
    iters: int
    seconds: float
    bound: Optional[object]              # the exact certificate when a chunk computed it
    y_eq: Optional[np.ndarray]           # multipliers in the data's units on R's rows
    y_in: Optional[np.ndarray]
    Z: Optional[np.ndarray]              # the point, None when SCS reports none


class SizeWorkspace:
    """The SCS workspaces (one a stage) of size ``m``, its cuts and its triangle
    rows; nodes differ by their fixings only."""

    def __init__(self, c, G, lam, m, cuts, triangles):
        lam = sf.dyadic_below(lam)
        self.c, self.m, self.p, self.n = c, m, len(c), len(c) + 1
        self.A = sf.objective(c, G, lam)
        self.af = fbound.objective_float(self.A)
        A = np.asarray(self.A, dtype=float)
        self.s = sf._pow2(float(np.abs(A).max()))
        self.cs = sf._pow2(float(np.max(np.abs(np.asarray(c, dtype=float)))))
        n, p = self.n, self.p
        R0 = sf.Rows(c, m, cuts=cuts, triangles=triangles, level="full")   # the layout, no fixings
        self.n_eq, self.n_in0, self.c_rows = R0.n_eq, R0.n_in, list(R0.c_rows)
        self.idx = _svec_index(n)
        nv = n * (n + 1) // 2
        Me, Mi = R0.matrices(self.cs)
        rows, cols, vals = [], [], []
        for j in range(p):                                    # u_j <= hi_j, -u_j <= -lo_j
            k = self.idx[(j + 1, 0)]
            rows += [2 * j, 2 * j + 1]; cols += [k, k]; vals += [1.0 / SQ2, -1.0 / SQ2]
        Mbox = sp.csc_matrix((vals, (rows, cols)), shape=(2 * p, nv))
        self.Ascs = sp.vstack([self._to_svec(Me), self._to_svec(Mi), Mbox, -sp.identity(nv, format="csc")],
                              format="csc")
        self.b_rows = np.concatenate([np.asarray(R0.beq, dtype=float), np.asarray(R0.bin, dtype=float)])
        A0 = np.zeros((n, n)); A0[1:, 1:] = A / self.s
        self.cvec = np.array([-A0[i, j] if i == j else -SQ2 * A0[i, j] for (i, j) in self.idx])
        self.cone = dict(z=self.n_eq, l=self.n_in0 + 2 * p, s=[n])
        self.stages = STAGES                                  # read at build, so a test can shorten them
        self.chunk = CHUNK
        self._ws = [None] * len(self.stages)
        self.last = None                                      # (x, y, s) of the last node solved

    def _to_svec(self, M):
        """Rows over vec(Z) (column-major, keys a <= b at a + n b) to rows over SCS's
        vectorisation."""
        n = self.n
        M = M.tocoo()
        cols = np.empty_like(M.col); vals = M.data.copy()
        for t, col in enumerate(M.col):
            a, b = col % n, col // n
            cols[t] = self.idx[(b, a)]
            if a != b:
                vals[t] /= SQ2
        return sp.csc_matrix((vals, (M.row, cols)), shape=(M.shape[0], n * (n + 1) // 2))

    def _b(self, fixes):
        hi, lo = np.ones(self.p), np.zeros(self.p)
        for j, v in fixes:
            hi[j] = lo[j] = v
        box = np.empty(2 * self.p); box[0::2] = hi; box[1::2] = -lo
        return np.concatenate([self.b_rows, box, np.zeros(self.n * (self.n + 1) // 2)])

    def _workspace(self, k, fixes):
        b = self._b(fixes)
        if self._ws[k] is None:
            kw = dict(eps_abs=EPS, eps_rel=EPS, max_iters=self.chunk, verbose=False)
            kw.update(self.stages[k][0])
            self._ws[k] = scs.SCS(dict(A=self.Ascs, b=b, c=self.cvec), self.cone, **kw)
        else:
            self._ws[k].update(b=b)
        return self._ws[k]

    def carry_from(self, prev: "SizeWorkspace", fixes=()):
        """Start from ``prev``'s last iterate, a workspace of the same size and cuts
        whose triangle rows are a prefix of these: x as it is, the old rows' y and s
        at their places, the new rows' multipliers zero and their slacks b - a'x
        clipped at zero."""
        if prev.last is None:
            return
        x, y, s = prev.last
        lo, hi = self.n_eq + prev.n_in0, self.n_eq + self.n_in0
        A_new = self.Ascs[lo:hi, :]
        s_new = np.maximum(self._b(fixes)[lo:hi] - A_new @ x, 0.0)
        self.last = (x.copy(), np.concatenate([y[:lo], np.zeros(hi - lo), y[lo:]]),
                     np.concatenate([s[:lo], s_new, s[lo:]]))

    def _multipliers(self, y, fixes):
        """SCS's dual y to R's multipliers in the data's units: R's equalities (each
        fixing's is y_hi - y_lo of its box rows), then R's inequalities, the free
        geos' box rows last."""
        s, p = self.s, self.p
        ye = y[:self.n_eq] * s
        yi = y[self.n_eq:self.n_eq + self.n_in0] * s
        yi[self.c_rows] /= self.cs
        ybox = y[self.n_eq + self.n_in0:self.n_eq + self.n_in0 + 2 * p] * s
        fixed = dict(fixes)
        y_eq = np.concatenate([ye, [ybox[2 * j] - ybox[2 * j + 1] for j, _ in fixes]])
        y_in = np.concatenate([yi, [ybox[i] for j in range(p) if j not in fixed for i in (2 * j, 2 * j + 1)]])
        return y_eq, y_in

    def _Z(self, x):
        Z = np.zeros((self.n, self.n))
        for (i, j), k in self.idx.items():
            Z[i, j] = Z[j, i] = x[k] if i == j else x[k] / SQ2
        return Z

    def solve(self, R: "sf.Rows") -> Outcome:
        """The node of R (this size, cuts and triangles, R's fixings, R's box rows)."""
        t0 = time.perf_counter()
        fixes = R.fixes
        iters, status, b = 0, "unstarted", None
        y_eq = y_in = Z = None
        start = self.last
        done = False
        for k, (_, chunks) in enumerate(self.stages):          # rule 7: each stage has its chunks
            ws = self._workspace(k, fixes)
            self.last = start
            for _ in range(chunks):
                if self.last is None:
                    sol = ws.solve(warm_start=False)
                else:
                    sol = ws.solve(warm_start=True, x=self.last[0], y=self.last[1], s=self.last[2])
                self.last = (sol["x"], sol["y"], sol["s"])
                iters += sol["info"]["iter"]; status = sol["info"]["status"]
                if status in ("infeasible", "infeasible_inaccurate"):
                    y_eq, y_in = self._multipliers(sol["y"], fixes)
                    b = fbound.certificate(self.A, R, y_eq, y_in, self.m, self.af)
                    kind = "certified" if b is not None and b <= 0 else "infeasible"
                    return Outcome(kind, status, iters, time.perf_counter() - t0, b, y_eq, y_in, None)
                y_eq, y_in = self._multipliers(sol["y"], fixes)
                Z = self._Z(sol["x"])
                done = status in ("solved", "solved_inaccurate")
                b = fbound.certificate(self.A, R, y_eq, y_in, self.m, self.af)
                if b is not None and b <= 0:
                    return Outcome("certified", status, iters, time.perf_counter() - t0, b, y_eq, y_in, Z)
                if done:
                    break                                         # this stage converged: the next one
        return Outcome("converged" if done else "cap", status, iters, time.perf_counter() - t0, b, y_eq, y_in, Z)


class Workspaces:
    """One search's workspaces, kept for the current ratio, size and cuts, and its
    triangle pool (module docstring)."""

    def __init__(self):
        self.spaces: Dict[tuple, SizeWorkspace] = {}
        self.pool: Tuple = ()

    def solve(self, c, G, lam, R: "sf.Rows") -> "sf.Solve":
        if R.level != "full":
            raise MlsynthConfigError(f"the workspace solves the full relaxation, not the {R.level!r} one.")
        if not R.box:
            raise MlsynthConfigError("the workspace's rows carry the box rows 0 <= u_j <= 1 (Rows(box=True)).")
        lam = sf.dyadic_below(lam)
        base = (lam, R.m, R.cuts)
        key = base + (R.triangles,)
        if key not in self.spaces:
            prev = None
            for kk, w in self.spaces.items():                # the last round's workspace: a prefix of the rows
                if kk[:3] == base and len(kk[3]) < len(R.triangles) and R.triangles[:len(kk[3])] == kk[3]:
                    prev = w
            self.spaces = {kk: w for kk, w in self.spaces.items() if kk[:3] == base}
            self.spaces[key] = SizeWorkspace(c, G, lam, R.m, [set(C) for C in R.cuts], R.triangles)
            if prev is not None:
                self.spaces[key].carry_from(prev, R.fixes)
        ws = self.spaces[key]
        t0 = time.perf_counter()
        try:
            o = ws.solve(R)
        except (RuntimeError, ValueError):                    # reported; Clarabel solves the node
            emit("solve", relaxation="full", solver="workspace", rows_eq=R.n_eq, rows_in=R.n_in, status="error",
                 seconds=time.perf_counter() - t0)
            ws.last = None
            return sf.solve(c, G, lam, R, solver="clarabel")
        emit("solve", relaxation="full", solver="workspace", rows_eq=R.n_eq, rows_in=R.n_in, status=o.kind,
             seconds=o.seconds)
        if not R.fixes and o.y_in is not None:
            lo, hi = R.triangle_rows
            yt = np.asarray(o.y_in[lo:hi], dtype=float)
            top = float(yt.max()) if yt.size else 0.0
            self.pool = tuple(t for t, v in zip(R.triangles, yt) if top > 0 and v > ACTIVE * top)
        if o.Z is None:
            value = -np.inf
        else:
            value = float(np.sum(np.asarray(ws.A, dtype=float) * o.Z[1:, 1:])) / ws.s
        return sf.Solve(value, ws.s, o.y_eq, o.y_in, o.Z, o.kind, solver="workspace", bound=o.bound)
