"""Warm-started simplex for the supergeo master LP, in numba, with the cut pool activated row by row.

The master over a fixed universe of groups (columns), with one artificial per market, the parity row for odd
N, and the cut rows that are active:

    min  c'x + BIG 1'a
    s.t. sum over c holding i of x_c + a_i = 1          (each market i)
         sum over odd c of x_c + sum_i a_i - s = 1       (odd N; each artificial counts, as in bcp._master)
         sum over c of coef_t(c) x_c + s_t = rhs_t       (each active cut t)
         x, a, s >= 0;  x_c = 0 for a column the node forbids.

Between a node and its child only bounds change (a forbidden column is fixed at 0) and cut rows are added, so
the parent's optimal basis stays dual feasible and the child is re-solved by the dual simplex from it. The cold
solve at the root starts from the basis of artificials, the parity surplus (n - 1 >= 0 with every artificial at
1) and the cut slacks, which is primal feasible.

Sources. P.-Q. Pan, Linear Programming Computation, 2nd ed., Springer, 2023 (doi:10.1007/978-981-19-0147-8):
  - Algorithm 3.3.1 (revised simplex), p. 76, and the basis-inverse update (3.21), p. 76;
  - Rule 3.5.2 (Bland's rule, finite), p. 85;
  - the pivot tolerance bounds (4.26) and Rule 4.7.1 (Harris's two-pass row rule), p. 112;
  - the bound violation (8.20) and row rule (8.21), p. 208; the entering set (8.51) and dual ratio test (8.52),
    p. 216; Algorithm 8.3.1 (generalized dual simplex for bounded variables), p. 218;
  - Rule 13.1.1 (dual steepest-edge row rule), p. 356, whose timing against Dantzig's rule is Table 13.1, p. 358.

Departures from the book, each stated where it is made:
  - the basis inverse is explicit and dense (the masters here have m <= about 100 active rows), updated by (3.21)
    and recomputed every REFACTOR pivots; Pan sections 4.3 to 4.5 (pp. 99-108) treat LU factors instead;
  - dual steepest-edge weights are computed exactly from the explicit inverse each pivot, in place of the
    recurrences (13.5) to (13.7), p. 357;
  - Rule 4.7.1 is applied to the dual ratio test (8.52) by the same two passes over columns;
  - row activation for the cut pool is ours: a cut row enters the LP when the solution violates it.

Floating point throughout, with tolerances; nothing here is exact. Never compile the exact layer (verify_nb,
exactnb, mitm_exact) with fastmath: its error-free transformations and its rounding margin assume each
operation is rounded as written. This module could take fastmath; it does not, so that its results match
HiGHS to the last digits the tests check."""
import numpy as np
from numba import njit
from . import bnp

TOL_P = 1e-9        # primal feasibility
TOL_D = 1e-9        # dual feasibility (reduced costs)
TOL_PIV = 1e-9      # pivot size, inside the range (4.26) of Pan, p. 112
HARRIS = 1e-9       # Harris's relaxation delta (Rule 4.7.1, p. 112)
REFACTOR = 64       # pivots between recomputations of the explicit inverse (ours)
DEGEN = 50          # consecutive degenerate pivots before Bland's rule (Rule 3.5.2, p. 85)
MAX_PIV = 20000


def cut_key(cut):
    return cut.key() if hasattr(cut, "key") else tuple(cut)


def cut_coef(cut, col):
    """Coefficient of a column in a cut row: a rank-1 cut's own, or the subset-row cut's 1 when the column holds
    two or more of the cut's three rows (Jepsen, Petersen, Spoorendonk and Pisinger 2008)."""
    if hasattr(cut, "coef"):
        return cut.coef(col)
    return 1 if len(set(col) & set(cut)) >= 2 else 0


def cut_rhs(cut):
    return cut.rhs if hasattr(cut, "rhs") else 1


@njit(cache=True)
def _update_inverse(Binv, col, r):
    """B^-1 of the basis with column r replaced: E_r B^-1, Pan (3.21), p. 76."""
    m = Binv.shape[0]
    piv = col[r]
    for k in range(m):
        Binv[r, k] /= piv
    for i in range(m):
        if i != r and col[i] != 0.0:
            f = col[i]
            for k in range(m):
                Binv[i, k] -= f * Binv[r, k]


@njit(cache=True)
def _primal(A, b, c, allowed, basis, Binv):
    """Revised primal simplex from a primal-feasible basis: Pan Algorithm 3.3.1, p. 76, pricing by Dantzig's rule
    (its step 3), the row by Harris's two passes (Rule 4.7.1, p. 112), and Bland's rule (Rule 3.5.2, p. 85) after
    DEGEN degenerate pivots in a row. A forbidden column that is basic has upper bound 0, so a step that would
    raise it is blocked at its current value (bounded variables as in Pan section 8.1). Returns (status, pivots):
    0 optimal, 2 pivot limit, 3 unbounded."""
    m, N = A.shape
    isb = np.zeros(N, np.bool_)
    for i in range(m): isb[basis[i]] = True
    piv = 0; degen = 0
    while piv < MAX_PIV:
        xB = Binv @ b
        y = c[basis] @ Binv
        # entering column
        q = -1; best = -TOL_D
        for j in range(N):
            if isb[j] or not allowed[j]: continue
            d = c[j]
            for i in range(m): d -= y[i] * A[i, j]
            if degen > DEGEN:
                if d < -TOL_D * (1.0 + abs(c[j])):
                    q = j; break                      # Bland: the smallest index with a negative reduced cost
            elif d < best:
                best = d; q = j
        if q < 0:
            return 0, piv
        col = Binv @ A[:, q]
        # pass 1 (Harris): the largest step with every bound relaxed by HARRIS
        a1 = np.inf
        for i in range(m):
            if col[i] > TOL_PIV:
                t = (max(xB[i], 0.0) + HARRIS) / col[i]
                if t < a1: a1 = t
            elif col[i] < -TOL_PIV and not allowed[basis[i]]:
                t = (max(-xB[i], 0.0) + HARRIS) / (-col[i])
                if t < a1: a1 = t
        if a1 == np.inf:
            return 3, piv
        # pass 2: among rows whose exact ratio is within a1, the largest pivot (Bland: the smallest basis index)
        r = -1; big = 0.0; rbest = np.inf
        for i in range(m):
            if col[i] > TOL_PIV:
                t = max(xB[i], 0.0) / col[i]
            elif col[i] < -TOL_PIV and not allowed[basis[i]]:
                t = max(-xB[i], 0.0) / (-col[i])
            else:
                continue
            if t <= a1:
                if degen > DEGEN:
                    if t < rbest - 1e-15 or (abs(t - rbest) <= 1e-15 and basis[i] < basis[r]):
                        rbest = t; r = i
                elif abs(col[i]) > big:
                    big = abs(col[i]); r = i
        theta = max(xB[r], 0.0) / col[r] if col[r] > 0 else 0.0
        degen = degen + 1 if theta <= 1e-12 else 0
        isb[basis[r]] = False; isb[q] = True; basis[r] = q
        _update_inverse(Binv, col, r)
        piv += 1
        if piv % REFACTOR == 0:
            Binv[:, :] = np.linalg.inv(A[:, basis])
    return 2, piv


@njit(cache=True)
def _dual(A, b, c, allowed, basis, Binv):
    """Dual simplex for bounds 0 <= x <= u with u = 0 on forbidden columns: Pan Algorithm 8.3.1, p. 218. The row is
    the largest bound violation (8.20) weighted by the dual steepest edge, Rule 13.1.1, p. 356, with the exact
    weights ||e_i' B^-1||^2 (not the recurrences (13.5) to (13.7)); every nonbasic column is at its lower bound 0,
    so the entering set (8.51) is the allowed nonbasic columns whose row entry has the opposite sign of the
    violation, and the ratio test (8.52) runs Harris's two passes (Rule 4.7.1, p. 112, applied to columns).
    Returns (status, pivots): 0 optimal, 1 infeasible (J of (8.51) empty), 2 pivot limit."""
    m, N = A.shape
    isb = np.zeros(N, np.bool_)
    for i in range(m): isb[basis[i]] = True
    piv = 0
    while piv < MAX_PIV:
        xB = Binv @ b
        # row: violation rho (8.20), weighted rho^2 / ||B^-T e_i||^2 (Rule 13.1.1)
        p = -1; best = 0.0; rho_p = 0.0
        for i in range(m):
            if xB[i] < -TOL_P:
                rho = -xB[i]
            elif xB[i] > TOL_P and not allowed[basis[i]]:
                rho = -xB[i]                         # above its upper bound 0: u - x < 0
            else:
                continue
            w = 0.0
            for k in range(m): w += Binv[i, k] * Binv[i, k]
            s = rho * rho / w
            if s > best:
                best = s; p = i; rho_p = rho
        if p < 0:
            return 0, piv
        y = c[basis] @ Binv
        alpha = Binv[p] @ A
        sgn = 1.0 if rho_p > 0 else -1.0
        # pass 1: the dual step with every reduced cost relaxed by HARRIS
        b1 = np.inf
        for j in range(N):
            if isb[j] or not allowed[j]: continue
            sj = sgn * alpha[j]                       # sigma_j of (8.46); J needs sigma_j < 0
            if sj < -TOL_PIV:
                d = c[j]
                for i in range(m): d -= y[i] * A[i, j]
                t = (max(d, 0.0) + HARRIS) / (-sj)
                if t < b1: b1 = t
        if b1 == np.inf:
            return 1, piv
        q = -1; big = 0.0
        for j in range(N):
            if isb[j] or not allowed[j]: continue
            sj = sgn * alpha[j]
            if sj < -TOL_PIV:
                d = c[j]
                for i in range(m): d -= y[i] * A[i, j]
                if max(d, 0.0) / (-sj) <= b1 and -sj > big:
                    big = -sj; q = j
        col = Binv @ A[:, q]
        isb[basis[p]] = False; isb[q] = True; basis[p] = q
        _update_inverse(Binv, col, p)
        piv += 1
        if piv % REFACTOR == 0:
            Binv[:, :] = np.linalg.inv(A[:, basis])
    return 2, piv


class Result:
    __slots__ = ("lp", "pi", "tau", "sigma", "x", "x_art", "art", "warm", "pivots", "active_rows", "status")


class Master:
    """The master over a fixed universe of columns; `solve` takes the node's allowed columns and the cut pool."""

    def __init__(self, cols, costs, n, odd, factor=1e3):
        costs = np.asarray(costs, float)
        if len(cols) != len(costs):
            raise ValueError(f"{len(cols)} columns and {len(costs)} costs")
        if not np.all(np.isfinite(costs)):
            raise ValueError("costs must be finite")
        self.cols = [tuple(int(v) for v in c) for c in cols]; self.costs = costs
        self.n, self.odd = int(n), bool(odd); self.C = len(cols)
        self.big = bnp.artificial_price(costs, factor)
        self.nfix = self.C + n + (1 if odd else 0)               # structural, artificials, parity surplus
        m0 = n + (1 if odd else 0)
        P = np.zeros((m0, self.nfix)); cf = np.zeros(self.nfix)
        for j, c in enumerate(self.cols):
            P[list(c), j] = 1.0
            if odd and len(c) % 2: P[n, j] = 1.0
        P[np.arange(n), self.C + np.arange(n)] = 1.0
        cf[:self.C] = costs; cf[self.C:self.C + n] = self.big
        if odd:
            P[n, self.C:self.C + n] = 1.0; P[n, self.C + n] = -1.0
        self.P, self.cf, self.m0 = P, cf, m0
        self.M = np.zeros((self.C, n))                          # column-by-market incidence
        for j, c in enumerate(self.cols): self.M[j, list(c)] = 1.0
        self._rows = {}                                         # cut key -> index into self._K
        self._K = np.zeros((64, self.C)); self._rhs = np.zeros(64); self._nk = 0

    def _row(self, cut):
        """Index of the cut's coefficient row: floor(sum of its multipliers over the column's rows / den),
        the subset-row cut's being multipliers 1 and den 2."""
        k = cut_key(cut)
        if k not in self._rows:
            S, num, den = (list(cut.S), np.array(cut.num, float), cut.den) if hasattr(cut, "S") else (list(cut), np.ones(3), 2)
            row = np.floor(self.M[:, S] @ num / den + 1e-9)
            if self._nk == len(self._rhs):                      # capacity doubling: no copy per cut
                self._K = np.vstack([self._K, np.zeros_like(self._K)]); self._rhs = np.concatenate([self._rhs, np.zeros_like(self._rhs)])
            self._rows[k] = self._nk; self._K[self._nk] = row; self._rhs[self._nk] = cut_rhs(cut); self._nk += 1
        return self._rows[k]

    def _system(self, active, cuts_by_key, big, phase1):
        k = len(active); m = self.m0 + k; N = self.nfix + k
        A = np.zeros((m, N)); A[:self.m0, :self.nfix] = self.P
        b = np.ones(m); c = np.zeros(N); c[:self.nfix] = self.cf
        if phase1:
            c[:self.C] = 0.0; c[self.C:self.C + self.n] = 1.0     # minimise the artificials alone
        else:
            c[self.C:self.C + self.n] = big
        for t, key in enumerate(active):
            A[self.m0 + t, :self.C] = self._K[self._rows[key]]
            A[self.m0 + t, self.nfix + t] = 1.0
            b[self.m0 + t] = cut_rhs(cuts_by_key[key])
        return A, b, c

    def solve(self, allowed, cuts, warm=None, add=20):
        """The node's LP. When an artificial is still in use, a phase-1 LP (zero column costs, unit artificials)
        decides: an infeasible node keeps its artificial, a feasible one is solved again with the artificials'
        price raised 1e3 times, up to 1e18 (the guard of bnp.solve_feasible)."""
        big = self.big
        while True:
            r = self._solve(allowed, cuts, warm, add, big, False)
            if r.art <= 1e-6 or big >= 1e18:
                return r
            if self._solve(allowed, cuts, None, add, 1.0, True).art > 1e-9:
                return r
            big *= 1e3; warm = None

    def _solve(self, allowed, cuts, warm, add, big, phase1):
        allowed = np.asarray(allowed, bool)
        if allowed.shape != (self.C,):
            raise ValueError(f"allowed has shape {allowed.shape}, the universe {self.C} columns")
        cuts_by_key = {}; pool_idx = []
        for cut in cuts:
            k = cut_key(cut)
            if k not in cuts_by_key:
                pool_idx.append(self._row(cut)); cuts_by_key[k] = cut
        pool_keys = list(cuts_by_key); pool_idx = np.array(pool_idx, np.int64)
        active, basis_ids = [], None
        if warm is not None and warm[0] is self:
            active = [k for k in warm[1] if k in cuts_by_key]
            ok = all(k in cuts_by_key for k in warm[1])
            basis_ids = list(warm[2]) if ok else None
            if not ok: active = []
        pivots = 0
        while True:
            A, b, c = self._system(active, cuts_by_key, big, phase1)
            m, N = A.shape
            allow = np.ones(N, np.bool_); allow[:self.C] = allowed
            pos = {("s", key): self.nfix + t for t, key in enumerate(active)}
            if basis_ids is None:
                basis = np.array(list(range(self.C, self.C + self.n)) + ([self.C + self.n] if self.odd else [])
                                 + [self.nfix + t for t in range(len(active))], np.int64)
                Binv = np.linalg.inv(A[:, basis])
                st, pv = _primal(A, b, c, allow, basis, Binv); pivots += pv
            else:
                basis = np.array([pos[v] if isinstance(v, tuple) else v for v in basis_ids], np.int64)
                Binv = np.linalg.inv(A[:, basis])
                st, pv = _dual(A, b, c, allow, basis, Binv); pivots += pv
                if st == 0:
                    st, pv = _primal(A, b, c, allow, basis, Binv); pivots += pv
                elif st == 2:                                   # stalled: a cold solve over the same rows
                    basis_ids = None; continue
            xB = np.linalg.inv(A[:, basis]) @ b
            x = np.zeros(N); x[basis] = xB; x[:self.C][~allowed] = 0.0
            back = {v: k for k, v in pos.items()}
            basis_ids = [back[j] if j in back else int(j) for j in basis]
            # row activation: the pool cuts this point violates, the worst first
            act = set(active)
            v = self._K[pool_idx] @ x[:self.C] - self._rhs[pool_idx] if len(pool_idx) else np.zeros(0)
            viol = [(float(v[t]), pool_keys[t]) for t in np.flatnonzero(v > TOL_P) if pool_keys[t] not in act]
            if not viol or st == 1:
                break
            viol.sort(reverse=True)
            for _, key in viol[:add]:
                active.append(key); basis_ids.append(("s", key))
        y = c[basis] @ np.linalg.inv(A[:, basis])
        r = Result()
        r.status = int(st); r.pivots = pivots; r.active_rows = len(active)
        r.lp = float(c @ x); r.x = x[:self.C].copy(); r.x_art = x[self.C:self.C + self.n].copy()
        r.art = float(x[self.C:self.C + self.n].sum())
        r.pi = y[:self.n].copy(); r.tau = max(float(y[self.n]), 0.0) if self.odd else 0.0
        sig = {key: min(float(y[self.m0 + t]), 0.0) for t, key in enumerate(active)}
        out, seen = [], set()                                   # a duplicate of a pool cut takes no dual of its own
        for cut in cuts:
            k = cut_key(cut); out.append(0.0 if k in seen else sig.get(k, 0.0)); seen.add(k)
        r.sigma = np.array(out)
        r.warm = (self, list(active), basis_ids)
        return r
