"""Lifts of the shared-slope problem on one slope piece [lo, hi] and size m (#965).

The exact set is w = b u, u binary, sum(u) = m, b in [lo, hi]. Levels, each a
relaxation of the one after it:

  shor  : the Shor lift [[1, w'], [w, W]] >= 0 (Shor 1987), with the
          reformulation-linearization products (Sherali and Adams 1999) of the
          bound w_j^2 = b w_j in [lo w_j, hi w_j], of the size constraint
          multiplied by each w_i, and W >= 0.
  persp : + indicators u, lo u <= w <= hi u, sum u = m, and the perspective
          W_jj u_j >= w_j^2 (De Rosa and Khajavirad 2026, the set S_n; Atamturk
          and Gomez 2025, the perspective as the singleton case of their eq. (7)).
  rlt   : + the products (hi - w_i) w_j >= 0 and (hi - w_i)(hi - w_j) >= 0
          (Sherali and Adams 1999).
  full  : the joint lift of x = (u, w), Z = [[1, u', w'], [u, U, V], [w, V', W]]:
          diag(U) = u (u binary), diag(V) = w (u_j w_j = w_j), the size equality
          times every variable, every product of the bound factors u, 1 - u,
          w - lo u, hi u - w (Sherali and Adams 1999), and V = V'.
          V = V' is derived here: u_i w_j = b u_i u_j = w_i u_j when every
          selected control carries the same b. Measured on the GeoLift markets
          (#965), removing it reopens a 12% root gap, and none of the hulls in
          Atamturk and Gomez (2025, Thm 2), Wei, Gomez and Kucukyavuz (2022,
          Thm 1) or Han, Gomez and Atamturk (2023) covers it: all three take the
          continuous variables as free given the support.
  pairs : optional 2x2 cones u_i u_j [[1, b], [b, b^2]] >= 0 and
          u_i (1 - u_j) [[1, b], [b, b^2]] >= 0, written in the lift's variables;
          the pairwise disjunction of Han, Gomez and Atamturk (2023), eq. (5),
          valid here because the slope is shared (derived here).

A piece's value is a lower bound by weak duality (#1087), not the solver's primal
value. Any multipliers (lambda >= 0 on the inequality rows, mu free on the
equalities, a point of the second-order cone on each cone row) give a Lagrangian
L <= objective on the feasible set. L is affine in (Z, u), with matrix S in Z, so
its minimum over the superset {Z PSD, trace(Z) <= T} x [0, 1]^p is
const + T min(0, lambda_min(S)) + sum_j min(0, a_j). Multipliers off their cones
are clipped or projected first, so the bound holds whatever the solver returned.
The full lift's feasible set lies on the face Zv = 0, v = (-m, 1, ..., 1, 0, ..., 0),
which has no strictly feasible point; there Clarabel stops "optimal_inaccurate"
with a primal value up to 3.75e-4 above the true minimum (#1087), and the
multipliers still give a bound within 1.5e-4 of it. The perspective rows are
written as cones ||(2 w_j, u_j - W_jj)|| <= u_j + W_jj, the same set as
w_j^2 <= u_j W_jj with u_j, W_jj >= 0, so that their multipliers are the cone's
own: the tangent of w_j^2 / u_j at u_j = 0 is undefined, and through it the bound
lost up to 2.4x on accurate solves.

Shor, N. Z. (1987). Quadratic optimization problems. Soviet J. Comput. Systems
    Sci. 25, 1-11.
Sherali, H. D. and Adams, W. P. (1999). A Reformulation-Linearization Technique
    for Solving Discrete and Continuous Nonconvex Problems. Kluwer.
De Rosa, A. and Khajavirad, A. (2026). Explicit convex hull description of
    bivariate quadratic sets with indicator variables. Math. Program. 218, 169-211.
Atamturk, A. and Gomez, A. (2025). Rank-one convexification for sparse
    regression. J. Mach. Learn. Res.
Wei, L., Gomez, A. and Kucukyavuz, S. (2022). Ideal formulations for constrained
    convex optimization problems with indicator variables. Math. Program. 192, 57-88.
Han, S., Gomez, A. and Atamturk, A. (2023). 2x2-convexifications for convex
    quadratic optimization with indicator variables. Math. Program. 202, 95-134.
"""
import numpy as np
import cvxpy as cp
from cvxpy.constraints import PSD, SOC, Equality, Inequality


class Lift:
    def __init__(self, y, X, level, skip=(), keep_sym=True, pairs=False):
        self.scale = float(np.linalg.norm(y))
        y, X = y / self.scale, X / self.scale
        p = X.shape[1]
        G, c, yy = X.T @ X, X.T @ y, float(y @ y)
        self.lo, self.hi = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
        self.m = cp.Parameter(nonneg=True)
        lo, hi, m = self.lo, self.hi, self.m
        self.level, self.status, self.value, self._u = level, None, None, None
        # the diagonal rows trace_bound reads: (w - lo u)(hi u - w) >= 0 and u (hi u - w) >= 0
        self._p = p
        self._capped = bool({("a", "c"), ("c", "a")} - set(skip))
        self._wcap = bool({("u", "c"), ("c", "u")} - set(skip))
        if level == "full":
            Z = cp.Variable((2 * p + 1, 2 * p + 1), symmetric=True)
            x = Z[1:, 0]; u, w = x[:p], x[p:]
            U, V, W = Z[1:p + 1, 1:p + 1], Z[1:p + 1, p + 1:], Z[p + 1:, p + 1:]
            one = np.ones(p)
            uc, wc = cp.reshape(u, (p, 1), order="F"), cp.reshape(w, (p, 1), order="F")
            ur, wr = cp.reshape(u, (1, p), order="F"), cp.reshape(w, (1, p), order="F")
            col = np.ones((p, 1)); row = np.ones((1, p))
            # rows-as-i, cols-as-j matrices of the linear terms
            Ui, Uj = uc @ row, col @ ur          # u_i, u_j
            Wi, Wj = wc @ row, col @ wr          # w_i, w_j
            Vt = V.T                             # (w_i u_j)
            # factors as (lifted products with u_j, with w_j, linear term in i)
            # f = u:      f_i u_j = U,       f_i w_j = V
            # f = 1-u:    u_j - U,           w_j - V
            # f = w-lo u: Vt - lo U,         W - lo V
            # f = hi u-w: hi U - Vt,         hi V - W
            F = {"u": (U, V, Ui), "1-u": (Uj - U, Wj - V, 1 - Ui),
                 "a": (Vt - lo * U, W - lo * V, Wi - lo * Ui),
                 "c": (hi * U - Vt, hi * V - W, hi * Ui - Wi)}
            # g_j for the same four factors as a function of f_i's (u_j, w_j, 1) terms
            def times(fi, g):
                pu, pw, lin = fi
                if g == "u": return pu
                if g == "1-u": return lin - pu
                if g == "a": return pw - lo * pu
                return hi * pu - pw
            cons = [Z >> 0, Z[0, 0] == 1, cp.diag(U) == u, cp.diag(V) == w,
                    cp.sum(u) == m,
                    cp.sum(U, axis=1) == m * u, cp.sum(V, axis=0) == m * w]
            if keep_sym:
                cons.append(V == V.T)
            for a in F:
                for g in F:
                    if (a, g) not in skip:
                        cons.append(times(F[a], g) >= 0)
            if pairs:
                # Pairwise cones (after Han, Gomez and Atamturk 2023, eq. 5, carried into
                # this lift): for a real design u_i u_j [[1,b],[b,b^2]] and
                # u_i (1 - u_j) [[1,b],[b,b^2]] are PSD because the slope is shared.
                iu, ju = np.triu_indices(p, 1)
                a, b_, c_ = U[iu, ju], V[iu, ju], W[iu, ju]
                cons.append(cp.SOC(a + c_, cp.vstack([2 * b_, a - c_])))
                I, J = np.nonzero(~np.eye(p, dtype=bool))
                a = u[I] - U[I, J]; b_ = w[I] - V[J, I]; c_ = cp.diag(W)[I] - W[I, J]
                cons.append(cp.SOC(a + c_, cp.vstack([2 * b_, a - c_])))
            obj = yy - 2 * c @ w + cp.trace(G @ W)
        else:
            Z = cp.Variable((p + 1, p + 1), symmetric=True)
            w, W = Z[1:, 0], Z[1:, 1:]
            cons = [Z >> 0, Z[0, 0] == 1, w >= 0, w <= hi,
                    cp.diag(W) >= lo * w, cp.diag(W) <= hi * w,
                    cp.sum(w) >= m * lo, cp.sum(w) <= m * hi,
                    cp.sum(W, axis=1) >= m * lo * w, cp.sum(W, axis=1) <= m * hi * w,
                    W >= 0]
            if level in ("persp", "rlt"):
                u = cp.Variable(p)
                cons += [u >= 0, u <= 1, cp.sum(u) == m, w >= lo * u, w <= hi * u]
                dW = cp.diag(W)
                cons.append(cp.SOC(u + dW, cp.vstack([2 * w, u - dW])))   # w_j^2 <= u_j W_jj
                self._u = u
            if level == "rlt":
                wc = cp.reshape(w, (p, 1), order="F"); row = np.ones((1, p))
                Wi = wc @ row
                cons += [W <= hi * Wi.T, W <= hi * Wi, W >= hi * (Wi + Wi.T) - hi * hi]
            obj = yy - 2 * c @ w + cp.trace(G @ W)
        self._Z = Z
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def piece(self, b_lo, b_hi, m, big=1e6, tol=None):
        """A lower bound on the lift's minimum over slopes in [b_lo, b_hi] and size m,
        in the units of y: ``bound()`` from the solve's multipliers. The solve's
        status and primal value are kept on ``status`` and ``value``. ``tol`` sets
        Clarabel's three stopping tolerances (absolute and relative gap,
        feasibility); None keeps Clarabel's own."""
        self.lo.value, self.hi.value, self.m.value = b_lo, min(b_hi, big), m
        kw = {} if tol is None else dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
        try:
            self.prob.solve(solver=cp.CLARABEL, max_threads=1, **kw)   # fork-safe (slopefree.solve)
        except cp.error.SolverError:  # pragma: no cover - Clarabel raises only on a numerical
            self.prob.solve(solver=cp.SCS, eps=1e-8, max_iters=20000)  # breakdown, never on the study's panels
        self.status = self.prob.status
        if self.status not in ("optimal", "optimal_inaccurate"):
            self.value = None
            return np.inf
        self.value = float(self.prob.value) * self.scale ** 2
        return self.bound()

    def trace_bound(self):
        """T with trace(Z) <= T on the feasible set at the current parameters.
        Below the full lift, 1 + m hi^2 from diag W <= hi w and sum w <= m hi.
        The full lift has trace U = sum u = m, and the diagonal of
        (w - lo u)(hi u - w) >= 0 gives W_jj <= (hi + lo) w_j - lo hi u_j, so
        trace W <= (hi + lo) sum w - lo hi m. The diagonal of u (hi u - w) >= 0
        gives sum w <= hi m, and trace W <= m hi^2; without it, w_j^2 <= W_jj
        (Z PSD) caps w_j at hi + lo, and sum w <= p (hi + lo). Without the first
        diagonal nothing caps W, and T is infinite."""
        m, lo, hi = float(self.m.value), float(self.lo.value), float(self.hi.value)
        if self.level != "full":
            return 1 + m * hi * hi
        if not self._capped:
            return np.inf
        sw = hi * m if self._wcap else self._p * (hi + lo)
        return 1 + m + (hi + lo) * sw - lo * hi * m

    def lagrangian(self, duals=None):
        """The Lagrangian of ``duals`` (one per constraint, in the problem's order; the
        last solve's by default) as an affine expression in (Z, u), in the scaled
        units the problem is built in; None when a multiplier is missing. It is at
        most the objective on the feasible set."""
        cons = self.prob.constraints
        duals = [c.dual_value for c in cons] if duals is None else list(duals)
        if len(duals) != len(cons):
            raise ValueError(f"one multiplier per constraint: {len(duals)} for {len(cons)}")
        terms = [self.prob.objective.args[0]]
        for c, d in zip(cons, duals):
            if isinstance(c, PSD):
                continue                        # the PSD cone is the minimization's own set
            if d is None:
                return None
            if isinstance(c, Inequality):       # expr <= 0, lambda >= 0
                terms.append(cp.sum(cp.multiply(np.maximum(np.asarray(d, float), 0.0), c.expr)))
            elif isinstance(c, Equality):
                terms.append(cp.sum(cp.multiply(np.asarray(d, float), c.expr)))
            elif isinstance(c, SOC):            # (t, x) in the cone: subtract <(dt, dx), (t, x)> >= 0
                t, x = c.args
                dt, dx = _soc_project(np.atleast_1d(np.asarray(d[0], float)),
                                      np.asarray(d[1], float).reshape(-1, t.size, order="F"))
                terms.append(-cp.sum(cp.multiply(dt, t)) - cp.sum(cp.multiply(dx.reshape(x.shape, order="F"), x)))
            else:  # pragma: no cover - the lift builds only these four kinds
                raise TypeError(f"no Lagrangian term for {type(c).__name__}")
        return cp.sum(cp.hstack([cp.reshape(t, (1,), order="F") for t in terms]))

    def bound(self, duals=None):
        """The weak-duality lower bound from ``duals`` (as for ``lagrangian``): the
        Lagrangian's minimum over {Z PSD, trace Z <= T} x [0, 1]^p, in the units of
        y. -inf when a multiplier is missing."""
        lag = self.lagrangian(duals)
        if lag is None:
            return -np.inf
        keep = {v: v.value for v in self.prob.variables()}
        for v in keep:
            v.value = np.zeros(v.shape)
        const, grad = float(lag.value), lag.grad
        for v, val in keep.items():
            v.value = val
        n = self._Z.shape[0]
        S = np.asarray(grad[self._Z].todense()).reshape(n, n, order="F")
        lam = float(np.linalg.eigvalsh((S + S.T) / 2)[0])
        out = const + (lam * self.trace_bound() if lam < 0 else 0.0)
        if self._u is not None:                  # u in [0, 1]^p
            out += float(np.minimum(np.asarray(grad[self._u].todense()).ravel(), 0.0).sum())
        return out * self.scale ** 2


def _soc_project(t, x):
    """Each column (t_k, x_k) projected onto the cone {|x| <= t}."""
    t, x = t.copy(), x.copy()
    nx = np.linalg.norm(x, axis=0)
    for k in range(len(t)):
        if nx[k] <= t[k]:
            continue
        if nx[k] <= -t[k]:
            t[k], x[:, k] = 0.0, 0.0
            continue
        a = (t[k] + nx[k]) / 2
        t[k], x[:, k] = a, x[:, k] * (a / nx[k])
    return t, x

