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


class Lift:
    def __init__(self, y, X, level, skip=(), keep_sym=True, pairs=False):
        self.scale = float(np.linalg.norm(y))
        y, X = y / self.scale, X / self.scale
        p = X.shape[1]
        G, c, yy = X.T @ X, X.T @ y, float(y @ y)
        self.lo, self.hi = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
        self.m = cp.Parameter(nonneg=True)
        lo, hi, m = self.lo, self.hi, self.m
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
                cons += [cp.quad_over_lin(w[j], u[j]) <= W[j, j] for j in range(p)]
            if level == "rlt":
                wc = cp.reshape(w, (p, 1), order="F"); row = np.ones((1, p))
                Wi = wc @ row
                cons += [W <= hi * Wi.T, W <= hi * Wi, W >= hi * (Wi + Wi.T) - hi * hi]
            obj = yy - 2 * c @ w + cp.trace(G @ W)
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def piece(self, b_lo, b_hi, m, big=1e6):
        self.lo.value, self.hi.value, self.m.value = b_lo, min(b_hi, big), m
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:  # pragma: no cover - Clarabel raises only on a numerical
            self.prob.solve(solver=cp.SCS, eps=1e-8, max_iters=20000)  # breakdown, never on the study's panels
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return np.inf
        return float(self.prob.value) * self.scale ** 2

