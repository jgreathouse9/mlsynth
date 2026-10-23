"""Predictor weights V that rationalise given donor weights w*, without cvxpy (paper, sec. 4.5).

Every reported V maximises a sum of logarithms over an affine slice of the cone:

  - analytic centre: log det V + sum_j log s_j(V) subject to the KKT equalities and
    trace one; one run of Newton's method with the equalities eliminated (Boyd and
    Vandenberghe, Convex Optimization, sec. 10.2);
  - log-det maximiser: log det V subject to the equalities and s_j(V) >= 0, by the
    barrier method (ch. 11): minimise -log det V - mu sum log s_j(V) as mu falls;
  - identified range of V_kk: a linear objective over the same set, by the barrier
    method with -log det V as the barrier of the positive semidefinite cone.

Positive definite class: the strictly feasible start is the paper's own Lemma B.1
construction, V0 = nu nu'/(nu'r) + I - r r'/||r||^2 with nu in the relative interior of
the normal cone (one LP), so no phase-I problem is solved. Diagonal class: the support
and the implicit equalities come from one LP over the cone of rationalisers, which also
gives the relative-interior start (``_diag_classes``).

Inside the donor hull w is an exact match, r = 0, and those conditions say nothing. V
rationalises the barycentric weights w_A of cell A when A is the selection under V:
<Q_j^A, V> > 0 for every donor j outside A (paper, eq. 18; interior.py). The centre and
the log-det maximiser are then taken over that set, from the metric that proved the
cell attainable, and the selection rows are the only inequalities: inside the hull no
predictor is forced to zero and no inequality holds with equality throughout.

Predictors are standardised across treated and donors (sd, ddof=1), as in the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .hull import outside_hull

TOL = 1e-9


def _standardize(X1, X0):
    sd = np.column_stack([X1, X0]).std(axis=1, ddof=1)
    return X1 / sd, X0 / sd[:, None]


def _null(A):
    """An orthonormal basis of the null space of A."""
    _, sv, Vt = np.linalg.svd(A)
    rank = int((sv > 1e-12 * max(sv[0], 1.0)).sum())
    return Vt[rank:].T


def _newton_eq(x, vgh, A, b, in_domain, fun=None, N=None, tol=1e-12, maxit=200):
    """Newton's method on {x : A x = b} from a strictly feasible x, in the null space of A.

    ``vgh`` gives the objective, gradient and Hessian, and ``fun`` the objective alone,
    +inf outside the domain, so each trial step of the line search costs one function
    value. ``N`` is a null-space basis of A, passed in when several runs share A.
    """
    if not in_domain(x):
        raise ValueError("Newton's method needs a strictly feasible start")
    if fun is None:
        fun = lambda z: vgh(z)[0] if in_domain(z) else np.inf   # noqa: E731
    N = _null(A) if N is None else N
    if N.shape[1] == 0:
        return x
    for _ in range(maxit):
        f, g, H = vgh(x)
        gz = N.T @ g
        Hz = N.T @ H @ N
        dz = -np.linalg.solve(Hz + 1e-14 * np.trace(Hz) / len(Hz) * np.eye(len(Hz)), gz)
        lam2 = float(-gz @ dz)                  # squared Newton decrement
        if lam2 / 2.0 <= tol:
            break
        dx = N @ dz
        step = 1.0
        while fun(x + step * dx) > f - 0.25 * step * lam2:
            step *= 0.5
            if step < 1e-16:
                return x
        x = x + step * dx
    return x


def _basis(K):
    """Symmetric basis B_a over the upper triangle, so that V = sum_a x_a B_a."""
    idx = [(k, l) for k in range(K) for l in range(k, K)]
    B = np.zeros((len(idx), K, K))
    for a, (k, l) in enumerate(idx):
        B[a, k, l] = 1.0
        B[a, l, k] = 1.0
    return B


def _relint_normal(X0, w, r):
    """nu in the relative interior of N(supp w) with nu'r > 0, by maximising the least slack."""
    K, J = X0.shape
    S = np.flatnonzero(w > 1e-7)
    a0 = S[0]
    others = np.setdiff1d(np.arange(J), S)
    A_ub = np.vstack([np.hstack([(X0[:, others] - X0[:, [a0]]).T, np.ones((len(others), 1))]),
                      np.concatenate([-r, [1.0]])[None, :]])
    A_eq = np.hstack([(X0[:, S[1:]] - X0[:, [a0]]).T, np.zeros((len(S) - 1, 1))]) if len(S) > 1 else None
    res = linprog(np.concatenate([np.zeros(K), [-1.0]]), A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]),
                  A_eq=A_eq, b_eq=np.zeros(len(S) - 1) if A_eq is not None else None,
                  bounds=[(-1, 1)] * K + [(None, None)], method="highs")
    if res.status != 0 or -res.fun <= TOL:
        raise ValueError("w is not rationalised by any positive definite V")
    return res.x[:K]


def pd_rationalizers(X1, X0, w, ranges=False):
    """Log-det maximiser and analytic centre (and optionally ranges) in the positive definite class.

    Returns trace-normalised diagonals ``logdet_diag`` and ``center_diag`` (the paper's
    V_kk / tr V), the matrices ``Vstar`` and ``Vcenter``, ``logdet_min_eig``, and with
    ``ranges=True`` the bounds ``lo`` and ``hi`` of each V_kk over the trace-one set.
    """
    if not outside_hull(np.asarray(X1, float), np.asarray(X0, float)):
        return _pd_interior(X1, X0, w, ranges)
    X1, X0 = _standardize(np.asarray(X1, float), np.asarray(X0, float))
    w = np.asarray(w, float)
    K, J = X0.shape
    S = np.flatnonzero(w > 1e-7)
    inactive = np.setdiff1d(np.arange(J), S)
    j0 = S[0]
    r = X1 - X0 @ w
    p = X0 @ w - X1                            # the paper's (X0 w* - X1)
    B = _basis(K)
    n = len(B)
    Bm = B.reshape(n, K * K)                   # x -> vec V(x)

    def lin(M):                                # <M, V(x)> as a row acting on x
        return Bm @ M.ravel()

    Aeq = [lin(np.eye(K))]                     # trace one
    beq = [1.0]
    for j in S[1:]:                            # active donors, equation (11)
        q = X0[:, j] - X0[:, j0]
        Aeq.append(lin(0.5 * (np.outer(q, p) + np.outer(p, q))))
        beq.append(0.0)
    Aeq, beq = np.array(Aeq), np.array(beq)
    C = np.array([lin(0.5 * (np.outer(X0[:, j] - X0[:, j0], p) + np.outer(p, X0[:, j] - X0[:, j0])))
                  for j in inactive])          # slacks s_j(V) = C_j x >= 0, inequality (12)
    nu = _relint_normal(X0, w, r)
    V0 = np.outer(nu, nu) / (nu @ r) + np.eye(K) - np.outer(r, r) / (r @ r)
    return _pd_centres(K, Aeq, beq, C, (V0 / np.trace(V0))[np.triu_indices(K)], ranges)


def _pd_centres(K, Aeq, beq, C, x0, ranges):
    """Log-det maximiser and analytic centre of {V >= 0 : Aeq x = beq, C x > 0}, from a strictly
    feasible x0, with V = sum_a x_a B_a (and with ``ranges`` the bounds of each V_kk)."""
    B = _basis(K)
    n = len(B)
    Bm = B.reshape(n, K * K)

    def mat(x):
        return (x @ Bm).reshape(K, K)

    def lin(M):
        return Bm @ M.ravel()

    def in_domain(x):
        try:
            np.linalg.cholesky(mat(x))
        except np.linalg.LinAlgError:
            return False
        return C.size == 0 or np.all(C @ x > 0)

    def barrier(mu):
        """-log det V - mu sum_j log s_j(V), with gradient and Hessian in x."""
        def vgh(x):
            V = mat(x)
            W = np.linalg.inv(V)
            f = -2.0 * np.log(np.diag(np.linalg.cholesky(V))).sum()
            g = -Bm @ W.ravel()
            WB = W @ B                         # W B_a for every basis matrix
            H = WB.reshape(n, -1) @ WB.transpose(0, 2, 1).reshape(n, -1).T   # tr(W B_a W B_b)
            if C.size:
                s = C @ x
                f -= mu * np.log(s).sum()
                g = g - mu * (C / s[:, None]).sum(axis=0)
                H = H + mu * (C.T / s ** 2) @ C
            return f, g, H
        return vgh

    def barrier_f(mu):
        """The barrier alone, +inf outside the domain: all a trial step of the line search needs."""
        def fun(x):
            try:
                f = -2.0 * np.log(np.diag(np.linalg.cholesky(mat(x)))).sum()
            except np.linalg.LinAlgError:
                return np.inf
            if C.size:
                s = C @ x
                if not np.all(s > 0):
                    return np.inf
                f -= mu * np.log(s).sum()
            return f
        return fun

    N = _null(Aeq)
    xc = _newton_eq(x0, barrier(1.0), Aeq, beq, in_domain, fun=barrier_f(1.0), N=N)
    x, mu, m = x0.copy(), 1.0, max(len(C), 1)
    while m * mu > 1e-10:
        x = _newton_eq(x, barrier(mu), Aeq, beq, in_domain, fun=barrier_f(mu), N=N)
        mu /= 20.0
    Vc = mat(xc)
    Vl = mat(x)
    out = dict(logdet_diag=np.diag(Vl) / np.trace(Vl), center_diag=np.diag(Vc) / np.trace(Vc),
               logdet_min_eig=float(np.linalg.eigvalsh(Vl).min()), Vstar=Vl, Vcenter=Vc)
    if ranges:
        lo, hi = np.zeros(K), np.zeros(K)
        degree = K + len(C)                    # barrier parameter of log det plus the slacks
        for k in range(K):
            e = lin(np.outer(np.eye(K)[k], np.eye(K)[k]))
            for sign, store in ((1.0, lo), (-1.0, hi)):
                def vgh_lin(y, mu_box, e=e, sign=sign):
                    f0, g0, H0 = barrier(1.0)(y)
                    return sign * (e @ y) + mu_box[0] * f0, sign * e + mu_box[0] * g0, mu_box[0] * H0

                def fun_lin(y, mu_box, e=e, sign=sign):
                    return sign * (e @ y) + mu_box[0] * barrier_f(1.0)(y)
                y, mu_box = xc.copy(), [1.0]
                while degree * mu_box[0] > 1e-9:
                    y = _newton_eq(y, lambda z: vgh_lin(z, mu_box), Aeq, beq, in_domain,
                                   fun=lambda z: fun_lin(z, mu_box), N=N)
                    mu_box[0] /= 10.0
                store[k] = e @ y
        out["lo"], out["hi"] = lo, hi
    return out


def _diag_system(X1, X0, w):
    """The cone of diagonal rationalisers of w: d >= 0 with Aeq d = 0 and Sm d >= 0.

    Predictors are standardised first; residuals at rounding size are exact matches.
    """
    X1, X0 = _standardize(np.asarray(X1, float), np.asarray(X0, float))
    w = np.asarray(w, float)
    S = np.flatnonzero(w > 1e-7)
    inactive = np.setdiff1d(np.arange(X0.shape[1]), S)
    j0 = S[0]
    r = X1 - X0 @ w
    r[np.abs(r) <= 1e-10 * max(1.0, np.abs(X1).max())] = 0.0   # exact matches, as solved to rounding
    Aeq = (X0[:, S[1:]] - X0[:, [j0]]).T * r          # (X_j - X_j0)' Diag(d) r = 0
    Sm = (X0[:, [j0]] - X0[:, inactive]).T * r        # slacks Sm d >= 0
    return Aeq, Sm


def _diag_classes(Aeq, Sm):
    """The support of the rationalisers, the inequalities they can make strict, and a start strict in both.

    Some point of a polyhedron is strict in every inequality that is not an implicit
    equality (Schrijver, Theory of Linear and Integer Programming, 1986, sec. 8.1), and
    the rationalisers form a cone, so one of them reaches 1 in all of those inequalities
    at once. One LP maximises how many of y_k <= d_k and z_i <= (Sm d)_i reach 1, with y
    and z in [0, 1], so z >= 0 also keeps d in the cone: y_k = 1 marks the support and
    z_i = 1 a strict inequality. The sum
    of d is capped at (K + q) / TOL, so a coordinate or slack counts as positive when it
    can exceed TOL once d is scaled to sum one, as with one LP per row, up to a factor of
    2 (K + q) at the threshold.
    """
    q, K = Sm.shape[0], Aeq.shape[1]
    A_ub = np.vstack([np.c_[-np.eye(K), np.eye(K), np.zeros((K, q))],          # y_k <= d_k
                      np.c_[-Sm, np.zeros((q, K)), np.eye(q)],                  # 0 <= z_i <= (Sm d)_i
                      np.r_[np.ones(K), np.zeros(K + q)][None, :]])             # 1'd <= (K + q) / TOL
    b_ub = np.r_[np.zeros(K + q), (K + q) / TOL]
    res = linprog(np.r_[np.zeros(K), -np.ones(K + q)], A_ub=A_ub, b_ub=b_ub,
                  A_eq=np.c_[Aeq, np.zeros((len(Aeq), K + q))] if len(Aeq) else None,
                  b_eq=np.zeros(len(Aeq)) if len(Aeq) else None,
                  bounds=[(0, None)] * K + [(0, 1)] * (K + q), method="highs")
    support = [k for k in range(K) if res.status == 0 and res.x[K + k] > 0.5]
    if not support:
        raise ValueError("no diagonal V rationalises w (w is outside the attainable closure)")
    strict = [i for i in range(q) if res.x[2 * K + i] > 0.5]
    d = res.x[:K]
    return support, strict, d[support] / d[support].sum()


def diag_rationalizers(X1, X0, w):
    """Log-det maximiser and analytic centre of the diagonal V that rationalise w.

    Returns ``logdet`` and ``center`` (sum one), the ``support`` of V, the predictors
    ``forced_zero`` in every rationaliser, and the count of implicit equalities.
    """
    if not outside_hull(np.asarray(X1, float), np.asarray(X0, float)):
        return _diag_interior(X1, X0, w)
    Aeq, Sm = _diag_system(X1, X0, w)
    K = Aeq.shape[1]
    support, strict, x0 = _diag_classes(Aeq, Sm)
    implicit = [i for i in range(len(Sm)) if i not in strict]
    sup = np.array(support)
    Ae = np.vstack([Aeq[:, sup], np.ones((1, len(sup)))] + ([Sm[implicit][:, sup]] if implicit else []))
    be = np.concatenate([np.zeros(len(Aeq)), [1.0]] + ([np.zeros(len(implicit))] if implicit else []))
    Cs = Sm[strict][:, sup] if strict else np.zeros((0, len(sup)))
    dl, dc = _diag_centres(K, sup, Ae, be, Cs, x0)
    return dict(logdet=dl, center=dc, support=support,
                forced_zero=[k for k in range(K) if k not in support], n_implicit=len(implicit))


def _diag_centres(K, sup, Ae, be, Cs, x0):
    """Log-det maximiser and analytic centre of {d_sup > 0 : Ae d_sup = be, Cs d_sup > 0} from a
    strictly feasible x0, as length-K vectors that are zero off the support ``sup``."""
    def in_domain(x):
        return np.all(x > 0) and (Cs.size == 0 or np.all(Cs @ x > 0))

    def barrier(mu):
        """-sum log d - mu sum log s(d)."""
        def vgh(x):
            f, g, H = -np.log(x).sum(), -1.0 / x, np.diag(1.0 / x ** 2)
            if Cs.size:
                s = Cs @ x
                f -= mu * np.log(s).sum()
                g = g - mu * (Cs / s[:, None]).sum(axis=0)
                H = H + mu * (Cs.T / s ** 2) @ Cs
            return f, g, H
        return vgh

    def barrier_f(mu):
        """The barrier alone, +inf outside the domain."""
        def fun(x):
            if not in_domain(x):
                return np.inf
            return -np.log(x).sum() - (mu * np.log(Cs @ x).sum() if Cs.size else 0.0)
        return fun

    N = _null(Ae)
    xc = _newton_eq(x0, barrier(1.0), Ae, be, in_domain, fun=barrier_f(1.0), N=N)
    x, mu, m = x0.copy(), 1.0, max(len(Cs), 1)
    while m * mu > 1e-10:
        x = _newton_eq(x, barrier(mu), Ae, be, in_domain, fun=barrier_f(mu), N=N)
        mu /= 20.0
    dl, dc = np.zeros(K), np.zeros(K)
    dl[sup], dc[sup] = x, xc
    return dl, dc


def _interior_cell(X1, X0, w):
    """w's cell inside the hull and the standardised donors; ValueError unless w is a cell's weights."""
    from . import interior
    X1, X0, w = (np.asarray(a, float) for a in (X1, X0, w))
    C1, C0 = interior._standardised(X1, X0)
    A = np.flatnonzero(w > 0)
    if len(A) == X0.shape[0] + 1:
        cell = interior.Cell(A, C1, C0)
        if cell.w.min() > interior.ADMIT_TOL and np.allclose(cell.w, w[A], rtol=0.0, atol=1e-9):
            return cell, C0
    raise ValueError("inside the donor hull a rationalised w is an exact match on K + 1 donors, the "
                     "barycentric weights of X1 in their simplex (paper, sec. 5.2)")


def _diag_interior(X1, X0, w):
    """Centre and log-det maximiser of the diagonal V under which w's cell is the selection."""
    from . import interior
    cell, _ = _interior_cell(X1, X0, w)
    R = cell.R
    K = R.shape[1]
    ok, d = interior.diag_attainable(R, {})
    if not ok:
        raise ValueError("no diagonal V selects w's cell: some donor outside it lies on or below the "
                         "cell's lifted hyperplane under every positive diagonal V")
    tau = 0.5                                  # move the certificate inside the orthant, keeping R d > 0
    while (R @ ((1.0 - tau) * d + tau / K)).min() <= 0.0:
        tau /= 2.0
    dl, dc = _diag_centres(K, list(range(K)), np.ones((1, K)), np.array([1.0]), R, (1.0 - tau) * d + tau / K)
    return dict(logdet=dl, center=dc, support=list(range(K)), forced_zero=[], n_implicit=0)


def _pd_interior(X1, X0, w, ranges):
    """The PD rationalisers of w's cell, on the trace-one slice."""
    from . import interior
    cell, C0 = _interior_cell(X1, X0, w)
    Q = interior.q_matrices(cell, C0)
    K = C0.shape[0]
    ok, V = interior.pd_attainable(Q, {}, V0=interior._steiner(C0[:, list(cell.A)]))
    if not ok:
        raise ValueError("no positive definite V selects w's cell")
    if V is None:                              # pragma: no cover - cvxpy decides only at the threshold,
        raise RuntimeError("the selecting metrics of this cell are too thin to centre from")  # no start
    Bm = _basis(K).reshape(-1, K * K)
    C = np.array([Bm @ Qj.ravel() for Qj in Q]).reshape(len(Q), -1)
    return _pd_centres(K, (Bm @ np.eye(K).ravel())[None, :], np.array([1.0]), C,
                       (V / np.trace(V))[np.triu_indices(K)], ranges)
