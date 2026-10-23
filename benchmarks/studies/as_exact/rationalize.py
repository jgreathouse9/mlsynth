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
the normal cone (one LP), so no phase-I problem is solved. Diagonal class: support and
implicit equalities are found by LPs, as in the reference, and the average of those LP
solutions is a relative-interior start.

Predictors are standardised across treated and donors (sd, ddof=1), as in the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


def _standardize(X1, X0):
    sd = np.column_stack([X1, X0]).std(axis=1, ddof=1)
    return X1 / sd, X0 / sd[:, None]


def _newton_eq(x, vgh, A, b, in_domain, tol=1e-12, maxit=200):
    """Newton's method on {x : A x = b} from a strictly feasible x, in the null space of A."""
    if not in_domain(x):
        raise ValueError("Newton's method needs a strictly feasible start")
    _, sv, Vt = np.linalg.svd(A)
    rank = int((sv > 1e-12 * max(sv[0], 1.0)).sum())
    N = Vt[rank:].T
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
        while not in_domain(x + step * dx) or vgh(x + step * dx)[0] > f - 0.25 * step * lam2:
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
    X1, X0 = _standardize(np.asarray(X1, float), np.asarray(X0, float))
    w = np.asarray(w, float)
    K, J = X0.shape
    S = np.flatnonzero(w > 1e-7)
    inactive = np.setdiff1d(np.arange(J), S)
    j0 = S[0]
    r = X1 - X0 @ w
    p = X0 @ w - X1                            # the paper's (X0 w* - X1)
    B = _basis(K)

    def lin(M):                                # <M, V(x)> as a row acting on x
        return np.einsum("akl,kl->a", B, M)

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
    x0 = (V0 / np.trace(V0))[np.triu_indices(K)]

    def in_domain(x):
        try:
            np.linalg.cholesky(np.tensordot(x, B, axes=1))
        except np.linalg.LinAlgError:
            return False
        return C.size == 0 or np.all(C @ x > 0)

    def barrier(mu):
        """-log det V - mu sum_j log s_j(V), with gradient and Hessian in x."""
        def vgh(x):
            V = np.tensordot(x, B, axes=1)
            W = np.linalg.inv(V)
            f = -2.0 * np.log(np.diag(np.linalg.cholesky(V))).sum()
            g = -np.einsum("akl,kl->a", B, W)
            WB = np.einsum("kl,alm->akm", W, B)
            H = np.einsum("akm,bmk->ab", WB, WB)
            if C.size:
                s = C @ x
                f -= mu * np.log(s).sum()
                g = g - mu * (C / s[:, None]).sum(axis=0)
                H = H + mu * (C.T / s ** 2) @ C
            return f, g, H
        return vgh

    xc = _newton_eq(x0, barrier(1.0), Aeq, beq, in_domain)
    x, mu, m = x0.copy(), 1.0, max(len(inactive), 1)
    while m * mu > 1e-10:
        x = _newton_eq(x, barrier(mu), Aeq, beq, in_domain)
        mu /= 20.0
    Vc = np.tensordot(xc, B, axes=1)
    Vl = np.tensordot(x, B, axes=1)
    out = dict(logdet_diag=np.diag(Vl) / np.trace(Vl), center_diag=np.diag(Vc) / np.trace(Vc),
               logdet_min_eig=float(np.linalg.eigvalsh(Vl).min()), Vstar=Vl, Vcenter=Vc)
    if ranges:
        lo, hi = np.zeros(K), np.zeros(K)
        degree = K + len(inactive)             # barrier parameter of log det plus the slacks
        for k in range(K):
            e = lin(np.outer(np.eye(K)[k], np.eye(K)[k]))
            for sign, store in ((1.0, lo), (-1.0, hi)):
                def vgh_lin(y, mu_box, e=e, sign=sign):
                    f0, g0, H0 = barrier(1.0)(y)
                    return sign * (e @ y) + mu_box[0] * f0, sign * e + mu_box[0] * g0, mu_box[0] * H0
                y, mu_box = xc.copy(), [1.0]
                while degree * mu_box[0] > 1e-9:
                    y = _newton_eq(y, lambda z: vgh_lin(z, mu_box), Aeq, beq, in_domain)
                    mu_box[0] /= 10.0
                store[k] = e @ y
        out["lo"], out["hi"] = lo, hi
    return out


def diag_rationalizers(X1, X0, w):
    """Log-det maximiser and analytic centre of the diagonal V that rationalise w.

    Returns ``logdet`` and ``center`` (sum one), the ``support`` of V, the predictors
    ``forced_zero`` in every rationaliser, and the count of implicit equalities.
    """
    X1, X0 = _standardize(np.asarray(X1, float), np.asarray(X0, float))
    w = np.asarray(w, float)
    K, J = X0.shape
    S = np.flatnonzero(w > 1e-7)
    inactive = np.setdiff1d(np.arange(J), S)
    j0 = S[0]
    r = X1 - X0 @ w
    r[np.abs(r) <= 1e-10 * max(1.0, np.abs(X1).max())] = 0.0   # exact matches, as solved to rounding
    Aeq = (X0[:, S[1:]] - X0[:, [j0]]).T * r          # (X_j - X_j0)' Diag(d) r = 0
    Sm = (X0[:, [j0]] - X0[:, inactive]).T * r        # slacks Sm d >= 0
    A_eq = np.vstack([Aeq, np.ones((1, K))])
    b_eq = np.concatenate([np.zeros(len(Aeq)), [1.0]])

    def lpmax(c):
        res = linprog(-c, A_ub=-Sm if len(Sm) else None, b_ub=np.zeros(len(Sm)) if len(Sm) else None,
                      A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if res.status != 0:
            raise ValueError("no diagonal V rationalises w (w is outside the attainable closure)")
        return -res.fun, res.x

    pts, support, strict = [], [], []
    for k in range(K):
        v, x = lpmax(np.eye(K)[k])
        pts.append(x)
        if v > TOL:
            support.append(k)
    for i in range(len(Sm)):
        v, x = lpmax(Sm[i])
        pts.append(x)
        if v > TOL:
            strict.append(i)
    implicit = [i for i in range(len(Sm)) if i not in strict]
    sup = np.array(support)
    Ae = np.vstack([A_eq[:, sup]] + ([Sm[implicit][:, sup]] if implicit else []))
    be = np.concatenate([b_eq] + ([np.zeros(len(implicit))] if implicit else []))
    Cs = Sm[strict][:, sup] if strict else np.zeros((0, len(sup)))
    x0 = np.mean(pts, axis=0)[sup]          # a mean of feasible points: relative interior, no projection

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

    xc = _newton_eq(x0, barrier(1.0), Ae, be, in_domain)
    x, mu, m = x0.copy(), 1.0, max(len(strict), 1)
    while m * mu > 1e-10:
        x = _newton_eq(x, barrier(mu), Ae, be, in_domain)
        mu /= 20.0
    dl, dc = np.zeros(K), np.zeros(K)
    dl[sup], dc[sup] = x, xc
    return dict(logdet=dl, center=dc, support=support,
                forced_zero=[k for k in range(K) if k not in support], n_implicit=len(implicit))
