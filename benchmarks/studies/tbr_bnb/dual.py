"""Two maximisers of the certified dual theta(y) (oracle.Oracle) (#965).

spectral_bundle : Helmberg and Kiwiel (2002), Algorithm 1, with the
                  trace-bounded model set {P V P' + alpha Wbar : V >= 0,
                  alpha >= 0, tr V + alpha <= T} (Liao, Ding and Zheng 2026,
                  eq. (44)) and the sign constraints by the Gauss-Seidel eta
                  step, eqs. (17)-(19). The model's value at a trial point is
                  its closed-form minimum over the model set. Measured on the
                  40-geo nodes it is far slower than the smoothed method: each
                  iteration solves a conic master QP over about 26,000
                  residuals, and at 79 x 79 a full eigendecomposition costs under
                  a millisecond, which removes the method's reason to exist.
smoothed_lbfgs  : the bounding procedure of BiqCrunch (Krislock, Malick and
                  Roupin), as mlsynth.utils.syndes_helpers.branch uses it:
                  L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995) on a smoothed dual,
                  the smoothing parameter decreased on a schedule. The smoothing
                  here is by Young's inequality, derived here:
                  <S, Z> >= -|S_-|_F |Z|_F >= -|S_-|_F T >= -|S_-|_F^2/(2a) - a T^2/2,
                  since |Z|_F <= tr Z <= T, so k - |S_-|_F^2/(2a) - a T^2/2 is a
                  valid lower bound. Nesterov (2007) gives the entropy smoothing
                  and its complexity; for a 79 x 79 matrix it would gain at most
                  about 3x over this one. The stage budget grows as
                  60 * 2^k up to 800, measured on the 40-geo nodes.

Both stop when the exact theta clears ``target`` and return the best y seen.

Helmberg, C. and Kiwiel, K. C. (2002). Math. Program. 93, 173-194.
Liao, F.-Y., Ding, L. and Zheng, Y. (2026). arXiv:2307.07651v2.
Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm
    for bound constrained optimization. SIAM J. Sci. Comput. 16, 1190-1208.
Nesterov, Y. (2007). Smoothing technique and its applications in semidefinite
    optimization. Math. Program. 110, 245-259.
"""
import time
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

#: The number of correction pairs L-BFGS-B stores (scipy's ``maxcor``; m in Byrd,
#: Lu, Nocedal and Zhu 1995). Measured on the four GeoLift groups (#965,
#: results/solver_settings.log): 10 gives the same nodes, prunes and branches as
#: 20 at 70-87% of the time, the work L-BFGS-B does per call falling with m; 5
#: leaves the hardest group uncertified, its multipliers converging too slowly
#: within the evaluation budget.
LBFGS_MEMORY = 10


def _svec_basis(r):
    idx = [(i, j) for i in range(r) for j in range(i, r)]
    Sel = np.zeros((len(idx), r * r))
    for t, (i, j) in enumerate(idx):
        Sel[t, i * r + j] = 1.0          # x_t = V[i, j] (row-major vec)
    return idx, Sel


def spectral_bundle(O, y0, target=None, *, iters=300, r_new=4, r_max=16, u=None, m_L=0.2,
                    tol=1e-7, gs_passes=4, log=None, time_limit=None):
    t0 = time.perf_counter()
    yh = np.array(y0, dtype=float); yh[O.ineq] = np.maximum(yh[O.ineq], 0)
    th, g, lam, Q = O.theta(yh)
    best = (th, yh.copy()); evals = 1
    e0 = O.expr(np.zeros((O.n, O.n)))
    P = Q[:, :r_new]; Wbar = None
    if u is None:
        u = max(np.linalg.norm(g) / max(abs(th), 1e-3), 1e-6)
    n = O.n
    Cobj = np.zeros((n, n)); p = O.p
    Cobj[p + 1:, p + 1:] = O.G; Cobj[0, p + 1:] = Cobj[p + 1:, 0] = -O.c
    history = []
    for it in range(iters):
        if target is not None and best[0] >= target:
            break
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            break
        r = P.shape[1]; idx, Sel = _svec_basis(r)
        cols, cobj = [], []
        for (i, j) in idx:
            E = np.outer(P[:, i], P[:, j]); E = E + E.T if i != j else E
            cols.append(O.expr(E) - e0); cobj.append(np.sum(Cobj * E))
        if Wbar is not None:
            cols.append(O.expr(Wbar) - e0); cobj.append(np.sum(Cobj * Wbar))
        B = np.array(cols).T; cobj = np.array(cobj); d = B.shape[1]
        H = B.T @ B + 1e-12 * np.eye(d)
        Lc = np.linalg.cholesky(H)
        V = cp.Variable((r, r), PSD=True); a = cp.Variable(nonneg=True) if Wbar is not None else None
        xv = Sel @ cp.vec(V, order="C")
        x = cp.hstack([xv, a]) if a is not None else xv
        lin = cp.Parameter(d)
        cons = [cp.trace(V) + (a if a is not None else 0) <= O.T]
        prob = cp.Problem(cp.Minimize(lin @ x + cp.sum_squares(Lc.T @ x) / (2 * u)), cons)
        eta = np.zeros(O.dim)
        for _ in range(gs_passes):
            base = e0 + eta
            lin.value = cobj + B.T @ yh + B.T @ base / u
            prob.solve(solver=cp.CLARABEL)
            xs = np.asarray(x.value).ravel()
            rr = e0 + B @ xs
            eta_new = np.where(O.ineq, np.maximum(0.0, -u * yh - rr), 0.0)
            if np.allclose(eta_new, eta, atol=1e-12, rtol=1e-6):  # pragma: no cover - an early exit from a
                eta = eta_new; break  # capped inner loop of the measured negative result (README)
            eta = eta_new
        rr = e0 + B @ xs
        y_new = yh + (rr + eta) / u
        y_new[O.ineq] = np.maximum(y_new[O.ineq], 0.0)
        Vs = V.value; al = float(a.value) if a is not None else 0.0
        Wst = P @ Vs @ P.T + (al * Wbar if Wbar is not None else 0)
        th_new, g_new, lam_n, Q_n = O.theta(y_new); evals += 1
        # the model's own value at y+: min over the model set, in closed form
        S_new, k_new = O.S_k(y_new)
        cand = [0.0, float(np.linalg.eigvalsh(P.T @ S_new @ P)[0])]
        if Wbar is not None: cand.append(float(np.sum(S_new * Wbar)))
        model = k_new + O.T * min(cand)
        if th_new > best[0]: best = (th_new, y_new.copy())
        pred = model - th
        history.append((it, th, th_new, pred, u, r))
        if log: log(it, th, th_new, pred, u, r)
        if pred <= tol * (abs(th) + 1.0):
            break
        if th_new - th >= m_L * pred:                       # serious step
            yh, th = y_new, th_new
            if th_new - th >= 0.8 * pred: u = max(u * 0.5, 1e-10)
        else:                                               # null step
            u = min(u * 1.5, 1e12) if th_new < th - pred else u
        # bundle update: keep the strong directions of V*, aggregate the rest, add new eigenvectors
        lv, Qv = np.linalg.eigh(Vs)
        order = np.argsort(-lv); lv, Qv = lv[order], Qv[:, order]
        keep = max(1, min(int(np.sum(lv > 1e-3 * max(lv[0], 1e-16))), r_max - r_new))
        Pk = P @ Qv[:, :keep]
        rest = P @ Qv[:, keep:] @ np.diag(lv[keep:]) @ Qv[:, keep:].T @ P.T
        agg = rest + (al * Wbar if Wbar is not None else 0)
        tr = float(np.trace(agg)) if np.ndim(agg) else 0.0
        Wbar = agg / tr if tr > 1e-14 else Wbar
        Pn, _ = np.linalg.qr(np.hstack([Pk, Q_n[:, :r_new]]))
        P = Pn
    return best[0], best[1], dict(evals=evals, iters=len(history), seconds=time.perf_counter() - t0, history=history)


def smoothed_lbfgs(O, y0, target=None, *, alpha0=1e-2, alpha_min=1e-9, shrink=0.1,
                   maxfun=400, time_limit=None, trace=None, handoff=None, ftol=None, stage_budget=None):
    t0 = time.perf_counter()
    y = np.array(y0, dtype=float); y[O.ineq] = np.maximum(y[O.ineq], 0)
    bounds = [(0.0, None) if b else (None, None) for b in O.ineq]
    best = [-np.inf, y.copy()]; evals = [0]; stopped = [None]

    class Done(Exception):
        pass

    def fun(yv, alpha):
        Sm, k = O.S_k(yv)
        lam, Q = np.linalg.eigh(Sm); evals[0] += 1
        exact = k + O.T * min(0.0, lam[0])
        if exact > best[0]: best[0], best[1] = exact, yv.copy()
        if trace is not None: trace.append(best[0])
        if target is not None and best[0] >= target: raise Done
        if handoff is not None and target is not None and handoff(evals[0], best[0], target):
            stopped[0] = "handoff"; raise Done
        if time_limit is not None and time.perf_counter() - t0 > time_limit: raise Done
        neg = np.minimum(lam, 0.0)
        Sneg = (Q * (-neg)) @ Q.T                       # S_- >= 0
        val = k - float(neg @ neg) / (2 * alpha) - alpha * O.T ** 2 / 2
        grad = O.expr(Sneg / alpha)
        return -val, -grad

    alpha = alpha0; stage = 0
    try:
        while alpha >= alpha_min:
            mf = stage_budget(stage) if stage_budget is not None else maxfun
            opts = dict(maxfun=mf, maxcor=LBFGS_MEMORY)
            if ftol is not None: opts["ftol"] = ftol
            res = minimize(fun, y, args=(alpha,), jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
            y = res.x; alpha *= shrink; stage += 1
    except Done:
        pass
    return best[0], best[1], dict(evals=evals[0], seconds=time.perf_counter() - t0, stopped=stopped[0])
