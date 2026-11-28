"""Support-restricted primal-dual interior point with pair pricing on the pair-product (first-level RLT) relaxation, after
Zanetti and Gondzio (arXiv:2206.11009v2, sections 3.2-3.4): each iteration is one interior-point step on the columns
in the support, then the reduced costs of every pair outside it are computed and the most negative enter (initialised
at v = s = sqrt(mu)), and pair columns whose value falls below a threshold near convergence leave. Prototype for the
property, not for speed: dense normal equations, numpy (no library change).

LP (rlt_scale.py's build, so the two are the same LP): rows lhs <= A v <= rhs; equality rows kept, one-sided rows get a
surplus column. Group columns x and the surpluses are always in the support (x <= 1 is implied by the cover rows);
only the pair columns w are priced. Standard form min c'v, A v = b, v >= 0; Mehrotra predictor-corrector, step 0.995
of the way to the boundary, normal equations A Theta A' dy = r with a 1e-10 relative diagonal shift, dense Cholesky.
Each iteration also reports the weak-duality bound B(y) of rlt_scale.py over the full LP (every pair), so the bound
is checked at every iterate, not only at the end. Reference: the full LP's value from HiGHS.
python -u sparse_ipm.py <worktree> <n> <reference_value> [init_factor] [max_iter] [enter_fraction_of_rows]"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
wt, n_s, REF = sys.argv[1], sys.argv[2], float(sys.argv[3])
INIT = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
MAXIT = int(sys.argv[5]) if len(sys.argv) > 5 else 200
ENTER = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
sys.argv = ["rlt_scale.py", wt, n_s, "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read()
src = src.split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.linalg as la, scipy.sparse as sp
A, lhs, rhs, cost, C, P, z, L, say = g["A"].tocsc(), g["lhs"], g["rhs"], g["cost"], g["C"], g["P"], g["z"], g["L"], g["say"]
rc0 = g["rc"]; PA, PB = g["PA"], g["PB"]
m = A.shape[0]
eq = lhs == rhs; ge = ~eq
nge = int(ge.sum())
# standard form: columns [x (C) | w (P) | surplus (nge)]
S_ = sp.csc_matrix((-np.ones(nge), (np.flatnonzero(ge), np.arange(nge))), shape=(m, nge))
Af = sp.hstack([A, S_]).tocsc()
b = np.where(eq, rhs, lhs)
cf = np.r_[cost, np.zeros(nge)]
NV = C + P + nge
always = np.r_[np.arange(C), C + P + np.arange(nge)]
is_pair = np.zeros(NV, bool); is_pair[C:C + P] = True
one = ~eq


def bound(y):
    yy = np.where(one, np.maximum(y, 0.0), y)
    r = cost - A.T @ yy
    return float(np.sum(np.where(yy > 0, yy * lhs, yy * np.where(np.isinf(rhs), 0.0, rhs))) + np.sum(np.minimum(r, 0.0)))


# initial support: the pairs with the smallest rc_G + rc_H, INIT * m of them
k0 = min(P, int(INIT * m))
first = C + np.argsort(rc0[PA] + rc0[PB], kind="stable")[:k0]
supp = np.zeros(NV, bool); supp[always] = True; supp[first] = True
v = np.ones(NV); s = np.ones(NV); y = np.zeros(m)
v[~supp] = 0.0; s[~supp] = 0.0
t0 = time.perf_counter(); best = -np.inf; peak = 0
for it in range(MAXIT):
    J = np.flatnonzero(supp)
    peak = max(peak, int(supp[C:C + P].sum()))
    Aj = Af[:, J]; vj = v[J]; sj = s[J]; cj = cf[J]
    rp = b - Aj @ vj
    rd = cj - Aj.T @ y - sj
    mu = float(vj @ sj) / J.size
    # full pricing and the bound at the current duals
    red = cf - Af.T @ y
    B = bound(y); best = max(best, B)
    out_neg = np.flatnonzero((~supp) & is_pair & (red < -1e-9 * max(1.0, abs(REF))))
    rel_p = np.linalg.norm(rp) / (1 + np.linalg.norm(b)); rel_d = np.linalg.norm(rd) / (1 + np.linalg.norm(cj))
    obj = float(cj @ vj)
    if it < 5 or it % 5 == 0:
        say(f"it {it:3d}: support {int(supp[C:C + P].sum()):,} of {P:,} pairs, mu {mu:.2e}, primal {rel_p:.1e}, dual "
            f"{rel_d:.1e}, objective {obj:.6f}, bound {B:.6f} (best {best:.6f}, ref {REF:.6f}), {out_neg.size} pairs "
            f"price out")
    if max(rel_p, rel_d, mu / (1 + abs(obj))) < 1e-8 and out_neg.size == 0:
        say(f"converged at it {it}: objective {obj:.8f}, bound {B:.8f}, ref {REF:.8f}, support "
            f"{int(supp[C:C + P].sum()):,} of {P:,} pairs (peak {peak:,}), {time.perf_counter() - t0:.1f} s")
        break
    theta = vj / sj
    M = (Aj.multiply(theta) @ Aj.T).toarray()
    M[np.diag_indices(m)] += 1e-10 * max(1.0, float(np.max(np.diag(M))))
    F = la.cho_factor(M, lower=True, check_finite=False)

    def direction(rc):
        rhs_ = rp + Aj @ (theta * rd) - Aj @ (rc / sj)
        dy = la.cho_solve(F, rhs_, check_finite=False)
        dv = theta * (Aj.T @ dy - rd) + rc / sj
        ds = rd - Aj.T @ dy
        return dv, dy, ds

    def step(x, dx):
        neg = dx < 0
        return min(1.0, float(np.min(-x[neg] / dx[neg]))) if neg.any() else 1.0

    dv, dy, ds = direction(-vj * sj)
    ap, ad = step(vj, dv), step(sj, ds)
    mu_aff = float((vj + ap * dv) @ (sj + ad * ds)) / J.size
    sigma = (mu_aff / mu) ** 3
    dv, dy, ds = direction(-vj * sj - dv * ds + sigma * mu)
    ap, ad = 0.995 * step(vj, dv), 0.995 * step(sj, ds)
    v[J] = vj + ap * dv; s[J] = sj + ad * ds; y = y + ad * dy
    # support update: pairs leave when tiny near convergence; the most negative priced pairs enter
    mu = float(v[J] @ s[J]) / J.size
    if mu < 1e-6:
        leave = J[is_pair[J] & (v[J] < 1e-3 * np.sqrt(mu)) & (s[J] > np.sqrt(mu))]
        supp[leave] = False; v[leave] = 0.0; s[leave] = 0.0
    red = cf - Af.T @ y
    cand = np.flatnonzero((~supp) & is_pair & (red < 0))
    if cand.size:
        enter = cand[np.argsort(red[cand], kind="stable")[:max(1, int(ENTER * m))]]
        supp[enter] = True; v[enter] = np.sqrt(mu); s[enter] = np.sqrt(mu)
else:
    say(f"stopped at {MAXIT} iterations: objective {obj:.8f}, best bound {best:.8f}, ref {REF:.8f}")
