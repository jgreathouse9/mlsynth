"""Support-restricted primal-dual interior point with pair pricing on the pair-product (first-level RLT) relaxation,
second version: the support rules of Zanetti and Gondzio's code (github.com/INFORMSJoC/2022.0184, MIT licence;
updateindexred_full.m and OT_IPM.m), and a log laid out after MOSEK's optimizer log (MOSEK Optimization Toolbox for
MATLAB 11.2.9, sections 8.1 and 13.3.4): input summary, setup, one line per iteration, solution summary.

Changes from sparse_ipm.py, each from the 24-market run (support collapse at mu ~ 1e-21 with the primal residual stuck
at 5e-5, after pairs needed for feasibility were dropped):
  leave    a pair leaves when its value is below 1e-6 (their threshold), at most m per iteration, smallest first, and
           only while the restricted primal residual is below LEAVE_PFEAS (no removals while primal infeasible);
  enter    pairs with reduced cost below -ENTER_TOL enter, at most ENTER * m, most negative first; the reduced costs
           are computed for every pair on every third iteration and otherwise only for the cheap pairs (the CHEAP
           share with the smallest rc_G + rc_H at the root), as their code does with c < cmax; once the restricted
           problem has converged, any pair with a negative reduced cost (below -1e-9) enters;
  log      written through logging to a file as each iteration ends, readable while the run goes.
Everything else as sparse_ipm.py: Mehrotra predictor-corrector, step 0.995 to the boundary, dense normal equations
with a 1e-10 relative shift, the weak-duality bound B(y) over the full LP at every iterate.
Columns of the iteration log (infinity norms on the restricted problem, as MOSEK's are on the whole one):
  ITE iteration; PFEAS |b - A_J v_J|; DFEAS |c_J - A_J'y - s_J|; GFEAS |c_J'v_J - b'y|; POBJ c_J'v_J; DOBJ b'y; MU
  v_J's_J / |J|; AP, AD the primal and dual step lengths; SUPP pairs in the support; IN, OUT pairs entering and
  leaving after the step; PR F (every pair priced) or C (cheap pairs only); BOUND B(y); TIME seconds.
RCA ablation copy of sparse_ipm2_dbg.py. Two switches, from the 24-market stall RCA (all else unchanged):
  SHIFT=maxdiag  the prototype's diagonal shift, 1e-10 * max diag(A Theta A') (grows with theta_max);
  SHIFT=diagrel  shift each diagonal entry by 1e-12 of itself (the normal matrix is singular: 648 of 8,867 rows of
                 A_J are dependent at the 24-market stall, so some shift is needed);
  GUARD=0        Mehrotra's sigma as before;
  FR=<npz>       facially reduced LP from rca_facial.py: the columns zero in every feasible solution are
                 never in the support and the dependent rows are dropped (the bound maps y back to every row);
  LEAVE=off      no pair leaves the support; LEAVE_CAP=f caps removals at f * m per iteration (default 1);
  GUARD=1        the centring target is held at mu0 * |rp|_inf / (BETA |rp0|_inf) or above while that exceeds sigma mu,
                 so complementarity falls no faster than the primal residual (BETA = 10).
Also, as before: env DUMP_AT="50,55,62,65" and DUMP_PREFIX
save the iterate (support J, v, s, y) before each listed iteration's step, and every iteration logs, at DEBUG to
the same file, the linear-system residuals of both directions: |A_J dv - rp|_inf / |rp|_inf (primal), |A_J'dy + ds -
rd|_inf / max(|rd|_inf, 1e-300) (dual), the smallest and largest theta, and the diagonal shift used.
python -u sparse_ipm2_dbg.py <worktree> <n> <reference_value> <log_path> [init] [enter] [enter_tol] [cheap] [max_iter]"""
import os, sys, time, logging
for v_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v_] = "1"
wt, n_s, REF, LOGP = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
arg = lambda i, d: float(sys.argv[i]) if len(sys.argv) > i else d
INIT, ENTER, ENTER_TOL, CHEAP, MAXIT = arg(5, 0.5), arg(6, 1.0), arg(7, 1e-3), arg(8, 0.25), int(arg(9, 200))
LEAVE_V, LEAVE_PFEAS, TOL = 1e-6, 1e-6, 1e-8

log = logging.getLogger("sparse_ipm_rca"); log.setLevel(logging.DEBUG); log.propagate = False
fh = logging.FileHandler(LOGP, mode="w"); fh.setFormatter(logging.Formatter("%(message)s")); log.addHandler(fh)
info = log.info

sys.argv = ["rlt_scale.py", wt, n_s, "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read()
src = src.split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.linalg as la, scipy.sparse as sp
A, lhs, rhs, cost, C, P, z, L = g["A"].tocsc(), g["lhs"], g["rhs"], g["cost"], g["C"], g["P"], g["z"], g["L"]
rc0 = g["rc"]; PA, PB = g["PA"], g["PB"]
m = A.shape[0]
eq = lhs == rhs; ge = ~eq; nge = int(ge.sum())
S_ = sp.csc_matrix((-np.ones(nge), (np.flatnonzero(ge), np.arange(nge))), shape=(m, nge))
Af = sp.hstack([A, S_]).tocsc()
b = np.where(eq, rhs, lhs)
cf = np.r_[cost, np.zeros(nge)]
NV = C + P + nge
always = np.r_[np.arange(C), C + P + np.arange(nge)]
is_pair = np.zeros(NV, bool); is_pair[C:C + P] = True
one = ~eq
M_FULL = m; ROWS = np.arange(m)
if os.environ.get("FR"):
    fr = np.load(os.environ["FR"], allow_pickle=True); Zm = np.zeros(NV, bool); Zm[fr["Z"]] = True; ROWS = fr["rows"]
    FR_Y = [np.asarray(v_, float) for v_ in fr["Y"]] if "Y" in fr.files else []; FR_Z = [np.asarray(v_, np.int64) for v_ in fr["Zs"]] if "Zs" in fr.files else []
    always = always[~Zm[always]]; is_pair &= ~Zm
    Af = Af.tocsr()[ROWS].tocsc(); b = b[ROWS]; m = ROWS.size
pair_score = rc0[PA] + rc0[PB]
cheap = np.zeros(NV, bool)
cheap[C + np.argsort(pair_score, kind="stable")[:max(1, int(CHEAP * P))]] = True


def nz_range(x):
    x = np.abs(np.asarray(x, float)); x = x[x > 0]
    return f"nnz: {x.size:<10,} min={x.min():.2e}  max={x.max():.2e}" if x.size else "nnz: 0"


def _bound_full(y):
    yy = np.where(one, np.maximum(y, 0.0), y)
    r = cost - A.T @ yy
    return float(np.sum(np.where(yy > 0, yy * lhs, yy * np.where(np.isinf(rhs), 0.0, rhs))) + np.sum(np.minimum(r, 0.0)))


def bound(y):
    if y.size != M_FULL:
        yf = np.zeros(M_FULL); yf[ROWS] = y; y = yf
        # facial-reduction certificates: y - sum_k lambda_k y_k, last round first. B(y - lambda y_k) is concave in
        # lambda, so each lambda_k maximises it over a log grid (and 0), instead of forcing every reduced cost on Z_k
        # to be nonnegative: with an inexact certificate (b'y_k != 0, z_k slightly negative) that lambda can be huge
        # and lambda * b'y_k then swamps the bound (30 markets, seed 0: -2.1e4 against 4.38).
        if FR_Y:
            for yk, Zk in reversed(list(zip(FR_Y, FR_Z))):
                lams = np.r_[0.0, np.logspace(-6, 12, 145)]
                vals = [_bound_full(y - lam * yk) for lam in lams]
                y = y - lams[int(np.argmax(vals))] * yk
    yy = np.where(one, np.maximum(y, 0.0), y)
    r = cost - A.T @ yy
    return float(np.sum(np.where(yy > 0, yy * lhs, yy * np.where(np.isinf(rhs), 0.0, rhs))) + np.sum(np.minimum(r, 0.0)))


info("Problem")
info(f"  Name                  : pair-product relaxation, seasonal panel seed {g['PANEL_SEED']}, {g['n']} markets, Q = 2")
info(f"  Objective sense       : minimize")
info(f"  Constraints           : {m:,} ({int(eq.sum()):,} equality, {nge:,} >=)")
info(f"  Scalar variables      : {NV:,} (groups {C:,}, pairs {P:,}, surplus {nge:,})")
info(f"  |c|                   {nz_range(cost)}")
info(f"  |A|                   {nz_range(Af.data)}")
info(f"  |b|                   {nz_range(b)}")
info(f"  Group LP bound L      : {L:.10e}; optimum z {z:.10e}; reference LP value {REF:.10e}")
info("Optimizer")
info(f"  Normal equations      : dense, dimension {m:,}, factor nonzeros {m * (m + 1) // 2:,}")
info(f"  Initial support       : {min(P, int(INIT * m)):,} pairs (smallest rc_G + rc_H), cheap set {int(cheap.sum()):,}")
info(f"  Support rules         : enter rc < -{ENTER_TOL:g} (cap {max(1, int(ENTER * m)):,}/it), leave v < {LEAVE_V:g} "
     f"while PFEAS rel < {LEAVE_PFEAS:g} (cap {m:,}/it), full pricing every 3rd iteration")
hdr = (f"{'ITE':>4} {'PFEAS':>8} {'DFEAS':>8} {'GFEAS':>8} {'POBJ':>16} {'DOBJ':>16} {'MU':>8} {'AP':>5} {'AD':>5} "
       f"{'SUPP':>7} {'IN':>6} {'OUT':>6} {'PR':>2} {'BOUND':>16} {'TIME':>7}")
info(hdr)

k0 = min(P, int(INIT * m))
supp = np.zeros(NV, bool); supp[always] = True
_ord = C + np.argsort(pair_score, kind="stable"); _ord = _ord[is_pair[_ord]]; supp[_ord[:k0]] = True
v = np.ones(NV); s = np.ones(NV); y = np.zeros(m)
v[~supp] = 0.0; s[~supp] = 0.0
t0 = time.perf_counter(); best = -np.inf; peak = 0; MU0 = RP0 = None; status = "UNKNOWN (iteration limit)"; ap = ad = 0.0
n_in = n_out = 0; pr = "-"
for it in range(MAXIT + 1):
    J = np.flatnonzero(supp)
    Aj = Af[:, J]; vj = v[J]; sj = s[J]; cj = cf[J]
    rp = b - Aj @ vj; rd = cj - Aj.T @ y - sj
    mu = float(vj @ sj) / J.size
    if MU0 is None:
        MU0, RP0 = mu, float(np.max(np.abs(rp)))
    pobj = float(cj @ vj); dobj = float(b @ y)
    B = bound(y); best = max(best, B)
    pf, df, gf = float(np.max(np.abs(rp))), float(np.max(np.abs(rd))), abs(pobj - dobj)
    nsup = int(supp[C:C + P].sum()); peak = max(peak, nsup)
    info(f"{it:>4} {pf:8.1e} {df:8.1e} {gf:8.1e} {pobj:16.9e} {dobj:16.9e} {mu:8.1e} {ap:5.3f} {ad:5.3f} "
         f"{nsup:>7,} {n_in:>6,} {n_out:>6,} {pr:>2} {B:16.9e} {time.perf_counter() - t0:7.1f}")
    rel_p = np.linalg.norm(rp) / (1 + np.linalg.norm(b)); rel_d = np.linalg.norm(rd) / (1 + np.linalg.norm(cj))
    converged_r = max(rel_p, rel_d, mu / (1 + abs(pobj))) < TOL
    if converged_r:
        red = cf - Af.T @ y
        neg = np.flatnonzero((~supp) & is_pair & (red < -1e-9))
        if neg.size == 0:
            status = "OPTIMAL (restricted problem converged, no pair prices out)"
            break
    if it == MAXIT:
        break
    theta = vj / sj
    if str(it) in os.environ.get("DUMP_AT", "").split(","):
        np.savez(f"{os.environ['DUMP_PREFIX']}_{it}.npz", J=J, v=vj, s=sj, y=y, rp=rp, rd=rd, mu=mu)
    M = (Aj.multiply(theta) @ Aj.T).toarray()
    if os.environ.get("SHIFT", "maxdiag") == "diagrel":
        shift = 1e-12 * float(np.max(np.diag(M)))
        M[np.diag_indices(m)] *= (1 + 1e-12)
        M[np.diag_indices(m)] += 1e-14
    else:
        shift = 1e-10 * max(1.0, float(np.max(np.diag(M))))
        M[np.diag_indices(m)] += shift
    F = la.cho_factor(M, lower=True, check_finite=False)

    def direction(rcomp):
        rhs_ = rp + Aj @ (theta * rd) - Aj @ (rcomp / sj)
        dy = la.cho_solve(F, rhs_, check_finite=False)
        return theta * (Aj.T @ dy - rd) + rcomp / sj, dy, rd - Aj.T @ dy

    def step(x, dx):
        neg = dx < 0
        return min(1.0, float(np.min(-x[neg] / dx[neg]))) if neg.any() else 1.0

    dv, dy, ds = direction(-vj * sj)
    a1, a2 = step(vj, dv), step(sj, ds)
    sigma = (float((vj + a1 * dv) @ (sj + a2 * ds)) / J.size / mu) ** 3
    target = sigma * mu
    if os.environ.get("GUARD", "0") == "1":
        floor = MU0 * float(np.max(np.abs(rp))) / (10.0 * RP0)
        if floor > target:
            target = min(floor, mu)
    sigma = target / mu
    dv, dy, ds = direction(-vj * sj - dv * ds + target)
    lin_p = float(np.max(np.abs(Aj @ dv - rp))) / max(float(np.max(np.abs(rp))), 1e-300)
    lin_d = float(np.max(np.abs(Aj.T @ dy + ds - rd))) / max(float(np.max(np.abs(rd))), 1e-300)
    log.debug(f"     it {it}: linear residual primal {lin_p:.2e} dual {lin_d:.2e}; theta min {theta.min():.1e} max "
              f"{theta.max():.1e}; shift {shift:.1e}; sigma {sigma:.1e}")
    ap, ad = 0.995 * step(vj, dv), 0.995 * step(sj, ds)
    v[J] = vj + ap * dv; s[J] = sj + ad * ds; y = y + ad * dy
    mu = float(v[J] @ s[J]) / J.size
    # leave: tiny pairs, only while the restricted problem is primal feasible
    rp_new = b - Af[:, J] @ v[J]
    n_out = 0
    if np.linalg.norm(rp_new) / (1 + np.linalg.norm(b)) < LEAVE_PFEAS:
        cand = J[is_pair[J] & (v[J] < LEAVE_V)] if os.environ.get("LEAVE", "on") != "off" else J[:0]
        if cand.size:
            go = cand[np.argsort(v[cand], kind="stable")[:max(1, int(float(os.environ.get("LEAVE_CAP", "1")) * m))]]
            log.debug(f"     it {it}: leave {go.size:,} pairs, removed mass |A_go v_go|_inf "
                      f"{float(np.max(np.abs(Af[:, go] @ v[go]))):.2e}")
            supp[go] = False; v[go] = 0.0; s[go] = 0.0; n_out = int(go.size)
    # enter: full pricing every third iteration, else the cheap pairs; any negative once the restricted LP converged
    pr = "F" if (it + 1) % 3 == 0 or converged_r else "C"
    red = cf - Af.T @ y
    pool = (~supp) & is_pair & (cheap if pr == "C" else True)
    thr = -1e-9 if converged_r else -ENTER_TOL
    cand = np.flatnonzero(pool & (red < thr))
    n_in = 0
    if cand.size:
        go = cand[np.argsort(red[cand], kind="stable")[:max(1, int(ENTER * m))]]
        supp[go] = True; v[go] = np.sqrt(mu); s[go] = np.sqrt(mu); n_in = int(go.size)
J = np.flatnonzero(supp); vj = v[J]
viol_con = float(np.max(np.abs(b - Af[:, J] @ vj)))
info("")
info(f"Problem status  : {'PRIMAL_AND_DUAL_FEASIBLE' if status.startswith('OPTIMAL') else 'UNKNOWN'}")
info(f"Solution status : {status}")
info(f"Primal.  obj: {float(cf[J] @ vj):.10e}   nrm: {float(np.max(np.abs(vj))):.0e}   Viol.  con: {viol_con:.0e}   "
     f"var: {float(max(0.0, -np.min(vj))):.0e}")
info(f"Dual.    obj: {float(b @ y):.10e}   nrm: {float(np.max(np.abs(y))):.0e}")
info(f"Bound B(y)  : {B:.10e} (best over the run {best:.10e}); reference {REF:.10e}; best is "
     f"{(REF - best) / REF:.2e} below it")
info(f"Support     : {int(supp[C:C + P].sum()):,} pairs at the end, peak {peak:,} of {P:,} ({peak / P:.1%}); rows {m:,}")
info(f"Time        : {time.perf_counter() - t0:.1f} s, {it} iterations; SHIFT={os.environ.get('SHIFT', 'maxdiag')} GUARD={os.environ.get('GUARD', '0')}")
print(open(LOGP).read().splitlines()[-7:] and "\n".join(open(LOGP).read().splitlines()[-7:]))
