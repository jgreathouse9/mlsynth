"""RCA link: the Newton solve. From dumped iterates of sparse_ipm2_dbg.py, recompute the affine-scaling direction
(complementarity target -v s) with each solve variant and report the primal and dual linear residuals,
|A_J dv - rp|_inf / |rp|_inf and |A_J'dy + ds - rd|_inf / |rd|_inf:
  V0  dense normal equations, diagonal shift 1e-10 * max diag (the prototype's solve);
  V1  dense normal equations, no shift (LDL' through scipy.linalg.solve, assume_a='sym');
  V2  V0 plus 5 steps of iterative refinement on M dy = r;
  V3  diagonal-relative shift 1e-12 * diag(M)_i;
  V4  the augmented system [[-Theta^-1, A_J'], [A_J, 0]] solved by sparse LU (SuperLU), no normal equations;
  V5  V4 plus 2 steps of iterative refinement on the augmented system.
V4 and V5 are the reference: they avoid forming A Theta A' (whose conditioning is the square of the augmented one's
in this regime, Andersen, Gondzio, Meszaros and Xu, in Terlaky (ed.) 1996, section 6.4.2).
python -u rca24_direction.py <worktree> <dump.npz> [<dump.npz> ...]"""
import os, sys, time
for v_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v_] = "1"
wt, dumps = sys.argv[1], sys.argv[2:]
sys.argv = ["rlt_scale.py", wt, "24", "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read().split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.sparse as sp, scipy.linalg as la, scipy.sparse.linalg as spla
A, lhs, rhs, cost, C, P = g["A"].tocsc(), g["lhs"], g["rhs"], g["cost"], g["C"], g["P"]
m = A.shape[0]; eq = lhs == rhs; ge = ~eq; nge = int(ge.sum())
S_ = sp.csc_matrix((-np.ones(nge), (np.flatnonzero(ge), np.arange(nge))), shape=(m, nge))
Af = sp.hstack([A, S_]).tocsc()
for dump in dumps:
    d = np.load(dump); J, v, s, y, rp, rd = d["J"], d["v"], d["s"], d["y"], d["rp"], d["rd"]
    Aj = Af[:, J].tocsc(); th = v / s; rc = -v * s
    r_normal = rp + Aj @ (th * rd) - Aj @ (rc / s)
    Md = (Aj.multiply(th) @ Aj.T).toarray()

    def finish(dy):
        dv = th * (Aj.T @ dy - rd) + rc / s
        ds = rd - Aj.T @ dy
        return (float(np.max(np.abs(Aj @ dv - rp))) / float(np.max(np.abs(rp))),
                float(np.max(np.abs(Aj.T @ dy + ds - rd))) / max(float(np.max(np.abs(rd))), 1e-300))

    out = {}
    M0 = Md.copy(); M0[np.diag_indices(m)] += 1e-10 * max(1.0, float(np.max(np.diag(Md))))
    F0 = la.cho_factor(M0, lower=True, check_finite=False)
    dy = la.cho_solve(F0, r_normal, check_finite=False); out["V0 shift 1e-10 max"] = finish(dy)
    for _ in range(5):
        dy = dy + la.cho_solve(F0, r_normal - Md @ dy, check_finite=False)
    out["V2 V0 + 5 refinement"] = finish(dy)
    try:
        out["V1 no shift (LDL')"] = finish(la.solve(Md, r_normal, assume_a="sym", check_finite=False))
    except Exception as e:
        out["V1 no shift (LDL')"] = (float("nan"), float("nan")); print("  V1 failed:", type(e).__name__, e)
    M3 = Md.copy(); M3[np.diag_indices(m)] *= (1 + 1e-12)
    try:
        F3 = la.cho_factor(M3, lower=True, check_finite=False)
        out["V3 shift 1e-12 diag_i"] = finish(la.cho_solve(F3, r_normal, check_finite=False))
    except Exception as e:
        out["V3 shift 1e-12 diag_i"] = (float("nan"), float("nan")); print("  V3 failed:", type(e).__name__)
    # augmented: [-Theta^-1 dv_part ... ] solve for (dv, dy) with  -Theta^-1 dv + A'dy = rd - rc/v ... derived from
    # S dv + V ds = rc, A'dy + ds = rd  =>  ds = rd - A'dy, S dv + V (rd - A'dy) = rc  =>  -Theta^-1 dv + A'dy = rd - rc/v
    tk = float("nan"); fnz = 0
    for delta in (0.0, 1e-12, 1e-9):
        K = sp.bmat([[sp.diags(-1.0 / th), Aj.T], [Aj, (-delta * sp.identity(m)) if delta else None]], format="csc")
        rhsK = np.r_[rd - rc / v, rp]
        try:
            t = time.perf_counter(); lu = spla.splu(K, permc_spec="COLAMD"); tk = time.perf_counter() - t
        except RuntimeError as e:
            out[f"V4 augmented LU, delta {delta:g}"] = (float("nan"), float("nan"))
            print(f"  augmented LU, delta {delta:g}: {e}"); continue
        fnz = lu.L.nnz + lu.U.nnz
        sol = lu.solve(rhsK); out[f"V4 augmented LU, delta {delta:g}"] = finish(sol[J.size:])
        K0 = sp.bmat([[sp.diags(-1.0 / th), Aj.T], [Aj, None]], format="csc")
        for _ in range(2):
            sol = sol + lu.solve(rhsK - K0 @ sol)
        out[f"V5 V4 + 2 refinement, d {delta:g}"] = finish(sol[J.size:])
    print(f"{dump}: support {J.size:,}, theta {th.min():.1e}..{th.max():.1e}, mu {float(d['mu']):.1e}; last augmented "
          f"LU {tk:.1f} s, factor nonzeros {fnz:,}")
    for k_, (lp_, ld_) in out.items():
        print(f"  {k_:26s} primal {lp_:.2e}  dual {ld_:.2e}")
