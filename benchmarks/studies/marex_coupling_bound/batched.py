"""The hull-distance bound for a batch of candidates per numpy call.

:func:`bound.hull_distance_bound` takes about 1.4 ms per candidate, most of it
the overhead of about sixteen small least-squares calls and the Python loop
around them, not arithmetic: a least-squares solve on a 90-by-8 corral is about
12,000 floating-point operations and takes about 40 microseconds as a numpy
call (``profile_bound.py``). These run one method in lockstep over a batch
instead.

:func:`wolfe_bound` is Wolfe's method with every quantity written through the
Gram matrix ``G = Bc'Bc`` of the target-centred markets. A corral point is a
pair ``(i, j)``, the difference ``b_i - b_j``; its products with every market
are ``G[i] - G[j]``, so the iterate's projections, its norm and the affine step
need no length-``T`` vector. Each candidate keeps its corral points' market
products ``R`` and their Gram ``Gc``, both updated when a point is added.
Dropped points leave holes, and the affine step measures from the first valid
slot. At every major step the bound is Wolfe's Theorem 2.2 from the current
point's products, ``(min over S - max over C) / |z|``, and a candidate leaves
the batch as soon as its square reaches what the caller needs.

:func:`frank_wolfe_bound` batches without corrals: Frank-Wolfe, plain or
pairwise (Lacoste-Julien and Jaggi 2015), on the two simplexes, the point
``z = B w - B u``, with the same bound at every iterate. It converges too slowly
to clear most candidates; ``batched_curve.py`` records how slowly.

Both bounds are valid for any iterate: they read the data only through the
products of the current point with the markets. :func:`wolfe_bound` reads them
from ``G``, so its rounding is that of forming ``G`` and of these products.
"""
from __future__ import annotations

import numpy as np

#: Corral slots per candidate; a candidate whose corral fills stops and is fitted.
CORRAL = 24

#: A weight at or below this is treated as zero, and a relative gap at or below
#: it as converged.
TOL = 1e-12


def _start(G, sets, pair_d2):
    """Each candidate's membership mask and its closest treated-control pair."""
    b, m = sets.shape
    n = G.shape[0]
    rows = np.arange(b)
    M = np.zeros((b, n), dtype=bool)
    M[rows[:, None], sets] = True
    D = np.where(M[:, None, :], np.inf, pair_d2[sets])
    flat = D.reshape(b, -1).argmin(axis=1)
    return M, sets[rows, flat // n], flat % n


def wolfe_bound(G, sets, need, pair_d2, max_iter=200, K=CORRAL):
    """Wolfe's method in lockstep: the best bound per candidate and a status.

    Status 1: the bound's square reached ``need``. 2: converged without
    reaching it. 3: the corral filled. 4: ``max_iter`` lockstep iterations.
    """
    b = len(sets)
    n = G.shape[0]
    rows = np.arange(b)
    M, i0, j0 = _start(G, sets, pair_d2)
    I = np.zeros((b, K), dtype=np.int64)
    J = np.zeros((b, K), dtype=np.int64)
    V = np.zeros((b, K), dtype=bool)
    L = np.zeros((b, K))
    R = np.zeros((b, K, n))
    Gc = np.zeros((b, K, K))
    I[:, 0], J[:, 0], V[:, 0], L[:, 0] = i0, j0, True, 1.0
    R[:, 0] = G[i0] - G[j0]
    Gc[:, 0, 0] = R[rows, 0, i0] - R[rows, 0, j0]
    major = np.ones(b, dtype=bool)
    best = np.full(b, -np.inf)
    status = np.zeros(b, dtype=int)
    act = rows
    for _ in range(max_iter):
        act = act[status[act] == 0]
        if not act.size:
            break
        # Major steps: the support point of P - Q along -z, the bound, the append.
        mj = act[major[act]]
        if mj.size:
            Kw = int((K - np.argmax(V[mj][:, ::-1], axis=1)).max())
            Lm, Vm = L[mj, :Kw], V[mj, :Kw]
            proj = np.einsum("kq,kqn->kn", Lm, R[mj, :Kw])
            zz = np.einsum("kq,kqr,kr->k", Lm, Gc[mj, :Kw, :Kw], Lm)
            pS = np.where(M[mj], proj, np.inf)
            pC = np.where(M[mj], -np.inf, proj)
            ifw, jfw = pS.argmin(axis=1), pC.argmax(axis=1)
            ar = np.arange(len(mj))
            ps, pc = pS[ar, ifw], pC[ar, jfw]
            best[mj] = np.maximum(best[mj], (ps - pc) / np.sqrt(np.maximum(zz, 1e-300)))
            bb = best[mj]
            cleared = (bb > 0) & (bb * bb >= need[mj])
            conv = ~cleared & (zz - (ps - pc) <= TOL * zz)
            dup = ~cleared & ~conv & np.any(
                Vm & (I[mj, :Kw] == ifw[:, None]) & (J[mj, :Kw] == jfw[:, None]), axis=1)
            full = ~cleared & ~conv & ~dup & V[mj].all(axis=1)
            status[mj[cleared]] = 1
            status[mj[conv | dup]] = 2
            status[mj[full]] = 3
            go = ~(cleared | conv | dup | full)
            if go.any():
                g, i_, j_ = mj[go], ifw[go], jfw[go]
                s = np.argmax(~V[g], axis=1)
                I[g, s], J[g, s], V[g, s], L[g, s] = i_, j_, True, 0.0
                R[g, s] = G[i_] - G[j_]
                Rg = R[g]
                col = (np.take_along_axis(Rg, i_[:, None, None], 2)[..., 0]
                       - np.take_along_axis(Rg, j_[:, None, None], 2)[..., 0]) * V[g]
                Gc[g, :, s] = col
                Gc[g, s, :] = col
                major[g] = False
        # Affine steps: the corral's affine minimiser, then Wolfe's minor cycle.
        mn = act[(~major[act]) & (status[act] == 0)]
        if mn.size:
            Kw = int((K - np.argmax(V[mn][:, ::-1], axis=1)).max())
            Vm, Lm, Gm = V[mn, :Kw], L[mn, :Kw], Gc[mn, :Kw, :Kw]
            ref = np.argmax(Vm, axis=1)
            ar = np.arange(len(mn))
            gr = Gm[ar, :, ref]
            grr = Gm[ar, ref, ref]
            E = Gm - gr[:, :, None] - gr[:, None, :] + grr[:, None, None]
            r = -(gr - grr[:, None])
            edge = Vm.copy()
            edge[ar, ref] = False
            scale = np.maximum(np.abs(np.diagonal(E, axis1=1, axis2=2)).max(axis=1), 1.0)
            E = (np.where(edge[:, :, None] & edge[:, None, :], E, 0.0)
                 + np.eye(Kw)[None] * np.where(edge, 1e-13 * scale[:, None], 1.0)[:, None, :])
            alpha = np.linalg.solve(E, np.where(edge, r, 0.0)[..., None])[..., 0]
            w = alpha * edge
            w[ar, ref] = 1.0 - w.sum(axis=1)
            ok = np.all(np.where(Vm, w > TOL, True), axis=1)
            Lnew = Lm.copy()
            Lnew[ok] = w[ok]
            bad = ~ok
            if bad.any():
                wb, lb_, Vb = w[bad], Lm[bad], Vm[bad]
                with np.errstate(divide="ignore", invalid="ignore"):
                    th = np.where(Vb & (wb <= TOL), lb_ / (lb_ - wb), np.inf)
                theta = np.clip(th.min(axis=1), 0.0, 1.0)
                lnew = lb_ + theta[:, None] * (wb - lb_)
                Vb = Vb & (lnew > TOL)
                lnew = np.where(Vb, lnew, 0.0)
                Lnew[bad] = lnew / lnew.sum(axis=1, keepdims=True)
                Vfull = V[mn]
                Vfull[bad, :Kw] = Vb
                V[mn] = Vfull
            Lfull = L[mn]
            Lfull[:, :Kw] = Lnew
            L[mn] = Lfull
            major[mn[ok]] = True
    status[act[status[act] == 0]] = 4
    return best, status


def frank_wolfe_bound(B, sets, need, pair_d2, max_iter=200, pairwise=True):
    """Frank-Wolfe in lockstep: the best bound per candidate and the iteration it
    reached ``need`` (-1 if it did not)."""
    T, n = B.shape
    b = len(sets)
    Bt = B.T
    M, i0, j0 = _start(B.T @ B, sets, pair_d2)
    rows = np.arange(b)
    W = np.zeros((b, n))
    U = np.zeros((b, n))
    W[rows, i0] = 1.0
    U[rows, j0] = 1.0
    z = Bt[i0] - Bt[j0]
    best = np.full(b, -np.inf)
    cleared_at = np.full(b, -1)
    act = rows
    for it in range(max_iter):
        proj = z @ B
        zz = np.einsum("ij,ij->i", z, z)
        pS = np.where(M, proj, np.inf)
        pC = np.where(M, -np.inf, proj)
        ifw, jfw = pS.argmin(axis=1), pC.argmax(axis=1)
        ar = np.arange(len(act))
        best[act] = np.maximum(best[act], (pS[ar, ifw] - pC[ar, jfw]) / np.sqrt(zz))
        bb = best[act]
        done = (bb > 0) & (bb * bb >= need[act])
        cleared_at[act[done]] = it
        keep = ~done
        if not keep.any():
            break
        act, z, W, U, M = act[keep], z[keep], W[keep], U[keep], M[keep]
        proj, ifw, jfw = proj[keep], ifw[keep], jfw[keep]
        ar = np.arange(len(act))
        if pairwise:
            ia = np.where(W > 0, proj, -np.inf).argmax(axis=1)
            ja = np.where(U > 0, proj, np.inf).argmin(axis=1)
            d = Bt[ifw] - Bt[ia] - Bt[jfw] + Bt[ja]
            gmax = np.minimum(W[ar, ia], U[ar, ja])
        else:
            d = Bt[ifw] - Bt[jfw] - z
            gmax = np.ones(len(act))
        dd = np.einsum("ij,ij->i", d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(dd > 0, -np.einsum("ij,ij->i", z, d) / dd, 0.0)
        g = np.clip(g, 0.0, gmax)
        z = z + g[:, None] * d
        if pairwise:
            W[ar, ifw] += g
            W[ar, ia] -= g
            U[ar, jfw] += g
            U[ar, ja] -= g
            np.maximum(W, 0.0, out=W)
            np.maximum(U, 0.0, out=U)
        else:
            W *= (1 - g)[:, None]
            W[ar, ifw] += g
            U *= (1 - g)[:, None]
            U[ar, jfw] += g
    return best, cleared_at
