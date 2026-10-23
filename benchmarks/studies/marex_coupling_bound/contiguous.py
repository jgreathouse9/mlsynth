"""Batched Wolfe with the running candidates contiguous, warm starts, and a pool worker.

:func:`wolfe_bound` is :func:`batched.wolfe_bound`'s algorithm with three
differences in how it runs, none in what it computes:

* the running candidates sit contiguous at the front of the batch. The major
  step's products run on a view of that block with masks, so nothing large is
  gathered; an append reads only the slice of products it needs. Finished
  candidates stay in the block until a quarter of it has finished, and one
  permutation then compacts it;
* it can start from a given corral (``init``), a warm start: the inherited
  points are re-solved first, by the affine step, as Wolfe (1976, sec. 9)
  re-enters his method after deleting points;
* it records the major steps each candidate took, and returns each candidate's
  final corral so the next candidate in a chain can start from it.

:func:`bound_batch` is the function a process pool maps over batches, with the
Gram matrix and the pair distances set once per worker by :func:`init_worker`.
:func:`bound_and_fit` also fits what the bound leaves in its batch, against its
own copy of the cutoff, with the panel set once per worker by
:func:`init_fit_worker`.
"""
from __future__ import annotations

import numpy as np

from mlsynth.utils.marex_helpers import exact

#: A weight at or below this is treated as zero, and a relative gap at or below
#: it as converged.
TOL = 1e-12

_G = None
_D2 = None
_B = None
_A = None


def init_worker(G, pair_d2):
    """Set the data a pool worker bounds against."""
    global _G, _D2
    _G, _D2 = G, pair_d2


def init_fit_worker(B, A, G, pair_d2):
    """Set the panel a pool worker bounds and fits against."""
    global _B, _A
    init_worker(G, pair_d2)
    _B, _A = B, A


def bound_batch(args):
    """Bound one batch in a pool worker: ``args`` is ``(sets, need)``; returns the status."""
    sets, need = args
    return wolfe_bound(_G, sets, need, _D2)[1]


def margin(c):
    """The search's margin on a cutoff."""
    return exact.PRUNE_MARGIN * (1.0 + abs(c))


def bound_and_fit(args):
    """Bound one batch and fit what the bound leaves, in a pool worker.

    ``args`` is ``(sets, lb, best, beta)``, the batch in ``lb`` order and the best
    value known when it was sent. The worker discards a candidate on its bound
    against ``best`` plus the margin, fits the rest in ``lb`` order as the shipped
    search does, and lowers its own ``best`` as it finds better designs. Returns
    the batch's best value and treated set (``None`` if it found nothing below
    ``best``), the coupled fits it ran and the candidates its bound discarded.
    """
    sets, lb, best, beta = args
    n = _B.shape[1]
    cut = best + margin(best)
    status = wolfe_bound(_G, sets, (cut - lb) / beta, _D2)[1]
    best_S = None
    fits = 0
    for k in np.flatnonzero(status != 1):
        cut = best + margin(best)
        if lb[k] >= cut:
            break
        S = [int(s) for s in sets[k]]
        g, w = exact.side_fit(_B, _A, S)
        if g >= best:
            continue
        C = [j for j in range(n) if j not in S]
        fits += 1
        fit = exact.coupled_fit(_B, _A, S, C, beta, w0=w, cutoff=cut)
        if fit is not None and (fit[0] < best or (
                fit[0] == best and best_S is not None and tuple(S) < best_S)):
            best, best_S = fit[0], tuple(S)
    return best, best_S, fits, int((status == 1).sum())


def closest_pairs(sets, M, pair_d2):
    """Each candidate's closest treated-control pair."""
    n = pair_d2.shape[0]
    D = np.where(M[:, None, :], np.inf, pair_d2[sets])
    flat = D.reshape(len(sets), -1).argmin(axis=1)
    return sets[np.arange(len(sets)), flat // n], flat % n


def wolfe_bound(G, sets, need, pair_d2, max_iter=200, K=24, init=None, compact=0.25):
    """Best bound, status, final corral ``(I, J, V, L)`` and major steps per candidate.

    Status as :func:`batched.wolfe_bound`: 1 cleared ``need``, 2 converged
    without clearing it, 3 corral full, 4 iteration cap.
    """
    b = len(sets)
    n = G.shape[0]
    rows = np.arange(b)
    M = np.zeros((b, n), dtype=bool)
    M[rows[:, None], sets] = True
    I = np.zeros((b, K), dtype=np.int64)
    J = np.zeros((b, K), dtype=np.int64)
    V = np.zeros((b, K), dtype=bool)
    L = np.zeros((b, K))
    major = np.ones(b, dtype=bool)
    if init is None:
        i0, j0 = closest_pairs(sets, M, pair_d2)
        I[:, 0], J[:, 0], V[:, 0], L[:, 0] = i0, j0, True, 1.0
    else:
        k0 = init[0].shape[1]
        I[:, :k0], J[:, :k0], V[:, :k0], L[:, :k0] = init
        major[:] = False
    R = np.where(V[:, :, None], G[I] - G[J], 0.0)                     # (b, K, n)
    Gc = (np.take_along_axis(R, I[:, None, :], 2)
          - np.take_along_axis(R, J[:, None, :], 2))
    Gc = np.where(V[:, :, None] & V[:, None, :], Gc, 0.0)            # (b, K, K)
    best = np.full(b, -np.inf)
    status = np.zeros(b, dtype=int)
    cycles = np.zeros(b, dtype=np.int64)
    orig = rows.copy()
    need = np.asarray(need, dtype=float).copy()
    state = [M, I, J, V, L, R, Gc, major, best, status, cycles, orig, need]
    kk = np.arange(K)
    na = b
    for _ in range(max_iter):
        if na == 0:
            break
        Vv = V[:na]
        live = status[:na] == 0
        Kw = int((K - np.argmax(Vv[:, ::-1], axis=1)).max())
        Lv = L[:na, :Kw]
        majv = major[:na] & live
        cycles[:na] += majv
        if majv.any():
            proj = np.einsum("kq,kqn->kn", Lv, R[:na, :Kw])
            zz = np.einsum("kq,kqr,kr->k", Lv, Gc[:na, :Kw, :Kw], Lv)
            pS = np.where(M[:na], proj, np.inf)
            pC = np.where(M[:na], -np.inf, proj)
            ifw, jfw = pS.argmin(axis=1), pC.argmax(axis=1)
            ar = np.arange(na)
            ps, pc = pS[ar, ifw], pC[ar, jfw]
            with np.errstate(invalid="ignore", divide="ignore"):
                bnd = (ps - pc) / np.sqrt(np.maximum(zz, 1e-300))
            bv, st = best[:na], status[:na]
            bv[majv] = np.maximum(bv[majv], bnd[majv])
            cleared = majv & (bv > 0) & (bv * bv >= need[:na])
            conv = majv & ~cleared & (zz - (ps - pc) <= TOL * zz)
            dup = majv & ~cleared & ~conv & np.any(
                Vv[:, :Kw] & (I[:na, :Kw] == ifw[:, None]) & (J[:na, :Kw] == jfw[:, None]), axis=1)
            full = majv & ~cleared & ~conv & ~dup & Vv.all(axis=1)
            st[cleared] = 1
            st[conv | dup] = 2
            st[full] = 3
            go = np.flatnonzero(majv & ~(cleared | conv | dup | full))
            if go.size:
                i_, j_ = ifw[go], jfw[go]
                s = np.argmax(~V[go], axis=1)
                I[go, s], J[go, s], V[go, s], L[go, s] = i_, j_, True, 0.0
                R[go, s] = G[i_] - G[j_]
                col = (R[go[:, None], kk[None, :], i_[:, None]]
                       - R[go[:, None], kk[None, :], j_[:, None]]) * V[go]
                Gc[go, :, s] = col
                Gc[go, s, :] = col
                major[go] = False
        mn = np.flatnonzero((~major[:na]) & (status[:na] == 0))
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
            Vnew = Vm.copy()
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
                Vnew[bad] = Vb
            L[mn, :Kw] = Lnew
            V[mn, :Kw] = Vnew
            major[mn[ok]] = True
        done = int((status[:na] != 0).sum())
        if done and (done >= compact * na or done == na):
            perm = np.argsort(status[:na] != 0, kind="stable")
            for a in state:
                a[:na] = a[:na][perm]
            na -= done
    status[:na][status[:na] == 0] = 4
    out = [np.empty_like(x) for x in (best, status, I, J, V, L, cycles)]
    for o, x in zip(out, (best, status, I, J, V, L, cycles)):
        o[orig] = x
    best_o, status_o, I_o, J_o, V_o, L_o, cycles_o = out
    return best_o, status_o, (I_o, J_o, V_o, L_o), cycles_o
