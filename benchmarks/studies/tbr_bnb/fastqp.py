"""The hull's simplex QP in one compiled loop, when numba is installed (#1068).

hull.ControlBound.piece and hull.Band._at call the library's capped simplex QP,
mlsynth.utils.solvers.active_set.solve_simplex_qp, tens of thousands of times a run,
about 0.3 ms a call and mostly the interpreter (bed s15_k3: 33,390 calls, 47% of its
time under cProfile). ``solve`` is that solver's loop compiled for the calls the
hull makes, upper and warm_start, operation by operation:

* the same LAPACK dgelsy (scipy's cython_lapack, the library the f2py wrapper
  reaches) on the same column-major bytes, with the same rcond and lwork;
* each matrix-vector product through BLAS dgemv with the transpose flag numpy uses
  for the operand's layout (numpy's B[:, idx] is F-contiguous, the hull's Z is C,
  the band's s X is F);
* numpy's pairwise summation for the sums and means of 1-D arrays, and the column
  order numpy's mean(axis=1) takes over an F array;
* r @ r and the norms as BLAS ddot.

So its weights are the library's bit for bit (test_fastqp.py), and the hull's bounds
and every decision on them are the same. Two steps stay outside: the FISTA seed of a
cold solve on 80 or more donors is the library's, computed in Python, and the
einsum that picks the power-of-two unit scale is not reproduced to the bit, so a
call whose scale lies within 1e-12 of its rounding boundary goes to the library.
So does every other call (a linear term, other options or storage), and every call
on a machine without numba.

The loop is the library's (active_set.solve_simplex_qp): a change there needs the
same change here, and the bit-for-bit property in test_fastqp.py fails until it is
made.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from mlsynth.utils.solvers import active_set as _as
from mlsynth.utils.solvers.accelerate import ACCEL_MIN_DONORS, fista_warm_start

try:  # numba is the optional ``fast`` extra
    import llvmlite.binding as _llvm
    from numba import njit, types
    from numba.extending import get_cython_function_address
    AVAILABLE = True
except ImportError:  # pragma: no cover - the suite runs with numba installed
    AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name (fastfloor.CANONICAL_NAME
#: gives the reason); under any other name the kernel compiles in memory.
CANONICAL_NAME = "benchmarks.studies.tbr_bnb.fastqp"
_CACHE = __name__ == CANONICAL_NAME

if AVAILABLE:
    # dgelsy as an external symbol resolved by name, so the kernel can be cached on
    # disk (a ctypes pointer is a constant of one process and turns caching off).
    _llvm.add_symbol("mlsynth_tbr_fastqp_dgelsy",
                     get_cython_function_address("scipy.linalg.cython_lapack", "dgelsy"))
    _dgelsy = types.ExternalFunction("mlsynth_tbr_fastqp_dgelsy", types.void(*([types.voidptr] * 13)))

_EPS = float(np.finfo(float).eps)
_SQRT_HALF = math.sqrt(0.5)


# ---------------------------------------------------------------- reductions
@njit(cache=_CACHE)
def _leaf(a, s, n):
    """numpy's pairwise_sum over a[s:s+n] for n <= 128: left to right below eight,
    else eight running sums combined in pairs and the remainder added in order."""
    if n < 8:
        res = -0.0
        for i in range(n):
            res += a[s + i]
        return res
    r0 = a[s]; r1 = a[s + 1]; r2 = a[s + 2]; r3 = a[s + 3]
    r4 = a[s + 4]; r5 = a[s + 5]; r6 = a[s + 6]; r7 = a[s + 7]
    lim = n - n % 8
    i = 8
    while i < lim:
        r0 += a[s + i]; r1 += a[s + i + 1]; r2 += a[s + i + 2]; r3 += a[s + i + 3]
        r4 += a[s + i + 4]; r5 += a[s + i + 5]; r6 += a[s + i + 6]; r7 += a[s + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[s + i]
        i += 1
    return res


@njit(cache=_CACHE)
def _pw(a, s, n):
    """numpy's pairwise_sum over a[s:s+n]: above 128 the halves at n2 = n // 2 less
    n2 % 8 summed apart and added. A stack in place of numpy's recursion; the
    additions are the same, in the same pairs."""
    if n <= 128:
        return _leaf(a, s, n)
    fs = np.empty(128, np.int64); fn = np.empty(128, np.int64); fp = np.empty(128, np.int64)
    vals = np.empty(128)
    top = 0; nv = 0
    fs[0] = s; fn[0] = n; fp[0] = 0; top = 1
    while top > 0:
        top -= 1
        s_, n_, p_ = fs[top], fn[top], fp[top]
        if n_ <= 128:
            vals[nv] = _leaf(a, s_, n_); nv += 1
        elif p_ == 1:                            # both halves done: left + right
            vals[nv - 2] = vals[nv - 2] + vals[nv - 1]; nv -= 1
        else:
            n2 = n_ // 2
            n2 -= n2 % 8
            fs[top] = s_; fn[top] = n_; fp[top] = 1; top += 1
            fs[top] = s_ + n2; fn[top] = n_ - n2; fp[top] = 0; top += 1     # the right half second
            fs[top] = s_; fn[top] = n2; fp[top] = 0; top += 1               # the left half first
    return vals[0]


@njit(cache=_CACHE)
def _sum(a):
    """np.sum of a 1-D array: the identity, plus the pairwise sum."""
    return 0.0 + _pw(a, 0, a.size)


@njit(cache=_CACHE)
def _norm(x):
    return math.sqrt(np.dot(x, x))


@njit(cache=_CACHE)
def _argmin(x):
    k = 0
    for i in range(x.size):
        if np.isnan(x[i]):
            return i
        if x[i] < x[k]:
            k = i
    return k


@njit(cache=_CACHE)
def _argmax(x):
    k = 0
    for i in range(x.size):
        if np.isnan(x[i]):
            return i
        if x[i] > x[k]:
            k = i
    return k


# --------------------------------------------------------- matrix storage
# A matrix B (m x J) is held as a C array S with a flag t: B = S when t is
# False (B C-contiguous), B = S.T when t is True (B F-contiguous). numba's dot
# then calls dgemv with the flag numpy's matmul uses for that layout.
@njit(cache=_CACHE)
def _mv(S, t, x):
    """B @ x."""
    if t:
        return np.dot(S.T, x)
    return np.dot(S, x)


@njit(cache=_CACHE)
def _mtv(S, t, x):
    """B.T @ x."""
    if t:
        return np.dot(S, x)
    return np.dot(S.T, x)


@njit(cache=_CACHE)
def _cols(S, t, idx, m):
    """B[:, idx] (F-contiguous in numpy) as its C transpose, shape (len(idx), m)."""
    out = np.empty((idx.size, m))
    for k in range(idx.size):
        j = idx[k]
        if t:
            for i in range(m):
                out[k, i] = S[j, i]
        else:
            for i in range(m):
                out[k, i] = S[i, j]
    return out


@njit(cache=_CACHE)
def _gelsy(Mt, rhs, m, n):
    """lstsq(M, rhs, gelsy)[0] with M (m x n) given as its C transpose Mt (n x m),
    the column-major bytes the f2py wrapper hands LAPACK."""
    a = Mt.copy()
    ldb = max(m, n)
    b = np.zeros(ldb)
    b[:m] = rhs
    minmn = min(m, n)
    lwork = max(minmn + 3 * n + 1, 2 * minmn + 1) * 4
    work = np.empty(lwork)
    jpvt = np.zeros(n, dtype=np.int32)
    ia = np.empty(7, dtype=np.int32)
    ia[0] = m; ia[1] = n; ia[2] = 1; ia[3] = max(1, m); ia[4] = max(1, ldb); ia[5] = lwork; ia[6] = 0
    rank = np.zeros(1, dtype=np.int32)
    info = np.zeros(1, dtype=np.int32)
    rc = np.empty(1)
    rc[0] = 1e-12
    m_ = ia[0:1]; n_ = ia[1:2]; nr_ = ia[2:3]; lda_ = ia[3:4]; ldb_ = ia[4:5]; lw_ = ia[5:6]
    _dgelsy(m_.ctypes, n_.ctypes, nr_.ctypes, a.ctypes, lda_.ctypes, b.ctypes, ldb_.ctypes,
            jpvt.ctypes, rc.ctypes, rank.ctypes, work.ctypes, lw_.ctypes, info.ctypes)
    return b[:n].copy()


# ------------------------------------------------------------------ kernel
@njit(cache=_CACHE)
def _kkt(DS, t, w, m, J, up, has_up):
    """simplex_kkt_error(B_in, A_in, w, None, upper) with D = B_in - A_in 1' held as DS."""
    g = _mtv(DS, t, _mv(DS, t, w))
    if not has_up:
        on = w > 1e-9
        if not on.any():
            on = w >= w.max()
        gon = g[on]
        nu = _sum(gon) / gon.size
        stat = _norm(gon - nu)
        if on.all():
            neg = 0.0
        else:
            neg = max(0.0, -(g[~on] - nu).min())
    else:
        cap = w >= up - 1e-9
        zero = (w <= 1e-9) & ~cap
        on = ~cap & ~zero
        if on.any():
            gon = g[on]
            nu = _sum(gon) / gon.size
            stat = _norm(gon - nu)
        else:
            lo_nu = g[cap].max() if cap.any() else -np.inf
            hi_nu = g[zero].min() if zero.any() else np.inf
            if np.isfinite(lo_nu) and np.isfinite(hi_nu):
                nu = (lo_nu + hi_nu) / 2.0
            elif np.isfinite(lo_nu):
                nu = lo_nu
            else:
                nu = hi_nu
            stat = 0.0
        neg = max(0.0, -(g[zero] - nu).min()) if zero.any() else 0.0
        neg = max(neg, (g[cap] - nu).max()) if cap.any() else neg
    fro = _norm(DS.ravel())                 # norm(D): ddot over memory order
    size = fro * fro * _norm(w) + 0.0
    e = max(stat, neg)
    return e / size if size > 0.0 else e


@njit(cache=_CACHE)
def _finish(w, DS, t, m, J, up, has_up):
    w = np.maximum(w, 0.0)
    total = _sum(w)
    if total > 0:
        w = w / total
    if has_up:
        w = np.minimum(w, up)
    return w, _kkt(DS, t, w, m, J, up, has_up)


@njit(cache=_CACHE)
def _prepare(S, t, A_in):
    """D = B - A 1' (as DS), its unit-scale copy (as US) and the near-boundary flag."""
    if t:
        J, m = S.shape
    else:
        m, J = S.shape
    DS = np.empty_like(S)
    energy_c = np.zeros(J)
    if t:
        for j in range(J):
            acc = 0.0
            for i in range(m):
                d = S[j, i] - A_in[i]
                DS[j, i] = d
                acc += d * d
            energy_c[j] = acc
    else:
        for i in range(m):
            for j in range(J):
                d = S[i, j] - A_in[i]
                DS[i, j] = d
                energy_c[j] += d * d
    energy = _sum(energy_c) / J
    scale = 1.0
    near = False
    if energy > 0.0 and np.isfinite(energy):
        mant, ex = math.frexp(math.sqrt(energy))
        # The einsum's summation order is not reproduced; within 1e-12 of the
        # rounding boundary of the power of two, let the caller fall back.
        if abs(mant - _SQRT_HALF) < 1e-12 or abs(mant - 0.5) < 1e-12 or mant > 1.0 - 1e-12:
            near = True
        scale = math.ldexp(1.0, ex - (1 if mant < _SQRT_HALF else 0))
    US = DS / scale
    return DS, US, m, J, near


@njit(cache=_CACHE)
def _core(S, t, A_in, ws, has_ws, up, has_up, tol):
    DS, B, m, J, near = _prepare(S, t, A_in)
    if near:
        return np.zeros(J), 0, False, 0.0, 2
    if J == 1:
        w, kkt = _finish(np.ones(1), DS, t, m, J, up, has_up)
        return w, 0, True, kkt, 0
    max_iter = 50 * J
    A = np.zeros(m)
    absB = np.abs(B)
    absA = np.abs(A)
    rel = min(tol, 0.1 * 1e-10)
    noise = (J + 1) * 0.5 * _EPS
    fro2 = _sum((B * B).ravel())
    # feasible start
    have_w = False
    w = np.zeros(J)
    if has_ws:
        ok = ws.size == J and np.all(np.isfinite(ws)) and ws.min() >= -tol and abs(_sum(ws) - 1.0) <= 1e-6
        if ok and has_up:
            ok = np.all(ws <= up + tol)
        if ok:
            w = np.maximum(ws, 0.0)
            w = w / _sum(w)
            if has_up:
                w = np.minimum(w, up)
            have_w = True
    if not have_w:
        if has_up:
            w = up / _sum(up)
        else:
            w = np.full(J, 1.0 / J)
    active = w <= tol
    if has_up:
        capped = (w >= up - tol) & ~active
        for j in range(J):
            if capped[j]:
                w[j] = up[j]
    else:
        capped = np.zeros(J, dtype=np.bool_)

    pivots = 0
    converged = False
    f_stop = np.inf
    have_stop = False
    w_stop = w.copy()
    for _ in range(max_iter):
        free = np.flatnonzero(~active & ~capped)
        anycap = capped.any()
        if free.size == 0 and anycap:
            rv = _mv(B, t, w) - A
            gv = _mtv(B, t, rv)
            sv = rel * (_mtv(absB, t, np.abs(rv)) + 0.0) + noise * _mtv(absB, t, _mv(absB, t, w) + absA)
            floor_v = noise * (fro2 * _norm(w) + 0.0)
            ci = np.flatnonzero(capped)
            ri = np.flatnonzero(active & (up > tol))
            if ri.size:
                i = ci[_argmax(gv[ci])]
                j = ri[_argmin(gv[ri])]
                if gv[i] - gv[j] > sv[i] + sv[j] + floor_v:
                    capped[i] = False
                    active[j] = False
                    pivots += 1
                    continue
            converged = True
            break
        if free.size == 0:
            active[_argmax(w)] = False
            free = np.flatnonzero(~active & ~capped)
        nF = free.size
        if anycap:
            cidx = np.flatnonzero(capped)
            uc = up[cidx]
            t_free = 1.0 - _sum(uc)
            Ct = _cols(B, t, cidx, m)
            A_free = A - np.dot(Ct.T, uc)
        else:
            t_free = 1.0
            A_free = A
        wF = np.empty(nF)
        if nF == 1:
            wF[0] = t_free
        else:
            BFt = _cols(B, t, free, m)
            # BF.mean(axis=1) of the F-contiguous BF: column by column
            cm = np.zeros(m)
            for k in range(nF):
                for i in range(m):
                    cm[i] += BFt[k, i]
            cm = cm / nF
            Mt = np.empty((nF - 1, m))
            for k in range(nF - 1):
                for i in range(m):
                    Mt[k, i] = BFt[k, i] - BFt[nF - 1, i]
            if not anycap:
                rhs = A - cm
            else:
                rhs = A_free - t_free * cm
            v = _gelsy(Mt, rhs, m, nF - 1)
            base = t_free / nF
            for k in range(nF - 1):
                wF[k] = base + v[k]
            wF[nF - 1] = base - _sum(v)

        feas = wF.min() >= -tol
        if feas and has_up:
            feas = np.all(wF <= up[free] + tol)
        if feas:
            w = np.zeros(J)
            w[free] = np.maximum(wF, 0.0)
            if has_up:
                w[free] = np.minimum(w[free], up[free])
                cidx = np.flatnonzero(capped)
                w[cidx] = up[cidx]
            low = wF < 0.0
            if has_up:
                high = wF > up[free]
            else:
                high = np.zeros(nF, dtype=np.bool_)
            if low.any() or high.any():
                active[free[low]] = True
                if has_up:
                    capped[free[high]] = True
                    room = 1.0 - _sum(w[capped])
                    rest = ~active & ~capped
                    if rest.any():
                        sr = _sum(w[rest])
                        if sr > 0.0:
                            fac = room / sr
                            ridx = np.flatnonzero(rest)
                            w[ridx] = w[ridx] * fac
                else:
                    w = w / _sum(w)
                pivots += 1
                continue
            r = _mv(B, t, w) - A
            g = _mtv(B, t, r)
            nu = _sum(g[free]) / nF
            pinned = active | capped
            anypin = pinned.any()
            k = -1
            ai = np.flatnonzero(pinned)
            reduced = np.empty(ai.size)
            if anypin:
                for q in range(ai.size):
                    if capped[ai[q]]:
                        reduced[q] = nu - g[ai[q]]
                    else:
                        reduced[q] = g[ai[q]] - nu
                size = rel * (_mtv(absB, t, np.abs(r)) + 0.0) + noise * _mtv(absB, t, _mv(absB, t, w) + absA)
                floor = noise * (fro2 * _norm(w) + 0.0)
                thr = size[ai] + (_sum(size[free]) / nF) + floor
                kk = _argmin(reduced / thr)
                if reduced[kk] < -thr[kk]:
                    active[ai[kk]] = False
                    capped[ai[kk]] = False
                    pivots += 1
                    continue
                k = _argmin(reduced)
            if have_stop or (k >= 0 and reduced[k] < 0.0):
                f = np.dot(r, r)
                f_err = np.dot(r, r) + 2.0 * np.dot(np.abs(r), _mv(absB, t, w) + absA)
                if have_stop and not f < f_stop - noise * f_err:
                    w = w_stop
                elif k >= 0 and reduced[k] < 0.0:
                    f_stop = f
                    w_stop = w.copy()
                    have_stop = True
                    active[ai[k]] = False
                    capped[ai[k]] = False
                    pivots += 1
                    continue
            converged = True
            break

        # blocking step
        cur = w[free]
        direction = wF - cur
        blocking = direction < -tol
        ratios = np.where(blocking, cur / np.maximum(-direction, tol), np.inf)
        if has_up:
            uF = up[free]
            rising = direction > tol
            ratios_up = np.where(rising, (uF - cur) / np.maximum(direction, tol), np.inf)
            step = min(1.0, ratios.min(), ratios_up.min())
        else:
            uF = up[:0]
            ratios_up = ratios
            step = min(1.0, ratios.min())
        cur = cur + step * direction
        w = np.zeros(J)
        w[free] = np.maximum(cur, 0.0)
        hit = free[cur <= tol]
        if has_up:
            cidx = np.flatnonzero(capped)
            w[cidx] = up[cidx]
            top = free[cur >= uF - tol]
            w[top] = up[top]
            capped[top] = True
            if hit.size == 0 and top.size == 0:
                jj = _argmin(np.minimum(ratios, ratios_up))
                if ratios_up[jj] < ratios[jj]:
                    capped[free[jj]] = True
                    w[free[jj]] = up[free[jj]]
                else:
                    hit = free[jj:jj + 1]
        elif hit.size == 0:
            hit = free[_argmin(cur):_argmin(cur) + 1]
        active[hit] = True
        pivots += 1

    w, kkt = _finish(w, DS, t, m, J, up, has_up)
    return w, pivots, converged, kkt, 0


@njit(cache=_CACHE)
def _check_upper(u):
    """0 ok, 1 non-finite, 2 negative, 3 sums below one; 4 all loose (>= 1)."""
    for x in u:
        if not np.isfinite(x):
            return 1
    for x in u:
        if x < 0:
            return 2
    if _sum(u) < 1.0:
        return 3
    for x in u:
        if x < 1.0:
            return 0
    return 4


_EMPTY = np.zeros(0)
_DEFAULT_TOL = 1e-9

#: The library's solver, which takes every call the kernel does not.
_library = _as.solve_simplex_qp


def solve(B, A, *, warm_start=None, linear=None, tol=_DEFAULT_TOL, max_iter=None, return_info=False,
          accelerate=True, upper=None):
    """solve_simplex_qp(B, A, ...), the same weights to the bit: compiled for the
    hull's calls (float64 arrays, B C- or F-contiguous, ``upper`` and ``warm_start``
    only), the library's solver for any other call and wherever numba is missing."""
    kw = dict(warm_start=warm_start, linear=linear, tol=tol, max_iter=max_iter, return_info=return_info,
              accelerate=accelerate, upper=upper)
    if (not AVAILABLE or linear is not None or max_iter is not None or return_info or not accelerate
            or tol != _DEFAULT_TOL or type(B) is not np.ndarray or type(A) is not np.ndarray
            or B.dtype != np.float64 or A.dtype != np.float64 or B.ndim != 2 or A.ndim != 1
            or A.shape[0] != B.shape[0] or B.shape[1] == 0 or not A.flags.c_contiguous):
        return _library(B, A, **kw)
    if B.flags.c_contiguous:
        S, t = B, False
    elif B.flags.f_contiguous:
        S, t = B.T, True
    else:
        return _library(B, A, **kw)
    J = B.shape[1]
    up, has_up = _EMPTY, False
    if upper is not None:
        u = np.asarray(upper, dtype=float)
        u = np.full(J, float(u)) if u.ndim == 0 else u.ravel()
        if u.shape != (J,):
            return _library(B, A, **kw)          # raises
        code = _check_upper(u)
        if code in (1, 2, 3):
            return _library(B, A, **kw)          # raises
        if code == 0:                            # a cap below one; all loose is the uncapped program
            up, has_up = np.ascontiguousarray(u), True
    ws = warm_start
    if ws is None and J >= ACCEL_MIN_DONORS:     # the library's cold seed, as it computes it
        Bu, Au, _ = _as._unit_problem(B, A, None)
        ws = fista_warm_start(Bu, Au)
    if ws is None:
        wsa, has_ws = _EMPTY, False
    else:
        wsa, has_ws = np.ascontiguousarray(np.asarray(ws, dtype=float).ravel()), True
    w, pivots, converged, kkt, status = _core(S, t, A, wsa, has_ws, up, has_up, float(tol))
    if status == 2:                              # the unit scale at its rounding boundary
        return _library(B, A, **kw)
    if not (converged and kkt <= _as.CERTIFY_TOL):
        warnings.warn(_as.SimplexQPUncertifiedWarning(
            f"solve_simplex_qp: the returned weights are not certified "
            f"(KKT error {kkt:.2e} against {_as.CERTIFY_TOL:.0e}, after {pivots} "
            f"pivots); pass return_info=True to read the outcome."),
            stacklevel=2)
    return w
