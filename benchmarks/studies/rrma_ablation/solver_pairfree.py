"""A pair-free, batched, exact solver for RRMA's weighted Wilcoxon fits.

Each RRMA fit minimises ``F(beta) = sum_{i<j} h_i h_j |e_i - e_j|`` with
``e = y - X beta``. ``oracle.wlad`` solves it as a linear programme with one
column per pair of periods. This module solves ``K`` fits at once with no array
carrying a pair dimension, and certifies each optimum exactly.

The reduction. In the dual, the multiplier of pair ``(i, j)`` is a flow on an
edge of the complete graph on the periods, with capacity ``h_i h_j``; objective
and constraints read only each period's net flow. By Hoffman's circulation
theorem (Schrijver 1986, Corollary 21.3a and Application 21.1, which allow
negative lower bounds, so each pair is an arc with bounds ``[-h_i h_j, h_i h_j]``;
Bazaraa, Jarvis and Sherali 2010, Exercise 11.22, states the same condition for
the out-of-kilter problem) a vector of net flows ``psi`` on a set of periods
``G`` is feasible when ``psi(U) <= h(U) h(G \\ U)`` for every ``U``. The best net flows against fixed
residuals come from a sort: ``a_i = h_i (h below - h above)``. That identity,
Edmonds' greedy algorithm on the cut function and Jaeckel's rank dispersion in
Hettmansperger and McKean (1998), is not taken from the LP texts cited here;
``benchmarks/tests/test_rrma_pairfree_solver.py`` checks it against the
pairwise objective and subgradient.

The algorithm is the bounded dual simplex (Pan 2023, Algorithm 8.4.1) written
on periods:

* a vertex is a labelling of the periods into tie groups, ``sum(|G| - 1) = p``
  tie equations ``e_i = e_root`` that fix ``beta``;
* pricing: rank scores with pairs inside a group excluded, then the ``p`` tie
  multipliers from one ``p x p`` solve, then each period's net flow ``psi``;
* certificate: the cut function ``h(U) (h(G) - h(U))`` is a concave function of
  ``h(U)``. Writing it as the minimum of its tangent lines turns
  ``min_U [h(U) (h(G) - h(U)) - psi(U)]`` into a minimum over thresholds on
  ``psi_i / h_i``, so the most violated ``U`` is a prefix of ``G`` in that order.
  Checking those prefixes is exact, and the structural degeneracy of pairwise
  LAD (``p + 1`` tied periods tie all their pairs) needs no perturbation;
* pivot: split a group along its most violated prefix, a descent edge, and take
  the long step to the first breakpoint where the slope turns non-negative.
  ``line_search="kinetic"`` finds it by swapping adjacent periods as their
  residuals cross (memory ``O(n)``); ``"pairs"`` lists the step's breakpoints in
  one transient ``(K, n, n)`` array and sorts them once;
* warm start: a new period joins as its own group, so the previous window's
  labelling is a vertex of the next, and it is dual feasible because every pair
  variable is boxed (Pan Section 8.2, condition (8.19)). It is the counterpart
  of restarting from the old basis when a variable is added (Pan Section
  7.3.2; Bazaraa, Jarvis and Sherali 2010, Section 6.7).

Run::

    python -m benchmarks.studies.rrma_ablation.solver_pairfree --reps 1 8

Results are written to ``results/solver_pairfree.json``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "solver_pairfree.json"
TOL = 1e-9


def _take(a, idx):
    return np.take_along_axis(a, idx, axis=1)


def _runs(keys_sorted):
    """Start and end position of each row's run of equal labels in a sorted (K, n) array."""
    K, n = keys_sorted.shape
    idx = np.broadcast_to(np.arange(n), (K, n))
    start = np.ones((K, n), bool)
    start[:, 1:] = keys_sorted[:, 1:] != keys_sorted[:, :-1]
    last = np.ones((K, n), bool)
    last[:, :-1] = start[:, 1:]
    s_pos = np.maximum.accumulate(np.where(start, idx, 0), axis=1)
    e_pos = np.minimum.accumulate(np.where(last, idx, n)[:, ::-1], axis=1)[:, ::-1]
    return idx, last, s_pos, e_pos


def rank_scores(h, labels, *keys):
    """``h_i (h below - h above)`` in the order of ``keys`` (primary first).

    Pairs of rows sharing a label are left out. Rows sharing a label must share
    every key, so they sort into one run.
    """
    order = np.lexsort(tuple([labels] + list(keys[::-1])), axis=1)
    hs = _take(h, order)
    _, _, s_pos, e_pos = _runs(_take(labels, order))
    c = np.cumsum(hs, axis=1)
    below = _take(c, s_pos) - _take(hs, s_pos)
    above = c[:, -1:] - _take(c, e_pos)
    a = np.empty_like(h)
    np.put_along_axis(a, order, hs * (below - above), axis=1)
    return a


def tie_rows(labels, m):
    """The ``m`` (row, root) tie pairs of a labelling: rows whose label is not themselves."""
    n = labels.shape[1]
    nonroot = labels != np.arange(n)[None, :]
    if not np.all(nonroot.sum(axis=1) == m):
        raise ValueError(f"labelling does not carry {m} tie equations")
    i = np.argsort(~nonroot, axis=1, kind="stable")[:, :m]
    return i, _take(labels, i)


def vertex(y, X, labels):
    """``beta`` at the vertex a labelling defines, with its tie matrix and tie pairs."""
    p = X.shape[2]
    i, j = tie_rows(labels, p)
    T = _take(X, i[..., None]) - _take(X, j[..., None])
    beta = np.linalg.solve(T, (_take(y, i) - _take(y, j))[..., None])[..., 0]
    return beta, T, i, j


def certificate(h, labels, psi):
    """Largest ``psi(U) - h(U) h(G \\ U)`` over prefixes of each group, and its group and prefix."""
    K, n = h.shape
    ratio = psi / np.where(h > 0, h, 1.0)
    order = np.lexsort((-ratio, labels), axis=1)
    ls, ps, hs = _take(labels, order), _take(psi, order), _take(h, order)
    idx, last, s_pos, e_pos = _runs(ls)
    cp, ch = np.cumsum(ps, axis=1), np.cumsum(hs, axis=1)
    base_p = _take(cp, s_pos) - _take(ps, s_pos)
    base_h = _take(ch, s_pos) - _take(hs, s_pos)
    P, Hk, hG = cp - base_p, ch - base_h, _take(ch, e_pos) - base_h
    V = np.where(last, -np.inf, P - Hk * (hG - Hk))
    pos = np.argmax(V, axis=1)
    rk = np.arange(K)
    g = ls[rk, pos]
    U = np.zeros((K, n), bool)
    np.put_along_axis(U, order, (ls == g[:, None]) & (idx <= pos[:, None]), axis=1)
    return V[rk, pos], g, U


def crash(y, X, h):
    """A starting vertex: weighted least squares, then the tightest ``p + 1`` residuals as one group.

    ``sum_{i<j} h_i h_j (e_i - e_j)^2 = h(.) sum_i h_i (e_i - e_bar)^2`` with
    ``e_bar`` the h-weighted mean, so the least-squares start is pair-free too.
    """
    K, n, p = X.shape
    w = h / h.sum(axis=1, keepdims=True)
    Xc = X - np.einsum("kn,knp->kp", w, X)[:, None, :]
    yc = y - (w * y).sum(axis=1, keepdims=True)
    A = np.einsum("knp,kn,knq->kpq", Xc, h, Xc) + 1e-12 * np.eye(p)
    beta = np.linalg.solve(A, np.einsum("knp,kn,kn->kp", Xc, h, yc)[..., None])[..., 0]
    e = y - np.einsum("knp,kp->kn", X, beta)
    order = np.argsort(e, axis=1)
    es = _take(e, order)
    st = np.argmin(es[:, p:] - es[:, :-p], axis=1)
    rows = _take(order, st[:, None] + np.arange(p + 1)[None, :])
    D = _take(X, rows[:, 1:, None]) - _take(X, rows[:, :1, None])
    with np.errstate(all="ignore"):
        bad = ~(np.linalg.cond(D) < 1e10)
    for k in np.flatnonzero(bad):
        # Repeated rows of X make the window's tie system singular: take rows
        # nearest the window's centre that raise the rank instead.
        centre = es[k, st[k] + p // 2]
        pick, Q = [], np.zeros((0, p))
        for r in np.argsort(np.abs(e[k] - centre), kind="stable"):
            if not pick:
                pick.append(r)
                continue
            d = X[k, r] - X[k, pick[0]]
            v = d - Q.T @ (Q @ d)
            if np.linalg.norm(v) > 1e-8 * max(1.0, np.linalg.norm(d)):
                pick.append(r)
                Q = np.vstack([Q, v / np.linalg.norm(v)])
                if len(pick) == p + 1:
                    break
        rows[k] = pick
    labels = np.broadcast_to(np.arange(n), (K, n)).copy()
    np.put_along_axis(labels, rows, np.repeat(rows.min(axis=1)[:, None], p + 1, axis=1), axis=1)
    return labels


def _long_step_pairs(e, u, ld, h, Fp):
    """Every breakpoint of the step in one transient (k, n, n) array; the entering pair."""
    k, n = e.shape
    app = (e[:, :, None] < e[:, None, :]) & (u[:, :, None] > u[:, None, :]) & (ld[:, :, None] != ld[:, None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(app, (e[:, None, :] - e[:, :, None]) / (u[:, :, None] - u[:, None, :]), np.inf)
    inc = np.where(app, 2 * h[:, :, None] * h[:, None, :] * (u[:, :, None] - u[:, None, :]), 0.0)
    o = np.argsort(tt.reshape(k, -1), axis=1, kind="stable")
    slope = Fp[:, None] + np.cumsum(_take(inc.reshape(k, -1), o), axis=1)
    hit = np.argmax(slope >= -TOL, axis=1)
    if not np.all(slope[np.arange(k), hit] >= -TOL):          # pragma: no cover - LAD is bounded below
        raise RuntimeError("unbounded line search")
    flat = o[np.arange(k), hit]
    return np.stack([flat // n, flat % n], axis=1), 1


def _long_step_kinetic(e, u, ld, h, Fp):
    """The same step with O(n) memory: swap adjacent periods as their residuals cross."""
    k, n = e.shape
    order = np.lexsort((ld, u, e), axis=1)
    live = np.ones(k, bool)
    enter = np.zeros((k, 2), int)
    rounds = 0
    while live.any():
        rounds += 1
        L = np.flatnonzero(live)
        o = order[L]
        lo, hi = o[:, :-1], o[:, 1:]
        uL, eL, lL, hL = u[L], e[L], ld[L], h[L]
        ulo, uhi = _take(uL, lo), _take(uL, hi)
        ok = (_take(lL, lo) != _take(lL, hi)) & (ulo > uhi)
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = np.where(ok, (_take(eL, hi) - _take(eL, lo)) / (ulo - uhi), np.inf)
        q = np.argmin(tc, axis=1)
        rr = np.arange(L.size)
        if not np.all(np.isfinite(tc[rr, q])):               # pragma: no cover - LAD is bounded below
            raise RuntimeError("unbounded line search")
        a, b = lo[rr, q], hi[rr, q]
        Fp[L] += 2 * hL[rr, a] * hL[rr, b] * (ulo[rr, q] - uhi[rr, q])
        stop = Fp[L] >= -TOL
        enter[L[stop]] = np.stack([a[stop], b[stop]], axis=1)
        sw, qs = L[~stop], q[~stop]
        order[sw, qs], order[sw, qs + 1] = order[sw, qs + 1], order[sw, qs].copy()
        live[L[stop]] = False
    return enter, rounds


#: Relative size of the row perturbation the pivots run on.
EPS_PERTURB = 1e-9
#: Problems whose final vertex failed the certificate on the unperturbed data.
UNCERTIFIED = [0]


def perturbation(y):
    """A deterministic, row-specific shift of size ``EPS_PERTURB * max|y|``.

    Ties in the data put zero-residual pairs outside the basis, a degenerate
    vertex on which the rank scores order tied periods arbitrarily. Pivoting on
    ``y`` plus this shift removes those ties; the perturbation argument
    (Luenberger and Ye 2021, Exercises 4.15-4.17) makes a basis optimal for the
    perturbed problem optimal for the original once the shift is small, and
    ``certify`` checks that on the original data.
    """
    n = y.shape[-1]
    xi = np.modf(np.sin((np.arange(n) + 1) * 12.9898) * 43758.5453)[0]
    return EPS_PERTURB * np.maximum(1.0, np.abs(y).max(axis=-1, keepdims=True)) * xi


def dual_flows(y, X, h, labels):
    """The net flows ``phi = a + psi`` at a vertex: a dual solution of the pairwise programme.

    ``a`` realises flows ``h_i h_j sign(e_i - e_j)`` on pairs across groups and
    ``psi`` the flows inside groups, so ``phi`` is dual feasible whenever the
    vertex passes the prefix-cut test, and ``X' phi = 0`` by construction.
    """
    K, n, p = X.shape
    beta, T, ti, tj = vertex(y, X, labels)
    e = _take(y - np.einsum("knp,kp->kn", X, beta), labels)
    a = rank_scores(h, labels, e)
    theta = np.linalg.solve(np.transpose(T, (0, 2, 1)),
                            -np.einsum("knp,kn->kp", X, a)[..., None])[..., 0]
    psi = np.zeros((K, n))
    rows = np.broadcast_to(np.arange(K)[:, None], ti.shape)
    np.add.at(psi, (rows, ti), theta)
    np.add.at(psi, (rows, tj), -theta)
    return a + psi


def pairwise_objective(y, X, h, beta):
    """``sum_{i<j} h_i h_j |e_i - e_j|`` from one sort; ties contribute nothing either way."""
    K, n = y.shape
    e = y - np.einsum("knp,kp->kn", X, beta)
    return (rank_scores(h, np.broadcast_to(np.arange(n), (K, n)).copy(), e) * e).sum(axis=1)


def certify(y, X, h, beta, phi):
    """Weak duality on the unperturbed data: ``F(beta) - phi'e(beta)`` and ``X'phi`` both vanish.

    For dual-feasible ``phi`` (net flows of pair flows within ``[-h_i h_j, h_i h_j]``),
    ``F(b) >= phi'(y - X b)`` for every ``b``; with ``X' phi = 0`` the right side
    is the same number for every ``b``, so a zero gap proves ``beta`` optimal.
    The dual constraints do not involve ``y``, so the dual found on the perturbed
    data is feasible here, and ties in the data cannot fool the test.
    """
    F = pairwise_objective(y, X, h, beta)
    e = y - np.einsum("knp,kp->kn", X, beta)
    gap = F - (phi * e).sum(axis=1)
    resid = np.abs(np.einsum("knp,kn->kp", X, phi)).max(axis=1)
    scale = np.maximum(1.0, np.abs(F))
    xscale = np.maximum(1.0, np.abs(X).max(axis=(1, 2)) * np.abs(phi).sum(axis=1))
    return (gap <= 1e-9 * scale) & (resid <= 1e-9 * xscale)


def solve(y, X, h, labels, line_search: str = "pairs", max_pivots: int = 1000,
          return_certified: bool = False):
    """Exact fits of ``K`` problems from vertex labellings.

    The pivots run on ``y + perturbation(y)``; ``beta`` is then the final
    vertex evaluated on ``y`` itself and checked by ``certify``. Returns
    ``(beta, labels, pivots, rounds)``, the slopes, the optimal labellings (a
    warm start for the next window), pivots per problem and the number of
    vectorised line-search passes, with the certificate as a fifth element when
    ``return_certified``. ``UNCERTIFIED`` counts problems that fail it.
    """
    y = np.asarray(y, dtype=float)
    yp = y + perturbation(y)
    _, labels, pivots, rounds = _pivot(yp, X, h, labels, line_search, max_pivots)
    beta = vertex(y, X, labels)[0]
    cert = certify(y, X, h, beta, dual_flows(yp, X, h, labels))
    UNCERTIFIED[0] += int((~cert).sum())
    if return_certified:
        return beta, labels, pivots, rounds, cert
    return beta, labels, pivots, rounds


def _pivot(y, X, h, labels, line_search: str = "pairs", max_pivots: int = 1000):
    """The pivot loop of ``solve``, on whatever ``y`` it is given."""
    K, n, p = X.shape
    step = {"pairs": _long_step_pairs, "kinetic": _long_step_kinetic}[line_search]
    labels = labels.copy()
    pivots = np.zeros(K, int)
    beta_out = np.zeros((K, p))
    active = np.arange(K)
    rounds = 0
    for _ in range(max_pivots + 1):
        if active.size == 0:
            return beta_out, labels, pivots, rounds
        ya, Xa, ha, la = y[active], X[active], h[active], labels[active]
        k = active.size
        beta, T, ti, tj = vertex(ya, Xa, la)
        e = _take(ya - np.einsum("knp,kp->kn", Xa, beta), la)          # exact ties within groups
        a_out = rank_scores(ha, la, e)
        theta = np.linalg.solve(np.transpose(T, (0, 2, 1)),
                                -np.einsum("knp,kn->kp", Xa, a_out)[..., None])[..., 0]
        psi = np.zeros((k, n))
        rows = np.broadcast_to(np.arange(k)[:, None], ti.shape)
        np.add.at(psi, (rows, ti), theta)
        np.add.at(psi, (rows, tj), -theta)
        viol, g, U = certificate(ha, la, psi)
        done = viol <= TOL * np.maximum(1.0, ha.sum(axis=1) ** 2 / 4)
        beta_out[active[done]] = beta[done]
        active, keep = active[~done], ~done
        if not keep.any():
            continue
        Xa, ha, la, e, U, g = Xa[keep], ha[keep], la[keep], e[keep], U[keep], g[keep]
        k = active.size
        rk = np.arange(k)
        W = (la == g[:, None]) & ~U
        rootU, rootW = np.argmax(U, axis=1), np.argmax(W, axis=1)
        ld = la.copy()
        ld[U] = np.repeat(rootU, U.sum(axis=1))
        ld[W] = np.repeat(rootW, W.sum(axis=1))
        i, j = tie_rows(ld, p - 1)
        Td = np.concatenate([_take(Xa, i[..., None]) - _take(Xa, j[..., None]),
                             (Xa[rk, rootU] - Xa[rk, rootW])[:, None, :]], axis=1)
        rhs = np.zeros((k, p, 1))
        rhs[:, -1, 0] = -1.0                                           # U rises against G \ U
        u = _take(-np.einsum("knp,kp->kn", Xa, np.linalg.solve(Td, rhs)[..., 0]), ld)
        Fp = (rank_scores(ha, ld, e, u) * u).sum(axis=1)
        enter, r = step(e, u, ld, ha, Fp)
        rounds += r
        la_, lb_ = ld[rk, enter[:, 0]], ld[rk, enter[:, 1]]
        merge = (ld == la_[:, None]) | (ld == lb_[:, None])
        labels[active] = np.where(merge, np.minimum(la_, lb_)[:, None], ld)
        pivots[active] += 1
    raise RuntimeError("pivot limit")                                  # pragma: no cover


# ----------------------------------------------------------------- measurement


def _models(case, reps, T1=60, T2=30, N=25):
    from . import oracle as O
    from . import simulate as S

    out = defaultdict(list)
    for rep in range(reps):
        rng = np.random.default_rng(S.seed_for(case, T1, N, rep))
        y0, X = S.draw(case, T1, T2, N, rng)
        order, _, _ = O.screen(y0[:T1], X[:T1])
        for groups in O.candidate_sets(order, X.shape[1], S.M_RANGE).values():
            for g in groups:
                h, _ = O.mallows_or_ones(X[:T1][:, g], rng)
                out[len(g)].append((y0[:T1], X[:T1][:, g], h))
    return out


def measure(case, reps, line_search, check, T1=60):
    """Every window of every model of ``reps`` replications, batched by group size."""
    from . import oracle as O
    from .highs_reference import wlad

    solves, piv, rounds, t_batch, t_highs, worst = 0, [], 0, 0.0, 0.0, -np.inf
    for p, specs in _models(case, reps, T1).items():
        Y, X, H = (np.stack([s[k] for s in specs]) for k in range(3))
        labels = None
        for n in range(T1 // 2, T1 + 1):
            y, Xn, h = Y[:, :n], X[:, :n], H[:, :n]
            t0 = time.perf_counter()
            lab0 = crash(y, Xn, h) if labels is None else np.concatenate(
                [labels, np.full((len(specs), 1), n - 1)], axis=1)
            beta, labels, pv, r = solve(y, Xn, h, lab0, line_search)
            t_batch += time.perf_counter() - t0
            solves += len(specs); piv.append(pv); rounds += r
            if check:
                t0 = time.perf_counter()
                i, j = O.pairs(n)
                for k in range(len(specs)):
                    w, D, rr = h[k, i] * h[k, j], Xn[k, i] - Xn[k, j], y[k, i] - y[k, j]
                    ref = wlad(rr, D, w)
                    f = lambda b: float(np.sum(w * np.abs(rr - D @ b)))
                    worst = max(worst, (f(beta[k]) - f(ref)) / f(ref))
                t_highs += time.perf_counter() - t0
    piv = np.concatenate(piv)
    return dict(reps=reps, line_search=line_search, solves=solves, pivots_mean=float(piv.mean()),
                pivots_max=int(piv.max()), rounds=rounds, seconds_per_rep=t_batch / reps,
                highs_seconds_per_rep=(t_highs / reps if check else None),
                worst_gap_vs_highs=(float(worst) if check else None))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 8])
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    from . import simulate as S

    rows = []
    for reps in args.reps:
        for ls in ("pairs", "kinetic"):
            for case in S.CASES:
                row = dict(case=case, **measure(case, reps, ls, check=(reps == 1)))
                rows.append(row)
                extra = (f" | HiGHS {row['highs_seconds_per_rep']:.2f}s/rep | worst gap "
                         f"{row['worst_gap_vs_highs']:+.1e}") if reps == 1 else ""
                print(f"case {case}, {reps} rep(s), {ls:7s}: {row['solves']} solves | pivots "
                      f"{row['pivots_mean']:.1f} (max {row['pivots_max']}) | passes {row['rounds']} | "
                      f"{row['seconds_per_rep']:.2f}s/rep{extra}")
    OUT.write_text(json.dumps(dict(python=platform.python_version(), rows=rows), indent=1) + "\n")
    return rows


if __name__ == "__main__":
    main()
