"""Au's (2018) TBR climb with every candidate of a step scored at once.

The climb in ``mlsynth.utils.tbr_helpers.design.search`` toggles one geo per
step. In ``_match`` a geo enters or leaves the control group; in ``_augment`` a
geo joins the treatment group and leaves the control group if it was there. So
every candidate of a step is the current pair of aggregates plus or minus one
geo's column, which is the structure ``forward_did_select`` exploits for FDID.

``batched_keys`` forms the candidates' aggregates as the columns of a ``T x M``
matrix and evaluates the reference objective's key on all of them: the centred
sums give the slope, residual scale, correlation and detectable impact; the
residual columns give the Durbin-Watson statistic and the Brownian-bridge
CUSUM; the same sums over the first ``T - n_test`` periods give the A/A test.
Each column is centred directly, as ``groupfit.group_sums`` does for one
candidate, so nothing is updated by differencing.

Exactness. The key compares four gates, a correlation rounded to two decimals
and an inverse impact. A batched key can differ from ``score_split``'s in its
last bits, and that only matters at a decision: a gate threshold, a rounding
boundary, or a near-tie for the best. A candidate within ``BAND`` of any of
these is re-scored by the exact scorer before anything is decided, and so is
every candidate within ``BAND`` of the best, so the step takes the geo the
exact climb takes and records the exact key. ``score_split`` stays the
authority; this module decides only which candidates it needs to see.

The ``"paper"`` objective (CUSUM and Breusch-Godfrey) is not batched here; its
steps run the exact functions.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Sequence, Set

import numpy as np
from scipy import special

from mlsynth.utils.tbr_helpers.design import objective as O
from mlsynth.utils.tbr_helpers.design import search as S

#: Distance from a decision below which a batched key is not trusted. Measured
#: batched-against-exact gaps are below 1e-15 relative.
BAND = 1e-9
CORR_CLIP = 0.999999
#: The rounding of one operation's terms, as agents_solvers.md rule 4 sizes it.
ZERO = 64 * np.finfo(float).eps

EXACT_MATCH = S._match
EXACT_AUGMENT = S._augment


def near_correlation_decision(corr) -> np.ndarray:
    """True where the correlation sits within ``BAND`` of the 0.8 gate, a rounding boundary or the clip."""
    c = np.asarray(corr, dtype=float)
    s = c * 100.0
    to_boundary = np.abs(s - (np.floor(s) + 0.5)) / 100.0
    return ((np.abs(c - O.MIN_CORR) <= BAND) | (to_boundary <= BAND)
            | (np.abs(np.abs(c) - CORR_CLIP) <= BAND))


def near_decision(corr, dw, excess, env_max, loc, half, fpp) -> np.ndarray:
    """True where any term of the key sits within ``BAND`` of the threshold that decides it.

    The thresholds are the reference objective's own: the correlation gate and
    rounding (``near_correlation_decision``), the Durbin-Watson band, the
    Brownian-bridge envelope (``excess`` is the largest cumulative residual
    above it), the A/A interval's zero crossing (``|loc| = half``) and, where
    the interval excludes zero, the false-positive cutoff.
    """
    dw, excess, loc, half, fpp = (np.asarray(v, dtype=float) for v in (dw, excess, loc, half, fpp))
    covers = np.abs(loc) < half
    return (near_correlation_decision(corr)
            | (np.abs(dw - O.DW_RANGE[0]) <= BAND * dw) | (np.abs(dw - O.DW_RANGE[1]) <= BAND * dw)
            | (np.abs(excess) <= BAND * env_max)
            | (np.abs(np.abs(loc) - half) <= BAND * half)
            | (~covers & (np.abs(fpp - O.AA_THRESHOLD_PROB) <= BAND)))


def batched_keys(Y, y, x, cols, a, b, n_test):
    """``(K, flagged)``: the reference key of candidates ``y + a_m Y[:, j_m]``, ``x + b_m Y[:, j_m]``.

    ``K`` is ``(M, 6)``, ordered as ``score_split``'s key. ``flagged`` marks the
    candidates whose key must come from the exact scorer: a degenerate fit, a
    non-finite term, or any quantity within ``BAND`` of the value that decides it.
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    M = len(cols)
    n0 = T - n_test
    if T < 3 or n0 < O.MIN_HOLDOUT_PRETEST:
        return np.full((M, 6), np.nan), np.ones(M, dtype=bool)   # the exact scorer reports the window
    G = Y[:, np.asarray(cols, dtype=int)]
    Yc = y[:, None] + np.asarray(a, dtype=float) * G
    Xc = x[:, None] + np.asarray(b, dtype=float) * G
    const = O.window_constants(T, n_test)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        dY, dX = Yc - Yc.mean(0), Xc - Xc.mean(0)
        sxx, syy, sxy = (dX * dX).sum(0), (dY * dY).sum(0), (dX * dY).sum(0)
        beta = sxy / sxx
        E = dY - beta * dX
        rss = (E * E).sum(0)
        sigma = np.sqrt(rss / (T - 2))
        corr_raw = sxy / np.sqrt(sxx * syy)
        corr = np.clip(corr_raw, -CORR_CLIP, CORR_CLIP)
        sq = np.sqrt(const.phi * (T + 1) / (T * n_test * (T - 1)) + 1.0 / T + 1.0 / n_test)
        impact = (const.tq_sig + const.tq_pow) * n_test * sq * sigma

        d = np.diff(E, axis=0)
        dw = (d * d).sum(0) / rss
        dw_ok = (dw > O.DW_RANGE[0]) & (dw < O.DW_RANGE[1])

        env = O.brownian_bridge_envelope(T)
        excess = (np.abs(np.cumsum(E / sigma, axis=0))[:-1] - env[:, None]).max(axis=0)
        bb_ok = ~(excess > 0)

        Y0, X0 = Yc[:n0], Xc[:n0]
        dY0, dX0 = Y0 - Y0.mean(0), X0 - X0.mean(0)
        sxx0, sxy0 = (dX0 * dX0).sum(0), (dX0 * dY0).sum(0)
        beta0 = sxy0 / sxx0
        alpha0 = Y0.mean(0) - beta0 * X0.mean(0)
        E0 = dY0 - beta0 * dX0
        sigma0 = np.sqrt((E0 * E0).sum(0) / (n0 - 2))
        h = float(n_test)
        xbar_h = Xc[n0:].mean(0)
        loc = h * (Yc[n0:].mean(0) - alpha0 - xbar_h * beta0)
        fitted_var = 1.0 / n0 + (xbar_h - X0.mean(0)) ** 2 / sxx0
        half = const.tq_holdout * h * sigma0 * np.sqrt(fitted_var + 1.0 / h)
        lower, upper = loc - half, loc + half
        covers = lower * upper < 0.0
        true_mean = np.minimum(np.abs(lower), np.abs(upper))
        tq = half / sigma0
        post_scale = sigma0 * np.sqrt(1.0 / n0 + 1.0 / n_test)
        fpp = ((1.0 - special.stdtr(n0 - 2, tq - true_mean / post_scale))
               + special.stdtr(n0 - 2, -tq - true_mean / post_scale))
        aa_ok = covers | (fpp <= O.AA_THRESHOLD_PROB)
        inv = 1.0 / impact

        # A computed zero is a zero only to the rounding of its terms. The
        # aggregates are formed at their level (``x - Y_j`` carries an error of
        # ``eps |x|``) and centring does not remove it, so each sum of squares is
        # compared with the rounding floor of the uncentred terms it came from. A
        # fit that is exact on paper reads 1e-13 here and has to go to the exact
        # scorer, which sees the zero.
        floor = ZERO * ZERO
        tiny_x = sxx <= floor * (Xc * Xc).sum(0)
        tiny_y = syy <= floor * (Yc * Yc).sum(0)
        tiny_r = rss <= floor * ((np.abs(Yc) + np.abs(beta) * np.abs(Xc)) ** 2).sum(0)
        tiny_x0 = sxx0 <= floor * (X0 * X0).sum(0)
        tiny_r0 = (E0 * E0).sum(0) <= floor * ((np.abs(Y0) + np.abs(beta0) * np.abs(X0)) ** 2).sum(0)

    K = np.column_stack([corr >= O.MIN_CORR, aa_ok, bb_ok, dw_ok, np.round(corr, 2), inv]).astype(float)
    finite = np.isfinite(np.column_stack([corr_raw, sigma, dw, excess, loc, half, fpp, inv])).all(axis=1)
    degenerate = tiny_x | tiny_y | tiny_r | tiny_x0 | tiny_r0 | ~finite
    with np.errstate(invalid="ignore"):
        near = near_decision(corr_raw, dw, excess, env.max(), loc, half, fpp)
    return K, degenerate | near


def _select(K, flagged, exact_key: Callable[[int], tuple]) -> int:
    """The index the exact climb would pick: the first maximum of the exact keys."""
    keys = [tuple(float(v) for v in row) for row in K]
    for i in np.flatnonzero(flagged):
        keys[i] = exact_key(int(i))
    best = max(range(len(keys)), key=keys.__getitem__)
    head, top = keys[best][:5], keys[best][5]
    group = [i for i, k in enumerate(keys) if k[:5] == head and abs(k[5] - top) <= BAND * abs(top)]
    for i in group:
        keys[i] = exact_key(i)
    return max(group, key=keys.__getitem__)


def match_step(scorer, treatment: Sequence[int], control: Set[int], toggleable: Sequence[int]):
    """``search._match``, scoring each round's toggles as one batch."""
    if scorer._objective != "reference":
        return EXACT_MATCH(scorer, treatment, control, toggleable)
    Y, n_test = scorer._y, scorer._n_test
    control = set(control)
    trace = [scorer(treatment, control).key]
    y = Y[:, sorted(treatment)].sum(axis=1)
    while True:
        current = scorer(treatment, control)
        cand = [g for g in toggleable if control ^ {g}]
        if not cand:
            return control, trace, True
        x = Y[:, sorted(control)].sum(axis=1)
        b = np.array([-1.0 if g in control else 1.0 for g in cand])
        K, flagged = batched_keys(Y, y, x, cand, np.zeros(len(cand)), b, n_test)
        i = _select(K, flagged, lambda m: scorer(treatment, control ^ {cand[m]}).key)
        best = scorer(treatment, control ^ {cand[i]})
        if best.key <= current.key:
            return control, trace, True
        control = control ^ {cand[i]}
        trace.append(best.key)


def augment_step(scorer, treatment: Sequence[int], control: Set[int], pools, *, from_pool: bool = False):
    """``search._augment``, scoring every treatment-eligible geo at once."""
    if scorer._objective != "reference":
        return EXACT_AUGMENT(scorer, treatment, control, pools, from_pool=from_pool)
    Y, n_test = scorer._y, scorer._n_test
    control_pool = set(pools["control"])
    base = (control_pool - set(treatment)) if from_pool else set(control)
    cands = []
    for geo in pools["treatment"]:
        if geo in treatment:
            continue
        cc = base - {geo}
        if not cc:
            cc = control_pool - set(treatment) - {geo}
        if not cc:
            continue
        cands.append((geo, cc))
    if not cands:
        return EXACT_AUGMENT(scorer, treatment, control, pools, from_pool=from_pool)
    regular = np.array([cc == base - {g} for g, cc in cands])
    geos = [g for g, _ in cands]
    y = Y[:, sorted(treatment)].sum(axis=1)
    x = Y[:, sorted(base)].sum(axis=1) if base else np.zeros(Y.shape[0])
    b = np.array([-1.0 if g in base else 0.0 for g in geos])
    K, flagged = batched_keys(Y, y, x, geos, np.ones(len(geos)), b, n_test)
    flagged = flagged | ~regular                     # a re-derived control group is scored exactly
    i = _select(K, flagged, lambda m: scorer(list(treatment) + [geos[m]], cands[m][1]).key)
    geo, cc = cands[i]
    return list(treatment) + [geo], cc


@contextlib.contextmanager
def batched():
    """Run ``greedy_search`` with the batched steps; the orchestration is the module's own."""
    S._match, S._augment = match_step, augment_step
    try:
        yield
    finally:
        S._match, S._augment = EXACT_MATCH, EXACT_AUGMENT


def batched_greedy_search(*args, **kwargs):
    with batched():
        return S.greedy_search(*args, **kwargs)
