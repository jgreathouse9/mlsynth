"""Pair exclusion: keeping k pairs of n markets and leaving the rest out of the experiment.

Two answers to "which k pairs":

* Nested trimming, Trimmed Match Design's (Chen, Longfils and Remy, "Trimmed Match Design for Randomized Paired
  Geo Experiments", arXiv:2105.07060, 2021, section 3): form the full optimal pairing, then drop its worst pair,
  then the next, so the k pairs kept are that pairing's k cheapest (:func:`nested_trim`).
* The cardinality-optimal k pairs, re-paired at each k: the k disjoint pairs of least total cost over every
  k-matching of the n markets.

  - For the supergeo paper's objective (Chen, Doudchenko, Jiang, Stein and Ying, "Supergeo design: generalized
    matching for geographic experiments", arXiv:2301.12044, eq. 3) the cost of a pair is the squared difference of
    two scalar totals. Some optimal k-matching then pairs only neighbours in sorted order. If a pair (a, c) has a
    market b between them, either b is unpaired, and (a, b) costs no more than (a, c); or b is paired with d, and the
    four markets re-pair as sorted neighbours at no more cost (for w <= x <= y <= z, (x - w)^2 + (z - y)^2 is no
    more than either other pairing: the exchange argument of :mod:`mlsynth.utils.pangeo_helpers.pairs`, and Bai,
    "Optimality of Matched-Pair Designs in Randomized Controlled Trials", AER 112(12) (2022), Theorem 1). Each
    exchange lowers the sum of the pairs' spans in sorted rank, so it ends at neighbour pairs. Choosing k disjoint
    neighbour pairs is a dynamic program over the sorted order, f(i, j) = min(f(i - 1, j), f(i - 2, j - 1) +
    (s_i - s_{i-1})^2), which gives every k at once in O(n^2) (:func:`sorted_k_pairs`). This is the closed-form
    case of CLAUDE.md invariant 8.
  - For any other cost, Edmonds' matcher on the n markets plus n - 2k dummy markets, each joined to every market
    at cost 0 and to every other dummy at a cost above any saving (:func:`matched_k_pairs`). A perfect matching of
    the 2n - 2k markets with no dummy-dummy pair is a k-matching of the real ones, and the reverse; the dummy costs
    make every optimal matching one of these. This is the construction behind the cardinality-optimality of
    Edmonds' intermediate matchings (Kaibel and Walter, "A note on matchings constructed during Edmonds' weighted
    perfect matching algorithm", arXiv:1703.09505, 2017; Lawler, "Combinatorial Optimization: Networks and
    Matroids", 1976, chapter 6).

The two are not nested in general: an augmenting path changes pairs already formed, so the best k pairs need not
lie in the full pairing. :func:`design_excluding` builds a design after a stated exclusion, by either method, and
refuses an exclusion that leaves no design.
"""
from __future__ import annotations

import numpy as np

from mlsynth.utils.pangeo_helpers.matching import min_weight_matching

METHODS = ("optimal", "nested")


def _check_int(value, name, lo, hi=None):
    """An integer (a bool is not one) from lo to hi, or ValueError naming it."""
    ok = isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    if not ok or value < lo or (hi is not None and value > hi):
        span = f"from {lo} to {hi}" if hi is not None else f"of at least {lo}"
        raise ValueError(f"{name} must be an integer {span}, not {value!r}")
    return int(value)


def _check_costs(W):
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"the cost matrix must be square, not of shape {W.shape}")
    off = ~np.eye(len(W), dtype=bool)
    if not np.isfinite(W[off]).all():
        raise ValueError("the cost matrix must be finite off the diagonal")
    if not np.array_equal(W[off], W.T[off]):
        raise ValueError("the cost matrix must be symmetric")
    return W


def nested_trim(W, pairs):
    """Trimming one pairing: its pairs sorted by cost (ties in the order given), and the cost of keeping the k
    cheapest, for k = 0 to len(pairs).

    Returns ``(costs, order)``: costs[k] is the sum of the k cheapest pairs' costs W[a, b], and order lists the pairs
    as ``(a, b)`` with a < b, cheapest first. Raises ValueError when a market appears twice or a pair is not of two
    markets of W."""
    W = np.asarray(W, dtype=float)
    pairs = [tuple(sorted((int(a), int(b)))) for a, b in pairs]
    used = [v for p in pairs for v in p]
    if len(set(used)) != len(used):
        raise ValueError("nested trimming starts from a matching: a market appears in two pairs")
    if any(a == b or a < 0 or b >= len(W) for a, b in pairs):
        raise ValueError(f"nested trimming starts from a matching of the {len(W)} markets of the cost matrix")
    c = np.array([W[a, b] for a, b in pairs], dtype=float)
    rank = np.argsort(c, kind="stable")
    costs = np.concatenate(([0.0], np.cumsum(c[rank])))
    return costs, [pairs[r] for r in rank]


def sorted_k_pairs(z):
    """The least-cost k disjoint pairs of the markets with scalar totals z under the cost (z_a - z_b)^2, for every k
    from 0 to n // 2 (the exchange argument and dynamic program of the module docstring).

    Returns ``(costs, pairs)``: costs[k] is the least cost of k pairs, and pairs[k] lists them as ``(a, b)`` with
    a < b, in sorted order of the totals; every pair is two neighbours in that order (equal totals ordered by
    position)."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    m = n // 2
    order = np.lexsort((np.arange(n), z))
    s = z[order]
    f = np.full((n + 1, m + 1), np.inf)
    f[:, 0] = 0.0
    paired = np.zeros((n + 1, m + 1), dtype=bool)
    for i in range(2, n + 1):
        d = s[i - 1] - s[i - 2]
        take = f[i - 2, :m] + d * d
        f[i, 1:] = np.minimum(f[i - 1, 1:], take)
        paired[i, 1:] = take < f[i - 1, 1:]
    out = []
    for k in range(m + 1):
        i, j, got = n, k, []
        while j > 0:
            if paired[i, j]:
                got.append(tuple(sorted((int(order[i - 2]), int(order[i - 1])))))
                i, j = i - 2, j - 1
            else:
                i -= 1
        out.append(got[::-1])
    return f[n].copy(), out


def matched_k_pairs(W, k):
    """The least-cost k disjoint pairs of the markets of the symmetric cost matrix W, by Edmonds' matcher with
    n - 2k dummy markets (the module docstring).

    Returns ``(pairs, cost)``, the pairs as ``(a, b)`` with a < b. Raises ValueError for a cost matrix that is not
    square, symmetric and finite off the diagonal, and for a k that is not an integer from 0 to n // 2."""
    W = _check_costs(W)
    n = len(W)
    k = _check_int(k, "k", 0, n // 2)
    if n == 0:
        return [], 0.0
    d = n - 2 * k
    off = W[~np.eye(n, dtype=bool)]
    # A dummy-dummy pair beside a real pair (a, b) re-pairs as two dummy-real pairs, saving big + W[a, b]; any
    # big above -min W makes that a saving, so no optimal matching keeps a dummy-dummy pair.
    big = float(np.abs(off).max()) + 1.0 if len(off) else 1.0
    M = np.zeros((n + d, n + d))
    M[:n, :n] = W
    M[n:, n:] = big
    pairs = [(a, b) for a, b in min_weight_matching(M)["pairs"] if b < n]
    return pairs, float(sum(W[a, b] for a, b in pairs))


def design_excluding(costs=None, *, totals=None, drop_pairs=0, excluded_units=(), method="optimal", min_pairs=1):
    """A design of pairs after a stated exclusion.

    The markets in ``excluded_units`` are left out by hand. The n' markets left form n' // 2 pairs (one more market
    is left out when n' is odd), and ``drop_pairs`` of those pairs are excluded: by re-pairing the n' markets at
    the least cost of the pairs kept (``method="optimal"``), or by trimming the worst pairs of their full optimal
    pairing (``method="nested"``). Pass the cost matrix of every pair as ``costs``, or the markets' scalar totals
    as ``totals`` for the squared-difference cost, which takes the closed form (:func:`sorted_k_pairs`) for both
    methods.

    Returns a dict: ``pairs`` (kept, as ``(a, b)`` with a < b, sorted), ``cost`` (their total), ``excluded_units``
    (every market in no kept pair, sorted), ``excluded_by_hand``, ``drop_pairs``, ``method``, and
    ``full_pairing``, the optimal pairing of the n' markets that nested trimming trimmed (None for ``"optimal"``).

    Raises ValueError when no design remains: the markets left form no pair, or fewer than ``min_pairs``; the
    exclusion drops every pair, or more pairs than the markets form; or the pairs kept number fewer than
    ``min_pairs``. Also for malformed arguments, each named."""
    if (costs is None) == (totals is None):
        raise ValueError("design_excluding reads one of costs (a cost matrix) or totals (scalar totals), not both")
    if totals is not None:
        z = np.asarray(totals, dtype=float)
        if z.ndim != 1 or not np.isfinite(z).all():
            raise ValueError("totals must be a finite vector, one per market")
        n = len(z)
    else:
        W = _check_costs(costs)
        n = len(W)
    min_pairs = _check_int(min_pairs, "min_pairs", 1)
    drop_pairs = _check_int(drop_pairs, "drop_pairs", 0)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    by_hand = [_check_int(u, "each entry of excluded_units", 0, n - 1) for u in excluded_units]
    if len(set(by_hand)) != len(by_hand):
        raise ValueError("excluded_units names a market twice")

    rest = [i for i in range(n) if i not in set(by_hand)]
    n_rest, available = len(rest), len(rest) // 2
    if available == 0:
        raise ValueError(f"no pair can form: {n_rest} market(s) remain of {n} after excluding {len(by_hand)} "
                         "by hand")
    if available < min_pairs:
        raise ValueError(f"the {n_rest} markets left after excluding {len(by_hand)} of {n} by hand form "
                         f"{available} pairs, fewer than min_pairs={min_pairs}: no design remains")
    if drop_pairs == available:
        raise ValueError(f"excluding {drop_pairs} of {available} pairs excludes every pair: no design remains")
    if drop_pairs > available:
        raise ValueError(f"excluding {drop_pairs} pairs, but the {n_rest} markets form only {available}: "
                         "no pair remains")
    kept = available - drop_pairs
    if kept < min_pairs:
        raise ValueError(f"excluding {drop_pairs} of {available} pairs leaves {kept} pairs, fewer than "
                         f"min_pairs={min_pairs}")

    full = None
    if totals is not None:
        zr = z[rest]
        Wr = (zr[:, None] - zr[None, :]) ** 2
        by_k = sorted_k_pairs(zr)[1]
        if method == "optimal":
            local = by_k[kept]
        else:
            full = by_k[available]
            local = nested_trim(Wr, full)[1][:kept]
    else:
        Wr = W[np.ix_(rest, rest)]
        if method == "optimal":
            local = matched_k_pairs(Wr, kept)[0]
        else:
            full = min_weight_matching(Wr)["pairs"]
            local = nested_trim(Wr, full)[1][:kept]

    def back(ps):
        return sorted(tuple(sorted((rest[a], rest[b]))) for a, b in ps)

    pairs = back(local)
    used = {v for p in pairs for v in p}
    return dict(pairs=pairs, cost=float(sum(Wr[a, b] for a, b in local)),
                excluded_units=[i for i in range(n) if i not in used], excluded_by_hand=sorted(by_hand),
                drop_pairs=drop_pairs, method=method, full_pairing=None if full is None else back(full))
