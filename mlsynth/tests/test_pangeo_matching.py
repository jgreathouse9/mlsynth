"""Minimum-weight matching by Edmonds' blossom algorithm (matching.py): the pairs, the market left out when N is
odd, and the dual certificate that proves the pairing optimal."""
from fractions import Fraction
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from mlsynth.utils.pangeo_helpers import matching


def _brute(W):
    """The least cost over every matching that leaves at most one market out (exactly one when N is odd), and the
    markets that can be the one left out at that cost; recursion on the lowest market."""
    n = len(W)
    memo = {}
    def best(U):
        if len(U) == 0:
            return 0.0
        if U in memo:
            return memo[U]
        i = min(U); rest = U - {i}
        out = min(W[i][j] + best(rest - {j}) for j in rest)
        memo[U] = out
        return out
    full = frozenset(range(n))
    if n % 2 == 0:
        return best(full), None
    costs = {d: best(full - {d}) for d in range(n)}
    m = min(costs.values())
    return m, {d for d, c in costs.items() if c == m}


def _check_structure(out, n):
    pairs = out["pairs"]
    seen = sorted(v for p in pairs for v in p)
    assert all(i < j for i, j in pairs)
    if n % 2 == 0:
        assert seen == list(range(n)) and out["unmatched"] is None
    else:
        assert out["unmatched"] not in seen and sorted(seen + [out["unmatched"]]) == list(range(n))
    assert len(pairs) == n // 2


def _sym(rng, n, lo, hi, integer=True):
    A = rng.integers(lo, hi, (n, n)).astype(float) if integer else rng.uniform(lo, hi, (n, n))
    W = np.triu(A, 1); W = W + W.T
    np.fill_diagonal(W, rng.uniform(-1e9, 1e9))            # the diagonal is never a pair and must be ignored
    return W


# --- smoke and the scalar closed form ----------------------------------------------------------------------------------

def test_four_markets_pair_as_neighbours():
    """Smoke: totals 1, 2, 10, 11 with the cost (z_a - z_b)^2 pair as (1, 2) and (10, 11), cost 2."""
    z = np.array([10.0, 1.0, 11.0, 2.0])
    out = matching.min_weight_matching((z[:, None] - z[None, :]) ** 2)
    assert sorted(out["pairs"]) == [(0, 2), (1, 3)] and out["cost"] == 2.0 and out["unmatched"] is None


@pytest.mark.parametrize("n", [2, 10, 51, 174, 400])
def test_the_scalar_objective_matches_the_sorted_pairing(n):
    """Unit: with the cost (z_a - z_b)^2 the sorted pairing is optimal (an exchange argument); for odd N the best
    market to leave out is found by a scan of the sorted order (prefix and suffix sums of neighbour costs at even
    positions). The matcher's cost equals that, to rounding."""
    z = np.sort(np.random.default_rng(n).lognormal(0, 1.3, n)) * 1000
    z = z[np.random.default_rng(n + 1).permutation(n)]
    out = matching.min_weight_matching((z[:, None] - z[None, :]) ** 2)
    s = np.sort(z)
    if n % 2 == 0:
        want = float(((s[1::2] - s[0::2]) ** 2).sum())
    else:
        want = min(float(((s[:k][1::2] - s[:k][0::2]) ** 2).sum() + ((s[k + 1:][1::2] - s[k + 1:][0::2]) ** 2).sum())
                   for k in range(0, n, 2))
    assert out["cost"] == pytest.approx(want, rel=1e-12)
    _check_structure(out, n)


# --- edge cases ----------------------------------------------------------------------------------------------------

def test_no_markets():
    out = matching.min_weight_matching(np.zeros((0, 0)))
    assert out["pairs"] == [] and out["unmatched"] is None and out["cost"] == 0.0


def test_one_market_is_left_out():
    out = matching.min_weight_matching(np.zeros((1, 1)))
    assert out["pairs"] == [] and out["unmatched"] == 0 and out["cost"] == 0.0


def test_two_markets_form_the_one_pair():
    out = matching.min_weight_matching(np.array([[0.0, 7.5], [7.5, 0.0]]))
    assert out["pairs"] == [(0, 1)] and out["cost"] == 7.5


@pytest.mark.parametrize("n", [6, 7, 12])
def test_all_costs_equal(n):
    """Edge: every pairing ties; any perfect (or near-perfect) matching is optimal and the certificate holds."""
    W = np.full((n, n), 3.0)
    out = matching.min_weight_matching(W)
    _check_structure(out, n)
    assert out["cost"] == 3.0 * (n // 2)
    assert matching.check_certificate(W, out) == 0


def test_negative_costs_are_allowed():
    """Every perfect matching has the same number of pairs, so the costs need not be positive."""
    rng = np.random.default_rng(3)
    W = _sym(rng, 8, -50, 50)
    want, _ = _brute(W)
    assert matching.min_weight_matching(W)["cost"] == want


# --- against brute force, over the whole domain ------------------------------------------------------------------------

@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(0, 11), lo=st.integers(-20, 5), span=st.integers(1, 40))
def test_integer_costs_against_brute_force_with_an_exact_certificate(seed, n, lo, span):
    """Property, over N from 0 to 11 (even and odd) and integer costs with many ties (spans as small as 1, negative
    costs included): the matcher's cost is the brute-force optimum; when N is odd the market left out is one of the
    brute force's best; and its dual certificate holds exactly (dual feasible, objective equal to the cost)."""
    rng = np.random.default_rng(seed)
    W = _sym(rng, n, lo, lo + span)
    want, drops = _brute(W)
    out = matching.min_weight_matching(W)
    _check_structure(out, n)
    assert out["cost"] == want
    if n % 2:
        assert out["unmatched"] in drops
    assert matching.check_certificate(W, out) == 0


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(2, 10), scale=st.sampled_from([1e-6, 1.0, 1e6, 1e12]))
def test_real_costs_against_brute_force(seed, n, scale):
    """Property, over real-valued costs at scales from 1e-6 to 1e12: the cost equals brute force to rounding and the
    certificate's largest violation is within rounding of the costs' scale."""
    rng = np.random.default_rng(seed)
    W = _sym(rng, n, 0.0, scale, integer=False)
    want, _ = _brute(W)
    out = matching.min_weight_matching(W)
    _check_structure(out, n)
    assert out["cost"] == pytest.approx(want, rel=1e-9, abs=1e-12 * scale)
    assert matching.check_certificate(W, out) <= 1e-9 * scale * n


@pytest.mark.parametrize("seed", range(6))
def test_against_networkx(seed):
    """Cross-check, where networkx is installed: its Edmonds implementation (min_weight_matching on the complete
    graph) gives the same cost on random 40 and 60-market instances with integer costs."""
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(seed)
    n = 40 + 20 * (seed % 2)
    W = _sym(rng, n, 0, 1000)
    G = nx.Graph()
    G.add_weighted_edges_from((i, j, W[i, j]) for i in range(n) for j in range(i + 1, n))
    want = sum(W[a, b] for a, b in nx.min_weight_matching(G))
    out = matching.min_weight_matching(W)
    assert out["cost"] == want and matching.check_certificate(W, out) == 0


# --- the result checks itself --------------------------------------------------------------------------------------

def test_a_wrong_matching_is_caught(monkeypatch):
    """Failure: the matcher checks its own certificate before returning. A solver that returns a feasible but
    worse pairing (with the duals of the right one) makes the call raise, not return it."""
    real = matching._solve
    def worse(Wp, eps):
        out = list(real(Wp, eps))
        mate = out[0].copy()
        p = [(i, int(mate[i])) for i in range(len(mate)) if i < mate[i]]
        (i1, j1), (i2, j2) = p[0], p[1]                    # swap partners between the first two pairs
        mate[i1], mate[j2], mate[i2], mate[j1] = j2, i1, j1, i2
        out[0] = mate
        return tuple(out)
    monkeypatch.setattr(matching, "_solve", worse)
    W = _sym(np.random.default_rng(5), 8, 0, 100)
    with pytest.raises(RuntimeError, match="certificate"):
        matching.min_weight_matching(W)


def test_the_certificate_rejects_a_wrong_pairing():
    """The checker itself has power: given the optimum's duals and a different pairing it reports a positive gap."""
    W = _sym(np.random.default_rng(6), 8, 0, 100)
    out = matching.min_weight_matching(W)
    p = out["pairs"]
    (a, b), (c, d) = p[0], p[1]
    bad = dict(out, pairs=[tuple(sorted((a, d))), tuple(sorted((c, b)))] + p[2:])
    bad["cost"] = float(sum(W[i, j] for i, j in bad["pairs"]))
    if bad["cost"] == out["cost"]:
        pytest.skip("the swap tied")
    assert matching.check_certificate(W, bad) > 0


# --- invalid input -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("W, what", [
    (np.zeros((3, 4)), "square"),
    (np.zeros(5), "square"),
    (np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
    (np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
    (np.array([[0.0, np.inf], [np.inf, 0.0]]), "finite"),
    (np.array([["a", "b"], ["b", "a"]]), "numbers"),
])
def test_bad_cost_matrices_are_rejected(W, what):
    """Failure: the cost matrix must be a square, symmetric array of finite numbers (off the diagonal); anything
    else raises before any work, naming what is wrong."""
    with pytest.raises(ValueError, match=what):
        matching.min_weight_matching(W)


@pytest.mark.parametrize("seed", [724, 1870])
def test_a_t_blossom_expanded_mid_stage(seed):
    """Unit: two small instances (integer costs 0 to 5, found by a search over seeds) on which a T blossom's dual
    reaches zero inside a stage and it is expanded there. In seed 724 a child off the even path holds a vertex an S
    vertex already reached, which is relabelled T; in seed 1870 an off-path child already carries S. The pairing is
    the brute-force optimum and the certificate is exact."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 13))
    A = rng.integers(0, 6, (n, n)).astype(float)
    W = np.triu(A, 1); W = W + W.T
    out = matching.min_weight_matching(W)
    assert out["cost"] == _brute(W)[0] and matching.check_certificate(W, out) == 0


def test_the_certificate_rejects_a_pairing_that_is_not_a_matching():
    """The checker returns infinity for pairings that are not matchings of the required size: a market used twice,
    too few pairs, a market out of range, an odd count whose left-out market is also paired, and duals of the
    wrong length."""
    W = _sym(np.random.default_rng(7), 6, 0, 10)
    out = matching.min_weight_matching(W)
    (a, b), (c, d), (e, f) = out["pairs"]
    assert matching.check_certificate(W, dict(out, pairs=[(a, b), (a, c), (e, f)])) == float("inf")
    assert matching.check_certificate(W, dict(out, pairs=[(a, b), (c, d)])) == float("inf")
    assert matching.check_certificate(W, dict(out, pairs=[(a, b), (c, d), (e, 9)])) == float("inf")
    assert matching.check_certificate(W, dict(out, p=out["p"][:-1])) == float("inf")
    W5 = _sym(np.random.default_rng(8), 5, 0, 10)
    out5 = matching.min_weight_matching(W5)
    paired = out5["pairs"][0][0]
    assert matching.check_certificate(W5, dict(out5, unmatched=paired)) == float("inf")
    assert matching.check_certificate(W5, dict(out5, unmatched=7)) == float("inf")


# --- the certificate check: exact, and fast where the pairing has slack ---------------------------------------------

def _oracle_check(W, result):
    """The certificate check written directly in exact rational arithmetic, every pair and every odd set: the oracle
    the fast check must equal."""
    W = np.asarray(W, dtype=float)
    n = len(W); m = n + n % 2
    pairs = list(result["pairs"]); covered = [v for pr in pairs for v in pr]
    if len(set(covered)) != len(covered) or len(pairs) != n // 2 or any(not 0 <= v < n for v in covered):
        return float("inf")
    if n % 2 and (result["unmatched"] in covered or not 0 <= result["unmatched"] < n):
        return float("inf")
    p = [Fraction(float(x)) for x in result["p"]]
    if len(p) != m:
        return float("inf")
    worst = Fraction(0); obj = sum(p)
    for members, q in result["blossoms"]:
        worst = max(worst, -Fraction(float(q))); obj -= (len(members) // 2) * Fraction(float(q))
    for i in range(m):
        for j in range(i + 1, m):
            cross = sum((Fraction(float(q)) for mem, q in result["blossoms"] if i in mem and j in mem), Fraction(0))
            w = Fraction(float(W[i, j])) if j < n else Fraction(0)
            worst = max(worst, p[i] + p[j] - cross - w)
    worst = max(worst, abs(obj - sum((Fraction(float(W[i, j])) for i, j in pairs), Fraction(0))))
    return float(worst)


def _laminar(rng, m, k):
    """k random odd sets over range(m) that are pairwise nested or disjoint: each is drawn inside a random earlier
    set (or the whole range) and disjoint from that set's other drawn children, so the family is laminar."""
    fam, kids = [], {None: []}
    for _ in range(k):
        par = None if not fam or rng.random() < 0.3 else int(rng.integers(len(fam)))
        room = sorted((set(range(m)) if par is None else set(fam[par])) - {v for c in kids[par] for v in fam[c]})
        size = len(room) if len(room) % 2 else len(room) - 1
        if size < 3 or (par is not None and size == len(fam[par])):
            continue
        size = int(rng.choice(range(3, size + 1, 2)))
        fam.append(tuple(sorted(int(v) for v in rng.choice(room, size, replace=False))))
        kids[par].append(len(fam) - 1); kids[len(fam) - 1] = []
    return fam


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), m=st.integers(1, 40), k=st.integers(0, 12))
def test_a_laminar_family_is_laid_out_as_intervals(seed, m, k):
    """Unit, over random laminar families: the vertex order puts every set's members in consecutive positions, and the
    interval bounds it reports are those positions."""
    fam = _laminar(np.random.default_rng(seed), m, k)
    order, lo, hi = matching._interval_order(fam, m)
    pos = np.empty(m, np.int64); pos[order] = np.arange(m)
    assert sorted(order.tolist()) == list(range(m))
    for b, mem in enumerate(fam):
        assert sorted(pos[list(mem)].tolist()) == list(range(lo[b], hi[b]))


def test_crossing_sets_are_not_intervals_and_are_reported_so():
    """Edge: {0, 1, 2}, {2, 3, 4} and {0, 4, 5} cross pairwise, so no order makes all three consecutive; the layout says
    so (None) and the check falls back to the direct one."""
    assert matching._interval_order([(0, 1, 2), (2, 3, 4), (0, 4, 5)], 6) is None


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(0, 9), kind=st.sampled_from(["int", "real", "tied"]),
       tamper=st.sampled_from(["none", "p-ulp", "p-shift", "q-shift", "drop-set", "add-set", "add-crossing",
                               "negative-q", "swap-pairs"]))
def test_the_check_equals_the_exact_oracle(seed, n, kind, tamper):
    """Property, over solved instances and certificates tampered with in every way the checker must notice (a dual
    moved by one ulp or by a step, an odd set dropped, added, crossing the others, or given a negative dual, partners
    swapped): the check returns exactly what the direct rational check returns."""
    rng = np.random.default_rng(seed)
    W = _sym(rng, n, 0, 3) if kind == "tied" else _sym(rng, n, -20, 20, integer=(kind == "int"))
    out = matching.min_weight_matching(W)
    m = n + n % 2
    r = dict(out, p=np.array(out["p"], float), blossoms=list(out["blossoms"]), pairs=list(out["pairs"]))
    if m and tamper == "p-ulp":
        i = int(rng.integers(m)); r["p"][i] = np.nextafter(r["p"][i], np.inf if rng.random() < .5 else -np.inf)
    elif m and tamper == "p-shift":
        r["p"][int(rng.integers(m))] += float(rng.choice([-1.0, 0.5, 2.0]))
    elif r["blossoms"] and tamper == "q-shift":
        b = int(rng.integers(len(r["blossoms"]))); mem, q = r["blossoms"][b]; r["blossoms"][b] = (mem, q + 0.25)
    elif r["blossoms"] and tamper == "drop-set":
        r["blossoms"].pop(int(rng.integers(len(r["blossoms"]))))
    elif m >= 3 and tamper in ("add-set", "add-crossing", "negative-q"):
        fam = [mem for mem, _ in r["blossoms"]]
        new = _laminar(rng, m, 3) if tamper != "add-crossing" else [
            tuple(sorted(int(v) for v in rng.choice(m, 3, replace=False))) for _ in range(3)]
        for mem in new:
            if mem not in fam:
                r["blossoms"].append((mem, float(rng.choice([-0.5, 0.5, 1.0])) * (-1 if tamper == "negative-q" else 1)))
    elif len(r["pairs"]) >= 2 and tamper == "swap-pairs":
        (a, b), (c, d) = r["pairs"][:2]; r["pairs"][:2] = [tuple(sorted((a, d))), tuple(sorted((c, b)))]
    assert matching.check_certificate(W, r) == _oracle_check(W, r)


def test_a_violation_hidden_by_rounding_is_found():
    """Unit: the odd set {0, 2, 3} (q = 1/2) puts markets 0 and 2 first in the screen's order, which computes the
    slack of (0, 2) as (W_02 - p_0) - p_2 + q. W_02 - p_0 = 2^53 + 1.25 rounds to 2^53 + 2 (the spacing there is 2),
    so the computed slack is +0.5 while the exact one is -0.25: the pair is violated by 1/4. Every other condition
    holds (the objective included), so the check reports 1/4 only if its rounding bound sends that pair to exact
    arithmetic."""
    B = 2.0 ** 53
    W = np.array([[0, 0, B, 0], [0, 0, 2 * B, 2], [B, 2 * B, 0, B + 2], [0, 2, B + 2, 0]])
    r = dict(pairs=[(0, 1), (2, 3)], unmatched=None, cost=B + 2, p=np.array([-1.25, 1.25, B + 2, 0.5]),
             blossoms=[((0, 2, 3), 0.5)], tolerance=0.0)
    order, lo, hi = matching._interval_order([(0, 2, 3)], 4)
    assert order.tolist()[:2] == [0, 2]
    assert (W[0, 2] - r["p"][0]) - r["p"][2] + 0.5 == 0.5           # the floating slack, as the screen computes it
    assert _oracle_check(W, r) == 0.25
    assert matching.check_certificate(W, r) == 0.25


def test_odd_sets_that_are_not_sets_of_vertices_are_rejected():
    """Failure: an odd set with a repeated vertex, or a vertex outside the m markets (and dummy), is not a set the
    certificate can use; the check returns infinity, the value for a malformed certificate."""
    W = _sym(np.random.default_rng(9), 6, 0, 10)
    out = matching.min_weight_matching(W)
    for bad in [(0, 0, 1), (0, 1, 6), (-1, 0, 1)]:
        assert matching.check_certificate(W, dict(out, blossoms=list(out["blossoms"]) + [(bad, 0.0)])) == float("inf")


def test_only_the_near_tight_pairs_are_settled_exactly():
    """Unit: on 400 markets with the scalar cost (199 odd sets with positive duals), the floating screen clears all but
    the pairs whose slack is within its rounding bound: fewer than 4 per market, of the 79800 pairs."""
    z = np.random.default_rng(400).lognormal(0, 1.3, 400) * 1000
    W = (z[:, None] - z[None, :]) ** 2
    out = matching.min_weight_matching(W)
    Wf = np.zeros((400, 400)); Wf[:] = W
    order, lo, hi = matching._interval_order([mem for mem, _ in out["blossoms"]], 400)
    ii, jj = matching._near_tight(Wf, np.asarray(out["p"], float), [q for _, q in out["blossoms"]], order, lo, hi)
    assert len(out["blossoms"]) > 100 and 200 <= len(ii) < 4 * 400


def _exact_slacks(Wf, p, family, qs):
    """Every pair's dual slack W_ij - p_i - p_j + sum of q over the sets holding both, in rationals."""
    m = len(p)
    F = Fraction
    out = {}
    for i in range(m):
        for j in range(i + 1, m):
            cross = sum((F(q) for mem, q in zip(family, qs) if i in mem and j in mem), F(0))
            out[i, j] = F(float(Wf[i, j])) - F(float(p[i])) - F(float(p[j])) + cross
    return out


def test_a_sum_of_many_duals_that_rounds_up_is_screened_in():
    """Unit: the pair (0, 1) lies in 21 nested odd sets, the largest with q = 1 and twenty with q = 3 * 2^-54 (three
    quarters of an ulp of 1). The screen adds them largest first, and each small one rounds the running sum up by a
    quarter ulp: the computed sum is 1 + 20 ulp against an exact 1 + 15 ulp. With W_01 = -(1 + 16 ulp) the computed
    slack is +4 ulp, more than the rounding bound of one operation, while the exact slack is -1 ulp. The pair must
    reach exact arithmetic, so the screen's bound counts the terms of the sum."""
    ulp = 2.0 ** -52
    m = 44
    family = [tuple(range(s)) for s in range(43, 2, -2)]
    qs = [1.0] + [3 * 2.0 ** -54] * 20
    p = np.zeros(m)
    Wf = np.ones((m, m)); np.fill_diagonal(Wf, 0.0)
    Wf[0, 1] = Wf[1, 0] = -(1 + 16 * ulp)
    order, lo, hi = matching._interval_order(family, m)
    pos = np.empty(m, np.int64); pos[order] = np.arange(m)
    G = np.zeros((m, m)); np.add.at(G, (lo, hi - 1), np.array(qs))
    C = np.cumsum(np.cumsum(G, axis=0)[:, ::-1], axis=1)[:, ::-1]
    a, b = sorted((pos[0], pos[1]))
    assert C[a, b] == 1 + 20 * ulp and Wf[0, 1] + C[a, b] > 2 * 2.0 ** -53 * 3   # the computed slack, past one-op bound
    assert _exact_slacks(Wf, p, family, qs)[0, 1] == -Fraction(ulp)
    ii, jj = matching._near_tight(Wf, p, qs, order, lo, hi)
    assert (0, 1) in {tuple(sorted(e)) for e in zip(ii.tolist(), jj.tolist())}


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), m=st.integers(2, 24), k=st.integers(0, 10),
       scale=st.sampled_from([1.0, 2.0 ** 40, 2.0 ** -40]))
def test_the_screen_returns_every_pair_without_positive_exact_slack(seed, m, k, scale):
    """Property, over laminar families whose duals mix 1 with values near an ulp of 1 (where floating sums round),
    duals p of mixed sign, and costs set to the floating value that makes each pair tight, moved by a few ulps either
    way: every pair whose exact slack is not positive is among the pairs the screen returns."""
    rng = np.random.default_rng(seed)
    family = _laminar(rng, m, k)
    qs = [float(rng.choice([1.0, 3 * 2.0 ** -54, 2.0 ** -53, 0.1, 5 * 2.0 ** -55])) * scale for _ in family]
    p = rng.choice([-1.0, 0.5, 1.0, 1 + 2.0 ** -52, 3 * 2.0 ** -54], m) * scale
    Wf = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            cross = sum(q for mem, q in zip(family, qs) if i in mem and j in mem)
            w = p[i] + p[j] - cross
            for _ in range(int(rng.integers(-3, 4))):
                w = np.nextafter(w, np.inf if rng.random() < 0.5 else -np.inf)
            Wf[i, j] = Wf[j, i] = w
    order, lo, hi = matching._interval_order(family, m)
    ii, jj = matching._near_tight(Wf, p, qs, order, lo, hi)
    got = {tuple(sorted(e)) for e in zip(ii.tolist(), jj.tolist())}
    assert all(e in got for e, s in _exact_slacks(Wf, p, family, qs).items() if s <= 0)


def test_a_crossing_family_is_checked_on_every_pair():
    """Unit: with the crossing sets {0, 1, 2}, {2, 3, 4}, {0, 4, 5} (duals 0) the check cannot lay the sets out and
    evaluates every pair exactly. The pair (0, 1) is violated by 5 (W_01 = -5, all duals 0), the largest violation:
    the objective is off by 3 and every other pair has slack 1."""
    W = np.ones((6, 6)); np.fill_diagonal(W, 0.0)
    W[0, 1] = W[1, 0] = -5.0
    r = dict(pairs=[(0, 1), (2, 3), (4, 5)], unmatched=None, cost=-3.0, p=np.zeros(6),
             blossoms=[((0, 1, 2), 0.0), ((2, 3, 4), 0.0), ((0, 4, 5), 0.0)], tolerance=0.0)
    assert matching._interval_order([mem for mem, _ in r["blossoms"]], 6) is None
    assert _oracle_check(W, r) == 5.0 and matching.check_certificate(W, r) == 5.0
