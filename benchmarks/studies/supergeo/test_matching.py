"""Minimum-weight matching by Edmonds' blossom algorithm (matching.py): the pairs, the market left out when N is
odd, and the dual certificate that proves the pairing optimal."""
from fractions import Fraction
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from . import matching


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
