"""Tests for lexsearch.select_treated_designs (centered-data regime)."""
import itertools
import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.lexsearch import (
    select_treated_designs, _afw_single, _afw_batched,
)


def centered_G(T, M, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, M))
    X = X - X.mean(axis=1, keepdims=True)   # f-uniform centering -> origin = centroid
    return X.T @ X


def brute_topk(G, M, m, K):
    res = []
    for S in itertools.combinations(range(M), m):
        loss, _, _ = _afw_single(G[np.ix_(S, S)], iters=600, tol=1e-14)
        res.append((loss, S))
    res.sort(key=lambda x: x[0])
    return res[:K]


# ---------------------------------------------------------------- solvers
def test_afw_batched_matches_single():
    G = centered_G(12, 14, 1)
    subs = list(itertools.combinations(range(14), 5))[:200]
    arr = np.array(subs)
    Lb = _afw_batched(G[arr[:, :, None], arr[:, None, :]], iters=120)
    for r, S in enumerate(subs[:40]):
        Ls, _, _ = _afw_single(G[np.ix_(S, S)], iters=600, tol=1e-14)
        assert abs(Lb[r] - Ls) < 1e-3


# ----------------------------------------------------- enumeration is exact
@pytest.mark.parametrize("M,m", [(12, 3), (14, 4), (13, 5)])
def test_enumeration_matches_bruteforce_centered(M, m):
    G = centered_G(12, M, 7)
    out = select_treated_designs(G, range(M), m, top_K=5, method="enumerate")
    got = sorted(round(d.loss, 6) for d in out["top_designs"])
    ref = sorted(round(l, 6) for l, _ in brute_topk(G, M, m, 5))
    assert np.allclose(got, ref, atol=1e-3)
    assert out["stats"]["termination"]["status"] == "OPTIMAL"
    assert out["stats"]["search"]["method"] == "enumeration"
    assert len(out["stats"]["solution_pool"]) == len(out["top_designs"]) <= 5


# --------------------------------------------- heuristic is near-optimal
@pytest.mark.parametrize("M,m", [(16, 5), (18, 6)])
def test_heuristic_near_optimal_centered(M, m):
    ratios = []
    for seed in range(5):
        G = centered_G(15, M, 100 + seed)
        h = select_treated_designs(G, range(M), m, top_K=5, method="heuristic",
                                   n_starts=16, random_state=0)
        e = select_treated_designs(G, range(M), m, top_K=5, method="enumerate")
        ratios.append(h["top_designs"][0].loss / (e["top_designs"][0].loss + 1e-15))
    ratios = np.array(ratios)
    assert ratios.mean() < 1.05         # within 5% of optimum on average
    assert ratios.max() < 1.30          # never wildly off


# ------------------------------------------------------------- weights ok
def test_weights_simplex_and_reproduce_loss():
    G = centered_G(12, 14, 3)
    out = select_treated_designs(G, range(14), 4, top_K=3, method="enumerate")
    for d in out["top_designs"]:
        assert abs(d.weights.sum() - 1.0) < 1e-6
        assert (d.weights >= -1e-9).all()
        S = d.indices
        assert abs(float(d.weights @ G[np.ix_(S, S)] @ d.weights) - d.loss) < 1e-6
        assert abs(d.imbalance - np.sqrt(d.loss)) < 1e-9


# ------------------------------------------------------------- budget
def test_budget_respected():
    M, m = 14, 4
    G = centered_G(12, M, 5)
    rng = np.random.default_rng(0)
    costs = rng.uniform(1, 5, size=M)
    budget = 12.0
    for method in ("enumerate", "heuristic"):
        out = select_treated_designs(G, range(M), m, top_K=5, method=method,
                                     unit_costs=costs, budget=budget)
        for d in out["top_designs"]:
            assert costs[d.indices].sum() <= budget + 1e-9


def test_budget_matches_filtered_bruteforce():
    M, m = 13, 4
    G = centered_G(12, M, 8)
    rng = np.random.default_rng(1)
    costs = rng.uniform(1, 5, size=M)
    budget = 11.0
    out = select_treated_designs(G, range(M), m, top_K=5, method="enumerate",
                                 unit_costs=costs, budget=budget)
    ref = []
    for S in itertools.combinations(range(M), m):
        if costs[list(S)].sum() <= budget:
            ref.append(_afw_single(G[np.ix_(S, S)], iters=600, tol=1e-14)[0])
    ref = sorted(ref)[:5]
    got = sorted(d.loss for d in out["top_designs"])
    assert np.allclose(got, [round(r, 6) for r in ref], atol=1e-3)


# ------------------------------------------------------------- dispatch
def test_auto_dispatch():
    G = centered_G(12, 12, 2)
    small = select_treated_designs(G, range(12), 3, top_K=3, method="auto",
                                   enumerate_max=10**9)
    assert small["stats"]["search"]["method"] == "enumeration"
    big = select_treated_designs(G, range(12), 3, top_K=3, method="auto",
                                 enumerate_max=1)
    assert big["stats"]["search"]["method"] == "multistart_local_search"


def test_consensus_reported_and_pool_mirrors_topK():
    G = centered_G(15, 20, 4)
    K = 6
    out = select_treated_designs(G, range(20), 5, top_K=K, method="heuristic",
                                 n_starts=12, random_state=0)
    # pool is exactly the K tuples of size m
    assert len(out["top_designs"]) == K
    assert len(out["stats"]["solution_pool"]) == K
    assert all(len(d.indices) == 5 for d in out["top_designs"])
    # consensus diagnostics present and sane (replaces the MIP gap)
    c = out["stats"]["search"]["consensus"]
    assert c is not None
    assert 0 <= c["starts_reaching_incumbent"] <= c["n_starts"]
    assert 0.0 <= c["consensus_rate"] <= 1.0
    assert c["distinct_local_optima"] >= 1
    assert out["stats"]["termination"]["status"] == "FEASIBLE"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
