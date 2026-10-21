"""Checks of the study's search against brute force and its own invariants."""
from itertools import combinations
from math import comb

import numpy as np
import pytest

from benchmarks.studies.syndes_per_unit_exact.search import (
    Fits, _fresh_price, brute_force, ordered_subsets, search)
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda


def _panel(N, T=12, seed=0, rank=3, noise=0.3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


@pytest.mark.parametrize("n,K", [(6, 1), (7, 3), (8, 4), (9, 8)])
def test_ordered_subsets_visits_every_set_once_in_order(n, K):
    a = np.sort(np.random.default_rng(n * K).uniform(size=n))
    out = list(ordered_subsets(a, K))
    sets = [p for _, p in out]
    assert len(sets) == comb(n, K) and len(set(sets)) == len(sets)
    sums = [a[list(p)].sum() for p in sets]
    assert all(x <= y + 1e-12 for x, y in zip(sums, sums[1:]))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("K", [2, 3])
def test_the_search_finds_the_brute_force_optimum(seed, K):
    Y = _panel(8, seed=seed)
    lam = estimate_lambda(Y)
    out = search(Y, K, lam)
    S, value = brute_force(Y, K, lam)
    assert out["closed"] and out["design"] == S
    assert out["value"] == pytest.approx(value, rel=1e-9)


def test_reused_prices_equal_fresh_ones():
    Y = _panel(9, seed=5)
    lam = estimate_lambda(Y)
    fits = Fits(Y, lam)
    for S in combinations(range(9), 3):
        assert fits.price(S) == pytest.approx(_fresh_price(Y, S, lam), rel=1e-9)


def test_the_root_bound_is_below_every_set():
    Y = _panel(9, seed=6)
    lam = estimate_lambda(Y)
    fits = Fits(Y, lam)
    for S in combinations(range(9), 3):
        assert fits.root_value[list(S)].mean() <= fits.price(S) * (1 + 1e-12)


def test_the_search_matches_the_library_mip():
    pytest.importorskip("pyscipopt")
    from benchmarks.studies.syndes_per_unit_exact.search import scip
    Y = _panel(8, seed=2)
    lam = estimate_lambda(Y)
    mip = scip(Y, 3, lam, time_limit=None, gap_limit=0.0)
    out = search(Y, 3, lam)
    assert mip["status"] == "optimal" and mip["design"] == out["design"]
    assert out["value"] <= mip["value"] * (1 + 1e-7)


def test_the_pair_bound_sits_between_the_root_bound_and_the_price():
    Y = _panel(9, seed=7)
    lam = estimate_lambda(Y)
    fits = Fits(Y, lam)
    for S in combinations(range(9), 3):
        removed = frozenset(S)
        pair = sum(fits.pair_term(i, removed) for i in S) / 3
        assert fits.root_value[list(S)].mean() <= pair * (1 + 1e-12)
        assert pair <= fits.price(S) * (1 + 1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_both_variants_find_the_optimum(seed):
    Y = _panel(8, seed=10 + seed)
    lam = estimate_lambda(Y)
    S, value = brute_force(Y, 3, lam)
    for pair in (True, False):
        out = search(Y, 3, lam, pair=pair)
        assert out["closed"] and out["design"] == S


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("K", [2, 3, 5])
def test_the_branch_and_bound_finds_the_brute_force_optimum(seed, K):
    from benchmarks.studies.syndes_per_unit_exact.search import branch_and_bound
    Y = _panel(8, seed=20 + seed)
    lam = estimate_lambda(Y)
    S, value = brute_force(Y, K, lam)
    out = branch_and_bound(Y, K, lam)
    assert out["closed"] and out["design"] == S
    assert out["value"] == pytest.approx(value, rel=1e-9)
