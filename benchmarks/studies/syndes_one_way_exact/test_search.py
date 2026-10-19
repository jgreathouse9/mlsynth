"""Checks of the one-way study's search against brute force and its bounds."""
from itertools import combinations

import numpy as np
import pytest

from benchmarks.studies.syndes_one_way_exact.search import OneWay, brute_force, search
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda


def _panel(N, T=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
            + 0.3 * rng.standard_normal((T, N)))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("K", [1, 2, 4])
def test_the_search_finds_the_brute_force_optimum(seed, K):
    Y = _panel(8, seed=seed)
    lam = estimate_lambda(Y)
    out = search(Y, K, lam)
    S, value = brute_force(Y, K, lam)
    assert out["design"] == S and out["value"] == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_both_bounds_sit_below_the_price_and_ridge_above_two_way(seed):
    Y = _panel(8, seed=10 + seed)
    inst = OneWay(Y, estimate_lambda(Y))
    sets = np.array(list(combinations(range(8), 3)))
    price = np.array([inst.price(S) for S in sets])
    tw, rg = inst.two_way_bound(sets), inst.ridge_bound(sets)
    assert np.all(tw <= price * (1 + 1e-9)) and np.all(rg <= price * (1 + 1e-9))
    assert np.all(rg >= tw * (1 - 1e-9))


def test_the_price_matches_the_library_objective():
    from mlsynth.utils.syndes_helpers.certificate import design_objective
    from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design
    pytest.importorskip("pyscipopt")
    Y = _panel(7, seed=20)
    Y = Y / Y.std()
    lam = estimate_lambda(Y)
    d = solve_synthetic_design(Y, 2, mode="global_equal_weights", gap_limit=0.0,
                               time_limit=None)
    S = tuple(int(i) for i in d.selected_unit_indices)
    assert OneWay(Y, lam).price(S) == pytest.approx(
        design_objective(Y, "global_equal_weights", d, lam), rel=1e-6)
    assert brute_force(Y, 2, lam)[0] == S
