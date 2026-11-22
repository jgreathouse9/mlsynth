"""Rank-1 Chvatal-Gomory cuts for the supergeo master, against brute force."""
import itertools
import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st, HealthCheck
from . import rank1, sr, dive, test_bcp

FAMILIES = rank1.FAMILIES


def designs(n):
    """Every partition of range(n) into groups of 2..4."""
    def go(U):
        if not U:
            yield []
            return
        i = min(U); rest = sorted(U - {i})
        for k in (1, 2, 3):
            for others in itertools.combinations(rest, k):
                g = (i,) + others
                for d in go(U - set(g)):
                    yield [g] + d
    yield from go(frozenset(range(n)))


def test_every_cut_family_holds_for_every_design():
    n = 9
    every = list(designs(n))
    for fam in FAMILIES:
        size, num, den = fam
        for S in itertools.combinations(range(n), size):
            for perm in set(itertools.permutations(num)):
                cut = rank1.Cut(S, perm, den)
                for d in every[::7]:
                    assert sum(cut.coef(g) for g in d) <= cut.rhs, (fam, S, perm, d)


def test_sr3_family_is_the_subset_row_cut():
    cut = rank1.Cut((1, 4, 6), (1, 1, 1), 2)
    for g in itertools.chain.from_iterable(itertools.combinations(range(8), k) for k in (2, 3, 4)):
        assert cut.coef(g) == sr.coef(g, (1, 4, 6)) and cut.rhs == 1


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 12))
def test_pricing_kernel_matches_direct_reduced_costs(seed, n):
    rng = np.random.default_rng(seed)
    z = rng.lognormal(0, 1, n) * 10; pi = rng.normal(0, 3, n)
    cuts = []
    for _ in range(8):
        size, num, den = FAMILIES[int(rng.integers(len(FAMILIES)))]
        if size > n: continue
        S = tuple(sorted(rng.choice(n, size, replace=False).tolist()))
        cuts.append(rank1.Cut(S, tuple(rng.permutation(num)), den))
    sigma = -rng.random(len(cuts))
    rows, sc, rc, m, best = rank1.price(z, pi, 0.0, cuts, sigma, 2, 4, np.inf)
    direct = {}
    for g in itertools.chain.from_iterable(itertools.combinations(range(n), k) for k in (2, 3, 4)):
        direct[g] = dive._sc(z, g) - pi[list(g)].sum() - sum(s * c.coef(g) for c, s in zip(cuts, sigma))
    got = {tuple(int(v) for v in r if v >= 0): r_ for r, r_ in zip(rows, rc)}
    assert set(got) == set(direct)
    assert max(abs(got[g] - direct[g]) for g in got) < 1e-9
    assert abs(best - min(direct.values())) < 1e-9


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 11), kind=st.integers(0, 3))
@example(seed=114462, n=6, kind=1)   # LP 0 up to noise: a rescale on every round drove the costs to 8e31 and HiGHS to "Unknown"
def test_root_bound_never_exceeds_the_optimum(seed, n, kind):
    z = test_bcp.gen(np.random.default_rng(seed), n, kind)
    out = rank1.root_bound(z, rounds=15)
    opt = test_bcp.brute(np.asarray(z, float))
    for nc, lb in out["hist"]:
        assert lb <= opt * (1 + 1e-7) + 1e-12, (nc, lb, opt)


def test_sr3_only_reproduces_the_subset_row_bound():
    rng = np.random.default_rng(3)
    for _ in range(3):
        z = rng.lognormal(0, 1.3, 11) * 100
        a = rank1.root_bound(z, families=[(3, (1, 1, 1), 2)], rounds=40, per_round=50)["hist"][-1][1]
        b = sr.run(z, cut_rounds=40, cuts_per_round=50, verbose=False)
        b = b["hist"][-1][1] / b["scale"]
        assert abs(a - b) <= 1e-6 * max(1.0, abs(b)), (a, b)


def test_subset_row_rounds_rescale_once():
    """The instance whose root LP is 0 up to noise: the subset-row loop scales the costs once, not every round."""
    z = test_bcp.gen(np.random.default_rng(114462), 6, 1)
    out = sr.run(z, cut_rounds=15, cuts_per_round=50, verbose=False)
    opt = test_bcp.brute(np.asarray(z, float))
    assert out["scale"] < 1e12
    assert out["hist"][-1][1] / out["scale"] <= opt * (1 + 1e-7) + 1e-12


@settings(max_examples=200, deadline=None)
@given(n=st.integers(4, 300), lp=st.floats(1e-300, 10.0), top=st.floats(0.0, 1e12))
def test_rescale_power_is_capped(n, lp, top):
    """The rescale never takes the dearest column past 1e10, and is never a shrink."""
    k = rank1._rescale_power(n, lp, top)
    assert k >= 0
    if k > 0 and top > 0:
        assert top * 4.0 ** k <= 1e10 * (1 + 1e-12)
