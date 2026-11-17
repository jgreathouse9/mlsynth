"""Pricing's bound pass in one compiled loop (fastprice), and price_subtree's use of it
when numba is installed (#965).

price_subtree bounds every design of a subtree in doubles (_upper_batch), keeps those
whose bound exceeds the incumbent's ratio and evaluates them exactly. The bound pass
built the designs with itertools in chunks and gathered a k x k block of G per design,
about a microsecond a design, nearly all of it the interpreter, and held the
interpreter lock. fastprice.bound_pass walks the combinations in place, compiled and
nogil, and computes each design's bound with _upper_batch's formulas and error terms,
which bound recursive summation in any order. At dyadic data every sum is exact and
the two passes keep the same designs at the same bounds; at any data every design
whose exact ratio exceeds the floor is kept, so pricing's answer is unchanged.
"""
from fractions import Fraction as Q
from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import fastprice, slopefree
from .brute import panel
from .certificate import ExactData
from .slopefree import _floats, _fixed, _ratio, _upper_batch
from .test_pricing import _best, _run

needs_numba = pytest.mark.skipif(not fastprice.AVAILABLE, reason="numba is not installed")

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _parts(c, G, fixes):
    """What price_subtree hands the bound pass: the fixed-in sums and the free block."""
    p = len(c)
    ins = sorted(j for j, v in fixes if v); F = [j for j in range(p) if j not in dict(fixes)]
    cf, Gf = _floats(c, G, {})
    fixed = _fixed(c, G, cf, ins, F)
    return fixed, cf[F], Gf[np.ix_(F, F)], ins, F


def _python_pass(fixed, cfF, GfF, k, floor):
    """_upper_batch over every design, kept above the floor: {combo: bound}."""
    every = list(combinations(range(len(cfF)), k))
    combos = np.array(every, dtype=np.int64).reshape(len(every), k)
    ub = _upper_batch(fixed, cfF, GfF, combos)
    return {tuple(int(x) for x in combos[i]): float(ub[i]) for i in np.flatnonzero(ub > floor)}


def _kernel_pass(fixed, cfF, GfF, k, floor):
    C, ub, total = fastprice.bound_pass(fixed, cfF, GfF, k, floor)
    return {tuple(int(x) for x in C[i]): float(ub[i]) for i in range(len(ub))}, total


def _dyadic(seed, p, n=12):
    """A panel whose c and G are small dyadic rationals: every float sum is exact."""
    rng = np.random.default_rng(seed)
    X = rng.integers(-4, 5, (n, p)) / 4; y = rng.integers(-4, 5, n) / 4
    c = [Q(float(v)) for v in X.T @ y]
    G = np.array([[Q(float(v)) for v in r] for r in X.T @ X], dtype=object)
    return c, G


# ---------------------------------------------------------------- smoke


@needs_numba
def test_smoke_the_kernel_keeps_the_python_pass_designs():
    c, G = _dyadic(1, 12)
    fixed, cfF, GfF, _, F = _parts(c, G, [])
    floor = float(np.median(list(_python_pass(fixed, cfF, GfF, 4, -np.inf).values())))
    got, total = _kernel_pass(fixed, cfF, GfF, 4, floor)
    assert total == comb(12, 4)
    assert got == _python_pass(fixed, cfF, GfF, 4, floor) and got


# ---------------------------------------------------------------- unit


@needs_numba
@pytest.mark.parametrize("fixes", [[], [(0, 1)], [(2, 0), (5, 1)]])
def test_the_kernel_counts_and_orders_every_design(fixes):
    c, G = _dyadic(2, 10)
    fixed, cfF, GfF, ins, F = _parts(c, G, fixes)
    k = 4 - len(ins)
    C, ub, total = fastprice.bound_pass(fixed, cfF, GfF, k, -np.inf)
    assert total == comb(len(F), k)
    keys = [tuple(int(x) for x in r) for r in C]
    assert keys == sorted(keys)                     # lexicographic, as combinations
    assert set(keys) == set(_python_pass(fixed, cfF, GfF, k, -np.inf))   # every design but those of bound -inf


@needs_numba
def test_every_design_above_the_floor_by_its_exact_ratio_is_kept():
    y, X = panel(3, p=11)
    D = ExactData(y, X)
    fixed, cfF, GfF, ins, F = _parts(D.c, D.G, [])
    ratios = sorted(_ratio(D.c, D.G, set(S))[1] for S in combinations(range(11), 4) if _ratio(D.c, D.G, set(S))[0] > 0)
    lam = ratios[len(ratios) // 2]
    got, _ = _kernel_pass(fixed, cfF, GfF, 4, float(lam))
    for S in combinations(range(11), 4):
        cu, r = _ratio(D.c, D.G, set(S))
        if cu > 0 and r > lam:
            assert S in got and Q(got[S]) >= r


@needs_numba
def test_price_subtree_runs_the_kernel_when_numba_is_installed(monkeypatch):
    calls, real = [], fastprice.bound_pass

    def spy(*a):
        calls.append(1)
        return real(*a)
    monkeypatch.setattr(fastprice, "bound_pass", spy)
    y, X = panel(1, p=9)
    D = ExactData(y, X)
    _run(D.c, D.G, 3, [], Q(0))
    assert calls == [1]


@pytest.mark.parametrize("switch", ["kernel-off", "numba-missing"])
def test_price_subtree_runs_the_python_pass_otherwise(monkeypatch, switch):
    if switch == "kernel-off":
        monkeypatch.setattr(slopefree, "PRICE_KERNEL", False)
    else:
        monkeypatch.setattr(fastprice, "AVAILABLE", False)

    def boom(*a):
        raise AssertionError("the kernel ran")
    monkeypatch.setattr(fastprice, "bound_pass", boom)
    y, X = panel(1, p=9)
    D = ExactData(y, X)
    out, inc = _run(D.c, D.G, 3, [], Q(0))
    best = _best(D.c, D.G, 9, 3, [], Q(0))
    assert inc["lam"] == best[0] and out.priced == comb(9, 3)


@needs_numba
@pytest.mark.parametrize("seed,m,fixes", [(1, 2, []), (2, 3, [(0, 1)]), (3, 3, [(1, 0), (4, 1)]), (4, 4, [(2, 1), (3, 1)])])
def test_pricing_on_the_kernel_finds_the_python_pass_answer(monkeypatch, seed, m, fixes):
    y, X = panel(seed, p=9)
    D = ExactData(y, X)
    out_k, inc_k = _run(D.c, D.G, m, fixes, Q(0))
    monkeypatch.setattr(slopefree, "PRICE_KERNEL", False)
    out_p, inc_p = _run(D.c, D.G, m, fixes, Q(0))
    assert (inc_k["lam"], inc_k["C"]) == (inc_p["lam"], inc_p["C"])
    assert out_k.priced == out_p.priced and out_k.exact == out_p.exact


@needs_numba
def test_the_kernel_releases_the_interpreter_lock():
    assert fastprice._bound_pass.targetoptions.get("nogil") is True


# ---------------------------------------------------------------- edges


@needs_numba
def test_no_free_control_to_choose_is_one_design():
    """k = 0: the subtree is the one design of the controls fixed in."""
    c, G = _dyadic(5, 8)
    fixed, cfF, GfF, ins, F = _parts(c, G, [(0, 1), (3, 1)])
    got, total = _kernel_pass(fixed, cfF, GfF, 0, -np.inf)
    assert total == 1 and list(got) == [()]
    assert got == _python_pass(fixed, cfF, GfF, 0, -np.inf)


@needs_numba
def test_every_free_control_chosen_is_one_design():
    c, G = _dyadic(6, 7)
    fixed, cfF, GfF, ins, F = _parts(c, G, [(1, 0)])
    got, total = _kernel_pass(fixed, cfF, GfF, len(F), -np.inf)
    assert total == 1 and got == _python_pass(fixed, cfF, GfF, len(F), -np.inf)


@needs_numba
def test_a_floor_above_every_bound_keeps_nothing():
    c, G = _dyadic(7, 9)
    fixed, cfF, GfF, _, _ = _parts(c, G, [])
    got, total = _kernel_pass(fixed, cfF, GfF, 3, np.inf)
    assert got == {} and total == comb(9, 3)


@needs_numba
def test_a_design_whose_sum_cannot_be_positive_is_never_kept():
    """c'u <= 0 for every design: no bound is finite and positive, nothing passes a floor of 0."""
    c, G = _dyadic(8, 8)
    c = [-abs(v) for v in c]
    fixed, cfF, GfF, _, _ = _parts(c, G, [])
    got, _ = _kernel_pass(fixed, cfF, GfF, 3, 0.0)
    assert got == {}


@needs_numba
def test_many_kept_designs_grow_the_output():
    """More designs pass than the kernel's first allocation (1024)."""
    c, G = _dyadic(9, 16)
    fixed, cfF, GfF, _, _ = _parts(c, G, [])
    got, total = _kernel_pass(fixed, cfF, GfF, 4, -np.inf)
    want = _python_pass(fixed, cfF, GfF, 4, -np.inf)
    assert len(want) > 1024 and got == want


# ---------------------------------------------------------------- failures


@needs_numba
@pytest.mark.parametrize("bad", ["k-negative", "k-too-large", "shape"])
def test_the_kernel_refuses_what_it_cannot_price(bad):
    c, G = _dyadic(10, 6)
    fixed, cfF, GfF, _, _ = _parts(c, G, [])
    k = {"k-negative": -1, "k-too-large": 7, "shape": 2}[bad]
    if bad == "shape":
        GfF = GfF[:, :-1]
    with pytest.raises(MlsynthConfigError, match="price"):
        fastprice.bound_pass(fixed, cfF, GfF, k, 0.0)


# ---------------------------------------------------------------- properties


@needs_numba
@SETTINGS
@given(st.integers(0, 10_000), st.integers(4, 11), st.integers(0, 4), st.integers(0, 2), st.floats(0.0, 1.0))
def test_property_at_dyadic_data_the_kernel_keeps_the_python_designs_at_the_same_bounds(seed, p, k, nfix, q):
    c, G = _dyadic(seed, p)
    fixes = [(j, j % 2) for j in range(nfix)]
    fixed, cfF, GfF, ins, F = _parts(c, G, fixes)
    k = min(k, len(F))
    every = _python_pass(fixed, cfF, GfF, k, -np.inf)
    finite = sorted(v for v in every.values() if np.isfinite(v))
    floor = finite[int(q * (len(finite) - 1))] if finite else 0.0
    got, total = _kernel_pass(fixed, cfF, GfF, k, floor)
    assert total == comb(len(F), k)
    assert got == _python_pass(fixed, cfF, GfF, k, floor)


@needs_numba
@SETTINGS
@given(st.integers(0, 10_000), st.integers(5, 10), st.integers(1, 4))
def test_property_pricing_on_the_kernel_matches_enumeration(seed, p, m):
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    out, inc = _run(D.c, D.G, m, [], Q(0))
    best = _best(D.c, D.G, p, m, [], Q(0))
    assert (inc["lam"] if inc["C"] else None) == (best[0] if best else None)
