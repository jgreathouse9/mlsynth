"""The large-neighbourhood search with a step budget: the same inputs and seed give the same design."""
import threading, time
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from . import lns, test_bcp


def _similar(n, seed=2026):
    pre = np.random.default_rng(seed).normal(100, 10, n)
    return pre / np.median(np.diff(np.sort(pre)))


def _pairs(z):
    o = [int(v) for v in np.argsort(z)]
    return [tuple(o[i:i + 2]) for i in range(0, len(z) - len(z) % 2, 2)]


class _Load:
    """Busy threads doing numpy work while the block runs, so a time-budgeted search would do fewer steps."""
    def __enter__(self):
        self.stop = False
        def burn():
            a = np.random.default_rng(0).normal(size=(300, 300))
            while not self.stop:
                a = np.tanh(a @ a.T / 300)
        self.ts = [threading.Thread(target=burn) for _ in range(3)]
        for t in self.ts: t.start()
        return self
    def __exit__(self, *exc):
        self.stop = True
        for t in self.ts: t.join()


def test_the_same_seed_and_steps_give_the_same_design_under_load():
    """A step budget makes the search a function of its inputs: the same design and loss with the machine idle and
    with three busy threads competing for it."""
    z = _similar(40)
    a = lns.lns_dp(z, _pairs(z), steps=25, seed=3, verbose=False)
    with _Load():
        b = lns.lns_dp(z, _pairs(z), steps=25, seed=3, verbose=False)
    assert sorted(a[0]) == sorted(b[0]) and a[1] == b[1]


def test_the_seed_decides_the_design():
    """Different seeds take different paths: on this instance seeds 0 and 1 end at different designs."""
    z = _similar(40)
    a = lns.lns_dp(z, _pairs(z), steps=25, seed=0, verbose=False)
    b = lns.lns_dp(z, _pairs(z), steps=25, seed=1, verbose=False)
    assert sorted(a[0]) != sorted(b[0])


def test_exactly_the_steps_asked_for():
    z = _similar(30)
    for k in (0, 1, 7):
        design, loss, trace = lns.lns_dp(z, _pairs(z), steps=k, verbose=False)
        assert len(trace) == k + 1
    design, loss, trace = lns.lns_dp(z, _pairs(z), steps=0, verbose=False)
    assert sorted(design) == sorted(_pairs(z))                  # no step, no change


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 30), kind=st.integers(0, 3), steps=st.integers(0, 12))
def test_the_search_only_improves_and_returns_a_partition(seed, n, kind, steps):
    """Every step keeps the loss or lowers it, and the design returned is a partition into groups of 2 to 4 whose
    loss is the one reported."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    start = [tuple(g) for g in test_bcp._sorted_pairs(z)]
    design, loss, trace = lns.lns_dp(z, start, steps=steps, seed=seed % 7, verbose=False)
    vals = [v for _, v in trace]
    assert all(b <= a for a, b in zip(vals, vals[1:]))
    assert sorted(i for g in design for i in g) == list(range(n)) and all(2 <= len(g) <= 4 for g in design)
    assert abs(loss - sum(lns.group_loss(z, g) for g in design)) <= 1e-9 * max(1.0, loss)


def test_bad_step_budgets_are_rejected():
    z = _similar(10)
    for k in (-1, 2.5, "many", True):
        with pytest.raises(ValueError):
            lns.lns_dp(z, _pairs(z), steps=k, verbose=False)
