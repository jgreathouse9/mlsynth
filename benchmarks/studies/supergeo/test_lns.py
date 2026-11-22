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


# --- stopping when the search stops paying ------------------------------------------------------------------------------
# Two rules end the search before its step budget, both counted in steps so the design stays a function of the
# inputs and the seed. patience p: stop once p steps in a row have not taken the loss below the lowest seen so far.
# target t: stop as soon as the loss is at most t, before the first step included. Neither draws from the random
# generator, so a run with either rule is the run without it, cut short: its trace is a prefix of the plain trace.

def _losses(trace):
    return [v for _, v in trace]


def _oracle_stop(plain, patience=None, target=None):
    """The step at which the rules end the search, read off the trace of the plain run with the same seed and budget:
    the first step s with loss <= target, or with s - (the last step that set a new lowest loss) >= patience;
    the budget when neither fires."""
    best, last = plain[0], 0
    for s, v in enumerate(plain):
        if s > 0 and v < best:
            best, last = v, s
        if target is not None and v <= target:
            return s
        if patience is not None and s > 0 and s - last >= patience:
            return s
    return len(plain) - 1


def test_patience_ends_a_search_that_cannot_improve():
    """Smoke: sorted pairs are the optimum with groups of exactly two, so no step improves them; with patience 5 the
    search ends after exactly 5 steps of a budget of 40, at the start design."""
    z = _similar(30)
    design, loss, trace = lns.lns_dp(z, _pairs(z), lo=2, hi=2, steps=40, patience=5, verbose=False)
    assert len(trace) == 6 and sorted(design) == sorted(_pairs(z))


def test_a_target_already_met_takes_no_step():
    """Edge: a start at or below the target ends the search before its first step."""
    z = _similar(20)
    start_loss = sum(lns.group_loss(z, g) for g in _pairs(z))
    for t in (start_loss, start_loss * 2, 1e300):
        design, loss, trace = lns.lns_dp(z, _pairs(z), steps=30, target=t, verbose=False)
        assert len(trace) == 1 and loss == start_loss


def _quadruples(k=8):
    """k quadruples a, a+1, a+2, a+3 spaced 10 apart: each splits evenly ({a, a+3} against {a+1, a+2}), so the
    optimum with groups of 2 to 4 costs 0, while the sorted pairs (a, a+1), (a+2, a+3) cost 1 each."""
    return np.concatenate([10.0 * j + np.arange(4.0) for j in range(k)])


def test_a_target_of_zero_stops_on_a_perfect_design():
    """Unit: on the quadruples, with groups of 2 to 4, the sorted pairs cost 16 and the optimum 0; with target 0
    the search stops at the first step whose design costs 0, which the plain run with the same seed reaches at the
    same step."""
    z = _quadruples()
    plain = lns.lns_dp(z, _pairs(z), steps=60, seed=4, verbose=False)
    assert plain[1] == 0.0
    s = _oracle_stop(_losses(plain[2]), target=0.0)
    design, loss, trace = lns.lns_dp(z, _pairs(z), steps=60, seed=4, target=0.0, verbose=False)
    assert loss == 0.0 and len(trace) == s + 1 and s < 60


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 26), kind=st.integers(0, 3), steps=st.integers(0, 40),
       sizes=st.sampled_from([(2, 2), (2, 3), (2, 4), (3, 4)]), patience=st.one_of(st.none(), st.integers(1, 12)),
       frac=st.one_of(st.none(), st.floats(0.0, 1.2)))
def test_the_rules_cut_the_plain_run_short_where_the_oracle_says(seed, n, kind, steps, sizes, patience, frac):
    """Property, over the data, the limits, the budget (0 included), the patience and the target: the run with the
    rules is the plain run with the same seed cut at the oracle's step. Its trace is a prefix of the plain trace,
    it ends where the oracle reads the rules as firing (or at the budget), and its design is a partition within the
    limits whose loss is the last value of its trace."""
    lo, hi = sizes
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    from . import certify_exact as ce
    if not ce._partition_exists(n, lo, hi):
        return
    start = ce._start(z, lo, hi)
    plain = _losses(lns.lns_dp(z, start, lo=lo, hi=hi, steps=steps, seed=seed % 5, verbose=False)[2])
    target = None if frac is None else frac * plain[0]
    design, loss, trace = lns.lns_dp(z, start, lo=lo, hi=hi, steps=steps, seed=seed % 5, patience=patience,
                                     target=target, verbose=False)
    got = _losses(trace)
    assert got == plain[:len(got)]
    assert len(got) - 1 == _oracle_stop(plain, patience, target)
    assert sorted(i for g in design for i in g) == list(range(n)) and all(lo <= len(g) <= hi for g in design)
    assert loss == got[-1]


def test_patience_applies_to_a_time_budget_too():
    """With a time budget the rules end the search the same way: sorted pairs with groups of two, a 60 s budget and
    patience 3 take 3 steps, far inside the time."""
    z = _similar(24)
    t = time.perf_counter()
    design, loss, trace = lns.lns_dp(z, _pairs(z), lo=2, hi=2, seconds=60, patience=3, verbose=False)
    assert len(trace) == 4 and time.perf_counter() - t < 30


def test_a_target_applies_to_a_time_budget_too():
    z = _similar(20)
    start_loss = sum(lns.group_loss(z, g) for g in _pairs(z))
    design, loss, trace = lns.lns_dp(z, _pairs(z), seconds=60, target=start_loss, verbose=False)
    assert len(trace) == 1


@pytest.mark.parametrize("kw", [dict(patience=0), dict(patience=-1), dict(patience=2.5), dict(patience=True),
                                dict(patience="5"), dict(patience=np.float64(3.0)), dict(target=float("nan")),
                                dict(target=float("inf")), dict(target=-float("inf")), dict(target=True),
                                dict(target="0"), dict(target=None, patience=False)])
def test_bad_stopping_rules_are_rejected(kw):
    """Failure: patience is None or an integer >= 1; target is None or a finite number. Anything else raises before
    any step, naming the argument."""
    z = _similar(10)
    name = [k for k in kw if kw[k] is not None][0]
    with pytest.raises(ValueError, match=name):
        lns.lns_dp(z, _pairs(z), steps=5, verbose=False, **kw)
