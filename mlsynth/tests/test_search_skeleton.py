"""The loop the certified branch and bounds share (#846): the work ledger, the
queue's order and its dive, the price-or-bound rule, the loop's handling of
the limit, and the bound over the nodes a stopped search leaves open."""
import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.search_skeleton import (Bound, Deferred, Ledger, NodeQueue, Price,
                                           cheaper_to_price, check_work_limit,
                                           least_open_bound, run)


class TestLedger:
    def test_it_adds_what_is_charged(self):
        ledger = Ledger(10.0)
        ledger.charge(3.0)
        ledger.charge(4.5)
        assert ledger.spent == 7.5 and ledger.left() == 2.5

    def test_a_charge_passes_the_limit_only_past_it(self):
        ledger = Ledger(10.0)
        ledger.charge(7.0)
        assert not ledger.passes(3.0) and ledger.passes(3.5)

    def test_it_is_exhausted_at_the_limit(self):
        ledger = Ledger(1.0)
        assert not ledger.exhausted()
        ledger.charge(1.0)
        assert ledger.exhausted() and ledger.left() == 0.0
        ledger.charge(1.0)
        assert ledger.left() == 0.0

    def test_no_limit_is_never_reached(self):
        ledger = Ledger(np.inf)
        ledger.charge(1e300)
        assert not ledger.exhausted() and not ledger.passes(1e300)


class TestCheckWorkLimit:
    @pytest.mark.parametrize("limit", [1, 0.5, 120.0, np.float64(3.0)])
    def test_a_positive_finite_limit_passes(self, limit):
        check_work_limit(limit)

    @pytest.mark.parametrize("limit", [0, -1.0, np.inf, np.nan, True, "120", None])
    def test_anything_else_is_refused(self, limit):
        with pytest.raises(MlsynthConfigError, match="work_limit must be a positive "
                           "finite number of seconds"):
            check_work_limit(limit)


class TestCheaperToPrice:
    def test_pricing_wins_when_it_costs_no_more(self):
        assert cheaper_to_price(10, 1.0, 10.0)
        assert not cheaper_to_price(11, 1.0, 10.0)

    def test_a_bound_that_rarely_prunes_is_worth_less(self):
        """At a prune rate of one in ten, a bound saves a tenth of the pricing."""
        assert cheaper_to_price(100, 1.0, 10.0, rate=0.1)
        assert not cheaper_to_price(100, 1.0, 10.0, rate=0.2)

    @given(st.integers(0, 10 ** 9), st.floats(0, 1), st.floats(0, 1e6))
    @settings(max_examples=200, deadline=None)
    def test_a_rate_of_one_is_the_plain_comparison(self, size, price, bound):
        """The unweighted rule is exactly ``size * price <= bound``: a rate of
        one changes no comparison."""
        assert cheaper_to_price(size, price, bound) == (size * price <= bound)
        assert cheaper_to_price(size, price, bound, rate=1.0) == (size * price <= bound)


def _drain(queue):
    out = []
    while queue:
        out.append(queue.pop())
    return out


class TestNodeQueue:
    def test_it_pops_the_smallest_key_first(self):
        queue = NodeQueue(last_first=False)
        for key in (3.0, 1.0, 2.0):
            queue.push(key, f"n{key:g}")
        assert [e[2] for e in _drain(queue)] == ["n1", "n2", "n3"]

    @pytest.mark.parametrize("last_first, order", [(False, "abc"), (True, "cba")])
    def test_equal_keys_go_in_the_order_asked(self, last_first, order):
        queue = NodeQueue(last_first=last_first)
        for node in "abc":
            queue.push(0.0, node)
        assert "".join(e[2] for e in _drain(queue)) == order

    def test_it_dives_until_a_design_is_found(self):
        """Pushed before a design exists, nodes go on a stack and pop last-in
        first whatever their keys; once one exists every node waiting there
        joins the best-first heap."""
        found = [False]
        queue = NodeQueue(last_first=True, found=lambda: found[0])
        queue.push(1.0, "a")
        queue.push(5.0, "b")
        assert queue.pop()[2] == "b"
        queue.push(9.0, "c")
        queue.push(0.5, "d")
        assert queue.pop()[2] == "d"
        found[0] = True
        queue.push(2.0, "e")
        assert [e[2] for e in _drain(queue)] == ["a", "e", "c"]

    def test_the_open_nodes_are_the_heap_then_the_dive_then_the_deferred(self):
        found = [True]
        queue = NodeQueue(last_first=True, found=lambda: found[0])
        queue.push(1.0, "a")
        found[0] = False
        queue.push(2.0, "b")
        entry = queue.pop()
        queue.put_back(entry, key=7.0)
        assert [(e[0], e[2]) for e in queue.open()] == [(1.0, "a"), (7.0, "b")]
        assert len(queue) == 1

    def test_a_node_put_back_keeps_its_key_unless_given_one(self):
        queue = NodeQueue(last_first=False)
        queue.push(4.0, "x")
        entry = queue.pop()
        queue.put_back(entry)
        assert queue.open() == [entry] and not queue


def _toy(values, K, limit=np.inf, price_cost=1.0, bound_cost=1.0, *, last_first=True,
         check_before_pop=True):
    """Choose ``K`` of ``values`` to maximise their sum: priced set by set when
    the subtree is small, bounded by the best completion otherwise."""
    values = list(values)
    n = len(values)
    best = [-np.inf, None]
    ledger = Ledger(limit)
    priced = [0]

    from math import comb

    def plan(key, node):
        I, F = node
        r = K - len(I)
        if r < 0 or r > len(F):
            return None
        count = comb(len(F), r)
        if r == 0 or r == len(F) or cheaper_to_price(count, price_cost, bound_cost):
            def price():
                for extra in itertools.combinations(F, r):
                    priced[0] += 1
                    v = sum(values[i] for i in I + extra)
                    if v > best[0]:
                        best[0], best[1] = v, tuple(sorted(I + extra))
            return Price(cost=count * price_cost, deferrable=True, run=price)

        def bound(ledger):
            if ledger.passes(bound_cost):
                return Deferred(key)
            ledger.charge(bound_cost)
            top = sum(values[i] for i in I) + sum(sorted((values[i] for i in F),
                                                         reverse=True)[:r])
            if top <= best[0]:
                return None
            u, rest = F[0], F[1:]
            return [(-top, (I, rest)), (-top, (I + (u,), rest))]

        return Bound(run=bound)

    queue = NodeQueue(last_first=last_first)
    queue.push(-np.inf, ((), tuple(range(n))))
    closed = run(queue, ledger, plan=plan, check_before_pop=check_before_pop)
    return best, closed, ledger, queue, priced[0]


class TestRun:
    @given(st.lists(st.integers(-50, 50), min_size=2, max_size=9), st.data(),
           st.booleans(), st.floats(0.0, 50.0))
    @settings(max_examples=200, deadline=None)
    def test_with_room_it_finds_the_optimum(self, values, data, last_first, bound_cost):
        K = data.draw(st.integers(1, len(values) - 1))
        best, closed, _, queue, _ = _toy(values, K, bound_cost=bound_cost,
                                         last_first=last_first)
        assert closed and not queue.open()
        assert best[0] == max(sum(c) for c in itertools.combinations(values, K))

    @given(st.lists(st.integers(-50, 50), min_size=4, max_size=9), st.data(),
           st.floats(0.5, 30.0), st.booleans())
    @settings(max_examples=200, deadline=None)
    def test_the_limit_is_kept(self, values, data, limit, check_before_pop):
        """No pricing or bound that would pass the limit is started, so the work
        stays inside it; a stopped search leaves the node it stopped at open."""
        K = data.draw(st.integers(1, len(values) - 1))
        best, closed, ledger, queue, _ = _toy(values, K, limit=limit, bound_cost=2.0,
                                              check_before_pop=check_before_pop)
        assert ledger.spent <= limit
        assert closed == (not queue.open())

    def test_a_pricing_that_would_pass_the_limit_is_deferred(self):
        best, closed, ledger, queue, priced = _toy([1, 2, 3, 4], 2, limit=5.0,
                                                   price_cost=1.0, bound_cost=100.0)
        assert not closed and priced == 0 and ledger.spent == 0.0
        assert [e[2] for e in queue.open()] == [((), (0, 1, 2, 3))]

    @pytest.mark.parametrize("check_before_pop", [True, False])
    def test_where_the_limit_is_read(self, check_before_pop):
        """The first node's bound spends the whole limit; the others cost
        nothing. Read before each node, the limit stops the search with them
        queued; read only before each charge, they are popped and dropped."""
        queue = NodeQueue(last_first=False)
        for key, node in ((0.0, "a"), (1.0, "b"), (2.0, "c")):
            queue.push(key, node)

        def plan(key, node):
            if node != "a":
                return None

            def bound(ledger):
                ledger.charge(5.0)
                return []
            return Bound(run=bound)

        closed = run(queue, Ledger(5.0), plan=plan, check_before_pop=check_before_pop)
        assert closed == (not check_before_pop)
        assert [e[2] for e in queue.open()] == (["b", "c"] if check_before_pop else [])

    def test_a_bound_that_would_pass_the_limit_is_put_back_with_its_key(self):
        """The bound stage may raise a node's key before it finds the limit in
        the way; the node stays open under the raised key."""
        queue = NodeQueue(last_first=False)
        queue.push(1.0, "a")
        plan = lambda key, node: Bound(run=lambda ledger: Deferred(9.0))
        assert not run(queue, Ledger(5.0), plan=plan, check_before_pop=False)
        assert [(e[0], e[2]) for e in queue.open()] == [(9.0, "a")]

    def test_a_dropped_node_has_no_children(self):
        queue = NodeQueue(last_first=False)
        queue.push(0.0, "root")
        seen = []

        def plan(key, node):
            seen.append(node)
            return None

        assert run(queue, Ledger(1.0), plan=plan, check_before_pop=False)
        assert seen == ["root"] and not queue.open()


class TestLeastOpenBound:
    def test_it_is_the_branch_modules_helper(self):
        from mlsynth.utils.syndes_helpers import branch
        assert branch.least_open_bound is least_open_bound
