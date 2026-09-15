"""A heuristic finding nothing is not a proof that nothing exists.

LEXSCM's multi-start search builds each treated tuple greedily: from a seed it
repeatedly adds the candidate that most lowers the batched loss, keeping only
additions that leave the partial tuple feasible. Feasibility of a *partial*
tuple is the wrong test under a budget. Spending is monotone, so a partial that
fits can still be impossible to finish, and the construction discovers this only
when it runs out of affordable candidates and returns nothing.

On a 60-market DMA design at m=4 that happened on 32 starts out of 32, at depth
2 or 3 of 4, while the four cheapest eligible markets together cost 40% of the
budget -- a feasible tuple was there for the taking. Every start failing left
the pool empty, and the branch downstream read an empty pool as proof that the
constraints admit no design, raising ``MlsynthConfigError`` and telling the
analyst to relax the adjacency constraint. Enumeration on the same inputs
returned 26,083 feasible tuples and status ``OPTIMAL``.

Two separate defects, pinned separately here:

1. The construction is myopic about the budget
   (:class:`TestGreedyCompletesUnderATightBudget`). Rejecting an addition needs
   a completion bound -- after adding ``j``, can the cheapest remaining
   ``m - |S| - 1`` candidates still be afforded? -- and that bound must never
   reject a partial that is completable (:class:`TestCompletionBound`).
2. An empty pool was read as infeasibility whichever path produced it
   (:class:`TestInfeasibilityIsOnlyClaimedWhenProved`). Only enumeration
   exhausts the region, so only enumeration can prove emptiness. The message
   also asserted ``largest conflict-free set is {largest} < m={m}`` without
   checking, and ``greedy_independent_set_size`` is a lower bound on the maximum
   independent set: when it reaches ``m`` it *proves* a conflict-free ``m``-tuple
   exists. The branch was printing a claim its own last computation refuted --
   ``37 < m=4`` in the case above.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.fast_scm_helpers import lexsearch as ls
from mlsynth.utils.fast_scm_helpers.conflict import greedy_independent_set_size
from mlsynth.utils.fast_scm_helpers.lexsearch import select_treated_designs


# =========================================================================
# Instances
# =========================================================================

def _starving_instance(n=20, T=12, seed=2, m=4, mult=2.5):
    """A design whose loss-greedy construction overspends before it finishes.

    Costs are geometric in the single-donor distance ``G_jj``, so the markets
    the greedy reaches for first are the dear ones -- the shape of the real DMA
    panel, where sales volume and closeness to the population centroid rise
    together. The budget is ``mult`` times the ``m`` cheapest, generous enough
    that a cheapest-first tuple fits with room to spare.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, n))
    X = X - X.mean(axis=1, keepdims=True)
    G = X.T @ X
    costs = np.empty(n)
    costs[np.argsort(np.diag(G))] = np.geomspace(8.0, 1.0, n)
    budget = float(np.sort(costs)[:m].sum() * mult)
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(0, n - 1, 3):
        conflict[i, i + 1] = conflict[i + 1, i] = True
    return dict(G=G, cand=list(range(n)), m=m, costs=costs, budget=budget,
                conflict=conflict)


def _clique_conflict(labels):
    """Conflict graph where units sharing a label are mutually exclusive."""
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]) & ~np.eye(len(labels), dtype=bool)


def _gram(n, T, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, n))
    X = X - X.mean(axis=1, keepdims=True)
    return X.T @ X


def _feasible(S, inst):
    return (inst["costs"][list(S)].sum() <= inst["budget"] + 1e-9
            and not any(inst["conflict"][a, b] for a, b in combinations(sorted(S), 2)))


@pytest.fixture(scope="module")
def inst():
    return _starving_instance()


@pytest.fixture(scope="module")
def certified(inst):
    return select_treated_designs(
        inst["G"], candidate_idx=inst["cand"], m=inst["m"], top_K=5,
        method="enumerate", unit_costs=inst["costs"], budget=inst["budget"],
        conflict=inst["conflict"])


# =========================================================================
# 1. The construction must finish when a feasible tuple exists
# =========================================================================

class TestGreedyCompletesUnderATightBudget:
    """The search returns a design whenever the region is not empty."""

    def test_the_region_is_not_empty(self, inst, certified):
        """Enumeration proves a feasible design exists, so the heuristic must
        not report otherwise."""
        assert certified["stats"]["termination"]["status"] == "OPTIMAL"
        assert certified["stats"]["search"]["subsets_evaluated"] > 1000
        assert _feasible(certified["top_designs"][0].indices, inst)

    @pytest.mark.parametrize("seed", range(6))
    def test_heuristic_returns_a_design(self, inst, seed):
        out = select_treated_designs(
            inst["G"], candidate_idx=inst["cand"], m=inst["m"], top_K=5,
            method="heuristic", n_starts=16, random_state=seed,
            unit_costs=inst["costs"], budget=inst["budget"],
            conflict=inst["conflict"])
        assert out["top_designs"], "search returned nothing where designs exist"
        assert out["stats"]["termination"]["status"] == "FEASIBLE"

    @pytest.mark.parametrize("seed", range(6))
    def test_returned_designs_respect_every_constraint(self, inst, seed):
        out = select_treated_designs(
            inst["G"], candidate_idx=inst["cand"], m=inst["m"], top_K=5,
            method="heuristic", n_starts=16, random_state=seed,
            unit_costs=inst["costs"], budget=inst["budget"],
            conflict=inst["conflict"])
        for d in out["top_designs"]:
            assert len(d.indices) == inst["m"]
            assert _feasible(d.indices, inst)

    @pytest.mark.parametrize("seed", range(6))
    def test_the_answer_is_close_to_the_certified_optimum(self, inst, certified, seed):
        """Completing the build is not enough; it has to find a good design."""
        out = select_treated_designs(
            inst["G"], candidate_idx=inst["cand"], m=inst["m"], top_K=5,
            method="heuristic", n_starts=16, random_state=seed,
            unit_costs=inst["costs"], budget=inst["budget"],
            conflict=inst["conflict"])
        gap = out["top_designs"][0].imbalance / certified["top_designs"][0].imbalance - 1
        assert gap < 0.10, f"imbalance {gap:.1%} above the certified optimum"

    def test_a_loose_budget_reaches_the_same_answer_as_no_budget(self):
        """The completion bound must not bite where the budget cannot."""
        G = _gram(16, 10, seed=5)
        costs = np.ones(16)
        free = select_treated_designs(G, list(range(16)), m=4, top_K=5,
                                      method="heuristic", n_starts=8, random_state=0)
        loose = select_treated_designs(G, list(range(16)), m=4, top_K=5,
                                       method="heuristic", n_starts=8, random_state=0,
                                       unit_costs=costs, budget=1000.0)
        assert ([d.indices for d in free["top_designs"]]
                == [d.indices for d in loose["top_designs"]])


# =========================================================================
# 2. The completion bound has to be sound
# =========================================================================

class TestCompletionBound:
    """Pruning a partial that could still be finished would lose the optimum."""

    def test_bound_never_rejects_a_completable_partial(self):
        """Brute force over every partial: the bound and the truth agree.

        Truth: some size-``m`` superset of the partial is within budget.
        The bound ignores the conflict graph, so it may admit a partial that
        conflicts block -- admitting too much is safe, rejecting is not.
        """
        rng = np.random.default_rng(0)
        n, m = 12, 4
        costs = rng.uniform(1.0, 9.0, n)
        free = list(range(n))
        for budget in (12.0, 16.0, 20.0, 30.0):
            for size in (1, 2, 3):
                for S in combinations(range(n), size):
                    S = list(S)
                    completable = any(
                        costs[list(S) + list(extra)].sum() <= budget + 1e-12
                        for extra in combinations([j for j in free if j not in S],
                                                  m - size))
                    allowed = ls._budget_allows_completion(
                        S, free, costs, budget, m)
                    if completable:
                        assert allowed, (
                            f"bound rejected a completable partial {S} "
                            f"at budget {budget}")

    def test_bound_is_a_no_op_without_a_budget(self):
        free = list(range(6))
        assert ls._budget_allows_completion([0], free, None, None, 3)
        assert ls._budget_allows_completion([0], free, np.ones(6), np.inf, 3)

    def test_bound_rejects_when_too_few_candidates_remain(self):
        """A partial cannot be finished from a pool that has run out."""
        costs = np.array([1.0, 1.0, 1.0])
        assert not ls._budget_allows_completion([0], [0, 1, 2], costs, 100.0, m=4)
        assert ls._budget_allows_completion([0], [0, 1, 2], costs, 100.0, m=3)

    def test_bound_rejects_a_partial_that_has_overspent(self):
        costs = np.array([10.0, 10.0, 1.0, 1.0, 1.0])
        free = list(range(5))
        # two dear units already take the whole budget; one more is needed
        assert not ls._budget_allows_completion([0, 1], free, costs, 20.5, 3)
        assert ls._budget_allows_completion([0], free, costs, 20.5, 3)
        # 10 + 10 + 1 is exactly 21, so at 21.0 the partial does complete
        assert ls._budget_allows_completion([0, 1], free, costs, 21.0, 3)


# =========================================================================
# 3. Infeasibility is claimed only where it is proved
# =========================================================================

class TestInfeasibilityIsOnlyClaimedWhenProved:
    """Enumeration can prove an empty region. A failed search cannot."""

    def test_enumeration_still_proves_spillover_infeasibility(self):
        """Three cliques cannot seat four mutually non-adjacent units."""
        G = _gram(10, 8, seed=1)
        conflict = _clique_conflict([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        with pytest.raises(MlsynthConfigError, match="spillover|conflict-free") as e:
            select_treated_designs(G, list(range(10)), m=4, top_K=5,
                                   method="enumerate", conflict=conflict)
        msg = str(e.value)
        assert greedy_independent_set_size(conflict, np.arange(10)) < 4
        assert "3 < m=4" in msg, msg          # the inequality it prints is true

    def test_message_never_asserts_a_false_inequality(self, monkeypatch):
        """When a conflict-free m-tuple provably exists, do not claim otherwise.

        The construction is forced to fail so the error branch is reached on an
        instance whose conflict graph plainly admits an ``m``-tuple.
        """
        inst = _starving_instance()
        largest = greedy_independent_set_size(inst["conflict"],
                                              np.asarray(inst["cand"]))
        assert largest >= inst["m"], "fixture must admit a conflict-free tuple"
        monkeypatch.setattr(ls, "_local_search",
                            lambda *a, **k: ([], 0, None))
        with pytest.raises(MlsynthConfigError) as e:
            select_treated_designs(
                inst["G"], candidate_idx=inst["cand"], m=inst["m"], top_K=5,
                method="heuristic", n_starts=16, random_state=0,
                unit_costs=inst["costs"], budget=inst["budget"],
                conflict=inst["conflict"])
        msg = str(e.value)
        assert f"< m={inst['m']}" not in msg, (
            f"message asserts an inequality the code disproved:\n{msg}")
        assert str(largest) in msg and "not binding on its own" in msg, msg

    def test_heuristic_failure_names_the_search_not_the_region(self, monkeypatch):
        """The error has to say the search found nothing, not that nothing is
        there, and point at the exhaustive path."""
        inst = _starving_instance()
        monkeypatch.setattr(ls, "_local_search",
                            lambda *a, **k: ([], 0, None))
        with pytest.raises(MlsynthConfigError) as e:
            select_treated_designs(
                inst["G"], candidate_idx=inst["cand"], m=inst["m"], top_K=5,
                method="heuristic", n_starts=16, random_state=0,
                unit_costs=inst["costs"], budget=inst["budget"],
                conflict=inst["conflict"])
        msg = str(e.value).lower()
        assert "multi-start search" in msg
        assert "not a proof" in msg
        assert "n_starts" in msg and "enumerate" in msg

    def test_error_with_no_named_constraint_still_explains_the_path(self, monkeypatch):
        """With neither a conflict graph nor a budget there is no line to add.

        The head has to carry the whole message on its own, and it still has to
        distinguish a search that found nothing from a region that holds
        nothing.
        """
        G = _gram(12, 9, seed=3)
        monkeypatch.setattr(ls, "_local_search", lambda *a, **k: ([], 0, None))
        with pytest.raises(MlsynthConfigError) as e:
            select_treated_designs(G, list(range(12)), m=3, top_K=5,
                                   method="heuristic", n_starts=8, random_state=0)
        msg = str(e.value)
        assert "multi-start search" in msg and "not a proof" in msg
        assert "\n  - " not in msg, f"nothing to itemise, yet it itemised:\n{msg}"

    def test_enumeration_failure_still_claims_the_region_is_empty(self):
        """Enumeration did exhaust the region, so the strong claim is earned."""
        G = _gram(10, 8, seed=1)
        conflict = _clique_conflict([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        with pytest.raises(MlsynthConfigError) as e:
            select_treated_designs(G, list(range(10)), m=4, top_K=5,
                                   method="enumerate", conflict=conflict)
        assert "infeasible" in str(e.value).lower()
        assert "not a proof" not in str(e.value).lower()
