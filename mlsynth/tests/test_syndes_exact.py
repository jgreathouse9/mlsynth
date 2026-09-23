"""Tests for the SYNDES exact two-way backend.

Contract. ``solve_synthetic_design_exact`` returns the same ``SYNDESDesign`` the
SCIP path returns, for ``mode="global_2way"``, by searching treated sets directly
instead of branch-and-bound. It scores every candidate with the closed form from
``gram``, settles the rows whose sign conditions are slack, prunes the rest
against the incumbent, and calls the inner solver only on survivors.

Parity is the test this module rests on: for every instance SCIP can finish,
both backends must select the same treated set and agree on the objective. Everything else pins how the search behaves -- that pruning is sound,
that restrictions are honoured as subset filters, that ``K=None`` ranges over
cardinalities, and that the pool comes out ranked.

Restrictions on the assignment (forced, forbidden, conflicting, stratum quotas,
costs and a budget) are pure predicates on the treated set, so the search applies
them while generating candidates. ``donor_exclusion`` is not: it ties the control
weights to which units are treated, so the exact backend refuses it.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.syndes_helpers.exact import (
    SubsetFilter,
    _stream_subsets,
    solve_synthetic_design_exact,
    solve_synthetic_design_exact_pool,
)
from mlsynth.utils.syndes_helpers.gram import TwoWayProblem
from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design
from mlsynth.utils.syndes_helpers.partition import (
    solve_partition_batch,
    split_indices,
)
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions


def _panel(N=9, T=14, rank=3, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _brute_force(Y, K, lam=None):
    """The optimum by evaluating every subset with the inner solver."""
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    subsets = np.array(list(combinations(range(problem.N), K)), dtype=np.int64)
    t, c = split_indices(problem.N, subsets)
    values = solve_partition_batch(problem, t, c, max_iter=8000, tol=1e-14).value
    best = int(np.argmin(values))
    return tuple(subsets[best]), float(values[best])


# ----------------------------------------------------------------------
# parity with the MIP
# ----------------------------------------------------------------------


class TestParityWithScip:
    @pytest.mark.parametrize("K", [2, 3, 4, 6])
    def test_same_treated_set_and_objective(self, K):
        Y = _panel()
        mip = solve_synthetic_design(Y=Y, K=K, mode="global_2way", solver="SCIP")
        exact = solve_synthetic_design_exact(Y=Y, K=K)
        assert tuple(exact.selected_unit_indices) == tuple(
            sorted(int(i) for i in np.flatnonzero(mip.assignment)))
        assert exact.objective_value == pytest.approx(
            float(mip.objective_value), rel=1e-6)

    def test_exact_is_never_worse_than_the_mip(self):
        """Both minimise the same objective, so the search cannot lose."""
        for seed in (1, 2, 3):
            Y = _panel(seed=seed)
            mip = solve_synthetic_design(Y=Y, K=3, mode="global_2way", solver="SCIP")
            exact = solve_synthetic_design_exact(Y=Y, K=3)
            assert exact.objective_value <= float(mip.objective_value) + 1e-8

    @pytest.mark.parametrize("lam", [0.01, 1.0, 7.5])
    def test_parity_across_penalties(self, lam):
        Y = _panel()
        mip = solve_synthetic_design(Y=Y, K=3, mode="global_2way",
                                     lam=lam, solver="SCIP")
        exact = solve_synthetic_design_exact(Y=Y, K=3, lam=lam)
        assert tuple(exact.selected_unit_indices) == tuple(
            sorted(int(i) for i in np.flatnonzero(mip.assignment)))


class TestAgreesWithBruteForce:
    @pytest.mark.parametrize("K", [1, 2, 5, 8])
    def test_finds_the_brute_force_optimum(self, K):
        Y = _panel()
        expected_set, expected_value = _brute_force(Y, K)
        got = solve_synthetic_design_exact(Y=Y, K=K)
        assert tuple(got.selected_unit_indices) == expected_set
        assert got.objective_value == pytest.approx(expected_value, rel=1e-7)

    def test_pruning_does_not_discard_the_optimum(self):
        """A deliberately bad seed must not survive as the answer."""
        Y = _panel()
        expected_set, _ = _brute_force(Y, 3)
        got = solve_synthetic_design_exact(Y=Y, K=3, incumbent_sets=np.array([[0, 1, 2]]))
        assert tuple(got.selected_unit_indices) == expected_set


# ----------------------------------------------------------------------
# the design object it returns
# ----------------------------------------------------------------------


class TestDesignContract:
    def test_populates_the_same_fields_as_the_mip(self):
        Y = _panel()
        mip = solve_synthetic_design(Y=Y, K=3, mode="global_2way", solver="SCIP")
        exact = solve_synthetic_design_exact(Y=Y, K=3)
        for field in ("mode", "objective_value", "lambda_value", "assignment",
                      "selected_unit_indices", "selected_unit_labels",
                      "assignment_by_unit", "w", "q", "treated_weights",
                      "control_weights", "contrast_weights", "contrast_series",
                      "pre_fit_rmse"):
            assert getattr(exact, field) is not None, field
            assert getattr(mip, field) is not None, field

    def test_weight_conventions_match_the_mip(self):
        """``q`` is the treated block, ``w - q`` the control block."""
        Y = _panel()
        exact = solve_synthetic_design_exact(Y=Y, K=3)
        np.testing.assert_allclose(exact.q, exact.treated_weights, atol=1e-12)
        np.testing.assert_allclose(exact.w - exact.q, exact.control_weights, atol=1e-12)
        np.testing.assert_allclose(exact.contrast_weights,
                                   exact.treated_weights - exact.control_weights,
                                   atol=1e-12)
        assert exact.treated_weights.sum() == pytest.approx(1.0)
        assert exact.control_weights.sum() == pytest.approx(1.0)

    def test_supports_are_disjoint_and_match_the_assignment(self):
        Y = _panel()
        exact = solve_synthetic_design_exact(Y=Y, K=3)
        treated = exact.assignment.astype(bool)
        assert np.all(exact.treated_weights[~treated] == 0.0)
        assert np.all(exact.control_weights[treated] == 0.0)
        assert int(exact.assignment.sum()) == 3

    def test_contrast_series_and_rmse(self):
        Y = _panel()
        exact = solve_synthetic_design_exact(Y=Y, K=3)
        np.testing.assert_allclose(exact.contrast_series,
                                   Y @ exact.contrast_weights, atol=1e-10)
        assert exact.pre_fit_rmse == pytest.approx(
            float(np.sqrt(np.mean(exact.contrast_series ** 2))))

    def test_unit_labels_are_carried_through(self):
        Y = _panel()
        labels = [f"u{i}" for i in range(Y.shape[1])]
        exact = solve_synthetic_design_exact(
            Y=Y, K=3, unit_index=IndexSet.from_labels(labels))
        assert set(exact.selected_unit_labels) <= set(labels)
        assert len(exact.selected_unit_labels) == 3
        assert set(exact.assignment_by_unit) == set(labels)

    def test_raw_results_record_how_the_search_went(self):
        Y = _panel()
        raw = solve_synthetic_design_exact(Y=Y, K=3).raw_results
        assert raw["solver"] == "exact"
        assert raw["n_designs"] == 84                      # C(9, 3)
        assert raw["n_closed_form"] + raw["n_solved"] <= raw["n_designs"]
        assert raw["certified"] is True
        assert raw["duality_gap"] >= 0.0


# ----------------------------------------------------------------------
# restrictions, costs, and the pool
# ----------------------------------------------------------------------


class TestSubsetStream:
    """A slice the filter empties is not the end of the stream."""

    def test_yields_every_admissible_subset_across_chunks(self):
        filt = SubsetFilter.build(DesignRestrictions(forced_in=[4]), N=6)
        blocks = list(_stream_subsets(6, 2, filt, chunk=3))
        seen = {tuple(row) for block in blocks for row in block}
        expected = {s for s in combinations(range(6), 2) if 4 in s}
        assert seen == expected

    def test_survives_a_fully_filtered_leading_chunk(self):
        """The first chunk is entirely rejected; later admissible sets still arrive."""
        filt = SubsetFilter.build(DesignRestrictions(forced_in=[5]), N=6)
        assert not filt.accepts((0, 1))                     # the first candidate
        blocks = list(_stream_subsets(6, 2, filt, chunk=2))
        seen = {tuple(row) for block in blocks for row in block}
        assert seen == {s for s in combinations(range(6), 2) if 5 in s}

    def test_empty_when_nothing_is_admissible(self):
        filt = SubsetFilter.build(DesignRestrictions(forbidden=list(range(6))), N=6)
        assert list(_stream_subsets(6, 2, filt, chunk=4)) == []

    def test_respects_the_chunk_size(self):
        filt = SubsetFilter.build(None, N=6)
        for block in _stream_subsets(6, 2, filt, chunk=4):
            assert block.shape[0] <= 4


class TestRestrictions:
    def test_forced_units_are_treated(self):
        Y = _panel()
        got = solve_synthetic_design_exact(
            Y=Y, K=3, restrictions=DesignRestrictions(forced_in=[0, 5]))
        assert {0, 5} <= set(int(i) for i in got.selected_unit_indices)

    def test_forbidden_units_are_not_treated(self):
        Y = _panel()
        got = solve_synthetic_design_exact(
            Y=Y, K=3, restrictions=DesignRestrictions(forbidden=[1, 2, 3]))
        assert not ({1, 2, 3} & set(int(i) for i in got.selected_unit_indices))

    def test_conflicting_pairs_are_never_both_treated(self):
        Y = _panel()
        got = solve_synthetic_design_exact(
            Y=Y, K=4, restrictions=DesignRestrictions(conflict_pairs=[(0, 1), (2, 3)]))
        chosen = set(int(i) for i in got.selected_unit_indices)
        assert not {0, 1} <= chosen
        assert not {2, 3} <= chosen

    def test_stratum_quotas_are_respected(self):
        Y = _panel()
        got = solve_synthetic_design_exact(
            Y=Y, K=4,
            restrictions=DesignRestrictions(strata=[((0, 1, 2, 3), 2, 3)]))
        inside = len({0, 1, 2, 3} & set(int(i) for i in got.selected_unit_indices))
        assert 2 <= inside <= 3

    def test_restricted_optimum_matches_a_filtered_brute_force(self):
        Y = _panel()
        restrictions = DesignRestrictions(forced_in=[0], forbidden=[8])
        got = solve_synthetic_design_exact(Y=Y, K=3, restrictions=restrictions)
        problem = TwoWayProblem.from_panel(Y)
        allowed = [s for s in combinations(range(problem.N), 3)
                   if 0 in s and 8 not in s]
        subsets = np.array(allowed, dtype=np.int64)
        t, c = split_indices(problem.N, subsets)
        values = solve_partition_batch(problem, t, c, max_iter=8000, tol=1e-14).value
        assert tuple(got.selected_unit_indices) == tuple(subsets[int(np.argmin(values))])

    def test_budget_caps_total_cost(self):
        Y = _panel()
        costs = np.arange(1.0, Y.shape[1] + 1.0)
        got = solve_synthetic_design_exact(Y=Y, K=3, costs=costs, budget=10.0)
        assert costs[got.selected_unit_indices].sum() <= 10.0 + 1e-9

    def test_infeasible_restrictions_raise(self):
        Y = _panel()
        with pytest.raises(MlsynthEstimationError):
            solve_synthetic_design_exact(
                Y=Y, K=2, restrictions=DesignRestrictions(conflict_pairs=[(0, 1)]),
                forbidden_sets=[np.array([i, j]) for i, j in
                                combinations(range(Y.shape[1]), 2)])

    def test_donor_exclusions_are_refused(self):
        """They couple the control weights to the assignment; the search cannot."""
        Y = _panel()
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(
                Y=Y, K=3,
                restrictions=DesignRestrictions(donor_exclusion=[(0, 4)]))


class TestPool:
    def test_returns_distinct_designs_ranked_by_objective(self):
        Y = _panel()
        pool = solve_synthetic_design_exact_pool(Y=Y, K=3, top_K=5)
        assert len(pool) == 5
        values = [d.objective_value for d in pool]
        assert values == sorted(values)
        chosen = [tuple(d.selected_unit_indices) for d in pool]
        assert len(set(chosen)) == 5

    def test_first_entry_is_the_single_optimum(self):
        Y = _panel()
        pool = solve_synthetic_design_exact_pool(Y=Y, K=3, top_K=4)
        single = solve_synthetic_design_exact(Y=Y, K=3)
        assert tuple(pool[0].selected_unit_indices) == tuple(single.selected_unit_indices)
        assert pool[0].objective_value == pytest.approx(single.objective_value, rel=1e-9)

    def test_stops_early_when_the_region_is_exhausted(self):
        Y = _panel(N=4, T=10)
        pool = solve_synthetic_design_exact_pool(Y=Y, K=3, top_K=99)
        assert len(pool) == 4                              # C(4, 3)

    def test_rejects_a_nonpositive_top_k(self):
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact_pool(Y=_panel(), K=3, top_K=0)


# ----------------------------------------------------------------------
# cardinality
# ----------------------------------------------------------------------


class TestCardinality:
    def test_k_none_searches_every_size(self):
        """The paper allows an unpinned K for the global modes."""
        Y = _panel(N=7, T=12)
        got = solve_synthetic_design_exact(Y=Y, K=None)
        assert 1 <= int(got.assignment.sum()) <= 6
        best = min(_brute_force(Y, k)[1] for k in range(1, 7))
        assert got.objective_value == pytest.approx(best, rel=1e-7)

    @pytest.mark.parametrize("K", [1, 8])
    def test_extreme_cardinalities(self, K):
        Y = _panel()
        got = solve_synthetic_design_exact(Y=Y, K=K)
        assert int(got.assignment.sum()) == K

    def test_rejects_out_of_range_k(self):
        Y = _panel()
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(Y=Y, K=0)
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(Y=Y, K=Y.shape[1])


class TestSurvivorPath:
    """When the sign conditions bind, the closed form only bounds and the QP runs.

    A small penalty is what makes them bind: the ridge pulls both weight vectors
    toward uniform, and as it shrinks the unconstrained minimiser starts putting
    negative mass on treated units. The panel is full rank so the Gram matrix
    stays invertible at that penalty.
    """

    # A full-rank panel at a small penalty where the incumbent from the exact
    # rows does not prune everything: only 1 of 120 designs settles in closed
    # form, and 19 survive the screen to reach the inner solver.
    LAM = 0.05
    K = 3

    def _full_rank_panel(self, N=10, T=12, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((T, N))

    def test_some_designs_reach_the_inner_solver(self):
        Y = self._full_rank_panel()
        got = solve_synthetic_design_exact(Y=Y, K=self.K, lam=self.LAM)
        assert got.raw_results["n_solved"] > 0, "the survivor path never ran"
        assert got.raw_results["n_closed_form"] < got.raw_results["n_designs"]

    def test_still_finds_the_brute_force_optimum(self):
        Y = self._full_rank_panel()
        expected_set, expected_value = _brute_force(Y, self.K, lam=self.LAM)
        got = solve_synthetic_design_exact(Y=Y, K=self.K, lam=self.LAM)
        assert tuple(got.selected_unit_indices) == expected_set
        assert got.objective_value == pytest.approx(expected_value, rel=1e-6)

    def test_matches_scip_on_the_binding_instance(self):
        Y = self._full_rank_panel()
        mip = solve_synthetic_design(Y=Y, K=self.K, mode="global_2way",
                                     lam=self.LAM, solver="SCIP")
        got = solve_synthetic_design_exact(Y=Y, K=self.K, lam=self.LAM)
        assert got.objective_value <= float(mip.objective_value) + 1e-8


class TestCostValidation:
    def test_costs_without_budget_raises(self):
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(Y=_panel(), K=3, costs=np.ones(9))

    def test_budget_without_costs_raises(self):
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(Y=_panel(), K=3, budget=5.0)

    def test_wrong_length_costs_raises(self):
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(Y=_panel(), K=3, costs=np.ones(3), budget=5.0)

    def test_negative_costs_raise(self):
        costs = np.ones(9); costs[2] = -1.0
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(Y=_panel(), K=3, costs=costs, budget=5.0)

    def test_nonpositive_budget_raises(self):
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(Y=_panel(), K=3, costs=np.ones(9), budget=0.0)


class TestLargeInstanceFallback:
    def test_falls_back_to_search_when_enumeration_is_too_large(self):
        """Above the candidate limit the answer is a design plus a bound."""
        Y = _panel(N=24, T=30, seed=5)
        got = solve_synthetic_design_exact(Y=Y, K=8, candidate_limit=1000)
        assert int(got.assignment.sum()) == 8
        assert got.raw_results["certified"] is False
        assert got.raw_results["lower_bound"] <= got.objective_value + 1e-9

    def test_enumerates_below_the_limit(self):
        Y = _panel()
        got = solve_synthetic_design_exact(Y=Y, K=3, candidate_limit=10_000)
        assert got.raw_results["certified"] is True

    def test_restrictions_past_the_limit_are_refused(self):
        """The fallback searches the unrestricted space, so it must not pretend."""
        Y = _panel(N=24, T=30, seed=5)
        with pytest.raises(MlsynthConfigError):
            solve_synthetic_design_exact(
                Y=Y, K=8, candidate_limit=1000,
                restrictions=DesignRestrictions(forced_in=[0]))

    def test_ill_conditioned_and_too_large_is_reported(self):
        rng = np.random.default_rng(9)
        Y = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 24))
        with pytest.raises(MlsynthEstimationError):
            solve_synthetic_design_exact(Y=Y, K=8, lam=1e-14, candidate_limit=1000)


def _brute_force_admissible(Y, K, filt, lam=None):
    """The optimum over the sets a filter admits, by generating all and testing.

    This is the semantics the backend had before the enumeration was taught to
    build candidates inside the restrictions, so it is the reference the new walk
    has to reproduce.
    """
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    admissible = [s for s in combinations(range(problem.N), K) if filt.accepts(s)]
    subsets = np.array(admissible, dtype=np.int64)
    t, c = split_indices(problem.N, subsets)
    values = solve_partition_batch(problem, t, c, max_iter=8000, tol=1e-14).value
    best = int(np.argmin(values))
    return tuple(subsets[best]), float(values[best])


class TestRestrictedEnumeration:
    """Structural restrictions shrink the space the enumeration limit reads.

    Forced units, forbidden units and stratum quotas are honoured while
    candidates are built, so the count that decides whether the search may
    enumerate is the count of admissible designs, not ``C(N, K)``. An instance
    whose unrestricted count is past the limit is now solved exactly whenever the
    restrictions leave few enough designs, where before it was refused.

    Conflict pairs, a cost budget and the pool's no-good sets do not decompose
    over a group, so they stay candidate tests and cannot move the gate. The last
    test here pins that, so the improvement is not overstated.
    """

    N, K, LIMIT = 14, 5, 500

    def _Y(self):
        return _panel(N=self.N, T=20, seed=3)

    def _solve(self, restrictions):
        return solve_synthetic_design_exact(
            Y=self._Y(), K=self.K, candidate_limit=self.LIMIT,
            restrictions=restrictions)

    def test_the_unrestricted_instance_is_past_the_limit(self):
        """Without this the rest of the class would prove nothing."""
        assert comb(self.N, self.K) > self.LIMIT
        with pytest.raises(MlsynthConfigError):
            self._solve(DesignRestrictions(conflict_pairs=[(0, 1)]))

    @pytest.mark.parametrize("restrictions", [
        DesignRestrictions(forbidden=[10, 11, 12, 13]),
        DesignRestrictions(forced_in=[0, 1, 2]),
        DesignRestrictions(strata=[(tuple(range(10)), None, 1)]),
        DesignRestrictions(forced_in=[0], forbidden=[11, 12, 13]),
    ])
    def test_restricted_instances_now_enumerate_exactly(self, restrictions):
        got = self._solve(restrictions)
        assert int(got.assignment.sum()) == self.K
        assert got.raw_results["certified"] is True

    @pytest.mark.parametrize("restrictions", [
        DesignRestrictions(forbidden=[10, 11, 12, 13]),
        DesignRestrictions(forced_in=[0, 1, 2]),
        DesignRestrictions(strata=[(tuple(range(10)), None, 1)]),
    ])
    def test_it_reaches_the_same_design_as_generate_then_filter(self, restrictions):
        Y = self._Y()
        filt = SubsetFilter.build(restrictions, N=self.N)
        expected_set, expected_value = _brute_force_admissible(Y, self.K, filt)
        got = self._solve(restrictions)
        assert tuple(got.selected_unit_indices) == expected_set
        assert got.objective_value == pytest.approx(expected_value, rel=1e-6)

    def test_the_reported_design_count_is_the_admissible_count(self):
        restrictions = DesignRestrictions(forbidden=[10, 11, 12, 13])
        got = self._solve(restrictions)
        assert got.raw_results["n_designs"] == comb(10, self.K)

    def test_a_budget_still_applies_inside_a_shrunken_space(self):
        """A non-structural rule keeps working once the gate has been cleared."""
        costs = np.ones(self.N); costs[:3] = 10.0
        got = solve_synthetic_design_exact(
            Y=self._Y(), K=self.K, candidate_limit=self.LIMIT,
            restrictions=DesignRestrictions(forbidden=[10, 11, 12, 13]),
            costs=costs, budget=7.0)
        assert not set(got.selected_unit_indices) & {0, 1, 2}
