"""Checks of the scale study's panels and instruments (#842).

The instruments wrap the library's searches from outside, so the checks are that
they count what the searches report, add what the searches do not, and leave the
library as they found it.
"""
from itertools import combinations

import numpy as np
import pytest

from benchmarks.studies.bnb_scale import marex_scale, panels, syndes_scale
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers import branch as marex_branch
from mlsynth.utils.syndes_helpers import branch as syndes_branch


# ------------------------------------------------------------------ the panels
class TestPanels:

    def test_bls_is_months_by_counties(self):
        Y = panels.bls(60)
        assert Y.shape == (36, 60) and np.isfinite(Y).all()

    def test_bls_sizes_are_nested(self):
        """60 counties are the first 60 of the 100, so the sizes differ only in
        how many units the search sees."""
        np.testing.assert_array_equal(panels.bls(60), panels.bls(100)[:, :60])

    def test_bls_mixes_states(self):
        """FIPS order groups counties by state; the fixed shuffle puts all four
        of the panel's states (Alabama, Oregon, Pennsylvania, Wyoming) in every
        size."""
        assert set(panels.bls_fips(60) // 1000) == {1, 41, 42, 56}

    def test_bls_refuses_more_counties_than_it_has(self):
        with pytest.raises(ValueError, match="193"):
            panels.bls(194)

    @pytest.mark.parametrize("kind", panels.KINDS)
    def test_every_kind_has_the_asked_width(self, kind):
        assert panels.panel(kind, 40).shape[1] == 40

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="kind"):
            panels.panel("smooth", 40)


# --------------------------------------------------------------------- SYNDES
@pytest.fixture(scope="module")
def small():
    """Every node bounded and lifted at 16 units, so each column is exercised."""
    Y = panels.panel("bls", 16)
    return Y, syndes_scale.measure(Y, 8, work_limit=30.0, **syndes_scale.EAGER)


class TestSyndesInstrument:

    def test_it_reports_what_the_search_reports(self, small):
        Y, row = small
        from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem
        from mlsynth.utils.syndes_helpers.exact import SubsetFilter
        problem = TwoWayProblem.from_panel(Y)
        o = syndes_branch.branch_and_bound(problem, BoundEngine.from_problem(problem),
                                           8, SubsetFilter(), work_limit=30.0,
                                           **syndes_scale.EAGER)
        for k in ("nodes", "lifts", "pruned", "priced"):
            assert row[k] == getattr(o, k), k
        assert row["work"] == pytest.approx(o.work)
        assert row["treated"] == [int(i) for i in o.treated]

    def test_it_counts_what_the_search_does_not(self, small):
        _, row = small
        assert row["lifts"] > 0, "the instance exercises no lift"
        assert row["evaluations"] > row["lifts"]        # every lift evaluates often
        assert row["certificates"] >= row["pruned"]     # every discard is certified
        assert row["max_queue"] >= 1
        assert row["cpu_seconds"] > 0 and row["swap_seconds"] > 0

    def test_the_gap_is_what_the_bound_leaves(self, small):
        _, row = small
        assert row["gap"] == pytest.approx((row["value"] - row["lower"]) / row["value"])
        assert row["gap"] >= -1e-9
        if row["certified"]:
            assert row["gap"] <= 1e-6

    def test_the_answer_is_the_enumerated_optimum(self, small):
        """At 16 units the enumeration is 12,870 sets: the search's design is it."""
        Y, row = small
        from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
        best = solve_synthetic_design_exact(Y=Y, K=8)
        assert row["treated"] == sorted(int(i) for i in best.selected_unit_indices)

    def test_the_swap_flag_reads_the_seeds(self, small):
        """``swap_found`` says whether the swap search's seeds held the design,
        complements included, which is when the search's work only proved it."""
        Y, row = small
        from mlsynth.utils.syndes_helpers.gram import (BoundEngine, TwoWayProblem,
                                                       maxcut_candidates)
        eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
        seeds = {tuple(sorted(int(i) for i in s))
                 for s in maxcut_candidates(eng, 8, n_restarts=32, seed=0)}
        seeds |= {tuple(sorted(set(range(16)) - set(s))) for s in seeds}
        assert row["swap_found"] == (tuple(row["treated"]) in seeds)

    def test_the_library_is_left_as_it_was(self, small):
        assert syndes_branch.lift is syndes_scale._ORIGINAL["lift"]
        assert syndes_branch.ExactR.certify is syndes_scale._ORIGINAL["certify"]
        assert syndes_branch.heapq is syndes_scale._ORIGINAL["heapq"]
        assert syndes_branch.maxcut_candidates is syndes_scale._ORIGINAL["maxcut"]


class TestSyndesFailures:

    def test_a_search_that_raises_still_restores_the_library(self):
        with pytest.raises(MlsynthConfigError, match="K must lie"):
            syndes_scale.measure(panels.panel("noise", 10), 11)
        assert syndes_branch.lift is syndes_scale._ORIGINAL["lift"]
        assert syndes_branch.heapq is syndes_scale._ORIGINAL["heapq"]


def test_the_queue_shim_tracks_the_largest_heap():
    shim = syndes_scale._QueueWatch()
    heap = []
    for x in (5, 3, 8, 1):
        shim.heappush(heap, x)
    assert shim.heappop(heap) == 1 and shim.largest == 4
    shim.heappush(heap, 0)
    assert shim.largest == 4


# ---------------------------------------------------------------------- MAREX
class TestMarexInstrument:

    def test_one_bound_is_timed_and_certified(self):
        out = marex_scale.root_bound(panels.panel("bls", 12), 3)
        assert out["seconds"] > 0 and out["lower"] is not None
        assert np.isfinite(out["lower"]) and out["markets"] == 12

    def test_a_closed_search_matches_the_enumeration(self):
        Y = panels.panel("bls", 12)
        row = marex_scale.search(Y, 3, max_lifts=200, price_seconds=1.0)
        assert row["closed"] and row["lifts"] > 0
        B, A = marex_scale.marex_panel(Y)
        from mlsynth.utils.marex_helpers.exact import side_fit
        best = None
        for S in combinations(range(12), 3):
            g, _ = side_fit(B, A, list(S), None)
            h, _ = side_fit(B, A, [j for j in range(12) if j not in S], None)
            if best is None or g + h < best[0]:
                best = (g + h, S)
        assert tuple(row["treated"]) == best[1]
        assert row["gap"] <= 1e-9

    def test_a_search_stopped_at_its_cap_reports_it(self):
        row = marex_scale.search(panels.panel("bls", 12), 3, max_lifts=1,
                                 price_seconds=1.0)
        assert not row["closed"] and row["lifts"] == 1
        assert "computed 1 bounds" in row["stopped"]
        # the bracket the message carries is read back, and brackets the design
        assert np.isfinite(row["lower"]) and row["lower"] <= row["upper"]
        assert row["gap"] == pytest.approx((row["upper"] - row["lower"]) / row["upper"])

    def test_the_library_is_left_as_it_was(self):
        marex_scale.search(panels.panel("bls", 12), 3, max_lifts=1, price_seconds=1.0)
        assert marex_branch.node_bound is marex_scale._ORIGINAL_NODE_BOUND


# -------------------------------------------------------------- inside a lift
class TestLiftProfile:
    """Where one root lift's time goes: the eigendecompositions, the rest of
    the dual's evaluation, or the optimiser and cut separation around them."""

    @pytest.fixture(scope="class")
    def prof(self):
        return syndes_scale.profile_lift(panels.panel("bls", 20), 10)

    def test_the_parts_nest(self, prof):
        assert 0 < prof["eigh_seconds"] <= prof["theta_seconds"] <= prof["profiled_seconds"]

    def test_the_shares_are_fractions_of_the_profiled_lift(self, prof):
        assert prof["eigh_share"] == pytest.approx(
            prof["eigh_seconds"] / prof["profiled_seconds"])
        assert 0 < prof["eigh_share"] <= prof["theta_share"] <= 1

    def test_the_shares_are_read_on_one_clock(self, prof):
        """The dual's evaluation is most of a lift. A share of a few percent
        means the profiler's seconds were divided by another clock's -- CPU
        seconds summed over BLAS threads, which the first version did."""
        assert prof["theta_share"] > 0.25

    def test_profiling_does_not_change_the_lift(self, prof):
        """The lift is deterministic, so the profiled run takes the evaluations
        the timed one did."""
        assert prof["evaluations"] == prof["profiled_evaluations"] > 0
        assert prof["seconds"] > 0
