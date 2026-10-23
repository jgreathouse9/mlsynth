"""Checks of the scale study's panels and instruments (#842).

The instruments wrap the library's searches from outside, so the checks are that
they count what the searches report, add what the searches do not, and leave the
library as they found it.
"""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
        """At 16 units the enumeration is 12,870 sets: the search's design is it.
        At K = N/2 a set and its complement are the same design, and which of
        the two comes back depends on the last digits, so either is the
        answer (on a CI runner the search returned the complement)."""
        Y, row = small
        from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
        best = solve_synthetic_design_exact(Y=Y, K=8)
        chosen = sorted(int(i) for i in best.selected_unit_indices)
        mirror = sorted(set(range(16)) - set(chosen))
        assert row["treated"] in (chosen, mirror)

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
        row = marex_scale.search(Y, 3, price_seconds=1.0)
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

    def test_a_search_stopped_at_its_limit_reports_it(self):
        row = marex_scale.search(panels.panel("bls", 12), 3, work_limit=1.5,
                                 price_seconds=1.0)
        assert not row["closed"] and row["lifts"] == 1
        assert "open at the limit" in row["stopped"] and row["work"] <= 1.5
        # the certificate's bracket holds the design
        assert np.isfinite(row["lower"]) and row["lower"] <= row["upper"]
        assert row["gap"] == pytest.approx((row["upper"] - row["lower"]) / row["upper"])

    def test_a_search_stopped_before_any_design_reports_it(self):
        row = marex_scale.search(panels.panel("bls", 12), 3, work_limit=0.5,
                                 price_seconds=1.0)
        assert not row["closed"] and row["treated"] is None
        assert "work_limit" in row["stopped"] and np.isnan(row["gap"])

    def test_the_library_is_left_as_it_was(self):
        marex_scale.search(panels.panel("bls", 12), 3, work_limit=1.5, price_seconds=1.0)
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

    def test_the_rest_of_the_lift_is_split_without_overlap(self, prof):
        """Cut separation and SciPy's L-BFGS-B (its Fortran core and Python
        driver, own time only) lie outside the dual's evaluation, so the three
        shares are disjoint parts of one lift."""
        assert prof["separate_share"] > 0 and prof["solver_share"] > 0
        assert (prof["theta_share"] + prof["separate_share"]
                + prof["solver_share"]) <= 1 + 1e-9
        assert prof["separate_share"] == pytest.approx(
            prof["separate_seconds"] / prof["profiled_seconds"])
        assert prof["solver_share"] == pytest.approx(
            prof["solver_seconds"] / prof["profiled_seconds"])

    def test_profiling_does_not_change_the_lift(self, prof):
        """The lift is deterministic, so the profiled run takes the evaluations
        the timed one did."""
        assert prof["evaluations"] == prof["profiled_evaluations"] > 0
        assert prof["seconds"] > 0


# ------------------------------------------------------------ the root's slack
class TestRootBounds:
    """Where the root's gap comes from: the bound the search uses, the same
    dual run to a finer stop, and the relaxation solved by interior point."""

    @pytest.fixture(scope="class")
    def small(self):
        from benchmarks.studies.bnb_scale import root_gap
        from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
        Y = panels.panel("bls", 12)
        best = solve_synthetic_design_exact(Y=Y, K=6)
        return root_gap.root_bounds(Y, 6, incumbent=float(best.objective_value))

    def test_every_bound_is_valid(self, small):
        """Each is a lower bound on the enumerated optimum."""
        for k in ("trust_region", "lift", "lift_long", "interior"):
            assert small[k] <= small["incumbent"] * (1 + 1e-9), k

    def test_the_lift_is_at_least_the_trust_region(self, small):
        assert small["lift"] >= small["trust_region"] * (1 - 1e-12)

    def test_a_finer_stop_never_falls_below_the_trust_region(self, small):
        """The finer stop is neither a better bound nor a longer run in
        general: on 40 BLS counties it ended after 702 evaluations to the
        shipped 761, on a certificate 8e-6 weaker, because the stall rule ended
        it first (``TestLiftTrace``). Malick and Roupin's Table 3 has the same
        reversal across alpha (ex108a, k = 63). What holds is the floor every
        lift keeps."""
        assert small["lift_long"] >= small["trust_region"] * (1 - 1e-12)
        assert small["evaluations_long"] > 0

    def test_the_gaps_read_the_bounds(self, small):
        for k in ("trust_region", "lift", "lift_long", "interior"):
            assert small[f"{k}_gap"] == pytest.approx(
                (small["incumbent"] - small[k]) / small["incumbent"])

    def test_the_library_schedule_is_restored(self, small):
        from benchmarks.studies.bnb_scale import root_gap
        for name, value in root_gap._DEFAULTS.items():
            assert getattr(syndes_branch, name) == value


class TestSchedule:
    """The lift's stopping constants, replaced for one call and put back."""

    def test_every_name_it_sets_is_restored(self):
        from benchmarks.studies.bnb_scale import root_gap
        before = {name: getattr(syndes_branch, name) for name in root_gap.NO_STALL}
        with root_gap.schedule(**root_gap.NO_STALL):
            assert syndes_branch.MIN_ROUNDS == root_gap.NO_STALL["MIN_ROUNDS"]
        assert {name: getattr(syndes_branch, name) for name in root_gap.NO_STALL} == before

    def test_a_raise_inside_still_restores(self):
        from benchmarks.studies.bnb_scale import root_gap
        before = syndes_branch.MIN_ROUNDS
        with pytest.raises(ZeroDivisionError):
            with root_gap.schedule(MIN_ROUNDS=7):
                1 / 0
        assert syndes_branch.MIN_ROUNDS == before

    def test_an_unknown_name_is_refused(self):
        """A misspelt constant would otherwise be set beside the real one and
        change nothing, so the run would measure the shipped schedule."""
        from benchmarks.studies.bnb_scale import root_gap
        with pytest.raises(ValueError, match="ALPHA_MN"):
            with root_gap.schedule(ALPHA_MN=1e-8):
                pass  # pragma: no cover - the context refuses before the body runs
        assert not hasattr(syndes_branch, "ALPHA_MN")


class TestLiftTrace:
    """Which rule ends a root lift: its rounds, read from the lift as it runs,
    and the lift's own stopping tests applied to them."""

    @pytest.fixture(scope="class")
    def traces(self):
        from benchmarks.studies.bnb_scale import root_gap
        Y = panels.panel("bls", 12)
        return Y, {"shipped": root_gap.trace_lift(Y, 6),
                   "long": root_gap.trace_lift(Y, 6, **root_gap.LONG),
                   "no_stall": root_gap.trace_lift(Y, 6, **root_gap.NO_STALL)}

    def test_it_reports_the_lift_it_traced(self, traces):
        Y, t = traces
        from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
        parts = syndes_branch.node_parts(engine.R, (), (), tuple(range(12)))
        U, _, evaluations = syndes_branch.lift(*parts, 0)
        assert t["shipped"]["U"] == U and t["shipped"]["evaluations"] == evaluations
        assert t["shipped"]["lower"] == pytest.approx(4 * engine.alpha / U)

    @pytest.mark.parametrize("run", ["shipped", "long", "no_stall"])
    def test_exactly_the_last_round_meets_a_stopping_rule(self, traces, run):
        """The lift breaks out at the first round that meets one, so a replica
        of its rules that is wrong anywhere marks a round too early or none."""
        rounds = traces[1][run]["rounds"]
        assert rounds and [r["stops"] for r in rounds] == [False] * (len(rounds) - 1) + [True]

    def test_the_reason_names_the_rule_that_held(self, traces):
        for run in traces[1].values():
            last = run["rounds"][-1]
            assert run["stopped"] in ("floor", "stall", "rounds")
            if run["stopped"] == "floor":
                assert last["at_floor"]
            else:
                assert last["giveup"] and not last["at_floor"]

    def test_with_the_stall_rule_off_the_run_ends_at_its_floor(self, traces):
        from benchmarks.studies.bnb_scale import root_gap
        run = traces[1]["no_stall"]
        assert run["stopped"] == "floor"
        assert run["final_alpha"] == root_gap.NO_STALL["ALPHA_MIN"]

    def test_the_library_is_left_as_it_was(self, traces):
        from benchmarks.studies.bnb_scale import root_gap
        assert syndes_branch._theta is root_gap._ORIGINAL["theta"]
        assert syndes_branch.separate is root_gap._ORIGINAL["separate"]
        assert syndes_branch.Cuts.concat is root_gap._ORIGINAL["concat"]
        for name, value in root_gap._DEFAULTS.items():
            assert getattr(syndes_branch, name) == value


def _round(alpha, smooth, added):
    """One traced round: the evaluation at the round's solution, the
    separation, and the concatenation when triangles were added."""
    return [("theta", alpha, smooth), ("separate",)] + ([("concat",)] if added else [])


class TestStoppingRules:
    """The replica of the lift's stopping rules, on rounds written by hand.
    The rounds a real lift takes depend on the BLAS kernel's last digits (on a
    CI runner the 40-county root took 824 evaluations to this machine's 761,
    and a finer floor ran on where it stalled here), so the rules are held on
    rounds whose answer is fixed."""

    def test_a_round_at_the_floor_that_adds_triangles_does_not_stop(self):
        from benchmarks.studies.bnb_scale import root_gap
        events = (_round(0.1, 10.0, True) + _round(5e-5, 9.0, True)
                  + _round(5e-5, 8.9, False))
        rounds = root_gap._rounds(events, alpha_min=5e-5, min_rounds=12, max_rounds=100)
        assert [r["stops"] for r in rounds] == [False, False, True]
        assert root_gap._stopped(rounds, max_rounds=100) == "floor"

    def test_a_round_meeting_both_rules_ended_at_its_floor(self):
        from benchmarks.studies.bnb_scale import root_gap
        events = (_round(0.1, 10.0, True) + _round(5e-5, 9.0, True)
                  + _round(5e-5, 9.0, False))
        rounds = root_gap._rounds(events, alpha_min=5e-5, min_rounds=2, max_rounds=100)
        assert rounds[-1]["at_floor"] and rounds[-1]["giveup"]
        assert root_gap._stopped(rounds, max_rounds=100) == "floor"

    def test_a_stall_above_the_floor_is_named(self):
        from benchmarks.studies.bnb_scale import root_gap
        events = _round(1e-5, 9.0, True) + _round(1e-5, 9.0, True)
        rounds = root_gap._rounds(events, alpha_min=1e-8, min_rounds=2, max_rounds=100)
        assert [r["stops"] for r in rounds] == [False, True]
        assert root_gap._stopped(rounds, max_rounds=100) == "stall"

    def test_the_round_limit_is_named(self):
        from benchmarks.studies.bnb_scale import root_gap
        events = _round(0.1, 10.0, True) + _round(0.1, 9.0, True)
        rounds = root_gap._rounds(events, alpha_min=1e-8, min_rounds=1, max_rounds=2)
        assert [r["stops"] for r in rounds] == [False, True]
        assert root_gap._stopped(rounds, max_rounds=2) == "rounds"

    def test_a_change_over_the_stall_tolerance_is_not_a_stall(self):
        """1.1e-5 relative is over the lift's 1e-6, so the run goes on."""
        from benchmarks.studies.bnb_scale import root_gap
        events = _round(5e-5, 9.0, True) + _round(5e-5, 9.0001, True)
        rounds = root_gap._rounds(events, alpha_min=1e-8, min_rounds=1, max_rounds=100)
        assert not rounds[-1]["stalled"] and not rounds[-1]["stops"]

    def test_a_stall_before_min_rounds_does_not_stop(self):
        from benchmarks.studies.bnb_scale import root_gap
        events = _round(5e-5, 9.0, True) + _round(5e-5, 9.0, True)
        rounds = root_gap._rounds(events, alpha_min=1e-8, min_rounds=12, max_rounds=100)
        assert rounds[-1]["stalled"] and not rounds[-1]["stops"]


class TestInstrumentsPinBlas:
    """Every instrument runs the library on one BLAS thread, whatever the
    caller's setting. Unpinned under four test workers, SciPy's L-BFGS-B core
    took 10.7 s of an 11.8 s lift at 20 units (#857), and the profile read
    the thread pool, not the lift."""

    @pytest.fixture
    def caller_threads(self):
        from threadpoolctl import threadpool_info, threadpool_limits
        with threadpool_limits(2, user_api="blas"):
            threads = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]
            if max(threads) < 2:  # pragma: no cover - CI and development machines run several threads; this guards a single-core runner
                pytest.skip("BLAS cannot run two threads here")
            yield threads

    @pytest.mark.parametrize("run", ["measure", "profile", "trace", "root"])
    def test_the_library_runs_on_one_thread(self, caller_threads, run, monkeypatch):
        from threadpoolctl import threadpool_info
        from benchmarks.studies.bnb_scale import root_gap
        seen = []
        eigh = np.linalg.eigh

        def spy(*args, **kwargs):
            if not seen:
                seen.append([p["num_threads"] for p in threadpool_info()
                             if p["user_api"] == "blas"])
            return eigh(*args, **kwargs)

        monkeypatch.setattr(np.linalg, "eigh", spy)
        Y = panels.panel("bls", 10)
        {"measure": lambda: syndes_scale.measure(Y, 5, work_limit=5.0, **syndes_scale.EAGER),
         "profile": lambda: syndes_scale.profile_lift(Y, 5),
         "trace": lambda: root_gap.trace_lift(Y, 5),
         "root": lambda: root_gap.root_bounds(Y, 5, incumbent=1.0, interior=False)}[run]()
        assert seen and set(seen[0]) == {1}
        assert [p["num_threads"] for p in threadpool_info()
                if p["user_api"] == "blas"] == caller_threads


class TestTriangleClosure:
    """The relaxation the lift's dual converges to, solved by interior point:
    Shor with the cardinality and its products, the RLT products, and every
    triangle cut a separation round finds, until none is violated."""

    @pytest.fixture(scope="class")
    def small(self):
        from benchmarks.studies.bnb_scale import root_gap
        from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
        Y = panels.panel("bls", 12)
        best = float(solve_synthetic_design_exact(Y=Y, K=6).objective_value)
        bounds = root_gap.root_bounds(Y, 6, incumbent=best)
        closure = root_gap.triangle_closure(Y, 6)
        return best, bounds, closure

    def test_it_is_a_lower_bound_on_the_optimum(self, small):
        best, _, closure = small
        assert closure["lower"] <= best * (1 + 1e-7)

    def test_triangles_only_tighten_the_interior_bound(self, small):
        _, bounds, closure = small
        assert closure["lower"] >= bounds["interior"] * (1 - 1e-7)

    def test_the_loop_ends_with_no_violated_triangle(self, small):
        _, _, closure = small
        assert closure["violated"] == 0 and closure["rounds"] >= 1


def test_a_seed_brings_its_complement():
    """At K = N/2 a set and its complement are the same design, and the search
    tries both, so ``swap_found`` must count a design the seeds held only as a
    complement. The 16-unit fixture's design is a seed itself, which is why
    this is checked directly."""
    held = syndes_scale._with_complements([np.array([[0, 1], [1, 3]])], 4)
    assert held == {(0, 1), (2, 3), (1, 3), (0, 2)}


@settings(max_examples=12, deadline=None)
@given(kind=st.sampled_from(panels.KINDS), N=st.integers(6, 14),
       run=st.sampled_from(["shipped", "long", "no_stall"]))
def test_property_only_the_last_traced_round_stops(kind, N, run):
    """Over panels, sizes and schedules: the replica of the lift's stopping
    rules marks the round the lift actually stopped at, and no other."""
    from benchmarks.studies.bnb_scale import root_gap
    values = {"shipped": {}, "long": root_gap.LONG, "no_stall": root_gap.NO_STALL}[run]
    rounds = root_gap.trace_lift(panels.panel(kind, N), N // 2, **values)["rounds"]
    assert [r["stops"] for r in rounds] == [False] * (len(rounds) - 1) + [True]


def test_the_runner_records_every_measurement(tmp_path, monkeypatch):
    """Every subcommand at 12 units, into a scratch results folder: each
    appends its rows with the panel and the load, ``root`` and ``trace`` read
    the design ``syndes`` found, and ``trace`` runs its three schedules."""
    import json
    from benchmarks.studies.bnb_scale import run
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    args = ["--kinds", "bls", "--sizes", "12"]
    assert run.main(["syndes", *args]) == 0
    assert run.main(["root", *args, "--closure"]) == 0
    assert run.main(["trace", *args]) == 0
    assert run.main(["profile", *args]) == 0
    assert run.main(["marex-bound", *args, "--m", "3"]) == 0
    assert run.main(["marex-search", *args, "--m", "3", "--work-limit", "5"]) == 0
    rows = {f.stem: [json.loads(line) for line in open(f)] for f in tmp_path.glob("*.jsonl")}
    assert set(rows) == {"syndes", "root", "trace", "lift_profile",
                         "marex-bound", "marex-search"}
    for name, written in rows.items():
        assert all(r["kind"] == "bls" and "load_before" in r for r in written), name
    assert [r["run"] for r in rows["trace"]] == ["shipped", "long", "no_stall"]
    best = rows["syndes"][0]["value"]
    assert rows["root"][0]["incumbent"] == best
    assert all(r["incumbent"] == best for r in rows["trace"])
    assert "closure_gap" in rows["root"][0]
