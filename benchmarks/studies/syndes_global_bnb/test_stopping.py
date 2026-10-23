"""The lift's stopping near the prune target (#837, candidate 1).

``stopping.lift`` is a study copy of the library's lift with three knobs. The
checks are that the copy is the library's lift when the knobs are off, that
every variant's bound is valid, and that each knob does what it says: a stall
above the floor halves ``alpha`` instead of ending the run, a node within a
margin of its target runs to a lower floor, and a forecast that cannot reach
the target ends the run early.
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

from . import stopping
from .test_dual import _best

SETTINGS = dict(max_examples=15, deadline=None)


def _shock(N, T, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 1)) + rng.standard_normal((T, N))


def _factor(N, T, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
            + 0.3 * rng.standard_normal((T, N)))


def _node(Y, K, I, O):
    R = BoundEngine.from_problem(TwoWayProblem.from_panel(Y)).R
    N = R.shape[0]
    F = [u for u in range(N) if u not in I and u not in O]
    c, b, A = branch.node_parts(R, I, O, F)
    return R, F, (c, b, A, 2 * (K - len(I)) - len(F))


#: A 14-unit node of a factor panel whose lift runs several rounds.
STEEP = dict(Y=_factor(14, 10, 3), K=5, I=[0], O=[1, 2])

nodes = st.builds(
    lambda N, seed, kind, fixed: (N, seed, kind, fixed),
    st.integers(8, 12), st.integers(0, 10 ** 6), st.booleans(), st.integers(0, 3))


def _drawn(N, seed, kind, fixed):
    Y = (_shock if kind else _factor)(N, 8, seed)
    K = N // 2
    I, O = list(range(fixed // 2)), list(range(fixed // 2, fixed))
    return Y, K, I, O


# ----------------------------------------------------------------- the copy
class TestTheCopy:

    @given(nodes, st.booleans())
    @settings(**SETTINGS)
    def test_with_its_knobs_off_it_is_the_librarys_lift(self, node, targeted):
        Y, K, I, O = _drawn(*node)
        _, _, parts = _node(Y, K, I, O)
        target = None
        if targeted:
            tr = branch.trust_region(*parts)[0]
            target = tr * (1 - 1e-3)
        U, _, evaluations = branch.lift(*parts, target=target)
        V, _, copied, info = stopping.lift(*parts, target=target)
        assert V == U and copied == evaluations
        assert info["ended"] in stopping.ENDINGS

    def test_its_multipliers_certify_its_bound(self):
        R, F, parts = _node(**STEEP)
        U, m, _, _ = stopping.lift(*parts, on_stall="halve", alpha_min=1e-7)
        xr = branch.ExactR(BoundEngine.from_problem(TwoWayProblem.from_panel(STEEP["Y"])))
        certified = float(xr.certify(STEEP["I"], STEEP["O"], F, STEEP["K"], m))
        assert certified >= U * (1 - 1e-9)


# ------------------------------------------------------------- the variants
@given(nodes, st.sampled_from(["give_up", "halve"]), st.sampled_from([5e-5, 1e-8]),
       st.booleans(), st.booleans())
@settings(**SETTINGS)
def test_property_every_variant_bounds_every_completion(node, on_stall, floor, margin, forecast):
    Y, K, I, O = _drawn(*node)
    R, F, parts = _node(Y, K, I, O)
    tr = branch.trust_region(*parts)[0]
    U, _, _, _ = stopping.lift(*parts, target=tr * (1 - 2e-3), on_stall=on_stall,
                               alpha_min=floor, margin=0.01 if margin else None,
                               forecast=forecast)
    top = _best(R, I, O, F, K)
    assert U >= top - 1e-10 * abs(top)
    assert U <= tr * (1 + 1e-12)


class TestHalveOnStall:

    def test_a_stall_never_ends_the_run_above_the_floor(self):
        """At the floor nothing is left to halve, so a stall may end the run
        there; whether one comes depends on the BLAS kernel's rounding."""
        _, _, parts = _node(**STEEP)
        _, _, _, info = stopping.lift(*parts, on_stall="halve", alpha_min=1e-8)
        assert info["ended"] != "stall" or info["final_alpha"] == 1e-8
        assert info["final_alpha"] == 1e-8 or info["ended"] == "rounds"

    def test_the_shipped_rule_stalls_above_the_floor_and_halving_reaches_it(self, monkeypatch):
        """With the stall test loosened so it fires, the shipped rule ends
        above its floor, and halving carries the run down to the floor, where
        a stall may end it since nothing is left to halve."""
        monkeypatch.setattr(stopping, "STALL", 1e-2)
        _, _, parts = _node(**STEEP)
        _, _, shipped, a = stopping.lift(*parts, alpha_min=1e-8)
        _, _, halved, b = stopping.lift(*parts, alpha_min=1e-8, on_stall="halve")
        assert a["ended"] == "stall" and a["final_alpha"] > 1e-8
        assert b["final_alpha"] == 1e-8 and halved > shipped


class TestMargin:

    def test_a_target_out_of_reach_never_goes_deep(self):
        _, _, parts = _node(**STEEP)
        tr = branch.trust_region(*parts)[0]
        far = tr * 0.5
        U, _, ev, info = stopping.lift(*parts, target=far, margin=0.01)
        U0, _, ev0, _ = stopping.lift(*parts, target=far)
        assert not info["deep"] and (U, ev) == (U0, ev0)

    def test_a_target_within_the_margin_runs_to_the_lower_floor(self):
        _, _, parts = _node(**STEEP)
        shipped, _, _, _ = stopping.lift(*parts)
        target = shipped * (1 - 1e-4)            # just under where the shipped run ends
        U, _, _, info = stopping.lift(*parts, target=target, margin=0.01)
        assert info["deep"] and U < shipped
        assert info["final_alpha"] < branch.ALPHA_MIN or info["ended"] == "target"


class TestForecast:

    def test_a_target_out_of_reach_ends_the_run_early(self):
        _, _, parts = _node(**STEEP)
        tr = branch.trust_region(*parts)[0]
        _, _, ev, info = stopping.lift(*parts, target=tr * 0.5, forecast=True)
        _, _, ev0, _ = stopping.lift(*parts, target=tr * 0.5)
        assert info["ended"] == "forecast" and ev < ev0

    def test_a_target_in_reach_is_still_reached(self):
        _, _, parts = _node(**STEEP)
        shipped, _, _, _ = stopping.lift(*parts)
        tr = branch.trust_region(*parts)[0]
        target = shipped + 0.5 * (tr - shipped)
        U, _, _, info = stopping.lift(*parts, target=target, forecast=True)
        assert U <= target and info["ended"] == "target"

    def test_it_waits_for_its_alpha(self):
        """A forecast that starts below any alpha the run reaches never
        applies, and the run is the plain one."""
        _, _, parts = _node(**STEEP)
        tr = branch.trust_region(*parts)[0]
        late = stopping.lift(*parts, target=tr * 0.5, forecast=True, forecast_from=1e-12)
        plain = stopping.lift(*parts, target=tr * 0.5)
        assert (late[0], late[2]) == (plain[0], plain[2]) and late[3]["ended"] != "forecast"

    def test_without_a_target_there_is_nothing_to_forecast(self):
        _, _, parts = _node(**STEEP)
        with_forecast, plain = stopping.lift(*parts, forecast=True), stopping.lift(*parts)
        assert (with_forecast[0], with_forecast[2]) == (plain[0], plain[2])


@pytest.mark.parametrize("bad", [dict(on_stall="wait"), dict(margin=-0.1),
                                 dict(alpha_min=0.0), dict(forecast_horizon=0),
                                 dict(forecast_from=0.0)])
def test_an_invalid_knob_is_refused(bad):
    _, _, parts = _node(**STEEP)
    with pytest.raises(ValueError):
        stopping.lift(*parts, **bad)


# ------------------------------------------------------------- the ratios
@pytest.fixture(scope="module")
def table():
    return stopping.ratios(_factor(10, 8, 1), 3, depths=(0, 2, 4))


class TestRatios:

    def test_a_row_per_node_with_every_variant(self, table):
        assert [r["fixed"] for r in table] == [0, 2, 4]
        for r in table:
            assert set(stopping.VARIANTS) <= set(r)
            assert {f"{k}_targeted" for k in stopping.TARGETED} <= set(r)

    def test_no_bound_passes_the_subtrees_minimum(self, table):
        for r in table:
            assert r["trust_region"] <= 1 + 1e-9
            assert all(r[k]["ratio"] <= 1 + 1e-9 for k in stopping.VARIANTS)

    def test_a_node_prunes_only_where_its_bound_could(self, table):
        """A prune needs a ratio past ``best / subtree``; the subtree holding
        the optimum has a need of one and can never prune."""
        for r in table:
            if r["need"] >= 1 - 1e-12:
                assert not any(r[f"{k}_targeted"]["pruned"] for k in stopping.TARGETED)


# ------------------------------------------------------------- the search
class TestSearch:

    def test_with_its_knobs_off_it_is_the_librarys_search(self):
        """Every node bounded and lifted, so every lift is compared."""
        from mlsynth.utils.syndes_helpers.exact import SubsetFilter
        eager = dict(price_seconds=1.0, bound_seconds=lambda n: 0.0,
                     lift_seconds=lambda n: 0.0)
        Y = _factor(14, 10, 3)
        problem = TwoWayProblem.from_panel(Y)
        o = branch.branch_and_bound(problem, BoundEngine.from_problem(problem), 5,
                                    SubsetFilter(), **eager)
        row = stopping.search(Y, 5, costs=eager)
        assert row["lifts"] > 3
        for k in ("nodes", "lifts", "pruned", "priced", "value", "lower"):
            assert row[k] == getattr(o, k), k
        assert row["closed"] and row["certified"]

    def test_the_library_lift_comes_back_after_a_raise(self, monkeypatch):
        shipped = branch.lift

        def boom(*a, **k):
            raise RuntimeError("search failed")

        monkeypatch.setattr(branch, "branch_and_bound", boom)
        with pytest.raises(RuntimeError, match="search failed"):
            stopping.search(_factor(10, 8, 0), 3)
        assert branch.lift is shipped

    def test_a_variant_reaches_the_same_optimum(self):
        Y = _factor(14, 10, 3)
        plain = stopping.search(Y, 5)
        margined = stopping.search(Y, 5, margin=0.01, forecast=True, forecast_from=1e-3)
        assert margined["value"] == plain["value"] and margined["certified"]
        assert sum(margined["endings"].values()) == margined["lifts"]


class TestNodes40:

    @pytest.fixture(scope="class")
    def rows(self):
        Y = _factor(12, 8, 2)
        optimum = stopping.search(Y, 4)["value"]
        return stopping.nodes40(Y, 4, optimum, depths=(0, 3), closure=False), optimum

    def test_every_variant_lies_under_the_trust_region(self, rows):
        for r in rows[0]:
            assert all(r[k]["U"] <= r["trust_region"] * (1 + 1e-12) for k in stopping.VARIANTS)

    def test_the_root_holds_the_optimum_and_never_prunes(self, rows):
        table, _ = rows
        assert [r["fixed"] for r in table] == [0, 3]
        assert not any(table[0][f"{k}_targeted"]["pruned"] for k in stopping.TARGETED)


def test_the_runner_appends_a_row_per_measurement(tmp_path, monkeypatch):
    import json
    from . import run
    monkeypatch.setattr(stopping, "RESULTS", tmp_path)
    monkeypatch.setitem(run.PANELS, "tiny", lambda N: _factor(N, 8, 2))
    assert stopping.main(["ratios", "tiny", "10", "3"]) == 0
    from . import cuts
    monkeypatch.setattr(cuts, "closures", lambda Y, K: [{"closed": K}])
    assert stopping.main(["closures", "tiny", "10", "3"]) == 0
    assert json.loads(open(tmp_path / "stopping_closures.jsonl").readline())["closed"] == 3
    assert stopping.main(["search", "tiny", "10", "3", "margin"]) == 0
    optimum = json.loads(open(tmp_path / "stopping_search.jsonl").readline())["value"]
    monkeypatch.setattr(stopping, "nodes40", lambda Y, K, opt: [{"optimum": opt}])
    assert stopping.main(["nodes", "tiny", "10", "3", str(optimum)]) == 0
    ratios = [json.loads(l) for l in open(tmp_path / "stopping_ratios.jsonl")]
    assert len(ratios) == len(stopping.DEPTHS) and all(r["panel"] == "tiny" for r in ratios)
    searched = json.loads(open(tmp_path / "stopping_search.jsonl").readline())
    assert searched["variant"] == "margin" and searched["knobs"] == {"margin": 0.005}
    assert json.loads(open(tmp_path / "stopping_nodes.jsonl").readline())["optimum"] == optimum
    with pytest.raises(ValueError, match="no mode"):
        stopping.main(["walk", "tiny", "10", "3"])
    assert stopping.main(["limit", "bls", "12", "forecast_late"]) == 0
    limited = json.loads(open(tmp_path / "stopping_limit.jsonl").readline())
    assert limited["panel"] == "bls" and limited["K"] == 6 and limited["variant"] == "forecast_late"


def test_a_node_with_nothing_free_to_vary_costs_no_evaluation():
    U, _, ev, info = stopping.lift(3.5, np.zeros(5), np.zeros((5, 5)), 1)
    assert U == 3.5 and ev == 0 and info["ended"] == "trust_region"


def test_the_closure_bounds_every_variant_from_below():
    """The closure is the value the lift's dual converges to, so no variant's
    bound passes under it."""
    Y = _factor(12, 8, 2)
    optimum = stopping.search(Y, 4)["value"]
    rows = stopping.nodes40(Y, 4, optimum, depths=(2,))
    r = rows[0]
    assert all(r[k]["U"] >= r["closure"] * (1 - 1e-6) for k in stopping.VARIANTS)
    assert r["closure"] <= r["trust_region"] * (1 + 1e-9)


def test_the_limit_mode_searches_half_treated_under_the_librarys_limit(tmp_path, monkeypatch):
    """The ``limit`` rows stand for the library's search at its default work
    limit; a run without the limit would close and read as a certified gap."""
    from mlsynth.utils.syndes_helpers.exact import WORK_LIMIT
    calls = []
    monkeypatch.setattr(stopping, "RESULTS", tmp_path)
    monkeypatch.setattr(stopping, "search", lambda Y, K, **kw: calls.append((Y.shape, K, kw)) or {})
    assert stopping.main(["limit", "noise", "14", "margin"]) == 0
    (shape, K, kw), = calls
    assert shape[1] == 14 and K == 7
    assert kw == {"work_limit": WORK_LIMIT, "margin": 0.005}
