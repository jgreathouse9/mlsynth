"""Checks of the stage ledger (#844).

The ledger times the library's two-way search by wrapping its stages. The
checks are that each wrapped call is charged its own time and no other, that
the wrapped search decides exactly what the library decides, that every
wrapper comes off whether the search returns or raises, and that the cost
model's charge per stage is the one the library adds to its work.
"""
import time

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.marex_helpers import branch as marex_branch
from mlsynth.utils.marex_helpers.certify import Dyadic
from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

from . import stages


def _factor(N, T, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
            + 0.3 * rng.standard_normal((T, N)))


def _library(Y, K, **kw):
    problem = TwoWayProblem.from_panel(Y)
    o = branch.branch_and_bound(problem, BoundEngine.from_problem(problem), K,
                                SubsetFilter(), **kw)
    return (o.nodes, o.lifts, o.pruned, o.priced, o.work, o.value, o.lower)


def _busy(seconds):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


# ----------------------------------------------------------------------
# the ledger
# ----------------------------------------------------------------------


def test_a_call_is_charged_its_own_time_and_its_callee_the_rest():
    ledger = stages.Ledger()
    inner = ledger.wrap("inner", lambda: _busy(0.02))

    def outer_body():
        _busy(0.01)
        inner()
    outer = ledger.wrap("outer", outer_body)
    outer()
    assert ledger.calls == {"outer": 1, "inner": 1}
    assert ledger.own["inner"] == pytest.approx(0.02, abs=0.008)
    assert ledger.own["outer"] == pytest.approx(0.01, abs=0.008)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.integers(0, 3), min_size=1, max_size=12))
def test_property_the_own_times_add_to_the_outermost_calls(path):
    """However the wrapped calls nest, their own times sum to the time of the
    calls made from outside every wrapper: nothing is charged twice or lost.
    Stage ``d`` calls stage ``d + 1``, so a call at ``d`` nests ``3 - d`` deep."""
    ledger = stages.Ledger()
    fns = {}

    def make(d):
        def body():
            _busy(0.0002)
            if d < 3:
                fns[d + 1]()
        return body
    for d in range(4):
        fns[d] = ledger.wrap(f"s{d}", make(d))
    t0 = time.perf_counter()
    for d in path:
        fns[d]()
    outside = time.perf_counter() - t0
    assert sum(ledger.calls.values()) == sum(4 - d for d in path)
    assert 0.5 * outside <= sum(ledger.own.values()) <= outside


def test_a_nested_label_names_its_caller():
    ledger = stages.Ledger()
    leaf = ledger.wrap("eigh", lambda: None, under_caller=True)
    a = ledger.wrap("lift", leaf)
    b = ledger.wrap("trust region", leaf)
    a(); b(); b(); leaf()
    assert ledger.calls == {"lift": 1, "lift > eigh": 1, "trust region": 2,
                            "trust region > eigh": 2, "eigh": 1}


def test_a_generator_is_charged_for_each_step_not_its_call():
    ledger = stages.Ledger()

    def blocks():
        for _ in range(3):
            _busy(0.005)
            yield 1
    stream = ledger.wrap("enumeration", blocks)
    outer = ledger.wrap("tree", lambda: [_busy(0.01) for _ in stream()])
    outer()
    assert ledger.calls["enumeration"] == 4          # three blocks and the end
    assert ledger.own["enumeration"] == pytest.approx(0.015, abs=0.006)
    assert ledger.own["tree"] == pytest.approx(0.03, abs=0.01)


def test_a_raising_call_is_still_charged_and_unwinds():
    ledger = stages.Ledger()

    def boom():
        _busy(0.005)
        raise ValueError("x")
    outer = ledger.wrap("outer", ledger.wrap("boom", boom))
    with pytest.raises(ValueError):
        outer()
    assert ledger.calls == {"outer": 1, "boom": 1}
    assert ledger.own["boom"] >= 0.004
    assert ledger._stack == []


# ----------------------------------------------------------------------
# the search under the ledger
# ----------------------------------------------------------------------


#: Pricing dear enough that a 14-unit search bounds, lifts and prices: under
#: #845's costs a lift takes about five times what it was charged before, and
#: at 1e-4 a set the search priced every subtree and lifted nothing.
COSTS = {"price_seconds": 1e-3}


@pytest.fixture(scope="module")
def flat():
    return _factor(14, 8, 3)


def test_the_wrapped_search_decides_what_the_library_decides(flat):
    row = stages.profile(flat, 5, costs=COSTS)
    expected = _library(flat, 5, **COSTS)
    got = tuple(row[k] for k in ("nodes", "lifts", "pruned", "priced", "work",
                                 "value", "lower"))
    assert got == expected


def test_every_stage_that_ran_is_charged_and_the_rest_is_the_tree(flat):
    row = stages.profile(flat, 5, costs=COSTS)
    st_ = row["stages"]
    assert row["lifts"] > 0 and row["priced"] > 0
    for label in ("pricing: closed form", "pricing: sign matrices", "trust region",
                  "lift", "lift: dual evaluation", "lift: dual evaluation > eigh",
                  "trust region > eigh", "lift: separation", "lift: L-BFGS-B",
                  "certificate: assembly", "seed: swap search"):
        assert st_[label]["calls"] > 0 and st_[label]["seconds"] > 0, label
    charged = sum(v["seconds"] for v in st_.values())
    assert charged <= row["seconds"]
    assert row["rest"] == pytest.approx(row["seconds"] - charged)


def test_the_modelled_charges_add_to_the_librarys_work(flat):
    """The cost model's charge per stage, summed, is the work the library
    reports, and one trust region more: the bound that orders the root's units,
    which the library does not charge. The ledger reads the model the search
    spends against."""
    row = stages.profile(flat, 5, costs=COSTS)
    modelled = sum(v["modelled"] for v in row["stages"].values())
    assert modelled == pytest.approx(row["work"] + branch.trust_region_seconds(14), rel=1e-12)
    assert row["stages"]["lift"]["modelled"] > 0
    assert row["stages"]["seed: swap search"]["modelled"] == 0.0


def test_the_certificates_parts_are_charged_when_a_node_prunes():
    """A noise panel prunes nodes, so the certificate runs; its exact parts
    are charged under their own labels."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((8, 1)) + rng.standard_normal((8, 14))
    row = stages.profile(Y, 5, costs={"price_seconds": 1e-3})
    assert row["pruned"] > 0
    for label in ("certificate: lambda_min_lower", "certificate: to floating point",
                  "certificate: exact product"):
        assert row["stages"][label]["calls"] > 0, label


def test_the_wrappers_come_off_whether_the_search_returns_or_raises(flat, monkeypatch):
    before = {(id(o), a): getattr(o, a) for o, a, *_ in stages.STAGES}
    stages.profile(flat, 5)
    assert {(id(o), a): getattr(o, a) for o, a, *_ in stages.STAGES} == before

    def fail(*a, **k):
        raise RuntimeError("search failed")
    monkeypatch.setattr(branch, "branch_and_bound", fail)
    with pytest.raises(RuntimeError, match="search failed"):
        stages.profile(flat, 5)
    assert {(id(o), a): getattr(o, a) for o, a, *_ in stages.STAGES} == before
    assert np.linalg.eigh is before[(id(np.linalg), "eigh")]
    assert marex_branch._float is before[(id(marex_branch), "_float")]
    assert Dyadic.__matmul__ is before[(id(Dyadic), "__matmul__")]


def test_the_work_limit_reaches_the_search(flat):
    row = stages.profile(flat, 5, work_limit=0.0)
    assert not row["closed"] and row["nodes"] == 0


def test_the_runner_appends_a_row(tmp_path, monkeypatch):
    import json
    from benchmarks.studies.syndes_global_bnb import run
    monkeypatch.setattr(stages, "RESULTS", tmp_path)
    monkeypatch.setitem(run.PANELS, "tiny", lambda N: _factor(N, 8, 3))
    assert stages.main(["tiny", "12", "4"]) == 0
    assert stages.main(["tiny", "12", "4", "limit"]) == 0
    rows = [json.loads(l) for l in open(tmp_path / "stages.jsonl")]
    assert [r["panel"] for r in rows] == ["tiny", "tiny"]
    assert np.isinf(rows[0]["work_limit"]) and rows[1]["work_limit"] == branch.WORK_LIMIT
    assert stages.main(["bls", "12", "6"]) == 0
    assert json.loads(open(tmp_path / "stages.jsonl").readlines()[-1])["panel"] == "bls"
