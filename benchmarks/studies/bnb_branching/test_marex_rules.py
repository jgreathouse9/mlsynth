"""Checks of the #839 MAREX search copy: under the library's choices it is the
library, count for count; under every rule, incumbent and node order it
returns the enumeration's optimum, closed; and the choices each change what the
search does."""
import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.marex_helpers.branch import branch_and_bound
from mlsynth.utils.marex_helpers.exact import side_fit
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

from . import marex_rules as mr

#: Every node is bounded until its subtree holds a single set.
ALWAYS_LIFT = dict(price_seconds=1.0, lift_seconds=lambda k: 0.0)


def _problem(seed, n, T=12):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = f @ rng.uniform(0.4, 1.6, (2, n)) + rng.normal(size=(T, n))
    return B, B.mean(axis=1)


def _brute(B, A, m):
    n = B.shape[1]
    best = np.inf
    for S in itertools.combinations(range(n), m):
        C = [j for j in range(n) if j not in S]
        best = min(best, side_fit(B, A, list(S), None)[0] + side_fit(B, A, C, None)[0])
    return best


@pytest.mark.parametrize("seed, n, m", [(0, 8, 3), (1, 9, 4), (2, 10, 3)])
def test_the_library_choices_are_the_library(seed, n, m):
    B, A = _problem(seed, n)
    lib = branch_and_bound(B, A, [m], filt=SubsetFilter(), **ALWAYS_LIFT)
    got = mr.search(B, A, m, **ALWAYS_LIFT)
    assert (got["lifts"], got["priced"], got["fitted"]) == (
        lib.lifts, lib.priced, lib.certificate.fitted)
    assert got["work"] == pytest.approx(lib.work, rel=1e-12)
    assert got["treated"] == list(lib.designs[0].treated)
    assert got["value"] == pytest.approx(lib.designs[0].value, rel=1e-12)
    assert got["lower"] == pytest.approx(float(lib.certificate.lower), rel=1e-12)
    assert got["closed"] == lib.closed and got["trials"] == 0


@given(st.integers(0, 10 ** 6), st.integers(7, 9), st.integers(2, 4),
       st.sampled_from(mr.RULES), st.booleans(), st.sampled_from(mr.ORDERS))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_every_choice_returns_the_optimum(seed, n, m, rule, guess, order):
    B, A = _problem(seed % 1000, n)
    optimum = _brute(B, A, m)
    got = mr.search(B, A, m, rule=rule, guess=guess, order=order, optimum=optimum,
                    **ALWAYS_LIFT)
    assert got["closed"]
    assert got["value"] == pytest.approx(optimum, rel=1e-9)
    assert got["lower"] <= got["value"] * (1 + 1e-9)
    assert got["near_work"] is not None and 0.0 <= got["near_work"] <= got["work"]


def test_each_rule_branches_differently():
    B, A = _problem(3, 10)
    runs = {rule: mr.search(B, A, 4, rule=rule, **ALWAYS_LIFT) for rule in mr.RULES}
    assert len({(r["lifts"], r["fitted"]) for r in runs.values()}) > 1
    assert runs["pseudo"]["trials"] > 0
    assert all(r["trials"] == 0 for k, r in runs.items() if k != "pseudo")


def test_without_the_guess_the_incumbent_comes_from_pricing():
    """On this panel the guesses fit four designs the bounds would have
    cleared, and the bounds are the same either way."""
    B, A = _problem(3, 11)
    with_guess = mr.search(B, A, 4, **ALWAYS_LIFT)
    without = mr.search(B, A, 4, guess=False, **ALWAYS_LIFT)
    assert without["value"] == pytest.approx(with_guess["value"], rel=1e-12)
    assert (with_guess["fitted"], without["fitted"]) == (5, 1)
    assert with_guess["lifts"] == without["lifts"] == 14


def test_depth_first_is_another_order():
    B, A = _problem(3, 11)
    best = mr.search(B, A, 4, **ALWAYS_LIFT)
    depth = mr.search(B, A, 4, order="depth", **ALWAYS_LIFT)
    assert depth["value"] == pytest.approx(best["value"], rel=1e-12)
    assert (best["lifts"], depth["lifts"]) == (14, 18)


def test_an_unknown_choice_is_refused():
    B, A = _problem(6, 7)
    with pytest.raises(ValueError):
        mr.search(B, A, 3, rule="random")
    with pytest.raises(ValueError):
        mr.search(B, A, 3, order="breadth")


def test_the_runner_writes_one_row_per_choice(monkeypatch, tmp_path):
    """``run marex`` appends a row per rule, guess and order, each naming its
    choices, without touching the study's own results."""
    import json

    from . import run

    calls = []

    def fake(name, m, rule, guess, order):
        calls.append((name, m, rule, guess, order))
        return {"value": 1.0, "work": 0.0}

    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "marex", fake)
    assert run.main(["marex", "walmart", "6", "largest,pseudo", "on,off", "best"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "marex.jsonl").read_text().splitlines()]
    assert [(r["rule"], r["guess"], r["order"]) for r in rows] == [
        ("largest", True, "best"), ("largest", False, "best"),
        ("pseudo", True, "best"), ("pseudo", False, "best")]
    assert calls[0] == ("walmart", 6, "largest", True, "best")
    assert all(r["panel"] == "walmart" and r["m"] == 6 for r in rows)
