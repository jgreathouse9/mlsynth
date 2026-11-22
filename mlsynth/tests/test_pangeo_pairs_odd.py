"""PANGEO's pairs (Q = 1) on the exact path for an arm with an odd number of units: one market is left out.

An odd arm has no perfect matching, so the design leaves out the one market whose removal lets the rest pair at
the least cost (CLAUDE.md invariant 8, closed forms first):

* under the supergeo paper's objective without covariates, the market left out sits at an even position of the
  sorted totals (dropping position k odd is never better than dropping k - 1: the two designs differ only in that
  the market at k + 1 pairs with k instead of k - 1, and k is nearer), so one scan of prefix and suffix sums of
  the neighbour costs finds it (``pairs.sorted_pairing_drop_one``);
* under every other cost, the blossom matcher's dummy market, joined to every market at cost 0, picks it
  (Kaibel and Walter, arXiv:1703.09505, 2017).

The market left out is recorded on ``ArmDesign.excluded_units`` and ``PangeoResults.excluded_units``, named in a
warning, and kept out of the assignment and the treated and control sets. An explicit MIP backend and the fast
path still raise on an odd arm.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import PANGEO
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import pairs as ppairs
from mlsynth.utils.pangeo_helpers import prepare_pangeo_inputs, run_pangeo
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs


def _long(Y, arms=None, cov=None):
    n, T = Y.shape
    df = pd.DataFrame({"unit": np.repeat([f"u{i:02d}" for i in range(n)], T),
                       "time": np.tile(np.arange(T), n), "sales": Y.ravel(),
                       "arm": np.repeat(["A"] * n if arms is None else list(arms), T)})
    if cov is not None:
        df["size"] = np.repeat(cov, T)
    return df


def _panel(seed, n, T=10):
    rng = np.random.default_rng(seed)
    return 100 + rng.normal(0, 8, (n, 1)) + rng.normal(0, 1, (n, T)).cumsum(axis=1)


def _design(Y, arms=None, cov=None, **kw):
    inp = prepare_pangeo_inputs(_long(Y, arms, cov), "sales", "arm", "unit", "time",
                                covariates=None if cov is None else ["size"])
    kw.setdefault("max_supergeo_size", 1)
    kw.setdefault("fast", False)
    kw.setdefault("compute_power", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_pangeo(inp, **kw)
    return inp, res, caught


def _matching_cost(W, units):
    """The least cost of a perfect matching of ``units`` under W, by recursion on the first unit."""
    units = tuple(units)
    if not units:
        return 0.0
    i, rest = units[0], units[1:]
    return min(W[i][j] + _matching_cost(W, tuple(k for k in rest if k != j)) for j in rest)


def _best_drops(W):
    """The least cost over every market left out, and the markets that attain it."""
    n = len(W)
    costs = {d: _matching_cost(W, [u for u in range(n) if u != d]) for d in range(n)}
    best = min(costs.values())
    return best, {d for d, c in costs.items() if c <= best * (1 + 1e-12) + 1e-9}


def _cost_matrix(inp, res, objective):
    idx = inp.arm_units["A"]
    e = np.arange(res.metadata["n_estimation"])
    w = None
    if objective == "weighted":
        raw = 0.97 ** (len(e) - 1 - np.arange(len(e)))
        w = raw / raw.sum() * len(e)
    cands = enumerate_candidate_pairs(idx, inp.Y[:, e], 1, objective=objective, weights=w,
                                      cov=inp.covariates, cov_scales=inp.covariate_scales)
    n = len(idx)
    W = np.zeros((n, n))
    for c in cands:
        a, b = (int(u) for u in c["members"])
        W[a, b] = W[b, a] = c["score"]
    return W


def _design_cost(inp, res, W):
    pos = {u: i for i, u in enumerate(inp.unit_names)}
    return sum(W[pos[p.treatment[0]], pos[p.control[0]]] for p in res.arm_designs["A"].pairs)


# --- the closed form ----------------------------------------------------------------------------------------------

def test_the_closed_form_leaves_out_the_market_that_frees_the_cheapest_pairing():
    """Smoke: totals 0, 1, 10, 11, 50 pair as (0, 1) and (10, 11) and leave out 50."""
    pairs, dropped = ppairs.sorted_pairing_drop_one(np.array([10.0, 50.0, 0.0, 11.0, 1.0]))
    assert dropped == 1 and sorted(pairs) == [(0, 3), (2, 4)]


def test_one_market_is_left_out_alone():
    """Edge: a single total is left out, with no pairs."""
    assert ppairs.sorted_pairing_drop_one(np.array([3.0])) == ([], 0)


def test_ties_leave_out_the_lowest_position_in_sorted_order():
    """Unit: equal totals tie every drop at cost 0; the market left out is the first in sorted order (the
    lowest total, then the lowest position), so the design is a function of the totals and their order."""
    pairs, dropped = ppairs.sorted_pairing_drop_one(np.array([5.0, 5.0, 5.0]))
    assert dropped == 0 and pairs == [(1, 2)]


@settings(max_examples=300, deadline=None)
@given(z=st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=7).map(
    lambda v: np.array(v if len(v) % 2 else v[:-1] or [0.0], dtype=float)))
def test_the_closed_form_is_the_brute_force_optimum(z):
    """Property, over odd counts up to 7 (ties included): every other position is paired once, the cost of
    the pairs equals the least over every market left out and every pairing of the rest, and the market left
    out is one that attains it."""
    pairs, dropped = ppairs.sorted_pairing_drop_one(z)
    assert sorted([v for p in pairs for v in p] + [dropped]) == list(range(len(z)))
    W = (z[:, None] - z[None, :]) ** 2
    best, drops = _best_drops(W)
    assert sum(W[a, b] for a, b in pairs) == pytest.approx(best, rel=1e-12, abs=1e-6)
    assert dropped in drops


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 10 ** 6), half=st.integers(0, 30))
def test_the_closed_form_and_the_matcher_agree_on_odd_counts(seed, half):
    """Property (invariant 8, the two solvers on the cases they share): on odd counts up to 61, the closed
    form's cost equals the blossom matcher's, which leaves one market out through its dummy."""
    from mlsynth.utils.pangeo_helpers import matching
    z = np.random.default_rng(seed).lognormal(0, 1.5, 2 * half + 1) * 1000
    W = (z[:, None] - z[None, :]) ** 2
    pairs, _ = ppairs.sorted_pairing_drop_one(z)
    assert sum(W[a, b] for a, b in pairs) == pytest.approx(matching.min_weight_matching(W)["cost"],
                                                           rel=1e-12, abs=1e-9)


def test_the_closed_form_rejects_an_even_count():
    """Failure: an even count leaves no market out; the drop-one closed form raises."""
    with pytest.raises(ValueError, match="odd"):
        ppairs.sorted_pairing_drop_one(np.arange(4.0))


# --- the design ---------------------------------------------------------------------------------------------------

def test_an_odd_arm_under_the_papers_objective_leaves_out_the_closed_forms_market():
    """Unit: an arm of 9 under supergeo_total designs 4 pairs by the closed form; the market left out is the
    closed form's on the estimation-window totals, it is recorded on the arm and the result, named in a
    warning, and absent from the assignment and the treated and control sets."""
    inp, res, caught = _design(_panel(0, 9), objective="supergeo_total")
    arm = res.arm_designs["A"]
    z = inp.Y[:, :res.metadata["n_estimation"]].sum(axis=1)
    _, dropped = ppairs.sorted_pairing_drop_one(z)
    gone = inp.unit_names[dropped]
    assert arm.excluded_units == [gone] and res.excluded_units == [gone]
    assert len(arm.pairs) == 4 and arm.n_units == 9
    assert gone not in res.assignment and gone not in arm.treatment_units + arm.control_units
    assert any(gone in str(w.message) and "left out" in str(w.message) for w in caught)
    d = res.metadata["solver_diagnostics"]["A"]
    assert d["path"] == "closed_form_sorted_pairs" and d["excluded_units"] == [int(dropped)]


@pytest.mark.parametrize("objective, with_cov", [("ss_res", False), ("r2", False), ("weighted", False),
                                                 ("supergeo_total", True)])
def test_an_odd_arm_under_any_other_cost_is_matched_with_one_left_out(objective, with_cov):
    """Unit: an arm of 7 under a cost with no closed form takes the matcher, whose dummy leaves one market
    out; the design costs the least over every market left out and every pairing of the rest."""
    cov = np.random.default_rng(1).normal(0, 1, 7) if with_cov else None
    inp, res, _ = _design(_panel(1, 7), cov=cov, objective=objective)
    d = res.metadata["solver_diagnostics"]["A"]
    assert d["path"] == "blossom_matching" and len(res.arm_designs["A"].pairs) == 3
    W = _cost_matrix(inp, res, objective)
    best, drops = _best_drops(W)
    assert _design_cost(inp, res, W) == pytest.approx(best, rel=1e-9, abs=1e-9)
    assert [inp.unit_names.index(u) for u in res.excluded_units][0] in drops


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10 ** 6), half=st.integers(1, 3),
       objective=st.sampled_from(["supergeo_total", "ss_res", "r2"]))
def test_odd_designs_over_the_domain(seed, half, objective):
    """Property, over odd arms of 3 to 7 units and three objectives: one market left out, the rest paired
    once, the left-out market in no pair and not in the assignment, and the cost the brute-force optimum."""
    inp, res, _ = _design(_panel(seed, 2 * half + 1), objective=objective)
    arm = res.arm_designs["A"]
    paired = [u for p in arm.pairs for u in p.treatment + p.control]
    assert len(res.excluded_units) == 1 and sorted(paired + res.excluded_units) == sorted(inp.unit_names)
    assert set(res.assignment) == set(paired)
    W = _cost_matrix(inp, res, objective)
    assert _design_cost(inp, res, W) == pytest.approx(_best_drops(W)[0], rel=1e-9, abs=1e-9)


def test_an_even_arm_leaves_nothing_out_and_warns_nothing():
    """Edge: an even arm records no excluded unit and emits no warning about one."""
    _, res, caught = _design(_panel(2, 8), objective="supergeo_total")
    assert res.arm_designs["A"].excluded_units == [] and res.excluded_units == []
    assert not any("left out" in str(w.message) for w in caught)


def test_two_arms_one_odd_one_even():
    """Unit: an odd arm and an even arm; only the odd arm leaves a market out, and the result lists it."""
    inp, res, _ = _design(_panel(3, 13), arms=["A"] * 6 + ["B"] * 7, objective="supergeo_total")
    assert res.arm_designs["A"].excluded_units == [] and len(res.arm_designs["B"].excluded_units) == 1
    assert res.excluded_units == res.arm_designs["B"].excluded_units
    assert res.excluded_units[0] in inp.unit_names[6:]


def test_three_units_make_one_pair():
    """Edge: an arm of 3 is one pair and one market left out."""
    _, res, _ = _design(_panel(4, 3), objective="ss_res")
    assert len(res.arm_designs["A"].pairs) == 1 and len(res.excluded_units) == 1


def test_the_analysis_run_leaves_out_the_same_market():
    """Unit: a post_col fit leaves out the market the design-only fit left out, and its effects use only the
    paired markets."""
    Y = _panel(5, 9, T=16)
    df = _long(Y)
    df["post"] = (df["time"] >= 12).astype(int)
    base = dict(outcome="sales", arm="arm", unitid="unit", time="time", max_supergeo_size=1, fast=False,
                objective="supergeo_total", display_graphs=False, compute_power=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = PANGEO(dict(base, df=df[df.post == 0].drop(columns="post"))).fit()
        analysis = PANGEO(dict(base, df=df, post_col="post")).fit()
    assert design.excluded_units == analysis.excluded_units and len(design.excluded_units) == 1
    assert analysis.effects is not None


# --- what still raises --------------------------------------------------------------------------------------------

def test_too_few_pairs_in_an_odd_arm_is_reported():
    """Failure: an arm of 7 gives 3 pairs; min_pairs 4 raises MlsynthEstimationError naming min_pairs."""
    with pytest.raises(MlsynthEstimationError, match="min_pairs"):
        _design(_panel(6, 7), objective="supergeo_total", min_pairs=4)


def test_one_unit_gives_no_pairs():
    """Failure: solve_pairs on a single unit leaves it out and has no pairs, fewer than min_pairs 1."""
    with pytest.raises(MlsynthEstimationError, match="min_pairs"):
        ppairs.solve_pairs(None, np.arange(1), totals=np.zeros(1))


@pytest.mark.parametrize("kw", [{"mip_backend": "cvxpy"}, {"fast": True}])
def test_the_mip_and_the_fast_path_still_raise_on_an_odd_arm(kw):
    """Failure: an explicit MIP backend and the fast path have no leave-one-out; an odd arm still raises."""
    with pytest.raises(MlsynthEstimationError):
        _design(_panel(7, 5), objective="ss_res", **kw)


# --- the sweep's candidates and automatic Q -----------------------------------------------------------------------

def _excluded_warnings(caught):
    from mlsynth.utils.pangeo_helpers.structures import PangeoExcludedUnitWarning
    return [w for w in caught if issubclass(w.category, PangeoExcludedUnitWarning)]


def test_the_sweeps_candidates_do_not_warn():
    """Unit: a fixed Q = 2 design with the diagnostic Q sweep on an odd arm leaves nothing out, and the sweep's
    Q = 1 candidate, which does leave one out, warns nothing: only the design returned names its markets."""
    _, res, caught = _design(_panel(8, 9, T=16), objective="ss_res", max_supergeo_size=2,
                             compute_power=True, compute_q_sweep=True)
    assert any(r["q"] == 1 and r["feasible"] for r in res.metadata["q_sweep"])
    assert res.excluded_units == [] and _excluded_warnings(caught) == []


def test_automatic_q_names_the_market_its_design_leaves_out(monkeypatch):
    """Unit: when automatic Q returns the pairs design of an odd arm, it warns once for the market that design
    leaves out, and the result records it; the sweep's other candidates add no warning."""
    from mlsynth.utils.pangeo_helpers import pipeline
    monkeypatch.setattr(pipeline, "_select_q", lambda sweep, **kw: 1)
    _, res, caught = _design(_panel(9, 9, T=16), objective="ss_res", max_supergeo_size=None,
                             compute_power=True)
    assert res.max_supergeo_size == 1 and len(res.excluded_units) == 1
    named = _excluded_warnings(caught)
    assert len(named) == 1 and res.excluded_units[0] in str(named[0].message)
