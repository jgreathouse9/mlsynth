"""PANGEO's pairs (Q = 1) on the exact path: a closed form where one exists, the blossom matcher elsewhere.

With Q = 1 every supergeo is one unit, so a design is a perfect matching of each arm's units and the exact-cover
MIP is a minimum-weight perfect matching (Edmonds, "Maximum matching and a polyhedron with 0,1-vertices", J. Res.
NBS 69B (1965) 125-130). Two cases follow (CLAUDE.md invariant 8, closed forms first):

* the supergeo paper's objective, (total_a - total_b)^2 over the estimation window (Chen, Doudchenko, Jiang, Stein
  and Ying, arXiv:2301.12044, eq. 3), with no covariate penalty: a cost of a difference of scalars, minimised by
  pairing neighbours in sorted order (an exchange argument: two crossing pairs are never cheaper than the two
  pairs that uncross them); ``pairs.sorted_pairing``;
* every other Q = 1 cost (the trajectory objectives, or any objective with a covariate penalty): no closed form,
  solved by the blossom matcher with its dual certificate; ``matching.min_weight_matching``.

``mip_backend="auto"`` takes these paths; ``"highspy"`` and ``"cvxpy"`` keep the MIP. The tests pin each path
against the MIP and against brute force, the routing, the diagnostics, and the failures (odd arms, too few
pairs).
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import PANGEO
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import mip as pmip
from mlsynth.utils.pangeo_helpers import pairs as ppairs
from mlsynth.utils.pangeo_helpers import pipeline as ppipe
from mlsynth.utils.pangeo_helpers import prepare_pangeo_inputs, run_pangeo
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs


def _long(Y, cov=None):
    n, T = Y.shape
    df = pd.DataFrame({"unit": np.repeat([f"u{i:02d}" for i in range(n)], T),
                       "time": np.tile(np.arange(T), n), "sales": Y.ravel(), "arm": "A"})
    if cov is not None:
        df["size"] = np.repeat(cov, T)
    return df


def _panel(seed, n, T=10):
    rng = np.random.default_rng(seed)
    return 100 + rng.normal(0, 8, (n, 1)) + rng.normal(0, 1, (n, T)).cumsum(axis=1)


def _design(Y, cov=None, **kw):
    inp = prepare_pangeo_inputs(_long(Y, cov), "sales", "arm", "unit", "time",
                                covariates=None if cov is None else ["size"])
    kw.setdefault("max_supergeo_size", 1)
    kw.setdefault("fast", False)
    kw.setdefault("compute_power", False)
    return inp, run_pangeo(inp, **kw)


def _cost(res, inp, objective="supergeo_total", cov=None):
    """The design's total score, recomputed from the candidates (the objective the paths share)."""
    idx = inp.arm_units["A"]
    n_e = res.metadata["n_estimation"]
    e = np.arange(n_e)
    w = None
    if objective == "weighted":                      # the pipeline's recency weights at the default decay
        raw = 0.97 ** (n_e - 1 - np.arange(n_e))
        w = raw / raw.sum() * n_e
    cands = enumerate_candidate_pairs(idx, inp.Y[:, e], 1, objective=objective, weights=w,
                                      cov=inp.covariates, cov_scales=inp.covariate_scales)
    score = {frozenset(int(u) for u in c["members"]): c["score"] for c in cands}
    pos = {u: i for i, u in enumerate(inp.unit_names)}
    return sum(score[frozenset(pos[u] for u in p.treatment + p.control)] for p in res.arm_designs["A"].pairs)


def _brute(W):
    """The least cost of a perfect matching of range(len(W)), by recursion on the lowest index."""
    def best(rest):
        if not rest:
            return 0.0
        i, others = rest[0], rest[1:]
        return min(W[i][j] + best(tuple(k for k in others if k != j)) for j in others)
    return best(tuple(range(len(W))))


# --- the closed form ----------------------------------------------------------------------------------------------

def test_sorted_pairing_pairs_neighbours():
    """Smoke: totals 10, 1, 11, 2 pair as (1, 3) and (0, 2), the neighbours in sorted order."""
    assert sorted(ppairs.sorted_pairing(np.array([10.0, 1.0, 11.0, 2.0]))) == [(0, 2), (1, 3)]


def test_sorted_pairing_breaks_ties_by_position():
    """Unit: equal totals are ordered by position, so the pairing is a function of the totals and their order."""
    assert ppairs.sorted_pairing(np.array([5.0, 5.0, 5.0, 5.0])) == [(0, 1), (2, 3)]


@settings(max_examples=200, deadline=None)
@given(z=st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=0, max_size=5).map(
    lambda v: np.array(v + v[:len(v) % 2], dtype=float)))
def test_sorted_pairing_is_the_brute_force_optimum(z):
    """Property, over even counts up to 6 (ties included, by duplication): the sorted pairing's cost
    sum (z_a - z_b)^2 equals the least over every perfect matching, and every position is paired once."""
    pairs = ppairs.sorted_pairing(z)
    assert sorted(v for p in pairs for v in p) == list(range(len(z)))
    W = (z[:, None] - z[None, :]) ** 2
    assert sum(W[a, b] for a, b in pairs) == pytest.approx(_brute(W), rel=1e-12, abs=1e-6)


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 10 ** 6), half=st.integers(1, 30))
def test_the_closed_form_and_the_matcher_agree_where_both_apply(seed, half):
    """Property (invariant 8: the two solvers on the cases they share): on even counts up to 60, the sorted
    pairing's cost equals the blossom matcher's on the same cost matrix."""
    from mlsynth.utils.pangeo_helpers import matching
    z = np.random.default_rng(seed).lognormal(0, 1.5, 2 * half) * 1000
    W = (z[:, None] - z[None, :]) ** 2
    got = sum(W[a, b] for a, b in ppairs.sorted_pairing(z))
    assert got == pytest.approx(matching.min_weight_matching(W)["cost"], rel=1e-12)


def test_sorted_pairing_rejects_an_odd_count():
    """Failure: an odd count has no perfect matching; the closed form raises instead of leaving one out."""
    with pytest.raises(ValueError, match="even"):
        ppairs.sorted_pairing(np.arange(5.0))


# --- the routing --------------------------------------------------------------------------------------------------

def _no_mip(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the MIP was called")
    monkeypatch.setattr(ppipe, "solve_partition", fail)


def test_auto_sorts_the_papers_objective_without_covariates(monkeypatch):
    """Unit: supergeo_total with no covariates takes the closed form; the MIP is not called, and the
    diagnostics name the path and the proof."""
    _no_mip(monkeypatch)
    inp, res = _design(_panel(0, 10), objective="supergeo_total")
    d = res.metadata["solver_diagnostics"]["A"]
    assert d["path"] == "closed_form_sorted_pairs" and d["proved_optimal"] is True
    assert d["n_selected_pairs"] == 5 == len(res.arm_designs["A"].pairs)
    assert d["objective_value"] == pytest.approx(_cost(res, inp))


@pytest.mark.parametrize("objective", ["ss_res", "r2", "weighted"])
def test_auto_matches_the_trajectory_objectives(monkeypatch, objective):
    """Unit: the trajectory objectives have no closed form; they take the matcher, whose certificate is on the
    diagnostics."""
    _no_mip(monkeypatch)
    _, res = _design(_panel(1, 10), objective=objective)
    d = res.metadata["solver_diagnostics"]["A"]
    assert d["path"] == "blossom_matching" and d["proved_optimal"] is True
    assert d["certificate_violation"] <= d["certificate_tolerance"]


def test_auto_matches_the_papers_objective_with_covariates(monkeypatch):
    """Unit: a covariate penalty breaks the closed form's condition (the cost is no longer a function of a
    difference of scalars), so supergeo_total with covariates takes the matcher."""
    _no_mip(monkeypatch)
    _, res = _design(_panel(2, 10), cov=np.random.default_rng(2).normal(0, 1, 10), objective="supergeo_total")
    assert res.metadata["solver_diagnostics"]["A"]["path"] == "blossom_matching"


@pytest.mark.parametrize("backend", ["cvxpy"] + (["highspy"] if pmip._highspy_available() else []))
def test_an_explicit_mip_backend_keeps_the_mip(backend):
    """Unit: asking for highspy or cvxpy runs the exact-cover MIP, as before."""
    _, res = _design(_panel(3, 8), objective="supergeo_total", mip_backend=backend)
    d = res.metadata["solver_diagnostics"]["A"]
    assert d["path"] == "exact_mip" and d["backend"] == backend


def test_larger_supergeos_keep_the_mip():
    """Unit: Q = 2 is set partitioning, not a matching; auto runs the MIP."""
    _, res = _design(_panel(4, 8), objective="ss_res", max_supergeo_size=2)
    assert res.metadata["solver_diagnostics"]["A"]["path"] == "exact_mip"


def test_the_solver_label_names_the_path():
    """Unit: metadata["solver"] says which path designed the pairs."""
    _, a = _design(_panel(5, 8), objective="supergeo_total")
    _, b = _design(_panel(5, 8), objective="ss_res")
    assert "sorted" in a.metadata["solver"] and "blossom" in b.metadata["solver"]


# --- each path against the MIP ------------------------------------------------------------------------------------

@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10 ** 6), half=st.integers(1, 5),
       objective=st.sampled_from(["supergeo_total", "ss_res", "r2", "weighted"]), with_cov=st.booleans())
def test_every_path_reaches_the_mips_optimum(seed, half, objective, with_cov):
    """Property, over panels of 2 to 10 units, every Q = 1 objective, with and without a covariate: the auto
    path's design costs no more than the MIP's (cvxpy, stopped at HiGHS's relative gap of 1e-4), and the MIP's
    no more than the auto path's within that gap."""
    Y = _panel(seed, 2 * half)
    cov = np.random.default_rng(seed + 1).normal(0, 1, 2 * half) if with_cov else None
    inp, auto = _design(Y, cov=cov, objective=objective)
    _, mip = _design(Y, cov=cov, objective=objective, mip_backend="cvxpy")
    a, m = _cost(auto, inp, objective), _cost(mip, inp, objective)
    assert a <= m * (1 + 1e-9) + 1e-9 and m <= a * (1 + 1e-4) + 1e-6


def test_the_trimmed_match_scale_is_solved_in_closed_form(monkeypatch):
    """Unit: 174 units, the size of google/trimmed_match's example, design by sorting with no MIP; the cost is
    the sorted pairing's of the estimation-window totals."""
    _no_mip(monkeypatch)
    Y = np.random.default_rng(6).lognormal(5, 1.3, (174, 1)) * np.random.default_rng(7).lognormal(0, 0.1, (174, 20))
    inp, res = _design(Y, objective="supergeo_total")
    z = inp.Y[:, :res.metadata["n_estimation"]].sum(axis=1)
    s = np.sort(z)
    assert _cost(res, inp) == pytest.approx(float(((s[1::2] - s[0::2]) ** 2).sum()), rel=1e-12)


# --- edges and failures -------------------------------------------------------------------------------------------

def test_two_units_form_the_one_pair():
    """Edge: an arm of two units is one pair on either path."""
    for objective in ("supergeo_total", "ss_res"):
        _, res = _design(_panel(8, 2), objective=objective)
        assert len(res.arm_designs["A"].pairs) == 1


def test_identical_units_tie_and_any_pairing_is_optimal():
    """Edge: every unit the same, every pairing costs 0; both paths return a design of cost 0."""
    Y = np.tile(_panel(9, 1), (6, 1))
    for objective in ("supergeo_total", "ss_res"):
        inp, res = _design(Y, objective=objective)
        assert _cost(res, inp, objective) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("objective", ["supergeo_total", "ss_res"])
def test_an_odd_arm_is_reported(objective):
    """Failure: pairs need an even arm; an arm of 5 raises MlsynthEstimationError naming the count, as the MIP
    path's infeasible cover does."""
    with pytest.raises(MlsynthEstimationError, match="even"):
        _design(_panel(10, 5), objective=objective)


@pytest.mark.parametrize("objective", ["supergeo_total", "ss_res"])
def test_too_few_pairs_for_min_pairs_is_reported(objective):
    """Failure: an arm of 6 gives 3 pairs; min_pairs 4 raises MlsynthEstimationError on either path."""
    with pytest.raises(MlsynthEstimationError, match="min_pairs"):
        _design(_panel(11, 6), objective=objective, min_pairs=4)


def test_an_empty_arm_gives_no_pairs():
    """Edge: solve_pairs on no units reports that zero pairs fall short of min_pairs."""
    with pytest.raises(MlsynthEstimationError, match="min_pairs"):
        ppairs.solve_pairs([], np.arange(0), totals=np.zeros(0))


def test_solve_pairs_rejects_a_missing_candidate():
    """Failure: the matcher needs every pair's score; a candidate list missing one raises."""
    Y = _panel(12, 4)
    idx = np.arange(4)
    cands = enumerate_candidate_pairs(idx, Y, 1, objective="ss_res")[1:]
    with pytest.raises(MlsynthEstimationError, match="score"):
        ppairs.solve_pairs(cands, idx)
