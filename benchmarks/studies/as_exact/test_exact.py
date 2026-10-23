"""The cvxpy-free solver against the authors' own code.

The oracle is the replication package at the pinned commit (reference.py), captured
with its inputs at full precision in ``results/reference.json`` by ``capture.py``: the
paper's two applications on the authors' data, the leakage-free Proposition 99 variant,
and 24 seeded random exterior instances. The tests read the bundle, so they run
offline; ``test_live_against_authors_code`` reruns the authors' code on fresh instances
when ``MLSYNTH_AS_EXACT_LIVE=1``.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from . import exact, rationalize
from .capture import random_instance
from mlsynth.utils.solvers.ridge_augment import simplex_qp

BUNDLE = json.loads((pathlib.Path(__file__).resolve().parent / "results" / "reference.json").read_text())["cases"]
CASES = {c["name"]: c for c in BUNDLE}


def _inputs(case):
    return tuple(np.array(case[k], float) for k in ("X1", "X0", "Z1", "Z0"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_diagonal_class_matches_authors_code(name):
    case = CASES[name]
    loss, w = exact.solve_diag(*_inputs(case))
    assert abs(loss - case["diag"]["loss"]) <= 1e-9 * max(1.0, abs(case["diag"]["loss"]))
    assert np.abs(w - np.array(case["diag"]["w"])).max() <= 1e-7
    assert w.min() >= 0.0 and abs(w.sum() - 1.0) <= 1e-12


@pytest.mark.parametrize("name", sorted(CASES))
def test_positive_definite_class_matches_authors_code(name):
    case = CASES[name]
    loss, w = exact.solve_pd(*_inputs(case))
    assert abs(loss - case["pd"]["loss"]) <= 1e-9 * max(1.0, abs(case["pd"]["loss"]))
    assert np.abs(w - np.array(case["pd"]["w"])).max() <= 1e-7


@pytest.mark.parametrize("name", [n for n, c in CASES.items() if "V" in c])
def test_predictor_weights_match_authors_code(name):
    """V from the authors' code, to the tolerance their solves used.

    The replication solves the diagonal programs and the analytic centres at Clarabel's
    default tolerances, so coordinates agree to about 1e-5 on a nearly flat objective;
    the log-det maximiser here must be at least as good by the objective itself. The PD
    log-det maximiser was solved at 1e-12 and is compared at 1e-6.
    """
    case = CASES[name]
    X1, X0, _, _ = _inputs(case)
    w = np.array(case["diag"]["w"])
    V = case["V"]
    d = rationalize.diag_rationalizers(X1, X0, w)
    theirs = np.array(V["diag_logdet"])
    sup = d["support"]
    assert np.sum(np.log(d["logdet"][sup])) >= np.sum(np.log(theirs[sup])) - 1e-9
    assert np.abs(d["logdet"] - theirs).max() <= 1e-4
    assert np.abs(d["center"] - np.array(V["diag_center"])).max() <= 1e-4
    assert [V["predictors"][k] for k in d["forced_zero"]] == V["diag_forced_zero"]
    p = rationalize.pd_rationalizers(X1, X0, w, ranges=True)
    assert np.abs(p["logdet_diag"] - np.array(V["pd_logdet_diag"])).max() <= 1e-6
    assert np.abs(p["center_diag"] - np.array(V["pd_center_diag"])).max() <= 1e-4
    assert np.abs(p["lo"] - np.array(V["pd_lo"])).max() <= 1e-6
    assert np.abs(p["hi"] - np.array(V["pd_hi"])).max() <= 1e-6
    assert abs(p["logdet_min_eig"] - V["pd_logdet_min_eig"]) <= 1e-4 * V["pd_logdet_min_eig"]


def test_proposition_99_diagonal_weights_are_uniform_on_the_retained_predictors():
    """Section 4.5.2: the match is exact on the three retained predictors, so the
    log-det maximiser puts one third on each (Table 2)."""
    case = CASES["prop99"]
    X1, X0, _, _ = _inputs(case)
    d = rationalize.diag_rationalizers(X1, X0, np.array(case["diag"]["w"]))
    retained = [case["V"]["predictors"][k] for k in d["support"]]
    assert retained == ["Share of population aged 15-24", "Retail cigarette price",
                        "Cigarette sales per capita (1975)"]
    assert np.abs(d["logdet"][d["support"]] - 1 / 3).max() <= 1e-8


def test_the_search_reports_its_work():
    """Proposition 99 reaches one face below a ridge, searched by branch and bound over sign sub-cubes."""
    st, st_pd = {}, {}
    exact.solve_diag(*_inputs(CASES["prop99"]), stats=st)
    exact.solve_pd(*_inputs(CASES["prop99"]), stats=st_pd)
    assert st["n_visible"] == st_pd["n_visible"] == 132
    assert st["pattern_tests"] > 0 and st["pattern_lp_checks"] == 0
    assert st["solves"] >= st["faces"] > 0


def test_proposition_99_headline_numbers():
    """Table 1 and Table D.1 of the paper, as the authors' code returns them."""
    case = CASES["prop99"]
    donors = case["donors"]
    loss, w = exact.solve_diag(*_inputs(case))
    got = {donors[j]: round(float(w[j]), 3) for j in np.flatnonzero(w)}
    assert got == {"Colorado": 0.160, "Connecticut": 0.068, "Montana": 0.202, "Nevada": 0.236, "Utah": 0.335}
    assert round(loss, 2) == 58.46
    loss_pd, _ = exact.solve_pd(*_inputs(case))
    assert round(loss_pd, 2) == 54.70


# --------------------------------------------------------------------------- failures
def test_an_interior_treated_unit_goes_to_the_interior_search():
    """Inside the donor hull both classes select among exact matches (interior.py, paper sec. 5)."""
    from . import interior
    from .test_interior import CASES as INTERIOR
    args = _inputs(INTERIOR["random_interior_K3_J12_seed4007"])
    for solve, inner in ((exact.solve_diag, interior.solve_interior_diag),
                         (exact.solve_pd, interior.solve_interior_pd)):
        st = {}
        loss, w = solve(*args, stats=st)
        want_loss, want_w = inner(*args)
        assert loss == want_loss and np.array_equal(w, want_w) and st["regime"] == "interior"


def test_the_search_reports_the_exterior_case():
    for solve in (exact.solve_diag, exact.solve_pd):
        st = {}
        solve(*_inputs(CASES["prop99"]), stats=st)
        assert st["regime"] == "exterior"


def test_a_treated_unit_on_the_hull_s_boundary_is_refused():
    """X1 on an edge of the hull is matched exactly by two donors, fewer than K + 1: it is outside the
    hull for neither case and strictly inside it for neither (Assumptions 1(i) and 2(ii))."""
    X0 = np.array([[0.0, 4.0, 0.0, 1.0], [0.0, 0.0, 4.0, 1.0]])
    for solve in (exact.solve_diag, exact.solve_pd):
        with pytest.raises(ValueError, match="boundary of the donor hull"):
            solve(np.array([2.0, 0.0]), X0, np.zeros(1), np.zeros((1, 4)))


def test_predictors_that_sum_to_a_constant_are_refused():
    """Shares of a whole make the donor hull lower-dimensional, which the search cannot use."""
    rng = np.random.default_rng(0)
    shares = rng.dirichlet(np.ones(3), size=21).T          # 3 x 21, columns sum to one
    X = np.vstack([shares, rng.normal(size=(2, 21))])
    X[:, 0] = X[:, 1:].max(axis=1) + 1.0                   # treated unit outside the hull
    X[:3, 0] = [0.2, 0.3, 0.5]
    Z = rng.normal(size=(8, 21))
    with pytest.raises(ValueError):
        exact.solve_diag(X[:, 0], X[:, 1:], Z[:, 0], Z[:, 1:])


def test_mismatched_shapes_are_refused():
    X1, X0, Z1, Z0 = _inputs(CASES["germany"])
    with pytest.raises(ValueError, match="expected"):
        exact.solve_diag(X1[:-1], X0, Z1, Z0)


# --------------------------------------------------------------------------- properties
def _instance(seed, K, J):
    inst = random_instance(seed, K, J)
    assume(inst is not None)
    return inst


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 4), extra=st.integers(4, 8))
def test_property_weights_do_not_depend_on_predictor_order_or_units(seed, K, extra):
    X1, X0, Z1, Z0 = _instance(seed, K, K + extra)
    _, w = exact.solve_diag(X1, X0, Z1, Z0)
    perm = np.random.default_rng(seed).permutation(K)
    scale = np.exp(np.random.default_rng(seed + 1).normal(size=K))
    _, w2 = exact.solve_diag(scale * X1[perm], scale[:, None] * X0[perm], Z1, Z0)
    assert np.abs(w - w2).max() <= 1e-6
    T = np.random.default_rng(seed + 2).normal(size=(K, K)) + 3 * np.eye(K)
    _, wp = exact.solve_pd(X1, X0, Z1, Z0)
    _, wp2 = exact.solve_pd(T @ X1, T @ X0, Z1, Z0)        # the PD class sees only the affine geometry
    assert np.abs(wp - wp2).max() <= 1e-6


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 4), extra=st.integers(4, 8))
def test_property_no_positive_diagonal_metric_beats_the_optimum(seed, K, extra):
    """The diagonal optimum is a minimum over every positive diagonal V; PD only enlarges the set."""
    X1, X0, Z1, Z0 = _instance(seed, K, K + extra)
    loss, _ = exact.solve_diag(X1, X0, Z1, Z0)
    loss_pd, _ = exact.solve_pd(X1, X0, Z1, Z0)
    assert loss_pd <= loss * (1 + 1e-9) + 1e-12
    rng = np.random.default_rng(seed + 3)
    for _ in range(5):
        s = np.sqrt(np.exp(2.0 * rng.normal(size=K)))
        b = simplex_qp(s[:, None] * X0, s * X1)              # the inner projection under Diag(s^2)
        assert loss <= float(np.sum((Z1 - Z0 @ b) ** 2)) * (1 + 1e-9) + 1e-9


# --------------------------------------------------------------------------- live
@pytest.mark.skipif(os.environ.get("MLSYNTH_AS_EXACT_LIVE") != "1",
                    reason="set MLSYNTH_AS_EXACT_LIVE=1 to fetch and run the authors' code")
def test_live_against_authors_code():
    from . import reference
    ref = reference.load(reference.fetch())
    vfs = ref["visible_facet_solver"]
    for seed in range(50_000, 50_012):
        inst = random_instance(seed, 5, 20)
        if inst is None:
            continue
        X1, X0, Z1, Z0 = inst
        lr, wr, _ = vfs.solve_exterior_hullfree(X0, X1, Z1, Z0)
        lo, wo = exact.solve_diag(X1, X0, Z1, Z0)
        assert abs(lo - lr) <= 1e-9 * max(1.0, lr) and np.abs(wo - wr).max() <= 1e-7
