"""The exact backend's interior case under positive definite V, checked against brute force and the authors' code.

Cell A is attainable for some positive definite V when <Q_j^A, V> > 0 for every donor j
outside it, Q_j^A = X_j X_j' - sum_i beta_ij X_i X_i' (paper, Section 5.2), and the
answer is the attainable cell of least validation loss (Proposition 5). The oracle
checks every set of K + 1 donors and solves the paper's program theta*(A), the largest
least slack over V >= 0 with tr(V) = K (the normalisation in the authors' code), with
cvxpy, written here independently of both implementations.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from _exact_support import (INTERIOR as CASES, admissible as _admissible, inputs as _inputs,
                            interior_design as _design, random_instance, selection as _selection,
                            standardise as _standardise)
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.bilevel.exact import interior

THETA_TOL = 1e-9


def _q(A, X0):
    """The paper's Q_j^A for the donors outside A, from its formula."""
    A = list(A)
    M = np.vstack([X0[:, A], np.ones(len(A))])
    out = []
    for j in range(X0.shape[1]):
        if j in A:
            continue
        beta = np.linalg.solve(M, np.append(X0[:, j], 1.0))
        out.append(np.outer(X0[:, j], X0[:, j]) - sum(b * np.outer(X0[:, i], X0[:, i]) for b, i in zip(beta, A)))
    return np.array(out).reshape(-1, X0.shape[0], X0.shape[0])


def _theta(Q, K):
    """theta*(A) by cvxpy: max theta subject to <Q_j, V> >= theta, V >= 0, tr(V) = K."""
    if not len(Q):
        return np.inf
    V, s = cp.Variable((K, K), PSD=True), cp.Variable()
    prob = cp.Problem(cp.Maximize(s), [cp.trace(V) == K] + [cp.trace(Qj @ V) >= s for Qj in Q])
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value)


def _oracle(X1, X0, Z1, Z0):
    """The best attainable cell under positive definite V, by brute force: (loss, w)."""
    S1, S0 = _standardise(X1, X0)
    K = X0.shape[0]
    best = (np.inf, None)
    for A, w in sorted(_admissible(S1, S0).items(), key=lambda a: float(np.sum((Z1 - Z0[:, list(a[0])] @ a[1]) ** 2))):
        if _theta(_q(A, S0), K) > THETA_TOL:
            full = np.zeros(X0.shape[1])
            full[list(A)] = w
            return float(np.sum((Z1 - Z0 @ full) ** 2)), full
    return best


def _check_evidence(ok, evidence, Q, K):
    """A V that selects the cell, or a y on the simplex that proves no V does."""
    if ok:
        assert np.linalg.eigvalsh(evidence).min() > 0.0 and abs(np.trace(evidence) - K) <= 1e-9
        assert min(np.trace(Qj @ evidence) for Qj in Q) > THETA_TOL
    elif evidence is not None:
        assert evidence.min() >= 0.0 and abs(evidence.sum() - 1.0) <= 1e-9
        assert K * np.linalg.eigvalsh(np.einsum("j,jkl->kl", evidence, Q)).max() <= THETA_TOL


# --------------------------------------------------------------------------- smoke
def test_an_interior_design_gets_an_exact_match_under_pd_metrics():
    X1, X0, Z1, Z0 = _design(4000, 3, 12)
    loss, w = interior.solve_interior_pd(X1, X0, Z1, Z0)
    assert np.isfinite(loss) and w.min() >= 0.0 and abs(w.sum() - 1.0) <= 1e-12
    assert np.allclose(X0 @ w, X1, atol=1e-10) and np.count_nonzero(w) == 4


# --------------------------------------------------------------------------- unit: one cell
def test_the_cell_matrices_follow_the_paper():
    X1, X0, _, _ = _design(4000, 3, 12)
    C1, C0 = interior._standardised(X1, X0)
    for A in list(_admissible(C1, C0))[:5]:
        cell = interior.Cell(A, C1, C0)
        Q = interior.q_matrices(cell, C0)
        assert np.allclose(Q, _q(A, C0), atol=1e-10)
        assert np.allclose(np.einsum("jkk->jk", Q), cell.R, atol=1e-10)    # the diagonal rows rho_j


@pytest.mark.parametrize("seed,K,J", [(4020, 3, 12), (4040, 4, 14)])
def test_the_pd_verdict_agrees_with_the_paper_s_program_on_every_cell(seed, K, J):
    X1, X0, _, _ = _design(seed, K, J)
    C1, C0 = interior._standardised(X1, X0)
    stats, decided = {}, 0
    for A in _admissible(C1, C0):
        Q = interior.q_matrices(interior.Cell(A, C1, C0), C0)
        theta = _theta(Q, K)
        ok, evidence = interior.pd_attainable(Q, stats)
        _check_evidence(ok, evidence, Q, K)
        if abs(theta) > 1e-6:                      # cvxpy's own accuracy leaves the rest open
            assert ok == (theta > THETA_TOL)
            decided += 1
    assert decided > 10 and stats["pd_tests"] > 0 and stats.get("pd_sdp_checks", 0) == 0


@pytest.mark.parametrize("theta,verdict", [(10.0, True), (1.0 + 1e-7, None), (0.1, False)])
def test_the_barrier_stops_at_checked_evidence_or_reports_none(theta, verdict):
    """One outside donor with Q_1 = diag(c, -1) has theta*(A) = K c, all of tr V = K on the first axis.
    Clearly above or below the 1e-9 threshold a certificate checks. Just above it neither can: the
    barrier's point stays about (n + K) / t short of theta*, and the dual point y = 1 reaches theta*."""
    c = THETA_TOL * theta / 2
    Q = np.array([np.diag([c, -1.0])])
    ok, evidence = interior._pd_barrier(Q, np.eye(2), {})
    assert ok is verdict
    if verdict is not None:
        _check_evidence(ok, evidence, Q, 2)


def test_an_undecided_barrier_goes_to_the_authors_program(monkeypatch):
    X1, X0, Z1, Z0 = _design(4020, 3, 12)
    loss, w = _oracle(X1, X0, Z1, Z0)
    monkeypatch.setattr(interior, "_pd_barrier", lambda Q, V0, stats: (None, None))
    stats = {}
    got_loss, got_w = interior.solve_interior_pd(X1, X0, Z1, Z0, stats=stats)
    assert np.isclose(got_loss, loss, rtol=1e-10) and np.allclose(got_w, w, atol=1e-12)
    assert stats["pd_sdp_checks"] >= 1


def test_the_walk_reaches_every_admissible_cell():
    X1, X0, _, _ = _design(4040, 4, 14)
    C1, C0 = interior._standardised(X1, X0)
    assert set(interior.walk_admissible(C1, C0, {})) == set(_admissible(C1, C0))


@pytest.mark.parametrize("seed,K,J", [(4000, 2, 8), (4020, 3, 12), (4040, 4, 14)])
def test_the_best_attainable_cell_matches_brute_force(seed, K, J):
    X1, X0, Z1, Z0 = _design(seed, K, J)
    loss, w = _oracle(X1, X0, Z1, Z0)
    got_loss, got_w = interior.solve_interior_pd(X1, X0, Z1, Z0)
    assert np.isclose(got_loss, loss, rtol=1e-10) and np.allclose(got_w, w, atol=1e-12)


# --------------------------------------------------------------------------- unit: the authors' answers
@pytest.mark.parametrize("name", sorted(CASES))
def test_pd_interior_matches_authors_code(name):
    case = CASES[name]
    ref = case["pd"]
    assert "error" not in ref and ref["regime"] == "interior" and not ref["baseline_fallback"]
    loss, w = interior.solve_interior_pd(*_inputs(case))
    assert abs(loss - ref["loss"]) <= 1e-9 * max(1.0, abs(ref["loss"]))
    assert np.abs(w - np.array(ref["w"])).max() <= 1e-8
    assert sorted(np.flatnonzero(w).tolist()) == ref["cell"]


def test_the_pd_optimum_never_exceeds_the_diagonal_one():
    """Every diagonal metric is positive definite, so the PD class attains at least as much."""
    for case in CASES.values():
        assert case["pd"]["loss"] <= case["diag"]["loss"] * (1 + 1e-12)
        assert interior.solve_interior_pd(*_inputs(case))[0] <= interior.solve_interior_diag(*_inputs(case))[0] * (1 + 1e-12)


# --------------------------------------------------------------------------- edge
def test_a_pool_of_k_plus_one_donors_is_its_own_pd_cell():
    X0 = np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    loss, w = interior.solve_interior_pd(np.array([1.0, 1.0]), X0, np.array([2.0]), np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(w, [0.5, 0.25, 0.25]) and np.isclose(loss, (2.0 - 1.75) ** 2)


def test_one_predictor_leaves_the_pd_and_diagonal_classes_the_same():
    X1, X0, Z1, Z0 = np.array([0.25]), np.array([[0.0, 1.0, 2.0, -1.0]]), np.array([0.5]), \
        np.array([[0.0, 2.0, 5.0, 1.0]])
    pd, diag = interior.solve_interior_pd(X1, X0, Z1, Z0), interior.solve_interior_diag(X1, X0, Z1, Z0)
    assert np.isclose(pd[0], diag[0]) and np.allclose(pd[1], diag[1])


# --------------------------------------------------------------------------- property
@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 3), extra=st.integers(1, 6))
def test_property_the_search_finds_the_best_pd_attainable_cell(seed, K, extra):
    X1, X0, Z1, Z0 = _design(seed, K, K + 1 + extra)
    loss, w = _oracle(X1, X0, Z1, Z0)
    got_loss, got_w = interior.solve_interior_pd(X1, X0, Z1, Z0)
    assert np.isclose(got_loss, loss, rtol=1e-10, atol=1e-12) and np.allclose(got_w, w, atol=1e-10)


# --------------------------------------------------------------------------- failure
def test_an_exterior_treated_unit_is_refused_under_pd_metrics():
    X1, X0, Z1, Z0 = random_instance(1000, 3, 10)
    with pytest.raises(MlsynthDataError, match="inside the donor hull"):
        interior.solve_interior_pd(X1, X0, Z1, Z0)


def test_a_pd_walk_past_its_cell_limit_raises():
    X1, X0, Z1, Z0 = _design(4020, 3, 12)
    with pytest.raises(MlsynthEstimationError, match="cell limit"):
        interior.solve_interior_pd(X1, X0, Z1, Z0, max_cells=1)


def test_no_attainable_cell_is_reported(monkeypatch):
    X1, X0, Z1, Z0 = _design(4020, 3, 12)
    monkeypatch.setattr(interior, "pd_attainable", lambda Q, stats, V0=None: (False, None))
    with pytest.raises(MlsynthDataError, match="no admissible cell is attainable"):
        interior.solve_interior_pd(X1, X0, Z1, Z0)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 4), extra=st.integers(1, 12), metric=st.integers(0, 2 ** 31))
def test_property_no_positive_definite_metric_selects_a_better_match(seed, K, extra, metric):
    """Proposition 5 without enumeration, for the full class: a random positive definite V selects a
    match that fits no better than the search's optimum."""
    X1, X0, Z1, Z0 = _design(seed, K, K + 1 + extra)
    loss, _ = interior.solve_interior_pd(X1, X0, Z1, Z0)
    S1, S0 = _standardise(X1, X0)
    L = np.random.default_rng(metric).normal(size=(K, K))
    w = _selection(S1, S0, L @ L.T + 1e-3 * np.eye(K))
    assert loss <= float(np.sum((Z1 - Z0 @ w) ** 2)) * (1 + 1e-9) + 1e-12
