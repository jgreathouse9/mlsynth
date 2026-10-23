"""The interior case (paper, Section 5), checked against brute force and the authors' code.

When X1 lies inside the donor hull every positive definite V leaves the same exact
matches, and the vanishing penalty selects the cell of K + 1 donors that is Delaunay
under V. A cell A is attainable for diagonal V when some d >= 0 has rho_j'd > 0 for
every donor j outside A, rho_j = diag(X_j X_j' - sum_i beta_ij X_i X_i'), and the
answer is the attainable cell of least validation loss (Proposition 5). The oracle here
checks every set of K + 1 donors: the barycentric weights by ``np.linalg.solve``, the
rows rho_j from the paper's formula, and attainability by the paper's program
theta*_+(A) with HiGHS, all on predictors standardised as ``synthx.estimate`` does.
"""
from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.optimize import linprog

from . import interior
from .capture import random_instance, random_interior_instance

HERE = pathlib.Path(__file__).resolve().parent
BUNDLE = json.loads((HERE / "results" / "reference_interior.json").read_text())["cases"]
CASES = {c["name"]: c for c in BUNDLE}
THETA_TOL = 1e-9                     # the authors' threshold on theta*_+, on standardised predictors


def _inputs(case):
    return tuple(np.array(case[k], float) for k in ("X1", "X0", "Z1", "Z0"))


def _design(seed, K, J):
    while (inst := random_interior_instance(seed, K, J)) is None:
        seed += 1
    return inst


def _standardise(X1, X0):
    sd = np.column_stack([X1, X0]).std(axis=1, ddof=1)
    return X1 / sd, X0 / sd[:, None]


def _rho(A, X0):
    """The paper's rows rho_j = diag(Q_j^A) for the donors outside A, from its formula."""
    A = list(A)
    M = np.vstack([X0[:, A], np.ones(len(A))])
    rows = []
    for j in range(X0.shape[1]):
        if j in A:
            continue
        beta = np.linalg.solve(M, np.append(X0[:, j], 1.0))
        Q = np.outer(X0[:, j], X0[:, j]) - sum(b * np.outer(X0[:, i], X0[:, i]) for b, i in zip(beta, A))
        rows.append(np.diag(Q))
    return np.array(rows).reshape(-1, X0.shape[0])


def _theta_plus(A, X0):
    """theta*_+(A): max theta subject to rho_j'd >= theta, d >= 0, 1'd = 1, by HiGHS."""
    R = _rho(A, X0)
    K = X0.shape[0]
    if not len(R):
        return np.inf
    res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.c_[-R, np.ones(len(R))], b_ub=np.zeros(len(R)),
                  A_eq=np.r_[np.ones(K), 0.0][None, :], b_eq=[1.0],
                  bounds=[(0, None)] * K + [(None, None)], method="highs")
    return -res.fun


def _admissible(X1, X0):
    """Every set of K + 1 donors whose simplex holds X1 strictly, with X1's barycentric weights."""
    K, J = X0.shape
    out = {}
    for A in itertools.combinations(range(J), K + 1):
        M = np.vstack([X0[:, A], np.ones(K + 1)])
        try:
            w = np.linalg.solve(M, np.append(X1, 1.0))
        except np.linalg.LinAlgError:          # pragma: no cover - general position holds on these designs
            continue
        if w.min() > 1e-9:
            out[A] = w
    return out


def _oracle(X1, X0, Z1, Z0):
    """The attainable cells and the best of them: (cells, loss, w)."""
    S1, S0 = _standardise(X1, X0)
    cells = {A: w for A, w in _admissible(S1, S0).items() if _theta_plus(A, S0) > THETA_TOL}
    best = min(cells, key=lambda A: float(np.sum((Z1 - Z0[:, list(A)] @ cells[A]) ** 2)))
    w = np.zeros(X0.shape[1])
    w[list(best)] = cells[best]
    return set(cells), float(np.sum((Z1 - Z0 @ w) ** 2)), w


# --------------------------------------------------------------------------- smoke
def test_an_interior_design_gets_an_exact_match():
    X1, X0, Z1, Z0 = _design(4000, 3, 12)
    loss, w = interior.solve_interior_diag(X1, X0, Z1, Z0)
    assert np.isfinite(loss) and w.min() >= 0.0 and abs(w.sum() - 1.0) <= 1e-12
    assert np.allclose(X0 @ w, X1, atol=1e-10) and np.count_nonzero(w) == 4
    assert np.isclose(loss, np.sum((Z1 - Z0 @ w) ** 2), rtol=1e-12)


# --------------------------------------------------------------------------- unit: one cell
def test_a_cell_s_weights_and_rows_follow_the_paper():
    X1, X0, _, _ = _design(4000, 3, 12)
    S1, S0 = _standardise(X1, X0)
    A, w = next(iter(_admissible(S1, S0).items()))
    cell = interior.Cell(A, *interior._standardised(X1, X0))
    assert np.allclose(cell.w, w, atol=1e-12)
    assert cell.out == [j for j in range(X0.shape[1]) if j not in A]
    _, C0 = interior._standardised(X1, X0)
    assert np.allclose(cell.R, _rho(A, C0), atol=1e-10)


def test_centring_leaves_the_rows_unchanged():
    """Q_j^A does not depend on the origin, since X_j = sum_i beta_ij X_i and sum_i beta_ij = 1."""
    X1, X0, _, _ = _design(4000, 3, 12)
    S1, S0 = _standardise(X1, X0)
    A = next(iter(_admissible(S1, S0)))
    c = np.array([3.0, -2.0, 7.5])
    assert np.allclose(_rho(A, S0), _rho(A, S0 + c[:, None]), atol=1e-8)


def test_the_attainability_verdict_agrees_with_the_paper_s_program_on_every_cell():
    X1, X0, _, _ = _design(4010, 3, 12)
    C1, C0 = interior._standardised(X1, X0)
    S1, S0 = _standardise(X1, X0)
    stats = {}
    for A in _admissible(S1, S0):
        cell = interior.Cell(A, C1, C0)
        ok, d = interior.diag_attainable(cell.R, stats)
        assert ok == (_theta_plus(A, S0) > THETA_TOL)
        if ok:                                   # d is a diagonal metric that selects the cell
            assert d.min() >= 0.0 and abs(d.sum() - 1.0) <= 1e-12 and (cell.R @ d).min() > 0.0
    assert stats["cell_tests"] > 0


def test_the_neighbours_are_the_cells_that_share_k_donors():
    X1, X0, _, _ = _design(4010, 3, 12)
    C1, C0 = interior._standardised(X1, X0)
    cells = _admissible(*_standardise(X1, X0))
    for A in list(cells)[:20]:
        expected = {B for B in cells if len(set(A) & set(B)) == len(A) - 1}
        assert set(interior.Cell(A, C1, C0).neighbours()) == expected


@pytest.mark.parametrize("seed,K,J", [(4000, 2, 8), (4020, 3, 12), (4040, 4, 14)])
def test_the_walk_reaches_every_attainable_cell(seed, K, J):
    """The attainable cells are connected by flips, so a walk from the seed finds all of them."""
    X1, X0, Z1, Z0 = _design(seed, K, J)
    cells, loss, w = _oracle(X1, X0, Z1, Z0)
    stats = {}
    got_loss, got_w = interior.solve_interior_diag(X1, X0, Z1, Z0, stats=stats)
    assert set(interior.walk_diag(*interior._standardised(X1, X0), {})) == cells
    assert stats["attainable_cells"] == len(cells)
    assert np.isclose(got_loss, loss, rtol=1e-10) and np.allclose(got_w, w, atol=1e-12)


@pytest.mark.parametrize("row,attainable", [([1.05e-9, 0.2e-9], True), ([0.9e-9, 0.5e-9], False)])
def test_a_slack_near_the_threshold_is_decided_by_the_authors_program(row, attainable):
    """Near the threshold the metric that selects a cell has 1'd near 1 / theta*_+, about 1e9 here, a
    point too large for the least-distance problem to verify (its residual identity falls below
    rounding), so HiGHS solves theta*_+ and its value decides on either side of 1e-9. One row [a, b]
    has theta*_+ = max(a, b)."""
    stats = {}
    ok, d = interior.diag_attainable(np.array([row]), stats)
    assert ok is attainable and stats == {"cell_tests": 1, "cell_lp_checks": 1}
    if ok:
        assert d.min() >= 0.0 and abs(d.sum() - 1.0) <= 1e-12 and float(np.array(row) @ d) > interior.THETA_TOL


def test_a_point_whose_slack_misses_the_threshold_does_not_count(monkeypatch):
    """A verified point meets R d >= 1 to a tolerance relative to |R| |d|, which is loose for a large d;
    the threshold reads its normalised slack min(R d) / 1'd, and below it HiGHS decides. Here
    theta*_+ = 0, reached at d = (1/2, 1/2), and the verdict's point has slack 0."""
    R = np.array([[1.0, -1.0], [-1.0, 1.0]])
    monkeypatch.setattr(interior.lsq, "ldp_verdict", lambda G, h: (True, np.array([1.0, 1.0])))
    stats = {}
    assert interior.diag_attainable(R, stats) == (False, None)
    assert stats == {"cell_tests": 1, "cell_lp_checks": 1}


def test_an_ambiguous_verdict_goes_to_highs(monkeypatch):
    X1, X0, Z1, Z0 = _design(4020, 3, 12)
    _, loss, w = _oracle(X1, X0, Z1, Z0)
    monkeypatch.setattr(interior.lsq, "ldp_verdict", lambda G, h: (None, None))
    stats = {}
    got_loss, got_w = interior.solve_interior_diag(X1, X0, Z1, Z0, stats=stats)
    assert np.isclose(got_loss, loss, rtol=1e-10) and np.allclose(got_w, w, atol=1e-12)
    assert stats["cell_lp_checks"] == stats["cell_tests"] > 1


# --------------------------------------------------------------------------- unit: the authors' answers
@pytest.mark.parametrize("name", sorted(CASES))
def test_diagonal_interior_matches_authors_code(name):
    case = CASES[name]
    ref = case["diag"]
    assert "error" not in ref and ref["regime"] == "interior" and not ref["baseline_fallback"]
    loss, w = interior.solve_interior_diag(*_inputs(case))
    assert abs(loss - ref["loss"]) <= 1e-9 * max(1.0, abs(ref["loss"]))
    assert np.abs(w - np.array(ref["w"])).max() <= 1e-8
    assert sorted(np.flatnonzero(w).tolist()) == ref["cell"]


# --------------------------------------------------------------------------- edge
def test_a_pool_of_k_plus_one_donors_is_its_own_cell():
    """With J = K + 1 there is one cell and no outside donor, so nothing can rule it out."""
    X0 = np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    X1 = np.array([1.0, 1.0])
    Z0 = np.array([[1.0, 2.0, 3.0]])
    loss, w = interior.solve_interior_diag(X1, X0, np.array([2.0]), Z0)
    assert np.allclose(w, [0.5, 0.25, 0.25]) and np.isclose(loss, (2.0 - 1.75) ** 2)


def test_one_predictor_brackets_the_treated_unit():
    """K = 1: a cell is a pair of donors on either side of X1."""
    X1, X0, Z1, Z0 = np.array([0.25]), np.array([[0.0, 1.0, 2.0, -1.0]]), np.array([0.5]), \
        np.array([[0.0, 2.0, 5.0, 1.0]])
    _, loss, w = _oracle(X1, X0, Z1, Z0)
    got_loss, got_w = interior.solve_interior_diag(X1, X0, Z1, Z0)
    assert np.isclose(got_loss, loss) and np.allclose(got_w, w)


# --------------------------------------------------------------------------- property
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 4), extra=st.integers(1, 7))
def test_property_the_walk_finds_the_best_attainable_cell(seed, K, extra):
    X1, X0, Z1, Z0 = _design(seed, K, K + 1 + extra)
    _, loss, w = _oracle(X1, X0, Z1, Z0)
    got_loss, got_w = interior.solve_interior_diag(X1, X0, Z1, Z0)
    assert np.isclose(got_loss, loss, rtol=1e-10, atol=1e-12)
    assert np.allclose(got_w, w, atol=1e-10) and np.allclose(X0 @ got_w, X1, atol=1e-9)


# --------------------------------------------------------------------------- failure
def test_an_exterior_treated_unit_is_refused():
    X1, X0, Z1, Z0 = random_instance(1000, 3, 10)
    with pytest.raises(ValueError, match="inside the donor hull"):
        interior.solve_interior_diag(X1, X0, Z1, Z0)


def test_a_treated_unit_on_a_face_of_the_hull_is_refused():
    """X1 on an edge of the hull is matched by two donors, fewer than K + 1 (Assumption 2(ii))."""
    X0 = np.array([[0.0, 4.0, 0.0, 1.0], [0.0, 0.0, 4.0, 1.0]])
    with pytest.raises(ValueError, match="inside the donor hull"):
        interior.solve_interior_diag(np.array([2.0, 0.0]), X0, np.zeros(1), np.zeros((1, 4)))


def test_affinely_dependent_donors_are_refused():
    X0 = np.array([[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 3.0]])            # donors 0, 1, 2 on a line
    with pytest.raises(ValueError, match="affinely dependent"):
        interior.Cell((0, 1, 2), np.array([0.5, 1.0]), X0)


def test_a_seed_that_never_reaches_k_plus_one_donors_is_refused(monkeypatch):
    X1, X0, _, _ = _design(4020, 3, 12)
    C1, C0 = interior._standardised(X1, X0)

    class Vertex:                                  # every attempt returns a two-donor vertex
        status, x = 0, np.r_[0.5, 0.5, np.zeros(10)]
    monkeypatch.setattr(interior, "linprog", lambda *a, **k: Vertex)
    with pytest.raises(ValueError, match="Assumption 2"):
        interior.seed_cell(C1, C0)


def test_a_seed_that_is_not_attainable_is_reported(monkeypatch):
    X1, X0, Z1, Z0 = _design(4020, 3, 12)
    monkeypatch.setattr(interior, "diag_attainable", lambda R, stats: (False, None))
    with pytest.raises(ValueError, match="seed cell is not attainable"):
        interior.solve_interior_diag(X1, X0, Z1, Z0)


def test_mismatched_or_non_finite_inputs_are_refused():
    X1, X0, Z1, Z0 = _design(4020, 3, 12)
    with pytest.raises(ValueError, match="expected X1"):
        interior.solve_interior_diag(X1[:2], X0, Z1, Z0)
    with pytest.raises(ValueError, match="finite"):
        interior.solve_interior_diag(X1, X0, Z1 * np.nan, Z0)


def test_a_walk_past_its_cell_limit_raises():
    X1, X0, Z1, Z0 = _design(4020, 3, 12)
    with pytest.raises(RuntimeError, match="cell limit"):
        interior.solve_interior_diag(X1, X0, Z1, Z0, max_cells=1)
