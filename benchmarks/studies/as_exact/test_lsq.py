"""The exact least-squares layer, lsq.py, tested on its own.

``_ldp`` decides whether G x >= h has a solution with one NNLS call (Lawson and Hanson,
Solving Least Squares Problems, ch. 23), and every face fit and piece fit of the
search goes through it. It once called an infeasible system feasible and returned a
point that broke its rows by 2.9: the sign-branching prototype asked it about the
normal-cone system in ``results/ldp_incident.json``, which HiGHS reports infeasible.
The tests follow the root-cause ladder for that incident, rung by rung.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st
from scipy.optimize import linprog

from . import lsq

INCIDENT = json.loads((pathlib.Path(__file__).resolve().parent / "results" / "ldp_incident.json").read_text())


def _incident():
    return np.array(INCIDENT["G"], float), np.array(INCIDENT["h"], float)


def _highs_feasible(G, h):
    res = linprog(np.zeros(G.shape[1]), A_ub=-G, b_ub=-h, bounds=(None, None), method="highs")
    return res.status == 0


def _violation(G, h, x):
    """Largest row violation of G x >= h, relative to the size of the row's terms."""
    v = np.maximum(h - G @ x, 0.0)
    return float((v / (np.linalg.norm(G, axis=1) * np.linalg.norm(x) + np.abs(h) + 1e-300)).max())


# --------------------------------------------------------------------------- rungs 0 and 1: the incident
def test_the_incident_system_is_refused():
    G, h = _incident()
    assert not _highs_feasible(G, h)
    assert lsq._ldp(G, h) is None


# --------------------------------------------------------------------------- rung 2: each step on its own
def test_the_decision_rule_reads_a_rounding_residual_as_no_solution():
    """The NNLS residual here is 1.1e-11 with r_{n+1} = +2.4e-12; a solution's has r_{n+1} = -||r||^2."""
    G, h = _incident()
    assert lsq._ldp_candidate(G, h) is None


def test_ldp_returns_only_points_it_has_verified(monkeypatch):
    G = np.array([[1.0, 0.0], [0.0, 1.0]])
    h = np.array([1.0, 1.0])
    monkeypatch.setattr(lsq, "_ldp_candidate", lambda G_, h_: np.array([1.0, 0.5]))
    assert lsq._ldp(G, h) is None
    monkeypatch.setattr(lsq, "_ldp_candidate", lambda G_, h_: np.array([1.0, 1.0]))
    assert np.array_equal(lsq._ldp(G, h), [1.0, 1.0])


def _piece_with_injected_reduction(monkeypatch, target):
    """simplex_ls on b_0 >= b_1 over three donors, with the reduction made to return `target`."""
    rng = np.random.default_rng(0)
    Z, z = rng.normal(size=(5, 3)), rng.normal(size=5)
    Gin, hin = np.array([[1.0, -1.0, 0.0]]), np.array([0.0])
    N = np.linalg.svd(np.ones((1, 3)))[2][1:].T          # the null-space basis simplex_ls builds
    b0 = np.full(3, 1.0 / 3.0)
    reduced = None if target is None else N.T @ (np.asarray(target) - b0)
    monkeypatch.setattr(lsq, "_lsi", lambda E, f, G, h: reduced)
    return z, Z, Gin, hin


def test_simplex_ls_returns_only_points_satisfying_its_inequalities(monkeypatch):
    """A reduced solution that keeps b on the simplex but breaks b_0 >= b_1 is an error, not an answer."""
    z, Z, Gin, hin = _piece_with_injected_reduction(monkeypatch, [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="breaks its constraints"):
        lsq.simplex_ls(z, Z, Gin=Gin, hin=hin)
    z, Z, Gin, hin = _piece_with_injected_reduction(monkeypatch, [1.0, 0.0, 0.0])
    val, b = lsq.simplex_ls(z, Z, Gin=Gin, hin=hin)
    assert np.allclose(b, [1.0, 0.0, 0.0]) and val == pytest.approx(float(np.sum((z - Z[:, 0]) ** 2)))


def test_a_failed_reduction_is_not_read_as_infeasible(monkeypatch):
    """The reduction finding nothing on a piece with points in it is an error; on an empty piece it is the answer."""
    z, Z, Gin, hin = _piece_with_injected_reduction(monkeypatch, None)
    with pytest.raises(ValueError, match="reduction failed"):
        lsq.simplex_ls(z, Z, Gin=Gin, hin=hin)                  # b = (1, 0, 0) satisfies b_0 >= b_1
    assert lsq.simplex_ls(z, Z, Gin=Gin, hin=np.array([1.5])) == (None, None)   # b_0 - b_1 <= 1 on the simplex


# --------------------------------------------------------------------------- rung 3: the invariants
def _planted_infeasible(rng, n, p, delta):
    """Unit rows with a Farkas certificate y >= 0, G'y = 0, y'h = delta * sum(y): no x has G x >= h."""
    G = rng.normal(size=(p, n))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    q = int(rng.integers(2, min(p, n + 2) + 1))
    y = np.zeros(p)
    y[:q] = rng.uniform(0.5, 1.5, size=q)
    G[q - 1] = -(y[: q - 1] @ G[: q - 1]) / y[q - 1]
    norm = np.linalg.norm(G[q - 1])
    G[q - 1] /= norm
    y[q - 1] *= norm
    h = G @ rng.normal(size=n) - np.abs(rng.normal(size=p))
    h[:q] = G[:q] @ rng.normal(size=n)
    h[:q] += (delta * y.sum() - y[:q] @ h[:q]) / y[:q].sum()
    return G, h


def _planted_feasible(rng, n, p):
    """Unit rows and a point x0 that meets a third of them with equality."""
    G = rng.normal(size=(p, n))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    x0 = rng.normal(size=n)
    s = np.abs(rng.normal(size=p))
    s[: max(1, p // 3)] = 0.0
    return G, G @ x0 - s, x0


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), n=st.integers(2, 7), extra=st.integers(2, 40),
       log_delta=st.floats(-7.0, -1.0))
def test_property_every_infeasible_system_is_refused(seed, n, extra, log_delta):
    """Margins from 1e-7 to 1e-1: below about 1e-4 the rounding residual is eps / delta."""
    G, h = _planted_infeasible(np.random.default_rng(seed), n, n + extra, 10.0 ** log_delta)
    assert lsq._ldp(G, h) is None


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), n=st.integers(2, 7), extra=st.integers(2, 40))
def test_property_a_feasible_system_gets_its_least_norm_point(seed, n, extra):
    G, h, x0 = _planted_feasible(np.random.default_rng(seed), n, n + extra)
    x = lsq._ldp(G, h)
    assert x is not None
    assert _violation(G, h, x) <= 1e-9
    assert np.linalg.norm(x) <= np.linalg.norm(x0) + 1e-9


def _piece_margin(Gin, hin):
    """Largest t with Gin b >= hin + t on the simplex (capped at 1); its sign is the piece's verdict."""
    m = Gin.shape[1]
    res = linprog(np.r_[np.zeros(m), -1.0], A_ub=np.c_[-Gin, np.ones(len(Gin))], b_ub=-hin,
                  A_eq=np.r_[np.ones(m), 0.0][None, :], b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, 1.0)], method="highs")
    return -res.fun


def _kkt_gap(z, Z, Gin, hin, b, active=1e-7):
    """How far b is from optimal: the gradient's L1 distance from the cone of the active constraints.

    The piece is convex, so b is optimal exactly when grad ||z - Z b||^2 = sum mu_j e_j +
    sum lam_i Gin_i + nu 1 with mu, lam >= 0. HiGHS finds the closest such combination.
    """
    m = len(b)
    g = 2.0 * Z.T @ (Z @ b - z)
    cols = [np.eye(m)[:, j] for j in np.flatnonzero(b <= active)]
    cols += [Gin[i] for i in np.flatnonzero(Gin @ b - hin <= active * (np.linalg.norm(Gin, axis=1) + np.abs(hin)))]
    C = np.column_stack(cols + [np.ones(m), -np.ones(m)]) if cols else np.column_stack([np.ones(m), -np.ones(m)])
    q = C.shape[1]
    res = linprog(np.r_[np.zeros(q), np.ones(2 * m)], A_eq=np.c_[C, np.eye(m), -np.eye(m)], b_eq=g,
                  bounds=[(0, None)] * (q + 2 * m), method="highs")
    return res.fun / max(1.0, float(np.abs(g).sum()))


@settings(max_examples=80, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), m=st.integers(2, 6), extra=st.integers(1, 8), k=st.integers(1, 6))
def test_property_a_piece_is_solved_when_feasible_and_refused_when_not(seed, m, extra, k):
    """More validation periods than donors, so the reduction is in its domain: the verdict and the optimum are exact."""
    rng = np.random.default_rng(seed)
    L = m + extra
    Z, z = rng.normal(size=(L, m)), rng.normal(size=L)
    Gin, hin = rng.normal(size=(k, m)), 0.3 * rng.normal(size=k)
    margin = _piece_margin(Gin, hin)
    assume(abs(margin) > 1e-6)
    sv = np.linalg.svd(Z @ np.linalg.svd(np.ones((1, m)))[2][1:].T, compute_uv=False)
    assume(sv[0] / sv[-1] <= lsq.COND_MAX)
    val, b = lsq.simplex_ls(z, Z, Gin=Gin, hin=hin)
    if margin < 0:
        assert b is None
        return
    assert b is not None
    assert b.min() >= 0.0 and abs(b.sum() - 1.0) <= 1e-12
    assert _violation(Gin, hin, b) <= 1e-7
    assert val == pytest.approx(float(np.sum((z - Z @ b) ** 2)), rel=1e-12, abs=1e-12)
    assert _kkt_gap(z, Z, Gin, hin, b) <= 1e-6


@settings(max_examples=80, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), m=st.integers(3, 6), extra=st.integers(1, 8), k=st.integers(1, 6),
       log_eta=st.floats(-12.0, 0.0))
def test_property_a_nearly_collinear_face_is_solved_or_refused_never_misread(seed, m, extra, k, log_eta):
    """Two donors with validation paths 10^log_eta apart: below the condition cap the answer is exact, above it an error."""
    rng = np.random.default_rng(seed)
    L = m + extra
    Z, z = rng.normal(size=(L, m)), rng.normal(size=L)
    Z[:, -1] = Z[:, 0] + 10.0 ** log_eta * rng.normal(size=L)
    Gin, hin = rng.normal(size=(k, m)), 0.3 * rng.normal(size=k)
    margin = _piece_margin(Gin, hin)
    assume(abs(margin) > 1e-6)
    sv = np.linalg.svd(Z @ np.linalg.svd(np.ones((1, m)))[2][1:].T, compute_uv=False)
    if sv[0] / sv[-1] > lsq.COND_MAX:
        with pytest.raises(ValueError, match="ill-conditioned"):
            lsq.simplex_ls(z, Z, Gin=Gin, hin=hin)
        return
    val, b = lsq.simplex_ls(z, Z, Gin=Gin, hin=hin)
    assert (b is None) == (margin < 0)
    if b is not None:
        assert _violation(Gin, hin, b) <= 1e-7
        assert _kkt_gap(z, Z, Gin, hin, b) <= 1e-6


# --------------------------------------------------------------------------- rung 4: degenerate inputs
@pytest.mark.parametrize("G, h, expected", [
    ([[1.0], [-1.0]], [1.0, 0.0], None),                       # x >= 1 and x <= 0
    ([[1.0], [-1.0]], [1.0, -1.0], [1.0]),                     # one feasible point
    ([[0.0, 0.0], [1.0, 0.0]], [1.0, 0.0], None),              # a zero row that asks 0 >= 1
    ([[0.0, 0.0], [1.0, 0.0]], [-1.0, 2.0], [2.0, 0.0]),       # a zero row that asks nothing
])
def test_degenerate_systems(G, h, expected):
    x = lsq._ldp(np.array(G), np.array(h))
    if expected is None:
        assert x is None
    else:
        assert x is not None and np.allclose(x, expected, atol=1e-9)


@pytest.mark.parametrize("L, eta", [(2, 1.0), (6, 1e-10)])
def test_a_face_outside_the_reduction_s_domain_is_refused(L, eta):
    """Fewer validation periods than free weights, or two donors 1e-10 apart: an error naming the condition."""
    rng = np.random.default_rng(3)
    Z, z = rng.normal(size=(L, 5)), rng.normal(size=L)
    Z[:, -1] = Z[:, 0] + eta * rng.normal(size=L)
    with pytest.raises(ValueError, match="rank-deficient or ill-conditioned"):
        lsq.simplex_ls(z, Z)


def test_a_design_with_more_predictors_than_validation_periods_is_refused():
    """West Germany's facets have more free weights than two validation periods can identify."""
    from .test_exact import CASES, _inputs
    from . import exact
    X1, X0, Z1, Z0 = _inputs(CASES["germany"])
    assert X0.shape[0] - 1 > 2
    with pytest.raises(ValueError, match="validation block"):
        exact.solve_diag(X1, X0, Z1[:2], Z0[:2])
