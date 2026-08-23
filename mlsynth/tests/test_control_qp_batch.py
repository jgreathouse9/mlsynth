"""The control QP, solved for a whole shortlist at once.

LEXSCM solves the same donor program once per shortlisted design. Each solve
was a fresh cvxpy problem -- graph construction, canonicalisation, an OSQP
setup -- and on a 211-market panel with twenty designs that was 39% of the fit's
wall clock at 0.217s a call, almost none of it arithmetic.

The program is Abadie and L'Hour's penalised synthetic control,

    min_v  ||A v - y||^2 + lam * sum_j v_j ||A[:, j] - y||^2,   v in simplex

and it reduces to the homogeneous form the batched Wolfe active set already
solves. Writing ``Q = A'A``, ``b = A'y``, ``c_j = ||A[:,j] - y||^2`` and
``d = lam c - 2b``, the objective is ``v'Qv + d'v + y'y``; and since ``1'v = 1``
on the simplex, ``d'v = v'(d 1')v``, so with

    S = Q + (d 1' + 1 d') / 2

the whole thing is ``v'Sv`` plus a constant. That is one Gram per design, and a
stack of them is one batched active set: iterations set by the hardest design in
the shortlist, not by their number.

What the tests check, and why in this order:

* the reduction is exact -- asserted as a KKT certificate on the *original*
  penalised objective, computed from ``A``, ``y`` and ``v`` alone, so it holds
  whatever produced ``v`` and does not stand one solver in for another;
* the batch agrees with the single solve it replaces, at the objective value
  (the meaningful quantity when the minimiser is a face);
* exclusions are exact zeros, since a treated unit weighted 1e-7 into its own
  control is a contamination the downstream assertion is there to catch;
* ragged shortlists -- spillover neighbourhoods differ in size, so donor pools
  do too -- still batch, by grouping on pool size;
* the degenerate pools: one donor, no donors, ``lam = 0``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.fast_scm_control_helpers import (
    solve_control_qp,
    solve_control_qps,
)


# --------------------------------------------------------------- fixtures


def _panel(T=30, J=24, seed=0, factor=0.9):
    """A panel shaped like a geo panel: one dominant factor, near-collinear."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal(T))
    load = rng.uniform(0.6, 1.6, J)
    X = factor * np.outer(f, load) + (1.0 - factor) * rng.standard_normal((T, J))
    X = X - X.mean(1, keepdims=True)
    sd = np.std(X, 1, keepdims=True)
    return X / np.where(sd < 1e-8, 1.0, sd)


def _objective(A, y, v, lam):
    """The penalised objective, evaluated on the donor submatrix."""
    c = ((A - y[:, None]) ** 2).sum(0)
    return float(((A @ v - y) ** 2).sum() + lam * (c @ v))


def _kkt_residual(A, y, v, lam):
    """Distance from the KKT conditions of the penalised program.

    With gradient ``g = 2 A'(A v - y) + lam c`` the simplex conditions are
    ``g_j >= mu`` everywhere and ``g_j == mu`` wherever ``v_j > 0``, for the
    multiplier ``mu = g'v``. Computed from the data and the point, so it
    certifies optimality without reference to a solver.
    """
    c = ((A - y[:, None]) ** 2).sum(0)
    g = 2.0 * (A.T @ (A @ v - y)) + lam * c
    mu = float(g @ v)
    scale = max(abs(mu), float(np.abs(g).max()), 1e-12)
    support = v > 1e-9
    slack = float(np.max(np.maximum(mu - g, 0.0))) / scale
    tight = (float(np.max(np.abs(g[support] - mu))) / scale
             if support.any() else 0.0)
    return max(slack, tight)


def _designs(X_E, n, m, seed=1):
    """``n`` treated tuples of size ``m`` with their synthetic-treated series."""
    rng = np.random.default_rng(seed)
    J = X_E.shape[1]
    out = []
    for _ in range(n):
        S = np.sort(rng.choice(J, size=m, replace=False))
        w = rng.dirichlet(np.ones(m))
        out.append((S, X_E[:, S] @ w))
    return out


# ------------------------------------------------- the answers are optima


@pytest.mark.parametrize("lam", [0.0, 0.1, 2.0])
def test_batched_weights_certify_against_the_penalised_program(lam):
    """Feasible, and satisfying the KKT conditions of the original objective."""
    X_E = _panel()
    designs = _designs(X_E, 8, 4)
    V = solve_control_qps(X_E, [y for _, y in designs],
                          [S for S, _ in designs], lambda_penalty=lam)

    for (S, y), v in zip(designs, V):
        assert v is not None
        assert v.min() >= -1e-12
        assert v.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(v[S] == 0.0)                     # exactly, not to 1e-8
        ctrl = np.setdiff1d(np.arange(X_E.shape[1]), S)
        assert _kkt_residual(X_E[:, ctrl], y, v[ctrl], lam) < 1e-6


def test_the_batch_matches_the_single_solve_it_replaces():
    """Same program, so the same objective value -- to well inside cvxpy's own
    tolerance. The weights themselves are compared loosely: with more donors
    than estimation periods the minimiser can be a face, and two correct solvers
    land in different places on it."""
    X_E = _panel()
    designs = _designs(X_E, 6, 3, seed=4)
    lam = 0.1
    V = solve_control_qps(X_E, [y for _, y in designs],
                          [S for S, _ in designs], lambda_penalty=lam)

    for (S, y), v in zip(designs, V):
        ref = solve_control_qp(X_E, y, list(S), lambda_penalty=lam)
        ctrl = np.setdiff1d(np.arange(X_E.shape[1]), S)
        A = X_E[:, ctrl]
        assert _objective(A, y, v[ctrl], lam) <= _objective(A, y, ref[ctrl], lam) + 1e-7


# ------------------------------------------------------- exclusions hold


def test_spillover_neighbours_get_exact_zeros():
    """The exclusion restriction is enforced by dropping columns, so the
    excluded weights are zero by construction and not to a solver tolerance."""
    X_E = _panel()
    designs = _designs(X_E, 4, 3, seed=7)
    spills = [[10, 11], [12], [], [13, 14, 15]]
    V = solve_control_qps(X_E, [y for _, y in designs],
                          [S for S, _ in designs], lambda_penalty=0.1,
                          exclude_idx_list=spills)
    for (S, _), sp, v in zip(designs, spills, V):
        dropped = np.concatenate([S, np.asarray(sp, dtype=int)]).astype(int)
        assert np.all(v[np.unique(dropped)] == 0.0)
        assert v.sum() == pytest.approx(1.0, abs=1e-9)


def test_pools_of_different_widths_still_batch():
    """Spillover neighbourhoods differ in size, so the donor pools do too. The
    grouping must not silently drop, reorder or mismatch a design: each returned
    vector has to belong to the design at its own position."""
    X_E = _panel()
    designs = _designs(X_E, 5, 3, seed=9)
    spills = [[], [10], [10, 11], [], [12, 13, 14]]
    V = solve_control_qps(X_E, [y for _, y in designs],
                          [S for S, _ in designs], lambda_penalty=0.1,
                          exclude_idx_list=spills)
    assert len(V) == len(designs)
    for (S, y), sp, v in zip(designs, spills, V):
        ctrl = np.setdiff1d(np.arange(X_E.shape[1]),
                            np.concatenate([S, np.asarray(sp, dtype=int)]))
        assert _kkt_residual(X_E[:, ctrl], y, v[ctrl], 0.1) < 1e-6


# ------------------------------------------------------------ degenerate


def test_a_single_surviving_donor_takes_all_the_weight():
    X_E = _panel(J=6)
    S = np.array([0, 1])
    y = X_E[:, S] @ np.array([0.5, 0.5])
    V = solve_control_qps(X_E, [y], [S], lambda_penalty=0.1,
                          exclude_idx_list=[[2, 3, 4]])
    assert V[0][5] == pytest.approx(1.0, abs=1e-12)
    assert V[0].sum() == pytest.approx(1.0, abs=1e-12)


def test_an_emptied_donor_pool_reports_none_in_place():
    """A design whose exclusions leave no donors is skipped downstream, so it
    must come back as ``None`` at its own index and not shift the others."""
    X_E = _panel(J=6)
    designs = _designs(X_E, 2, 2, seed=11)
    ok, dead = designs[0], designs[1]
    V = solve_control_qps(X_E, [ok[1], dead[1]], [ok[0], dead[0]],
                          lambda_penalty=0.1,
                          exclude_idx_list=[[], list(range(6))])
    assert V[1] is None
    assert V[0] is not None and V[0].sum() == pytest.approx(1.0, abs=1e-9)


def test_an_empty_shortlist_returns_an_empty_list():
    X_E = _panel(J=6)
    assert solve_control_qps(X_E, [], [], lambda_penalty=0.1) == []


# ------------------------------------------------------- the work contract


def test_the_shortlist_costs_one_solve_not_one_per_design(monkeypatch):
    """The point of the reduction: designs sharing a pool width go through the
    active set together, so the solver is entered once per width and not once
    per design. Asserted on the call count, which is machine-independent."""
    from mlsynth.utils.fast_scm_helpers import fast_scm_control_helpers as h

    calls = {"n": 0}
    real = h.solve_simplex_minnorm_batch

    def spy(G, **kw):
        calls["n"] += 1
        return real(G, **kw)

    monkeypatch.setattr(h, "solve_simplex_minnorm_batch", spy)
    X_E = _panel(J=30)
    designs = _designs(X_E, 20, 4, seed=13)
    solve_control_qps(X_E, [y for _, y in designs], [S for S, _ in designs],
                      lambda_penalty=0.1)
    assert calls["n"] == 1, (
        f"{calls['n']} solver entries for 20 designs of equal pool width -- "
        "the batch is being run one design at a time"
    )


def test_cvxpy_is_not_entered_at_all(monkeypatch):
    """cvxpy's per-problem overhead is the cost being removed; a fallback that
    quietly re-enters it would restore the cost while the tests still passed."""
    import cvxpy as cp
    from mlsynth.utils.fast_scm_helpers import fast_scm_control_helpers as h

    def explode(*a, **kw):
        raise AssertionError("the batched control QP entered cvxpy")

    monkeypatch.setattr(cp.Problem, "solve", explode)
    monkeypatch.setattr(h, "_solve_qp_problem", explode)
    X_E = _panel()
    designs = _designs(X_E, 5, 3, seed=15)
    V = solve_control_qps(X_E, [y for _, y in designs],
                          [S for S, _ in designs], lambda_penalty=0.1)
    assert all(v is not None for v in V)
