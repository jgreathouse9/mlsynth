"""MAREX's objective, scored by an oracle that shares no code with the library.

The exact-versus-MIQP tests compare two backends, and that comparison is only
as independent as the two paths are. Since #756 the MIQP reports the design
its selection defines by refitting the weights with the exact backend's own
``_penalties`` and ``side_fit`` (``optimization._refit_by_simplex_fits``). So
a defect in how a penalty vector is built moves both backends together, and the
differential agrees with itself: with the unit penalty's ``xi`` term dropped,
both reported treated ``u1`` at 1.967236407 where the design is ``u1, u3`` at
1.975428419 (#783).

The oracle here rebuilds every penalty from the definitions -- the squared
distance of a market to its cluster's mean, and for the unit penalty each
unit's best simplex fit on every other unit in the panel (Abadie and Zhao,
eq. 10) -- fits each candidate's two sides with Clarabel, and enumerates the
treated sets. It imports nothing from :mod:`mlsynth.utils.marex_helpers`
except the private ``_penalties`` it checks directly.

Levels: unit (the penalty vectors), differential (the design and objective),
edge, property.
"""
from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.tests.test_marex_exact_backend import _fit, _panel


# ------------------------------------------------------------------ oracle
def _predictors(df, T0):
    """Each unit's outcomes over the fit window, one row per unit, in panel order."""
    order = list(dict.fromkeys(df["unit"]))
    wide = df.pivot(index="unit", columns="time", values="y").loc[order]
    return order, wide.to_numpy(float)[:, :T0]


def _simplex_fit(D, y, lin=None, weights=False):
    """``min ||y - D w||^2 + lin'w`` over the simplex, by Clarabel; with
    ``weights``, the minimum and its weights."""
    w = cp.Variable(D.shape[1])
    obj = cp.sum_squares(y - D @ w)
    if lin is not None:
        obj = obj + lin @ w
    prob = cp.Problem(cp.Minimize(obj), [w >= 0, cp.sum(w) == 1])
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12,
               tol_feas=1e-12)
    if weights:
        return float(prob.value), np.asarray(w.value, dtype=float)
    return float(prob.value)


def _reproducibility(X):
    """Each unit's best simplex fit on every other unit of the panel (eq. 10)."""
    N = X.shape[0]
    return np.array([
        _simplex_fit(X[[i for i in range(N) if i != j]].T, X[j])
        for j in range(N)])


def _oracle_penalties(X, members, design, p):
    """The penalty vectors over one cluster's members, from the definitions."""
    A = X[members].mean(axis=0)
    d1 = ((X[members] - A) ** 2).sum(axis=1)
    if design == "penalized":
        return (p.get("lambda1", 0.0) * d1, p.get("lambda2", 0.0) * d1)
    if design == "unit_penalized":
        r = _reproducibility(X)[members]
        return (p.get("xi", 0.0) * r + p.get("lambda1_unit", 0.0) * d1,
                np.zeros(len(members)))
    return np.zeros(len(members)), np.zeros(len(members))


#: A treated market carries weight above this, or the set is not a design of
#: its size (Abadie and Zhao bound the nonzero treated weights; jgreathouse9/mlsynth#950).
CARRIED = 1e-8


def _oracle(X, groups, m, design, p):
    """The minimum of MAREX's objective by enumeration, and its treated sets.

    A set counts only when its own treated fit weights every market in it. A
    cluster with no such set makes the program empty: ``(inf, None)``.
    """
    total, treated = 0.0, []
    for members in groups:
        members = list(members)
        a, b = _oracle_penalties(X, members, design, p)
        A = X[members].mean(axis=0)
        best = (np.inf, None)
        for S in itertools.combinations(range(len(members)), m):
            C = [i for i in range(len(members)) if i not in S]
            g, w = _simplex_fit(X[[members[i] for i in S]].T, A, a[list(S)],
                                weights=True)
            if np.any(w <= CARRIED):
                continue
            f = g + _simplex_fit(X[[members[i] for i in C]].T, A, b[C])
            if f < best[0]:
                best = (f, S)
        if best[1] is None:
            return np.inf, None
        total += best[0]
        treated.append([members[i] for i in best[1]])
    return total, treated


DESIGNS = [
    ("standard", {}),
    ("penalized", {"lambda1": 0.05, "lambda2": 0.02}),
    ("penalized", {"lambda1": 0.3, "lambda2": 0.0}),
    ("penalized", {"lambda1": 0.0, "lambda2": 0.3}),
    ("unit_penalized", {"xi": 0.5, "lambda1_unit": 0.05}),
    ("unit_penalized", {"xi": 2.0, "lambda1_unit": 0.0}),
]


# -------------------------------------------------- unit: the penalty vectors
@pytest.mark.parametrize("design, p", DESIGNS[1:])
def test_the_penalty_vectors_match_their_definitions(design, p):
    """Rung 2: the shared helper against the definitions, market by market."""
    from mlsynth.utils.marex_helpers.exact import _penalties
    df, T0 = _panel(seed=1)
    _, X = _predictors(df, T0)
    members = np.arange(X.shape[0])
    D1 = ((X - X.mean(axis=0)) ** 2).sum(axis=1)[:, None]
    a, b = _penalties(design, 0, members, X, D1, p.get("lambda1", 0.0),
                      p.get("lambda2", 0.0), p.get("xi", 0.0),
                      p.get("lambda1_unit", 0.0))
    ra, rb = _oracle_penalties(X, list(members), design, p)
    np.testing.assert_allclose(np.zeros_like(ra) if a is None else a, ra,
                               rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.zeros_like(rb) if b is None else b, rb,
                               rtol=1e-6, atol=1e-9)


def test_each_cluster_is_priced_by_its_own_mean():
    """With two clusters the distances are to each cluster's own mean."""
    from mlsynth.utils.marex_helpers.exact import _penalties
    df, T0 = _panel(J=12, seed=3, region_sizes=(6, 6))
    _, X = _predictors(df, T0)
    groups = [np.arange(6), np.arange(6, 12)]
    means = [X[g].mean(axis=0) for g in groups]
    D1 = np.column_stack([((X - mu) ** 2).sum(axis=1) for mu in means])
    for k, g in enumerate(groups):
        a, b = _penalties("penalized", k, g, X, D1, 0.3, 0.1, 0.0, 0.0)
        ra, rb = _oracle_penalties(X, list(g), "penalized",
                                   {"lambda1": 0.3, "lambda2": 0.1})
        np.testing.assert_allclose(a, ra, rtol=1e-9)
        np.testing.assert_allclose(b, rb, rtol=1e-9)


def _at_oracle(want, fit, rel):
    """The fit's objective is the oracle's, or both find the program empty."""
    if not np.isfinite(want):
        with pytest.raises(MlsynthEstimationError, match="weight"):
            fit()
        return
    assert fit().globres.objective == pytest.approx(want, rel=rel)


# ------------------------------------------- differential: design and objective
@pytest.mark.parametrize("design, p", DESIGNS)
def test_the_exact_design_is_the_oracle_optimum(design, p):
    """The exact backend's objective is the enumerated minimum of the stated one."""
    df, T0 = _panel(seed=1)
    labels, X = _predictors(df, T0)
    want, sets = _oracle(X, [range(X.shape[0])], 3, design, p)
    _at_oracle(want, lambda: _fit(df, T0, "exact", m_eq=3, design=design, **p), 1e-6)


@pytest.mark.parametrize("design, p", DESIGNS[1:])
def test_the_mip_design_is_the_oracle_optimum(design, p):
    """The MIQP's reported objective too: the comparison #783 lost."""
    df, T0 = _panel(seed=1)
    _, X = _predictors(df, T0)
    want, _ = _oracle(X, [range(X.shape[0])], 3, design, p)
    _at_oracle(want, lambda: _fit(df, T0, "mip", m_eq=3, design=design, **p), 1e-5)


def test_two_clusters_are_each_at_their_oracle_optimum():
    df, T0 = _panel(J=12, seed=3, region_sizes=(6, 6))
    _, X = _predictors(df, T0)
    p = {"lambda1": 0.3, "lambda2": 0.1}
    want, _ = _oracle(X, [range(6), range(6, 12)], 2, "penalized", p)
    _at_oracle(want, lambda: _fit(df, T0, "exact", m_eq=2, cluster="region", design="penalized", **p), 1e-6)


# ----------------------------------------------------------- edge cases
def test_a_single_control_market_is_at_the_oracle_optimum():
    df, T0 = _panel(J=6, seed=11)
    _, X = _predictors(df, T0)
    p = {"xi": 1.0, "lambda1_unit": 0.1}
    want, _ = _oracle(X, [range(6)], 5, "unit_penalized", p)
    _at_oracle(want, lambda: _fit(df, T0, "exact", m_eq=5, design="unit_penalized", **p), 1e-6)


# ------------------------------------------------------------ property
@given(seed=st.integers(0, 10_000), J=st.integers(5, 7), m=st.integers(1, 3),
       design=st.sampled_from(["penalized", "unit_penalized"]),
       scale=st.sampled_from([0.01, 0.1, 1.0]))
@settings(max_examples=12, deadline=None)
def test_the_exact_objective_is_the_oracle_minimum_over_the_domain(
        seed, J, m, design, scale):
    """Rung 3: over random panels and penalties, against the independent oracle."""
    df, T0 = _panel(J=J, seed=seed, n_good=2)
    _, X = _predictors(df, T0)
    rng = np.random.default_rng(seed)
    p = ({"lambda1": scale * rng.random(), "lambda2": scale * rng.random()}
         if design == "penalized" else
         {"xi": scale * 10 * rng.random(), "lambda1_unit": scale * rng.random()})
    want, _ = _oracle(X, [range(J)], m, design, p)
    _at_oracle(want, lambda: _fit(df, T0, "exact", m_eq=m, design=design, **p), 1e-6)
