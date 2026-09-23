"""Generative property tests for the LEXSCM treated-design objective.

The design search ranks candidate treated tuples by

    L(S) = min_{w in simplex} w' G_SS w,     G = Xtilde' Xtilde

which, because the predictors are f-centred, is the squared distance from the
population centroid to the convex hull of the selected units. Everything the
search reports rests on that number being the minimum: `L` is a *ranking*
criterion, so an error that looks small in absolute terms reorders the
shortlist and changes which markets get treated.

Why this layer. Property tests generate inside the specification, which is
where a fault of omission lives (see the instrument contract in
``agents/agents_tests.md``). `L` is close to the Layer 1 target that contract
prefers -- a pure function of one PSD matrix -- with the caveat that a solver
runs inside it.

The properties are the geometric ones that carry the objective's meaning, and
each would be violated by a plausible defect rather than only by nonsense:

* non-negativity and the vertex bound -- a feasible point is always available
  by putting all the weight on one unit, so `L` can never exceed the smallest
  diagonal, and PSD says it cannot go below zero;
* monotonicity -- enlarging `S` enlarges the hull, so `L` can only fall. This
  is the property a search that returns a *near*-optimum silently breaks;
* set semantics -- `L` depends on which units are chosen, not the order they
  are listed in;
* scale -- scaling the panel by `c` scales `L` by `c^2`, since the objective is
  a squared distance in the same units;
* batch/single agreement -- the two solvers are the same program at different
  arities, and the search uses one to rank and the other to report;
* the KKT certificate -- computed from `Q` and `w` alone, so it holds whatever
  solver produced `w`, which is what lets it check the arithmetic without
  standing one implementation in for another.

Conditioning is generated deliberately, not left to chance. A Gram of
independent gaussians is well enough separated that even a sublinearly
convergent method solves it, so a suite built only on those passes while the
estimator is wrong in the field -- which is exactly what happened here. The
``factor_strength`` parameter spans from that easy regime to the one a geo
panel actually presents, where a single component carries most of the variance
and the columns are near-collinear.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.fast_scm_helpers.lexsearch import _afw_batched, _afw_single

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _panel(T, M, factor_strength, seed):
    """An f-centred, standardised panel and its Gram, as ``build_X_tilde`` makes it.

    ``factor_strength`` interpolates between independent columns (0.0) and a
    panel dominated by one common factor (1.0), which is the geo-panel regime.
    """
    rng = np.random.default_rng(seed)
    idio = rng.standard_normal((T, M))
    f = np.cumsum(rng.standard_normal(T))
    load = rng.uniform(0.6, 1.6, M)
    X = factor_strength * np.outer(f, load) + (1.0 - factor_strength) * idio
    X = X - X.mean(1, keepdims=True)                    # f-centred, f uniform
    sd = np.std(X, 1, keepdims=True)
    X = X / np.where(sd < 1e-8, 1.0, sd)
    return X, X.T @ X


def _loss(G, S):
    S = np.asarray(sorted(S))
    return float(_afw_batched(G[np.ix_(S, S)][None], iters=80)[0])


def _kkt_residual(Q, w):
    """Distance from the KKT conditions of ``min_w w'Qw`` on the simplex.

    With ``g = Q w`` and ``nu = w'Qw`` the conditions are ``g_j >= nu``
    everywhere, tight wherever ``w_j > 0``. Computed from ``Q`` and ``w``, so
    it certifies a point without reference to the solver that produced it.
    """
    g = Q @ w
    nu = float(w @ g)
    scale = max(abs(nu), float(np.abs(np.diag(Q)).max()), 1e-12)
    tight = w > 1e-10
    return max(float(np.max(np.maximum(nu - g, 0.0))) / scale,
               float(np.max(np.abs(g[tight] - nu))) / scale if tight.any() else 0.0)


PANELS = st.tuples(
    st.integers(min_value=8, max_value=40),      # T
    st.integers(min_value=6, max_value=25),      # M
    st.floats(min_value=0.0, max_value=1.0),     # factor strength
    st.integers(min_value=0, max_value=10_000),  # seed
)


@given(PANELS, st.integers(min_value=2, max_value=6))
@SETTINGS
def test_loss_is_nonnegative_and_under_the_best_vertex(panel, m):
    """PSD from below; a single-unit design is feasible, so from above too."""
    T, M, fs, seed = panel
    assume(m <= M)
    _, G = _panel(T, M, fs, seed)
    S = list(range(m))
    L = _loss(G, S)
    assert L >= -1e-9
    assert L <= min(G[j, j] for j in S) + 1e-7


@given(PANELS, st.integers(min_value=2, max_value=5))
@SETTINGS
def test_adding_a_unit_cannot_raise_the_loss(panel, m):
    """The hull only grows, so the distance to it only falls.

    A solver stopping short of the optimum can violate this, since its error is
    a property of each subproblem and the larger hull may come back worse. It
    did not do so here: this is one of the four properties the pre-existing
    Frank-Wolfe implementation still satisfied, so it guards the invariant
    without having caught that defect.
    """
    T, M, fs, seed = panel
    assume(m + 1 <= M)
    _, G = _panel(T, M, fs, seed)
    S = list(range(m))
    assert _loss(G, S + [m]) <= _loss(G, S) + 1e-7


@given(PANELS, st.integers(min_value=2, max_value=6))
@SETTINGS
def test_loss_depends_on_the_set_not_the_order(panel, m):
    T, M, fs, seed = panel
    assume(m <= M)
    _, G = _panel(T, M, fs, seed)
    rng = np.random.default_rng(seed + 1)
    S = list(rng.choice(M, size=m, replace=False))
    assert _loss(G, S) == pytest.approx(_loss(G, list(reversed(S))), rel=1e-9,
                                        abs=1e-12)


@given(PANELS, st.integers(min_value=2, max_value=5),
       st.floats(min_value=0.2, max_value=5.0))
@SETTINGS
def test_scaling_the_panel_scales_the_loss_quadratically(panel, m, c):
    """``L`` is a squared distance, so it carries ``c^2``."""
    T, M, fs, seed = panel
    assume(m <= M)
    X, _ = _panel(T, M, fs, seed)
    S = list(range(m))
    base = _loss(X.T @ X, S)
    scaled = _loss((c * X).T @ (c * X), S)
    assert scaled == pytest.approx(c * c * base, rel=1e-7, abs=1e-12)


@given(PANELS, st.integers(min_value=2, max_value=6))
@SETTINGS
def test_batched_and_single_solvers_agree(panel, m):
    """One ranks the candidates, the other reports the winner's loss."""
    T, M, fs, seed = panel
    assume(m <= M)
    _, G = _panel(T, M, fs, seed)
    rng = np.random.default_rng(seed + 2)
    subs = np.sort(rng.choice(M, size=(6, m), replace=True), axis=1)
    Qs = G[subs[:, :, None], subs[:, None, :]]
    batched = _afw_batched(Qs, iters=80)
    for k, Q in enumerate(Qs):
        single, _, _ = _afw_single(Q, iters=600, tol=1e-14)
        assert batched[k] == pytest.approx(single, rel=1e-7, abs=1e-10)


@given(PANELS, st.integers(min_value=2, max_value=6))
@SETTINGS
def test_the_reported_weights_certify_as_optimal(panel, m):
    """Feasible, and satisfying the KKT conditions of the program."""
    T, M, fs, seed = panel
    assume(m <= M)
    _, G = _panel(T, M, fs, seed)
    rng = np.random.default_rng(seed + 3)
    S = np.sort(rng.choice(M, size=m, replace=False))
    Q = G[np.ix_(S, S)]
    loss, w, lb = _afw_single(Q, iters=600, tol=1e-14)
    assert w.min() >= -1e-12
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert _kkt_residual(Q, w) < 1e-6
    assert loss == pytest.approx(float(w @ Q @ w), rel=1e-9, abs=1e-12)
    assert lb <= loss + 1e-9


@given(PANELS)
@SETTINGS
def test_a_hull_containing_the_origin_has_zero_loss(panel):
    """The centroid of every candidate is the origin by construction, so the
    full candidate set always attains it -- the degenerate case the module
    docstring names as the reason an exact branch-and-bound bound is weak."""
    T, M, fs, seed = panel
    _, G = _panel(T, M, fs, seed)
    assert _loss(G, list(range(M))) < 1e-8
