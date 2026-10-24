"""The donor hull of the exact backend (``mlsynth.utils.bilevel.exact.hull``), against Qhull.

``Hull`` discovers facets on demand by gift wrapping: the facets visible from X1 by a
walk over the facet-ridge graph, the star of a face by a walk over the ridges that
contain it, and each neighbour by one QR factorisation and one angle sweep. The oracle
is SciPy's ``ConvexHull`` (Qhull), which lists every facet with its outward equation.
"""
from __future__ import annotations

import numpy as np
import pytest
from _exact_support import interior_design, random_instance
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.spatial import ConvexHull

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.bilevel.exact import hull


def _qhull(X0):
    """{facet: (outward unit normal, offset)} with n'x <= b inside."""
    qh = ConvexHull(X0.T)
    return {frozenset(int(i) for i in s): (eq[:-1], -eq[-1]) for s, eq in zip(qh.simplices, qh.equations)}


def _design(seed, K, extra):
    inst = random_instance(seed, K, K + extra)
    assume(inst is not None)
    return inst[0], inst[1]


# --------------------------------------------------------------------------- properties against Qhull
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 5), extra=st.integers(2, 10))
def test_property_the_visible_facets_are_qhull_s(seed, K, extra):
    X1, X0 = _design(seed, K, extra)
    facets = _qhull(X0)
    want = {F for F, (n, b) in facets.items() if n @ X1 > b + hull.TOL}
    H = hull.Hull(X0, X1)
    assert H.visible_facets() == want
    for F in want:                                       # the same outward hyperplane
        n, b = H.normal[F]
        assert np.allclose(n, facets[F][0], atol=1e-9) and np.isclose(b, facets[F][1], atol=1e-9)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 5), extra=st.integers(2, 10), m=st.integers(1, 4),
       pick=st.integers(0, 10_000))
def test_property_a_star_is_every_facet_holding_the_face(seed, K, extra, m, pick):
    X1, X0 = _design(seed, K, extra)
    assume(m < K)
    facets = _qhull(X0)
    H = hull.Hull(X0, X1)
    vis = sorted(sorted(F) for F in H.visible_facets())
    F0 = vis[pick % len(vis)]
    sigma = frozenset(F0[:m])
    assert H.star(sigma, frozenset(F0)) == {F for F in facets if sigma <= F}


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 5), extra=st.integers(2, 10), pick=st.integers(0, 10_000))
def test_property_a_neighbour_shares_the_ridge(seed, K, extra, pick):
    """Across ridge F - {apex} lies the one other facet of Qhull that contains it, and the cache is symmetric."""
    X1, X0 = _design(seed, K, extra)
    facets = _qhull(X0)
    H = hull.Hull(X0, X1)
    F = sorted(H.visible_facets(), key=sorted)[pick % len(H.visible_facets())]
    for apex in F:
        G = H.neighbour(F, apex)
        ridge = F - {apex}
        assert G == next(E for E in facets if ridge <= E and E != F)
        assert H.neighbour(G, next(iter(G - ridge))) == F


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 5), extra=st.integers(2, 10), inward=st.booleans())
def test_property_outside_hull_agrees_with_the_facet_equations(seed, K, extra, inward):
    """X1 is outside exactly when some facet's equation is positive at it; the designs keep a margin."""
    X1, X0 = (interior_design(seed, K, K + extra + 1) if inward else _design(seed, K, extra))[:2]
    margin = max(n @ X1 - b for n, b in _qhull(X0).values())
    assume(abs(margin) > 1e-6)
    assert hull.outside_hull(X1, X0) == (margin > 0)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(1, 6), rank=st.integers(0, 6), J=st.integers(2, 20))
def test_property_affine_rank_is_the_dimension_of_the_affine_hull(seed, K, rank, J):
    rank = min(rank, K, J - 1)
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(K, rank)) @ rng.normal(size=(rank, J)) + rng.normal(size=(K, 1))
    assert hull.affine_rank(X0) == rank


# --------------------------------------------------------------------------- edge cases
def test_a_ray_through_a_vertex_is_cast_again():
    """From the centre of a square toward (2, 2) the ray leaves through the vertex (1, 1), which lies
    in two facets; the start is moved and the ray recast until it exits through one facet's interior."""
    X0 = np.array([[1.0, -1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]])
    X1 = np.array([2.0, 2.0])
    assert hull.Hull(X0, X1).visible_facets() == {frozenset({0, 1}), frozenset({0, 3})}


def test_points_without_a_hyperplane_through_them_have_none():
    P = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])     # three collinear points in 3-D
    assert hull._hyperplane(P, np.zeros(3)) is None


def test_a_degenerate_facet_is_refused():
    """Three collinear donors in 3-D span no plane, so no facet can be built on them."""
    X0 = np.array([[0.0, 1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 2.0, 3.0, 0.0], [0.0, 1.0, 2.0, 0.0, 3.0]])
    H = hull.Hull(X0, np.array([5.0, 5.0, 5.0]))
    with pytest.raises(MlsynthDataError, match="degenerate facet"):
        H.add(frozenset({0, 1, 2}))


def test_a_seed_that_never_lands_on_a_facet_is_reported(monkeypatch):
    """Every cast ray exits through a lower-dimensional face: the walk has nowhere to start."""
    X0 = np.array([[1.0, -1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]])

    class Vertex:
        status, x = 0, np.r_[1.0, 0.0, 0.0, 0.0, 1.0]
    monkeypatch.setattr(hull, "linprog", lambda *a, **k: Vertex)
    with pytest.raises(MlsynthDataError, match="could not find a visible facet"):
        hull._seed_facet(X0, np.array([2.0, 2.0]), X0.mean(axis=1))


# --------------------------------------------------------------------------- failure
@pytest.mark.parametrize("X0", [
    np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]),          # donors on a line in 2-D
    np.array([[0.0, 1.0], [0.0, 1.0]]),                                # fewer than K + 1 donors
])
def test_a_hull_of_fewer_than_k_dimensions_is_refused(X0):
    with pytest.raises(MlsynthDataError, match="fewer than K dimensions"):
        hull.Hull(X0, np.array([5.0, 4.0]))


def test_no_predictors_have_rank_zero():
    assert hull.affine_rank(np.empty((0, 4))) == 0
