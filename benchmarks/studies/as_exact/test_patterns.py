"""Sign patterns of a face by branching, checked against one LP per sign pattern.

The sign patterns S(sigma) of the normal cone N(sigma) decide which part of a face is
attainable: a point w of the face is attainable exactly when some nonzero c in N(sigma)
has c_k r_k(w) >= 0 for every k. ``branch_patterns`` lists patterns, with 0 for a free
sign, whose completions are exactly the sign vectors s in {-1, +1}^K whose closed
orthant meets N(sigma) away from the origin. The oracle tests each of the 2^K sign
vectors with HiGHS and uses every donor outside the face as a wall of the cone, so it
also checks that the link vertices the search passes in are enough.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st
from scipy.optimize import linprog

from . import patterns
from .capture import random_instance
from .hull import Hull


def _face(seed, K, J, m, pick):
    """A face of m donors inside a visible facet of the first exterior design from `seed` on.

    The link is the star's vertices outside the face. A facet's star is the facet itself,
    so its cone, the ray of its normal, takes its one wall from the apex of a neighbour.
    """
    while (inst := random_instance(seed, K, J)) is None:
        seed += 1
    X1, X0, _, _ = inst
    hull = Hull(X0, X1)
    facets = sorted(sorted(F) for F in hull.visible_facets())
    F = facets[pick % len(facets)]
    sigma = frozenset(F[:m])
    star = hull.star(sigma, frozenset(F))
    link = set().union(*star) - sigma
    if not link:
        link = {next(iter(hull.neighbour(sigma, F[0]) - sigma))}
    return X0, sigma, link, hull, star


def _feasible_sign_vectors(X0, sigma):
    """Every s in {-1, +1}^K whose closed orthant meets N(sigma) away from 0, one HiGHS LP each."""
    K, J = X0.shape
    cols = sorted(sigma)
    a = cols[0]
    E = (X0[:, cols[1:]] - X0[:, [a]]).T
    W = np.array([X0[:, i] - X0[:, a] for i in range(J) if i not in sigma])
    g = -W.sum(axis=0)                      # g'c > 0 on the pointed cone minus the origin
    out = set()
    for s in itertools.product((-1, 1), repeat=K):
        res = linprog(np.zeros(K), A_ub=W, b_ub=np.zeros(len(W)), A_eq=np.vstack([E, g[None, :]]),
                      b_eq=np.r_[np.zeros(len(E)), 1.0], bounds=[(0, None) if x > 0 else (None, 0) for x in s],
                      method="highs")
        if res.status == 0:
            out.add(s)
    return out


def _completions(pats, K):
    """The sign vectors a list of patterns stands for: every 0 replaced by both signs."""
    out = set()
    for t in pats:
        free = [k for k in range(K) if t[k] == 0]
        for signs in itertools.product((-1, 1), repeat=len(free)):
            u = [int(x) for x in t]
            for k, s in zip(free, signs):
                u[k] = s
            out.add(tuple(u))
    return out


# --------------------------------------------------------------------------- smoke
def test_a_lower_face_gets_a_list_of_sign_patterns():
    X0, sigma, link, _, _ = _face(1003, 6, 16, 3, 0)
    pats = patterns.branch_patterns(X0, sigma, link)
    assert pats and all(p.shape == (6,) and set(np.unique(p)) <= {-1, 0, 1} for p in pats)


# --------------------------------------------------------------------------- unit: every face size, and the exact methods
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_every_face_size_matches_one_lp_per_sign_vector(m):
    X0, sigma, link, _, _ = _face(1003, 6, 16, m, 1)
    assert _completions(patterns.branch_patterns(X0, sigma, link), 6) == _feasible_sign_vectors(X0, sigma)


def test_a_facet_gets_the_signs_of_its_normal():
    X0, sigma, link, hull, _ = _face(1003, 6, 16, 6, 2)
    assert _completions(patterns.branch_patterns(X0, sigma, link), 6) == \
        _completions(patterns.facet_patterns(hull.normal[sigma][0]), 6)


def test_a_ridge_matches_the_sweep_between_its_two_facet_normals():
    X0, sigma, link, hull, star = _face(1003, 6, 16, 5, 3)
    a1, a2 = (hull.normal[G][0] for G in star)
    assert _completions(patterns.branch_patterns(X0, sigma, link), 6) == \
        _completions(patterns.ridge_patterns(a1, a2), 6)


def test_an_ambiguous_verdict_goes_to_highs(monkeypatch):
    """When neither a point nor a certificate checks, HiGHS decides, and the answer is unchanged."""
    X0, sigma, link, _, _ = _face(1003, 6, 16, 3, 4)
    expected = _feasible_sign_vectors(X0, sigma)
    stats, asked = {}, []
    monkeypatch.setattr(patterns.lsq, "ldp_verdict", lambda G, h: asked.append(1) or (None, None))
    assert _completions(patterns.branch_patterns(X0, sigma, link, stats=stats), 6) == expected
    assert asked and stats["pattern_lp_checks"] >= len(asked)


def test_a_witness_outranks_a_verdict_that_contradicts_it(monkeypatch):
    """If the zero test calls a sub-cube empty while a checked point with zeros on its free signs
    exists, the sub-cube is closed by that point: a verified point is the stronger evidence."""
    X0, sigma, link, _, _ = _face(1003, 6, 16, 3, 0)
    e0 = np.eye(6)[0]

    def contradicting(E, W, g, assign, stats, cache):
        if np.any(assign == patterns.FREE) and assign[0] in (1, patterns.FREE):
            return True, e0                     # the free test always finds e0
        return False, None                      # the zero test always says empty
    monkeypatch.setattr(patterns, "_cone_test", contradicting)
    assert [p.tolist() for p in patterns.branch_patterns(X0, sigma, link)] == [[1, 0, 0, 0, 0, 0]]


# --------------------------------------------------------------------------- property
@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 6), extra=st.integers(3, 10),
       drop=st.integers(0, 5), pick=st.integers(0, 50))
def test_property_the_listed_patterns_are_exactly_the_cone_s_sign_vectors(seed, K, extra, drop, pick):
    m = K - drop
    assume(m >= 1)
    X0, sigma, link, _, _ = _face(seed, K, K + extra, m, pick)
    pats = patterns.branch_patterns(X0, sigma, link)
    assert _completions(pats, K) == _feasible_sign_vectors(X0, sigma)


# --------------------------------------------------------------------------- failure
def test_a_face_without_a_link_is_refused():
    X0, sigma, _, _, _ = _face(1003, 6, 16, 3, 0)
    with pytest.raises(ValueError, match="link"):
        patterns.branch_patterns(X0, sigma, set())
