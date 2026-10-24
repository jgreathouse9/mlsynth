"""Sign patterns and the best attainable fit of a face (exact backend), checked against one LP per sign vector.

A point w of face sigma is attainable exactly when some nonzero c in the normal cone
N(sigma) has c_k r_k(w) >= 0 for every k, with r = X1 - X0 w, so the attainable part of
the face is the union of the pieces {w : s_k r_k(w) >= 0}, one for each sign vector s in
{-1, +1}^K whose closed orthant meets N(sigma) away from the origin. The oracle tests
each of the 2^K sign vectors with HiGHS and uses every donor outside the face as a wall
of the cone, so it also checks that the link vertices the code passes in are enough.
``facet_patterns`` and ``ridge_patterns`` are compared with it directly, and
``face_min``, the branch and bound over sign sub-cubes, with one least-squares solve per
sign vector the oracle admits.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from _exact_support import random_instance
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import linprog

from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.bilevel.exact import lsq, patterns
from mlsynth.utils.bilevel.exact.hull import Hull


def _face(seed, K, J, m, pick):
    """A face of m donors inside a visible facet F of the first exterior design from `seed` on.

    The link is the star's vertices outside the face. A facet's star is the facet itself,
    so its cone, the ray of its normal, takes its one wall from the apex of a neighbour.
    """
    while (inst := random_instance(seed, K, J)) is None:
        seed += 1
    X1, X0, Z1, Z0 = inst
    hull = Hull(X0, X1)
    facets = sorted(sorted(G) for G in hull.visible_facets())
    F = facets[pick % len(facets)]
    sigma = frozenset(F[:m])
    star = hull.star(sigma, frozenset(F))
    link = set().union(*star) - sigma
    if not link:
        link = {next(iter(hull.neighbour(sigma, F[0]) - sigma))}
    return SimpleNamespace(X1=X1, X0=X0, Z1=Z1, Z0=Z0, sigma=sigma, link=link, hull=hull, star=star,
                           F=frozenset(F))


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


def _oracle_min(f):
    """The best fit over the pieces of the sign vectors the oracle admits, one solve each."""
    best = (np.inf, None)
    for s in sorted(_feasible_sign_vectors(f.X0, f.sigma)):
        val, beta = lsq.sign_piece_ls(f.X0, f.X1, f.Z1, f.Z0, sorted(f.sigma), np.array(s))
        if val is not None and val < best[0]:
            best = (val, beta)
    return best


def _face_min(f, U=np.inf, **kw):
    return patterns.face_min(f.X0, f.X1, f.Z1, f.Z0, f.sigma, f.link, U, **kw)


def _same(got, want):
    if not np.isfinite(want[0]):
        assert got == (np.inf, None)
        return
    assert np.isclose(got[0], want[0], rtol=1e-9, atol=1e-12)
    assert np.allclose(got[1], want[1], atol=1e-6)


# (m, pick) on the K = 6 design from seed 1010: the face's own fit is unattainable on each,
# so the search branches before it finds the best piece
BRANCHING = {1: 0, 2: 47, 3: 33, 4: 6, 5: 6, 6: 23}


# --------------------------------------------------------------------------- smoke
def test_a_lower_face_gets_its_best_attainable_fit():
    f = _face(1010, 6, 16, 3, BRANCHING[3])
    val, beta = _face_min(f)
    assert np.isfinite(val) and beta.shape == (3,) and beta.min() >= 0.0 and abs(beta.sum() - 1.0) <= 1e-9
    assert np.isclose(val, np.sum((f.Z1 - f.Z0[:, sorted(f.sigma)] @ beta) ** 2), rtol=1e-9)


# --------------------------------------------------------------------------- unit: every face size
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_every_face_size_matches_one_solve_per_sign_vector(m):
    f = _face(1010, 6, 16, m, BRANCHING[m])
    _same(_face_min(f), _oracle_min(f))


def test_a_face_with_no_attainable_piece_gets_none():
    f = _face(1003, 6, 16, 3, 0)
    assert _oracle_min(f) == (np.inf, None)
    assert _face_min(f) == (np.inf, None)


def test_a_facet_gets_the_signs_of_its_normal():
    f = _face(1003, 6, 16, 6, 2)
    assert _completions(patterns.facet_patterns(f.hull.normal[f.sigma][0]), 6) == _feasible_sign_vectors(f.X0, f.sigma)


def test_a_ridge_matches_the_sweep_between_its_two_facet_normals():
    f = _face(1003, 6, 16, 5, 3)
    a1, a2 = (f.hull.normal[G][0] for G in f.star)
    assert _completions(patterns.ridge_patterns(a1, a2), 6) == _feasible_sign_vectors(f.X0, f.sigma)


# --------------------------------------------------------------------------- unit: the search
def test_an_attainable_face_fit_settles_the_face_in_one_test():
    """Passed in as the root's minimiser, an attainable fit costs one cone test and no solve."""
    f = _face(1010, 6, 16, 4, 33)
    u, beta_u = lsq.simplex_ls(f.Z1, f.Z0[:, sorted(f.sigma)])
    assert np.isclose(_oracle_min(f)[0], u, rtol=1e-9)
    stats = {}
    val, beta = _face_min(f, root=(np.zeros(6, int), u, beta_u), stats=stats)
    assert val == u and np.array_equal(beta, beta_u)
    assert stats == {"pattern_tests": 1}


def test_the_incumbent_prunes_what_cannot_beat_it():
    f = _face(1010, 6, 16, 3, BRANCHING[3])
    want = _oracle_min(f)
    assert _face_min(f, U=want[0] * (1 - 1e-9)) == (np.inf, None)
    _same(_face_min(f, U=want[0] * (1 + 1e-9)), want)


def test_an_ambiguous_verdict_goes_to_highs(monkeypatch):
    """When neither a point nor a certificate checks, HiGHS decides every test, and the answer
    is unchanged, including the sub-cubes HiGHS calls empty, which are split on every sign."""
    f = _face(1010, 6, 16, 4, BRANCHING[4])
    want = _oracle_min(f)
    stats = {}
    monkeypatch.setattr(patterns.lsq, "ldp_verdict", lambda G, h: (None, None))
    _same(_face_min(f, stats=stats), want)
    assert stats["pattern_lp_checks"] == stats["pattern_tests"] > 1


# --------------------------------------------------------------------------- edge
def test_signs_no_point_of_the_face_meets_give_no_fit():
    """On a face of a visible facet with normal n, n'r(w) > 0, so the signs -sign(n) leave no point;
    the root's solve finds the region empty and no cone test is made."""
    f = _face(1010, 6, 16, 3, BRANCHING[3])
    n = f.hull.normal[f.F][0]
    stats = {}
    assert _face_min(f, root=(-np.sign(n).astype(int), 0.0, None), stats=stats) == (np.inf, None)
    assert stats == {"solves": 1}


# --------------------------------------------------------------------------- property
@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 6), extra=st.integers(3, 10),
       drop=st.integers(0, 5), pick=st.integers(0, 50))
def test_property_the_search_finds_the_best_attainable_piece(seed, K, extra, drop, pick):
    m = K - drop
    assume(m >= 1)
    f = _face(seed, K, K + extra, m, pick)
    _same(_face_min(f), _oracle_min(f))


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 7), extra=st.integers(3, 10),
       ridge=st.booleans(), pick=st.integers(0, 50))
def test_property_the_closed_forms_list_the_cone_s_sign_vectors(seed, K, extra, ridge, pick):
    f = _face(seed, K, K + extra, K - 1 if ridge else K, pick)
    if ridge:
        assume(len(f.star) == 2)
        pats = patterns.ridge_patterns(*(f.hull.normal[G][0] for G in f.star))
    else:
        pats = patterns.facet_patterns(f.hull.normal[f.sigma][0])
    assert _completions(pats, K) == _feasible_sign_vectors(f.X0, f.sigma)


# --------------------------------------------------------------------------- failure
def test_a_face_without_a_link_is_refused():
    f = _face(1010, 6, 16, 3, BRANCHING[3])
    with pytest.raises(MlsynthEstimationError, match="link"):
        patterns.face_min(f.X0, f.X1, f.Z1, f.Z0, f.sigma, set(), np.inf)


def test_a_link_inside_the_face_s_affine_hull_is_refused():
    """A wall in the span of the face leaves the normalisation g'c = 1 inconsistent with E c = 0."""
    X0 = np.array([[0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    Z0 = np.eye(3, 4)
    with pytest.raises(MlsynthDataError, match="normal cone"):
        patterns.face_min(X0, np.array([0.5, -1.0]), np.zeros(3), Z0, frozenset({0, 1}), {2}, np.inf)


# --------------------------------------------------------------------------- the cone test and the edge search
def _cone_oracle(X0, sigma, sgn, zero):
    """Some orthant agreeing with sgn off the zero set (either sign on it) meets N(sigma) away from 0."""
    admitted = _feasible_sign_vectors(X0, sigma)
    return any(all(s[k] == sgn[k] for k in range(len(sgn)) if not zero[k]) for s in admitted)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 5), extra=st.integers(3, 8), m=st.integers(1, 3),
       pick=st.integers(0, 50), signs=st.integers(0, 2 ** 31), nzero=st.integers(0, 2))
def test_property_cone_compatible_matches_orthant_enumeration(seed, K, extra, m, pick, signs, nzero):
    f = _face(seed, K, K + extra, m, pick)
    rng = np.random.default_rng(signs)
    sgn = rng.choice([-1.0, 1.0], size=K)
    zero = np.zeros(K, bool)
    zero[rng.choice(K, size=nzero, replace=False)] = True
    sgn[zero] = 0.0
    assert patterns.cone_compatible(f.X0, f.sigma, sgn, zero) == _cone_oracle(f.X0, f.sigma, sgn, zero)


def _edge_oracle(X0, X1, z, Z, sigma):
    """The best fit over the union of the edge's pieces: each admitted orthant s leaves an interval of
    t, the fit ||a - t b||^2 is convex in t, so its minimum on an interval is the clipped free minimiser."""
    i, j = sorted(sigma)
    g, h = X1 - X0[:, i], X0[:, j] - X0[:, i]
    a, b = z - Z[:, i], Z[:, j] - Z[:, i]
    t_free = (a @ b) / (b @ b)
    best = np.inf
    for s in _feasible_sign_vectors(X0, sigma):
        lo, hi = 0.0, 1.0
        for k in range(len(s)):                         # s_k (g_k - t h_k) >= 0
            if abs(h[k]) <= 1e-12:
                if s[k] * g[k] < 0:
                    lo, hi = 1.0, 0.0
            elif s[k] * h[k] > 0:
                hi = min(hi, g[k] / h[k])
            else:
                lo = max(lo, g[k] / h[k])
        if lo <= hi:
            t = float(np.clip(t_free, lo, hi))
            best = min(best, float(np.sum((a - t * b) ** 2)))
    return best


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 6), extra=st.integers(3, 8), pick=st.integers(0, 50))
def test_property_an_edge_gets_its_best_attainable_fit(seed, K, extra, pick):
    f = _face(seed, K, K + extra, 2, pick)
    (val, t), _ = patterns.edge_min(f.X0, f.X1, f.Z1, f.Z0, f.sigma, np.inf)
    want = _edge_oracle(f.X0, f.X1, f.Z1, f.Z0, f.sigma)
    if not np.isfinite(want):
        assert val == np.inf and t is None
        return
    assert np.isclose(val, want, rtol=1e-9, atol=1e-12)
    i, j = sorted(f.sigma)
    a, b = f.Z1 - f.Z0[:, i], f.Z0[:, j] - f.Z0[:, i]
    assert np.isclose(np.sum((a - t * b) ** 2), val, rtol=1e-12)


def test_an_edge_attainable_at_one_point_gets_that_point():
    """On edge {2, 3} of this design two residuals vanish together at t = 1/2: the open intervals on
    either side are not attainable, the point between them is, and it is the edge's answer."""
    X0 = np.array([[-1.0, -2.0, 0.0, 1.0, 2.0], [2.0, -2.0, 3.0, -3.0, 1.0], [3.0, 1.0, 3.0, -1.0, 3.0]])
    X1 = np.array([2.0, 0.0, 1.0])
    z, Z = np.zeros(5), np.eye(5)
    (val, t), n_lp = patterns.edge_min(X0, X1, z, Z, frozenset({2, 3}), np.inf)
    assert t == 0.5 and np.isclose(val, 0.5) and n_lp >= 3
    assert np.allclose(X1 - (0.5 * X0[:, 2] + 0.5 * X0[:, 3]), [1.5, 0.0, 0.0])
    assert _edge_oracle(X0, X1, z, Z, frozenset({2, 3})) == pytest.approx(0.5)


def test_a_cone_vector_supported_on_the_zero_residuals_counts():
    """At vertex (0, 0) of this triangle every normal has c_1 <= 0, so no c meets r = (0, 2) with
    c_1 > 0; but c = (1, 0), supported where the residual vanishes, is in the normal cone."""
    X0 = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.5]])
    sgn, zero = np.array([0.0, 1.0]), np.array([True, False])
    assert patterns.cone_compatible(X0, frozenset({0}), sgn, zero)
    assert _cone_oracle(X0, frozenset({0}), sgn, zero)
    assert not patterns.cone_compatible(X0, frozenset({0}), np.array([0.0, 1.0]), np.array([False, False]))
