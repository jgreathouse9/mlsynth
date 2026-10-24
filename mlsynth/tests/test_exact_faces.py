"""Faces of the donor hull by dimension (the paper's Table 3), from the exact backend.

``faces.face_counts`` counts every face of conv(X0), those of the facets visible from
X1, and the visible faces that hold an attainable point: some w on the face and some
nonzero c in its normal cone with c_k r_k(w) >= 0 for every k, r = X1 - X0 w. The
authors' counts come from the captured ``run_all.py`` (California and West Germany);
on small designs the oracle tests each visible face by brute force, one HiGHS LP per
sign vector of {-1, +1}^K for the cone and one per sign vector for the face's piece.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from _exact_support import EXTERIOR, INTERIOR, inputs, random_instance
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.bilevel.exact import faces
from mlsynth.utils.bilevel.exact.structures import FaceCounts

RUN_ALL = json.loads((Path(__file__).resolve().parents[2] / "benchmarks" / "reference" / "vanillasc_exact"
                      / "reference.json").read_text())["values"]


def _authors(tag, K):
    return {kind: tuple(int(RUN_ALL[f"{tag}.faces.{kind}[{d}]"]) for d in range(K))
            for kind in ("total", "visible", "attainable")}


# --------------------------------------------------------------------------- the authors' Table 3
@pytest.mark.parametrize("name,tag", [("prop99", "california"), ("germany", "germany")])
def test_face_counts_match_the_authors(name, tag):
    X1, X0, _, _ = inputs(EXTERIOR[name])
    got = faces.face_counts(X1, X0)
    assert isinstance(got, FaceCounts)
    want = _authors(tag, X0.shape[0])
    assert (got.total, got.visible, got.attainable) == (want["total"], want["visible"], want["attainable"])


# --------------------------------------------------------------------------- oracle on small designs
def _facets(X0):
    hull = ConvexHull(X0.T)
    out = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        out[frozenset(int(i) for i in simplex)] = eq
    return out


def _cone_orthants(X0, sigma):
    """The sign vectors s whose closed orthant meets N(sigma) away from 0 (every donor a wall)."""
    K, J = X0.shape
    cols = sorted(sigma)
    a = cols[0]
    E = (X0[:, cols[1:]] - X0[:, [a]]).T
    W = np.array([X0[:, i] - X0[:, a] for i in range(J) if i not in sigma])
    g = -W.sum(axis=0)
    out = []
    for s in itertools.product((-1, 1), repeat=K):
        res = linprog(np.zeros(K), A_ub=W, b_ub=np.zeros(len(W)), A_eq=np.vstack([E, g[None, :]]),
                      b_eq=np.r_[np.zeros(len(E)), 1.0], bounds=[(0, None) if x > 0 else (None, 0) for x in s],
                      method="highs")
        if res.status == 0:
            out.append(np.array(s))
    return out


def _piece_nonempty(X1, X0, sigma, s):
    """Some w on the face with s_k r_k(w) >= 0, i.e. s_k (X0 w)_k <= s_k X1_k, for every k."""
    cols = sorted(sigma)
    m = len(cols)
    res = linprog(np.zeros(m), A_ub=s[:, None] * X0[:, cols], b_ub=s * X1,
                  A_eq=np.ones((1, m)), b_eq=[1.0], bounds=(0, None), method="highs")
    return res.status == 0


def _brute_force(X1, X0):
    K = X0.shape[0]
    facets = _facets(X0)
    visible = [F for F, eq in facets.items() if eq[:-1] @ X1 + eq[-1] > 1e-9]
    every = {frozenset(c) for F in facets for k in range(1, K + 1) for c in itertools.combinations(sorted(F), k)}
    seen = {frozenset(c) for F in visible for k in range(1, K + 1) for c in itertools.combinations(sorted(F), k)}
    att = {f for f in seen if any(_piece_nonempty(X1, X0, f, s) for s in _cone_orthants(X0, f))}
    count = lambda fs: tuple(sum(1 for f in fs if len(f) == d + 1) for d in range(K))  # noqa: E731
    return count(every), count(seen), count(att)


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 4), extra=st.integers(3, 7))
def test_property_face_counts_match_brute_force(seed, K, extra):
    inst = random_instance(seed, K, K + extra) if K > 2 else None
    if K == 2:
        rng = np.random.default_rng(seed)
        X0 = rng.normal(size=(2, 2 + extra))
        X1 = X0.max(axis=1) + 0.5 * rng.uniform(size=2)
        inst = (X1, X0, None, None)
    assume(inst is not None)
    X1, X0 = inst[0], inst[1]
    got = faces.face_counts(X1, X0)
    assert (got.total, got.visible, got.attainable) == _brute_force(X1, X0)


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 5), extra=st.integers(3, 10))
def test_property_counts_obey_euler_and_nest(seed, K, extra):
    """The boundary of a simplicial K-polytope is a (K-1)-sphere: sum (-1)^d f_d = 1 - (-1)^K;
    and attainable faces are visible, visible faces are faces."""
    inst = random_instance(seed, K, K + extra)
    assume(inst is not None)
    got = faces.face_counts(inst[0], inst[1])
    assert sum((-1) ** d * f for d, f in enumerate(got.total)) == 1 - (-1) ** K
    assert all(a <= v <= t for a, v, t in zip(got.attainable, got.visible, got.total))
    assert got.visible[K - 1] >= 1


# --------------------------------------------------------------------------- edge and failure
def test_a_simplex_hull_has_every_subset_as_a_face():
    X0 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    got = faces.face_counts(np.array([2.0, 2.0]), X0)
    assert got.total == (3, 3) and got.visible == (2, 1)


def test_an_interior_treated_unit_sees_no_face():
    X1, X0, _, _ = inputs(INTERIOR["random_interior_K3_J12_seed4007"])
    got = faces.face_counts(X1, X0)
    assert got.visible == (0, 0, 0) and got.attainable == (0, 0, 0) and sum(got.total) > 0


def test_a_facet_with_more_than_k_donors_is_refused():
    """A cube's faces hold four donors each, so its hull is not simplicial (general position fails)."""
    X0 = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]).T
    with pytest.raises(MlsynthDataError, match="holds 4 donors"):
        faces.face_counts(np.array([2.0, 2.5, 3.0]), X0)


def test_a_lower_dimensional_hull_is_refused():
    X0 = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(MlsynthDataError, match="fewer than K dimensions"):
        faces.face_counts(np.array([5.0, 4.0]), X0)
