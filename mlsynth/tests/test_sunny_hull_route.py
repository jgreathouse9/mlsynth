"""Reading the sunny set off one convex hull instead of one program per donor.

Test-first (per ``agents/agents_tests.md``): written before the route exists, so
every test here is RED until it lands.

Eq (9) solves one linear program per donor. Below a small reduced row count the
same answer is a single convex hull: donor ``j`` is sunny exactly when ``x_j``
lies on a facet whose supporting hyperplane separates the origin. qhull returns
the hull as ``{x : A x + b <= 0}``, so facet ``i`` has ``h_H(a_i) = -b_i`` and is
lit exactly when ``b_i > 0``.

Two things about this route cannot be established by random designs, and both are
pinned below.

The predicate is membership of a lit facet, not membership of ``hull.vertices``.
A sunny donor can lie in the relative interior of a lit facet, where qhull does
not report it as a vertex. Over 360 random designs across eleven shapes the two
readings agreed every time, because Gaussian points are in general position and
no donor ever lands inside a facet; only a constructed collinear design separates
them.

And qhull raises on a degenerate point set, so the route needs a fallback that no
random design will ever exercise.

The threshold is measured, not assumed. The hull route wins by 21x or more for a
reduced row count of 6 at every donor count up to 500; at 8 it is about 3x slower
than the programs once there are 100 donors or more, because a hull of ``n``
points in dimension ``d`` carries up to about ``n**(d // 2)`` facets while the
programs cost only one solve per donor.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.solvers import sunny as S
from mlsynth.utils.solvers.sunny import (
    DEFAULT_TOL,
    _centred,
    _reduced,
    sunny_alphas,
    sunny_donors,
)


# --------------------------------------------------------------------------- #
# the design that separates a lit facet from a hull vertex
# --------------------------------------------------------------------------- #
def _facet_interior_design():
    """Centred columns whose middle donor lies inside a lit facet.

    Three columns are collinear on the face ``x = 1``, which is the face the
    origin sees, and a fourth sits behind it at ``(3, 0)``. The middle of the
    three is on that face but is not a vertex of the hull, so a route that reads
    ``hull.vertices`` calls it shady. ``alpha* = (1, 1, 1, 1/3)``.
    """
    return np.array([[1.0, 1.0, 1.0, 3.0],
                     [-1.0, 0.0, 1.0, 0.0]]), np.zeros(2)


def test_a_donor_inside_a_lit_facet_is_sunny():
    B, A = _facet_interior_design()
    assert list(sunny_donors(B, A)) == [True, True, True, False]


def test_that_design_really_does_put_a_donor_inside_a_facet():
    """Guards the guard: if the construction stopped being degenerate the test
    above would pass for the wrong reason, against a hull whose every boundary
    point is a vertex."""
    from scipy.spatial import ConvexHull

    B, A = _facet_interior_design()
    Xt = _reduced(_centred(B, A))
    assert np.allclose(sunny_alphas(B, A), [1.0, 1.0, 1.0, 1.0 / 3.0])
    hull = ConvexHull(Xt.T)
    assert 1 not in set(hull.vertices)          # sunny, yet not a hull vertex


# --------------------------------------------------------------------------- #
# the two routes agree
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,J", [(2, 8), (3, 10), (3, 30), (4, 12), (5, 20),
                                 (5, 74), (6, 15), (6, 40), (2, 50)])
def test_the_hull_route_agrees_with_the_programs(m, J):
    rng = np.random.default_rng(m * 100 + J)
    for rep in range(12):
        B = rng.normal(size=(m, J))
        A = rng.normal(size=m) * (1.0 if rep % 3 else 0.2)   # mix 0 in / out of H
        assert np.array_equal(sunny_donors(B, A),
                              sunny_alphas(B, A) >= 1.0 - DEFAULT_TOL)


def test_no_linear_program_runs_below_the_threshold(monkeypatch):
    """The point of the route is that the programs do not run at all."""
    calls = {"n": 0}
    real = S.linprog

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(S, "linprog", counted)
    rng = np.random.default_rng(0)
    B, A = rng.normal(size=(5, 40)), rng.normal(size=5)
    sunny_donors(B, A, certify=False)
    assert calls["n"] == 0


def test_the_counter_actually_intercepts(monkeypatch):
    """Control for the test above: patched the same way, a design over the
    threshold must still reach the programs. Without this, a counter wired to the
    wrong name would report zero for both."""
    calls = {"n": 0}
    real = S.linprog

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(S, "linprog", counted)
    rng = np.random.default_rng(0)
    B, A = rng.normal(size=(9, 30)), rng.normal(size=9)
    sunny_donors(B, A, certify=False)
    assert calls["n"] > 0


# --------------------------------------------------------------------------- #
# the threshold itself
# --------------------------------------------------------------------------- #
def test_the_threshold_is_six():
    """Measured, not assumed: at seven reduced rows the margin decays to about 3x
    by 200 donors, and at eight the hull route is roughly 3x slower than the
    programs once there are 100 donors or more."""
    assert S._HULL_MAX_ROWS == 6


def test_the_route_is_taken_at_the_threshold_and_not_above(monkeypatch):
    seen = {}

    def spy(Xt, tol):
        seen["rows"] = Xt.shape[0]
        return None                                   # force the fallback

    monkeypatch.setattr(S, "_sunny_via_hull", spy)
    rng = np.random.default_rng(1)
    sunny_donors(rng.normal(size=(6, 20)), rng.normal(size=6), certify=False)
    assert seen.get("rows") == 6

    seen.clear()
    sunny_donors(rng.normal(size=(7, 20)), rng.normal(size=7), certify=False)
    assert "rows" not in seen


# --------------------------------------------------------------------------- #
# the fallback no random design reaches
# --------------------------------------------------------------------------- #
def test_a_qhull_failure_falls_back_to_the_programs(monkeypatch):
    from scipy.spatial import QhullError

    def boom(*a, **k):
        raise QhullError("forced")

    monkeypatch.setattr(S, "ConvexHull", boom)
    rng = np.random.default_rng(3)
    B, A = rng.normal(size=(4, 20)), rng.normal(size=4)
    assert np.array_equal(sunny_donors(B, A),
                          sunny_alphas(B, A) >= 1.0 - DEFAULT_TOL)


def test_a_degenerate_point_set_still_classifies():
    """Duplicate donors make the reduced design rank deficient, which is the
    shape qhull refuses. Whether it raises or not, the verdict must be right and
    the duplicates must classify alike."""
    rng = np.random.default_rng(5)
    B = rng.normal(size=(4, 12))
    B[:, 3] = B[:, 2]                                 # exact duplicate
    A = rng.normal(size=4)
    got = sunny_donors(B, A)
    assert np.array_equal(got, sunny_alphas(B, A) >= 1.0 - DEFAULT_TOL)
    assert got[2] == got[3]


# --------------------------------------------------------------------------- #
# edges the route must not get wrong
# --------------------------------------------------------------------------- #
def test_the_origin_inside_the_hull_reports_every_donor_shady():
    """No lit facet exists, and by Proposition 1 that says an exact fit exists."""
    X0 = np.array([[0.0, 2.0, 1.0, 3.0],
                   [0.0, 0.0, 2.0, 2.0]])
    X1 = X0 @ np.full(4, 0.25)
    assert not sunny_donors(X0, X1).any()


def test_a_donor_equal_to_the_treated_unit_is_shady():
    """Its centred column is the origin, so ``alpha* = 0``: shady, and dropping it
    would discard an exact fit, which is why Proposition 2 carries its hypothesis."""
    B = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, -1.0]])
    A = np.zeros(2)
    assert not sunny_donors(B, A)[0]


def test_every_column_equal_to_the_treated_unit():
    B = np.zeros((3, 4))
    A = np.zeros(3)
    assert not sunny_donors(B, A).any()


def test_a_single_reduced_row_skips_the_hull_route(monkeypatch):
    """qhull has no one-dimensional hull. A single pre-period reduces to one row,
    and routing it through the hull raised ``ValueError: Need at least 2-D data``
    where the screen had always worked."""
    seen = {}

    def spy(Xt, tol):
        seen["called"] = True
        return None

    monkeypatch.setattr(S, "_sunny_via_hull", spy)
    B = np.array([[1.0, 2.0, 3.0]])
    A = np.array([0.5])
    got = sunny_donors(B, A)
    assert "called" not in seen
    assert np.array_equal(got, sunny_alphas(B, A) >= 1.0 - DEFAULT_TOL)


def test_qhull_refusing_with_a_value_error_also_falls_back(monkeypatch):
    def boom(*a, **k):
        raise ValueError("Need at least 2-D data")

    monkeypatch.setattr(S, "ConvexHull", boom)
    rng = np.random.default_rng(9)
    B, A = rng.normal(size=(4, 15)), rng.normal(size=4)
    assert np.array_equal(sunny_donors(B, A),
                          sunny_alphas(B, A) >= 1.0 - DEFAULT_TOL)
