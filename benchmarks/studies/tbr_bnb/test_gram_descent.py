"""The incumbent's RSS descent scored from the Gram (#965).

search.incumbent descends by add, drop and swap moves from random starts. With
c = X'y, G = X'X and v = G u kept for the current design, a move's c'u and u'Gu are
a few entries of c, v and G, so every move of a step is scored at once, and the
design's RSS is y'y - (c'u)^2 / u'Gu when c'u > 0, else y'y (hull.positive_rss).
The scores differ from positive_rss's only by rounding, so the descent reaches the
designs the series-scored descent does; that one is kept here as the reference."""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from .brute import geo_panel, panel
from .hull import positive_rss
from .search import incumbent
from .treatment import _split

SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _reference(y, X, *, restarts, seed=0):
    """The descent as it was: every move's RSS from the summed series."""
    p = X.shape[1]
    rng = np.random.default_rng(seed)

    def descend(C):
        cur = positive_rss(y, X, C)
        while True:
            moves = [mv for mv in ([C ^ {j} for j in range(p)]
                                   + [(C - {a}) | {b} for a in sorted(C) for b in range(p) if b not in C]) if mv]
            if not moves:
                return cur, C
            vals = [positive_rss(y, X, mv) for mv in moves]
            i = int(np.argmin(vals))
            if vals[i] < cur * (1 - 1e-12):
                C, cur = moves[i], vals[i]
            else:
                return cur, C

    starts = [set(rng.choice(p, size=int(rng.integers(1, p + 1)), replace=False).tolist())
              for _ in range(restarts)]
    return min((descend(C) for C in starts), key=lambda r: r[0])


def _local_optimum(y, X, C, v):
    """No add, drop or swap move from C lowers the positive-slope RSS below v by more
    than the descent's own tolerance."""
    p = X.shape[1]
    moves = [C ^ {j} for j in range(p)] + [(C - {a}) | {b} for a in C for b in range(p) if b not in C]
    return all(positive_rss(y, X, mv) >= v * (1 - 1e-9) for mv in moves if mv)


# ---------------------------------------------------------------- smoke


def test_smoke_a_geo_group_reaches_the_reference_design():
    M = geo_panel(3, n=60, p=12)
    y, X, _ = _split(M, (0, 5))
    v, C = incumbent(y, X, restarts=8)
    v0, C0 = _reference(y, X, restarts=8)
    assert C == C0 and v == pytest.approx(v0, rel=1e-10)


# ---------------------------------------------------------------- invariants


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(2, 9), restarts=st.integers(1, 4))
def test_property_the_descent_reaches_the_reference_designs(seed, p, restarts):
    y, X = panel(seed % 500, p=p)
    v, C = incumbent(y, X, restarts=restarts, seed=seed)
    v0, C0 = _reference(y, X, restarts=restarts, seed=seed)
    assert v == pytest.approx(v0, rel=1e-9)
    assert C == C0 or positive_rss(y, X, C) == pytest.approx(v0, rel=1e-9)


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(1, 9), restarts=st.integers(1, 4))
def test_property_the_design_is_a_local_optimum_scored_as_reported(seed, p, restarts):
    y, X = panel(seed % 500, p=p)
    v, C = incumbent(y, X, restarts=restarts, seed=seed)
    assert C and C <= set(range(p))
    assert v == pytest.approx(positive_rss(y, X, C), rel=1e-9)
    assert _local_optimum(y, X, C, v)


def test_the_best_restart_is_kept():
    y, X = panel(7, p=8)
    each = [incumbent(y, X, restarts=1, seed=s)[0] for s in range(6)]
    v, _ = incumbent(y, X, restarts=1, seed=int(np.argmin(each)))
    assert v == min(each)


# ---------------------------------------------------------------- edges


def test_one_candidate_control():
    y, X = panel(2, p=1)
    v, C = incumbent(y, X, restarts=3)
    assert C == {0} and v == pytest.approx(positive_rss(y, X, {0}), rel=1e-12)


def test_no_design_has_a_positive_slope():
    """Every control positive and y negative: y'x < 0 for every design, every design
    scores y'y, and the descent stops at its start."""
    _, X = panel(4, p=5)
    X = np.abs(X) + 1.0
    y = -X.sum(axis=1)
    v, C = incumbent(y, X, restarts=2)
    assert v == pytest.approx(float(y @ y), rel=1e-12) and C


def test_identical_controls():
    y, X = panel(5, p=4)
    X = np.column_stack([X, X[:, 0]])                       # a duplicate column ties its moves
    v, C = incumbent(y, X, restarts=4)
    assert v == pytest.approx(positive_rss(y, X, C), rel=1e-9) and _local_optimum(y, X, C, v)


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_restart_count_is_refused(bad):
    y, X = panel(1, p=4)
    with pytest.raises(MlsynthDataError, match="restarts"):
        incumbent(y, X, restarts=bad)
