"""Every lift level is a relaxation, the nested ones are ordered, and an infeasible
piece contributes nothing."""
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from .brute import panel, piece_optimum
from .hull import ControlBound, rss_of
from .lifts import Lift

SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
F = ["u", "1-u", "a", "c"]
FAMILIES = {(a, g) for a in F for g in F}
VARIANTS = {"shor": dict(level="shor"), "persp": dict(level="persp"), "rlt": dict(level="rlt"),
            "full": dict(level="full"), "full+pairs": dict(level="full", pairs=True),
            "no V=V'": dict(level="full", keep_sym=False),
            "slope products only": dict(level="full", skip=FAMILIES - {("a", "a"), ("a", "c"), ("c", "a"), ("c", "c")})}


def _value(y, X, lo, hi, m, kw):
    kw = dict(kw)
    return Lift(y, X, kw.pop("level"), **kw).piece(lo, hi, m)


#: The solver's own value of a lift, compared with the enumerated optimum, is good
#: to Clarabel's accuracy, which on these lifts reaches 8e-5 relative (#965: 3e-5
#: on ten geos, 8e-5 on 40). Validity beyond it is the exact certificate's job
#: (test_certificate.py); here the levels are checked to the solver's accuracy.
SOLVER_ACCURACY = 2e-4


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.05, 1.0), width=st.floats(1.1, 3.0),
       m=st.integers(1, 5), name=st.sampled_from(sorted(VARIANTS)))
@example(seed=137, lo=1.0, width=2.0, m=1, name="full+pairs")   # 5.7e-5 above, by the solver
def test_every_level_is_a_relaxation(seed, lo, width, m, name):
    y, X = panel(seed, p=5)
    hi = lo * width
    v = _value(y, X, lo, hi, m, VARIANTS[name])
    assert v <= piece_optimum(y, X, lo, hi, m) * (1 + SOLVER_ACCURACY) + 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_the_nested_levels_are_ordered(seed):
    """Each level adds constraints to the one before it, so its value can only rise."""
    y, X = panel(seed, p=5)
    vals = [_value(y, X, 0.3, 0.9, 3, VARIANTS[k]) for k in ("shor", "persp", "rlt")]
    tol = 1e-6 * abs(vals[-1])
    assert vals[0] <= vals[1] + tol and vals[1] <= vals[2] + tol
    full = _value(y, X, 0.3, 0.9, 3, VARIANTS["full"])
    assert _value(y, X, 0.3, 0.9, 3, VARIANTS["no V=V'"]) <= full + 1e-6 * abs(full)
    assert full <= _value(y, X, 0.3, 0.9, 3, VARIANTS["full+pairs"]) + 1e-6 * abs(full)


def test_the_joint_lift_is_exact_on_a_small_panel():
    """The finding on the ten-geo subset (#965), in miniature: at p = 5 the joint
    lift reaches the enumerated optimum of the piece to solver accuracy."""
    y, X = panel(0, p=5)
    v = _value(y, X, 0.3, 0.9, 3, VARIANTS["full"])
    assert v == pytest.approx(piece_optimum(y, X, 0.3, 0.9, 3), rel=1e-4)


def test_an_infeasible_piece_contributes_nothing():
    """lo > hi admits no slope: the hull and the lift both report +inf."""
    y, X = panel(1, p=4)
    assert ControlBound(y, X).piece(2.0, 1.0, 4) == np.inf
    assert Lift(y, X, "shor").piece(2.0, 1.0, 4) == np.inf


def test_the_rss_of_an_empty_aggregate_is_the_total_sum_of_squares():
    y, X = panel(2, p=3)
    X = X.copy(); X[:, 1] = -X[:, 0]
    assert rss_of(y, X, {0, 1}) == pytest.approx(float(y @ y))
