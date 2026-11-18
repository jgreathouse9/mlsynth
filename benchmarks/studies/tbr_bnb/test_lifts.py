"""Every lift level is a relaxation, the nested ones are ordered, and an infeasible
piece contributes nothing.

A piece's value is a lower bound by weak duality, read from the solve's multipliers
(#1087): the solver's primal value is not a bound, and on the full lift's face Zv = 0,
which has no strictly feasible point, Clarabel stops with it 3.75e-4 above the
optimum. The bound holds for any multipliers, so the tests below hold it to float
precision, perturb the multipliers, and check its tightness only where the solve
reports optimal."""
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


#: The bound against the enumerated optimum: weak duality is exact arithmetic up to
#: the float sums that evaluate it, so the slack is float precision, not solver accuracy.
FLOAT = 1e-9


def _lift(y, X, kw):
    kw = dict(kw)
    return Lift(y, X, kw.pop("level"), **kw)


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.05, 1.0), width=st.floats(1.1, 3.0),
       m=st.integers(1, 5), name=st.sampled_from(sorted(VARIANTS)))
@example(seed=137, lo=1.0, width=2.0, m=1, name="full+pairs")   # the primal value was 5.7e-5 above
@example(seed=2117, lo=0.9375, width=1.25, m=3, name="full")    # #1087: the primal value was 3.75e-4 above
def test_every_level_is_a_relaxation(seed, lo, width, m, name):
    y, X = panel(seed, p=5)
    hi = lo * width
    v = _value(y, X, lo, hi, m, VARIANTS[name])
    opt = piece_optimum(y, X, lo, hi, m)
    assert v <= opt + FLOAT * abs(opt)


def test_the_solve_is_recorded_beside_the_bound():
    """#1087's case: the solve stops inaccurate above the optimum; the piece reports
    the bound, and keeps the status and the primal value for the caller to read."""
    y, X = panel(2117, p=5)
    L = Lift(y, X, "full")
    v = L.piece(0.9375, 0.9375 * 1.25, 3)
    opt = piece_optimum(y, X, 0.9375, 0.9375 * 1.25, 3)
    assert L.status == L.prob.status == "optimal_inaccurate"
    assert L.value == pytest.approx(float(L.prob.value) * L.scale ** 2, rel=1e-12)
    assert L.value > opt * (1 + 2e-4)          # the primal value is not a bound here
    assert v == L.bound() <= opt
    assert v > opt * (1 - 1e-3)                # and the bound is still close


def _perturb(d, rng, size):
    if d is None:
        return None
    if isinstance(d, (list, tuple)):
        return [_perturb(e, rng, size) for e in d]
    d = np.asarray(d, float)
    return d * rng.uniform(0.0, 2.0, d.shape) + rng.normal(0.0, size, d.shape)


@SETTINGS
@given(seed=st.integers(0, 10_000), noise=st.integers(0, 2 ** 32 - 1),
       size=st.sampled_from([1e-3, 1e-1, 1.0, 10.0]), m=st.integers(1, 5), name=st.sampled_from(sorted(VARIANTS)))
def test_any_multipliers_give_a_bound(seed, noise, size, m, name):
    """Weak duality does not need the solver's multipliers to be right: wrong signs,
    wrong scales and points outside the cones are clipped or projected, and the bound
    stays below the optimum."""
    y, X = panel(seed, p=5)
    L = _lift(y, X, VARIANTS[name])
    L.piece(0.3, 0.9, m)
    rng = np.random.default_rng(noise)
    duals = [_perturb(c.dual_value, rng, size) for c in L.prob.constraints]
    opt = piece_optimum(y, X, 0.3, 0.9, m)
    assert L.bound(duals) <= opt + FLOAT * abs(opt)


@pytest.mark.parametrize("name", sorted(set(VARIANTS) - {"no V=V'"}))   # inaccurate on every case tried;
@pytest.mark.parametrize("seed, lo, width, m", [(752, 0.8226, 2.834, 1), (485, 0.841, 2.595, 1),
                                                (6920, 0.7454, 2.267, 1), (0, 0.3, 3.0, 3)])
def test_the_bound_meets_an_accurate_solve(name, seed, lo, width, m):
    """Where the solver reports optimal, its multipliers are good and the bound sits
    at the primal value. The m = 1 cases are where the perspective rows' tangent at
    u_j = 0 lost up to 2.4x; written as cones, their multipliers are exact."""
    y, X = panel(seed, p=5)
    L = _lift(y, X, VARIANTS[name])
    v = L.piece(lo, lo * width, m)
    if L.status != "optimal":
        pytest.skip(f"the solve reported {L.status}")
    assert v == pytest.approx(L.value, rel=1e-4)


def test_the_trace_bound_holds_at_every_design():
    """The bound minimizes over trace(Z) <= T, so T must cover the lifted point of every
    real design: with each slope at hi, trace = 1 + m + m hi^2 for the full lift
    (1 + m hi^2 without the u block). Every variant has a finite T."""
    y, X = panel(1, p=5)
    for name in sorted(VARIANTS):
        L = _lift(y, X, VARIANTS[name])
        L.piece(0.4, 1.7, 3)
        tr = 1 + 3 * 1.7 ** 2 + (3 if VARIANTS[name]["level"] == "full" else 0)
        assert tr * (1 - 1e-12) <= L.trace_bound() < np.inf
    L = _lift(y, X, VARIANTS["full"])
    L.piece(0.4, 1.7, 3)
    assert L.trace_bound() == pytest.approx(1 + 3 + 3 * 1.7 ** 2, rel=1e-12)   # tight at hi


@pytest.mark.parametrize("name", sorted(VARIANTS))
@pytest.mark.parametrize("seed", range(2))
def test_the_trace_bound_covers_the_relaxation(name, seed):
    """T must hold on the relaxation, not only at real designs: the largest trace over
    each lift's feasible set stays at or below it."""
    import cvxpy as cp
    y, X = panel(seed, p=5)
    L = _lift(y, X, VARIANTS[name])
    L.piece(0.4, 1.7, 3)
    top = cp.Problem(cp.Maximize(cp.trace(L._Z)), L.prob.constraints)
    top.solve(solver=cp.CLARABEL)
    assert top.status in ("optimal", "optimal_inaccurate")
    assert top.value <= L.trace_bound() * (1 + 1e-6)


def test_a_full_lift_without_the_slope_diagonal_has_no_trace_bound():
    """Without the (w - lo u)(hi u - w) rows nothing caps diag W, so T is infinite and
    the bound is -inf unless the Lagrangian's matrix is PSD; it is still a bound."""
    y, X = panel(1, p=5)
    L = Lift(y, X, "full", skip={("a", "c"), ("c", "a")})
    v = L.piece(0.4, 1.7, 3)
    assert L.trace_bound() == np.inf
    assert v <= piece_optimum(y, X, 0.4, 1.7, 3)


def test_missing_multipliers_give_no_bound():
    y, X = panel(1, p=5)
    L = Lift(y, X, "shor")
    L.piece(0.3, 0.9, 3)
    duals = [c.dual_value for c in L.prob.constraints]
    duals[3] = None
    assert L.bound(duals) == -np.inf


def test_multipliers_must_match_the_constraints():
    y, X = panel(1, p=5)
    L = Lift(y, X, "shor")
    L.piece(0.3, 0.9, 3)
    with pytest.raises(ValueError, match="one multiplier per constraint"):
        L.bound([None])


@pytest.mark.parametrize("seed", range(3))
def test_the_nested_levels_are_ordered(seed):
    """Each level adds constraints to the one before it, so its optimum can only rise;
    the bounds track the optima to the solver's accuracy."""
    y, X = panel(seed, p=5)
    vals = [_value(y, X, 0.3, 0.9, 3, VARIANTS[k]) for k in ("shor", "persp", "rlt")]
    tol = 1e-6 * abs(vals[-1])
    assert vals[0] <= vals[1] + tol and vals[1] <= vals[2] + tol
    full = _value(y, X, 0.3, 0.9, 3, VARIANTS["full"])
    assert _value(y, X, 0.3, 0.9, 3, VARIANTS["no V=V'"]) <= full + 1e-6 * abs(full)
    assert full <= _value(y, X, 0.3, 0.9, 3, VARIANTS["full+pairs"]) + 1e-6 * abs(full)


def test_the_joint_lift_is_exact_on_a_small_panel():
    """The finding on the ten-geo subset (#965), in miniature: at p = 5 the joint
    lift's bound reaches the enumerated optimum of the piece to solver accuracy."""
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
