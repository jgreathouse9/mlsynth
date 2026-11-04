"""Upper bounds in the simplex QP: ``min ||A - Bw||^2 + l'w`` over the capped
simplex ``{0 <= w <= u, sum(w) = 1}``.

The capped simplex is the hull of the shared-slope design problem at a fixed
total (#965, roadmap step 5): with ``w = s v``, ``min ||y - X w||^2`` over
``0 <= w <= hi, sum(w) = s`` is ``solve_simplex_qp(s X, y, upper=hi / s)``.

The reference is exact and independent of any solver: on a few donors every
assignment of the weights to "at zero", "at the cap" and "free" is tried, the
free set's equality-constrained least squares solved, and the best point that is
feasible kept. A convex QP's minimum is attained at one of those points.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.solvers.active_set import simplex_kkt_error, solve_simplex_qp

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _obj(B, A, w, lin=None):
    r = A - B @ w
    return float(r @ r + (0.0 if lin is None else lin @ w))


def _brute(B, A, u, lin=None):
    """The exact minimiser over the capped simplex by enumerating faces."""
    J = B.shape[1]
    best, arg = np.inf, None
    for roles in product((0, 1, 2), repeat=J):          # 0 at zero, 1 at cap, 2 free
        w = np.zeros(J)
        cap = [j for j in range(J) if roles[j] == 1]
        free = [j for j in range(J) if roles[j] == 2]
        w[cap] = u[cap]
        t = 1.0 - w.sum()
        if not free:
            if abs(t) > 1e-12:
                continue
        else:
            BF = B[:, free]
            target = A - B @ w
            # min ||target - BF wF||^2 + lin_F' wF s.t. 1'wF = t, by its KKT system
            k = len(free)
            K = np.zeros((k + 1, k + 1))
            K[:k, :k] = 2 * BF.T @ BF
            K[:k, k] = K[k, :k] = 1.0
            rhs = np.r_[2 * BF.T @ target - (0 if lin is None else lin[free]), t]
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            w[free] = sol[:k]
        if w.min() < -1e-10 or np.any(w > u + 1e-10):
            continue
        v = _obj(B, A, w, lin)
        if v < best:
            best, arg = v, w
    return best, arg


def _case(seed, J=5, T=8, tight=True):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, J))
    k = min(2, J)
    A = B[:, :k] @ np.array([0.7, 0.3])[:k] + 0.3 * rng.normal(size=T)
    lo = 1.0 / J + 0.05
    u = rng.uniform(lo, max(lo + 0.1, 0.6 if tight else 2.0), size=J)
    return B, A, u


# ---------------------------------------------------------------- smoke
def test_smoke_a_capped_solve_is_feasible_and_certified():
    B, A, u = _case(0)
    w, info = solve_simplex_qp(B, A, upper=u, return_info=True)
    assert w.shape == (5,) and abs(w.sum() - 1) < 1e-12
    assert w.min() >= 0 and np.all(w <= u + 1e-12)
    assert info["converged"] and info["certified"]


# ---------------------------------------------------------------- unit
@pytest.mark.parametrize("seed", range(8))
def test_the_capped_solve_reaches_the_exact_minimum(seed):
    B, A, u = _case(seed)
    w = solve_simplex_qp(B, A, upper=u)
    best, _ = _brute(B, A, u)
    assert _obj(B, A, w) == pytest.approx(best, rel=1e-10, abs=1e-12)


def test_a_binding_cap_is_held_at_the_cap():
    """A donor that fits the target alone wants all the weight; its cap holds it."""
    rng = np.random.default_rng(1)
    B = rng.normal(size=(10, 4)); A = B[:, 0].copy()
    u = np.array([0.4, 1.0, 1.0, 1.0])
    w = solve_simplex_qp(B, A, upper=u)
    assert w[0] == pytest.approx(0.4, abs=1e-12)


def test_caps_that_do_not_bind_change_no_bit():
    """Caps of one or more cannot bind on the simplex: the answer is today's."""
    for seed in range(5):
        B, A, _ = _case(seed)
        plain = solve_simplex_qp(B, A)
        for u in (np.ones(5), np.full(5, 3.0), 1.0):
            np.testing.assert_array_equal(solve_simplex_qp(B, A, upper=u), plain)


def test_a_scalar_cap_is_every_donors_cap():
    B, A, _ = _case(2)
    np.testing.assert_array_equal(solve_simplex_qp(B, A, upper=0.35),
                                  solve_simplex_qp(B, A, upper=np.full(5, 0.35)))


def test_the_linear_term_and_the_caps_together():
    for seed in range(5):
        B, A, u = _case(seed)
        lin = np.random.default_rng(seed + 100).uniform(0, 1, size=5)
        w = solve_simplex_qp(B, A, upper=u, linear=lin)
        best, _ = _brute(B, A, u, lin)
        assert _obj(B, A, w, lin) == pytest.approx(best, rel=1e-9, abs=1e-12)


def test_the_hull_of_the_shared_slope_problem():
    """The use that motivates the caps: min ||y - Xw||^2 over 0 <= w <= hi,
    sum(w) = s, as a capped simplex in v = w / s."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 6)) * 40; y = X[:, :3].sum(1) * 0.5 + rng.normal(size=30)
    hi, s = 0.8, 2.0
    v = solve_simplex_qp(s * X, y, upper=hi / s)
    best, _ = _brute(s * X, y, np.full(6, hi / s))
    assert _obj(s * X, y, v) == pytest.approx(best, rel=1e-10)
    assert np.all(s * v <= hi + 1e-12)


def test_the_certificate_reads_the_caps():
    B, A, u = _case(4)
    w = solve_simplex_qp(B, A, upper=u)
    assert simplex_kkt_error(B, A, w, upper=u) <= 1e-10
    # the plain certificate does not hold at a capped optimum whose cap binds
    if np.any(w >= u - 1e-12):
        assert simplex_kkt_error(B, A, w) > 1e-10


def test_the_certificate_refuses_a_point_moved_along_the_face():
    B, A, u = _case(5)
    w = solve_simplex_qp(B, A, upper=u)
    free = np.where((w > 1e-6) & (w < u - 1e-6))[0]
    if free.size >= 2:
        d = np.zeros(5); d[free[0]], d[free[1]] = 1e-3, -1e-3
        assert simplex_kkt_error(B, A, w + d, upper=u) > 1e-10


def test_a_warm_start_outside_the_caps_is_ignored():
    B, A, u = _case(6)
    bad = np.zeros(5); bad[0] = 1.0                       # sums to one, above u[0] < 1
    w = solve_simplex_qp(B, A, upper=u, warm_start=bad)
    np.testing.assert_allclose(w, solve_simplex_qp(B, A, upper=u), atol=1e-12)


def test_a_wide_capped_pool_seeds_itself_feasibly():
    """Past the acceleration threshold the solver seeds itself; the seed must
    respect the caps or be discarded, and the answer must not change."""
    rng = np.random.default_rng(7)
    B = rng.normal(size=(12, 40)); A = B[:, :3].mean(1) + 0.1 * rng.normal(size=12)
    u = np.full(40, 0.2)
    w1 = solve_simplex_qp(B, A, upper=u)
    w2 = solve_simplex_qp(B, A, upper=u, accelerate=False)
    assert np.all(w1 <= u + 1e-12)
    assert _obj(B, A, w1) == pytest.approx(_obj(B, A, w2), rel=1e-10)


# ---------------------------------------------------------------- properties
@SETTINGS
@given(seed=st.integers(0, 100_000), J=st.integers(2, 6), tight=st.booleans())
def test_property_the_capped_solve_is_the_exact_minimum(seed, J, tight):
    B, A, u = _case(seed, J=J, tight=tight)
    w, info = solve_simplex_qp(B, A, upper=u, return_info=True)
    best, _ = _brute(B, A, u)
    assert w.min() >= 0 and np.all(w <= u + 1e-12) and abs(w.sum() - 1) < 1e-12
    assert _obj(B, A, w) <= best * (1 + 1e-10) + 1e-12
    assert info["certified"]


@SETTINGS
@given(seed=st.integers(0, 100_000), J=st.integers(2, 6), k=st.integers(-20, 20))
def test_property_a_power_of_two_scale_changes_no_bit(seed, J, k):
    B, A, u = _case(seed, J=J)
    c = 2.0 ** k
    np.testing.assert_array_equal(solve_simplex_qp(c * B, c * A, upper=u), solve_simplex_qp(B, A, upper=u))


@SETTINGS
@given(seed=st.integers(0, 100_000), J=st.integers(2, 6))
def test_property_tightening_a_cap_never_lowers_the_minimum(seed, J):
    B, A, u = _case(seed, J=J)
    v1 = _obj(B, A, solve_simplex_qp(B, A, upper=u))
    u2 = u.copy(); u2[0] = max(1.0 / J + 0.01, u[0] * 0.8)
    if u2.sum() >= 1:
        assert _obj(B, A, solve_simplex_qp(B, A, upper=u2)) >= v1 * (1 - 1e-10) - 1e-12


# ---------------------------------------------------------------- edge and failure
def test_caps_that_sum_to_one_leave_one_point():
    B, A, _ = _case(8, J=4)
    u = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(solve_simplex_qp(B, A, upper=u), u, atol=1e-12)


def test_one_donor_with_room_takes_all_the_weight():
    B, A, _ = _case(9, J=1)
    np.testing.assert_array_equal(solve_simplex_qp(B, A, upper=1.0), [1.0])


@pytest.mark.parametrize("u, match", [
    (np.array([0.1, 0.1, 0.1, 0.1, 0.1]), "sum to at least one"),
    (np.array([0.5, 0.5, -0.1, 0.5, 0.5]), "nonnegative"),
    (np.array([0.5, 0.5, np.nan, 0.5, 0.5]), "finite"),
    (np.array([0.5, 0.5, 0.5]), "one cap per donor"),
])
def test_invalid_caps_are_refused(u, match):
    B, A, _ = _case(10)
    with pytest.raises(ValueError, match=match):
        solve_simplex_qp(B, A, upper=u)


def test_one_donor_whose_cap_is_below_one_is_infeasible():
    B, A, _ = _case(11, J=1)
    with pytest.raises(ValueError, match="sum to at least one"):
        solve_simplex_qp(B, A, upper=0.5)


def test_a_feasible_capped_warm_start_is_used_and_changes_nothing():
    B, A, u = _case(12)
    w0, info0 = solve_simplex_qp(B, A, upper=u, return_info=True)
    w1, info1 = solve_simplex_qp(B, A, upper=u, warm_start=w0, return_info=True)
    assert _obj(B, A, w1) == pytest.approx(_obj(B, A, w0), rel=1e-12)
    assert info1["pivots"] <= info0["pivots"]


@pytest.mark.parametrize("seed", range(6))
def test_a_wide_design_with_a_linear_term_meets_the_caps_along_its_ray(seed):
    """More donors than rows makes the free block rank-deficient, and a linear term
    along its null space is a ray the loop follows until a weight meets zero or
    its cap."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(3, 6)); A = rng.normal(size=3)
    lin = rng.uniform(-1, 1, size=6)
    u = rng.uniform(0.25, 0.5, size=6)
    w = solve_simplex_qp(B, A, upper=u, linear=lin)
    best, _ = _brute(B, A, u, lin)
    assert np.all(w <= u + 1e-12) and w.min() >= 0
    assert _obj(B, A, w, lin) == pytest.approx(best, rel=1e-9, abs=1e-12)


def test_the_certificate_refuses_a_weight_held_at_its_cap_that_should_come_down():
    """Hold the worst-fitting donor at its cap and fit the rest optimally: the
    zero side and the free set are then optimal, and only the cap side is not."""
    rng = np.random.default_rng(13)
    B = rng.normal(size=(12, 5)); A = B[:, 1:3].mean(1) + 0.05 * rng.normal(size=12)
    u = np.full(5, 0.5)
    j = 0                                                     # A is built without donor 0
    held = 0.3; u_held = u.copy(); u_held[j] = held
    rest = [k for k in range(5) if k != j]
    v = solve_simplex_qp(B[:, rest] * (1 - held), A - held * B[:, j], upper=u[rest] / (1 - held))
    w = np.zeros(5); w[j] = held; w[rest] = v * (1 - held)
    opt = solve_simplex_qp(B, A, upper=u_held)
    assert _obj(B, A, opt) < _obj(B, A, w) - 1e-6               # the held weight should come down
    assert simplex_kkt_error(B, A, w, upper=u_held) > 1e-10


def test_unequal_loose_caps_change_no_bit():
    """Caps of one or more that differ from donor to donor still cannot bind."""
    for seed in range(5):
        B, A, _ = _case(seed)
        np.testing.assert_array_equal(solve_simplex_qp(B, A, upper=np.array([1.0, 2.0, 5.0, 1.5, 3.0])),
                                      solve_simplex_qp(B, A))


@SETTINGS
@given(seed=st.integers(0, 100_000), J=st.integers(2, 6))
def test_property_caps_below_the_uniform_weight(seed, J):
    """Some caps below 1 / J, so the uniform point is not in the capped simplex."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(8, J)); A = B[:, :1].sum(1) + 0.3 * rng.normal(size=8)
    u = rng.uniform(0.02, 1.0, size=J)
    u[0] = 0.5 / J                                            # below the uniform weight
    if u.sum() < 1:
        u[1:] *= (1.0 - u[0]) / u[1:].sum() * 1.01
    w = solve_simplex_qp(B, A, upper=u)
    best, _ = _brute(B, A, u)
    assert w.min() >= 0 and np.all(w <= u + 1e-12) and abs(w.sum() - 1) < 1e-12
    assert _obj(B, A, w) <= best * (1 + 1e-10) + 1e-12
