"""The cascade of certificates for one size (#965): reused multipliers, the Shor
relaxation, then the full one.

Every certificate in the cascade is the exact Lagrangian bound of slopefree.bound,
valid at any multipliers, so each tier is checked for two things: that it never
certifies a size holding a design above zero (against enumeration), and that a
tier which does not close hands the size to the next one."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from . import slopefree as sf
from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import (Multipliers, Rows, bound, certify_slope_free, design_value, dyadic_below,
                        objective, size_certificate)

SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    """Every node is bounded: on panels this small pricing would close every size."""
    monkeypatch.setattr(sf, "PRICE_LIMIT", 0)


def _exact(y, X):
    D = ExactData(y, X)
    return D, D.c, D.G


def _designs(p, m):
    for C in combinations(range(p), m):
        u = np.zeros(p, dtype=int); u[list(C)] = 1
        yield set(C), u


def _lifted(u):
    z = np.concatenate([[1], u]).astype(object)
    return np.array([[Q(int(a * b)) for b in z] for a in z], dtype=object)


def _ratio(c, G, C):
    u = np.zeros(len(c), dtype=object); u[:] = Q(0); u[list(C)] = Q(1)
    return (c @ u) ** 2 / (u @ G @ u)


def _rowset(R):
    key = lambda r: tuple(sorted(r.items()))  # noqa: E731
    return ({(key(r), b) for r, b in zip(R.eq, R.beq)}, {(key(r), b) for r, b in zip(R.ineq, R.bin)})


def _solves_by_level(monkeypatch):
    seen = []
    real = sf.solve

    def wrapped(c, G, lam, R):
        seen.append(R.level)
        return real(c, G, lam, R)

    monkeypatch.setattr(sf, "solve", wrapped)
    return seen


def _no_dual(monkeypatch):
    """The dual tier off, for the tests of the tiers after it."""
    monkeypatch.setattr(sf, "dual_multipliers", lambda *a, **k: (None, {"evals": 0}))


def _at_optimum(seed, p=6):
    y, X = panel(seed, p=p)
    D, c, G = _exact(y, X)
    _, C = positive_optimum(y, X)
    return c, G, _ratio(c, G, C), C


# ---------------------------------------------------------------- smoke


def test_smoke_every_size_closes_somewhere_in_the_cascade():
    c, G, lam, C = _at_optimum(0)
    for m in range(1, 7):
        res = size_certificate(c, G, lam, m, cuts=[C])
        assert res.certified and res.tier in ("dual", "shor", "full")
        assert res.bound <= 0 and res.multipliers is not None


def test_smoke_certify_records_its_tiers(monkeypatch):
    """One tier per bounded node; a one-design node (here the size with every geo)
    is evaluated, not bounded, so it has none."""
    y, X = panel(1, p=6)
    best, C = positive_optimum(y, X)
    real, calls = sf.size_certificate, []
    monkeypatch.setattr(sf, "size_certificate", lambda *a, **k: calls.append(1) or real(*a, **k))
    out = certify_slope_free(y, X, hull=False)
    assert out.certified and set(out.controls) == C
    assert sum(out.tiers.values()) == len(calls) > 0 and set(out.tiers) <= {"reuse", "dual", "shor", "full"}


# ---------------------------------------------------------------- the Shor rows


@pytest.mark.parametrize("seed", range(3))
def test_the_shor_rows_are_a_subset_of_the_full_rows(seed):
    """Dropping rows only enlarges the relaxation, so the Shor bound is at least the
    full one and holds every design."""
    y, X = panel(seed, p=5)
    _, c, G = _exact(y, X)
    for m in range(1, 6):
        full = Rows(c, m, cuts=[{0, 2}], fixes=[(1, 1)])
        shor = Rows(c, m, cuts=[{0, 2}], fixes=[(1, 1)], level="shor")
        fe, fi = _rowset(full); se, si = _rowset(shor)
        assert se <= fe and si <= fi
        assert shor.n_in < full.n_in and shor.n_eq == full.n_eq


def test_the_shor_rows_keep_the_size_and_the_slope_sign():
    """The trace bound needs diag U = u and sum u = m; c'u >= 0 and the cuts are
    what the Shor relaxation keeps of the inequalities."""
    y, X = panel(0, p=5)
    _, c, _ = _exact(y, X)
    R = Rows(c, 3, cuts=[{0, 1}], level="shor")
    assert R.n_in == 2 and R.c_rows == [0]
    assert R.beq[-1 - 0] == 0 and 3 in R.beq          # the size row and its products
    for C, u in _designs(5, 3):
        if c @ u.astype(object) >= 0:
            assert R.holds(_lifted(u)) == (C != {0, 1})


def test_an_unknown_level_is_refused():
    y, X = panel(0, p=4)
    _, c, _ = _exact(y, X)
    with pytest.raises(ValueError, match="level"):
        Rows(c, 2, level="lp")


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 5), size=st.floats(0.0, 2.0))
def test_property_the_shor_bound_holds_at_any_multipliers(seed, m, size):
    y, X = panel(seed % 40, p=5)
    _, c, G = _exact(y, X)
    lam = dyadic_below(Q(seed % 7, 10))
    R = Rows(c, m, level="shor")
    rng = np.random.default_rng(seed)
    A = objective(c, G, lam)
    b = bound(A, R, rng.normal(size=R.n_eq) * size, rng.normal(size=R.n_in) * size, m)
    best = max((design_value(A, u) for C, u in _designs(5, m) if c @ u.astype(object) >= 0), default=None)
    if best is not None:
        assert b >= best


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 5), f=st.fractions(Q(1, 2), Q(3, 2)))
@example(seed=0, m=1, f=Q(15243782109491573, 15200979701598040))   # the dual's start at F = 0 overflowed
def test_property_a_certified_size_holds_no_design_above_zero(seed, m, f):
    """Whatever tier closes it, a certified size has no design with u'Au > 0 other
    than the cut ones, here at ratios on both sides of the optimum's."""
    y, X = panel(seed % 40, p=5)
    _, c, G = _exact(y, X)
    _, C = positive_optimum(y, X)
    lam = dyadic_below(_ratio(c, G, C) * f)
    prev = size_certificate(c, G, lam, m - 1 if m > 1 else 1, cuts=[C])
    res = size_certificate(c, G, lam, m, cuts=[C], warm=prev.multipliers)
    A = objective(c, G, lam)
    above = [D for D, u in _designs(5, m) if D != C and c @ u.astype(object) >= 0 and design_value(A, u) > 0]
    if res.certified:
        assert not above, (res.tier, above)


# ---------------------------------------------------------------- the reuse tier


def test_a_sizes_own_multipliers_close_it_without_a_solve(monkeypatch):
    c, G, lam, C = _at_optimum(2)
    first = size_certificate(c, G, lam, 3, cuts=[C])
    assert first.certified
    monkeypatch.setattr(sf, "solve", lambda *a, **k: pytest.fail("a reused certificate solved"))
    again = size_certificate(c, G, lam, 3, cuts=[C], warm=first.multipliers)
    assert again.certified and again.tier == "reuse" and again.bound == first.bound


def test_reused_multipliers_are_checked_exactly(monkeypatch):
    """The float screen only decides whether to try the exact bound; a size the exact
    bound does not close goes on to the Shor relaxation."""
    c, G, lam, C = _at_optimum(2)
    first = size_certificate(c, G, lam, 3, cuts=[C])
    monkeypatch.setattr(sf, "_screen", lambda *a, **k: -1.0)
    real, calls = sf.bound, []

    def bound(*a, **k):
        calls.append(1)
        return Q(1) if len(calls) == 1 else real(*a, **k)

    monkeypatch.setattr(sf, "bound", bound)
    res = size_certificate(c, G, lam, 3, cuts=[C], warm=first.multipliers)
    assert res.certified and res.tier != "reuse" and len(calls) >= 2


def test_a_screen_above_zero_skips_the_exact_check(monkeypatch):
    """Multipliers the float screen rules out cost no exact bound: the size makes
    the same exact evaluations as one offered nothing."""
    c, G, lam, C = _at_optimum(2)
    first = size_certificate(c, G, lam, 3, cuts=[C])
    monkeypatch.setattr(sf, "_screen", lambda *a, **k: 1.0)
    real, calls = sf.bound, []
    monkeypatch.setattr(sf, "bound", lambda *a, **k: calls.append(1) or real(*a, **k))
    cold = size_certificate(c, G, lam, 3, cuts=[C]); n_cold = len(calls)
    res = size_certificate(c, G, lam, 3, cuts=[C], warm=first.multipliers)
    assert res.tier == cold.tier != "reuse" and len(calls) - n_cold == n_cold


def test_the_screen_agrees_with_the_exact_bound():
    c, G, lam, C = _at_optimum(1)
    res = size_certificate(c, G, lam, 2, cuts=[C])
    w = res.multipliers
    R = Rows(c, 2, cuts=[C], level=w.level)
    A = objective(c, G, dyadic_below(lam))
    exact = float(bound(A, R, w.y_eq, w.y_in, 2))
    assert sf._screen(A, R, w.y_eq, w.y_in, 2) == pytest.approx(exact, rel=1e-6, abs=1e-6 * abs(exact) + 1e-9)


def test_multipliers_of_another_shape_are_not_reused(monkeypatch):
    c, G, lam, C = _at_optimum(0)
    first = size_certificate(c, G, lam, 2, cuts=[C])
    short = Multipliers(first.multipliers.level, first.multipliers.y_eq[:-1], first.multipliers.y_in)
    res = size_certificate(c, G, lam, 2, cuts=[C], warm=short)
    assert res.certified and res.tier != "reuse"


# ---------------------------------------------------------------- the Shor tier


def test_a_size_the_shor_relaxation_closes_solves_nothing_more(monkeypatch):
    _no_dual(monkeypatch)
    c, G, lam, C = _at_optimum(0)
    tiers = {m: size_certificate(c, G, lam, m, cuts=[C], shor=True).tier for m in range(1, 7)}
    m = next(m for m, t in tiers.items() if t == "shor")
    seen = _solves_by_level(monkeypatch)
    res = size_certificate(c, G, lam, m, cuts=[C], shor=True)
    assert res.tier == "shor" and seen == ["shor"]


def test_a_shor_bound_above_zero_goes_on_to_the_full_relaxation(monkeypatch):
    _no_dual(monkeypatch)
    c, G, lam, C = _at_optimum(0)
    real = sf.bound
    monkeypatch.setattr(sf, "bound", lambda A, R, *a, **k: Q(1) if R.level == "shor" else real(A, R, *a, **k))
    seen = _solves_by_level(monkeypatch)
    res = size_certificate(c, G, lam, 2, cuts=[C], shor=True)
    assert res.certified and res.tier == "full" and seen[:2] == ["shor", "full"]


def test_a_shor_solver_failure_goes_on_to_the_full_relaxation(monkeypatch):
    """Clarabel can fail on the smaller relaxation (it did on a variant with the
    u_i c'u rows); the size is then the full relaxation's, not an error."""
    _no_dual(monkeypatch)
    import cvxpy as cp

    c, G, lam, C = _at_optimum(0)
    real = sf.solve

    def solve(c_, G_, lam_, R):
        if R.level == "shor":
            raise cp.error.SolverError("Solver 'CLARABEL' failed.")
        return real(c_, G_, lam_, R)

    monkeypatch.setattr(sf, "solve", solve)
    res = size_certificate(c, G, lam, 2, cuts=[C], shor=True)
    assert res.certified and res.tier == "full"


def test_a_full_solver_failure_is_reported():
    """The last tier hands a failure to the branch and bound: the size is reported
    open, with no bound, and is branched like any size that does not close
    (test_solver_failure.py)."""
    import cvxpy as cp

    c, G, lam, C = _at_optimum(0)
    with pytest.MonkeyPatch.context() as mp:
        _no_dual(mp)
        mp.setattr(sf, "solve", lambda *a, **k: (_ for _ in ()).throw(cp.error.SolverError("failed")))
        res = size_certificate(c, G, lam, 2, cuts=[C])
    assert not res.certified and res.bound is None and res.u is None


# ---------------------------------------------------------------- the search hands multipliers on


def test_each_root_size_is_offered_the_previous_sizes_multipliers(monkeypatch):
    y, X = panel(1, p=6)
    real, offered = sf.size_certificate, []

    def wrapped(c, G, lam, m, cuts=(), fixes=(), warm=None, **k):
        res = real(c, G, lam, m, cuts=cuts, fixes=fixes, warm=warm, **k)
        offered.append((m, list(fixes), warm, res))
        return res

    monkeypatch.setattr(sf, "size_certificate", wrapped)
    out = certify_slope_free(y, X, hull=False)
    assert out.certified
    roots = [r for r in offered if not r[1]]
    assert roots[0][2] is None
    for (_, _, _, before), (_, _, warm, _) in zip(roots, roots[1:]):
        if before.certified:
            assert warm is before.multipliers
    assert all(w is None for _, f, w, _ in offered if f)       # a branched node starts cold


# ---------------------------------------------------------------- the dual tier


@SETTINGS
@given(v=st.lists(st.floats(-50, 50), min_size=1, max_size=12), T=st.floats(0.5, 20))
def test_property_the_capped_projection_is_the_nearest_point(v, T):
    """proj onto {z >= 0, sum z <= T}: feasible, and no feasible point nearer v
    (checked against the KKT form z = max(v - theta, 0), theta >= 0, theta > 0 only
    when the sum is at T)."""
    v = np.array(v); z = sf._capped(v, T)
    assert (z >= 0).all() and z.sum() <= T * (1 + 1e-12) + 1e-12
    theta = 0.0 if np.maximum(v, 0).sum() <= T else None
    if theta is None:
        on = z > 0
        theta = float(np.mean(v[on] - z[on]))
        assert theta > 0 and abs(z.sum() - T) <= 1e-9 * max(1, T)
    assert np.allclose(z, np.maximum(v - theta, 0), atol=1e-9)


@pytest.mark.parametrize("v", [[1e17, 3.0, -1.0], [1e17, 1e17 - 64, 5.0], [1e300, 1.0]])
def test_the_capped_projection_survives_entries_far_above_the_cap(v):
    """At 1e17 and T = 2, u_1 - (u_1 - T) rounds to zero, so no index passes the
    sort-and-scan test in doubles; the projection is still feasible."""
    z = sf._capped(np.array(v), 2.0)
    assert np.isfinite(z).all() and (z >= 0).all() and z.sum() <= 2.0 * (1 + 1e-12)
    assert z[0] > 0


@SETTINGS
@given(seed=st.integers(0, 10_000), a=st.floats(1e-3, 10), T=st.floats(1, 10))
def test_property_the_smoothing_is_within_a_t_squared_over_two(seed, a, T):
    """g(S) = T max(0, lambda_max(S)) and its smoothing g_a satisfy
    g - a T^2 / 2 <= g_a <= g."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(6, 6)); S = (B + B.T) / 2
    lam = np.linalg.eigvalsh(S)
    g = T * max(0.0, lam[-1])
    z = sf._capped(lam / a, T)
    ga = float(z @ lam - a / 2 * (z @ z))
    assert g - a * T * T / 2 - 1e-9 <= ga <= g + 1e-9


def test_the_dual_gradient_matches_finite_differences():
    c, G, lam, C = _at_optimum(0)
    A = objective(c, G, dyadic_below(lam))
    R = Rows(c, 3, cuts=[C])
    D = sf._Dual(A, R, 3, c)
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(size=D.size)) * 1e-2
    f, g = D.smoothed(y, 1e-2)
    for j in rng.choice(D.size, 8, replace=False):
        e = np.zeros(D.size); e[j] = 1e-7
        fd = (D.smoothed(y + e, 1e-2)[0] - D.smoothed(y - e, 1e-2)[0]) / 2e-7
        assert fd == pytest.approx(g[j], rel=1e-4, abs=1e-6)


def test_smoke_the_dual_closes_a_size_without_a_solver(monkeypatch):
    c, G, lam, C = _at_optimum(0)
    monkeypatch.setattr(sf, "solve", lambda *a, **k: pytest.fail("the dual tier solved"))
    for m in (2, 3, 4):
        res = size_certificate(c, G, lam, m, cuts=[C])
        assert res.certified and res.tier == "dual" and res.bound <= 0
        assert res.multipliers.level == "full"


def test_the_dual_returns_nothing_it_cannot_close():
    """Below the optimum's ratio the optimum's size holds a design above zero, so no
    multipliers can close it: the dual gives up."""
    c, G, lam, C = _at_optimum(0)
    lam = dyadic_below(lam * Q(1, 2))
    A = objective(c, G, lam)
    R = Rows(c, len(C), cuts=[])
    w, info = sf.dual_multipliers(A, R, len(C), c)
    assert w is None and info["evals"] > 1


def test_the_dual_starts_from_closing_multipliers_and_stops():
    c, G, lam, C = _at_optimum(0)
    first = size_certificate(c, G, lam, 3, cuts=[C])
    A = objective(c, G, dyadic_below(lam))
    R = Rows(c, 3, cuts=[C])
    w, info = sf.dual_multipliers(A, R, 3, c, warm=first.multipliers)
    assert w is not None and info["evals"] == 1


def test_the_dual_multipliers_are_checked_exactly(monkeypatch):
    """Multipliers the dual hands over are certified only by the exact bound."""
    c, G, lam, C = _at_optimum(0)
    R = Rows(c, 2, cuts=[C])
    zeros = Multipliers("full", np.zeros(R.n_eq), np.zeros(R.n_in))
    monkeypatch.setattr(sf, "dual_multipliers", lambda *a, **k: (zeros, {"evals": 1}))
    res = size_certificate(c, G, lam, 2, cuts=[C])
    assert res.certified and res.tier in ("shor", "full")


def test_a_dual_that_gives_up_hands_the_size_on(monkeypatch):
    c, G, lam, C = _at_optimum(0)
    monkeypatch.setattr(sf, "dual_multipliers", lambda *a, **k: (None, {"evals": 5}))
    seen = _solves_by_level(monkeypatch)
    res = size_certificate(c, G, lam, 2, cuts=[C])
    assert res.certified and res.tier == "full" and seen[0] == "full"     # the Shor tier is off (SHOR_TIER)


def test_the_dual_reads_the_c_rows_at_the_solvers_scale():
    """The rows carrying c are divided by a power of two near max |c| before the
    dual sees them, and multiplied back after: without it the dual stalls above
    zero on GeoLift's groups (scratch spike, 6 of 6)."""
    c, G, lam, C = _at_optimum(0)
    A = objective(c, G, dyadic_below(lam))
    R = Rows(c, 3, cuts=[C])
    D = sf._Dual(A, R, 3, c)
    assert D.cs == sf._pow2(float(np.max(np.abs(np.asarray(c, dtype=float)))))
    y = np.zeros(D.size); y[D.c_idx] = 1.0
    eq, ineq = D.back(y)
    assert np.allclose(ineq[R.c_rows], D.s / D.cs)


def test_the_dual_gives_up_within_its_budget():
    c, G, lam, C = _at_optimum(0)
    lam = dyadic_below(lam * Q(1, 2))
    A = objective(c, G, lam)
    R = Rows(c, len(C), cuts=[])
    w, info = sf.dual_multipliers(A, R, len(C), c, budget=20)
    assert w is None and info["evals"] <= 21
