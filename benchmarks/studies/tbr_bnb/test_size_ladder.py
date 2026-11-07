"""The size certificates' schedule ladder and first-order triangle rounds (#965).

On the 100-geo, four-treated search the incumbent's own group, (4, 35, 58, 60),
took 2,321 s, 2,250 of them in nine sizes (19 to 27) that went to Clarabel: sizes
19, 20 and 27 closed by the full relaxation with no triangles, the others after one
or two rounds, at 144 to 392 s a size. Run on first-order multipliers alone, the
screen's longer schedules closed 19, 20 and 27 in 8 to 37 s, and rounds of triangles
separated from the dual's own smoothed point, each round restarted from zero, closed
21, 22 and 26 after one round and 23 and 24 after two; 25 was still open after three.
Warm-starting a round from the last one's multipliers left 23, 24 and 25 open after
six. So from SCREEN_LADDER_FROM candidate controls a size tries the screen's
schedules, then up to FIRST_ORDER_ROUNDS such rounds, before the Shor and full
relaxations."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel
from .certificate import ExactData
from .search import incumbent
from .slopefree import (FIRST_ORDER_ROUNDS, FIRST_ORDER_ROUNDS_FROM, SCREEN_LADDER_FROM, SCREEN_SCHEDULES, Rows, dual_multipliers,
                        dyadic_below, objective, size_certificate)
from .treatment import _split

#: The schedules and the triangle rounds forced on, whatever the group's size.
LADDER = dict(schedules=SCREEN_SCHEDULES, first_order_rounds=FIRST_ORDER_ROUNDS)

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _case(seed, T, C, n=20, p=30):
    """A treatment group of a simulated panel, at the ratio of the design C."""
    y, X, _ = _split(geo_panel(seed, n=n, p=p), T)
    D = ExactData(y, X)
    q = len(D.c)
    u = np.zeros(q, dtype=object); u[:] = Q(0); u[sorted(C)] = Q(1)
    cu = D.c @ u
    return D.c, D.G, dyadic_below(cu * cu / (u @ D.G @ u))


#: 26 candidate controls; at size 7 the dual gives up on every schedule and Clarabel
#: closes the full relaxation after one round of triangles; so does one first-order round.
NEEDS_TRIANGLES = dict(seed=2, T=(6, 11, 16, 18), C=[0, 6, 11, 15, 17, 19, 20, 25], m=7)
#: 26 candidate controls; at size 7 the first schedule gives up and a longer one closes.
NEEDS_LADDER = dict(seed=8, T=(4, 9, 19, 29), C=[2, 3, 4, 5, 9, 10, 20, 21, 25], m=7)


@pytest.fixture(scope="module")
def needs_triangles():
    k = NEEDS_TRIANGLES
    return (*_case(k["seed"], k["T"], k["C"]), k["m"], [set(k["C"])])


@pytest.fixture(scope="module")
def needs_ladder():
    k = NEEDS_LADDER
    return (*_case(k["seed"], k["T"], k["C"]), k["m"], [set(k["C"])])


def _spy(monkeypatch, fail=False):
    """Record each dual call's (schedule, warm given, triangle rows); ``fail`` makes
    every call give up, with the last smoothed point seen when it closed."""
    real = slopefree.dual_multipliers
    calls, points = [], []

    def dual(A, R, m, c, warm=None, **k):
        calls.append(((k.get("maxfun", 150), k.get("stages", 6), k.get("budget")), warm is not None, R.n_triangles))
        w, info = real(A, R, m, c, warm=warm, **k)
        if info["point"] is not None:
            points.append(info["point"])
        if fail:
            return None, dict(info, point=points[-1])
        return w, info

    monkeypatch.setattr(slopefree, "dual_multipliers", dual)
    return calls


def test_the_rows_count_their_triangles():
    c = np.array([Q(1), Q(2), Q(-1), Q(3)], dtype=object)
    assert Rows(c, 2).n_triangles == 0
    R = Rows(c, 2, triangles=[(0, (0, 1, 2)), (1, (0, 1, 3))])
    assert R.n_triangles == 2 and R.n_in == Rows(c, 2).n_in + 2


# ---------------------------------------------------------------- smoke


def test_smoke_first_order_triangle_rounds_close_what_the_dual_leaves(needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    before = size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES[:1])
    after = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    assert before.certified and before.tier == "full" and before.rounds == 1
    assert after.certified and after.tier == "triangles" and after.rounds == 1 and after.triangles > 0
    assert after.bound <= 0 and after.multipliers is None


# ---------------------------------------------------------------- unit


def test_a_longer_schedule_closes_a_size_the_first_gives_up(needs_ladder):
    c, G, lam, m, cuts = needs_ladder
    assert size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES[:1]).tier != "dual"
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    assert res.certified and res.tier == "dual" and res.rounds == 0


def test_the_schedules_run_in_turn_and_only_the_first_is_warm(monkeypatch, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    warm = size_certificate(c, G, lam, m - 1, cuts=cuts).multipliers
    calls = _spy(monkeypatch)
    size_certificate(c, G, lam, m, cuts=cuts, **LADDER, warm=warm)
    ladder = [cl for cl in calls if cl[2] == 0]
    assert [cl[0] for cl in ladder] == [SCREEN_SCHEDULES[0]] + list(SCREEN_SCHEDULES[1:])
    assert [cl[1] for cl in ladder] == [warm is not None] + [False] * (len(SCREEN_SCHEDULES) - 1)


def test_a_triangle_round_runs_on_the_last_schedule_with_the_separated_rows(monkeypatch, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    calls = _spy(monkeypatch)
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    rounds = [cl for cl in calls if cl[2] > 0]
    assert len(rounds) == res.rounds == 1 and rounds[0][2] == res.triangles
    assert rounds[0][0] == SCREEN_SCHEDULES[-1] and not rounds[0][1]


def test_at_most_the_round_limit_then_clarabel_decides(monkeypatch, needs_triangles):
    """Every round finds new triangles and none closes: FIRST_ORDER_ROUNDS rounds run."""
    c, G, lam, m, cuts = needs_triangles
    fresh = iter(combinations(range(len(c)), 3))
    real, n = slopefree.separate_triangles, []

    def separate(u, U, k=400, tol=1e-6):            # fresh triangles for the first-order rounds only
        n.append(1)
        return [(1.0, 1, next(fresh)) for _ in range(5)] if len(n) <= FIRST_ORDER_ROUNDS else real(u, U, k=k, tol=tol)

    monkeypatch.setattr(slopefree, "separate_triangles", separate)
    calls = _spy(monkeypatch, fail=True)
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    assert sum(1 for cl in calls if cl[2] > 0) == FIRST_ORDER_ROUNDS
    assert res.certified and res.tier == "full"


def test_no_violated_triangle_ends_the_rounds(monkeypatch, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    real = slopefree.separate_triangles
    seen = []

    def separate(u, U, k=400, tol=1e-6):
        seen.append(k)
        return [] if len(seen) == 1 else real(u, U, k=k, tol=tol)

    monkeypatch.setattr(slopefree, "separate_triangles", separate)
    calls = _spy(monkeypatch)
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    assert not any(cl[2] > 0 for cl in calls) and res.certified and res.tier == "full"


def test_a_round_whose_exact_bound_misses_goes_on(monkeypatch, needs_triangles):
    """Float multipliers that the exact bound does not confirm are not a certificate:
    the next round runs, and Clarabel when the rounds are spent."""
    c, G, lam, m, cuts = needs_triangles
    real = slopefree.bound
    monkeypatch.setattr(slopefree, "bound", lambda A, R, ye, yi, m_: Q(1) if R.n_triangles else real(A, R, ye, yi, m_))
    calls = _spy(monkeypatch)
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    assert res.tier == "full" and sum(1 for cl in calls if cl[2] > 0) >= 1


def test_the_dual_keeps_its_smoothed_point_when_it_gives_up(needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    A = objective(c, G, lam)
    R = Rows(c, m, cuts=cuts)
    w, info = dual_multipliers(A, R, m, c)
    Z = info["point"]
    assert w is None and Z.shape == (R.n, R.n) and Z[0, 0] == pytest.approx(1.0)
    assert np.allclose(Z, Z.T) and np.linalg.eigvalsh(Z)[0] >= -1e-9


def test_the_dual_reports_no_point_when_it_closes():
    c, G, lam = _case(0, (0, 4, 19, 23), [0, 9, 21, 22, 24])
    A = objective(c, G, lam * 2)
    R = Rows(c, 5)
    w, info = dual_multipliers(A, R, 5, c)
    assert w is not None and info["point"] is None


def test_the_default_schedules_follow_the_number_of_controls(monkeypatch, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    assert len(c) < SCREEN_LADDER_FROM
    calls = _spy(monkeypatch)
    size_certificate(c, G, lam, m, cuts=cuts)
    assert len([cl for cl in calls if cl[2] == 0]) == 1 and not any(cl[2] > 0 for cl in calls)
    calls.clear()
    monkeypatch.setattr(slopefree, "SCREEN_LADDER_FROM", 1)
    size_certificate(c, G, lam, m, cuts=cuts)
    assert len([cl for cl in calls if cl[2] == 0]) == len(SCREEN_SCHEDULES)


def test_the_branch_and_bound_certifies_with_the_ladder_of_its_size(monkeypatch):
    seen = []
    real = slopefree.size_certificate

    def size(*a, **k):
        seen.append(k.get("schedules", "default"))
        return real(*a, **k)

    monkeypatch.setattr(slopefree, "size_certificate", size)
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)            # every node is bounded
    y, X, _ = _split(geo_panel(0, n=10, p=12), (0, 1, 2, 3))
    out = slopefree.certify_slope_free(y, X, restarts=5)
    assert out.certified and seen and set(seen) == {"default"}


def test_the_default_rounds_follow_the_number_of_controls(monkeypatch, needs_triangles):
    """With the schedules given, the rounds still run only from FIRST_ORDER_ROUNDS_FROM
    controls unless asked for."""
    c, G, lam, m, cuts = needs_triangles
    assert len(c) < FIRST_ORDER_ROUNDS_FROM
    calls = _spy(monkeypatch)
    res = size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES)
    assert not any(cl[2] > 0 for cl in calls) and res.tier == "full"
    calls.clear()
    monkeypatch.setattr(slopefree, "FIRST_ORDER_ROUNDS_FROM", 1)
    res = size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES)
    assert any(cl[2] > 0 for cl in calls) and res.tier == "triangles"


def test_no_rounds_when_asked_for_none(monkeypatch, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    calls = _spy(monkeypatch)
    res = size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES, first_order_rounds=0)
    assert not any(cl[2] > 0 for cl in calls) and res.tier == "full"


def test_the_rounds_need_the_ladder(monkeypatch, needs_triangles):
    """One schedule is the size certificates' cascade of before: no rounds by default."""
    c, G, lam, m, cuts = needs_triangles
    monkeypatch.setattr(slopefree, "FIRST_ORDER_ROUNDS_FROM", 1)
    calls = _spy(monkeypatch)
    size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES[:1])
    assert not any(cl[2] > 0 for cl in calls)


# ---------------------------------------------------------------- properties


def _best_of_size(c, G, m, excluded):
    """max (c'u)^2 / u'Gu over the designs of size m with c'u > 0, other than
    ``excluded``, in doubles over every design; None when there is none."""
    p = len(c)
    cf = np.array([float(v) for v in c]); Gf = np.array([[float(v) for v in r] for r in G])
    best = None
    for C in combinations(range(p), m):
        if set(C) in excluded:
            continue
        idx = list(C)
        cu = cf[idx].sum()
        if cu > 0:
            r = cu * cu / Gf[np.ix_(idx, idx)].sum()
            best = r if best is None or r > best else best
    return best


def _geo_size(seed, f, dm, P=20, n=12):
    """A size of a 16-control group of a simulated panel near its incumbent's size,
    at the incumbent's ratio times f, and the incumbent as the cut."""
    rng = np.random.default_rng(seed)
    T = tuple(sorted(rng.choice(P, 4, replace=False).tolist()))
    y, X, _ = _split(geo_panel(seed % 40, n=n, p=P), T)
    D = ExactData(y, X)
    C = incumbent(y, X, restarts=3, seed=0)[1]
    u = np.zeros(len(D.c), dtype=object); u[:] = Q(0); u[sorted(C)] = Q(1)
    cu = D.c @ u
    if not cu > 0:
        return None
    return D.c, D.G, cu * cu / (u @ D.G @ u) * f, min(max(1, len(C) + dm), len(D.c)), [set(C)]


def test_a_size_closed_by_a_triangle_round_holds_no_better_design():
    """Seed 12 at size 8: the first-order round closes it, and no design of size 8 has
    a larger ratio (enumerated)."""
    c, G, lam, m, cuts = _geo_size(12, Q(1), -1)
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    assert m == 8 and res.certified and res.tier == "triangles"
    best = _best_of_size(c, G, m, cuts)
    assert best is None or best <= float(dyadic_below(lam)) * (1 + 1e-9)


@SETTINGS
@given(seed=st.integers(0, 10_000), f=st.sampled_from([Q(99, 100), Q(1), Q(1001, 1000)]), dm=st.integers(-1, 1))
def test_property_a_certified_size_holds_no_better_design(seed, f, dm):
    """16 candidate controls, the ladder and the rounds forced on, every design of the
    size enumerated; the ladder closes whatever the first schedule closes."""
    case = _geo_size(seed, f, dm)
    if case is None:
        return
    c, G, lam, m, cuts = case
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    first = size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES[:1])
    if res.certified:
        best = _best_of_size(c, G, m, cuts)
        assert best is None or best <= float(dyadic_below(lam)) * (1 + 1e-9)
    if first.certified and first.tier == "dual":
        assert res.certified and res.tier == "dual"


# ---------------------------------------------------------------- edges


def test_a_node_with_fixings_holds_no_better_design_when_certified():
    """Seed 12 at size 8 with one geo of the incumbent fixed in and another out:
    certified only when no design of size 8 with those fixings has a larger ratio."""
    c, G, lam, m, cuts = _geo_size(12, Q(1), -1)
    C = sorted(cuts[0])
    out = next(j for j in range(len(c)) if j not in cuts[0])
    fixes = [(C[0], 1), (out, 0)]
    res = size_certificate(c, G, lam, m, cuts=cuts, fixes=fixes, **LADDER)
    designs = [D for D in combinations(range(len(c)), m) if C[0] in D and out not in D and set(D) != cuts[0]]
    cf = np.array([float(v) for v in c]); Gf = np.array([[float(v) for v in r] for r in G])
    best = max((cf[list(D)].sum() ** 2 / Gf[np.ix_(D, D)].sum() for D in designs if cf[list(D)].sum() > 0), default=None)
    assert res.certified
    assert best is None or best <= float(dyadic_below(lam)) * (1 + 1e-9)


def test_a_point_that_offers_only_known_triangles_ends_the_rounds(monkeypatch, needs_triangles):
    """A round whose smoothed point separates only triangles already in the rows adds
    nothing: the rounds stop there, not at the limit."""
    c, G, lam, m, cuts = needs_triangles
    calls = _spy(monkeypatch, fail=True)            # each round's point is the one before it
    res = size_certificate(c, G, lam, m, cuts=cuts, **LADDER)
    assert sum(1 for cl in calls if cl[2] > 0) == 1 < FIRST_ORDER_ROUNDS and res.tier == "full"


def test_no_schedule_goes_straight_to_the_relaxations(monkeypatch, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    calls = _spy(monkeypatch)
    res = size_certificate(c, G, lam, m, cuts=cuts, schedules=())
    assert calls == [] and res.certified and res.tier == "full"


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("schedules", [((0, 6, 400),), ((150, 6),), [(150, 6, 400)]])
def test_a_malformed_schedule_is_refused(schedules, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    with pytest.raises(MlsynthConfigError, match="schedule"):
        size_certificate(c, G, lam, m, cuts=cuts, schedules=schedules)


@pytest.mark.parametrize("rounds", [-1, 1.0, "3", True])
def test_a_malformed_round_count_is_refused(rounds, needs_triangles):
    c, G, lam, m, cuts = needs_triangles
    with pytest.raises(MlsynthConfigError, match="first_order_rounds"):
        size_certificate(c, G, lam, m, cuts=cuts, schedules=SCREEN_SCHEDULES, first_order_rounds=rounds)
