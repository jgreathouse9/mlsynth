"""The deep descent in certify_slope_free (#965 item 6).

A group whose screen does not close is searched at its incumbent's ratio, and every
size's branch and bound prunes against it. When ``restarts`` descents miss the
group's optimum the search runs at a weaker ratio until rounding finds it: on the
certified design's own group at 100 geos, k = 4, 1,959 of 2,057 s. A second descent
with ``deep_restarts`` restarts, run only on a group that reaches the hull stage,
gives the search the stronger ratio from the start.

Panel 1 at eight controls is the fixture: one restart reaches an RSS of 5,853.9, the
enumerated optimum is 302.3, and 200 restarts reach it."""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import search, slopefree
from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import _ratio, certify_slope_free

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _fixture():
    y, X = panel(1, p=8)
    best, C = positive_optimum(y, X)
    return y, X, best, C


@pytest.fixture
def calls(monkeypatch):
    """The ``restarts`` of every RSS descent certify_slope_free runs."""
    seen = []
    real = search.incumbent

    def spy(y, X, *, restarts, seed):
        seen.append(restarts)
        return real(y, X, restarts=restarts, seed=seed)

    monkeypatch.setattr(search, "incumbent", spy)
    return seen


def _collect():
    seen = []
    return seen, dict(progress=seen.append, verbose=1)


def test_the_fixture_needs_the_deep_descent():
    y, X, best, C = _fixture()
    shallow, _ = search.incumbent(y, X, restarts=1, seed=0)
    deep, D = search.incumbent(y, X, restarts=200, seed=0)
    assert shallow > 10 * best and D == C and deep == pytest.approx(best, rel=1e-9)


# ---------------------------------------------------------------- smoke


def test_smoke_the_deep_descent_returns_the_enumerated_optimum():
    y, X, best, C = _fixture()
    out = certify_slope_free(y, X, restarts=1, deep_restarts=200)
    assert out.certified and set(out.controls) == C
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


# ---------------------------------------------------------------- unit


def test_the_deep_descent_runs_once_after_the_first(calls):
    y, X, _, _ = _fixture()
    certify_slope_free(y, X, restarts=1, deep_restarts=200)
    assert calls == [1, 200]


def test_the_default_deep_descent_has_500_restarts(calls):
    y, X, _, _ = _fixture()
    certify_slope_free(y, X, restarts=1)
    assert calls == [1, 500]


def test_the_default_is_read_at_the_call(calls, monkeypatch):
    """None takes DEEP_RESTARTS as it is when certify_slope_free is called."""
    y, X, _, _ = _fixture()
    monkeypatch.setattr(slopefree, "DEEP_RESTARTS", 7)
    certify_slope_free(y, X, restarts=1, deep_restarts=None)
    monkeypatch.setattr(slopefree, "DEEP_RESTARTS", 0)
    certify_slope_free(y, X, restarts=1)
    assert calls == [1, 7, 1]


def test_a_bad_default_is_refused(monkeypatch):
    y, X, _, _ = _fixture()
    monkeypatch.setattr(slopefree, "DEEP_RESTARTS", -3)
    with pytest.raises(MlsynthConfigError, match="-3"):
        certify_slope_free(y, X, restarts=1)


def test_the_beds_run_without_it(calls):
    """The beds measure the branch and bound from their five-restart incumbent."""
    from .beds import BED_DEEP_RESTARTS, BED_RESTARTS, run_group
    assert BED_DEEP_RESTARTS == 0
    run_group("s15_k3", 0)
    assert calls == [BED_RESTARTS]
    calls.clear()
    run_group("s15_k3", 0, deep_restarts=50)
    assert calls == [BED_RESTARTS, 50]


def test_every_size_is_searched_at_the_deep_designs_ratio(monkeypatch):
    """The branch and bound starts from the deep design: the incumbent every size is
    first tested at is the optimum's ratio, not the one-restart design's."""
    y, X, _, C = _fixture()
    D = ExactData(y, X)
    first = []
    real = slopefree._branch

    def spy(c, G, out, m, *a):
        inc = a[-2]
        first.append((m, inc["lam"], set(inc["C"])))
        return real(c, G, out, m, *a)

    monkeypatch.setattr(slopefree, "_branch", spy)
    certify_slope_free(y, X, restarts=1, deep_restarts=200)
    lam = _ratio(D.c, D.G, C)[1]
    seen = {}
    for m, lam_m, C_m in first:
        seen.setdefault(m, (lam_m, C_m))
    assert seen and all(v == (lam, C) for v in seen.values())


def test_the_deep_design_is_reported_as_a_descent_incumbent():
    y, X, _, C = _fixture()
    D = ExactData(y, X)
    seen, kw = _collect()
    certify_slope_free(y, X, restarts=1, deep_restarts=200, **kw)
    found = [e for e in seen if e.kind == "incumbent" and e.fields["source"] == "descent"]
    assert len(found) == 1
    assert found[0].fields["controls"] == sorted(C)
    assert found[0].fields["ratio"] == pytest.approx(float(_ratio(D.c, D.G, C)[1]), rel=1e-12)


def test_a_deep_design_no_better_is_not_adopted(monkeypatch):
    """When the first descent already holds the optimum and the deep one returns a
    worse design, the optimum stays and no descent incumbent is reported."""
    y, X, _, C = _fixture()
    worse = set(search.incumbent(y, X, restarts=1, seed=0)[1])
    monkeypatch.setattr(search, "incumbent",
                        lambda y, X, *, restarts, seed: (None, set(C) if restarts == 1 else worse))
    seen, kw = _collect()
    out = certify_slope_free(y, X, restarts=1, deep_restarts=200, **kw)
    assert set(out.controls) == C and out.certified
    assert not [e for e in seen if e.kind == "incumbent" and e.fields["source"] == "descent"]


def test_an_equal_deep_design_is_not_reported(monkeypatch):
    """A deep design at the same ratio is not a new incumbent."""
    y, X, _, C = _fixture()
    monkeypatch.setattr(search, "incumbent", lambda y, X, *, restarts, seed: (None, set(C)))
    seen, kw = _collect()
    certify_slope_free(y, X, restarts=1, deep_restarts=200, **kw)
    assert not [e for e in seen if e.kind == "incumbent" and e.fields["source"] == "descent"]


# ---------------------------------------------------------------- when it runs


@pytest.mark.parametrize("restarts,deep", [(1, 0), (1, 1), (5, 5), (5, 3)])
def test_no_deep_descent_unless_it_has_more_restarts(calls, restarts, deep):
    y, X, _, _ = _fixture()
    certify_slope_free(y, X, restarts=restarts, deep_restarts=deep)
    assert calls == [restarts]


def test_one_more_restart_is_enough_to_run_it(calls):
    y, X, _, _ = _fixture()
    certify_slope_free(y, X, restarts=1, deep_restarts=2)
    assert calls == [1, 2]


def test_no_deep_descent_with_excluded_designs(calls):
    """With designs excluded the incumbent comes from the neighbour search; the deep
    descent is not run, so it cannot return an excluded design."""
    y, X, _, C = _fixture()
    out = certify_slope_free(y, X, restarts=1, deep_restarts=200, exclude=[C])
    assert calls == [1]
    assert set(out.controls) != C


def test_no_deep_descent_when_the_screen_closes(calls):
    """A ceiling under the optimum is held by the group screen, before the hull
    stage, and the deep descent is not run."""
    y, X, best, _ = _fixture()
    out = certify_slope_free(y, X, restarts=1, deep_restarts=200, ceiling=Q(best) * Q(99, 100))
    assert out.certified and out.ceiling_held and out.sizes_after_hull == []
    assert calls == [1]


def test_the_deep_descent_runs_after_an_open_screen(calls):
    """A ceiling between the optimum and the one-restart design's RSS cannot be held,
    so the screen stays open and the deep descent runs; the deep design is below the
    ceiling, which is then not held."""
    y, X, best, C = _fixture()
    seen, kw = _collect()
    out = certify_slope_free(y, X, restarts=1, deep_restarts=200, ceiling=Q(best) * 2, **kw)
    screens = [e for e in seen if e.kind == "screen"]
    assert len(screens) == 1 and not screens[0].fields["certified"]
    assert calls == [1, 200]
    assert out.certified and not out.ceiling_held and set(out.controls) == C


def test_a_ceiling_above_the_deep_design_keeps_its_ratio(calls):
    """With the ceiling's ratio above the deep design's, the search keeps testing at
    the ceiling, and the deep design is reported when nothing beats it."""
    y, X, best, C = _fixture()
    out = certify_slope_free(y, X, restarts=1, deep_restarts=200, ceiling=Q(best) * Q(99, 100),
                             group_screen=False)
    assert calls == [1, 200]
    assert out.certified and out.ceiling_held and set(out.controls) == C


# ---------------------------------------------------------------- edge


def test_one_control():
    """With one control there is one design, and both descents return it."""
    y, X = panel(1, p=1)
    best, C = positive_optimum(y, X)
    if best == np.inf:
        pytest.skip("panel 1 at one control has no positive slope")
    out = certify_slope_free(y, X, restarts=1, deep_restarts=10)
    assert out.certified and set(out.controls) == C


def test_deep_restarts_zero_turns_it_off_and_still_certifies(calls):
    y, X, best, C = _fixture()
    out = certify_slope_free(y, X, restarts=1, deep_restarts=0)
    assert calls == [1]
    assert out.certified and set(out.controls) == C
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [-1, 1.5, "500", True, np.int64(5)])
def test_a_bad_deep_restarts_is_refused(bad):
    y, X, _, _ = _fixture()
    with pytest.raises(MlsynthConfigError, match="deep_restarts"):
        certify_slope_free(y, X, restarts=1, deep_restarts=bad)


def test_a_bad_deep_restarts_is_refused_before_any_descent(calls):
    y, X, _, _ = _fixture()
    with pytest.raises(MlsynthConfigError):
        certify_slope_free(y, X, restarts=1, deep_restarts=-1)
    assert calls == []


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(3, 7), restarts=st.integers(1, 3),
       deep=st.integers(0, 40))
def test_property_the_deep_descent_changes_no_answer(seed, p, restarts, deep):
    """Whatever the deep descent finds, the certified answer is the enumerated
    optimum, the reported ratio is the reported design's, and the incumbent is the
    one found without it."""
    y, X = panel(seed % 80, p=p)
    best, _ = positive_optimum(y, X)
    if best == np.inf:
        return
    D = ExactData(y, X)
    out = certify_slope_free(y, X, restarts=restarts, deep_restarts=deep)
    off = certify_slope_free(y, X, restarts=restarts, deep_restarts=0)
    assert out.certified and off.certified
    assert out.ratio == _ratio(D.c, D.G, set(out.controls))[1]
    assert out.incumbent == off.incumbent
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(3, 6), deep=st.integers(2, 40))
def test_property_the_deep_descent_runs_exactly_when_the_search_reaches_the_hull(seed, p, deep):
    """Without exclusions or a ceiling the search always reaches the hull stage, so
    the deep descent runs whenever it has more restarts."""
    y, X = panel(seed % 80, p=p)
    if positive_optimum(y, X)[0] == np.inf:
        return
    seen = []
    real = search.incumbent

    def spy(y, X, *, restarts, seed):
        seen.append(restarts)
        return real(y, X, restarts=restarts, seed=seed)

    old, search.incumbent = search.incumbent, spy
    try:
        certify_slope_free(y, X, restarts=1, deep_restarts=deep)
    finally:
        search.incumbent = old
    assert seen == [1, deep]
