"""Blank periods held back from the certified search (#965).

certify_treatment holds back the last ``blank_periods`` periods of the panel, none
by default and three test lengths when None, and certifies on the periods before
them: the design window. No design choice sees the blank periods, so the A/A test at each
blank origin (fit on every period before it, the cumulative effect over the next
``n_test`` tested against zero) is a window the search could not have fitted to.
The four gates are read on the whole panel, as the library reports a design.
``blank_periods=0`` searches the whole panel, the study's search before blank
periods. Every claim is checked against enumeration on the design window and
against the library's own A/A test."""
import inspect

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import numpy as np

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.objective import MIN_HOLDOUT_PRETEST
from mlsynth.utils.tbr_helpers.design.objective import aa_test_ok

from .brute import raw_panel, relative_optimum
from .gates import gates_of
from .treatment import BLANK_TESTS, _checked_blank, certify_treatment

N_TEST = 4
SETTINGS = settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _panel(seed=5, n=40, p=6):
    return raw_panel(seed, n=n, p=p)


def _origins(n, B, n_test):
    D0 = n - B
    return [D0 + i * n_test for i in range(B // n_test)]


# ---------------------------------------------------------------- smoke


def test_smoke_none_holds_back_three_test_lengths():
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=None)
    assert out.certified and out.treatment is not None
    assert out.blank_periods == BLANK_TESTS * N_TEST == 12
    assert out.design_periods == M.shape[0] - 12
    assert len(out.blank_aa) == BLANK_TESTS


# ---------------------------------------------------------------- unit


def test_the_default_holds_back_nothing():
    """The default searches the whole panel: a design chosen with blank periods held
    back detected a planted 3% lift less often on panels of 40 to 104 periods and on
    three data panels, at the same false-positive rate (README, "Blank periods")."""
    assert inspect.signature(certify_treatment).parameters["blank_periods"].default == 0
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3)
    assert out.blank_periods == 0 and out.design_periods == M.shape[0] and out.blank_aa == []


def test_the_search_is_the_design_windows():
    """The design is the enumerated optimum of the design window, and the same as a
    search given only that window."""
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=None)
    D = M[:out.design_periods]
    key, T, C = relative_optimum(D, 2, N_TEST)
    assert tuple(out.treatment) == tuple(T) and tuple(out.controls) == tuple(C)
    alone = certify_treatment(D, 2, n_test=N_TEST, restarts=3, blank_periods=0)
    assert (out.treatment, out.controls, out.key) == (alone.treatment, alone.controls, alone.key)


def test_each_blank_origin_is_the_librarys_aa_test():
    M = _panel()
    B = 3 * N_TEST
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=B)
    y = M[:, out.treatment].sum(axis=1)
    x = M[:, out.controls].sum(axis=1)
    want = [aa_test_ok(y[:o + N_TEST], x[:o + N_TEST], N_TEST) for o in _origins(M.shape[0], B, N_TEST)]
    assert out.blank_aa == want


def test_the_gates_are_read_on_the_whole_panel():
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=None)
    assert out.gates == gates_of(M[:, out.treatment].sum(axis=1), M, out.controls, N_TEST)


def test_a_blank_window_the_search_never_sees():
    """Rewriting the blank periods changes nothing the search reports: the design,
    its key and its certificate come from the design window alone."""
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=None)
    N = M.copy()
    N[out.design_periods:] = N[out.design_periods:][::-1] * 1.7
    other = certify_treatment(N, 2, n_test=N_TEST, restarts=3, blank_periods=None)
    assert (other.treatment, other.controls, other.key, other.certified) == \
           (out.treatment, out.controls, out.key, out.certified)


@given(seed=st.integers(0, 10_000), tests=st.integers(2, 4), scale=st.floats(0.1, 10.0))
@SETTINGS
def test_property_the_blank_periods_never_reach_the_search(seed, tests, scale):
    """Over panels, blank lengths and rewrites of the blank periods, the design is
    the design window's enumerated optimum and does not move when they are rewritten."""
    M = raw_panel(seed, n=36, p=5)
    B = tests * N_TEST
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=B)
    D0 = M.shape[0] - B
    assert out.design_periods == D0 and out.certified
    key, T, C = relative_optimum(M[:D0], 2, N_TEST)
    assert tuple(out.treatment) == tuple(T) and tuple(out.controls) == tuple(C)
    N = M.copy()
    N[D0:] *= scale
    other = certify_treatment(N, 2, n_test=N_TEST, restarts=3, blank_periods=B)
    assert (other.treatment, other.controls, other.key) == (out.treatment, out.controls, out.key)
    assert len(out.blank_aa) == tests


def test_the_pool_holds_back_the_same_periods():
    M = _panel()
    one = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=None)
    two = certify_treatment(M, 2, n_test=N_TEST, restarts=3, n_jobs=2, blank_periods=None)
    assert (two.treatment, two.controls, two.key, two.blank_periods, two.blank_aa) == \
           (one.treatment, one.controls, one.key, one.blank_periods, one.blank_aa)


# ---------------------------------------------------------------- edge


def test_zero_blank_periods_is_the_whole_panel():
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
    assert out.blank_periods == 0 and out.design_periods == M.shape[0] and out.blank_aa == []
    key, T, C = relative_optimum(M, 2, N_TEST)
    assert tuple(out.treatment) == tuple(T)


def test_the_shortest_blank_window_is_two_test_lengths():
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=2 * N_TEST)
    assert out.blank_periods == 2 * N_TEST and len(out.blank_aa) == 2


def test_a_blank_window_not_a_multiple_of_the_test_tests_whole_windows():
    """Ten blank periods at a test of four hold two whole windows; the last two
    periods are fitted on by the second origin's successors, never tested alone."""
    M = _panel()
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=10)
    assert out.design_periods == M.shape[0] - 10 and len(out.blank_aa) == 2


def test_a_level_positive_only_in_the_design_window_is_enough():
    """The relative key reads the design window's levels, so a geo whose mean over
    the blank periods is negative does not stop the search."""
    M = _panel()
    N = M.copy()
    N[-12:, 3] = -1000.0
    out = certify_treatment(N, 2, n_test=N_TEST, restarts=3, blank_periods=None)
    assert out.certified and out.design_periods == N.shape[0] - 12


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [1, N_TEST, 2 * N_TEST - 1])
def test_a_blank_window_shorter_than_two_tests_raises(bad):
    with pytest.raises(MlsynthDataError, match="blank_periods"):
        certify_treatment(_panel(), 2, n_test=N_TEST, restarts=3, blank_periods=bad)


@pytest.mark.parametrize("bad", [-1, True, 2.0, "8"])
def test_a_blank_period_count_that_is_not_a_count_raises(bad):
    with pytest.raises(MlsynthDataError, match="blank_periods"):
        certify_treatment(_panel(), 2, n_test=N_TEST, restarts=3, blank_periods=bad)


def test_a_design_window_too_short_for_the_search_raises():
    """The search's start scores designs with the A/A test, which fits on the
    window less one test length and needs MIN_HOLDOUT_PRETEST periods there. A hold
    back leaving fewer is refused naming blank_periods, not left to fail inside the
    start's local search."""
    n = 3 * N_TEST + N_TEST + MIN_HOLDOUT_PRETEST - 1
    with pytest.raises(MlsynthDataError, match="blank_periods"):
        certify_treatment(_panel(n=n), 2, n_test=N_TEST, restarts=3, blank_periods=None)


def test_the_shortest_design_window_the_search_takes():
    n = 3 * N_TEST + N_TEST + MIN_HOLDOUT_PRETEST
    out = certify_treatment(_panel(n=n), 2, n_test=N_TEST, restarts=3, blank_periods=None)
    assert out.design_periods == N_TEST + MIN_HOLDOUT_PRETEST and out.treatment is not None


@given(n=st.integers(8, 60), tests=st.integers(2, 4), n_test=st.integers(2, 6))
@settings(max_examples=200, deadline=None)
def test_property_a_hold_back_is_refused_exactly_when_the_search_cannot_run(n, tests, n_test):
    B = tests * n_test
    if n - B < n_test + MIN_HOLDOUT_PRETEST:
        with pytest.raises(MlsynthDataError, match="blank_periods"):
            _checked_blank(B, n_test, n)
    else:
        assert _checked_blank(B, n_test, n) == B


def test_a_blank_window_that_leaves_too_few_design_periods_raises():
    M = _panel(n=14)
    with pytest.raises(MlsynthDataError, match="blank"):
        certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=None)
