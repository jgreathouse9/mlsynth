"""A panel with a missing or infinite value is refused at the door.

certify_treatment and guided_treatment both start from the library's greedy and
local search, whose least squares raised numpy's LinAlgError ("SVD did not converge")
on a panel with a nan or an inf, a solver's failure in place of the data error the
library's contract promises. Both entries now check the panel first
(treatment._checked_panel) and raise MlsynthDataError naming the geo and the period.
"""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import guided, treatment
from .brute import geo_panel

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _certify(M, **kw):
    kw = {"n_test": N_TEST, "restarts": 3, "pass_restarts": 2, "gap": Q(0), "n_jobs": 1, **kw}
    return treatment.certify_treatment(M, 2, **kw)


def _guided(M, **kw):
    return guided.guided_treatment(M, 2, n_test=N_TEST, restarts=3, **kw)


def _bad(value, t=3, j=2):
    M = geo_panel(5).copy()
    M[t, j] = value
    return M


# ---------------------------------------------------------------- smoke


def test_smoke_a_finite_panel_passes_the_check_unchanged():
    M = geo_panel(5)
    out = treatment._checked_panel(M)
    assert out.dtype == float and out.shape == M.shape and np.array_equal(out, M)


def test_smoke_a_finite_panel_still_certifies():
    assert _certify(geo_panel(5)).certified


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_the_message_names_the_geo_and_the_period(value):
    with pytest.raises(MlsynthDataError, match=r"geo 2 .*period 3"):
        treatment._checked_panel(_bad(value, t=3, j=2))
    with pytest.raises(MlsynthDataError, match=r"geo 4 .*period 0"):
        treatment._checked_panel(_bad(value, t=0, j=4))


def test_the_first_bad_entry_is_named_in_period_order():
    M = geo_panel(5).copy()
    M[7, 1] = np.nan
    M[2, 4] = np.inf
    with pytest.raises(MlsynthDataError, match=r"geo 4 .*period 2"):
        treatment._checked_panel(M)


def test_the_count_of_bad_entries_is_reported():
    M = geo_panel(5).copy()
    M[:, 3] = np.nan
    with pytest.raises(MlsynthDataError, match=rf"{M.shape[0]} entries"):
        treatment._checked_panel(M)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_certify_treatment_refuses_the_panel(value, n_jobs):
    with pytest.raises(MlsynthDataError, match="finite"):
        _certify(_bad(value), n_jobs=n_jobs)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_certify_treatment_refuses_it_with_a_given_start(value):
    with pytest.raises(MlsynthDataError, match="finite"):
        _certify(_bad(value), start=[0, 1])


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("start", [None, [0, 1]])
def test_guided_treatment_refuses_the_panel(value, start):
    with pytest.raises(MlsynthDataError, match="finite"):
        _guided(_bad(value), start=start)


def test_the_check_comes_before_the_search(monkeypatch):
    """The refusal is the check's, not a later stage's: the start is never reached."""
    called = []
    monkeypatch.setattr(treatment, "_start", lambda *a, **k: called.append(1))
    monkeypatch.setattr(guided, "_start", lambda *a, **k: called.append(1))
    for run in (_certify, _guided):
        with pytest.raises(MlsynthDataError, match="finite"):
            run(_bad(np.nan))
    assert not called


@pytest.mark.parametrize("bad", [np.arange(6.0), np.zeros((2, 3, 4))])
def test_a_panel_that_is_not_two_dimensional_is_refused(bad):
    with pytest.raises(MlsynthDataError, match="periods by geos"):
        treatment._checked_panel(bad)


# ---------------------------------------------------------------- edge


def test_a_panel_of_one_period():
    M = geo_panel(5)[:1]
    assert np.array_equal(treatment._checked_panel(M), M)


def test_integer_and_list_panels_become_floats():
    out = treatment._checked_panel([[1, 2, 3], [4, 5, 6]])
    assert out.dtype == float and out.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


# ---------------------------------------------------------------- property


@SETTINGS
@given(t=st.integers(0, 23), j=st.integers(0, 6), value=st.sampled_from([np.nan, np.inf, -np.inf]),
       seed=st.integers(0, 30))
def test_property_any_non_finite_entry_is_refused_where_it_is(t, j, value, seed):
    M = geo_panel(seed, n=24, p=7).copy()
    M[t, j] = value
    with pytest.raises(MlsynthDataError, match=rf"geo {j} .*period {t}"):
        treatment._checked_panel(M)


@SETTINGS
@given(seed=st.integers(0, 30), n=st.integers(1, 30), p=st.integers(2, 9))
def test_property_a_finite_panel_passes_unchanged(seed, n, p):
    M = geo_panel(seed, n=n, p=p)
    assert np.array_equal(treatment._checked_panel(M), M)
