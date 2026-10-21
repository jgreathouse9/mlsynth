"""The unit-level penalty fits the members it is asked for, and only those.

``per_unit_reproducibility(Y_fit, members)`` is equation (10)'s charge: for each
unit, the smallest squared distance from its fitting-period outcomes to the
simplex hull of every other unit's. Each unit's charge depends on the panel
and not on the cluster asking for it. The function used to ignore ``members``
and fit every unit in the panel, so a design with ``K`` clusters fitted the
panel ``K`` times on the exact backend and ``2K`` times on the MIQP path (the
objective and the refit each ask), to keep one cluster's rows each time.

It now fits ``members`` alone, each with the same solver on the same program,
so a member's charge is the same number bit for bit, and a unit nobody asked
for is ``nan`` instead of a value a caller could read by mistake.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers import formulation as form
from mlsynth.utils.marex_helpers.formulation import per_unit_reproducibility
from mlsynth.utils.solvers.active_set import solve_simplex_qp


def _every_unit(Y_fit, members=None):
    """The previous implementation: every unit against every other one."""
    N = Y_fit.shape[0]
    out = np.zeros(N, dtype=float)
    for j in range(N):
        others = [i for i in range(N) if i != j]
        if not others:
            continue
        D = Y_fit[others, :].T
        a = solve_simplex_qp(D, Y_fit[j, :])
        residual = Y_fit[j, :] - D @ a
        out[j] = float(residual @ residual)
    return out


def _panel(N=12, T=20, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    return rng.normal(size=(N, 2)) @ F.T + 0.3 * rng.standard_normal((N, T)) + 5.0


def _counting(monkeypatch):
    calls = []

    def spy(D, y, **kw):
        calls.append(D.shape[1])
        return solve_simplex_qp(D, y, **kw)

    monkeypatch.setattr(form, "solve_simplex_qp", spy)
    return calls


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_charges_each_member():
    d = per_unit_reproducibility(_panel(), [1, 4, 7])
    assert np.all(np.isfinite(d[[1, 4, 7]])) and np.all(d[[1, 4, 7]] >= 0)


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_a_members_charge_is_the_previous_number_bit_for_bit():
    Y = _panel()
    members = [0, 3, 5, 11]
    np.testing.assert_array_equal(per_unit_reproducibility(Y, members)[members],
                                  _every_unit(Y)[members])


def test_the_whole_panel_as_members_reproduces_every_charge():
    Y = _panel()
    np.testing.assert_array_equal(per_unit_reproducibility(Y, list(range(12))),
                                  _every_unit(Y))


def test_it_fits_each_member_once_against_every_other_unit(monkeypatch):
    calls = _counting(monkeypatch)
    per_unit_reproducibility(_panel(N=12), [2, 6, 9])
    assert calls == [11, 11, 11]


def test_a_unit_nobody_asked_for_is_nan():
    d = per_unit_reproducibility(_panel(), [2, 6])
    rest = np.setdiff1d(np.arange(12), [2, 6])
    assert np.isnan(d[rest]).all()
    assert d.shape == (12,)


def test_members_as_an_array_or_a_list_agree():
    Y = _panel()
    np.testing.assert_array_equal(per_unit_reproducibility(Y, [1, 5]),
                                  per_unit_reproducibility(Y, np.array([1, 5])))


def _clustered(N=18, T=30, seed=1):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    Y = rng.normal(size=(N, 2)) @ F.T + 0.3 * rng.standard_normal((N, T)) + 10.0
    rows = [{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]),
             "grp": f"c{j % 3}"} for j in range(N) for t in range(T)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("backend", ["exact", "mip"])
def test_a_clustered_design_is_unchanged(monkeypatch, backend):
    """Same charges, so the same search and the same design, on both backends."""
    cfg = dict(df=_clustered(), outcome="y", unitid="unit", time="time", T0=24,
               m_eq=1, cluster="grp", design="unit_penalized", xi=1.0,
               backend=backend)
    new = MAREX(cfg).fit()
    monkeypatch.setattr(form, "per_unit_reproducibility", _every_unit)
    monkeypatch.setattr(exact_mod, "per_unit_reproducibility", _every_unit)
    old = MAREX(cfg).fit()
    np.testing.assert_array_equal(new.globres.treated_weights_agg,
                                  old.globres.treated_weights_agg)
    np.testing.assert_array_equal(new.globres.control_weights_agg,
                                  old.globres.control_weights_agg)
    assert new.globres.objective == old.globres.objective


@pytest.mark.parametrize("backend", ["exact", "mip"])
def test_a_clustered_design_fits_each_unit_at_most_twice(monkeypatch, backend):
    """Three clusters of six: 18 fits a pass, not 18 per cluster."""
    calls = _counting(monkeypatch)
    MAREX(dict(df=_clustered(), outcome="y", unitid="unit", time="time", T0=24,
               m_eq=1, cluster="grp", design="unit_penalized", xi=1.0,
               backend=backend)).fit()
    passes = 1 if backend == "exact" else 2           # the MIQP refits once
    assert len(calls) == 18 * passes


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_no_members_fits_nothing(monkeypatch):
    calls = _counting(monkeypatch)
    d = per_unit_reproducibility(_panel(), [])
    assert calls == [] and np.isnan(d).all()


def test_a_one_unit_panel_charges_nothing():
    """No other unit, no hull: the charge is zero, as before."""
    d = per_unit_reproducibility(_panel(N=1), [0])
    assert d.tolist() == [0.0]


def test_a_repeated_member_is_fitted_once(monkeypatch):
    calls = _counting(monkeypatch)
    per_unit_reproducibility(_panel(), [3, 3, 3])
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [[12], [-1], [-13], [0, 99]])
def test_a_member_outside_the_panel_is_refused(bad):
    with pytest.raises(IndexError):
        per_unit_reproducibility(_panel(N=12), bad)


# --------------------------------------------------------------------------- #
# property
# --------------------------------------------------------------------------- #
from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def _panel_and_members(draw):
    N = draw(st.integers(1, 9))
    T = draw(st.integers(1, 12))
    seed = draw(st.integers(0, 2 ** 31 - 1))
    Y = np.random.default_rng(seed).normal(size=(N, T)) * draw(
        st.floats(0.01, 100.0)) + draw(st.floats(-50.0, 50.0))
    members = draw(st.lists(st.integers(0, N - 1), max_size=2 * N))
    return Y, members


@settings(max_examples=150, deadline=None)
@given(_panel_and_members())
def test_members_get_the_previous_charge_and_no_one_else_gets_one(case):
    Y, members = case
    d = per_unit_reproducibility(Y, members)
    asked = np.zeros(Y.shape[0], dtype=bool)
    asked[np.asarray(members, dtype=int)] = True
    np.testing.assert_array_equal(d[asked], _every_unit(Y)[asked])
    assert np.isnan(d[~asked]).all()
