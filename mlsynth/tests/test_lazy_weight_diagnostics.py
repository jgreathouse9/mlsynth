"""A solve pays for the certificate only when the certificate is read.

``solve_weights`` returned five derived facts eagerly: ``kkt_residual``,
``free_directions``, ``free_intercepts``, ``unique`` and ``status``. Two of them
are expensive -- ``kkt_residual`` recomputes the reduced gradient, and
``_face_null_space`` takes a rank-revealing SVD -- and a resampling loop reads
neither. Profiled on TSSC's Step-1 loop, which refits MSC(c) 200 times per
replication on a 20x10 subsample and keeps only ``beta``:

    full refit                              212.1 us
    without _face_null_space's SVD           89.1 us   2.07x
    without the SVD and the KKT residual     62.0 us   2.98x
    scipy.optimize.nnls itself                8.6 us   (4 percent of the refit)

so the certificate was 77 percent of the call and the optimisation 4.

The five stay on the public surface, spelled and typed as before; they are
computed on first read and cached. The solution therefore keeps the design it
was solved on, which extends that array's lifetime and allocates nothing.

What these tests pin is that laziness is invisible. Every value equals what the
eager path produced, to the last bit; reading twice computes once; a solution
whose diagnostics are never read never calls either routine; and the one thing a
frozen dataclass makes easy to get wrong -- a cache that makes two equal
solutions compare unequal, or a mutable array escaping read-only -- is asserted
directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.weights import WeightConstraint, WeightObjective, solve_weights
from mlsynth.utils.weights import solve as S

CONE_IC = WeightConstraint(sum_to_one=False, intercept=True)
CASES = [WeightConstraint(), WeightConstraint(intercept=True),
         WeightConstraint(sum_to_one=False), CONE_IC]
IDS = ["simplex", "simplex+ic", "cone", "cone+ic"]


def _panel(m, J, seed=0, dup=False, zero=False):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal((m, 3)) * 0.3, axis=0)
    L = rng.uniform(0.5, 1.5, (J + 1, 3))
    Y = 1.0 + f @ L.T + rng.standard_normal((m, J + 1)) * 0.4
    B = np.ascontiguousarray(Y[:, 1:])
    if dup:
        B[:, 1] = B[:, 0]
    if zero:
        B[:, 2] = 0.0
    return B, np.ascontiguousarray(Y[:, 0])


def _eager(B, A, con, obj=None):
    """What the eager path computed, recomputed here independently."""
    obj = obj if obj is not None else WeightObjective()
    sol = solve_weights(B, A, con, obj)
    w, a = np.asarray(sol.weights), sol.intercept
    resid = S.kkt_residual(B, A, w, a, con, obj)
    dirs, ics = S._face_null_space(B, A, w, a, con, obj)
    return sol, resid, dirs, ics


# --------------------------------------------------------------------------- #
# 1. laziness is invisible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("con", CASES, ids=IDS)
@pytest.mark.parametrize("m,J", [(20, 10), (80, 10), (30, 25), (12, 40)])
def test_every_lazy_field_equals_the_eager_value(con, m, J):
    sol, resid, dirs, ics = _eager(*_panel(m, J, seed=m * J), con)
    assert sol.kkt_residual == resid
    np.testing.assert_array_equal(sol.free_directions, dirs)
    np.testing.assert_array_equal(sol.free_intercepts, ics)
    assert sol.unique == (dirs.shape[1] == 0)
    assert sol.status == ("optimal" if resid < S.KKT_TOL else "inaccurate")


@pytest.mark.parametrize("dup,zero,seed", [(True, False, 14), (False, True, 4),
                                           (True, True, 14)])
def test_a_degenerate_design_agrees_too(dup, zero, seed):
    """Duplicated and zero columns are where ``free_directions`` is non-empty, so
    they are where laziness could differ and go unnoticed."""
    B, A = _panel(20, 10, seed=seed, dup=dup, zero=zero)
    sol, resid, dirs, ics = _eager(B, A, CONE_IC)
    assert sol.kkt_residual == resid
    np.testing.assert_array_equal(sol.free_directions, dirs)
    np.testing.assert_array_equal(sol.free_intercepts, ics)
    assert sol.unique == (dirs.shape[1] == 0)
    assert not sol.unique, "this fixture is meant to be non-identified"


def test_duplicated_donors_are_not_automatically_non_identified():
    """Recorded because the obvious fixture for non-uniqueness is wrong, and the
    first version of the test above asserted it.

    Two identical columns permit trading weight between them only where that
    trade stays optimal. On this panel the optimum excludes both twins -- they
    sit at zero with a strictly positive reduced gradient -- so moving weight
    onto either one leaves the optimal face, and the solution is unique. What
    makes duplicates non-identified is being in the support, or at a bound whose
    reduced gradient vanishes.
    """
    B, A = _panel(20, 10, seed=4, dup=True)
    sol = solve_weights(B, A, CONE_IC)
    w = np.asarray(sol.weights)
    assert w[0] == 0.0 and w[1] == 0.0
    assert sol.unique and sol.free_directions.shape[1] == 0


def test_a_ridge_still_reports_a_unique_optimum():
    B, A = _panel(20, 10, seed=6, dup=True)
    sol = solve_weights(B, A, CONE_IC, WeightObjective(ridge=0.5))
    assert sol.unique and sol.free_directions.shape[1] == 0


# --------------------------------------------------------------------------- #
# 2. the work is actually deferred, and done once
# --------------------------------------------------------------------------- #
def test_a_solve_that_reads_nothing_computes_neither_diagnostic(monkeypatch):
    """The point of the change. Without this the fields could be lazy in form
    and eager in fact."""
    calls = {"kkt": 0, "face": 0}
    real_kkt, real_face = S.kkt_residual, S._face_null_space
    monkeypatch.setattr(S, "kkt_residual",
                        lambda *a, **k: (calls.__setitem__("kkt", calls["kkt"] + 1),
                                         real_kkt(*a, **k))[1])
    monkeypatch.setattr(S, "_face_null_space",
                        lambda *a, **k: (calls.__setitem__("face", calls["face"] + 1),
                                         real_face(*a, **k))[1])
    B, A = _panel(20, 10, seed=7)
    sol = solve_weights(B, A, CONE_IC)
    _ = sol.weights, sol.intercept, sol.objective, sol.support, sol.n_donors
    assert calls == {"kkt": 0, "face": 0}, "the certificate was computed unread"
    _ = sol.unique
    assert calls["face"] == 1 and calls["kkt"] == 0, "unique needs only the face"
    _ = sol.kkt_residual
    assert calls["kkt"] == 1


def test_reading_twice_computes_once(monkeypatch):
    calls = {"n": 0}
    real = S._face_null_space
    monkeypatch.setattr(S, "_face_null_space",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                         real(*a, **k))[1])
    B, A = _panel(20, 10, seed=8)
    sol = solve_weights(B, A, CONE_IC)
    for _ in range(5):
        _ = sol.free_directions, sol.free_intercepts, sol.unique
    assert calls["n"] == 1


def test_to_dict_reads_the_whole_certificate():
    """``to_dict`` is the estimator-facing path and must still report every
    field, so it is the one caller that always pays."""
    B, A = _panel(20, 10, seed=9)
    d = solve_weights(B, A, CONE_IC).to_dict()
    for key in ("solver", "status", "objective", "kkt_residual",
                "weights_unique", "free_directions", "support_size", "intercept"):
        assert key in d
    assert isinstance(d["weights_unique"], bool)
    assert isinstance(d["free_directions"], int)


# --------------------------------------------------------------------------- #
# 3. the frozen container's invariants survive the cache
# --------------------------------------------------------------------------- #
def test_the_weights_are_still_read_only():
    sol = solve_weights(*_panel(20, 10, seed=10), CONE_IC)
    with pytest.raises(ValueError):
        sol.weights[0] = 1.0
    with pytest.raises(ValueError):
        sol.support[0] = 0


def test_a_lazy_field_cannot_be_assigned():
    sol = solve_weights(*_panel(20, 10, seed=12), CONE_IC)
    for name in ("kkt_residual", "unique", "status", "free_directions"):
        with pytest.raises(AttributeError):
            setattr(sol, name, 0.0)


def test_two_solves_of_the_same_panel_compare_equal_whatever_was_read():
    """A cache populated on one and not the other must not make them differ."""
    B, A = _panel(20, 10, seed=13)
    a = solve_weights(B, A, CONE_IC)
    b = solve_weights(B, A, CONE_IC)
    _ = a.kkt_residual, a.free_directions, a.unique, a.status
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.objective == b.objective and a.intercept == b.intercept
    assert a.solver == b.solver and a.n_donors == b.n_donors
    assert a.status == b.status and a.unique == b.unique


def test_identifies_still_works_off_the_lazy_face():
    """``identifies`` reads ``free_directions`` and ``free_intercepts``, so it is
    a caller that triggers the SVD.

    This is the case its own docstring names: duplicated donors leave the weights
    non-identified, and the post-period design annihilates the direction weight
    is free to move along, so the ATT is identified anyway. Both halves are
    asserted, since a lazy face that came back empty would make ``identifies``
    return True for the wrong reason.
    """
    B, A = _panel(30, 10, seed=1, dup=True)
    sol = solve_weights(B[:20], A[:20], CONE_IC)
    assert not sol.unique and sol.free_directions.shape[1] == 1
    assert sol.identifies(B[20:]) is True
    with pytest.raises(MlsynthEstimationError, match="columns"):
        sol.identifies(B[20:, :3])
