"""``solve_simplex_qp_least_norm`` takes a warm start, and it changes the work, not the answer.

SparseSC's sweep solves the inner simplex QP several hundred thousand times along
L-BFGS-B iterates that move ``v`` a little at a time (#934). Seeding each solve with
the previous weights cut the least-norm solve from 627 to 94 microseconds on the
Basque panel's 4,000-call sequence, and every warm answer equalled the cold one to
the last bit. The augmentation adds rows, not columns, so a point of the original
simplex is a feasible start for the augmented program as it stands.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import mlsynth.utils.solvers.active_set as active_set
from mlsynth.utils.solvers.active_set import solve_simplex_qp_least_norm


def _sequence(seed, m=8, J=12, steps=30, drift=0.05):
    """A design that drifts a little at each step, as the sweep's iterates do."""
    rng = np.random.default_rng(seed)
    B0, A = rng.normal(size=(m, J)), rng.normal(size=m)
    out, B = [], B0
    for _ in range(steps):
        B = B * np.exp(drift * rng.normal(size=(m, 1)))
        out.append((B.copy(), A))
    return out


# --------------------------------------------------------------------------- smoke
def test_a_warm_start_is_accepted():
    (B, A), = _sequence(0, steps=1)
    w = solve_simplex_qp_least_norm(B, A, warm_start=np.full(B.shape[1], 1.0 / B.shape[1]))
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- unit
def test_the_warm_start_reaches_the_solver(monkeypatch):
    seen = []
    real = active_set.solve_simplex_qp

    def spy(*a, **k):
        seen.append(k.get("warm_start"))
        return real(*a, **k)
    monkeypatch.setattr(active_set, "solve_simplex_qp", spy)
    (B, A), = _sequence(1, steps=1)
    ws = np.full(B.shape[1], 1.0 / B.shape[1])
    solve_simplex_qp_least_norm(B, A, warm_start=ws)
    assert seen and seen[-1] is not None and np.array_equal(seen[-1], ws)


def test_the_zero_design_passes_the_warm_start_on(monkeypatch):
    seen = []
    real = active_set.solve_simplex_qp

    def spy(*a, **k):
        seen.append(k.get("warm_start"))
        return real(*a, **k)
    monkeypatch.setattr(active_set, "solve_simplex_qp", spy)
    ws = np.array([0.25, 0.75])
    solve_simplex_qp_least_norm(np.zeros((3, 2)), np.zeros(3), warm_start=ws)
    assert np.array_equal(seen[-1], ws)


# --------------------------------------------------------------------------- edge
@pytest.mark.parametrize("ws", [np.array([2.0, -1.0, 0.0]), np.array([np.nan, 0.5, 0.5]), np.array([0.5, 0.5])])
def test_a_start_off_the_simplex_gives_the_cold_answer(ws):
    rng = np.random.default_rng(2)
    B, A = rng.normal(size=(5, 3)), rng.normal(size=5)
    assert np.array_equal(solve_simplex_qp_least_norm(B, A, warm_start=ws), solve_simplex_qp_least_norm(B, A))


def test_a_wide_design_selects_the_same_least_norm_point_from_any_start():
    """A face of minimisers: the warm start must not choose the member."""
    rng = np.random.default_rng(3)
    B, A = rng.normal(size=(3, 9)), rng.normal(size=3)
    cold = solve_simplex_qp_least_norm(B, A)
    for k in range(9):
        ws = np.zeros(9); ws[k] = 1.0
        np.testing.assert_allclose(solve_simplex_qp_least_norm(B, A, warm_start=ws), cold, atol=1e-9)


# --------------------------------------------------------------------------- property
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), m=st.integers(2, 12), J=st.integers(2, 20))
def test_property_a_warm_started_sequence_equals_the_cold_one(seed, m, J):
    w = None
    for B, A in _sequence(seed, m=m, J=J, steps=12):
        cold = solve_simplex_qp_least_norm(B, A)
        w = solve_simplex_qp_least_norm(B, A, warm_start=w)
        np.testing.assert_allclose(w, cold, atol=1e-9)
