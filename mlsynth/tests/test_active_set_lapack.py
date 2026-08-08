"""The active set's free-set solve calls LAPACK directly, and must not move a digit.

``solve_simplex_qp`` spends nearly all of its time in one ``lstsq`` per pivot on
a small free-set system. At those sizes ``scipy.linalg.lstsq``'s Python wrapper
-- argument validation, ``asarray_chkfinite``, a workspace query per call --
costs more than the LAPACK work it wraps, so the solver calls ``dgelsy`` itself
with the workspace size cached per shape.

That is a speed change and nothing else. ``dgelsy`` is the driver
``scipy.linalg.lstsq(..., lapack_driver="gelsy")`` already dispatched to, so
these tests pin the substitution as *bit-identical*, not merely close: every
estimator built on the bilevel engine (MEDSC, SCD, the ridge augmentation, the
conformal and jackknife loops) reads its weights through this function, and a
change of one ulp there would ripple into published replication numbers.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import lstsq as scipy_lstsq

from mlsynth.utils.bilevel.active_set import solve_simplex_qp


def _reference(M, b):
    """What the solver used to call, verbatim."""
    return scipy_lstsq(M, b, lapack_driver="gelsy")[0]


SHAPES = [
    (19, 37),    # Prop 99 at the first pivot: underdetermined, m < nF
    (19, 20),    # mid-sweep
    (19, 8),     # late, once the support has formed
    (30, 5),     # comfortably overdetermined
    (6, 6),      # square
    (1, 3),      # degenerate single row
    (12, 1),     # single column
]


class TestTheLapackCallMatchesScipyExactly:
    @pytest.mark.parametrize("m,n", SHAPES)
    def test_bit_identical_on_well_posed_systems(self, m, n):
        from mlsynth.utils.bilevel.active_set import _gelsy_lstsq

        rng = np.random.default_rng(m * 100 + n)
        M = rng.normal(size=(m, n))
        b = rng.normal(size=m)
        got, want = _gelsy_lstsq(M, b), _reference(M, b)
        assert got.shape == want.shape
        assert np.array_equal(got, want), f"max|d|={np.abs(got - want).max():.3e}"

    def test_bit_identical_on_a_rank_deficient_system(self):
        # Collinear donors are the case the free-set solve must survive without
        # an epsilon-I fudge; gelsy is rank-revealing and both paths must agree.
        from mlsynth.utils.bilevel.active_set import _gelsy_lstsq

        rng = np.random.default_rng(7)
        M = rng.normal(size=(10, 4))
        M[:, 3] = M[:, 0] + 2.0 * M[:, 1]          # exact linear dependence
        b = rng.normal(size=10)
        assert np.array_equal(_gelsy_lstsq(M, b), _reference(M, b))

    def test_bit_identical_on_a_zero_matrix(self):
        from mlsynth.utils.bilevel.active_set import _gelsy_lstsq

        M = np.zeros((5, 3))
        b = np.arange(5, dtype=float)
        assert np.array_equal(_gelsy_lstsq(M, b), _reference(M, b))

    def test_alternating_shapes_do_not_poison_the_workspace_cache(self):
        # The workspace size is cached per shape. Interleaving shapes must not
        # let one shape's cached lwork or pivot array be used for another.
        from mlsynth.utils.bilevel.active_set import _gelsy_lstsq

        rng = np.random.default_rng(3)
        problems = [(rng.normal(size=(m, n)), rng.normal(size=m))
                    for m, n in SHAPES] * 3
        for M, b in problems:
            assert np.array_equal(_gelsy_lstsq(M, b), _reference(M, b))

    def test_the_input_matrix_is_not_overwritten(self):
        # gelsy overwrites its arguments in place when told to; the solver slices
        # a fresh M each pivot but the helper must not mutate what it is handed.
        from mlsynth.utils.bilevel.active_set import _gelsy_lstsq

        rng = np.random.default_rng(11)
        M = rng.normal(size=(9, 4))
        b = rng.normal(size=9)
        M0, b0 = M.copy(), b.copy()
        _gelsy_lstsq(M, b)
        assert np.array_equal(M, M0) and np.array_equal(b, b0)


class TestTheSolverAnswerIsUnchanged:
    """A differential test against the pre-change code path.

    The free-set solve is the only thing that moved, so re-running the whole
    active set with the scipy call restored must reproduce the shipped solver
    weight-for-weight on every problem.
    """

    @staticmethod
    def _solve_via_scipy(B, A, monkeypatch):
        import mlsynth.utils.bilevel.active_set as mod

        monkeypatch.setattr(mod, "_gelsy_lstsq",
                            lambda M, b: _reference(M, b))
        return solve_simplex_qp(B, A)

    @pytest.mark.parametrize("m,J", [(19, 38), (8, 40), (30, 10), (5, 5), (25, 3)])
    def test_random_problems_agree_bit_for_bit(self, m, J, monkeypatch):
        rng = np.random.default_rng(m * 1000 + J)
        for _ in range(12):
            B = rng.normal(size=(m, J))
            A = rng.normal(size=m)
            fast = solve_simplex_qp(B, A)
            with monkeypatch.context() as mp:
                slow = self._solve_via_scipy(B, A, mp)
            assert np.array_equal(fast, slow), (
                f"max|d|={np.abs(fast - slow).max():.3e}")

    def test_collinear_donor_pool_agrees(self, monkeypatch):
        rng = np.random.default_rng(21)
        B = rng.normal(size=(15, 9))
        B[:, 4] = B[:, 0]                          # duplicate donor
        B[:, 8] = 0.5 * B[:, 1] - B[:, 2]
        A = rng.normal(size=15)
        fast = solve_simplex_qp(B, A)
        with monkeypatch.context() as mp:
            slow = self._solve_via_scipy(B, A, mp)
        assert np.array_equal(fast, slow)


class TestTheSolutionIsStillOptimal:
    """Solver-independent optimality, so the speed change cannot hide a defect."""

    @pytest.mark.parametrize("m,J", [(19, 38), (12, 25), (40, 6)])
    def test_kkt_certificate_holds(self, m, J):
        rng = np.random.default_rng(m + J)
        B = rng.normal(size=(m, J))
        A = rng.normal(size=m)
        w = solve_simplex_qp(B, A)

        assert w.min() >= -1e-9
        assert abs(w.sum() - 1.0) <= 1e-9
        g = 2.0 * (B.T @ (B @ w - A))              # gradient of ||A - Bw||^2
        lam = float(g.min())                       # the sum-to-one multiplier
        supported = w > 1e-8
        # stationarity: on the support the gradient is flat at the multiplier
        assert float(np.max(g[supported] - lam)) <= 1e-6 * (1.0 + abs(lam))
