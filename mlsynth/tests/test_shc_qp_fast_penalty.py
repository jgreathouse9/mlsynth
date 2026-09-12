"""The SHC matching program's penalty, in the form that does not cost N^3.

``solve_shc_qp`` minimises the pre-window mismatch over the simplex plus a tiny
penalty on the components of ``w`` lying in the near-null space of
:math:`L^\\top L`. That penalty is what pins the solution down: with ``m`` rows
and ``N`` columns the Gram matrix has rank at most ``m``, so on a real panel
(``m = 24``, ``N = 229``) roughly 205 of the 229 directions are unidentified by
the fit and the penalty is the only thing that chooses among them.

Written with the null-space basis ``C`` it is a dense ``(N - r) x N`` operator,
and obtaining ``C`` means an eigendecomposition of the ``N x N`` Gram matrix.
Both are avoidable. Because the eigenvectors are a complete orthonormal set,
``C C' + V V' = I`` with ``V`` the ``r`` identified directions, so

.. math::

    \\lVert C^\\top w \\rVert^2 = w^\\top (I - VV^\\top) w
                               = \\lVert (I - VV^\\top) w \\rVert^2,

which is a rank-``r`` affine map -- 24 columns instead of 205 -- and ``V`` is
the leading right singular vectors of ``L``, from a thin SVD of an ``m x N``
matrix instead of a decomposition of an ``N x N`` one.

These tests pin the identity, not the speed: the two forms must agree on the
weights, not merely on the objective, because the unidentified directions move
the post-window counterfactual even where they leave the pre-window fit alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.shc_helpers.kernels import solve_shc_qp

_TOL = 1e-8


def _problem(m=12, N=60, seed=0):
    """A rank-deficient matching problem: N columns, at most m identified."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, m)
    L = np.column_stack([np.sin(3 * t + k / 7.0) + 0.05 * rng.normal(size=m)
                         for k in range(N)])
    target = L[:, :4] @ np.array([0.4, 0.3, 0.2, 0.1])
    return L, target


def _null_basis(L):
    """The dense form's ``C``: eigenvectors of ``L'L`` below the threshold."""
    from scipy.linalg import eigh
    ev, evec = eigh(L.T @ L)
    return evec[:, ev < _TOL]


class TestPenaltyIdentity:

    def test_the_two_penalty_forms_are_the_same_quadratic(self):
        """The algebra, before any solver is involved."""
        L, _target = _problem()
        C = _null_basis(L)
        _u, sv, Vt = np.linalg.svd(L, full_matrices=False)
        V = Vt[sv ** 2 >= _TOL].T
        assert C.shape[1] > 0 and V.shape[1] < L.shape[1]
        rng = np.random.default_rng(1)
        for _ in range(20):
            w = rng.normal(size=L.shape[1])
            dense = float(C.T @ w @ (C.T @ w))
            thin = float(np.sum((w - V @ (V.T @ w)) ** 2))
            assert dense == pytest.approx(thin, rel=1e-9, abs=1e-12)

    def test_the_projectors_are_complementary(self):
        L, _t = _problem()
        C = _null_basis(L)
        _u, sv, Vt = np.linalg.svd(L, full_matrices=False)
        V = Vt[sv ** 2 >= _TOL].T
        identity = C @ C.T + V @ V.T
        assert identity == pytest.approx(np.eye(L.shape[1]), abs=1e-9)

    def test_the_thin_basis_is_much_smaller(self):
        """The reason to prefer it, as a shape assertion."""
        L, _t = _problem(m=12, N=60)
        C = _null_basis(L)
        _u, sv, Vt = np.linalg.svd(L, full_matrices=False)
        V = Vt[sv ** 2 >= _TOL].T
        assert V.shape[1] <= L.shape[0]
        assert V.shape[1] < C.shape[1]


class TestSolverAgreement:

    def test_the_solved_weights_agree(self):
        """Not just the objective: the weights, since they move the post window."""
        L, target = _problem()
        w, obj = solve_shc_qp(L, target)
        assert w is not None
        # the reference: the dense form, solved here
        import cvxpy as cp
        C = _null_basis(L)
        v = cp.Variable(L.shape[1])
        pen = 1e-6 * cp.sum_squares(C.T @ v) if C.size else 0
        prob = cp.Problem(cp.Minimize(cp.sum_squares(target - L @ v) + pen),
                          [cp.sum(v) == 1, v >= 0])
        prob.solve(solver=cp.CLARABEL)
        assert w == pytest.approx(v.value, abs=1e-7)
        assert obj == pytest.approx(prob.value, rel=1e-7)

    def test_the_solution_is_on_the_simplex(self):
        L, target = _problem()
        w, _obj = solve_shc_qp(L, target)
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert (w >= -1e-9).all()

    def test_a_full_rank_problem_needs_no_penalty_and_still_solves(self):
        """When nothing is unidentified the penalty term drops out entirely."""
        rng = np.random.default_rng(3)
        L = rng.normal(size=(8, 3))
        target = L @ np.array([0.5, 0.3, 0.2])
        w, _obj = solve_shc_qp(L, target)
        assert w == pytest.approx(np.array([0.5, 0.3, 0.2]), abs=1e-6)

    def test_the_augmented_program_is_unchanged(self):
        L, target = _problem()
        w_shc, _ = solve_shc_qp(L, target)
        w_a, obj_a = solve_shc_qp(L, target, use_augmented=True,
                                  w_shc=w_shc, lam=1.0)
        assert w_a is not None
        assert w_a.sum() == pytest.approx(1.0, abs=1e-7)

    def test_the_augmented_program_still_requires_its_arguments(self):
        L, target = _problem()
        with pytest.raises(ValueError, match="lam and w_shc"):
            solve_shc_qp(L, target, use_augmented=True)
