"""Each migrated simplex solve returns what its cvxpy program returned.

``tools/simplex_qp_audit.py`` says which cvxpy problems are the program
``solve_simplex_qp`` solves. Moving one onto the active set is a claim that
the two agree, and a claim about a solver is only as good as the panel it was
checked on: a swap that agrees on one well-conditioned example and diverges on
a degenerate one has not been checked.

So each migrated helper is called here on random panels, against the cvxpy
program it replaced, written out in the case so the comparison does not read
the implementation it is testing. The weights are the headline. The objective
is what settles a disagreement, because a program with a flat direction has
more than one minimiser and two solvers may return different ones: the
question is then whether the active set reached the same value, not whether it
reached the same point.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import cvxpy as cp
import numpy as np
import pytest

Case = Callable[[np.random.Generator],
                Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], float]]]
CASES: Dict[str, Case] = {}


def case(name: str) -> Callable[[Case], Case]:
    def register(fn: Case) -> Case:
        CASES[name] = fn
        return fn
    return register


def _panel(rng: np.random.Generator, rows=(8, 25), cols=(3, 12), scale=1.0):
    """A donor block and a treated series inside its hull, plus noise."""
    T = int(rng.integers(*rows))
    J = int(rng.integers(*cols))
    X = scale * rng.normal(size=(T, J))
    return X, X @ rng.dirichlet(np.ones(J)) + 0.05 * scale * rng.normal(size=T)


def _cvxpy_simplex(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = cp.Variable(X.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(y - X @ w)),
               [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    return np.asarray(w.value, dtype=float).ravel()


def _lsq(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda v: float((y - X @ v) @ (y - X @ v))


@case("clustersc/spannability::_best_convex_fit")
def _spannability(rng):
    from mlsynth.utils.clustersc_helpers.spannability import _best_convex_fit

    X, y = _panel(rng, scale=float(rng.choice([1.0, 1e4])))
    scale = float(np.sqrt(np.mean(np.square(X))))
    got, _ = _best_convex_fit(X, y, scale)
    return got, _cvxpy_simplex(X / scale, y / scale), _lsq(X, y)


@case("drosc/estimation::sc_weights")
def _drosc(rng):
    from mlsynth.utils.drosc_helpers.estimation import sc_weights

    X, y = _panel(rng)
    got, _ = sc_weights(y, X)
    return got, _cvxpy_simplex(X, y), _lsq(X, y)


@case("dsc/weights::_refine_exact")
def _dsc(rng):
    from mlsynth.utils.dsc_helpers.weights import _refine_exact

    X, y = _panel(rng)
    warm = np.full(X.shape[1], 1.0 / X.shape[1])
    return _refine_exact(X, y, warm), _cvxpy_simplex(X, y), _lsq(X, y)


@case("dtwsc/pipeline::_simplex_weights")
def _dtwsc(rng):
    from mlsynth.utils.dtwsc_helpers.pipeline import _simplex_weights

    X, y = _panel(rng, cols=(2, 12))
    return _simplex_weights(y, X), _cvxpy_simplex(X, y), _lsq(X, y)


@case("inferutils::_outcome_only_simplex")
def _inferutils(rng):
    from mlsynth.utils.inferutils import _outcome_only_simplex

    X, y = _panel(rng)
    return _outcome_only_simplex(y, X), _cvxpy_simplex(X, y), _lsq(X, y)


@case("orthsc/gmm_sce::gmm_sc_weights")
def _gmm_sce(rng):
    """One-step GMM: the moment vector is the residual seen through ``YK'``."""
    from mlsynth.utils.orthsc_helpers.gmm_sce.solver import (
        _row_normalize, gmm_sc_weights)

    T0 = int(rng.integers(10, 25))
    J, K = int(rng.integers(3, 8)), int(rng.integers(3, 8))
    YJ = rng.normal(size=(T0, J))
    y0 = YJ @ rng.dirichlet(np.ones(J)) + 0.05 * rng.normal(size=T0)
    YK = rng.normal(size=(T0, K))
    got = np.asarray(gmm_sc_weights(y0, YJ, YK)["weights"], dtype=float).ravel()

    scaled = _row_normalize(np.hstack(
        [y0[:, None], YJ, np.hstack([np.ones((T0, 1)), YK])]))
    y0s, YJs, YKs = scaled[:, 0], scaled[:, 1:1 + J], scaled[:, 1 + J:]
    B, A = YKs.T @ YJs, YKs.T @ y0s
    return got, _cvxpy_simplex(B, A), _lsq(B, A)


@case("scmo/solvers::simplex_weights")
def _scmo(rng):
    from mlsynth.utils.scmo_helpers.solvers import simplex_weights

    Z, z = _panel(rng)
    return simplex_weights(z, Z.T), _cvxpy_simplex(Z, z), _lsq(Z, z)


@case("spotsynth/sc::simplex_weights")
def _spotsynth(rng):
    from mlsynth.utils.spotsynth_helpers.sc import simplex_weights

    T0 = int(rng.integers(8, 20))
    X, y = _panel(rng, rows=(T0 + 3, T0 + 4), scale=float(rng.choice([1.0, 1e6])))
    got, _ = simplex_weights(y, X, T0)
    scale = float(np.sqrt(np.mean(np.square(X[:T0]))))
    return (got, _cvxpy_simplex(X[:T0] / scale, y[:T0] / scale),
            _lsq(X[:T0], y[:T0]))


@case("ssc/weights::sc_weights_one")
def _ssc(rng):
    """Demeaned, which is how this site profiles out its intercept."""
    from mlsynth.utils.ssc_helpers.weights import sc_weights_one

    X, y = _panel(rng)
    y = y + 3.2
    _, got = sc_weights_one(y, X)
    Xd, yd = X - X.mean(axis=0), y - y.mean()
    return got, _cvxpy_simplex(Xd, yd), _lsq(Xd, yd)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_migrated_solve_agrees_with_the_cvxpy_program_it_replaced(name):
    rng = np.random.default_rng(20250924)
    for trial in range(6):
        got, ref, objective = CASES[name](rng)
        assert got.shape == ref.shape
        here, there = objective(got), objective(ref)
        excess = (here - there) / max(abs(there), 1e-12)
        assert excess <= 1e-8, (
            f"{name}, trial {trial}: the active set finished {excess:.3e} above "
            f"cvxpy's objective ({here:.6e} vs {there:.6e})"
        )


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_migrated_solve_returns_a_point_on_the_simplex(name):
    """The constraint is not the solver's to relax, whatever the panel."""
    rng = np.random.default_rng(1234567)
    for _ in range(6):
        got, _, _ = CASES[name](rng)
        assert got.min() >= -1e-12, got.min()
        assert abs(got.sum() - 1.0) <= 1e-8, got.sum()


def test_naming_a_solver_keeps_the_cvxpy_path_and_the_same_answer():
    """GMM-SCE's ``solver`` argument is public, so the swap cannot ignore it.

    MASC set the precedent: the default goes to the active set, and an
    explicitly requested cvxpy solver keeps the cvxpy path. Both have to
    return the same weights, or the argument changes the estimate instead of
    the route to it.
    """
    from mlsynth.utils.orthsc_helpers.gmm_sce.solver import gmm_sc_weights

    rng = np.random.default_rng(99)
    YJ = rng.normal(size=(18, 5))
    y0 = YJ @ rng.dirichlet(np.ones(5)) + 0.05 * rng.normal(size=18)
    YK = rng.normal(size=(18, 4))

    native = gmm_sc_weights(y0, YJ, YK)
    viacp = gmm_sc_weights(y0, YJ, YK, solver="SCS")
    assert np.max(np.abs(native["weights"] - viacp["weights"])) < 1e-4
    assert native["status"] == "optimal"
    assert viacp["status"] in ("optimal", "optimal_inaccurate")
