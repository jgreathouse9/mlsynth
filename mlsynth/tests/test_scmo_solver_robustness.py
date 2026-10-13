"""Robustness of SCMO's simplex weight solver to a failed solve.

The regression this file was written for was cvxpy-shaped: OSQP terminated
without a primal solution, ``w.value`` stayed ``None``, and it went straight
into ``np.clip``, which raised an opaque ``TypeError`` (``'>=' not supported
between instances of 'NoneType' and 'float'``). The answer at the time was to
try CLARABEL as well and translate the error if both failed.

``simplex_weights`` is on the active set now, under the ridge its reference
program carries (see ``test_scmo_reference_ridge.py``). That returns a point on
the simplex or raises, so there is no ``None`` to clip and no second solver to
fall back to. The two tests pinning the fallback chain are gone with it.

What survives is the invariant the chain existed to protect: a failed solve
reaches the caller as a translated ``MlsynthEstimationError`` naming what to
look at, and a solve that succeeds returns a point on the simplex.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.scmo_helpers import solvers


def _problem():
    """A small, well-posed matching problem (treated is a convex mix of donors)."""
    Z_donors = np.array(
        [[1.0, 0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 1.0]]
    )
    # treated = 0.5 * donor0 + 0.5 * donor1 -> a genuine simplex target
    Z_treated = 0.5 * Z_donors[0] + 0.5 * Z_donors[1]
    return Z_treated, Z_donors


def test_simplex_weights_smoke_returns_valid_simplex():
    Z_treated, Z_donors = _problem()
    w = solvers.simplex_weights(Z_treated, Z_donors)
    assert w.shape == (Z_donors.shape[0],)
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_a_failed_solve_is_reported_as_a_translated_error():
    """Not a raw exception, and not a ``TypeError`` from clipping ``None``."""
    Z_treated, Z_donors = _problem()
    with mock.patch(
        "mlsynth.utils.scmo_helpers.solvers.solve_simplex_qp_least_norm",
        side_effect=RuntimeError("singular design"),
    ):
        with pytest.raises(MlsynthEstimationError, match="degenerate or ill-conditioned"):
            solvers.simplex_weights(Z_treated, Z_donors)


def test_a_degenerate_matching_matrix_still_returns_a_simplex_point():
    """Duplicated and collinear donors are where OSQP used to give up."""
    Z_donors = np.array([[1.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],      # an exact duplicate
                         [0.0, 1.0, 0.0],
                         [2.0, 0.0, 0.0]])     # collinear with the first
    Z_treated = np.array([0.5, 0.5, 0.0])
    w = solvers.simplex_weights(Z_treated, Z_donors)
    assert w.shape == (4,)
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_the_duplicated_donors_are_given_the_same_weight():
    """Donors 0 and 1 are identical, so any split of their shared mass is
    optimal and the returned one is even. This holds with or without the ridge
    -- the solve starts from the uniform weights and nothing here breaks the
    symmetry -- so it pins the behaviour and is not evidence about the ridge.
    The check that separates the two programs is in
    ``test_scmo_reference_ridge.py``, on a one-column match."""
    Z_donors = np.array([[1.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])
    Z_treated = np.array([0.5, 0.5, 0.0])
    w = solvers.simplex_weights(Z_treated, Z_donors)
    assert w[0] == pytest.approx(w[1], abs=1e-9)
