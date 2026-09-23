"""Mode-aware optimality certificate for the SYNDES design.

The SYNDES design MIP is NP-hard; SCIP's cost is almost entirely in *proving*
optimality, because its dual bound is the continuous (McCormick) relaxation,
which is very loose for the two-way objective. This module supplies a valid
lower bound on the optimal objective *without* branch-and-bound, so a fitted
design can be reported with a validated optimality gap (``LB <= optimum <=
incumbent``) -- the rigorous form of "close enough, stop".

The right lower bound depends on the mode's geometry:

* ``global_equal_weights`` (one-way): the treated weights are pinned to ``1/K``,
  so ``D`` enters the residual linearly -- no bilinear term -- and the plain
  continuous relaxation is already tight (a single convex QP).
* ``global_2way`` (two-way): the ``q = w*D`` bilinear makes the continuous
  relaxation useless (~80% gap). Naming the treated set removes the bilinear
  entirely (see :mod:`mlsynth.utils.syndes_helpers.gram`), and the closed form
  that follows admits a Rayleigh bound over every size-``K`` design at once:
  ``f* >= alpha N / (K (N - K) lam_max(R))``. One matrix inverse and one
  eigenvalue, no relaxation solve, and no size gate.
* ``per_unit``: the weights are an ``(N, N)`` matrix, so no lift is tractable
  and the continuous relaxation is very loose (~70%). There is no cheap tight
  bound; the certificate is returned ``certified=False``.

The returned lower bound is always *valid* (a relaxation optimum), modulo the
solver's numerical tolerance; ``certified`` flags whether it is tight enough to
be a useful certificate for the mode.

The two-way bound replaced an SDP / moment (Shor--Lasserre level-1) lift, which
added ``D_i^2 = D_i`` and ``w_i D_i = q_i`` as exact second moments. Measured on
the same instances, the Rayleigh bound reached 88.8 / 90.6 / 92.0 percent of the
optimum at ``K = 3 / 5 / 7`` against the lift's 83.2 / 84.9 / 86.2, taking about
0.1 ms against 0.1--0.23 s, with no size gate and no iteration cap to exhaust.
The lift went with the MIP accelerator that was its last consumer.

This certificate is an mlsynth addition, not part of Doudchenko et al. (2021):
the paper proves the design NP-hard and gives the mixed-integer program but
derives no relaxation-based lower bound. It is a design-time diagnostic computed
on the pre-treatment panel and does not change the design SYNDES returns.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthConfigError
from .formulation import build_syndes_problem_components
from .gram import BoundEngine, TwoWayProblem
from .optimization import estimate_lambda

_ONE_WAY = "global_equal_weights"
_TWO_WAY = "global_2way"
_PER_UNIT = "per_unit"
_MODES = (_ONE_WAY, _TWO_WAY, _PER_UNIT)


@dataclass(frozen=True)
class SYNDESCertificate:
    """A validated optimality gap for a fitted SYNDES design.

    Attributes
    ----------
    lower_bound : float or None
        A valid lower bound on the optimal objective (``None`` only if the
        bound solve failed).
    optimality_gap : float or None
        ``(incumbent - lower_bound) / |incumbent|`` clamped at 0 -- the fraction
        by which the incumbent design may exceed the true optimum. ``None`` when
        there is no bound.
    certified : bool
        Whether ``lower_bound`` is tight enough to be a meaningful certificate
        for this mode (True for one-way and two-way; False for per-unit, where
        the bound is valid but loose).
    method : str
        ``"continuous_relaxation"`` or ``"rayleigh"``.
    note : str
        Human-readable caveat (empty when fully certified).
    """

    lower_bound: Optional[float]
    optimality_gap: Optional[float]
    certified: bool
    method: str
    note: str = ""


def _bound_value(objective: cp.Expression) -> Optional[float]:
    """The solved objective, or ``None`` when the solve produced no value.

    The continuous relaxation is a convex solve that can end without an optimum.
    CVXPY leaves ``.value`` at ``None`` in that case, so every caller reads the
    bound through this helper and gets an absent bound instead of a
    ``TypeError``.
    """
    value = objective.value
    return None if value is None else float(value)


def _continuous_relaxation_bound(
    Y: np.ndarray, K: int, mode: str, lam: float
) -> Optional[float]:
    """Continuous relaxation optimum (``D`` in ``[0, 1]``): a valid lower bound.

    ``None`` when the relaxation does not solve.
    """
    _, N = Y.shape
    D = cp.Variable(N, nonneg=True)
    comp = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam, mode=mode)
    prob = cp.Problem(cp.Minimize(comp.objective), list(comp.constraints) + [D <= 1])
    prob.solve(solver=cp.CLARABEL)
    return _bound_value(comp.objective)

def _rayleigh_bound_two_way(
    Y: np.ndarray, K: int, lam: float
) -> Optional[float]:
    """Closed-form lower bound on the two-way optimum over every size-``K`` design.

    Reduces the instance to its Gram matrix and applies Rayleigh's inequality to
    the cut form ``sigma' R sigma`` (see
    :mod:`mlsynth.utils.syndes_helpers.gram`). Costs one matrix inverse and one
    eigenvalue, needs no relaxation solve, and cannot fail to converge -- so the
    only way it returns ``None`` is an ill-conditioned Gram matrix, which happens
    when ``lam`` approaches zero on a panel with fewer pre-periods than units.
    """
    engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))
    return engine.global_lower_bound(K)


def _gap(incumbent: float, lb: float) -> float:
    denom = abs(incumbent) if abs(incumbent) > 1e-12 else 1.0
    return max(0.0, (incumbent - lb) / denom)


def _certificate(
    lb: Optional[float],
    incumbent_obj: float,
    certified: bool,
    method: str,
    note: str = "",
    absent_note: str = "",
) -> SYNDESCertificate:
    """Assemble a certificate, degrading to "no bound" when none is available.

    An absent bound is the documented ``lower_bound is None`` case: there is no
    gap to report and nothing is certified. ``absent_note`` overrides the default
    explanation for methods that go absent for a reason other than a solve that
    did not converge.
    """
    if lb is None:
        return SYNDESCertificate(
            None, None, False, method,
            note=absent_note or (
                f"the {method} bound solve did not converge, so no lower "
                "bound is available and this design carries no certified gap."))
    return SYNDESCertificate(lb, _gap(incumbent_obj, lb), certified, method, note=note)


def syndes_certificate(
    Y: np.ndarray,
    K: int,
    mode: str,
    incumbent_obj: float,
    *,
    lam: Optional[float] = None,
) -> SYNDESCertificate:
    """Certify a SYNDES design's optimality gap with a mode-appropriate bound.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix ``(T, N)`` -- the same matrix the design
        was optimized on.
    K : int
        Number of treated units.
    mode : {"global_equal_weights", "global_2way", "per_unit"}
        SYNDES formulation the design used.
    incumbent_obj : float
        Objective value of the fitted design (``SYNDESDesign.objective_value``).
    lam : float, optional
        Regularization; estimated from ``Y`` when ``None`` (must match the fit).

    Returns
    -------
    SYNDESCertificate
    """
    if mode not in _MODES:
        raise MlsynthConfigError(
            f"syndes_certificate: unknown mode {mode!r}; expected one of {_MODES}.")
    _, N = Y.shape
    lam_value = float(estimate_lambda(Y)) if lam is None else float(lam)

    if mode == _ONE_WAY:
        return _certificate(_continuous_relaxation_bound(Y, K, mode, lam_value),
                            incumbent_obj, True, "continuous_relaxation")

    if mode == _TWO_WAY:
        return _certificate(
            _rayleigh_bound_two_way(Y, K, lam_value),
            incumbent_obj, True, "rayleigh",
            absent_note=(
                "the two-way Gram matrix is too ill-conditioned for the "
                "closed-form bound, so no lower bound is available and this "
                "design carries no certified gap. Raise lam, or add "
                "pre-treatment periods."))

    # per_unit: no cheap tight bound (SDP is O(N^4); continuous relaxation ~70% loose)
    return _certificate(
        _continuous_relaxation_bound(Y, K, mode, lam_value),
        incumbent_obj, False, "continuous_relaxation",
        note=("per-unit has an (N,N) weight matrix, so the SDP lift is intractable "
              "and the continuous relaxation is loose; the gap is not a tight "
              "certificate."))
