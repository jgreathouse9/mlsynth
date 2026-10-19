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

The returned lower bound is *valid* (a relaxation optimum, or a closed form),
modulo the solver's numerical tolerance, and only a relaxation that solved to
``optimal`` gives one. The incumbent is the returned design scored on the data
and penalty the bound uses (:func:`design_objective`), so the two refer to one
problem. ``certified`` says the gap can be read as a certificate: the bound
exists, is tight enough for the mode to matter, and does not exceed the
incumbent beyond the bound's own tolerance.

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

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

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

#: A bound may sit above the incumbent by this much, relative to the larger of
#: the two, before the excess counts as a contradiction: the tolerance of the
#: relaxation's solve. Within it the gap is zero.
_BOUND_TOLERANCE = 1e-8


class SYNDESCertificateWarning(UserWarning):
    """A SYNDES certificate's lower bound exceeds the design it bounds."""


@dataclass(frozen=True)
class SYNDESCertificate:
    """A validated optimality gap for a fitted SYNDES design.

    Attributes
    ----------
    lower_bound : float or None
        A valid lower bound on the optimal objective. ``None`` when there is
        none: the relaxation did not solve to ``optimal``, or the two-way Gram
        matrix is too ill-conditioned for the closed form.
    optimality_gap : float or None
        ``(incumbent - lower_bound) / max(|incumbent|, |lower_bound|)``, the
        fraction by which the incumbent may exceed the true optimum. Zero when
        the bound exceeds the incumbent within the bound's tolerance; negative,
        with ``certified=False`` and a :class:`SYNDESCertificateWarning`, when
        it exceeds it by more. ``None`` when there is no bound.
    certified : bool
        Whether the gap can be read as a certificate: the bound exists, does not
        contradict the incumbent, and is tight enough to matter for the mode
        (one-way and two-way; per-unit's bound is valid but loose).
    method : str
        ``"continuous_relaxation"`` or ``"rayleigh"``.
    note : str
        Human-readable caveat (empty when fully certified).
    incumbent : float or None
        The objective the gap is measured from: the returned design scored on
        the bound's data and penalty.
    """

    lower_bound: Optional[float]
    optimality_gap: Optional[float]
    certified: bool
    method: str
    note: str = ""
    incumbent: Optional[float] = None


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
) -> Tuple[Optional[float], str]:
    """Continuous relaxation optimum (``D`` in ``[0, 1]``): a valid lower bound.

    Valid only at an optimum: an inexact relaxation's objective can sit above the
    relaxation's minimum, and then it bounds nothing. So any status but
    ``optimal``, and any solver error, gives no bound and says why.
    """
    _, N = Y.shape
    D = cp.Variable(N, nonneg=True)
    comp = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam, mode=mode)
    prob = cp.Problem(cp.Minimize(comp.objective), list(comp.constraints) + [D <= 1])
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as exc:
        return None, f"the continuous relaxation's solver failed ({exc})"
    if prob.status != "optimal":
        return None, (f"the continuous relaxation failed to solve to optimality "
                      f"(status {prob.status!r})")
    value = _bound_value(comp.objective)
    if value is None:                       # pragma: no cover - optimal carries a value
        return None, "the continuous relaxation failed to return a value"
    return value, ""

def _rayleigh_bound_two_way(
    Y: np.ndarray, K: Optional[int], lam: float
) -> Optional[float]:
    """Closed-form lower bound on the two-way optimum over every size-``K`` design.

    Reduces the instance to its Gram matrix and applies Rayleigh's inequality to
    the cut form ``sigma' R sigma`` (see
    :mod:`mlsynth.utils.syndes_helpers.gram`). Costs one matrix inverse and one
    eigenvalue, needs no relaxation solve, and cannot fail to converge -- so the
    only way it returns ``None`` is an ill-conditioned Gram matrix, which happens
    when ``lam`` approaches zero on a panel with fewer pre-periods than units.

    With ``K=None`` the design searches every size, so the bound is the smallest
    over sizes ``1 .. N - 1``.
    """
    engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))
    if K is not None:
        return engine.global_lower_bound(K)
    bounds = [engine.global_lower_bound(k) for k in range(1, engine.N)]
    return None if any(b is None for b in bounds) else float(min(bounds))


def _gap(incumbent: float, lb: float) -> float:
    """``(incumbent - lb) / max(|incumbent|, |lb|)``; zero when both are zero.

    Relative to the larger of the two, with no floor, so the gap does not depend
    on the units of the outcome.
    """
    scale = max(abs(float(incumbent)), abs(float(lb)))
    if scale == 0.0:
        return 0.0
    return (float(incumbent) - float(lb)) / scale


def design_objective(Y: np.ndarray, mode: str, design: Any, lam: float) -> float:
    """The mode's objective at a design's weights, on the panel ``Y``.

    The weights are feasible for the mode's program on any panel, so this is an
    upper bound on that panel's optimum: the incumbent a lower bound on the same
    panel and penalty is compared with.

    Two-way and one-way, with treated weights ``a`` and control weights ``b``:
    ``(1/T) ||Y (a - b)||^2 + lam (||a||^2 + ||b||^2)``. Per-unit, with ``Q``
    the treated units' control rows:
    ``(1 / (K T)) sum_{i in S} ||y_i - Y q_i||^2 + (lam / K) ||Q_S||^2``.
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    if mode == _PER_UNIT:
        S = np.flatnonzero(np.asarray(design.assignment) == 1)
        Q = np.asarray(design.treated_weights, dtype=float)[S]
        resid = Y[:, S] - Y @ Q.T
        return float((resid ** 2).sum() / (len(S) * T) + lam / len(S) * (Q ** 2).sum())
    a = np.asarray(design.treated_weights, dtype=float).reshape(-1)
    b = np.asarray(design.control_weights, dtype=float).reshape(-1)
    r = Y @ (a - b)
    return float(r @ r / T + lam * (a @ a + b @ b))


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
                "bound is available and this design carries no certified gap."),
            incumbent=float(incumbent_obj))
    gap = _gap(incumbent_obj, lb)
    if gap < 0.0:
        if gap >= -_BOUND_TOLERANCE:
            gap = 0.0
        else:
            certified = False
            note = (f"the lower bound {lb:.6g} exceeds the incumbent "
                    f"{incumbent_obj:.6g} by {-gap:.3g} of the larger, beyond the "
                    "bound's tolerance: the bound or the incumbent is wrong, and "
                    "the gap certifies nothing.")
            warnings.warn(f"SYNDES certificate: {note}", SYNDESCertificateWarning,
                          stacklevel=3)
    return SYNDESCertificate(lb, gap, certified, method, note=note,
                             incumbent=float(incumbent_obj))


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
        The design's objective on ``Y`` with penalty ``lam``
        (:func:`design_objective`). The bound and the incumbent must refer to
        the same panel and penalty: a design fitted on part of the pre-period
        (holdout selection) is scored on the whole of it here.
    lam : float, optional
        Regularization; estimated from ``Y`` when ``None``. Pass the penalty the
        incumbent was scored with.

    Returns
    -------
    SYNDESCertificate
    """
    if mode not in _MODES:
        raise MlsynthConfigError(
            f"syndes_certificate: unknown mode {mode!r}; expected one of {_MODES}.")
    if K is None and mode != _TWO_WAY:
        raise MlsynthConfigError(
            f"syndes_certificate: K=None is bounded for the two-way mode only; "
            f"mode {mode!r} needs K.")
    _, N = Y.shape
    lam_value = float(estimate_lambda(Y)) if lam is None else float(lam)

    if mode == _ONE_WAY:
        lb, why = _continuous_relaxation_bound(Y, K, mode, lam_value)
        return _certificate(lb, incumbent_obj, True, "continuous_relaxation",
                            absent_note=_absent(why))

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
    lb, why = _continuous_relaxation_bound(Y, K, mode, lam_value)
    return _certificate(
        lb, incumbent_obj, False, "continuous_relaxation",
        note=("per-unit has an (N,N) weight matrix, so the SDP lift is intractable "
              "and the continuous relaxation is loose; the gap is not a tight "
              "certificate."),
        absent_note=_absent(why))


def _absent(why: str) -> str:
    return (f"{why}, so no lower bound is available and this design carries no "
            "certified gap.")
