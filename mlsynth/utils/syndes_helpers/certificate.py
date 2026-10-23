"""Optimality certificate for a SYNDES design, from the exact searches' bounds.

Every SYNDES mode has an exact backend that searches treated sets and reports a
lower bound on the optimum with its design (``design.solve_outcome``):

* ``global_2way`` (:mod:`~.exact`): the enumeration under the closed-form bound
  of :mod:`~.gram`, which reports the Rayleigh bound over every size-``K``
  design, or the smallest priced bound where ``G + lam I`` is singular; above
  its candidate limit the branch and bound of :mod:`~.branch`, which reports
  the optimum when it closes and its smallest open node bound at its work
  limit;
* ``global_equal_weights`` (:mod:`~.one_way_exact`): the enumeration under the
  ridge bound, which reports the optimum itself; above its candidate limit the
  same branch and bound with one-way's leaves, which reports the optimum when
  it closes and its smallest open bound at its work limit;
* ``per_unit`` (:mod:`~.per_unit_exact`): the branch and bound on units, which
  reports the optimum when it closes and its smallest open node bound at its node
  limit.

The certificate is ``lower_bound <= optimum <= incumbent``. The bound comes from
the mode's search on the panel, penalty and restrictions the incumbent is scored
on: the design's own search when the caller passes it (``search``), and a fresh
search otherwise, which is what a design fitted on part of the pre-period
(holdout selection) needs. The incumbent is the returned design scored on that
panel and penalty (:func:`design_objective`), so the two refer to one problem.
``certified`` says the search certified its design, its bound reaching the
design, and that bound does not contradict the incumbent beyond
:data:`_BOUND_TOLERANCE`.

The certificate is an mlsynth addition, not part of Doudchenko et al. (2021): the
paper proves the design NP-hard and gives the mixed-integer program but derives
no lower bound. It is a design-time diagnostic computed on the pre-treatment
panel and does not change the design SYNDES returns.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from .optimization import estimate_lambda

_ONE_WAY = "global_equal_weights"
_TWO_WAY = "global_2way"
_PER_UNIT = "per_unit"
_MODES = (_ONE_WAY, _TWO_WAY, _PER_UNIT)

#: A bound may sit above the incumbent by this much, relative to the larger of
#: the two, before the excess counts as a contradiction: the tolerance the
#: searches certify their designs to. Within it the gap is zero.
_BOUND_TOLERANCE = 1e-8


class SYNDESCertificateWarning(UserWarning):
    """A SYNDES certificate's lower bound exceeds the design it bounds."""


@dataclass(frozen=True)
class SYNDESCertificate:
    """A validated optimality gap for a fitted SYNDES design.

    Attributes
    ----------
    lower_bound : float or None
        A valid lower bound on the optimal objective: the bound the mode's exact
        search reported. ``None`` when the search raised, which the note
        explains.
    optimality_gap : float or None
        ``(incumbent - lower_bound) / max(|incumbent|, |lower_bound|)``, the
        fraction by which the incumbent may exceed the true optimum. Zero when
        the bound exceeds the incumbent within the bound's tolerance; negative,
        with ``certified=False`` and a :class:`SYNDESCertificateWarning`, when
        it exceeds it by more. ``None`` when there is no bound.
    certified : bool
        Whether the gap can be read as a certificate: the search closed, and
        its bound does not contradict the incumbent.
    method : str
        ``"search"``: the bound is the mode's exact search's.
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


def _search(Y: np.ndarray, K: Optional[int], mode: str, lam: float,
            restrictions: Any, costs: Any, budget: Any) -> Any:
    """The mode's exact search on ``Y``. A search stopped at its limit warns,
    as it does for the design, and the certificate says so in its note."""
    from .exact import solve_synthetic_design_exact
    from .one_way_exact import solve_one_way_exact
    from .per_unit_exact import solve_per_unit_exact
    solve = {_TWO_WAY: solve_synthetic_design_exact,
             _ONE_WAY: solve_one_way_exact,
             _PER_UNIT: solve_per_unit_exact}[mode]
    return solve(Y, K, lam=lam, restrictions=restrictions, costs=costs,
                 budget=budget)


def _bound(search: Any) -> Tuple[Optional[float], bool, str]:
    """A search's lower bound, whether it closed, and why not if it did not."""
    outcome = getattr(search, "solve_outcome", None)
    if outcome is not None and outcome.solver_status == "exact":
        closed = outcome.status == "optimal"
        why = "" if closed else (
            f"the search stopped at its limit (status {outcome.status!r}), so "
            "its bound is valid but the design is not proven optimal.")
        return outcome.dual_bound, closed, why
    raw = getattr(search, "raw_results", None) or {}
    if raw.get("solver") != "exact" or raw.get("lower_bound") is None:
        raise MlsynthConfigError(
            "syndes_certificate: search must be a design from one of SYNDES's "
            "exact backends, which report a lower bound.")
    closed = bool(raw.get("certified"))
    why = "" if closed else (
        "the search did not certify its design, so its bound is valid but the "
        "design is not proven optimal.")
    return float(raw["lower_bound"]), closed, why


def syndes_certificate(
    Y: np.ndarray,
    K: Optional[int],
    mode: str,
    incumbent_obj: float,
    *,
    lam: Optional[float] = None,
    restrictions: Optional[Any] = None,
    costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    search: Optional[Any] = None,
) -> SYNDESCertificate:
    """Certify a SYNDES design's optimality gap with its mode's search bound.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix ``(T, N)``: the panel the incumbent is
        scored on.
    K : int or None
        Number of treated units. ``None`` searches every size, which only the
        two-way mode does.
    mode : {"global_equal_weights", "global_2way", "per_unit"}
        SYNDES formulation the design used.
    incumbent_obj : float
        The design's objective on ``Y`` with penalty ``lam``
        (:func:`design_objective`).
    lam : float, optional
        Penalty; estimated from ``Y`` when ``None``. Pass the penalty the
        incumbent was scored with.
    restrictions, costs, budget : optional
        The design's restrictions, so the bound is the restricted problem's.
        Without them it is the unrestricted problem's, which bounds the
        restricted one too, more loosely.
    search : SYNDESDesign, optional
        A design the mode's exact backend returned on ``Y``, ``K``, ``lam`` and
        the same restrictions; its bound is read instead of searching again.

    Returns
    -------
    SYNDESCertificate

    Raises
    ------
    MlsynthConfigError
        For an unknown mode, a ``search`` that carries no exact-search bound,
        or a ``K`` the mode's search refuses (``K=None`` outside the two-way
        mode among them).
    """
    if mode not in _MODES:
        raise MlsynthConfigError(
            f"syndes_certificate: unknown mode {mode!r}; expected one of {_MODES}.")
    lam_value = float(estimate_lambda(Y)) if lam is None else float(lam)
    if search is None:
        try:
            search = _search(Y, K, mode, lam_value, restrictions, costs, budget)
        except MlsynthEstimationError as exc:
            return _certificate(
                None, incumbent_obj, False, "search",
                absent_note=(f"the {mode} search raised ({exc}), so no lower "
                             "bound is available and this design carries no "
                             "certified gap."))
    lb, closed, why = _bound(search)
    return _certificate(lb, incumbent_obj, closed, "search", note=why)
