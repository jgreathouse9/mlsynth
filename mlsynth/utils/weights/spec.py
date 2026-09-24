"""What is being asked of the solver, separated from how it is answered.

Two frozen records. A :class:`WeightConstraint` names a polyhedron in weight
space; a :class:`WeightObjective` names the discrepancy minimised over it. Both
validate on construction, so a misspecified program fails where it is written
and not several frames down inside a backend.

The polyhedra are the ones the library already builds by hand in fourteen
places: the simplex (Abadie-Diamond-Hainmueller), the nonnegative cone
(Bayani's Eq. 1.30 and the robust-SC family), the affine hyperplane, the whole
space (Amjad's Theorem 4.2.1 is a span condition, not a hull condition), and
the two of those with a per-donor cap. The intercept is a field because the
distinction between a free intercept and one inside the simplex is a modelling
choice with consequences -- see :class:`WeightConstraint.intercept`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mlsynth.exceptions import MlsynthConfigError

#: Cressie-Read exponents this layer solves exactly. ``gamma = 1`` is the
#: quadratic member, whose programs are quadratic over a polyhedron and admit a
#: finite active-set method. ``gamma < 1`` (entropy at 0, empirical likelihood
#: at -1) is exponential-cone and is refused here by name.
QUADRATIC_DIVERGENCE = "quadratic"
_EXPONENTIAL_CONE = {"entropy": 0.0, "empirical_likelihood": -1.0}


@dataclass(frozen=True)
class WeightConstraint:
    """The polyhedron the weights are drawn from.

    Parameters
    ----------
    nonneg : bool
        Require ``w >= 0``. Dropping it admits extrapolation past the donor
        cloud, which is what separates a span condition from a hull condition.
    sum_to_one : bool
        Require ``sum(w) == 1``.
    upper : float, optional
        A common per-donor cap ``w <= upper``, used to bound the influence of
        any single donor. ``None`` leaves the weights uncapped above.
    intercept : bool
        Fit an additive level shift alongside the weights, unconstrained in
        sign. An intercept placed inside the simplex instead -- bounded to
        ``[0, 1]`` and competing with the donors for the unit budget -- cannot
        represent a treated unit lying below its donors at all, and silently
        distorts the weights when it tries.
    """

    nonneg: bool = True
    sum_to_one: bool = True
    upper: Optional[float] = None
    intercept: bool = False

    def __post_init__(self) -> None:
        if self.upper is not None:
            if not np.isfinite(self.upper) or self.upper <= 0.0:
                raise MlsynthConfigError(
                    f"upper must be a finite positive cap; got {self.upper!r}."
                )
            if self.sum_to_one and self.upper < 1.0 and not self.nonneg:
                raise MlsynthConfigError(
                    "An upper cap below 1 with sum_to_one and no non-negativity "
                    "describes a polyhedron this layer does not cover."
                )

    @property
    def shape(self) -> tuple[bool, bool, bool]:
        """The dispatch key: (non-negative, sums to one, capped above)."""
        return (self.nonneg, self.sum_to_one, self.upper is not None)

    def describe(self) -> str:
        """The polyhedron's usual name, for messages and for ``solver`` fields."""
        return {
            (True, True, False): "simplex",
            (True, False, False): "cone",
            (False, True, False): "affine",
            (False, False, False): "free",
            (True, False, True): "box",
            (False, False, True): "box",
            (True, True, True): "capped-simplex",
            (False, True, True): "capped-affine",
        }[self.shape]

    def contains(self, w: np.ndarray, *, tol: float = 1e-8) -> bool:
        """Whether ``w`` is feasible to within ``tol``."""
        w = np.asarray(w, dtype=float).ravel()
        if self.nonneg and w.min(initial=0.0) < -tol:
            return False
        if self.sum_to_one and abs(float(w.sum()) - 1.0) > tol:
            return False
        if self.upper is not None and w.max(initial=0.0) > self.upper + tol:
            return False
        return True


@dataclass(frozen=True)
class WeightObjective:
    """The discrepancy minimised over the polyhedron.

    Parameters
    ----------
    ridge : float
        Coefficient on ``||w - toward||^2``. Zero leaves the plain least-squares
        fit. A positive value makes the program strictly convex, so the
        minimiser is unique whatever the donor rank.
    toward : np.ndarray, optional
        The shrinkage target. ``None`` means uniform weights ``1_J / J``, which
        is the Bregman reference point the Cressie-Read family is built around.
    divergence : str
        Which Cressie-Read member. Only ``"quadratic"`` (``gamma = 1``) is
        solved here; the others are exponential-cone programs and raise.
    """

    ridge: float = 0.0
    toward: Optional[np.ndarray] = None
    divergence: str = QUADRATIC_DIVERGENCE

    def __post_init__(self) -> None:
        if self.divergence != QUADRATIC_DIVERGENCE:
            if self.divergence in _EXPONENTIAL_CONE:
                gamma = _EXPONENTIAL_CONE[self.divergence]
                raise MlsynthConfigError(
                    f"divergence={self.divergence!r} is the Cressie-Read member at "
                    f"gamma={gamma}, an exponential cone program. This layer solves "
                    f"gamma=1 exactly over polyhedra; build the cone program with "
                    f"cvxpy instead."
                )
            raise MlsynthConfigError(
                f"Unknown divergence {self.divergence!r}; expected "
                f"{QUADRATIC_DIVERGENCE!r}."
            )
        if not np.isfinite(self.ridge) or self.ridge < 0.0:
            raise MlsynthConfigError(
                f"ridge must be a finite non-negative coefficient; got {self.ridge!r}."
            )
        if self.toward is not None:
            tgt = np.asarray(self.toward, dtype=float).ravel()
            if tgt.size == 0 or not np.all(np.isfinite(tgt)):
                raise MlsynthConfigError(
                    "toward must be a finite, non-empty shrinkage target."
                )
            object.__setattr__(self, "toward", tgt)

    def target(self, n_donors: int) -> np.ndarray:
        """The shrinkage target for ``n_donors`` donors, defaulting to uniform."""
        if self.toward is None:
            return np.full(n_donors, 1.0 / n_donors)
        if self.toward.size != n_donors:
            raise MlsynthConfigError(
                f"toward has {self.toward.size} entries but there are {n_donors} donors."
            )
        return self.toward
