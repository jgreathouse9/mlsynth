"""What comes back from the solver.

A weight vector on its own does not say whether it is optimal, whether it is
the only optimum, or which method produced it. Those three facts decide how the
number downstream should be read, so they travel with the weights instead of
being recomputed -- or assumed -- at each call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlsynth.exceptions import MlsynthEstimationError


@dataclass(frozen=True)
class WeightSolution:
    """An optimal weight vector and its certificate.

    Attributes
    ----------
    weights : np.ndarray
        The minimiser, shape ``(J,)``.
    intercept : float
        The fitted level shift, ``0.0`` when the constraint set has none.
    objective : float
        The attained value of the full objective, ridge penalty included.
    kkt_residual : float
        Scale-free violation of the Karush-Kuhn-Tucker conditions, recomputed
        from ``weights`` alone. Slater's condition holds on every polyhedron
        here, so KKT is necessary and sufficient (Boyd and Vandenberghe 2004,
        section 5.5.3) and this number is a proof of optimality that does not
        depend on what the backend reported about itself.
    unique : bool
        Whether the minimiser is the only one, i.e. whether ``free_directions``
        is empty. False means a continuum of weight vectors attains the same
        fit, so the individual weights carry no interpretation.

        It says nothing on its own about whether an estimate built from the
        weights is identified. A duplicated donor makes the optimum a continuum
        and leaves every counterfactual untouched, because the direction weight
        moves along is annihilated wherever the twins stay identical. Donors
        collinear only before treatment make the same verdict mean the opposite.
        Which of the two holds is a question about the periods being predicted,
        so it is asked of them, with :meth:`identifies`.
    free_directions : np.ndarray
        Orthonormal basis, shape ``(J, k)``, of the directions the weights can
        move along at no cost while staying feasible. Empty when the minimiser
        is unique.
    free_intercepts : np.ndarray
        The intercept component of each free direction, shape ``(k,)``; zeros
        when the constraint set has no intercept.
    solver : str
        Which backend ran, as ``"<polyhedron>:<method>"``.
    status : str
        ``"optimal"`` when the certificate is within tolerance, otherwise
        ``"inaccurate"``.
    n_donors : int
        Number of donor columns.
    support : np.ndarray
        Indices of the donors carrying weight.
    """

    weights: np.ndarray
    intercept: float
    objective: float
    kkt_residual: float
    unique: bool
    solver: str
    status: str
    n_donors: int
    support: np.ndarray = field(repr=False)
    free_directions: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    free_intercepts: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float).ravel())
        object.__setattr__(self, "weights", self.weights.copy())
        self.weights.setflags(write=False)
        object.__setattr__(self, "support", np.asarray(self.support, dtype=int).ravel())
        self.support.setflags(write=False)

    def identifies(self, B: np.ndarray, *, tol: float = 1e-8) -> bool:
        """Whether ``B @ w + intercept`` is the same for every minimiser.

        The weights are solved on the pre-treatment periods, so ``unique``
        rules on those periods alone. An ATT is built from the periods after
        treatment, and a continuum in the weights reaches it only if the
        post-treatment design fails to annihilate the directions weight is free
        to move along. Pass the post-treatment donor block here to ask that.

        True on a unique solution, and True whenever the continuum exists but
        cancels -- duplicated donors being the case where it does.
        """
        B = np.asarray(B, dtype=float)
        if B.ndim != 2 or B.shape[1] != self.n_donors:
            raise MlsynthEstimationError(
                f"Expected a design with {self.n_donors} columns; got shape {B.shape}."
            )
        if self.free_directions.shape[1] == 0:
            return True
        shift = B @ self.free_directions + self.free_intercepts
        scale = max(float(np.abs(B).max(initial=0.0)), 1e-300)
        return bool(np.abs(shift).max(initial=0.0) <= tol * scale)

    def fitted(self, B: np.ndarray) -> np.ndarray:
        """The synthetic series ``B @ w + intercept`` for any period block."""
        return np.asarray(B, dtype=float) @ self.weights + self.intercept

    def to_dict(self) -> dict:
        """A flat mapping for the ``MethodDetailsResults`` bag on a result."""
        return {
            "solver": self.solver,
            "status": self.status,
            "objective": self.objective,
            "kkt_residual": self.kkt_residual,
            "weights_unique": self.unique,
            "free_directions": int(self.free_directions.shape[1]),
            "support_size": int(self.support.size),
            "intercept": self.intercept,
        }
