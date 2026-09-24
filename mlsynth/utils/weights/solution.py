"""What comes back from the solver.

A weight vector on its own does not say whether it is optimal, whether it is
the only optimum, or which method produced it. Those three facts decide how the
number downstream should be read, so they travel with the weights instead of
being recomputed -- or assumed -- at each call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
        Whether the minimiser is the only one. False means a continuum of weight
        vectors attains the same fit, so the individual weights carry no
        interpretation even though the counterfactual does.
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float).ravel())
        object.__setattr__(self, "weights", self.weights.copy())
        self.weights.setflags(write=False)
        object.__setattr__(self, "support", np.asarray(self.support, dtype=int).ravel())
        self.support.setflags(write=False)

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
            "support_size": int(self.support.size),
            "intercept": self.intercept,
        }
