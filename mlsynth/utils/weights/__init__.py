"""The library's weight programs, behind one entry point.

Every estimator here that picks donor weights is solving the same thing: a
convex quadratic over a polyhedron in weight space. The polyhedron differs --
simplex, cone, affine hyperplane, the whole space, any of them capped -- and
almost nothing else does. This package makes the polyhedron an argument and the
solver a table lookup, so a call site names the econometrics and nothing else.

    >>> from mlsynth.utils.weights import WeightConstraint, solve_weights
    >>> sol = solve_weights(donors, treated)                       # simplex
    >>> sol = solve_weights(donors, treated, WeightConstraint(sum_to_one=False))
    >>> sol.unique, sol.kkt_residual                    # doctest: +SKIP

The design and its sources are in ``agents/agents_solver.md``.
"""
from .solution import WeightSolution
from .solve import KKT_TOL, SUPPORT_TOL, WEAK_ACTIVE_TOL, kkt_residual, solve_weights
from .spec import QUADRATIC_DIVERGENCE, WeightConstraint, WeightObjective

__all__ = [
    "KKT_TOL",
    "QUADRATIC_DIVERGENCE",
    "SUPPORT_TOL",
    "WEAK_ACTIVE_TOL",
    "WeightConstraint",
    "WeightObjective",
    "WeightSolution",
    "kkt_residual",
    "solve_weights",
]
