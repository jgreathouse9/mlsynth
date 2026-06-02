"""Bilevel SCM optimization.

A self-contained implementation of the optimistic bilevel program for jointly
optimizing predictor weights ``V`` and donor weights ``W``, used as a drop-in
replacement for ``Opt.SCopt`` inside FSCM's predictor mode. No external QP
solver is used -- the lower-level problems are solved by the FISTA
simplex-least-squares primitive in :mod:`simplex`.

Two interchangeable backends are available via ``solve_bilevel(..., method=)``:

* ``"malo"`` (default) -- Malo, Eskelinen, Zhou & Kuosmanen (2024): staged
  corner search with an early optimality certificate.
* ``"mscmt"`` -- Becker & Kloessner (2018): global differential-evolution
  search over ``log10(V)`` (the MSCMT outer optimisation).
"""

from .structure import BilevelProblem, BilevelSolution
from .simplex import project_simplex, simplex_lstsq, mspe
from .solver import solve_bilevel, lower_level_weights
from .mscmt import solve_mscmt

__all__ = [
    "BilevelProblem",
    "BilevelSolution",
    "project_simplex",
    "simplex_lstsq",
    "mspe",
    "solve_bilevel",
    "solve_mscmt",
    "lower_level_weights",
]
