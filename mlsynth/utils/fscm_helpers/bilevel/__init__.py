"""Bilevel SCM optimization (Malo, Eskelinen, Zhou & Kuosmanen 2024).

A self-contained implementation of the optimistic bilevel program for jointly
optimizing predictor weights ``V`` and donor weights ``W``, used as a drop-in
replacement for ``Opt.SCopt`` inside FSCM's predictor mode. No external QP
solver is used -- the lower-level problems are solved by the FISTA
simplex-least-squares primitive in :mod:`simplex`.
"""

from .structure import BilevelProblem, BilevelSolution
from .simplex import project_simplex, simplex_lstsq, mspe
from .solver import solve_bilevel, lower_level_weights

__all__ = [
    "BilevelProblem",
    "BilevelSolution",
    "project_simplex",
    "simplex_lstsq",
    "mspe",
    "solve_bilevel",
    "lower_level_weights",
]
