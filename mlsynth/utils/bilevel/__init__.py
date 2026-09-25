"""Bilevel SCM optimization.

A self-contained implementation of the optimistic bilevel program for jointly
optimizing predictor weights ``V`` and donor weights ``W``, used as a drop-in
replacement for ``Opt.SCopt`` inside FSCM's predictor mode. No external QP
solver is used: the lower-level problems are solved by the active sets in
:mod:`mlsynth.utils.solvers.active_set` (one problem, design matrix at hand)
and :mod:`mlsynth.utils.solvers.minnorm` (a whole population at once, in Gram
form), with the FISTA primitive of :mod:`mlsynth.utils.solvers.simplex` for the
first-order paths.

Those solvers are general numerics and live in :mod:`mlsynth.utils.solvers`.
This package keeps only the bilevel program itself, and does not re-export
them: that re-export was a shim by another name, and it is what kept their
misfiling invisible at the call site. Import a solver from where it lives.

Two interchangeable backends are available via ``solve_bilevel(..., method=)``:

* ``"malo"`` (default) -- Malo, Eskelinen, Zhou & Kuosmanen (2024): staged
  corner search with an early optimality certificate.
* ``"mscmt"`` -- Becker & Kloessner (2018): global differential-evolution
  search over ``log10(V)`` (the MSCMT outer optimisation).
* ``"penalized"`` -- Abadie & L'Hour (2021): pairwise-penalized estimator with
  leave-one-out ``lambda`` selection and an optional bias correction.
"""

from .structure import BilevelProblem, BilevelSolution
from .solver import solve_bilevel, lower_level_weights
from .mscmt import solve_mscmt
from .regression_v import regression_v, solve_regression
from .penalized import bias_corrected_gaps, penalized_weights, solve_penalized
from .determine_v import (
    canonical_v,
    canonical_v_diagnostics,
    check_v,
    kkt_matrix,
    max_order_v,
    min_loss_w_v,
)
from .engine import BilevelSCM, BilevelSCMResult

__all__ = [
    "regression_v",
    "solve_regression",
    "BilevelProblem",
    "BilevelSolution",
    "BilevelSCM",
    "BilevelSCMResult",
    "solve_bilevel",
    "solve_mscmt",
    "solve_penalized",
    "penalized_weights",
    "bias_corrected_gaps",
    "lower_level_weights",
    "canonical_v",
    "canonical_v_diagnostics",
    "check_v",
    "kkt_matrix",
    "max_order_v",
    "min_loss_w_v",
]
