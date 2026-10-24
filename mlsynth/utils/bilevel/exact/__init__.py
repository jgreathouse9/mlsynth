"""The exact synthetic control of Abadie and Shang (2026), VanillaSC's ``backend="exact"``.

Abadie and Shang, "Synthetic Control: An Exact Geometric Formulation" (2026), replace
the bilevel choice of predictor weights V with a search over the geometry of the donor
hull. The donor weights are the best validation fit over every weight vector some
admissible V selects, the exact optimum of the bilevel program, and the predictor
weights are then a canonical member of the set of V that select it.

- ``search``: the donor weights. A treated unit outside the donor hull is matched on a
  face of the facets visible from it (the exterior case); one inside the hull is
  matched exactly, and the vanishing penalty selects among the exact matches
  (``interior``, the interior case).
- ``rationalize``: the predictor weights that select given donor weights, diagonal or
  positive definite: the log-det maximiser, the analytic centre and, for the positive
  definite class, the identified range of each weight.
- ``hull``, ``patterns``, ``lsq``: the facets of the donor hull, the sign patterns and
  attainability tests of a face, and the exact least-squares solves they use.

Everything is NumPy and SciPy; cvxpy is called only to decide the rare interior
attainability test that the barrier method leaves within rounding of its threshold.
"""

from .solve import V_CLASSES, solve_exact
from .structures import DiagonalRationalizers, ExactDetails, FaceCounts, PDRationalizers

__all__ = ["V_CLASSES", "solve_exact", "DiagonalRationalizers", "ExactDetails", "FaceCounts", "PDRationalizers"]
