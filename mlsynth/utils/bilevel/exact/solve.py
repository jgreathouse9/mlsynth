"""The exact backend as a bilevel solver: ``solve_exact(BilevelProblem) -> BilevelSolution``.

The validation outcomes of the problem (``y1_pre``, ``Y0_pre``) are the paper's Z1 and
Z0 and its predictors are X1 and X0. The donor weights are the search's optimum in the
class asked for: ``"diagonal"`` (the paper's problem (5), positive diagonal V) or
``"pd"`` (problem (6), positive definite V). The predictor weights reported as ``V`` are
the log-det maximiser of the V that select those donor weights, as shares summing to
one: the diagonal entries of a diagonal V, or V_kk / tr V of a positive definite one.
They refer to the predictors scaled to unit standard deviation across the treated unit
and the donors (ddof=1), the scaling the paper and VanillaSC use, so predictors already
on that scale are reported as given.

``metadata["exact"]`` holds the ``ExactDetails``: the case the search took, both
rationalisers' centres (the diagonal ones only in the diagonal class), the identified
ranges when ``ranges=True``, the visible facets and those holding a positive definite
optimum outside the hull, and the face counts when ``face_counts=True``.
"""
from __future__ import annotations

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthDataError
from ...solvers.ridge_augment import simplex_qp
from ..structure import BilevelProblem, BilevelSolution
from . import faces, interior, rationalize, search
from .structures import DiagonalRationalizers, ExactDetails, PDRationalizers

V_CLASSES = ("diagonal", "pd")


def solve_exact(prob: BilevelProblem, *, v_class: str = "diagonal", ranges: bool = False,
                face_counts: bool = False, max_cells: int = interior.MAX_CELLS) -> BilevelSolution:
    """Solve the bilevel program exactly (Abadie and Shang 2026).

    Parameters
    ----------
    prob : BilevelProblem
        Validation outcomes and predictors; at least one predictor.
    v_class : {"diagonal", "pd"}
        The class of predictor weights the donor weights are chosen over.
    ranges : bool
        Also bound each V_kk over the positive definite rationalisers (2K barrier solves).
    face_counts : bool
        Also count the faces of the donor hull by dimension (Table 3); outside the hull only.
    max_cells : int
        Limit on the cells an interior walk visits before it raises.
    """
    if v_class not in V_CLASSES:
        raise MlsynthConfigError(f"v_class must be one of {V_CLASSES}; got {v_class!r}.")
    if prob.n_predictors == 0:
        raise MlsynthDataError("the exact backend needs predictors: it searches the donor weights some "
                               "predictor weights select, and with no predictors there are none to weight")
    X1, X0, Z1, Z0 = prob.X1, prob.X0, prob.y1_pre, prob.Y0_pre
    stats = {}
    if v_class == "diagonal":
        loss, W = search.solve_diag(X1, X0, Z1, Z0, stats=stats, max_cells=max_cells)
        dg = rationalize.diag_rationalizers(X1, X0, W)
        diagonal = DiagonalRationalizers(logdet=dg["logdet"], center=dg["center"],
                                         support=tuple(int(k) for k in dg["support"]),
                                         forced_zero=tuple(int(k) for k in dg["forced_zero"]),
                                         n_implicit=int(dg["n_implicit"]))
    else:
        loss, W = search.solve_pd(X1, X0, Z1, Z0, stats=stats, max_cells=max_cells)
        diagonal = None
    pr = rationalize.pd_rationalizers(X1, X0, W, ranges=ranges)
    pd = PDRationalizers(logdet=pr["Vstar"], center=pr["Vcenter"], logdet_min_eig=float(pr["logdet_min_eig"]),
                         range_lo=pr.get("lo"), range_hi=pr.get("hi"))
    V = diagonal.logdet if diagonal is not None else pd.logdet_shares
    regime = stats.pop("regime")
    details = ExactDetails(
        v_class=v_class, regime=regime, predictor_names=tuple(str(n) for n in prob.predictor_names),
        validation_loss=loss, pd=pd, diagonal=diagonal,
        n_visible_facets=stats.pop("n_visible", None),
        facets=stats.pop("facets", None),
        face_counts=faces.face_counts(X1, X0) if face_counts and regime == "exterior" else None,
        search=stats)
    S1, S0 = rationalize._standardize(X1, X0)
    r = S1 - S0 @ W
    lower = float(np.sum(V * r ** 2)) if diagonal is not None else float(r @ pd.logdet @ r / np.trace(pd.logdet))
    w_free = simplex_qp(Z0, Z1)
    return BilevelSolution(V=np.asarray(V, float), W=W, upper_loss=loss, lower_loss=lower,
                           lower_bound=float(np.sum((Z1 - Z0 @ w_free) ** 2)), stage="exact",
                           metadata={"exact": details, "v_class": v_class, "regime": regime})
