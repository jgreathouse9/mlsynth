"""Synthetic-control weight solvers for SCMO.

The simplex solver replaces ``Opt.SCopt(scm_model_type="SIMPLEX")``: it
minimizes the squared imbalance between the treated and donor matching
vectors subject to the convex-combination constraint (weights >= 0, sum 1),
and carries the ridge the Tian-Lee-Panchenko / Abadie program carries.

That ridge is not decoration. Their ``fn_W`` builds
``Dmat <- ZJ %*% V %*% t(ZJ) + (10^-7) * diag(J)``, which with ``V = (1/p) I``
adds ``p * 1e-7 ||w||^2`` to the objective. Where the imbalance has one
minimiser the term changes nothing -- 1.5e-7 in the German concatenated
weights. Where it has a face of them the term is the only thing choosing a
point, and SCMO's averaged scheme on a single-period multiple-outcome spec
produces exactly that: every outcome averages into one column, so the German
panel is one equation in sixteen donors. Solved without the ridge, relabelling
the donors moved the weights by 0.419 and the reported ATT by hundreds.

:func:`~mlsynth.utils.bilevel.active_set.solve_simplex_qp_least_norm` is that
term as a selection rule, scaled by the design so it survives a rescaling of
the matching columns that the authors' absolute constant does not.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..bilevel.active_set import solve_simplex_qp_least_norm


def simplex_weights(Z_treated: np.ndarray, Z_donors: np.ndarray) -> np.ndarray:
    """Convex (simplex) SC weights minimizing ``||Z_treated - Z_donors' w||^2``.

    Ties are broken by the smallest ``||w||``, which is what the reference
    program's ridge selects.

    Parameters
    ----------
    Z_treated : np.ndarray
        Treated matching vector, shape ``(P,)``.
    Z_donors : np.ndarray
        Donor matching matrix, shape ``(J, P)``.

    Returns
    -------
    np.ndarray
        Donor weights, shape ``(J,)``; non-negative and summing to one.
    """
    try:
        w_hat = np.clip(
            solve_simplex_qp_least_norm(np.asarray(Z_donors, dtype=float).T,
                                        np.asarray(Z_treated, dtype=float)),
            0.0, None,
        )
    except Exception as exc:
        raise MlsynthEstimationError(
            "SCMO simplex weight solve failed: the donor matching matrix may "
            f"be degenerate or ill-conditioned ({exc})."
        ) from exc
    total = w_hat.sum()
    return w_hat / total if total > 0 else w_hat            # exact simplex (sum == 1)
