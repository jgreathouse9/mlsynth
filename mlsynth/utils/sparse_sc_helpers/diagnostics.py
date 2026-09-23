"""Degeneracy diagnostics for a completed SparseSC solve.

The outer problem

    min over v >= 0   F(v) = (1/T) ||Z1 - Z0 w*(v)||^2 + lam ||v||_1

is neither convex nor smooth, so the returned ``v`` is whichever critical point
the solve reached. Critical points here are not interchangeable: some sit in
heavily kinked regions where the gradient sees almost nothing, and the solve
stops there for reasons that have nothing to do with fit. This module reads
that off the finished design and reports it, without re-solving anything or
changing any estimate.

Two readings, both available in closed form.

The U-space -- the directions in which ``F`` varies smoothly -- is the support
of ``v``. Liu and Sagastizabal (Example 9.4, pp. 322-323, in Bagirov, Gaudioso,
Karmitsa, Makela and Taheri, *Numerical Nonsmooth Optimization*) treat exactly
``h(x) = q(x) + ||x||_1`` with ``q`` smooth and print the bases: "respective
bases for V(x) and U(x) are ``{e_j : x_j = 0}`` and ``{e_j : x_j != 0}``". So
``dim U`` is a count of nonzeros and needs no computation. Their *algorithms*
do not transfer -- Sect. 9.8, p. 327 states "the VU-algorithms are designed for
convex functions only" and ``F`` is not convex -- only the subspace reading.

The inner problem gives the second. ``w*(v)`` lies on a face of the donor
simplex carrying ``|A|`` active donors, where it has ``|A| - 1`` degrees of
freedom, so the envelope gradient carries no information about the donors that
are out, and at ``|A| = 1`` it is identically zero.

Two conditions warn rather than merely report, because a caller who does not
look will otherwise act on a number that does not mean what it appears to.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np

from .objective import ACTIVE_TOL
from .structures import SparseSCDegeneracy, SparseSCDesign

# Predictors below this are treated as out of the support. The anchor is pinned
# at 1 and the penalised weights that leave the support are driven to exactly
# 0 by the bound, so this separates "at the bound" from "small but present"
# rather than discriminating among small values.
SUPPORT_TOL = 1e-10


def assess_degeneracy(
    design: SparseSCDesign,
    *,
    support_tol: float = SUPPORT_TOL,
    active_tol: float = ACTIVE_TOL,
) -> SparseSCDegeneracy:
    """Read the degeneracy of a finished solve off its design.

    Parameters
    ----------
    design : SparseSCDesign
        The completed design: ``v``, ``w`` and ``v_path``.
    support_tol : float, optional
        Predictors with ``|v_p| > support_tol`` are in the support.
    active_tol : float, optional
        Donors with ``|w_j| > active_tol`` are active. Defaults to the
        constant the envelope gradient itself uses, so the reported ``|A|``
        is the one the gradient saw.
    """
    v = np.asarray(design.v, dtype=float).ravel()
    w = np.asarray(design.w, dtype=float).ravel()
    v_path = np.atleast_2d(np.asarray(design.v_path, dtype=float))

    in_support = np.abs(v_path) > support_tol
    # Column 0 is the anchor, pinned at 1; the corner is every other entry out.
    free = in_support[:, 1:] if in_support.shape[1] > 1 else in_support[:, :0]
    n_anchor_only = int(np.count_nonzero(~free.any(axis=1))) if free.size or free.shape[0] else 0

    return SparseSCDegeneracy(
        dim_u=int(np.count_nonzero(np.abs(v) > support_tol)),
        n_predictors=int(v.size),
        n_active_donors=int(np.count_nonzero(np.abs(w) > active_tol)),
        n_donors=int(w.size),
        n_anchor_only_grid=n_anchor_only,
        n_distinct_supports=int(np.unique(in_support, axis=0).shape[0]),
        n_grid=int(v_path.shape[0]),
        support_tol=float(support_tol),
        active_tol=float(active_tol),
    )


def warn_if_degenerate(
    degeneracy: SparseSCDegeneracy,
    predictor_names: Optional[Sequence[Any]] = None,
    stacklevel: int = 3,
) -> None:
    """Warn on the two conditions a caller would otherwise act on unawares.

    Both are statements about what the returned fit *is*, not guesses about
    whether it is good.
    """
    if degeneracy.anchor_only:
        who = ""
        if predictor_names is not None and len(predictor_names):
            who = f" ({predictor_names[0]!r})"
        warnings.warn(
            f"SparseSC: the selected fit matches on the anchor predictor"
            f"{who} alone -- every other predictor weight is at the bound. "
            f"The penalty's high-lambda limit is the anchor, so the reported "
            f"predictor set is a consequence of which predictor is listed "
            f"first, not of the data. Reorder `covariates`, or lower the "
            f"lambda grid's upper end, to see whether the fit survives.",
            UserWarning, stacklevel=stacklevel,
        )
    if degeneracy.n_active_donors <= 1:
        warnings.warn(
            f"SparseSC: the counterfactual rests on "
            f"{degeneracy.n_active_donors} donor(s) of {degeneracy.n_donors}. "
            f"On a face this narrow the outer objective's gradient carries no "
            f"information about the donors that are out (at one active donor "
            f"it vanishes identically), so the solve stopped for a reason "
            f"unrelated to fit.",
            UserWarning, stacklevel=stacklevel,
        )
