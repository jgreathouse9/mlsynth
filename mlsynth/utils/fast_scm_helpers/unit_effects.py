r"""Each treated unit against its own synthetic control.

The design LEXSCM solves reports one aggregate: a synthetic treated unit, a
synthetic control, and the gap between them. Abadie and Zhao's equation (10)
also speaks of each treated unit matched to its own synthetic control, drawn
from the same control pool, and that is the object
:func:`~mlsynth.utils.fast_scm_helpers.post_inference.unit_level_cumulative`
consumes: period-by-unit gaps, one column per treated unit.

Computing them is one simplex least-squares solve per treated unit, over the
control pool the design chose, fitted on the window the design fitted on. The
work is proportional to the number of treated units and is done once for the
winning design, never for a candidate during the search.

The two windows do different jobs. The fit window is where each unit's donor
weights are solved, so its gap is in-sample and says how well the pool can
track that unit at all. The blank window is held out of every solve, so its gap
is the out-of-sample residual: the null that the cumulative path resamples, and
the series :func:`~mlsynth.utils.fast_scm_helpers.post_inference.approximability`
tests for a location offset the resampling cannot shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthDataError
from ..solvers.active_set import solve_simplex_qp
from .post_inference import approximability


@dataclass(frozen=True)
class TreatedUnitEffect:
    """One treated unit, its own synthetic control, and the three gap series.

    Attributes
    ----------
    unit : str
        The treated unit's label.
    weight : float
        Its weight in the design's synthetic treated unit.
    donor_weights : dict
        Its own synthetic control over the design's control pool, on the
        simplex. These are not the design's control weights, which fit the
        treated aggregate; they fit this unit alone.
    gap_fit, gap_blank, gap_post : np.ndarray
        Observed minus synthetic over the fit, blank and post windows.
    approximability_t, approximability_ok : float, bool
        The blank-window gate: whether the pool reproduces this unit
        off-sample, and the statistic behind the verdict.
    """

    unit: str
    weight: float
    donor_weights: Dict[str, float]
    gap_fit: np.ndarray
    gap_blank: np.ndarray
    gap_post: np.ndarray
    approximability_t: float
    approximability_ok: bool


def per_treated_unit_effects(
    Y: np.ndarray,
    labels: Sequence[str],
    treated: Mapping[str, float],
    control: Mapping[str, float],
    fit_idx: np.ndarray,
    blank_idx: np.ndarray,
    post_idx: np.ndarray,
    *,
    tol: float = 1e-10,
) -> Tuple[TreatedUnitEffect, ...]:
    """Fit every treated unit to its own synthetic control.

    Parameters
    ----------
    Y : np.ndarray, shape (T, N)
        The observed outcome matrix, time by unit.
    labels : sequence of str
        Unit labels in ``Y``'s column order.
    treated, control : mapping
        The design's treated and control weights, keyed by label. Treated units
        carrying no weight are skipped; the control keys name the donor pool.
    fit_idx, blank_idx, post_idx : np.ndarray
        Row indices of the three windows.

    Returns
    -------
    tuple of TreatedUnitEffect
        One per treated unit with weight above ``tol``, in the order ``treated``
        gives them.
    """
    Y = np.asarray(Y, dtype=float)
    pos = {lab: i for i, lab in enumerate(labels)}

    kept = [k for k, v in treated.items() if abs(float(v)) > tol]
    if not kept:
        raise MlsynthDataError(
            "per_treated_unit_effects needs at least one treated unit carrying "
            f"weight; got {len(treated)} entr(y/ies), all within {tol} of zero. "
            "A design whose treated support collapsed has no per-unit "
            "decomposition to report."
        )
    donors = [k for k, v in control.items() if abs(float(v)) > tol]
    if not donors:
        raise MlsynthDataError(
            "per_treated_unit_effects needs at least one control unit carrying "
            "weight; the donor pool is what each treated unit is fitted "
            "against, and an empty pool leaves nothing to fit."
        )
    missing = [k for k in list(kept) + list(donors) if k not in pos]
    if missing:
        raise MlsynthDataError(
            f"unit(s) {missing} are not in the panel. The labels must match "
            f"Y's column order, which holds {len(labels)} unit(s)."
        )
    both = sorted(set(kept) & set(donors))
    if both:
        raise MlsynthDataError(
            f"unit(s) {both} are both treated and control. A unit in its own "
            "donor pool reproduces itself exactly and reports a zero gap, so "
            "the design's treated and control sets must be disjoint."
        )

    donor_cols = [pos[k] for k in donors]
    out = []
    for name in kept:
        j = pos[name]
        D = Y[fit_idx][:, donor_cols]
        v = solve_simplex_qp(D, Y[fit_idx, j])
        v = np.asarray(v[0] if isinstance(v, tuple) else v, dtype=float)

        def gap(idx):
            idx = np.asarray(idx, dtype=int)
            if idx.size == 0:
                return np.zeros(0, dtype=float)
            return Y[idx, j] - Y[idx][:, donor_cols] @ v

        blank = gap(blank_idx)
        check = approximability(blank) if blank.size >= 3 else None
        out.append(TreatedUnitEffect(
            unit=name,
            weight=float(treated[name]),
            donor_weights={k: float(v[i]) for i, k in enumerate(donors)},
            gap_fit=gap(fit_idx),
            gap_blank=blank,
            gap_post=gap(post_idx),
            approximability_t=float(check.t_stat) if check else float("nan"),
            approximability_ok=bool(check.ok) if check else False,
        ))
    return tuple(out)
