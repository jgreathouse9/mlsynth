"""The SYNDES treated-set search against SCIP proving optimality.

``mode="two_way_global"`` defaults to ``backend="exact"``: naming the treated set
removes every binary from the design MIP, leaving a convex program, so the design
is found by searching treated sets instead of by branch-and-bound. The claim that
default rests on is that the search reaches the design SCIP would prove optimal.

This pins it on the panel Doudchenko et al. (2021) use themselves --
``basedata/urate_cps.csv``, the BLS state unemployment series behind
:mod:`benchmarks.cases.syndes_bls` -- across several sub-panels and treated-set
sizes. SCIP is asked to prove optimality (``gap_limit=0``), which makes it an
independent referent: it is a separate, widely used exact solver handed the same
mixed-integer program.

This is not a replication of a published number. It is a cross-check of two
solvers on one program, which is why it reports agreement and not a match to a
table. The estimator's own replication is
:mod:`benchmarks.cases.syndes_bls` (Path B, Table 1).

What the comparison has to control for
---------------------------------------
``gap_limit`` defaults to ``0.05``, so the MIP path may stop at a design five
percent above the optimum, and on some panels it does. The two backends agree on
the treated set only once the MIP is asked to prove optimality, which is the
configuration used here. Where they would otherwise differ, the search returns
the better design.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "urate_cps.csv"

#: Sub-panels drawn from the 40-month x 50-state matrix. Sized so SCIP can close
#: the gap in seconds while the search still ranks thousands of designs.
_PANELS = ((12, 20), (15, 20))
_KS = (3, 5)
_DRAWS = 2
SEED = 0


def _subpanel(Y: np.ndarray, n_units: int, n_periods: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.choice(Y.shape[1], size=n_units, replace=False))
    return Y[:n_periods][:, cols]


def run() -> dict:
    from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
    from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design

    Y = np.loadtxt(_DATA, delimiter=",")
    agree, gaps, cells = [], [], 0
    search_never_worse = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n_units, n_periods in _PANELS:
            for draw in range(_DRAWS):
                panel = _subpanel(Y, n_units, n_periods, SEED + draw)
                for K in _KS:
                    search = solve_synthetic_design_exact(Y=panel, K=K)
                    mip = solve_synthetic_design(panel, K=K, mode="global_2way",
                                                 gap_limit=0.0, time_limit=300.0)
                    agree.append(float(
                        tuple(int(i) for i in search.selected_unit_indices)
                        == tuple(int(i) for i in mip.selected_unit_indices)))
                    s_obj = float(search.objective_value)
                    m_obj = float(mip.objective_value)
                    gaps.append(abs(s_obj - m_obj))
                    search_never_worse.append(float(s_obj <= m_obj + 1e-6))
                    cells += 1

    return {
        "n_cells": float(cells),
        "treated_set_agreement": float(np.mean(agree)),
        "max_objective_gap": float(np.max(gaps)),
        "search_never_worse": float(np.mean(search_never_worse)),
    }


# Tolerances.
#
# The two agreement figures are exact. Every cell is a program SCIP has proven
# optimal, so the search must select the same treated set in all of them and its
# objective can never exceed SCIP's. A single disagreeing cell means the search
# missed an optimum, which is the defect this case exists to catch, and no
# fraction below one is acceptable.
#
# The objective gap is bounded by the two solvers' numerical tolerances, not by
# any modelling difference: both minimise the same quadratic over the same
# feasible set. Measured 1.2e-09 to 2.2e-09 across these cells, where the
# outcomes are unemployment rates on [0.01, 0.17] so the objective scale is
# small. 1e-05 leaves four orders of headroom for a differently-built SCIP while
# staying far below the spacing between distinct designs, which is where a
# genuine disagreement would show.
#
# n_cells guards the loop itself: a case that silently stopped covering panels
# would still report perfect agreement over whatever remained.
EXPECTED = {
    "n_cells": (8.0, 0.0),
    "treated_set_agreement": (1.0, 0.0),
    "max_objective_gap": (0.0, 1e-5),
    "search_never_worse": (1.0, 0.0),
}
