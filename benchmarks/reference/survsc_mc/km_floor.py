"""How much of the Table 1 error is the Kaplan-Meier step?

    python benchmarks/reference/survsc_mc/km_floor.py

Measures the sup-norm distance between a single unit-period Kaplan-Meier curve
and its own analytic truth, on the same grid the estimator uses and at the same
rates and censoring the Section 4 design produces. Compare the result against
Table 1: it sets the floor the SSC error is measured on top of.
"""
from __future__ import annotations

import numpy as np

from survsc_oracle import (
    N_PERIODS,
    N_UNITS,
    TREATED,
    draw_design,
    kaplan_meier,
    sample_cell,
    solve_censoring_rate,
)

KS = (100, 300, 700)
REPS = 20
SEED0 = 777

PAPER_SSC = {
    ("cox", 100): 0.1177, ("cox", 300): 0.0652, ("cox", 700): 0.0542,
    ("aalen", 100): 0.0621, ("aalen", 300): 0.0507, ("aalen", 700): 0.0245,
}


def per_curve_error(dgp: str, K: int) -> float:
    out = []
    for s in range(REPS):
        rng = np.random.default_rng(SEED0 + s)
        design = draw_design(rng, dgp)
        nu = solve_censoring_rate(design.rate)
        cells = {}
        for p in range(N_PERIODS):
            for n in range(N_UNITS):
                if p == 1 and n == TREATED:
                    continue
                cells[(p, n)] = sample_cell(design.rate[p, n], nu, K, rng)
        pooled = np.concatenate([t for t, _ in cells.values()])
        grid = np.linspace(0.0, float(np.quantile(pooled, 0.90)), 100)
        out.append(np.mean([
            np.max(np.abs(kaplan_meier(t, e, grid)
                          - np.exp(-design.rate[p, n] * grid)))
            for (p, n), (t, e) in cells.items()
        ]))
    return float(np.mean(out))


def main() -> None:
    print(f"{'DGP':6} {'K':>4} {'KM sup-error':>13} {'Table 1 SSC':>12} {'ratio':>7}")
    for dgp in ("cox", "aalen"):
        for K in KS:
            km = per_curve_error(dgp, K)
            ssc = PAPER_SSC[(dgp, K)]
            print(f"{dgp:6} {K:>4} {km:>13.4f} {ssc:>12.4f} {ssc / km:>7.2f}")


if __name__ == "__main__":
    main()
