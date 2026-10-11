"""Is each bound a lower bound, and how tight is it?

For every treated set on each panel, both bounds are compared with the exact
treated-side fit ``g(S)`` they claim to bound. A violation is a candidate the
bound could wrongly discard: ``lb(S) > g(S)`` beyond the relative tolerance the
search prunes with. ``settled`` counts the candidates where the bound equals
``g(S)``, which for the centred bound is every set whose sum-to-one minimiser
is already non-negative.

    python check_bounds.py results/check_bounds.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panels import fit_matrices
from search import BOUNDS, PRUNE_MARGIN, all_subsets, fit_value

CASES = [(J, m, seed) for J, m in ((12, 3), (16, 4), (20, 5))
         for seed in (3, 11, 42)]


def main(out: str) -> None:
    rows = []
    for J, m, seed in CASES:
        B, A = fit_matrices(J, seed)
        subs = all_subsets(J, m)
        g = np.array([fit_value(B, A, S) for S in subs])
        tol = PRUNE_MARGIN * (1.0 + np.abs(g))
        for name, bound in BOUNDS.items():
            lb = bound(B, A, subs)
            excess = lb - g
            rows.append(dict(
                J=J, m=m, seed=seed, bound=name, candidates=len(subs),
                violations=int((excess > tol).sum()),
                worst_excess=float(excess.max()),
                settled=int((np.abs(excess) <= tol).sum()),
                median_gap=float(np.median(g - lb))))
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/check_bounds.csv")
