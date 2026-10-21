"""Is ``treated_fit_bound`` a lower bound on the fit it prunes with, and how tight?

``timing.py`` measures what the search saves. It cannot say whether the saving
is sound, because a bound that sometimes exceeds the fit discards the optimum
and still returns fast. This enumerates every admissible treated set on each
panel and compares the bound with the exact fit it claims to bound.

Two quantities per panel:

``violations``
    sets where ``bound > fit`` by more than ``PRUNE_MARGIN`` relative to the
    fit. Each one is a candidate the search could wrongly discard. The count
    must be zero; the bound is what makes the search exact.
``settled``
    sets where the bound equals the fit. For this bound that is every set whose
    sum-to-one minimiser is already non-negative, so the simplex constraint is
    slack and the closed form is the answer. A high share means the bound is
    tight, which is why so few candidates need fitting.

Both the plain treated fit and the penalized one are checked. They take
different closed forms -- ``1 / 1'G^-1 1`` against
``(2 + t)^2 / 4s - u / 4`` -- and only the second reads the linear term, so a
panel run without penalties never exercises it.

    python -m benchmarks.studies.marex_exact_backend.check_bounds [out.csv]
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from mlsynth.utils.marex_helpers.exact import (
    PRUNE_MARGIN,
    side_fit,
    treated_fit_bound,
)

from .timing import SEEDS, SIZES, panel


def _fit_block(df, T0):
    """The predictor block the search matches on: pre-period outcomes."""
    Y = df.to_numpy(dtype=float)
    return Y[:, :T0].T                      # (T0, J), columns are markets


def check(J: int, m: int, seed: int, penalty: bool) -> dict:
    df, T0 = panel(J=J, seed=seed)
    B = _fit_block(df, T0)
    A = B.mean(axis=1)                      # the cluster mean is the target
    subsets = np.array(list(itertools.combinations(range(J), m)), dtype=np.int64)

    rng = np.random.default_rng(seed)
    linear = rng.uniform(0.0, 0.5, size=J) if penalty else None

    lb = treated_fit_bound(B, A, subsets, linear=linear)
    fit = np.empty(len(subsets))
    for i, cols in enumerate(subsets):
        lin_i = None if linear is None else linear[cols]
        fit[i], _ = side_fit(B, A, cols, linear=lin_i)

    scale = np.maximum(np.abs(fit), 1.0)
    excess = (lb - fit) / scale
    violations = int((excess > PRUNE_MARGIN).sum())
    settled = int(np.isclose(lb, fit, rtol=1e-9, atol=1e-12).sum())
    return {
        "J": J, "m": m, "seed": seed,
        "penalty": bool(penalty),
        "candidates": len(subsets),
        "violations": violations,
        "worst_excess": float(excess.max()),
        "settled": settled,
        "settled_share": settled / len(subsets),
        "mean_slack": float(np.mean((fit - lb) / scale)),
    }


def main(out: str | None) -> None:
    rows = []
    # The full enumeration is factorial, so this runs the sizes where every
    # candidate can be fitted exactly; the bound's algebra does not change
    # with J, and a violation is a violation at any size.
    for J, m in [s for s in SIZES if J_ok(s)]:
        for seed in SEEDS:
            for penalty in (False, True):
                rows.append(check(J, m, seed, penalty))
    d = pd.DataFrame(rows)
    print(d.to_string(index=False,
                      formatters={"worst_excess": "{:.2e}".format,
                                  "settled_share": "{:.3f}".format,
                                  "mean_slack": "{:.4f}".format}))
    total_v = int(d["violations"].sum())
    print(f"\n{len(d)} panel-configurations, "
          f"{int(d['candidates'].sum()):,} treated sets enumerated")
    print(f"violations of the lower bound: {total_v}")
    print(f"settled (bound == fit): {d['settled_share'].mean():.1%} of sets on average")
    if out:
        d.to_csv(out, index=False)
        print(f"wrote {out}")
    raise SystemExit(0 if total_v == 0 else 1)


def J_ok(size) -> bool:
    J, m = size
    from math import comb
    return comb(J, m) <= 60_000


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
