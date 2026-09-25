"""Cross-validation: SNN's greedy anchor search vs exact maximum biclique.

SNN's contribution is that identification is local: an entry is recoverable
whenever the observed mask holds a fully observed cross around it. Finding the
largest such cross is the maximum-biclique problem. The reference
implementation (``deshen24/syntheticNN``) solves it exactly, enumerating every
maximal clique of an augmented graph and keeping the most square one. mlsynth
uses a dependency-free greedy search: strip the emptiest row or column until
the block is complete.

``snn_prop99`` cross-validates the two implementations on Proposition 99 and
reports machine-precision agreement, but it cannot speak to any of this. Under
block missingness the neighborhood submatrix is already complete, so both
searches return it on the first check and neither the clique enumeration nor
the greedy loop ever runs. That case validates the PCR half of SNN on the panel
geometry SI and RSC already cover; the anchor search -- the half that is SNN's
own -- went unmeasured.

This case measures it, on scattered MNAR masks where the search is the whole
problem. Observation probability rises with the entry's value, so the pattern
is missing not at random and irregular, with no block structure to fall back
on. Both engines run on the same masks with the same rank rule, against a known
low-rank signal.

Three things are pinned.

Coverage. The share of missing entries for which each search finds any cross at
all. An entry the search loses is an entry SNN cannot impute, so this is a
capability measure and not an accuracy one. The exact search never fails; the
greedy search must not either.

Block quality. The mean min-dimension of the discovered cross, greedy against
exact. Larger is better: the entrywise error bound in the paper falls with
``min(|AR|, |AC|)``, so a smaller block is a weaker guarantee.

Accuracy. Mean absolute imputation error against the planted signal, greedy
against exact, over the entries both engines reach. The greedy search is a
heuristic, so it is expected to lose some accuracy; what is pinned is how much.

This case is why the greedy search's tie-break compares shares and not counts.
Comparing raw missing counts made the longer side always look worse, so on
these masks the search stripped one side to nothing and lost the cross on 55
percent of missing entries while the exact search lost none, with a mean
min-dimension of 2.86 against 5.94 and 74 percent more error where it did
return a block. No panel benchmark could see it, because no panel reaches that
line. The regression is pinned directly in
``mlsynth/tests/test_snn_anchor_search.py``; the rows below are its empirical
companion.

Provenance
----------
* Reference: ``deshen24/syntheticNN`` @ ``a95b511``, fetched on demand into the
  gitignored ``benchmarks/reference/.cache``. The case skips if neither git nor
  codeload is reachable.
* ``snn.py`` calls ``nx.from_numpy_matrix``, removed in networkx 3.0, so its
  biclique branch raises on any modern install and only the fast path runs.
  ``benchmarks/reference/clone_syntheticnn.py`` aliases it to
  ``from_numpy_array``, which is what lets the reference's clique enumeration
  run here at all.
* Data: generated, not empirical. Five 24x18 rank-3 matrices with MNAR masks
  (observation probability from 0.55 to 0.95, rising with the entry's value),
  seeds 0-4. Exhaustive maximal-clique enumeration is exponential, so the
  matrices are kept small enough for the reference to finish.
"""
from __future__ import annotations

import warnings

import numpy as np

from benchmarks.reference.clone_syntheticnn import import_syntheticnn

SEEDS = range(5)
SHAPE = (24, 18)
RANK = 3
P_LO, P_HI = 0.55, 0.95


def _mnar_matrix(seed: int):
    """A rank-``RANK`` signal and an MNAR mask whose rate rises with the value."""
    rng = np.random.default_rng(seed)
    m, n = SHAPE
    A = rng.standard_normal((m, RANK)) @ rng.standard_normal((RANK, n))
    z = (A - A.min()) / (A.max() - A.min())
    observed = rng.random((m, n)) < P_LO + (P_HI - P_LO) * z
    X = A.copy()
    X[~observed] = np.nan
    return A, X


def _measure():
    """Per-entry comparison of the two anchor searches across all seeds."""
    from mlsynth.utils.snn_helpers.completion import _find_anchors, snn_predict

    RefSNN = import_syntheticnn().SyntheticNearestNeighbors

    n_missing = g_lost = r_lost = 0
    g_dim, r_dim, g_err, r_err = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in SEEDS:
            A, X = _mnar_matrix(seed)
            mask = (~np.isnan(X)).astype(int)
            ref = RefSNN(n_neighbors=1, verbose=False)
            ref_out = RefSNN(n_neighbors=1, verbose=False).fit_transform(X.copy())

            for i, j in np.argwhere(np.isnan(X)):
                i, j = int(i), int(j)
                n_missing += 1
                gAR, gAC = _find_anchors(mask, i, j)
                rAR, rAC = ref._find_anchors(X, (i, j))
                rAR, rAC = np.atleast_1d(rAR), np.atleast_1d(rAC)
                g_ok = gAR.size > 0 and gAC.size > 0
                r_ok = rAR.size > 0 and rAC.size > 0
                g_lost += not g_ok
                r_lost += not r_ok
                if not (g_ok and r_ok):
                    continue
                g_dim.append(min(gAR.size, gAC.size))
                r_dim.append(min(rAR.size, rAC.size))
                # Same rank rule on both sides (Donoho-Gavish), so the
                # comparison is of the anchor search and not of the truncation.
                val, ok = snn_predict(X, mask, i, j, universal=True)
                if ok and np.isfinite(val):
                    g_err.append(abs(val - A[i, j]))
                if np.isfinite(ref_out[i, j]):
                    r_err.append(abs(ref_out[i, j] - A[i, j]))
    return {
        "n_missing": n_missing,
        "greedy_lost_pct": 100.0 * g_lost / n_missing,
        "exact_lost_pct": 100.0 * r_lost / n_missing,
        "greedy_min_dim": float(np.mean(g_dim)),
        "exact_min_dim": float(np.mean(r_dim)),
        "greedy_mae": float(np.mean(g_err)),
        "exact_mae": float(np.mean(r_err)),
    }


def run() -> dict:
    m = _measure()
    return {
        "n_missing_entries": float(m["n_missing"]),
        "greedy_lost_cross_pct": m["greedy_lost_pct"],
        "exact_lost_cross_pct": m["exact_lost_pct"],
        "greedy_min_dim": m["greedy_min_dim"],
        "exact_min_dim": m["exact_min_dim"],
        # The greedy block as a share of the exact one: 1.0 would mean the
        # heuristic matches exhaustive enumeration.
        "min_dim_ratio": m["greedy_min_dim"] / m["exact_min_dim"],
        "greedy_mae": m["greedy_mae"],
        "exact_mae": m["exact_mae"],
        # How much accuracy the heuristic costs, as a share of the exact
        # search's error.
        "mae_ratio": m["greedy_mae"] / m["exact_mae"],
    }


def comparison() -> dict:
    """mlsynth's greedy anchor search against exhaustive maximal-clique search.

    Returns ``{"rows": [...], "mlsynth_call": {...}, "reference": {...}}`` with
    one row per measured quantity: coverage, block quality, and imputation
    accuracy, each for both searches on the same MNAR masks.
    """
    m = _measure()
    rows = [
        {"quantity": "missing entries", "mlsynth": m["n_missing"],
         "reference": m["n_missing"]},
        {"quantity": "entries with no cross found (%)",
         "mlsynth": round(m["greedy_lost_pct"], 2),
         "reference": round(m["exact_lost_pct"], 2)},
        {"quantity": "anchor cross min-dimension (mean)",
         "mlsynth": round(m["greedy_min_dim"], 3),
         "reference": round(m["exact_min_dim"], 3)},
        {"quantity": "imputation mean |error|",
         "mlsynth": round(m["greedy_mae"], 4),
         "reference": round(m["exact_mae"], 4)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {
            "engine": "mlsynth.utils.snn_helpers.completion",
            "search": "greedy: strip the emptiest row or column by share",
            "rank": "Donoho-Gavish universal threshold, both sides",
        },
        "reference": {
            "impl": "deshen24/syntheticNN SyntheticNearestNeighbors "
                    "(live run), exact maximum biclique via "
                    "networkx.find_cliques",
            "version": "deshen24/syntheticNN @ a95b511, with "
                       "from_numpy_matrix aliased to from_numpy_array",
        },
    }


# Tolerances.
#
# These rows are a heuristic measured against an exact algorithm, so only the
# coverage rows are equalities; the rest are pinned at what the heuristic
# currently achieves, with bands wide enough to survive ordinary drift and
# narrow enough to catch a collapse.
#
# `greedy_lost_cross_pct` and `exact_lost_cross_pct` are both 0 and pinned at
# 0 with zero tolerance. This is the row the anchor-search fix exists for: any
# entry losing its cross is an entry SNN cannot impute, and the exact search
# proves one was available. Before the fix this row read 55.07.
#
# `n_missing_entries` pins the design: 592 missing cells across the five seeded
# masks. It moves only if the generator changes, which would invalidate every
# other row, so it carries zero tolerance.
#
# `greedy_min_dim` 5.36 against `exact_min_dim` 5.68, a ratio of 0.944 -- the
# greedy search recovers 94% of the block dimension exhaustive enumeration
# finds. The paper's entrywise bound falls with min(|AR|, |AC|), so this is the
# quantity that decides what the heuristic costs in guarantee. The ratio is
# pinned with a 0.15 band: it fails if the greedy block drops below 79% of the
# exact one, which is where the old count-based rule sat (2.86/5.94 = 0.48).
#
# `mae_ratio` 1.124 -- the greedy search's mean absolute error is 12% above the
# exact search's. Pinned with a 0.35 band, so the row fails if the heuristic's
# error exceeds ~1.47x the exact search's. The old rule sat at 1.74 on the
# entries it could reach at all.
#
# The two absolute error rows carry wide bands. They depend on the planted
# signal's scale and on the rank rule, so they are recorded to make the ratio
# auditable and are not themselves the claim.
EXPECTED = {
    "n_missing_entries": (592.0, 0),
    "greedy_lost_cross_pct": (0.0, 0),
    "exact_lost_cross_pct": (0.0, 0),
    "greedy_min_dim": (5.36, 0.8),
    "exact_min_dim": (5.68, 0.8),
    "min_dim_ratio": (0.944, 0.15),
    "greedy_mae": (0.7148, 0.25),
    "exact_mae": (0.6361, 0.25),
    "mae_ratio": (1.124, 0.35),
}
