"""Cross-validation: mlsynth's GRC engine against Yamamoto's own R package.

``mlsynth.utils.clustersc_helpers.rpca.fgrc`` is a pure-Python transcription of
the ``OptimGRC_C`` kernel in Michio Yamamoto's ``grc`` package
(https://github.com/michioyamamoto/grc), which backs the ``cluster_method="fgrc"``
donor-selection step. ``fgrc_toy_subspace`` already validates that the engine
recovers the authors' documented toy structure. This case is the other half:
does the port compute the same thing the authors' code computes.

The split of labour
-------------------
R generates the panels, with the authors' ``GRC.Rd`` DGP and their RNG, draws
the ALS start, and fits. Python reads the panel and the start and estimates
from them. Two implementations left to their own random number streams land in
two different local optima and the comparison measures nothing, so the start is
drawn once, in R, and handed over.

Why the reference's own optimum is not the target
-------------------------------------------------
``GRC()`` is not reproducible across R processes. With the panel, the start and
the seed all pinned, the ``c1=1`` configuration returned 1217.30913315 in one
run and 1217.46083840 in the next -- and those are precisely the two optima the
R and Python optimisers each reported when first compared. Four calls inside a
single process agree exactly, so the instability is across processes, not
within one. Pinning the value ``GRC()`` happens to return would therefore pin a
coin flip.

Two things are stable, and they are what this case measures.

The objective, value-for-value. Python's :func:`grc_loss` is evaluated on the
solution R reported -- R's own ``A`` and ``cluster`` -- and compared against the
``lossfunc`` R printed for it. That asks whether the two implementations agree
on what the objective *is*, and it does not depend on which optimum either
optimiser found. It agrees to 2e-15 on all three configurations.

The optimiser, by direction. From R's start, the port's ALS reaches a loss no
worse than R's. On the ``c1=2, c2=0`` panel it finds a strictly better one
(1109.77 against 1111.87), which is the bistability above seen from the other
side.

Provenance
----------
``benchmarks/R/fgrc_grc_crossval.R`` regenerates the vendored fits; see
``benchmarks/reference/fgrc_grc_crossval/provenance.json``. The R package is
installed from a clone of the GitHub repository, whose ``src/OptimGRC_C.c``
ships with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_REF = Path(__file__).resolve().parents[1] / "reference" / "fgrc_grc_crossval"


def run() -> dict:
    from sklearn.metrics import adjusted_rand_score

    from mlsynth.utils.clustersc_helpers.rpca.fgrc import (
        _als_once,
        grc_loss,
        indicator,
    )

    summary_path = _REF / "summary.csv"
    if not summary_path.exists():
        raise BenchmarkSkipped(
            f"reference fits missing at {_REF}; regenerate with "
            f"benchmarks/R/fgrc_grc_crossval.R (needs the michioyamamoto/grc "
            f"package installed into an R library)."
        )
    summary = pd.read_csv(summary_path)

    rel_errors, improvements, aris, never_worse = [], [], [], []
    for _, row in summary.iterrows():
        tag = str(row["tag"])
        c1, c2, k = int(row["c1"]), int(row["c2"]), int(row["k"])
        r_loss = float(row["lossfunc"])

        X = pd.read_csv(_REF / f"{tag}_X.csv").values.astype(float)
        A0 = pd.read_csv(_REF / f"{tag}_A0.csv").values.astype(float)
        A_ref = pd.read_csv(_REF / f"{tag}_A.csv").values.astype(float)
        cl_ref = pd.read_csv(_REF / f"{tag}_cl.csv").values.ravel().astype(int)

        # 1. the objective, on R's own solution
        loss_on_ref = grc_loss(X, indicator(cl_ref, k), A_ref, c1, 1.0, 0.0)
        rel_errors.append(abs(loss_on_ref - r_loss) / abs(r_loss))

        # 2. the optimiser, from R's start
        _A, U, loss_py = _als_once(
            X, A0, c1, c2, k, 1.0, 0.0, 100, 100, 1e-5, np.random.default_rng(0)
        )
        improvements.append((r_loss - loss_py) / abs(r_loss))
        never_worse.append(loss_py <= r_loss + 1e-8)
        aris.append(adjusted_rand_score(cl_ref, U.argmax(axis=1) + 1))

    return {
        "n_configs": float(len(summary)),
        "loss_max_rel_error_on_reference_solution": float(max(rel_errors)),
        "n_configs_loss_matches_to_1e_12": float(sum(e < 1e-12 for e in rel_errors)),
        "python_never_worse_than_reference": float(all(never_worse)),
        "min_relative_improvement": float(min(improvements)),
        "n_configs_same_optimum": float(sum(a > 0.999 for a in aris)),
        "mean_ari_vs_reference_partition": float(np.mean(aris)),
    }


# All three configurations are read from vendored R output, and the Python side
# is deterministic given that input, so re-runs are exact. The binding
# assertions are the two stable properties: the objective agrees value-for-value
# on the reference's own solution (zero tolerance on the count, a ceiling on the
# error), and the port's optimiser is never worse than the reference from the
# same start. `n_configs_same_optimum` is 2 of 3 and banded to admit 1 or 3,
# because which optimum is reached is exactly the coin flip documented above --
# asserting 3 would be asserting the reference's instability went our way.
EXPECTED = {
    "n_configs": (3.0, 0.0),
    "loss_max_rel_error_on_reference_solution": (2.2e-15, 1e-12),
    "n_configs_loss_matches_to_1e_12": (3.0, 0.0),          # binding headline
    "python_never_worse_than_reference": (1.0, 0.0),        # binding headline
    "min_relative_improvement": (0.0, 1e-6),                # >= 0 up to solver noise
    "n_configs_same_optimum": (2.0, 1.0),
    "mean_ari_vs_reference_partition": (0.946, 0.15),
}
