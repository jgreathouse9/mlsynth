"""RPCA-SC reference (Bayani), driving the author's own vendored code.

Cross-validation reference for mlsynth's ``CLUSTERSC`` RPCA-SC family. Rather
than reimplement the method, this module runs Mani Bayani's *own* Robust PCA
routine -- loaded verbatim from the vendored dissertation source at
``vendor/bayani_rpca_synth/RPCA_2.py`` (see the ``NOTICE.md`` there for
attribution) -- and wraps it in the same non-negative fit and West-Germany
cluster his code uses.

The Robust PCA decomposition (Principal Component Pursuit; Candès, Li, Ma &
Wright 2011) is deterministic, and the NNLS is convex, so the reference is
recomputed live rather than pinned to a captured constant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import cvxpy as cp
import numpy as np
import pandas as pd

_VENDOR = Path(__file__).resolve().parent / "vendor" / "bayani_rpca_synth"
_TREATMENT_YEAR = 1990

# West-Germany cluster from the author's FPCA.R (recorded in his RPCA_2.py's own
# ``countries`` list): West Germany plus the eleven donors his functional-PCA
# k-means groups with it. mlsynth reaches the same cluster via its own SVD-based
# clustering; hardcoding it here avoids re-running the R functional-PCA step.
WEST_GERMANY_CLUSTER: List[str] = [
    "UK", "Austria", "Belgium", "Denmark", "France", "West Germany",
    "Italy", "Netherlands", "Norway", "Japan", "Australia", "New Zealand",
]


def _load_authors_rpca() -> Callable[[np.ndarray], tuple]:
    """Return the author's verbatim ``RPCA`` function from the vendored source.

    Executes only the function-definition block of ``RPCA_2.py`` (up to the
    first top-level side effect), in a NumPy namespace, so the reference runs the
    author's exact code without importing the script's plotting side effects.
    """
    source = (_VENDOR / "RPCA_2.py").read_text()
    start = source.index("def shrink")
    end = source.index("\nabadie = np.array")     # first top-level side effect
    namespace: Dict[str, object] = {"np": np}
    exec(compile(source[start:end], str(_VENDOR / "RPCA_2.py"), "exec"), namespace)
    return namespace["RPCA"]                       # type: ignore[return-value]


def rpca_sc_west_germany() -> Dict[str, object]:
    """Run Bayani's RPCA-SC for West Germany on the author's vendored panel.

    Returns a dict with ``weights`` (donor -> weight), the per-year
    ``counterfactual``, ``years``, ``pre_rmse`` and ``att`` (mean post-treatment
    gap), using the author's own ``RPCA`` routine.
    """
    robust_pca = _load_authors_rpca()

    wide = pd.read_csv(_VENDOR / "Data_Germany.csv", index_col=0)
    years = np.array([int(c) for c in wide.columns])
    outcomes = wide.loc[WEST_GERMANY_CLUSTER].to_numpy(float)   # 12 x T, WG row 5
    treated_row = WEST_GERMANY_CLUSTER.index("West Germany")
    n_pre = int(np.sum(years < _TREATMENT_YEAR))

    donor_matrix = np.delete(outcomes, treated_row, axis=0)     # 11 donors x T
    low_rank, _sparse = robust_pca(donor_matrix)                # author's PCP
    low_rank_pre = low_rank[:, :n_pre]
    treated_pre = outcomes[treated_row, :n_pre]

    weights = cp.Variable(donor_matrix.shape[0])
    cp.Problem(cp.Minimize(cp.sum_squares(treated_pre - weights @ low_rank_pre)),
               [weights >= 0]).solve()
    w = np.asarray(weights.value, dtype=float).flatten()
    counterfactual = low_rank.T @ w

    donors = [c for c in WEST_GERMANY_CLUSTER if c != "West Germany"]
    treated = outcomes[treated_row]
    pre_rmse = float(np.sqrt(np.mean((treated[:n_pre] - counterfactual[:n_pre]) ** 2)))
    att = float(np.mean((treated - counterfactual)[n_pre:]))
    return {
        "weights": {d: float(wi) for d, wi in zip(donors, w)},
        "counterfactual": counterfactual,
        "years": years,
        "pre_rmse": pre_rmse,
        "att": att,
    }
