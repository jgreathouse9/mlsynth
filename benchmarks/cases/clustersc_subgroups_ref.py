"""Cross-validation: the authors' ClusterSC code reproduces their paper's headline.

Companion to ``clustersc_subgroups`` (which validates *mlsynth's* ClusterSC). This
case runs the **authors' own** reference implementation
(https://github.com/srho1/ClusterSC) on the **authors' own** two-subgroup sine
DGP (``generate_sine_dataset_A`` / ``_B``) and confirms it reproduces the paper's
central synthetic-experiment finding (Rho et al. 2025, Section 6.1): donor
clustering substantially lowers the median post-period test MSE versus the
whole-pool RSC / PCR-SC baseline.

Why a separate, reference-only case
-----------------------------------
mlsynth and the authors' code implement ClusterSC differently (mlsynth clusters
on the rank-r-truncated *pre-period* features per the paper's Algorithm 3 and
denoises pre-only with no intercept; the reference clusters on the full,
untruncated panel and fits OLS with an intercept). As a result the two are
strongest in *different* regimes, so a per-target numeric cross-validation
between them is not meaningful. This case therefore cross-validates the
**reference against the paper** (a provenance anchor confirming the comparison
baseline is faithful); ``clustersc_subgroups`` independently validates mlsynth.

Provenance
----------
* Reference: srho1/ClusterSC @ b223e1e (pinned), cloned on demand by
  :mod:`benchmarks.reference.clone_clustersc`; MIT-licensed, imported not vendored.
* DGP: the authors' ``syclib.gendata.generate_sine_dataset_A`` / ``_B``
  (T=10, T0=8, rank 3 per subgroup, n=500 each), seeded with ``np.random.seed``.
* Headline: paper Section 6.1 reports ~50% median OLS test-MSE reduction from
  clustering (n=1000, 500 reps). On a single seed with 30 targets the median
  improvement is large and positive at every noise level.
"""
from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference.clone_clustersc import import_syclib

K1 = 3            # rank of subgroup A
K2 = 3            # rank of subgroup B
K = 2             # number of clusters
T = 10
T0 = 8
N_PER_GROUP = 500
SEED = 0
N_TARGETS = 30
NOISE_LEVELS = [0.10, 0.25, 0.40]


def _ref_test_mse(syclib, dataset: pd.DataFrame, target_id, clustering: bool) -> float:
    """Reference post-period prediction MSE for one placebo target."""
    from syclib import Matrix, SyntheticControl
    from syclib.cluster import ClusterSC

    if clustering:
        csc = ClusterSC(dataset.drop(columns=target_id).T)
        csc.perform_clustering(k=K)
        cluster = csc.predict_target_cluster(dataset[target_id][:T0])
        donors = csc.get_donor_group(cluster)
        cds = pd.concat([pd.DataFrame(dataset[target_id]).T, donors], axis=0)
        M = Matrix(cds.T, T0=T0, target_name=target_id)
        M.denoise(num_sv=K1, transform=False)
    else:
        M = Matrix(dataset.copy(), T0, target_name=target_id)
        M.denoise(num_sv=K1 + K2, transform=False)
    sc = SyntheticControl()
    sc.fit(M.pre_donor, M.pre_target, method="ols")
    return float(sc.predict_and_mse(M.post_donor, M.post_target))


def run() -> dict:
    syclib = import_syclib()
    from syclib.gendata import generate_sine_dataset_A, generate_sine_dataset_B

    out: dict = {}
    improvements: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for noise in NOISE_LEVELS:
            np.random.seed(SEED)
            dsA = generate_sine_dataset_A(N_PER_GROUP, T, noise, num_signals=K1)
            dsB = generate_sine_dataset_B(N_PER_GROUP, T, noise, num_signals=K2)
            dataset = pd.concat([dsA, dsB], axis=1)
            dataset.columns = range(dataset.shape[1])
            targets = list(dsA.columns[:N_TARGETS])
            rsc = [_ref_test_mse(syclib, dataset, t, clustering=False) for t in targets]
            clu = [_ref_test_mse(syclib, dataset, t, clustering=True) for t in targets]
            rsc_med, clu_med = float(np.median(rsc)), float(np.median(clu))
            improve = (rsc_med - clu_med) / rsc_med * 100.0
            out[f"ref_improve_pct_noise{int(noise * 100):03d}"] = improve
            improvements.append(improve)
    out["ref_clustering_wins_all"] = float(all(i > 0 for i in improvements))
    out["ref_min_improve_pct"] = float(min(improvements))
    return out


def _mlsynth_test_mse(dataset: pd.DataFrame, target_id, clustering: bool) -> float:
    """mlsynth CLUSTERSC post-period prediction MSE for one placebo target.

    Mirrors ``_ref_test_mse`` but drives mlsynth's public ``CLUSTERSC`` API on
    the same ``time x unit`` panel: whole-pool RSC denoises at the pooled rank
    ``K1 + K2``, ClusterSC at the subgroup rank ``K1``.
    """
    from mlsynth import CLUSTERSC

    long = (dataset.reset_index()
            .melt(id_vars="index", var_name="unit", value_name="y")
            .rename(columns={"index": "time"}))
    long["treat"] = ((long["unit"] == target_id) & (long["time"] >= T0)).astype(int)
    rank = K1 if clustering else K1 + K2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC({
            "df": long, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "method": "pcr", "clustering": clustering, "pcr_objective": "OLS",
            "rank": rank, "rank_method": "fixed",
            "k_clusters": (K if clustering else None),
            "project_denoised": True, "display_graphs": False,
        }).fit()
    gap = np.asarray(res.gap).ravel()
    return float(np.mean(gap[T0:] ** 2))


def comparison() -> dict:
    """mlsynth CLUSTERSC vs the authors' code, clustering test-MSE gain by noise.

    On the authors' own two-subgroup sine DGP, pairs -- at each noise level --
    the median post-period test-MSE improvement from clustering (whole-pool RSC
    minus ClusterSC, as a percent of RSC) computed by mlsynth and by the
    reference. Loads the reference clone first so a blocked git/network skips
    cleanly via ``BenchmarkSkipped``.
    """
    from benchmarks.reference.clone_clustersc import import_syclib

    syclib = import_syclib()        # triggers the on-demand clone; skips if blocked
    from syclib.gendata import generate_sine_dataset_A, generate_sine_dataset_B

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for noise in NOISE_LEVELS:
            np.random.seed(SEED)
            dsA = generate_sine_dataset_A(N_PER_GROUP, T, noise, num_signals=K1)
            dsB = generate_sine_dataset_B(N_PER_GROUP, T, noise, num_signals=K2)
            dataset = pd.concat([dsA, dsB], axis=1)
            dataset.columns = range(dataset.shape[1])
            targets = list(dsA.columns[:N_TARGETS])

            ml_rsc = [_mlsynth_test_mse(dataset, t, clustering=False) for t in targets]
            ml_clu = [_mlsynth_test_mse(dataset, t, clustering=True) for t in targets]
            ref_rsc = [_ref_test_mse(syclib, dataset, t, clustering=False) for t in targets]
            ref_clu = [_ref_test_mse(syclib, dataset, t, clustering=True) for t in targets]

            ml_improve = (np.median(ml_rsc) - np.median(ml_clu)) / np.median(ml_rsc) * 100.0
            ref_improve = (np.median(ref_rsc) - np.median(ref_clu)) / np.median(ref_rsc) * 100.0
            rows.append({"quantity": f"clustering_improve_pct/noise{int(noise * 100):03d}",
                         "mlsynth": round(float(ml_improve), 6),
                         "reference": round(float(ref_improve), 6)})

    from benchmarks.reference.clone_clustersc import _COMMIT
    cfg = {"outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
           "method": "pcr", "pcr_objective": "OLS", "rank_method": "fixed",
           "project_denoised": True}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "CLUSTERSC", "config": cfg},
        "reference": {"impl": "authors' ClusterSC code (srho1/ClusterSC syclib), "
                              "cloned on demand by benchmarks.reference.clone_clustersc",
                      "version": f"srho1/ClusterSC @ {_COMMIT[:7]}"},
    }


# The authors' code must show clustering winning at every noise level on its own
# DGP (the paper's headline). Per-noise values are pinned with wide bands: the
# single-seed median over 30 targets is noisier than the paper's 500-rep median,
# and k-means is mildly sklearn-version-sensitive. The binding assertions are
# `ref_clustering_wins_all == 1` and a positive floor on the smallest gain.
EXPECTED = {
    "ref_improve_pct_noise010": (38.8, 22.0),
    "ref_improve_pct_noise025": (71.3, 22.0),
    "ref_improve_pct_noise040": (70.6, 22.0),
    "ref_clustering_wins_all": (1.0, 0.0),
    "ref_min_improve_pct": (38.8, 30.0),   # lower bound stays well above 0
}
