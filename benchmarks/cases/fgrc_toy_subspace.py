"""Path B benchmark: fGRC's subspace separation recovers cluster structure that
is invisible to ordinary clustering, on the Yamamoto-Hwang toy example.

CLUSTERSC's ``cluster_method="fgrc"`` donor-selection step is a port of the
generalized reduced clustering of Yamamoto and Hwang (2014, 2017). Because that
clustering is not part of Bayani's (2021) original RPCA-SC, it is validated here
against its *own* authors' documented toy example rather than against Bayani.

The toy example (authors' ``GRC.Rd``)
-------------------------------------
The reference package's help page (https://github.com/michioyamamoto/grc,
``man/GRC.Rd``) ships this example: 300 subjects and 5 variables, with three
clusters of 100 planted in a *one-dimensional* subspace ::

    X1 <- rnorm(100, -10, 1); X2 <- rnorm(100, 0, 1); X3 <- rnorm(100, 10, 1)
    X  <- cbind(c(X1, X2, X3), matrix(rnorm(300 * 4), 300, 4))
    X  <- scale(X)
    res <- GRC(X, 1, 0, 3)          # comp1=1, comp2=0, k=3

Only the first variable carries the grouping; the other four are pure noise.
The subtlety is that ``scale()`` gives every column unit variance, so after
scaling each of the four noise variables carries as much spread as the single
structured variable -- the cluster signal is a needle in a five-dimensional
haystack. Ordinary k-means on the raw variables is swamped by the noise
dimensions; GRC's contribution is to *find* the one-dimensional subspace on
which the groups separate and cluster there.

What this case validates
------------------------
The port's GRC engine (:func:`mlsynth.utils.clustersc_helpers.rpca.fgrc.optim_grc`,
the core that the functional wrapper ``fgrc_cluster`` calls) is run on the
authors' exact DGP, ``optim_grc(X, c1=1, c2=0, k=3)``, and compared against
k-means over the same Lloyd engine in two feature spaces: the raw five
variables (the naive baseline) and the single signal variable (the oracle that
knows the subspace). Recovery is scored by the adjusted Rand index (ARI)
against the planted 100/100/100 partition.

* On a representative draw (seed 0) GRC recovers the three groups exactly
  (``grc_seed0_ari == 1``) and concentrates its loading on the signal variable
  (``grc_seed0_signal_loading ~ 1``) -- it identifies the one-dimensional
  cluster subspace, reproducing the documented ``res$cluster`` / ``res$A1``.
* Averaged over draws, GRC's subspace search roughly doubles the recovery ARI
  of naive five-variable k-means (``grc_ari_mean`` ~ 0.71 vs
  ``kmeans_full_ari_mean`` ~ 0.30) and moves most of the way to the oracle
  (``oracle_ari_mean == 1``). The binding assertion is that GRC beats naive
  k-means on average (``grc_beats_kmeans == 1``), with the advantage floored
  well above zero.

Not every draw recovers: because ``scale()`` inflates the noise variables, a
minority of finite-sample draws admit a spurious lower-loss partition, so GRC's
per-draw ARI is bimodal and its mean sits below 1. This is a property of the
example, not the port -- the R help page itself defaults to a single random
start and only notes that "multiple random initial starts may work". The port's
numerical fidelity to the authors' compiled ``OptimGRC_C`` routine is pinned
separately, to machine precision, in ``mlsynth/tests/test_fgrc.py``.

Provenance
----------
* Method / DGP: Yamamoto & Hwang (2014), *A general formulation of cluster
  analysis with dimension reduction and subspace separation*, Behaviormetrika
  41(1):115-129; and (2017), *Dimension-reduced clustering of functional data
  via subspace separation*, J. Classification 34(2):294-326. Toy DGP from
  ``michioyamamoto/grc`` ``man/GRC.Rd``.
* Self-contained: reproduces the DGP with NumPy's version-stable ``default_rng``
  stream and clusters with the port's own Lloyd engine (no R, no sklearn), so
  the case is deterministic and needs no reference toolchain.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import List

import numpy as np

from mlsynth.utils.clustersc_helpers.rpca.fgrc import optim_grc, _kmeans_best

N_PER_GROUP = 100          # authors' N.sub.k
N_NOISE = 4                # authors' matrix(rnorm(300 * 4), 300, 4)
N_SEEDS = 15               # replications for the aggregate metrics
GRC_N_RANDOM = 10          # ALS restarts (modest: biases the comparison *against* GRC)
GRC_NSTART = 20            # k-means restarts inside each U-update
KM_NSTART = 30             # k-means restarts for the baselines
TRUTH = np.repeat([1, 2, 3], N_PER_GROUP)


def _scale(X: np.ndarray) -> np.ndarray:
    """R ``scale()``: centre each column and divide by its sd (ddof=1)."""
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


def _toy_panel(seed: int) -> np.ndarray:
    """The authors' ``GRC.Rd`` DGP: one structured variable (three N(mu, 1)
    groups at mu in {-10, 0, 10}) plus four N(0, 1) noise variables, scaled."""
    rng = np.random.default_rng(seed)
    signal = np.concatenate([
        rng.normal(-10.0, 1.0, N_PER_GROUP),
        rng.normal(0.0, 1.0, N_PER_GROUP),
        rng.normal(10.0, 1.0, N_PER_GROUP),
    ])
    noise = rng.normal(size=(3 * N_PER_GROUP, N_NOISE))
    return _scale(np.column_stack([signal, noise]))


def _ari(labels: np.ndarray, truth: np.ndarray = TRUTH) -> float:
    """Adjusted Rand index (Hubert-Arabie), self-contained."""
    a, b = np.asarray(labels), np.asarray(truth)
    contingency: dict = {}
    for pair in zip(a.tolist(), b.tolist()):
        contingency[pair] = contingency.get(pair, 0) + 1
    comb2 = lambda x: x * (x - 1) // 2
    sum_cells = sum(comb2(v) for v in contingency.values())
    sum_a = sum(comb2(v) for v in Counter(a.tolist()).values())
    sum_b = sum(comb2(v) for v in Counter(b.tolist()).values())
    total = comb2(a.size)
    expected = sum_a * sum_b / total
    return float((sum_cells - expected) / ((sum_a + sum_b) / 2 - expected))


def _kmeans_ari(features: np.ndarray, seed: int) -> float:
    """ARI of the port's Lloyd k-means (k=3) run in a given feature space."""
    F = features if features.ndim == 2 else features[:, None]
    labels = _kmeans_best(F, 3, KM_NSTART, np.random.default_rng(seed))
    return _ari(labels + 1)


def run() -> dict:
    grc_ari: List[float] = []
    km_full_ari: List[float] = []
    oracle_ari: List[float] = []
    seed0_signal_loading = None

    for seed in range(N_SEEDS):
        X = _toy_panel(seed)
        labels, A, _loss = optim_grc(
            X, c1=1, c2=0, k=3, n_random=GRC_N_RANDOM, nstart=GRC_NSTART, seed=seed,
        )
        grc_ari.append(_ari(labels))
        km_full_ari.append(_kmeans_ari(X, seed))          # naive: all five variables
        oracle_ari.append(_kmeans_ari(X[:, [0]], seed))   # oracle: signal variable only
        if seed == 0:
            # share of the estimated 1-D loading on the signal variable
            seed0_signal_loading = float(abs(A[0, 0]) / np.sqrt(np.sum(A[:, 0] ** 2)))

    grc_mean = float(np.mean(grc_ari))
    km_mean = float(np.mean(km_full_ari))
    return {
        "grc_seed0_ari": grc_ari[0],
        "grc_seed0_signal_loading": seed0_signal_loading,
        "grc_ari_mean": grc_mean,
        "kmeans_full_ari_mean": km_mean,
        "oracle_ari_mean": float(np.mean(oracle_ari)),
        "grc_advantage": grc_mean - km_mean,
        "grc_beats_kmeans": float(grc_mean > km_mean),
    }


# Deterministic: NumPy's PCG64 ``default_rng`` streams are version-stable and the
# clustering uses the port's own Lloyd engine (no sklearn), so re-runs are exact.
# Bands are generous to absorb any future drift; the binding assertions are
# `grc_beats_kmeans == 1` (subspace separation beats naive k-means on average),
# an advantage floored well above zero, exact recovery on the representative
# draw, and the oracle confirming the structure is recoverable in the subspace.
EXPECTED = {
    "grc_seed0_ari": (1.0, 0.05),                # recovers the 3 groups on seed 0
    "grc_seed0_signal_loading": (0.997, 0.03),   # identifies the 1-D cluster subspace
    "grc_ari_mean": (0.707, 0.15),
    "kmeans_full_ari_mean": (0.302, 0.15),
    "oracle_ari_mean": (1.0, 0.05),              # structure is perfectly recoverable in-subspace
    "grc_advantage": (0.405, 0.20),              # lower bound stays well above 0
    "grc_beats_kmeans": (1.0, 0.0),              # binding headline
}
