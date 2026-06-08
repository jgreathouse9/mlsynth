"""Path B benchmark: ClusterSC beats whole-pool RSC under donor subgroups.

Reproduces the central claim of Rho, Tang, Bergam, Cummings & Misra (2025),
*ClusterSC: Advancing Synthetic Control with Donor Selection* (Section 6.1):
when the donor pool splits into structurally distinct subgroups, restricting the
synthetic control to the target's own subgroup (ClusterSC) lowers the
post-intervention prediction error relative to using the whole donor pool (plain
RSC / PCR-SC). Both modes are exercised through mlsynth's single ``CLUSTERSC``
estimator -- ``clustering=False`` is RSC, ``clustering=True`` is ClusterSC.

The regime
----------
mlsynth's RSC denoises the *pre-period* donor matrix to a fixed rank before
regressing, so -- unlike the authors' reference code -- it is robust to the raw
donor count ``n``. The lever that actually exercises the paper's
curse-of-dimensionality argument is the **pooled signal rank**: with ``K``
well-separated subgroups of rank ``r`` each, the pooled donor matrix has rank
``K*r``. Once ``K*r`` exceeds the pre-period length ``T0`` the whole-pool fit
must under-denoise, while each subgroup stays low-rank and well-conditioned. We
use ``K=6`` subgroups of rank 3 with ``T0=8`` (pooled rank 18 >> 8), the
high-dimensional-subgroup regime the paper targets.

Placebo design: the DGP plants no treatment effect, so the post-period gap is
pure prediction error and its mean square is the test MSE the paper reports.

Provenance
----------
* DGP: :func:`mlsynth.utils.clustersc_helpers.simulation.simulate_subgroup_panel`
  -- a faithful ``K``-subgroup generalisation of the authors' two-subgroup sine
  mixture (``generate_sine_dataset_A`` / ``_B`` in https://github.com/srho1/ClusterSC).
* Headline (paper Section 6.1, Figure on synthetic OLS): donor clustering reduces
  median test MSE; the gain is largest where the whole-pool fit is most
  ill-posed. A per-replication cross-validation against the authors' own RSC /
  ClusterSC code lives in ``clustersc_subgroups_ref``.
"""
from __future__ import annotations

import warnings
from typing import List

import numpy as np

from mlsynth.utils.clustersc_helpers.simulation import simulate_subgroup_panel

K = 6
RANK = 3
N_PER_GROUP = 150
T = 16
T0 = 8
SEED = 0
N_TARGETS = 25
NOISE_LEVELS = [0.10, 0.25, 0.40]


def _test_mse(panel, target_id: int, clustering: bool) -> float:
    """Post-period prediction MSE for one placebo target via the public API."""
    from mlsynth import CLUSTERSC

    wide = panel.wide
    long = (wide.reset_index()
            .melt(id_vars="index", var_name="unit", value_name="y")
            .rename(columns={"index": "time"}))
    long["treat"] = ((long["unit"] == target_id) & (long["time"] >= panel.T0)).astype(int)
    # Whole-pool RSC denoises at the pooled rank K*r; ClusterSC at the subgroup rank r.
    rank = panel.rank if clustering else panel.K * panel.rank
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC({
            "df": long, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "method": "pcr", "clustering": clustering, "pcr_objective": "OLS",
            "rank": rank, "rank_method": "fixed",
            "k_clusters": (panel.K if clustering else None),
            "project_denoised": True, "display_graphs": False,
        }).fit()
    gap = np.asarray(res.gap).ravel()
    return float(np.mean(gap[panel.T0:] ** 2))


def run() -> dict:
    out: dict = {}
    improvements: List[float] = []
    for noise in NOISE_LEVELS:
        panel = simulate_subgroup_panel(
            K=K, rank=RANK, n_per_group=N_PER_GROUP, T=T, T0=T0, noise=noise, seed=SEED,
        )
        rsc = [_test_mse(panel, t, clustering=False) for t in panel.target_ids[:N_TARGETS]]
        clu = [_test_mse(panel, t, clustering=True) for t in panel.target_ids[:N_TARGETS]]
        rsc_med, clu_med = float(np.median(rsc)), float(np.median(clu))
        improve = (rsc_med - clu_med) / rsc_med * 100.0
        out[f"improve_pct_noise{int(noise * 100):03d}"] = improve
        improvements.append(improve)
    # 1.0 iff ClusterSC's median test MSE beats whole-pool RSC at *every* noise level.
    out["clustering_wins_all"] = float(all(i > 0 for i in improvements))
    out["min_improve_pct"] = float(min(improvements))
    return out


# Deterministic (fixed simulator seed + KMeans random_state=0) => exact re-runs.
# The per-noise improvements are pinned with a generous +/-10 band to absorb
# sklearn / numpy version drift in the SVD / k-means; the headline assertions are
# `clustering_wins_all == 1` (ClusterSC beats RSC at every noise) and a positive
# floor on the smallest improvement.
EXPECTED = {
    "improve_pct_noise010": (60.8, 10.0),
    "improve_pct_noise025": (43.2, 10.0),
    "improve_pct_noise040": (24.3, 10.0),
    "clustering_wins_all": (1.0, 0.0),
    "min_improve_pct": (24.3, 14.0),   # lower bound stays well above 0
}
