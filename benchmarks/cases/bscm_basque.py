"""BSCM cross-validation + Path A: the Basque Country / ETA-terrorism study.

Kim, S., Lee, C., & Gupta, S. (2020), "Bayesian Synthetic Control Methods,"
Journal of Marketing Research 57(5):831-852, propose two Bayesian synthetic
controls that relax the simplex: the donor weights are regularised by a
``horseshoe`` (global-local shrinkage) or a ``spike_slab`` (variable-selection)
prior, so the counterfactual may extrapolate and the weights need not be
non-negative or sum to one. Posterior samples give a counterfactual with
credible bands and an ATT credible interval.

The paper's empirical application (a Washington soda tax) uses proprietary
Nielsen data, so this durable case validates on the public Basque panel
instead. Primary validation is a **cross-validation against the authors'
reference Stan implementation** (``clarencejlee/bscm``: ``Horseshoe_Publish.stan``
and ``Spike_Slab_Publish.stan``, sampled with rstan). On the identical Basque
matching problem (treated = Basque Country, 16 donor regions, treatment 1970),
the pure-numpy Gibbs sampler reproduces the reference posterior:

  =======================  ==============  ==================================
  Quantity (1970)          Stan reference  note
  =======================  ==============  ==================================
  horseshoe ATT            ~-0.72          credibly negative (ETA cost)
  spike-slab ATT           ~-0.69
  pre-treatment RMSE       ~0.004-0.005    near-interpolation (large p, small n)
  weights                  6-7 negative    unconstrained extrapolation
  =======================  ==============  ==================================

The counterfactual paths of the numpy port and the reference Stan agree to a
correlation above 0.999 across both priors (see ``docs/replications/bscm.rst``).
Point estimates are matched; the exact credible-band width is subject to the
global-shrinkage hyperprior detail discussed on the replication page.

Provenance: Kim, Lee & Gupta (2020); the reference Stan ``clarencejlee/bscm``;
Abadie & Gardeazabal (2003) for the original study.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "basque_data.csv")
_TREATED = "Basque Country (Pais Vasco)"


def run() -> dict:
    from mlsynth import BSCM

    df = pd.read_csv(os.path.abspath(_DATA))
    df["treat"] = ((df["regionname"] == _TREATED) & (df["year"] >= 1970)).astype(int)
    base = dict(df=df, outcome="gdpcap", treat="treat", unitid="regionname",
                time="year", n_iter=4000, burn_in=2000, chains=4, seed=2019,
                display_graphs=False)

    hs = BSCM({**base, "prior": "horseshoe"}).fit()
    ss = BSCM({**base, "prior": "spike_slab"}).fit()

    hw = np.array(list(hs.donor_weights.values()))
    return {
        "hs_att": float(hs.att),
        "hs_pre_rmse": float(hs.pre_rmse),
        "hs_n_negative": float((hw < 0).sum()),
        "ss_att": float(ss.att),
        "ss_pre_rmse": float(ss.pre_rmse),
        # spike-slab selects a sparse signal: at least one donor is clearly "in"
        "ss_max_incl": float(max(ss.inclusion_probs.values())),
    }


# Cross-validation targets frozen from the reference Stan (clarencejlee/bscm)
# on the identical Basque 1970 problem. Tolerances bracket Monte-Carlo noise
# across seeds (the samplers agree on the posterior, not bit-for-bit draws).
EXPECTED = {
    "hs_att": (-0.72, 0.35),          # Stan horseshoe ATT ~ -0.716
    "hs_pre_rmse": (0.005, 0.01),     # near-interpolation pre-fit
    "hs_n_negative": (6.0, 4.0),      # unconstrained: several negative weights
    "ss_att": (-0.69, 0.40),          # Stan spike-slab ATT ~ -0.692
    "ss_pre_rmse": (0.005, 0.01),
    "ss_max_incl": (0.85, 0.25),      # a signal donor is clearly selected (>= 0.6)
}
