r"""BPSCS self-contained validation: effect recovery + distance-based shrinkage.

mlsynth's :class:`mlsynth.BPSCS` (Penalized Synthetic Control under Spillovers,
Fernandez-Morales, Oganisian & Lee 2026) down-weights spatially close -- and
therefore likely spillover-contaminated -- donors through a covariate-and-distance
shrinkage prior. The paper's empirical panel (NielsenIQ retail scanner data) is
proprietary, and the authors' example data and Stan reference are GPL-licensed, so
this durable case is self-contained: it simulates a spatial panel with a known
treatment effect and a set of spatially-close donors contaminated by a post-period
spillover, then checks that BPSCS (a) recovers the effect sign and rough magnitude
and (b) shrinks the contaminated near donors harder than the clean far donors --
the method's whole point. The live cell-for-cell cross-validation against the
authors' GPL Stan (fetched at runtime) is documented on the replication page.

Requires the ``[bayes]`` extra (NumPyro); skips gracefully when it is absent.

Provenance: Fernandez-Morales, Oganisian & Lee (2026), Biometrics 82(2), ujag054;
reference repo github.com/estfernan/penalized-sc-spillovers (GPL-3, not shipped).
"""
from __future__ import annotations

import warnings

_EFFECT = -4.0
_SPILL = 3.0
_NEAR = 0.30            # donors within this distance of the treated unit are contaminated


def _panel(n_units=12, T=20, T0=14, seed=0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n_units, 2)); coords[0] = [0.0, 0.0]
    dist0 = np.linalg.norm(coords - coords[0], axis=1)
    f = np.cumsum(rng.standard_normal(T))                      # smooth common factor
    rows, near = [], []
    for i in range(n_units):
        load = 1.0 + rng.standard_normal() * 0.4
        cov = coords[i] + rng.standard_normal(2) * 0.05
        y = 20.0 + load * f + rng.standard_normal(T) * 0.3
        if i == 0:
            y[T0:] += _EFFECT
        elif dist0[i] < _NEAR:
            y[T0:] += _SPILL                                   # spillover contamination
            near.append(f"u{i:02d}")
        for t in range(T):
            rows.append({"unit": f"u{i:02d}", "t": t, "y": float(y[t]),
                         "treat": int(i == 0 and t >= T0),
                         "cov1": float(cov[0]), "cov2": float(cov[1]),
                         "lat": float(coords[i, 0]), "lon": float(coords[i, 1])})
    return pd.DataFrame(rows), near


def _fit(prior):
    from mlsynth import BPSCS
    try:
        import numpyro  # noqa: F401
    except ImportError as exc:
        from benchmarks.compare import BenchmarkSkipped
        raise BenchmarkSkipped("BPSCS needs NumPyro (pip install 'mlsynth[bayes]').") \
            from exc
    df, near = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = BPSCS({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit", "time": "t",
            "covariates": ["cov1", "cov2"], "coords": ["lat", "lon"],
            "prior": prior, "kappa_d": 0.0, "n_warmup": 800, "n_samples": 800,
            "n_chains": 2, "target_accept": 0.95, "max_tree_depth": 12, "seed": 0,
        }).fit()
    return res, near


def run() -> dict:
    import numpy as np

    res_dhs, near = _fit("dhs")
    res_ds2, _ = _fit("ds2")
    w = res_dhs.weights.donor_weights
    near_abs = np.mean([abs(w[k]) for k in near]) if near else np.nan
    far_abs = np.mean([abs(v) for k, v in w.items() if k not in near])
    return {
        "att_dhs": float(res_dhs.att),
        "att_ds2": float(res_ds2.att),
        "att_negative": 1.0 if (res_dhs.att < 0 and res_ds2.att < 0) else 0.0,
        # near (contaminated) donors shrunk harder than far (clean) donors
        "near_shrunk_more": 1.0 if near_abs < far_abs else 0.0,
        "pre_rmse_dhs": float(res_dhs.fit_diagnostics.rmse_pre),
    }


# Self-contained recovery + shrinkage check on a simulated spatial panel (treated
# effect -4, near donors contaminated by a +3 spillover). Tolerances are loose:
# the free-running counterfactual is noisy, so we assert direction and rough
# magnitude, not a tight point match.
EXPECTED = {
    "att_dhs": (-4.0, 3.0),            # recovers the effect (widely bracketed)
    "att_ds2": (-4.0, 3.0),
    "att_negative": (1.0, 0.0),        # both priors give a negative effect
    "near_shrunk_more": (1.0, 0.0),    # contaminated near donors down-weighted
    "pre_rmse_dhs": (0.0, 2.0),        # tracks the pre-period
}
