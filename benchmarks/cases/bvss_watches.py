"""Cross-validation: mlsynth BVSS vs the authors' own Gibbs sampler (China, p>n).

Cross-validation against the canonical implementation. mlsynth's BVSS
(:class:`mlsynth.estimators.bvss.BVSS`, Bayesian Synthetic Control with a Soft
Simplex Constraint -- Xu & Zhou 2025, arXiv:2503.06454) is a direct port of the
authors' two-coordinate Gibbs sampler; this case checks the port against that
original R, taken verbatim from the authors' fsPDA replication script
(``example2_fspda_2.R``, the China anti-corruption empirical example). The panel
is the in-repo ``china_watches_long.csv`` -- one treated import series
(``watches``), 87 donor product-category series, 35 pre-treatment months: the
high-dimensional ``p > n`` regime the soft-simplex BVS is built for.

Two complementary checks
------------------------
1. Deterministic engine (exact). The sampler is Metropolis-within-Gibbs, so its
   draws are stochastic, but every draw is built from deterministic math
   primitives -- the posterior covariance log-determinant ``VM``, the quadratic
   and bilinear forms ``RSS`` / ``RSS2``, the model-complexity term ``AM`` and
   the marginal log-likelihood ``loglike`` (Eqs. (4)-(5) and Lemmas S1-S2 of the
   paper). On a fixed ``(gamma, tau, mu, phi)`` mlsynth reproduces the authors'
   R primitives to ~1e-9, so the two samplers draw from the identical target.

2. Posterior summary (within Monte-Carlo error). Both samplers, seeded and run
   on the same panel, put the posterior-mean ATT at about -0.021 (monthly import
   growth) with a negative 95% credible set -- the anti-corruption campaign
   depressed luxury-watch imports -- and select a sparse model (~5-6 donors out
   of 87). mlsynth's ATT (-0.0212) sits within Monte-Carlo error of the R
   reference (-0.0208); the two use independent RNG streams, so this is a
   distributional agreement, not a value-for-value one.

Reference (live captured run)
-----------------------------
``benchmarks/reference/bvss_watches/reference.R`` carries the authors' sampler
primitives verbatim and runs both blocks (the fixed-input engine dump and a
seeded Gibbs run) on the in-repo panel; the captured ``reference.json`` holds the
five primitive values and the posterior summary, pinned here via
:func:`reference_value` / :func:`load_reference`. Only ``truncnorm`` is needed
for the reference (the sampler's one non-base dependency; ``fsPDA`` is only the
authors' data loader and is not used). Regenerate with
``python benchmarks/reference/generate.py bvss_watches``.

Provenance
----------
* Data: ``basedata/china_watches_long.csv`` -- the fsPDA-style China
  anti-corruption panel (Shi & Huang 2023): treated ``watches`` import series +
  87 donor category series, monthly, 35 pre-treatment periods.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED = "watches"
_THETA = 0.25

_REF = load_reference("bvss_watches")
REF_ATT = reference_value("bvss_watches", "post_att")
_DET_KEYS = ["det_vm_logdet", "det_rss", "det_rss2", "det_am", "det_loglike"]


def _demeaned_panel():
    from mlsynth.utils.bvss_helpers.posterior import VM  # noqa: F401 (import check)

    d = pd.read_csv(_BASE / "china_watches_long.csv")
    units = list(dict.fromkeys(d["unit"]))
    times = sorted(d["time"].unique())
    W = d.pivot(index="time", columns="unit", values="y").reindex(index=times, columns=units)
    donors = [u for u in units if u != _TREATED]
    tr = d[d["unit"] == _TREATED]
    M = int((tr["treat"] == 0).sum())
    Y0 = W.loc[times[:M], _TREATED].to_numpy().reshape(-1, 1)
    X0 = W.loc[times[:M], donors].to_numpy()
    mean_X = X0.mean(0)
    mean_Y = float(Y0.mean())
    Y = (Y0 - mean_Y).ravel()
    X = X0 - mean_X
    return d, Y, X, X.T @ X


def _deterministic_primitives():
    """mlsynth's sampler primitives on the reference's fixed (gamma, tau, mu, phi)."""
    from scipy.linalg import det
    from mlsynth.utils.bvss_helpers.posterior import VM, RSS, RSS2, AM, loglike

    _d, Y, X, Gram = _demeaned_panel()
    N = X.shape[1]
    gamma = np.zeros(N, dtype=int)
    gamma[:5] = 1
    tau, phi = 0.5, 1.3
    mu = np.zeros(N)
    mu[:5] = [0.4, 0.25, 0.15, 0.1, 0.1]
    z = Y - X @ mu
    return {
        "det_vm_logdet": float(np.log(det(VM(gamma, tau, Gram)))),
        "det_rss": RSS(gamma, tau, z, X, Gram),
        "det_rss2": RSS2(gamma, tau, X[:, 1] - X[:, 0], z, X, Gram),
        "det_am": AM(gamma, tau, _THETA, Gram, N),
        "det_loglike": loglike(gamma, tau, mu, phi, Y, X, Gram),
    }


def _mlsynth_bvss_att(df):
    from mlsynth import BVSS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = BVSS({
            "df": df, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "n_iter": 50, "burn_in": 25, "theta": _THETA,
            "kappa1": 1.0, "kappa2": 1.0, "seed": 1, "display_graphs": False,
        }).fit()
    return float(res.att)


def run() -> dict:
    prim = _deterministic_primitives()
    det_max_abs_diff = max(abs(prim[k] - reference_value("bvss_watches", k)) for k in _DET_KEYS)

    df, _Y, _X, _G = _demeaned_panel()
    att_ml = _mlsynth_bvss_att(df)

    return {
        # (1) exact: mlsynth's Gibbs primitives vs the authors' R, value for value.
        "det_max_abs_diff_vs_R": det_max_abs_diff,
        # (2) distributional: posterior-mean ATT within Monte-Carlo error of R.
        "mls_att": att_ml,
        "att_abs_diff_vs_R": float(abs(att_ml - REF_ATT)),
        "att_negative": 1.0 if att_ml < 0 else 0.0,
    }


def comparison() -> dict:
    """mlsynth BVSS vs the authors' own Gibbs sampler, quantity by quantity.

    The deterministic engine (``VM`` log-det, ``RSS``, ``RSS2``, ``AM``,
    ``loglike``) is laid value-for-value against the authors' R on a fixed
    ``(gamma, tau, mu, phi)``; the posterior-mean ATT is laid against a seeded R
    Gibbs run (within Monte-Carlo error). The reference side is a live captured
    run in ``benchmarks/reference/bvss_watches/``. Returns ``{"rows": [...],
    "mlsynth_call": {...}, "reference": {...}}``.
    """
    prim = _deterministic_primitives()
    df, _Y, _X, _G = _demeaned_panel()
    att_ml = _mlsynth_bvss_att(df)

    rows = [{"quantity": k, "mlsynth": round(prim[k], 9),
             "reference": round(reference_value("bvss_watches", k), 9)}
            for k in _DET_KEYS]
    rows.append({"quantity": "posterior-mean ATT",
                 "mlsynth": round(att_ml, 6), "reference": round(REF_ATT, 6)})

    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "BVSS (two-coordinate Gibbs, soft simplex)",
                         "config": {"outcome": "y", "treat": "treat",
                                    "unitid": "unit", "time": "time",
                                    "theta": _THETA, "n_iter": 50, "seed": 1}},
        "reference": {"impl": "authors' two-coordinate Gibbs (example2_fspda_2.R "
                              "primitives), live run, captured",
                      "version": "Xu & Zhou (2025) arXiv:2503.06454, on in-repo "
                                 "china_watches_long.csv"},
    }


# The Gibbs draws are stochastic, but their deterministic primitives (VM log-det,
# RSS, RSS2, AM, loglike) are exact: mlsynth reproduces the authors' R to ~1e-9
# on a fixed (gamma, tau, mu, phi), so both samplers target the identical
# posterior. The seeded posterior-mean ATT agrees within Monte-Carlo error
# (mlsynth -0.0212 vs R -0.0208, independent RNG streams), and is negative -- the
# anti-corruption campaign depressed luxury-watch imports. mls_att is
# deterministic under the fixed seed (n_iter=50); det/att targets are pinned from
# the live captured R run via reference_value.
EXPECTED = {
    "det_max_abs_diff_vs_R": (0.0, 1e-6),      # exact sampler-engine match
    "mls_att": (-0.021234, 1e-4),              # deterministic at seed=1, n_iter=50
    "att_abs_diff_vs_R": (0.0, 0.01),          # within Monte-Carlo error of R
    "att_negative": (1.0, 0.0),                # anti-corruption depresses imports
}
