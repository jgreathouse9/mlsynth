"""Path B: L-infinity SC vs classic SC under latent factors (Wang-Xing-Ye 2025).

Reproduces the *direction* of Wang, Xing & Ye (2025), Section 5 / Table 4: under
a two-factor DGP the dense L-infinity SC beats the sparse classic SC at
estimating the ATT when the true donor weights are themselves dense, while the
mixed L1+L-infinity method wins when the truth is sparse.

DGP (Table 3, p. 13), with deterministic loadings ``lambda_j = (j-1)/J`` and
factors ``F ~ N(0, I_2)``; treated outcome ``Y1 = mu + Y0 w0 + u`` with a
constant post-treatment effect ``delta = 3``:

* DGP 1 -- equal weights ``w0 = 1/J`` (SC's home turf).
* DGP 2 -- ``w0_j ~ U(-3/J, 3/J)`` (dense).
* DGP 3 -- ``w0_j ~ (Beta(0.2,0.2) - 0.5) * 3/J`` (dense).
* DGP 4 -- half the DGP-3 draw, half zeros, shuffled (sparse).

Metric: RMSE of the ATT estimator over ``B`` replications, ``RMSE =
sqrt(mean((ATT_hat - 3)^2))``.

Runtime note
------------
The paper uses ``B = 2000`` and CV-selected ``lambda``. For a CI-affordable
durable guard we use ``B = 50`` and a fixed penalty, asserting the **ordering**
(L-infinity < SC in the dense DGPs 2/3; L1+L-infinity < SC in the sparse DGP 4),
not the 4-decimal Table-4 cells. The penalized solve now runs through the
OSQP / Gram fast path (``fast_solve``), so a larger ``B`` is affordable for a
manual durable run; the full ``B = 2000`` Table-4 cells remain a manual target.

Provenance / scenario
---------------------
* Paper Monte Carlo (scenario 1, Path B). Self-contained -- always runs.
"""
from __future__ import annotations

import warnings

import numpy as np

J, T0, T1 = 30, 100, 10
DELTA = 3.0
B = 50
LAM = 4.0          # fixed mlsynth penalty for the L-infinity methods
SEED = 7


def _weights(rng, dgp):
    if dgp == 1:
        return np.full(J, 1.0 / J)
    if dgp == 2:
        return rng.uniform(-3.0 / J, 3.0 / J, J)
    if dgp == 3:
        return (rng.beta(0.2, 0.2, J) - 0.5) * 3.0 / J
    # dgp == 4: half dense, half zero, shuffled
    w = np.zeros(J)
    half = J // 2
    w[:half] = (rng.beta(0.2, 0.2, half) - 0.5) * 3.0 / J
    rng.shuffle(w)
    return w


def _panel(rng, w0):
    T = T0 + T1
    load = np.arange(1, J + 1) / J
    F = rng.normal(size=(T, 2))
    Y0 = load[None, :] + F[:, 0:1] + load[None, :] * F[:, 1:2] + rng.normal(0, 2.0, (T, J))
    Y1 = Y0 @ w0 + rng.normal(0, 1.0, T)
    Y1[T0:] += DELTA
    return Y0, Y1


def _att(model_kwargs, Y0, Y1):
    from mlsynth.utils import effectutils as eff
    from mlsynth.utils.laxscm_helpers.crossval import ElasticNetCV
    m = ElasticNetCV(**model_kwargs)
    m.fit(Y0[:T0], Y1[:T0])
    y_post_hat = m.predict(Y0[T0:])
    # Route the ATT through the shared effects helper rather than re-deriving it.
    return eff.att(eff.gap(Y1[T0:], y_post_hat))


_METHODS = {
    "sc": dict(alpha=[0.0], lam=[0.0], second_norm="L1_L2",
               constraint_type="simplex", fit_intercept=False, n_splits=2),
    "linf": dict(alpha=[0.0], lam=[LAM], second_norm="L1_INF",
                 constraint_type="unconstrained", fit_intercept=True, n_splits=2),
    "l1linf": dict(alpha=[0.5], lam=[LAM], second_norm="L1_INF",
                   constraint_type="unconstrained", fit_intercept=True, n_splits=2),
}


def run() -> dict:
    rng = np.random.default_rng(SEED)
    rmse = {dgp: {m: [] for m in _METHODS} for dgp in (1, 2, 3, 4)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dgp in (1, 2, 3, 4):
            for _ in range(B):
                w0 = _weights(rng, dgp)
                Y0, Y1 = _panel(rng, w0)
                for name, kw in _METHODS.items():
                    rmse[dgp][name].append((_att(kw, Y0, Y1) - DELTA) ** 2)

    def R(dgp, m):
        return float(np.sqrt(np.mean(rmse[dgp][m])))

    out = {f"rmse_dgp{d}_{m}": R(d, m) for d in (1, 2, 3, 4) for m in _METHODS}
    # Directional claims from the paper.
    out["linf_beats_sc_dgp2"] = float(R(2, "linf") < R(2, "sc"))
    out["linf_beats_sc_dgp3"] = float(R(3, "linf") < R(3, "sc"))
    out["l1linf_beats_sc_dgp4"] = float(R(4, "l1linf") < R(4, "sc"))
    return out


# Directional reproduction of Table 4 (dense regimes favour L-infinity; the
# sparse regime favours L1+L-infinity). RMSE cells are reported for the record
# but only loosely bounded; the booleans are the assertion.
EXPECTED = {
    "linf_beats_sc_dgp2": (1.0, 0.0),
    "linf_beats_sc_dgp3": (1.0, 0.0),
    "l1linf_beats_sc_dgp4": (1.0, 0.0),
    "rmse_dgp2_sc": (1.0, 1.5),
    "rmse_dgp2_linf": (0.5, 1.0),
    "rmse_dgp3_sc": (1.0, 1.5),
    "rmse_dgp3_linf": (0.5, 1.0),
}
