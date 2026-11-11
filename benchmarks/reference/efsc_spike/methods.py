"""Estimators compared in the EFSC spike.

* ``efsc_fits``      -- the package's two fits (counterfactual mask, treated mask),
                       via ``efsc.fitting.fit_factorization``: the same calls
                       ``EFSC.fit`` makes in causal mode (estimator.py).
* ``cell_mle``       -- cell-wise maximum likelihood natural parameters.
* ``sc_complete``    -- the paper's synthetic-control baseline: simplex weights
                       on pre-period rows, one treated unit at a time (C.5).
* ``mcnnm_complete`` -- ``mlsynth.MCNNM`` on a matrix of cell estimates, through
                       its long-panel ingestion. The competent low-rank baseline
                       the paper does not run.
* ``dsc_counterfactual`` -- ``mlsynth.DSC`` on the microdata, one treated unit at
                       a time; counterfactual quantile functions.
* ``compsc_counterfactual`` -- ``mlsynth.COMPSC`` on cell shares, one treated
                       unit at a time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import efsc.fitting
from efsc import InferenceConfig

PAPER = InferenceConfig(epochs=3000, learning_rate=0.1, prior_scale=1.0)  # C.5
PACKAGE_DEFAULT = InferenceConfig()  # prior 2.0, 2000 epochs, family lr


def efsc_fits(panel, truth, *, rank, config, seed=0):
    tp = truth["treated_post"]
    cf = efsc.fitting.fit_factorization(panel, ~tp, rank=rank, config=config, seed=seed)
    tr = efsc.fitting.fit_factorization(panel, tp, rank=rank, config=config, seed=seed + 1)
    return cf.natural_parameters.numpy().astype(float), tr.natural_parameters.numpy().astype(float)


# ----------------------------------------------------------------------------- cells
def cell_stats(truth):
    f = truth["frame"]
    n, T = truth["m"].shape
    g = f.groupby(["unit", "time"])["y"]
    mean = g.mean().to_numpy().reshape(n, T)
    var = g.var(ddof=0).to_numpy().reshape(n, T)
    return mean, var


def cell_mle(truth, family):
    n, T = truth["m"].shape
    if family == "gaussian":
        mean, var = cell_stats(truth)
        var = np.maximum(var, 1e-6)
        return np.stack([mean / var, -1.0 / (2.0 * var)], axis=-1)
    if family == "poisson":
        mean, _ = cell_stats(truth)
        return np.log(np.maximum(mean, 0.5 / truth["m"]))[..., None]
    if family == "multinomial":
        counts = truth["counts"] + 0.5  # Haldane correction: zero cells have no MLE
        return np.log(counts[..., :-1] / counts[..., -1:])
    raise ValueError(family)


# ------------------------------------------------------------------- completions
def _simplex_weights(target, donors):
    k = donors.shape[0]
    obj = lambda w: np.sum((target - w @ donors) ** 2)
    grad = lambda w: -2.0 * donors @ (target - w @ donors)
    res = minimize(obj, np.full(k, 1.0 / k), jac=grad, method="SLSQP",
                   bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def sc_complete(matrix, truth):
    """Simplex SC per treated unit on the pre-period columns of ``matrix``."""
    tp = truth["treated_post"]
    T = matrix.shape[1]
    post = truth["post"]
    controls = np.where(~tp.any(axis=1))[0]
    out = matrix.copy()
    for i in truth["treated_idx"]:
        w = _simplex_weights(matrix[i, : T - post], matrix[controls, : T - post])
        out[i, T - post:] = w @ matrix[controls, T - post:]
    return out


def mcnnm_complete(matrix, truth, seed=0):
    """``mlsynth.MCNNM`` on a cell-level matrix; returns the completed matrix."""
    from mlsynth import MCNNM

    n, T = matrix.shape
    tp = truth["treated_post"]
    df = pd.DataFrame({
        "unit": np.repeat(np.arange(n), T),
        "time": np.tile(np.arange(T), n),
        "y": matrix.ravel(),
        "d": tp.ravel().astype(int),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MCNNM({"df": df, "outcome": "y", "treat": "d", "unitid": "unit",
                     "time": "time", "display_graphs": False, "random_state": seed}).fit()
    cf = np.asarray(res.counterfactual_matrix, dtype=float)
    out = matrix.copy()
    out[tp] = cf[tp]
    return out


def gaussian_component_complete(eta, truth, completer):
    """Complete the two Gaussian components on their unconstrained scales."""
    z1 = completer(eta[..., 0], truth)
    z2 = completer(np.log(-eta[..., 1]), truth)
    return np.stack([z1, -np.exp(z2)], axis=-1)


def dsc_counterfactual(truth, seed=0):
    """``mlsynth.DSC`` per treated unit -> counterfactual Gaussian natural params.

    DSC returns counterfactual quantile functions; the mean and variance of the
    quantile curve give the Gaussian the comparison needs.
    """
    from mlsynth import DSC

    f = truth["frame"]
    n, T = truth["m"].shape
    tp = truth["treated_post"]
    post = truth["post"]
    controls = np.where(~tp.any(axis=1))[0]
    grid = np.linspace(0.005, 0.995, 199)
    out = np.full((n, T, 2), np.nan)
    for i in truth["treated_idx"]:
        sub = f[f["unit"].isin(np.append(controls, i))].copy()
        sub["d"] = ((sub["unit"] == i) & (sub["time"] >= T - post)).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DSC({"df": sub, "outcome": "y", "treat": "d", "unitid": "unit",
                       "time": "time", "display_graphs": False, "random_state": seed,
                       "qte_quantiles": list(grid), "compute_inference": False}).fit()
        for curve in res.qte_curves:
            j = int(curve.time_label)
            q = np.asarray(curve.counterfactual, dtype=float)
            mu, var = q.mean(), max(q.var(), 1e-6)
            out[i, j] = (mu / var, -1.0 / (2.0 * var))
    return out


def compsc_counterfactual(truth, zero_replacement=None):
    """``mlsynth.COMPSC`` per treated unit -> counterfactual probabilities."""
    from mlsynth import COMPSC

    n, T = truth["m"].shape
    C = truth["counts"].shape[-1]
    tp = truth["treated_post"]
    post = truth["post"]
    controls = np.where(~tp.any(axis=1))[0]
    shares = truth["counts"] / truth["counts"].sum(axis=-1, keepdims=True)
    cols = [f"s{c}" for c in range(C)]
    out = np.full((n, T, C), np.nan)
    for i in truth["treated_idx"]:
        units = np.append(controls, i)
        rows = []
        for u in units:
            for j in range(T):
                rows.append([u, j, int(u == i and j >= T - post), *shares[u, j]])
        df = pd.DataFrame(rows, columns=["unit", "time", "d", *cols])
        cfg = {"df": df, "outcome": cols[0], "outcomes": cols, "treat": "d",
               "unitid": "unit", "time": "time", "display_graphs": False,
               "inference": "none"}
        if zero_replacement is not None:
            cfg["zero_replacement"] = zero_replacement
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = COMPSC(cfg).fit()
        cf = np.asarray(res.counterfactual_composition, dtype=float)
        out[i, T - post:] = cf[-post:]
    return out


# --------------------------------------------------------------------- divergences
def gaussian_kl(eta_p, eta_q):
    """KL(N_p || N_q) from natural parameters, cell-wise."""
    s2p, s2q = -1 / (2 * eta_p[..., 1]), -1 / (2 * eta_q[..., 1])
    mp, mq = eta_p[..., 0] * s2p, eta_q[..., 0] * s2q
    return 0.5 * np.log(s2q / s2p) + (s2p + (mp - mq) ** 2) / (2 * s2q) - 0.5


def softmax_ref(logits):
    full = np.concatenate([logits, np.zeros(logits.shape[:-1] + (1,))], axis=-1)
    full = np.exp(full - full.max(axis=-1, keepdims=True))
    return full / full.sum(axis=-1, keepdims=True)


def poisson_kl(eta_p, eta_q):
    lp, lq = np.exp(eta_p[..., 0]), np.exp(eta_q[..., 0])
    return lp * (eta_p[..., 0] - eta_q[..., 0]) - lp + lq
