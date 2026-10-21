"""A readable port of Kato & Ohda (2026) MMSCM, used as the replication oracle.

Reference: mmscm.py from https://github.com/MasaKat0/mmscm --
  `MMSCM._data_setup`      (column scaling by per-feature max |.|)
  `MMSCM.train_param`      (moment construction, SLSQP on the simplex)
  `distscm_obj_func`       (the loss)
  `MMSCM.bias_train`       (the pre-period level correction)

The method in one line: choose simplex weights w so that, for each moment
k = 1..K, the TIME AVERAGE of the residual is zero.

    loss(w) = sum_k v_k * ( mean_t [ y_k,t - sum_j w_j * y_k,j,t ] )^2

Abadie's objective is the same expression with the square inside the average,
`mean_t[ (.)^2 ]`. That one transposition is the whole method: Abadie matches
the pre-period path, MMSCM matches K scalar averages of it.
"""
from __future__ import annotations
import numpy as np
import scipy.optimize


def build_moments(features: np.ndarray, poly: int) -> np.ndarray:
    """(K0, T) features -> (K0 + poly - 2, T) moments.

    Mirrors `train_param`'s `for dim in range(2, self.poly)` loop exactly: the
    raw features are kept, and powers 2..poly-1 of the FIRST one are appended.
    So `poly=2` appends nothing, and with covariates present the polynomial
    moments are built from whichever feature comes first -- which is why the
    caller must put the outcome first.
    """
    F = np.atleast_2d(np.asarray(features, dtype=float))
    extra = [F[0] ** d for d in range(2, poly)]
    return np.vstack([F] + extra) if extra else F


def mmscm_loss(w, treated_mom, donor_mom, v):
    """sum_k v_k * (time-average residual of moment k)^2."""
    resid = treated_mom - np.einsum("j,kjt->kt", w, donor_mom)
    return float(np.sum(v * resid.mean(axis=1) ** 2))


def abadie_loss(w, treated_mom, donor_mom, v):
    """The contrast: sum_k v_k * time-average of the SQUARED residual."""
    resid = treated_mom - np.einsum("j,kjt->kt", w, donor_mom)
    return float(np.sum(v * (resid ** 2).mean(axis=1)))


def fit(treated_pre, donors_pre, poly=2, loss="mmscm", w0=None, scale=True):
    """Simplex-constrained fit on the pre-period. Returns (w, loss, success).

    `treated_pre` is (K0, T) and `donors_pre` is (J, K0, T); a 1-D outcome and a
    (J, T) donor block are accepted and treated as K0 = 1.

    `scale` reproduces `_data_setup`: every feature is divided by the largest
    absolute value it takes across all units, so the moments are comparable
    before they are summed.
    """
    treated_pre = np.atleast_2d(np.asarray(treated_pre, float))   # (K0, T)
    donors_pre = np.asarray(donors_pre, float)                     # (J, K0, T)
    if donors_pre.ndim == 2:                                       # (J, T) -> (J, 1, T)
        donors_pre = donors_pre[:, None, :]
    J = donors_pre.shape[0]
    tm = build_moments(treated_pre, poly)
    dm = np.stack([build_moments(donors_pre[j], poly) for j in range(J)], axis=1)
    n_moments = tm.shape[0]
    if scale:
        m = np.abs(np.concatenate([tm[:, None, :], dm], axis=1)).max(axis=(1, 2))
        tm, dm = tm / m[:, None], dm / m[:, None, None]
    v = np.ones(n_moments) / n_moments
    obj = mmscm_loss if loss == "mmscm" else abadie_loss
    w0 = np.ones(J) / J if w0 is None else np.asarray(w0, float)
    res = scipy.optimize.minimize(
        lambda w: obj(w, tm, dm, v), w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints=({"type": "eq", "fun": lambda x: 1 - np.sum(x)},))
    return res.x, float(res.fun), bool(res.success)


def counterfactual(w, donors_all):
    """Weighted donor OUTCOMES over every period (no bias correction).

    Takes the outcome row only: the counterfactual is in outcome units however
    many features the weights were fitted on.
    """
    D = np.asarray(donors_all, float)
    return w @ (D if D.ndim == 2 else D[:, 0, :])


def bias_correction(w, treated_pre, donors_pre):
    """`bias_train`: the mean pre-period OUTCOME gap, added to the counterfactual."""
    t = np.atleast_2d(np.asarray(treated_pre, float))[0]
    D = np.asarray(donors_pre, float)
    return float(np.mean(t - w @ (D if D.ndim == 2 else D[:, 0, :])))
