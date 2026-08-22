"""The panel the calibration study was measured on, and the DGP built from it.

The study's real-data arms need a wide panel of positive outcomes -- the one used
was 211 US media markets by 128 weeks of retail sales. That panel is proprietary
and is not shipped, so this module reads whatever panel the environment points it
at and rebuilds the same structure from it:

  ``MLSYNTH_CAL_PANEL``   path to a CSV (required for the real-data arms)
  ``MLSYNTH_CAL_TIME``    column holding the period      (default ``start_date``)
  ``MLSYNTH_CAL_UNIT``    column holding the unit         (default ``dma``)
  ``MLSYNTH_CAL_VALUE``   column holding the outcome      (default ``total``)

Draws are synthetic in composition and real in every component. From an SVD of
the unit-demeaned log panel the module keeps the factor paths, the loadings, the
unit means and the idiosyncratic residuals; a draw selects ``J`` units and applies
one circular shift to the time index. The shift is common to all units on purpose,
since shifting each unit independently would preserve every unit's own temporal
dependence while destroying the cross-sectional dependence among residuals, which
is the part a synthetic control has to survive.

Nothing here is fitted or tuned: the factor path carries the actual trend and
seasonality, and each unit's residual series carries its own autocorrelation,
variance and departures from normality.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

T0 = 104                       # pre-periods
H = 13                         # horizon
TB = int(round(0.45 * T0))     # 47 blank periods, the latter 45 percent
TE = T0 - TB                   # 57 fitting periods
T = T0 + H
ALPHA = 0.05
LIFT = 0.02
DELTA = float(np.log1p(LIFT))  # additive shift in logs, so tau_t is known exactly
K = 2                          # factors kept


class PanelUnavailable(RuntimeError):
    """Raised when the study panel is not configured; arms skip on it."""


def _path() -> str:
    p = os.environ.get("MLSYNTH_CAL_PANEL")
    if not p:
        raise PanelUnavailable(
            "set MLSYNTH_CAL_PANEL to a CSV with unit / period / outcome columns; "
            "see the module docstring for the column-name variables.")
    if not os.path.exists(p):
        raise PanelUnavailable(f"MLSYNTH_CAL_PANEL points at a missing file: {p}")
    return p


def load():
    """Factor paths, loadings, unit means and residuals of the configured panel."""
    d = pd.read_csv(_path())
    tcol = os.environ.get("MLSYNTH_CAL_TIME", "start_date")
    ucol = os.environ.get("MLSYNTH_CAL_UNIT", "dma")
    vcol = os.environ.get("MLSYNTH_CAL_VALUE", "total")
    wide = d.pivot_table(index=tcol, columns=ucol, values=vcol).sort_index()
    values = np.log(wide.values)
    mean = values.mean(axis=0)
    centred = values - mean[None, :]
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    factors = U[:, :K] * S[:K]
    loadings = Vt[:K].T
    residuals = centred - factors @ loadings.T
    return mean, factors, loadings, residuals


_CACHE: dict = {}


def _components():
    if "v" not in _CACHE:
        _CACHE["v"] = load()
    return _CACHE["v"]


def draw(seed: int, J: int = 20):
    """A ``J``-unit panel of ``T`` log outcomes, untreated."""
    mean, factors, loadings, residuals = _components()
    n_all, t_full = loadings.shape[0], factors.shape[0]
    if t_full < T:
        raise PanelUnavailable(
            f"the panel has {t_full} periods; the study design needs {T}.")
    rng = np.random.default_rng(seed)
    units = rng.choice(n_all, size=J, replace=False)
    shift = int(rng.integers(0, t_full))
    idx = (np.arange(T) + shift) % t_full
    return (mean[units][None, :]
            + factors[idx] @ loadings[units].T
            + residuals[idx][:, units]).T


def frame(Y: np.ndarray) -> pd.DataFrame:
    """Long frame in the shape the design estimators ingest."""
    J = Y.shape[0]
    return pd.DataFrame([{"unit": f"d{j:03d}", "time": t, "y": float(Y[j, t])}
                         for j in range(J) for t in range(Y.shape[1])])
