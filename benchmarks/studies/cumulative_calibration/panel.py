"""The panel the calibration study was measured on, and the DGP built from it.

The study's arms need a wide panel of positive outcomes. The one they were
measured on is proprietary and is not shipped, so by default this module builds a
synthetic stand-in -- see :mod:`synthetic_geo_panel` -- drawn to behave the same
way: one dominant factor, mildly persistent idiosyncratic errors, market sizes
spread over orders of magnitude, and a small correlation shared inside a region.
Every arm therefore runs with nothing configured.

Point ``MLSYNTH_CAL_PANEL`` at a CSV to use a real panel instead. A configured
path is never replaced by the stand-in: if it cannot be read, the error stands,
so a results file cannot report real-panel numbers that a substitution produced.
Call :func:`source` to record which panel an arm actually used.

  ``MLSYNTH_CAL_PANEL``   path to a CSV (optional; the stand-in is the default)
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


def source() -> str:
    """``"synthetic"``, or the path of the panel the environment points at."""
    return os.environ.get("MLSYNTH_CAL_PANEL") or "synthetic"


def _path() -> str:
    p = os.environ.get("MLSYNTH_CAL_PANEL")
    if not p:                                   # pragma: no cover - source() gates it
        raise PanelUnavailable(
            "set MLSYNTH_CAL_PANEL to a CSV with unit / period / outcome columns; "
            "see the module docstring for the column-name variables.")
    if not os.path.exists(p):
        raise PanelUnavailable(f"MLSYNTH_CAL_PANEL points at a missing file: {p}")
    return p


def _decompose(values: np.ndarray):
    """Split a log panel into unit means, ``K`` factors, loadings and residuals."""
    mean = values.mean(axis=0)
    centred = values - mean[None, :]
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    factors = U[:, :K] * S[:K]
    loadings = Vt[:K].T
    return mean, factors, loadings, centred - factors @ loadings.T


def _synthetic():
    """The stand-in panel, decomposed by the same code path as a real one.

    A panel drawn to behave like the one the study was measured on: dominated by
    a single factor, mildly persistent once that factor is out, with a small
    correlation shared inside a regional block. Running the same decomposition
    over it means the arms cannot tell the two apart by their interface.
    """
    try:                                     # imported as a package member
        from . import synthetic_geo_panel as sgp
    except ImportError:                      # imported as a script module, as the
        import synthetic_geo_panel as sgp    # arms do via sys.path
    n_weeks = max(sgp.N_WEEKS, T)
    values, _ = sgp.make_panel(np.random.default_rng(20240101),
                               n_markets=sgp.N_MARKETS, n_weeks=n_weeks)
    return _decompose(values)


def load():
    """Factor paths, loadings, unit means and residuals of the configured panel.

    Falls back to :func:`_synthetic` when nothing is configured, so every arm runs
    without the panel the study was measured on. A configured path is never
    replaced: if it cannot be read, the error stands.
    """
    if source() == "synthetic":
        return _synthetic()
    d = pd.read_csv(_path())
    tcol = os.environ.get("MLSYNTH_CAL_TIME", "start_date")
    ucol = os.environ.get("MLSYNTH_CAL_UNIT", "dma")
    vcol = os.environ.get("MLSYNTH_CAL_VALUE", "total")
    wide = d.pivot_table(index=tcol, columns=ucol, values=vcol).sort_index()
    return _decompose(np.log(wide.values))


_CACHE: dict = {}


def _components():
    """Cached components, keyed by source so a reconfigured panel is re-read."""
    key = source()
    if _CACHE.get("key") != key:
        _CACHE.clear()
        _CACHE["key"], _CACHE["v"] = key, load()
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
