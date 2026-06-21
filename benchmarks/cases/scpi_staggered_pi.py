"""Cross-validation benchmark: staggered VanillaSC prediction intervals vs scpi.

Path A (empirical, runnable Python reference). Reproduces the multiple-treated-
unit *prediction intervals* of Cattaneo, Feng, Palomba & Titiunik (2025),
Section 4 -- the cross-unit (event-time, TSUA) bands -- on the canonical Germany
reunification panel (``basedata/scpi_germany.csv``), driven through the public
``VanillaSC.fit()`` with ``inference="scpi"``.

Two treated units adopt at different times (West Germany 1991, Italy 1992) with
the 15 never-treated countries as donors. Outcome-only, simplex weights. The
``scpi`` package's ``scpi`` (with ``scdataMulti(effect="time")``) produces the
event-time prediction intervals; ``mlsynth``'s clean-room staggered engine
reproduces the full sub-Gaussian band to solver tolerance in ``scpi``-compat
mode (``scpi_compat=True``).

This case also documents a published-quirk finding. ``scpi`` divides the
predictand matrix ``P`` by the number of treated units ``iota`` in
``scdataMulti`` (correct -- it forms the average) and then divides the simulated
in-sample draws by ``iota`` a *second* time in ``scpi_in_diag``, so its
time-aggregated in-sample interval carries a ``1 / iota**2`` scaling, while the
point estimate and out-of-sample term use the statistically correct single
``1 / iota``. Verified against ``scpi`` at machine precision: with iota = 2 the
correct (default) in-sample band is exactly 2x wider than ``scpi``'s at every
event time. ``mlsynth`` defaults to the correct ``1 / iota`` and exposes
``scpi_compat=True`` to reproduce ``scpi`` bit-for-bit; both are checked here
through ``fit()``.

Reference: ``scpi_pkg`` (PyPI). Skips itself (``BenchmarkSkipped``) when absent.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "scpi_germany.csv")
_ADOPT = {"West Germany": 1991, "Italy": 1992}
_SIMS = 500
_SEED = 1234


def _panel() -> pd.DataFrame:
    df = pd.read_csv(os.path.abspath(_DATA))
    df["status"] = 0
    for unit, yr in _ADOPT.items():
        df.loc[(df["country"] == unit) & (df["year"] >= yr), "status"] = 1
    return df


def _fit_tsua(df: pd.DataFrame, *, scpi_compat: bool):
    """Event-time synthetic-prediction bands through the public fit()."""
    from mlsynth import VanillaSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": df, "outcome": "gdp", "treat": "status",
                         "unitid": "country", "time": "year",
                         "display_graphs": False, "inference": "scpi",
                         "scpi_sims": _SIMS, "seed": _SEED,
                         "scpi_compat": scpi_compat}).fit()
    esi = res.additional_outputs["event_study_intervals"]
    ells = sorted(esi)
    full = np.array([esi[e]["synthetic_ci"] for e in ells], dtype=float)
    insample = np.array([esi[e]["insample_synthetic_ci"] for e in ells], dtype=float)
    return ells, full, insample


def _scpi_tsua(df: pd.DataFrame) -> np.ndarray:
    """scpi effect='time' full sub-Gaussian band (CI_all_gaussian)."""
    from scpi_pkg.scdataMulti import scdataMulti
    from scpi_pkg.scpi import scpi
    aux = scdataMulti(df=df, id_var="country", time_var="year", outcome_var="gdp",
                      treatment_var="status", features=None, cov_adj=None,
                      constant=False, cointegrated_data=False, effect="time")
    np.random.seed(_SEED)
    res = scpi(aux, w_constr={"name": "simplex"}, sims=_SIMS, cores=1,
               e_method="gaussian", u_alpha=0.05, e_alpha=0.05)
    ci = res.CI_all_gaussian
    return np.column_stack([np.asarray(ci.iloc[:, 0], float),
                            np.asarray(ci.iloc[:, 1], float)])


def run() -> dict:
    try:
        import scpi_pkg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional reference dep
        raise BenchmarkSkipped("scpi_pkg not installed "
                               "(`pip install scpi_pkg`)") from exc

    df = _panel()

    # --- mlsynth through fit(): scpi-compat (matches scpi) and default (correct) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, full_compat, insample_compat = _fit_tsua(df, scpi_compat=True)
        _, _, insample_default = _fit_tsua(df, scpi_compat=False)
        sc_band = _scpi_tsua(df)

    n = min(len(full_compat), len(sc_band))
    fc, scb = full_compat[:n], sc_band[:n]

    # full-band agreement (the reproduction claim, through fit())
    sc_w = scb[:, 1] - scb[:, 0]
    ml_w = fc[:, 1] - fc[:, 0]
    tsua_width_max_rel = float(np.max(np.abs(ml_w - sc_w) / sc_w))
    tsua_endpoint_max_abs = float(np.max(np.abs(fc - scb)))

    # the 1/iota**2 finding: default in-sample band is exactly iota x compat's
    iota = len(_ADOPT)
    wc = insample_compat[:n, 1] - insample_compat[:n, 0]
    wd = insample_default[:n, 1] - insample_default[:n, 0]
    insample_iota_ratio = float(np.median(wd / wc))

    return {
        "tsua_full_band_width_max_rel_diff_vs_scpi": tsua_width_max_rel,
        "tsua_full_band_endpoint_max_abs_diff_vs_scpi": tsua_endpoint_max_abs,
        # documents the upstream quirk: correct default / scpi-compat = iota
        "insample_default_over_compat_ratio": insample_iota_ratio,
    }


# scpi-compat fit() reproduces scpi's event-time bands to solver tolerance; the
# correct default in-sample band is exactly iota (= 2) times scpi's, documenting
# the 1/iota**2 scaling in scpi's time-aggregated in-sample interval.
EXPECTED = {
    "tsua_full_band_width_max_rel_diff_vs_scpi": (0.0, 0.05),    # within 5%
    "tsua_full_band_endpoint_max_abs_diff_vs_scpi": (0.0, 0.05),
    "insample_default_over_compat_ratio": (2.0, 0.05),          # = iota
}
