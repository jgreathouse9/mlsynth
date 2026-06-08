"""Cross-validation benchmark: SDID vs ``causaltensor`` on Proposition 99.

Path A (empirical, scenario 3 -- runnable Python reference). Reproduces the
canonical Synthetic DiD estimate of California's Proposition 99 tobacco-control
effect on the Abadie-Diamond-Hainmueller smoking panel, and cross-validates the
mlsynth implementation cell-for-cell against the ``causaltensor`` reference
implementation.

Provenance
----------
* Data: ``mlsynth/basedata/smoking_data.csv`` -- the canonical Abadie et al.
  (2010) sample: **39 states x 31 years (1970-2000)**, California treated from
  1989. The ``Proposition 99`` column flags the treated unit/period cells.
* Headline: Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), "Synthetic
  Difference-in-Differences," *AER* 111(12), report an SDID ATT of about
  **-15.6** packs per capita (matched by the R ``synthdid`` package, -15.604).
* Reference: ``causaltensor.SDID`` (PyPI ``causaltensor`` >= 0.1.12), the
  ``SDIDPanelSolver`` implementation. Installed via ``pip install causaltensor``.

The two implementations agree on the ATT to ~3e-3 packs; the residual is the
unit-weight ridge (zeta) optimiser, not a methodological difference.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_BASE = Path(__file__).resolve().parents[2] / "basedata"
TREAT_UNIT = "California"
TREAT_YEAR = 1989


def _load_panel() -> pd.DataFrame:
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["treat"] = df["Proposition 99"].astype(int)
    return df[["state", "year", "cigsale", "treat"]]


def run() -> dict:
    try:
        import causaltensor as ct  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional reference dep
        raise BenchmarkSkipped("causaltensor not installed "
                               "(`pip install causaltensor`)") from exc

    from mlsynth import SDID

    df = _load_panel()

    # --- mlsynth ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SDID({"df": df, "outcome": "cigsale", "treat": "treat",
                    "unitid": "state", "time": "year",
                    "display_graphs": False}).fit()
    ml_att = float(res.inference.att)

    # --- causaltensor reference: O (N x T) outcome matrix + Z treatment mask ---
    wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    states, years = wide.index.tolist(), wide.columns.tolist()
    O = wide.values.astype(float)
    ti, sc = states.index(TREAT_UNIT), years.index(TREAT_YEAR)
    Z = np.zeros_like(O)
    Z[ti, sc:] = 1
    ct_att = float(ct.SDID(O, Z, treat_units=[ti], starting_time=sc))

    return {
        "sdid_att": ml_att,
        "sdid_vs_causaltensor_abs_diff": abs(ml_att - ct_att),
    }


# mlsynth's ATT must land on the published SDID value, and match the reference
# implementation tightly. Tolerances: 0.05 brackets display rounding of the
# AER/synthdid -15.6 headline; 5e-3 is the optimiser-level agreement with
# causaltensor (different ridge solver, same estimand).
EXPECTED = {
    "sdid_att": (-15.604, 0.05),
    "sdid_vs_causaltensor_abs_diff": (0.0, 5e-3),
}
