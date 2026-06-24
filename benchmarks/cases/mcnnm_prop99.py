"""Cross-validation benchmark: MC-NNM vs ``causaltensor`` on Proposition 99.

Path A (empirical, scenario 3 -- runnable Python reference). Reproduces the
Matrix-Completion (MC-NNM) estimate of California's Proposition 99 effect on the
canonical smoking panel and cross-validates mlsynth against ``causaltensor``.

Why this is an estimand-level (not cell-exact) cross-check
---------------------------------------------------------
Both implementations solve the same SOFT-IMPUTE objective (Athey et al. 2021,
eq. 4.3) but differ in the two-way fixed-effect sub-solver: mlsynth fits the
unit/time effects on the **observed cells only** (the Athey et al. observed-set
:math:`\\mathcal{O}` convention), whereas ``causaltensor`` fits them on the full
``O - M`` matrix. With each implementation choosing its own regulariser by
cross-validation, the two completed matrices are therefore close but not
identical -- so we cross-validate the **ATT** (the public estimand), not the raw
low-rank fit.

Provenance
----------
* Data: ``mlsynth/basedata/smoking_data.csv`` -- the canonical Abadie et al.
  (2010) sample: 39 states x 31 years (1970-2000), California treated 1989.
* Headline: Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), "Matrix
  Completion Methods for Causal Panel Data Models," *JASA* 116(536), report an
  MC-NNM Prop 99 effect of roughly **-20** packs per capita.
* Reference: ``causaltensor.MC_NNM_with_cross_validation`` (PyPI
  ``causaltensor`` >= 0.1.12). Installed via ``pip install causaltensor``.

mlsynth's CV ATT (-19.83) and causaltensor's (-20.27) agree to ~0.44 packs
(~2% of the estimand) -- the residual is the documented FE-solver difference.
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

    from mlsynth import MCNNM

    df = _load_panel()

    # --- mlsynth (selects lambda by its own cross-validation) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MCNNM({"df": df, "outcome": "cigsale", "treat": "treat",
                     "unitid": "state", "time": "year",
                     "display_graphs": False}).fit()
    ml_att = float(res.att)

    # --- causaltensor reference: O outcome matrix, Omega observed mask ---
    wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    states, years = wide.index.tolist(), wide.columns.tolist()
    O = wide.values.astype(float)
    ti, sc = states.index(TREAT_UNIT), years.index(TREAT_YEAR)
    Omega = np.ones_like(O)
    Omega[ti, sc:] = 0  # 1 = observed, 0 = treated/missing
    _, _, _, ct_tau = ct.MC_NNM_with_cross_validation(O, Omega)
    ct_att = float(ct_tau)

    return {
        "mcnnm_att": ml_att,
        "mcnnm_vs_causaltensor_abs_diff": abs(ml_att - ct_att),
    }


def comparison() -> dict:
    """mlsynth MC-NNM vs ``causaltensor``, the Prop 99 ATT side by side.

    Re-runs both solvers (each picking its own regulariser by cross-validation)
    and pairs the ATT estimands. Propagates ``BenchmarkSkipped`` when
    ``causaltensor`` is not installed, so the comparison export mirrors ``run``.
    """
    try:
        import causaltensor as ct
    except ImportError as exc:  # pragma: no cover - optional reference dep
        raise BenchmarkSkipped("causaltensor not installed "
                               "(`pip install causaltensor`)") from exc

    from mlsynth import MCNNM

    df = _load_panel()
    cfg = {"outcome": "cigsale", "treat": "treat", "unitid": "state",
           "time": "year"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MCNNM({**cfg, "df": df, "display_graphs": False}).fit()
    ml_att = float(res.att)

    wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    states, years = wide.index.tolist(), wide.columns.tolist()
    O = wide.values.astype(float)
    ti, sc = states.index(TREAT_UNIT), years.index(TREAT_YEAR)
    Omega = np.ones_like(O)
    Omega[ti, sc:] = 0  # 1 = observed, 0 = treated/missing
    _, _, _, ct_tau = ct.MC_NNM_with_cross_validation(O, Omega)
    ct_att = float(ct_tau)

    rows = [{"quantity": "ATT", "mlsynth": round(ml_att, 6),
             "reference": round(ct_att, 6)}]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MCNNM", "config": cfg},
        "reference": {"impl": "Python package causaltensor",
                      "version": getattr(ct, "__version__", "causaltensor (pip)")},
    }


# mlsynth's ATT must land on the published MC-NNM value (~-20, generous 1.5 to
# absorb the implementation's CV choice) and agree with causaltensor to within
# 1.0 pack -- the tolerance brackets the documented FE-solver difference plus the
# two independent cross-validation grids.
EXPECTED = {
    "mcnnm_att": (-20.0, 1.5),
    "mcnnm_vs_causaltensor_abs_diff": (0.0, 1.0),
}
