#!/usr/bin/env python3
"""Live captured run of the original Robust Synthetic Control (tslib) on Prop 99.

Runs Amjad, Shah & Shin's canonical ``RobustSyntheticControl``
(https://github.com/jehangiramjad/tslib, *Robust Synthetic Control*,
JMLR 19(22):1-51, 2018) on the Abadie-Diamond-Hainmueller California
Proposition 99 panel: California treated, the 38-state donor pool, the
1970-1988 pre-period (T0=19) and the 1989-2000 post-period. The OG RSC is hard
singular-value thresholding (HSVT) of the stacked donor+treated pre-matrix at a
fixed rank ``k``, followed by an unconstrained pseudo-inverse (OLS) of the
de-noised treated row on the de-noised donor rows.

``k`` is fixed to ``3`` -- the ``singvals = 3`` that tslib's own bundled
California Prop 99 study (``tests/testScriptSynthControlSVD.py::prop99``) uses --
and the SAME ``k`` is fed to both tslib here and mlsynth's PCR in the case, so
the de-noising rank matches.

The reference is fetched at a pinned commit into the gitignored
``benchmarks/reference/.cache`` (git clone, else codeload tarball); see
``benchmarks/reference/clone_tslib.py`` and the bundle ``NOTICE``. Nothing from
tslib is redistributed in this tree.

Prints the ``== REFERENCE VALUES ==`` block that ``generate.py`` parses (the
learned donor weights as ``weight\\t<donor>\\t<value>`` rows, the pre-period
RMSE, the post-period mean counterfactual, and the ATT) followed by a
``== SESSION INFO ==`` block recording numpy / pandas / scikit-learn / Python
versions.

Run from the repository root::

    python benchmarks/reference/pcr_rsc_ref/reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reference.clone_tslib import build_rsc_weights

_DATA = ROOT / "basedata" / "smoking_data.csv"
_TREAT_YEAR = 1989
_K = 3  # tslib's own prop99 study uses singvals = 3
_TREATED = "California"


def _run() -> dict:
    df = pd.read_csv(_DATA)
    wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    years = list(wide.columns)
    states = list(wide.index)
    donors = [s for s in states if s != _TREATED]
    pre_years = [y for y in years if y < _TREAT_YEAR]
    post_years = [y for y in years if y >= _TREAT_YEAR]
    T0 = len(pre_years)

    target_pre = wide.loc[_TREATED, pre_years].values.astype(float)
    donor_pre = wide.loc[donors, pre_years].values.T          # (T0, J)
    weights = build_rsc_weights(target_pre, donor_pre, donors, _K)

    # Counterfactual over the full period: raw donor outcomes @ learned weights
    # (RSC predicts with the donor pool's observed series, as in tslib.predict).
    donor_full = wide.loc[donors, years].values.T             # (T, J)
    y_treated = wide.loc[_TREATED, years].values.astype(float)
    cf = donor_full @ weights

    pre_idx = [years.index(y) for y in pre_years]
    post_idx = [years.index(y) for y in post_years]
    pre_rmse = float(np.sqrt(np.mean((y_treated[pre_idx] - cf[pre_idx]) ** 2)))
    post_cf_mean = float(np.mean(cf[post_idx]))
    att = float(np.mean(y_treated[post_idx] - cf[post_idx]))

    return {
        "weights": dict(zip(donors, weights.tolist())),
        "pre_rmse": pre_rmse,
        "post_cf_mean": post_cf_mean,
        "att": att,
        "k": _K,
    }


def main() -> int:
    res = _run()

    print("== REFERENCE VALUES ==")
    for donor, w in res["weights"].items():
        print(f"weight\t{donor}\t{w:.10f}")
    print(f"pre_rmse\t{res['pre_rmse']:.10f}")
    print(f"post_cf_mean\t{res['post_cf_mean']:.10f}")
    print(f"att\t{res['att']:.10f}")
    print(f"k\t{float(res['k']):.1f}")

    print("== SESSION INFO ==")
    import platform
    import sklearn

    print(f"python {platform.python_version()}")
    print(f"numpy {np.__version__}")
    print(f"pandas {pd.__version__}")
    print(f"scikit-learn {sklearn.__version__}")
    print("reference: jehangiramjad/tslib @ "
          "3e50bc1fbe0178bf2c1b21b2ce9fd0f0ca2d5f76 "
          "(RobustSyntheticControl, codeload tarball)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
