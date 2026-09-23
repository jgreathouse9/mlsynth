#!/usr/bin/env python3
"""Live captured run of SucreRouge's Bayesian Robust SC on Proposition 99.

Runs the Bayesian branch of ``SucreRouge/synth_control``
(https://github.com/SucreRouge/synth_control) -- the closed-form Gaussian
posterior over the donor weights of Amjad, Shah & Shen (*Robust Synthetic
Control*, JMLR 19(22):1-51, 2018) -- on the Abadie-Diamond-Hainmueller
California Proposition 99 panel: California treated, the 38-state donor pool,
the 1970-1988 pre-period (T0=19) and the 1989-2000 post-period. The algorithm is
rank-``k`` HSVT of the full donor matrix, a data-driven prior precision (a
forward-chaining ridge-CV penalty), and the conjugate Gaussian posterior over
the weights.

``k`` is fixed to ``3`` -- the same retained rank the point-estimator
cross-validation (``pcr_rsc_ref``) uses -- and the SAME ``k`` is fed to mlsynth's
HSVT denoiser in the case, so the de-noising matches.

The reference is fetched at a pinned commit into the gitignored
``benchmarks/reference/.cache`` (git clone, else codeload tarball); see
``benchmarks/reference/clone_synth_control.py`` and the bundle ``NOTICE``.
Nothing from ``synth_control`` is redistributed in this tree.

Prints the ``== REFERENCE VALUES ==`` block that ``generate.py`` parses (the
learned donor weights as ``weight\\t<donor>\\t<value>`` rows; the observation
noise ``obs_noise_var``, the data-driven ``prior_precision``, the pre-period
RMSE, the post-period mean counterfactual, the ATT, and ``k``) followed by a
``== SESSION INFO ==`` block recording numpy / pandas / scikit-learn / Python
versions.

Run from the repository root::

    python benchmarks/reference/bayesian_rsc_ref/reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reference.clone_synth_control import build_bayesian_rsc

_DATA = ROOT / "basedata" / "smoking_data.csv"
_TREAT_YEAR = 1989
_K = 3
_TREATED = "California"


def _run() -> dict:
    df = pd.read_csv(_DATA)
    wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    years = list(wide.columns)
    states = list(wide.index)
    donors = [s for s in states if s != _TREATED]
    T0 = sum(1 for y in years if y < _TREAT_YEAR)

    target_full = wide.loc[_TREATED, years].values.astype(float)
    donor_full = wide.loc[donors, years].values.T          # (T, J)
    return build_bayesian_rsc(target_full, donor_full, donors, T0, _K)


def main() -> int:
    res = _run()

    print("== REFERENCE VALUES ==")
    for donor, w in res["weights"].items():
        print(f"weight\t{donor}\t{w:.10f}")
    print(f"obs_noise_var\t{res['obs_noise_var']:.10f}")
    print(f"prior_precision\t{res['prior_precision']:.10f}")
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
    print("reference: SucreRouge/synth_control @ "
          "a55ee1482d5db8cda32e18327de0072e7ba5e0b6 "
          "(learn method='bayesian', codeload tarball)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
