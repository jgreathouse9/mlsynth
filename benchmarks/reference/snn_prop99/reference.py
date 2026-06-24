#!/usr/bin/env python3
"""Live captured run of deshen24/syntheticNN on California Proposition 99.

Runs Dwivedi/Shah/Shen's canonical Synthetic Nearest Neighbors implementation
(https://github.com/deshen24/syntheticNN, ``SyntheticNearestNeighbors``) on the
Abadie-Diamond-Hainmueller Prop-99 smoking panel, with California's
post-treatment block (1989-2000) set to ``NaN`` -- the exact block-missingness
problem mlsynth's ``SNN`` solves. The reference is fetched at a pinned commit
into the gitignored ``benchmarks/reference/.cache`` (git clone, else the
codeload tarball); see ``benchmarks/reference/clone_syntheticnn.py`` and the
bundle ``NOTICE``.

Prints the ``== REFERENCE VALUES ==`` block that ``generate.py`` parses (the
imputed counterfactual per post-year, the ATT, and the 2000 gap) followed by a
``== SESSION INFO ==`` block recording numpy / pandas / scikit-learn / networkx
/ Python versions.

Run from the repository root::

    python benchmarks/reference/snn_prop99/reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reference.clone_syntheticnn import import_syntheticnn

_DATA = ROOT / "basedata" / "smoking_data.csv"
_TREAT_YEAR = 1989


def _run() -> dict:
    snn_mod = import_syntheticnn()
    SyntheticNearestNeighbors = snn_mod.SyntheticNearestNeighbors

    df = pd.read_csv(_DATA)
    wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    years = list(wide.columns)
    states = list(wide.index)
    i = states.index("California")
    post_years = [y for y in years if y >= _TREAT_YEAR]

    X = wide.values.astype(float).copy()
    observed_cal = X[i].copy()
    for y in post_years:
        X[i, years.index(y)] = np.nan

    # Donoho-Gavish (universal) rank, single synthetic neighbour -- the default
    # rank rule and the n_neighbors=1 setting mlsynth's SNN uses.
    snn = SyntheticNearestNeighbors(n_neighbors=1, weights="uniform", verbose=False)
    Xhat = snn.fit_transform(X)

    cf = {y: float(Xhat[i, years.index(y)]) for y in post_years}
    gaps = {y: float(observed_cal[years.index(y)] - cf[y]) for y in post_years}
    att = float(np.mean(list(gaps.values())))
    return {"cf": cf, "att": att, "gap_2000": gaps[2000]}


def main() -> int:
    res = _run()

    print("== REFERENCE VALUES ==")
    for y in sorted(res["cf"]):
        print(f"cf_{y}\t{res['cf'][y]:.6f}")
    print(f"snn_att\t{res['att']:.6f}")
    print(f"snn_gap_2000\t{res['gap_2000']:.6f}")

    print("== SESSION INFO ==")
    import platform
    import sklearn
    import networkx
    print(f"python {platform.python_version()}")
    print(f"numpy {np.__version__}")
    print(f"pandas {pd.__version__}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"networkx {networkx.__version__}")
    print("reference: deshen24/syntheticNN @ a95b511 (codeload tarball)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
