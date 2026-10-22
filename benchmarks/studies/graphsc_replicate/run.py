"""Graph-SC, demonstrated before it is built: Prop 99 (Path A) and the simulation (Path B).

Run::

    python -m benchmarks.studies.graphsc_replicate.run

Results are written to ``results/graphsc.json``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import oracle as G

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results" / "graphsc.json"


def prop99():
    """``(y, X, T0)`` for California and the 38 donors, through ``dataprep``."""
    from mlsynth.utils.datautils import dataprep

    df = pd.read_csv(ROOT / "basedata" / "smoking_data.csv")
    df["treat"] = df["Proposition 99"].astype(int)
    p = dataprep(df, "state", "year", "cigsale", "treat")
    return np.asarray(p["y"], float), np.asarray(p["donor_matrix"], float), int(p["pre_periods"])
