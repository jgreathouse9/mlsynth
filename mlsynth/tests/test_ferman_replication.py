"""TSSC's MSCa reproduces Ferman & Pinto (2021)'s demeaned SC on the Basque panel.

The demeaned SC of Ferman & Pinto (2021, Quantitative Economics 12:1197-1221) --
simplex donor weights plus a free intercept -- is exactly the ``MSCa`` variant of
:class:`~mlsynth.TSSC`. On the identified Basque/ETA panel (1975 cutoff, C < n)
mlsynth's ``MSCa`` matches the authors' own R ``quadprog`` QP value-for-value.

The golden, dashboard-facing version runs the authors' R code live
(``benchmarks/cases/ferman_demeaned_basque.py``). This pins the same numbers in
the fast unit suite, offline, against the captured R reference (no R required).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import TSSC

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "basedata" / "basque_data.csv"
_REF = json.loads(
    (_ROOT / "benchmarks" / "reference" / "ferman_demeaned_basque" / "reference.json").read_text()
)
_TREATED = "Basque Country (Pais Vasco)"
_DROP = ["Spain (Espana)", "Syntetic Basque Country"]


def _msca(treat_year: int):
    df = pd.read_csv(_DATA)
    df = df[~df.regionname.isin(_DROP)].copy()
    df["treat"] = ((df.regionname == _TREATED) & (df.year >= treat_year)).astype(int)
    df = df[["regionname", "year", "gdpcap", "treat"]].dropna()
    return TSSC({"df": df, "outcome": "gdpcap", "treat": "treat",
                "unitid": "regionname", "time": "year",
                "display_graphs": False}).fit().variants["MSCa"]


def test_msca_matches_demeaned_sc_weights_1975():
    """Identified case (C < n): donor weights reproduce the authors' demeaned SC."""
    m = _msca(1975)
    w = pd.Series(dict(m.donor_weights), dtype=float)
    ref = pd.Series(_REF["weights"], dtype=float)
    l1 = float((w.reindex(ref.index).fillna(0.0) - ref).abs().sum())
    assert l1 < 1e-3                                       # value-for-value, QP-tight


def test_msca_matches_att_and_intercept_1975():
    m = _msca(1975)
    assert float(m.att) == pytest.approx(_REF["values"]["att"], abs=1e-2)
    assert float(m.intercept) == pytest.approx(_REF["values"]["intercept"], abs=1e-3)


def test_rank_deficient_1970_is_a_different_solution():
    """Cautionary half: at 1970 the panel is rank-deficient (C > n), so the
    demeaned-SC weights are non-unique -- mlsynth's MSCa lands on a different
    (equally pre-fitting) minimiser than the identified 1975 solution."""
    w70 = pd.Series(dict(_msca(1970).donor_weights), dtype=float)
    w75 = pd.Series(dict(_msca(1975).donor_weights), dtype=float)
    idx = w70.index.union(w75.index)
    l1 = float((w70.reindex(idx).fillna(0) - w75.reindex(idx).fillna(0)).abs().sum())
    assert l1 > 0.3                                        # materially different corners
