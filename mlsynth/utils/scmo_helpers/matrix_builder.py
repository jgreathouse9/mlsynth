"""Spec-driven construction of the SCMO matching matrix ``Z`` (pure NumPy).

A *spec* describes how to assemble the columns of ``Z`` from a long panel:

    spec = {
        "year": 1989,                       # int, or list[int] to stack periods
        "vars": {
            "private_social_exp": "Private social expenditure",   # raw column
            "electricity_pc": ("Electricity generation", "per_capita"),
            "patents_pc":     ("Triadic patent families", "per_capita"),
            "gdp_growth":     "Real GDP growth",
            "gdp_pc":         "gdp",
            ...
        },
        "per_capita_denominator": "Population levels",   # optional, default
    }

Rules are either a bare column name (raw level) or ``(column, op)`` with
``op`` in ``{"level", "log", "per_capita", "raw"}``. Each resulting column is
standardized by its cross-unit SD (Tian-Lee-Panchenko footnote 5); columns
that are not complete across all units are dropped (``complete.cases``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..fast_scm_helpers.structure import IndexSet

_DEFAULT_POP = "Population levels"


def _column_for_year(
    df_year: pd.DataFrame, unit_index: IndexSet, rule: Any, pop_col: str
) -> np.ndarray:
    """Resolve one spec rule to a length-N ordered NumPy column."""
    if isinstance(rule, str):
        col, op = rule, "level"
    elif isinstance(rule, (tuple, list)) and len(rule) == 2:
        col, op = rule
    else:
        raise ValueError(f"Bad spec rule: {rule!r}")

    series = df_year[col]
    if op in ("level", "raw"):
        vals = series
    elif op == "log":
        vals = np.log(series)
    elif op == "per_capita":
        vals = series / df_year[pop_col]
    else:
        raise ValueError(f"Unknown operation: {op!r}")
    # order to the canonical unit index
    return vals.reindex(unit_index.labels).to_numpy(dtype=float)


def build_matching_matrix(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    spec: Dict[str, Any],
    unit_index: IndexSet,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble the standardized matching matrix ``Z`` from ``spec``.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel (one row per unit-period) with the spec's columns.
    unitid, time : str
        Unit and time column names.
    spec : dict
        Matching specification (see module docstring).
    unit_index : IndexSet
        Canonical unit ordering for the rows of ``Z``.

    Returns
    -------
    Z : np.ndarray
        Standardized matching matrix, shape ``(N, P)``.
    labels : list of str
        Length-``P`` predictor labels (``name`` or ``name@year`` when several
        periods are stacked).
    """
    years = spec["year"]
    years = [years] if np.isscalar(years) else list(years)
    pop_col = spec.get("per_capita_denominator", _DEFAULT_POP)
    var_rules: Dict[str, Any] = spec["vars"]

    cols: List[np.ndarray] = []
    labels: List[str] = []
    for yr in years:
        df_year = df[df[time] == yr].set_index(unitid)
        for name, rule in var_rules.items():
            cols.append(_column_for_year(df_year, unit_index, rule, pop_col))
            labels.append(name if len(years) == 1 else f"{name}@{yr}")

    Z_raw = np.column_stack(cols)                       # (N, P_raw)

    # complete.cases(t(Z)): drop columns with any non-finite entry across units
    keep = np.all(np.isfinite(Z_raw), axis=0)
    Z_raw, labels = Z_raw[:, keep], [l for l, k in zip(labels, keep) if k]

    # standardize each column by its cross-unit SD (no centering)
    sd = Z_raw.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Z_raw / sd
    return Z, labels
