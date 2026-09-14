"""Spec-driven construction of the SCMO matching matrix ``Z`` (pure NumPy).

A *spec* describes how to assemble the columns of ``Z`` from a long panel::

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

Rules are either a bare column name (raw level), ``(column, op)`` with ``op``
in ``{"level", "log", "per_capita", "raw"}``, or a dict
``{"column": ..., "op": ..., "year": ...}``. A dict rule that names its own
``year`` is read at that period and enters the matrix once, which is how a
time-invariant predictor is matched on without being repeated at every stacked
period.

With ``demean=True`` each variable's block of stacked columns is centered on
that unit's own mean across the block before the columns are standardized
(Tian-Lee-Panchenko 2026, Online Appendix B.1.1): stable level differences
between units then stop driving the weights. A block of a single column carries
no within-block mean and is passed through untouched.

Each resulting column is standardized by its cross-unit SD
(Tian-Lee-Panchenko footnote 5); columns that are not complete across all units
are dropped (``complete.cases``), as are columns every unit shares once the
matrix is centered, which carry nothing to match on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError
from ..fast_scm_helpers.structure import IndexSet

_DEFAULT_POP = "Population levels"


def _normalize(name: str, rule: Any) -> Tuple[Any, Any]:
    """Split a spec rule into the ``(column, op)`` pair and its pinned period.

    The pinned period is ``None`` for the usual rules, which are read at every
    period in ``spec["year"]``.
    """
    if isinstance(rule, dict):
        if "column" not in rule:
            raise MlsynthConfigError(
                f"Spec rule {name!r} is a dict without a 'column' key: {rule!r}.")
        return (rule["column"], rule.get("op", "level")), rule.get("year")
    return rule, None


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
    demean: bool = False,
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
    demean : bool, default False
        Center each variable's stacked columns on the unit's own block mean
        before standardizing (Online Appendix B.1.1).

    Returns
    -------
    Z : np.ndarray
        Standardized matching matrix, shape ``(N, P)``.
    labels : list of str
        Length-``P`` predictor labels (``name`` or ``name@year`` when several
        periods are stacked).
    col_period : np.ndarray
        Length-``P`` period that each column belongs to (used by the
        ``averaged`` scheme to average across outcomes within a period).
    """
    years = spec["year"]
    years = [years] if np.isscalar(years) else list(years)
    pop_col = spec.get("per_capita_denominator", _DEFAULT_POP)
    var_rules: Dict[str, Any] = spec["vars"]
    normalized = {name: _normalize(name, rule) for name, rule in var_rules.items()}
    stacked = {n: r for n, (r, yr) in normalized.items() if yr is None}

    cols: List[np.ndarray] = []
    labels: List[str] = []
    periods: List[Any] = []
    col_var: List[str] = []
    for yr in years:
        df_year = df[df[time] == yr].set_index(unitid)
        for name, rule in stacked.items():
            cols.append(_column_for_year(df_year, unit_index, rule, pop_col))
            labels.append(name if len(years) == 1 else f"{name}@{yr}")
            periods.append(yr)
            col_var.append(name)
    for name, (rule, pinned_year) in normalized.items():
        if pinned_year is None:
            continue
        rows = df[df[time] == pinned_year]
        if rows.empty:
            raise MlsynthConfigError(
                f"Spec rule {name!r} pins period {pinned_year!r}, which is not in "
                f"the panel's '{time}' column.")
        cols.append(_column_for_year(rows.set_index(unitid), unit_index, rule, pop_col))
        labels.append(name)
        periods.append(pinned_year)
        col_var.append(name)

    Z_raw = np.column_stack(cols)                       # (N, P_raw)
    col_period = np.asarray(periods)

    # complete.cases(t(Z)): drop columns with any non-finite entry across units
    keep = np.all(np.isfinite(Z_raw), axis=0)
    Z_raw = Z_raw[:, keep]
    labels = [l for l, k in zip(labels, keep) if k]
    col_period = col_period[keep]
    col_var = [v for v, k in zip(col_var, keep) if k]

    if demean:
        Z_raw = _demean_blocks(Z_raw, col_var)

    # Then drop what no unit is distinguished by, judged on the matrix as it
    # will be matched on: a column identical across units contributes the same
    # constant to every donor's distance. Centering comes first, so a column
    # that is flat in levels but separates units once its block is centered
    # stays -- which is what the paper's published weights are computed from.
    keep = np.full(Z_raw.shape[1], True)
    if Z_raw.shape[0] > 1:
        with np.errstate(invalid="ignore"):
            keep = np.nan_to_num(Z_raw.std(axis=0, ddof=1), nan=1.0) > 0
    Z_raw = Z_raw[:, keep]
    labels = [l for l, k in zip(labels, keep) if k]
    col_period = col_period[keep]

    # standardize each column by its cross-unit SD (no centering)
    sd = Z_raw.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Z_raw / sd
    return Z, labels, col_period


def _demean_blocks(Z_raw: np.ndarray, col_var: List[str]) -> np.ndarray:
    """Center each variable's block of columns on that unit's block mean.

    A one-column block is left alone: it carries no within-block mean, and
    centering it would erase the variable (Tian-Lee-Panchenko match the
    appendix simulation's observed predictors in levels alongside the demeaned
    outcome blocks).
    """
    out = Z_raw.copy()
    for var in dict.fromkeys(col_var):
        idx = [j for j, v in enumerate(col_var) if v == var]
        if len(idx) < 2:
            continue
        out[:, idx] -= out[:, idx].mean(axis=1, keepdims=True)
    return out
