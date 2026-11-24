"""The typed results of :func:`mlsynth.utils.datautils.dataprep`.

``dataprep`` turns a long panel into the matrices an estimator works on. With one treated unit it returns a
:class:`SinglePanel`: the treated unit's outcome vector, the donors' outcome matrix, and the split of the periods
into pre- and post-treatment. With several treated units it returns a :class:`CohortPanel`, the treated units
grouped into cohorts by the period their treatment starts, each cohort a :class:`Cohort`. Covariates, when asked
for, come as a :class:`CovariateBlock`, and the stacked predictor matrix of the MAREX-family design estimators
(Abadie and Zhao, "Synthetic controls for experimental design", arXiv:2108.02196) as a :class:`PredictorBlock`.

Each type checks, when it is built, that its pieces describe one panel: the outcome vector and every matrix have a
row per period, the pre- and post-periods are non-negative and add up to the total, there is a name for every donor
column, and the covariate and predictor blocks are shaped to the panel's units. A mismatch raises
:class:`~mlsynth.exceptions.MlsynthDataError`, so a panel whose parts disagree stops at ingestion instead of
reaching an estimator as numbers of the wrong shape.

Each type is also a read-only :class:`collections.abc.Mapping` with exactly the keys the dictionary ``dataprep``
returned before these types existed, in the same order, each key's value the field itself. Callers that read
``prep["y"]`` or test ``"cohorts" in prep`` keep working while they move to the fields (issue #1138).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mlsynth.exceptions import MlsynthDataError

__all__ = ["CovariateBlock", "PredictorBlock", "Cohort", "SinglePanel", "CohortPanel"]


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise MlsynthDataError(f"Prepared panel is inconsistent: {message}")


def _periods(pre: int, post: int, total: int) -> None:
    _check(pre >= 0 and post >= 0, f"pre_periods ({pre}) and post_periods ({post}) must not be negative.")
    _check(pre + post == total, f"pre_periods ({pre}) + post_periods ({post}) must equal total_periods ({total}).")


class _View(Mapping):
    """The read-only mapping over a prepared panel's fields, keyed as ``dataprep``'s dictionary was."""

    __slots__ = ()

    def _pairs(self) -> Dict[str, Any]:  # pragma: no cover - every subclass defines it
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self._pairs()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._pairs())

    def __len__(self) -> int:
        return len(self._pairs())


@dataclass(frozen=True, eq=False, slots=True)
class CovariateBlock:
    """Covariates aggregated to one row per unit.

    Attributes
    ----------
    matrix : np.ndarray
        ``(N, M)``: a row per unit, a column per covariate (donors first, then the treated unit, for one treated
        unit; the wide panel's column order for several).
    names : tuple of str
        The ``M`` covariate names.
    means, scales : np.ndarray
        ``(M,)``: the column means and scales used to standardise ``matrix``.
    donor_matrix, treated_matrix : np.ndarray or None
        For one treated unit, ``matrix`` split into the donors' rows and the treated unit's row; None for several.
    """

    matrix: np.ndarray
    names: Tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    donor_matrix: Optional[np.ndarray] = None
    treated_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        _check(np.ndim(self.matrix) == 2, "the covariate matrix must have a row per unit and a column per covariate.")
        n, m = np.shape(self.matrix)
        _check(len(self.names) == m, f"{len(self.names)} covariate names for {m} covariate columns.")
        _check(np.shape(self.means) == (m,), f"covariate means of shape {np.shape(self.means)} for {m} covariates.")
        _check(np.shape(self.scales) == (m,), f"covariate scales of shape {np.shape(self.scales)} for {m} covariates.")
        split = (self.donor_matrix is None, self.treated_matrix is None)
        _check(split in ((True, True), (False, False)),
               "donor_matrix and treated_matrix must be given together or not at all.")
        if self.donor_matrix is not None:
            d, t = np.shape(self.donor_matrix), np.shape(self.treated_matrix)
            _check(len(d) == 2 and len(t) == 2 and d[1] == m and t[1] == m and d[0] + t[0] == n,
                   f"donor_matrix {d} and treated_matrix {t} must split the {n} by {m} covariate matrix by rows.")


@dataclass(frozen=True, eq=False, slots=True)
class PredictorBlock:
    """The stacked predictor matrix of the MAREX-family design estimators.

    Attributes
    ----------
    matrix : np.ndarray
        ``(T_pre + M, N)``: the pre-period outcomes stacked on the transposed covariate matrix, a column per unit.
    block_sizes : tuple of int
        ``(T_pre, M)``: the rows of each block.
    f_weights : np.ndarray or None
        ``(N,)``: the population weights, equal by default.
    population_mean : np.ndarray or None
        ``(T_pre + M,)``: ``matrix @ f_weights``, the design objective's target.
    """

    matrix: np.ndarray
    block_sizes: Tuple[int, int]
    f_weights: Optional[np.ndarray] = None
    population_mean: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        _check(np.ndim(self.matrix) == 2, "the predictor matrix must have a row per predictor and a column per unit.")
        rows, cols = np.shape(self.matrix)
        _check(len(self.block_sizes) == 2 and min(self.block_sizes) >= 0 and sum(self.block_sizes) == rows,
               f"predictor block_sizes {tuple(self.block_sizes)} must add up to the matrix's {rows} rows.")
        _check(self.f_weights is None or np.shape(self.f_weights) == (cols,),
               f"f_weights of shape {np.shape(self.f_weights)} for {cols} units.")
        _check(self.population_mean is None or np.shape(self.population_mean) == (rows,),
               f"population_mean of shape {np.shape(self.population_mean)} for {rows} predictors.")


def _covariate_pairs(cov: Optional[CovariateBlock], out: Dict[str, Any]) -> None:
    if cov is None:
        return
    out["covariate_matrix"] = cov.matrix
    out["covariate_names"] = cov.names
    out["covariate_means"] = cov.means
    out["covariate_scales"] = cov.scales
    if cov.donor_matrix is not None:
        out["donor_covariate_matrix"] = cov.donor_matrix
        out["treated_covariate_matrix"] = cov.treated_matrix


def _predictor_pairs(pred: Optional[PredictorBlock], out: Dict[str, Any]) -> None:
    if pred is None:
        return
    out["predictor_matrix"] = pred.matrix
    out["predictor_block_sizes"] = pred.block_sizes
    out["f_weights"] = pred.f_weights
    out["X_population_mean"] = pred.population_mean


def _blocks(cov: Optional[CovariateBlock], pred: Optional[PredictorBlock], n_units: int) -> None:
    _check(cov is None or np.shape(cov.matrix)[0] == n_units,
           f"the covariate block has {None if cov is None else np.shape(cov.matrix)[0]} rows for {n_units} units.")
    _check(pred is None or np.shape(pred.matrix)[1] == n_units,
           f"the predictor block has {None if pred is None else np.shape(pred.matrix)[1]} columns for {n_units} "
           "units.")


@dataclass(frozen=True, eq=False, slots=True)
class SinglePanel(_View):
    """A panel with one treated unit.

    Attributes
    ----------
    treated_unit_name : hashable
        The treated unit's identifier.
    Ywide : pd.DataFrame
        ``(T, J + 1)``: the outcomes, a row per period and a column per unit.
    y : np.ndarray
        ``(T,)``: the treated unit's outcomes.
    donor_names : pd.Index
        The ``J`` donors, in the order of ``donor_matrix``'s columns.
    donor_matrix : np.ndarray
        ``(T, J)``: the donors' outcomes.
    total_periods, pre_periods, post_periods : int
        ``T``, and its split at the period the treatment starts.
    time_labels : pd.Index
        The ``T`` periods, in order.
    covariates : CovariateBlock or None
        With ``dataprep(covariates=...)``.
    predictors : PredictorBlock or None
        With ``dataprep(marex=True)``.
    """

    treated_unit_name: Hashable
    Ywide: pd.DataFrame
    y: np.ndarray
    donor_names: pd.Index
    donor_matrix: np.ndarray
    total_periods: int
    pre_periods: int
    post_periods: int
    time_labels: pd.Index
    covariates: Optional[CovariateBlock] = None
    predictors: Optional[PredictorBlock] = None

    def __post_init__(self) -> None:
        T = self.total_periods
        _periods(self.pre_periods, self.post_periods, T)
        _check(np.shape(self.y) == (T,), f"y has shape {np.shape(self.y)} for {T} periods.")
        d = np.shape(self.donor_matrix)
        _check(len(d) == 2 and d[0] == T, f"donor_matrix has shape {d} for {T} periods.")
        _check(len(self.donor_names) == d[1], f"{len(self.donor_names)} donor_names for {d[1]} donor columns.")
        _check(len(self.time_labels) == T, f"{len(self.time_labels)} time_labels for {T} periods.")
        _check(self.Ywide.shape == (T, d[1] + 1),
               f"Ywide has shape {self.Ywide.shape} for {T} periods and {d[1] + 1} units.")
        _blocks(self.covariates, self.predictors, d[1] + 1)

    def _pairs(self) -> Dict[str, Any]:
        out = {"treated_unit_name": self.treated_unit_name, "Ywide": self.Ywide, "y": self.y,
               "donor_names": self.donor_names, "donor_matrix": self.donor_matrix,
               "total_periods": self.total_periods, "pre_periods": self.pre_periods,
               "post_periods": self.post_periods, "time_labels": self.time_labels}
        _covariate_pairs(self.covariates, out)
        _predictor_pairs(self.predictors, out)
        return out


@dataclass(frozen=True, eq=False, slots=True)
class Cohort(_View):
    """The treated units whose treatment starts in one period, with the never-treated units as donors.

    Attributes
    ----------
    treated_units : list
        The cohort's units, in the order of ``y``'s columns.
    y : np.ndarray
        ``(T, k)``: their outcomes.
    donor_names : list
        The never-treated units, in the order of ``donor_matrix``'s columns.
    donor_matrix : np.ndarray
        ``(T, J)``: their outcomes.
    total_periods, pre_periods, post_periods : int
        ``T``, and its split at the cohort's start.
    """

    treated_units: List[Hashable]
    y: np.ndarray
    donor_names: Sequence[Hashable]
    donor_matrix: np.ndarray
    total_periods: int
    pre_periods: int
    post_periods: int

    def __post_init__(self) -> None:
        T = self.total_periods
        _periods(self.pre_periods, self.post_periods, T)
        k = len(self.treated_units)
        _check(np.shape(self.y) == (T, k), f"y has shape {np.shape(self.y)} for {T} periods and {k} treated units.")
        J = len(self.donor_names)
        _check(np.shape(self.donor_matrix) == (T, J),
               f"donor_matrix has shape {np.shape(self.donor_matrix)} for {T} periods and {J} donors.")

    def _pairs(self) -> Dict[str, Any]:
        return {"treated_units": self.treated_units, "y": self.y, "donor_names": self.donor_names,
                "donor_matrix": self.donor_matrix, "total_periods": self.total_periods,
                "pre_periods": self.pre_periods, "post_periods": self.post_periods}


@dataclass(frozen=True, eq=False, slots=True)
class CohortPanel(_View):
    """A panel with several treated units, grouped by the period their treatment starts.

    Attributes
    ----------
    Ywide : pd.DataFrame
        ``(T, N)``: the outcomes, a row per period and a column per unit.
    cohorts : dict
        Each start period's :class:`Cohort`.
    time_labels : pd.Index
        The ``T`` periods, in order.
    covariates : CovariateBlock or None
        With ``dataprep(covariates=...)``, over every unit; no donor and treated split.
    predictors : PredictorBlock or None
        With ``dataprep(marex=True)``, over the earliest cohort's pre-periods.
    """

    Ywide: pd.DataFrame
    cohorts: Dict[Hashable, Cohort]
    time_labels: pd.Index
    covariates: Optional[CovariateBlock] = None
    predictors: Optional[PredictorBlock] = None

    def __post_init__(self) -> None:
        T = len(self.time_labels)
        _check(self.Ywide.shape[0] == T, f"{T} time_labels for Ywide's {self.Ywide.shape[0]} periods.")
        for start, c in self.cohorts.items():
            _check(isinstance(c, Cohort), f"the cohort starting at {start!r} is a {type(c).__name__}, not a Cohort.")
            _check(c.total_periods == T, f"the cohort starting at {start!r} has {c.total_periods} periods; the "
                                         f"panel has {T}.")
        _blocks(self.covariates, self.predictors, self.Ywide.shape[1])

    def _pairs(self) -> Dict[str, Any]:
        out = {"Ywide": self.Ywide, "cohorts": self.cohorts, "time_labels": self.time_labels}
        _covariate_pairs(self.covariates, out)
        _predictor_pairs(self.predictors, out)
        return out
