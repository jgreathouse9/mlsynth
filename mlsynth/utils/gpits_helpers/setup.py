"""Data preparation for GPITS.

Wraps :func:`mlsynth.utils.datautils.dataprep` in single-series mode -- GPITS
needs no cross-sectional donors, so any untreated units in the frame are
ignored -- and assembles the Gaussian-process design matrix.

The design is the time index plus whatever covariates the caller declares.
Column order matters and is not cosmetic: one-hot encoded categoricals come
first, then the continuous columns with the time index leading them. The
period rescaling in :mod:`.pipeline` reads the standard deviation of the first
continuous column, so time has to be it.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from ..helperutils import IndexSet
from .structures import GPITSInputs


def _one_hot(values: np.ndarray, levels: Sequence[Any]) -> np.ndarray:
    """Indicator columns for ``levels``, in the given order."""
    out = np.zeros((len(values), len(levels)), dtype=float)
    for j, lv in enumerate(levels):
        out[:, j] = (values == lv).astype(float)
    return out


def prepare_gpits_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[List[str]] = None,
    categorical_covariates: Optional[List[str]] = None,
) -> GPITSInputs:
    """Pivot a single-treated-unit panel into :class:`GPITSInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel. Untreated units may be present and are ignored.
    outcome, treat, unitid, time : str
        Column names.
    covariates : list of str, optional
        Columns to fold into the design alongside the time index.
    categorical_covariates : list of str, optional
        Subset of ``covariates`` to one-hot encode.

    Returns
    -------
    GPITSInputs

    Raises
    ------
    MlsynthDataError
        If the pre/post split cannot be identified, there is no post-treatment
        period, a declared covariate is missing, or the pre-treatment outcome
        is constant (its standard deviation is the scale the GP standardises
        by, so a constant series has no scale to work with).
    """
    covariates = list(covariates or [])
    categorical = set(categorical_covariates or [])

    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Declared covariates absent from the frame: {missing}. "
            f"Available columns: {sorted(df.columns)}."
        )

    prepared = dataprep(df, unitid, time, outcome, treat, allow_no_donors=True)
    T0 = prepared.get("pre_periods")
    n_post = prepared.get("post_periods")
    if T0 is None or n_post is None:  # pragma: no cover - dataprep returns
        # both counts whenever it returns at all; this guards its contract
        raise MlsynthDataError(
            "dataprep did not return pre/post period counts; ensure the "
            "treated unit and treatment timing are identified."
        )
    T0, n_post = int(T0), int(n_post)
    if n_post < 1:  # pragma: no cover - dataprep rejects a panel with no
        # treated observation before this guard is reached; kept so the
        # requirement is stated where a reader of this module will look
        raise MlsynthDataError(
            "GPITS needs at least one post-treatment period; the treatment "
            "indicator never turns on."
        )
    if T0 < 3:
        raise MlsynthDataError(
            f"GPITS needs at least 3 pre-treatment periods to fit a trend; got {T0}."
        )

    y = np.asarray(prepared["y"], dtype=float).ravel()
    treated_label = prepared.get("treated_unit_name")
    time_labels = np.asarray(prepared["Ywide"].index)
    if y.shape[0] != time_labels.shape[0]:  # pragma: no cover - dataprep
        # builds both from the same pivot, so a mismatch is a contract breach
        raise MlsynthDataError(
            f"Outcome length {y.shape[0]} does not match the {time_labels.shape[0]} "
            "period labels returned by dataprep."
        )

    if float(np.std(y[:T0], ddof=1)) <= 0.0:
        raise MlsynthDataError(
            "The pre-treatment outcome is constant, so it has no scale to "
            "standardise by and the Gaussian process is not identified. "
            "Check the outcome column and the treatment date."
        )

    # Rows of the treated unit only, ordered to match the pivoted series.
    cov_frame: Optional[pd.DataFrame] = None
    if covariates:
        unit_rows = df[df[unitid] == treated_label]
        if unit_rows.empty:  # pragma: no cover - dataprep names a unit that
            unit_rows = df   # is in the frame; this is a belt-and-braces path
        unit_rows = unit_rows.drop_duplicates(subset=[time]).set_index(time)
        try:
            cov_frame = unit_rows.reindex(time_labels)[covariates]
        except KeyError as exc:  # pragma: no cover - covariate presence is
            # checked above, so reindex cannot miss a declared column
            raise MlsynthDataError(
                f"Could not align covariates to the period index: {exc}"
            ) from exc
        if cov_frame.isna().any().any():
            bad = cov_frame.columns[cov_frame.isna().any()].tolist()
            raise MlsynthDataError(
                f"Covariates contain missing values after aligning to the "
                f"period index: {bad}. GPITS needs a complete design."
            )

    cat_blocks: List[np.ndarray] = []
    cat_names: List[str] = []
    cont_cols: List[np.ndarray] = [np.arange(1, len(y) + 1, dtype=float)]
    cont_names: List[str] = ["__time__"]

    if cov_frame is not None:
        for name in covariates:
            col = cov_frame[name].to_numpy()
            if name in categorical:
                # Levels come from the pre-period, which is the training
                # window; a level appearing only after treatment carries no
                # fitted baseline and would be an unfilled column.
                levels = list(pd.unique(pd.Series(col[:T0]).sort_values()))
                cat_blocks.append(_one_hot(col, levels))
                cat_names.extend(f"{name}={lv}" for lv in levels)
            else:
                v = pd.to_numeric(cov_frame[name], errors="coerce").to_numpy(dtype=float)
                if np.isnan(v).any():
                    raise MlsynthDataError(
                        f"Covariate {name!r} is not numeric; declare it in "
                        "'categorical_covariates' to one-hot encode it."
                    )
                if float(np.std(v[:T0], ddof=1)) <= 0.0:
                    raise MlsynthDataError(
                        f"Continuous covariate {name!r} is constant over the "
                        "pre-period, so it cannot be standardised."
                    )
                cont_cols.append(v)
                cont_names.append(name)

    cat = np.hstack(cat_blocks) if cat_blocks else np.empty((len(y), 0))
    design = np.column_stack([cat, np.column_stack(cont_cols)])

    return GPITSInputs(
        time_index=IndexSet.from_labels(time_labels),
        y=y,
        design=design,
        T0=T0,
        n_categorical=cat.shape[1],
        column_names=cat_names + cont_names,
        treated_label=treated_label,
        metadata={"covariates": covariates,
                  "categorical_covariates": sorted(categorical)},
    )
