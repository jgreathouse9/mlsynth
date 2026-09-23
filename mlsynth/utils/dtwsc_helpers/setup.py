"""Panel ingestion for DTWSC -- a thin wrapper over :func:`dataprep`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import DTWSCInputs


def unit_rescale_factors(
    df: pd.DataFrame,
    unitid: str,
    time: str,
    outcome: str,
    last_pre_period,
):
    """Per-unit ``(offset, multiplier)`` putting every unit on one range.

    Reproduces ``dsc(rescale = TRUE)``, the reference's default. Each unit is
    mapped by ``(x - min_i) * mean_range / (max_i - min_i)``, where the min and
    max are taken over the PRE-treatment window only and ``mean_range`` is the
    mean of those ranges across units. Afterwards every unit spans exactly the
    same pre-treatment interval.

    The point is what the synthetic control then optimises over. Untransformed,
    a donor whose outcome swings widely contributes more squared error than a
    flat one and is fitted harder for reasons unrelated to how well its path
    resembles the treated unit's. Equalising the ranges makes the fit a
    comparison of shape.

    Restricting to the pre-period matters: taking the range over the whole
    panel would let post-treatment variation -- including the treatment effect
    itself -- set the weights.

    A unit that is flat across the entire pre-period has no range to normalise
    and is left alone rather than divided by zero.
    """
    pre = df[df[time] <= last_pre_period]
    if pre.empty:
        raise MlsynthDataError(
            "DTWSC: rescaling needs at least one pre-treatment period; none "
            f"of the panel's {time} values are <= {last_pre_period}."
        )
    ranges = pre.groupby(unitid, observed=True)[outcome].agg(["min", "max"])
    spans = ranges["max"] - ranges["min"]
    mean_span = float(spans.mean())
    factors = {}
    for unit, lo in ranges["min"].items():
        span = float(spans.loc[unit])
        factors[unit] = (float(lo), mean_span / span if span > 0 else 1.0)
    return factors


def rescale_panel(
    df: pd.DataFrame,
    unitid: str,
    time: str,
    outcome: str,
    last_pre_period,
) -> pd.DataFrame:
    """Return ``df`` with ``outcome`` put on the common pre-treatment range."""
    factors = unit_rescale_factors(df, unitid, time, outcome, last_pre_period)
    out = df.copy()
    lo = out[unitid].map(lambda u: factors[u][0])
    mult = out[unitid].map(lambda u: factors[u][1])
    out[outcome] = (out[outcome] - lo) * mult
    return out


def prepare_dtwsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates=None,
) -> DTWSCInputs:
    """Pivot a long panel into :class:`DTWSCInputs` via :func:`dataprep`.

    DTWSC is a single-treated-unit, block-treatment method: it learns one warp
    per donor against one treated path.
    """
    prepped = dataprep(
        df=df,
        unit_id_column_name=unitid,
        time_period_column_name=time,
        outcome_column_name=outcome,
        treatment_indicator_column_name=treat,
    )
    if "y" not in prepped or "donor_matrix" not in prepped:
        raise MlsynthDataError(
            "DTWSC requires a single treated unit with a common treatment "
            "date; the panel resolved to staggered cohorts instead."
        )

    y = np.asarray(prepped["y"], dtype=float).ravel()
    donors = np.asarray(prepped["donor_matrix"], dtype=float)
    if donors.ndim == 1:  # pragma: no cover - dataprep returns (T, J) even for
        # a single donor; this normalises defensively for a future caller.
        donors = donors[:, None]
    n_pre = int(prepped["pre_periods"])

    if donors.shape[1] < 1:  # pragma: no cover - dataprep raises on an empty
        # donor pool before returning, unless allow_no_donors is set, which
        # DTWSC never does; kept so the requirement is stated at this seam.
        raise MlsynthDataError("DTWSC needs at least one donor unit.")
    if n_pre < 3:
        raise MlsynthDataError(
            f"DTWSC needs at least 3 pre-treatment periods to learn a warp; "
            f"the panel has {n_pre}."
        )
    if int(prepped["post_periods"]) < 1:  # pragma: no cover - a panel with no
        # treated period fails dataprep's own treatment-detection first; kept
        # so the requirement is visible where DTWSC depends on it.
        raise MlsynthDataError("DTWSC needs at least one post-treatment period.")
    if not np.isfinite(y).all() or not np.isfinite(donors).all():
        raise MlsynthDataError("DTWSC: outcome contains NaN or infinite values.")

    cov_frame = None
    if covariates:
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise MlsynthDataError(
                f"DTWSC: covariate column(s) {missing} missing from `df`."
            )
        cov_frame = df[[unitid, time, *covariates]].copy()
        cov_frame = cov_frame.rename(columns={unitid: "__unit", time: "__time"})
        cov_frame["__unit"] = cov_frame["__unit"].astype(str)

    return DTWSCInputs(
        y=y,
        donor_matrix=donors,
        donor_names=list(prepped["donor_names"]),
        time_labels=np.asarray(prepped["time_labels"]),
        n_pre=n_pre,
        treated_name=prepped.get("treated_unit_name"),
        covariate_frame=cov_frame,
    )
