"""Long-DataFrame -> NumPy boundary for SCMO (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..fast_scm_helpers.structure import IndexSet
from .matrix_builder import build_matching_matrix
from .structures import SCMOInputs


def prepare_scmo_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    spec: Dict[str, Any],
    treated_unit: Any,
    intervention_time: Any,
    demean: bool = False,
    donors: Optional[Sequence[Any]] = None,
) -> SCMOInputs:
    """Pivot the panel to NumPy, build ``IndexSet``\\ s and the matching matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel (one row per unit-period).
    unitid, time, outcome : str
        Column names for unit id, time, and the primary outcome.
    spec : dict
        Matching specification consumed by
        :func:`mlsynth.utils.scmo_helpers.matrix_builder.build_matching_matrix`.
    treated_unit : Any
        Label of the treated unit.
    intervention_time : Any
        First treated period; pre-period is ``time < intervention_time``.
    demean : bool, default False
        Center each variable's stacked matching columns on the unit's own block
        mean before standardizing (Tian-Lee-Panchenko Online Appendix B.1.1).
    donors : sequence, optional
        Units allowed to carry weight. Default: every unit but the treated one.
        The restriction is a choice set, not a change of the data -- the
        matching matrix is still built and standardized on every unit -- which
        is what makes leave-one-out refits comparable to the full fit.

    Returns
    -------
    SCMOInputs
        Pure-NumPy container for the estimation engine. Its ``time_index`` and
        ``Y`` cover the periods the outcome is observed in, which can be a
        subset of the periods the matching matrix spans; ``metadata`` records
        the rest under ``dropped_outcome_periods``.
    """
    units = list(pd.unique(df[unitid]))
    if treated_unit not in units:
        raise MlsynthDataError(f"treated_unit {treated_unit!r} not found in '{unitid}'.")
    unit_index = IndexSet.from_labels(units)

    times = np.sort(pd.unique(df[time]))

    Ywide = df.pivot(index=unitid, columns=time, values=outcome).reindex(unit_index.labels)[times]
    # A period no unit observes the outcome in carries nothing for the outcome
    # panel, and is dropped from it: the matching matrix keeps every period, so
    # outcomes observed at different frequencies share one panel (the COVID
    # application of Tian-Lee-Panchenko, where a daily series is matched
    # alongside a quarterly one). A period only *some* units are missing is a
    # broken panel, and still raises below.
    unobserved = Ywide.isna().all(axis=0).to_numpy()
    dropped = [t for t, gone in zip(times, unobserved) if gone]
    if dropped:
        Ywide = Ywide.loc[:, ~unobserved]
        times = times[~unobserved]
    if Ywide.isna().any().any():
        raise MlsynthDataError("SCMO requires a complete outcome panel after pivoting.")
    Y = Ywide.to_numpy(dtype=float)
    time_index = IndexSet.from_labels(times)

    T0 = int(np.sum(times < intervention_time))
    if T0 < 1:
        raise MlsynthDataError("No pre-treatment periods (check intervention_time).")

    Z, predictor_labels, col_period = build_matching_matrix(
        df, unitid=unitid, time=time, spec=spec, unit_index=unit_index,
        demean=demean,
    )

    treated_idx = int(unit_index.get_index([treated_unit])[0])
    donor_idx = _donor_index(unit_index, treated_idx, donors)

    return SCMOInputs(
        unit_index=unit_index,
        time_index=time_index,
        treated_idx=treated_idx,
        donor_idx=donor_idx,
        Y=Y,
        T0=T0,
        Z=Z,
        predictor_labels=predictor_labels,
        col_period=col_period,
        metadata={"spec": spec, "outcome": outcome,
                  "intervention_time": intervention_time, "demean": demean,
                  "dropped_outcome_periods": dropped},
    )


def _donor_index(unit_index: IndexSet, treated_idx: int,
                 donors: Optional[Sequence[Any]]) -> np.ndarray:
    """Row indices of the units allowed to carry weight."""
    if donors is None:
        return np.array([i for i in range(len(unit_index)) if i != treated_idx],
                        dtype=int)
    labels = list(unit_index.labels)
    treated = labels[treated_idx]
    chosen = list(donors)
    if not chosen:
        raise MlsynthDataError("donors must name at least one donor unit.")
    unknown = [d for d in chosen if d not in labels]
    if unknown:
        raise MlsynthDataError(f"donors names units absent from the panel: {unknown}.")
    if treated in chosen:
        raise MlsynthDataError(
            f"donors names the treated unit {treated!r}; a unit cannot be its own donor.")
    keep = set(chosen)
    return np.array([i for i, label in enumerate(labels) if label in keep], dtype=int)
