"""The realized panel on a design's units and periods, and each unit's role."""
from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np

from ...exceptions import MlsynthDataError
from ..datautils import geoex_dataprep
from .structures import CorrectionInputs

#: The corrections were validated with one to three corrected controls. With
#: eleven or twelve, correcting cost more than the spillover did, and a design
#: that excluded the neighbours did better.
MAX_CORRECTED = 3
#: Above this share of the control composite, the corrected controls are most of
#: the counterfactual, and the estimate is mostly the rebuilding fits.
MAX_SHARE = 0.5
#: Relative tolerance for the design's composites reproducing on the panel.
#: The composite weight maps drop weights below 1e-8, and a mixed-integer
#: backend's weights sum to one within its own tolerance.
REPRODUCE_TOL = 1e-6


def prepare_correction(config) -> CorrectionInputs:
    """Pivot the realized panel onto the design's units and timeline, check that
    the design reproduces on it, and sort the units into roles.

    Raises
    ------
    MlsynthDataError
        If the panel lacks a design unit or period, if the design's composites
        do not reproduce on its fitting window, or if no clean control is left
        to rebuild the contaminated ones from.
    """
    design = config.design
    w, c = design.windows, design.composite
    labels = [str(u) for u in (c.members or c.treated_labels + c.control_labels)]

    prep = geoex_dataprep(config.df, config.unitid, config.time, config.outcome)
    wide = prep["Ywide"]                                  # time x unit
    by_label: Dict[str, Any] = {str(col): col for col in wide.columns}
    missing = [u for u in labels if u not in by_label]
    if missing:
        raise MlsynthDataError(
            f"SPILLCORRECT: the panel is missing the design's units {missing}.")

    T = w.T0 + w.post_periods
    if w.time_labels is not None:
        times = list(w.time_labels)[:T]
        absent = [t for t in times if t not in set(wide.index)]
        if absent:
            raise MlsynthDataError(
                f"SPILLCORRECT: the panel is missing {len(absent)} of the design's "
                f"periods, starting at {absent[0]!r}.")
    else:                                               # pragma: no cover - MAREX records its labels
        if wide.shape[0] < T:
            raise MlsynthDataError(
                f"SPILLCORRECT: the design spans {T} periods and the panel "
                f"{wide.shape[0]}.")
        times = list(wide.index[:T])
    Y = wide.loc[times, [by_label[u] for u in labels]].to_numpy(dtype=float).T

    pos = {u: i for i, u in enumerate(labels)}
    wt = np.zeros(len(labels))
    wc = np.zeros(len(labels))
    for u, x in c.treated.items():
        wt[pos[str(u)]] = x
    for u, x in c.control.items():
        wc[pos[str(u)]] = x

    _check_reproduces(design, Y, wt, wc, w.fit_end)

    pool = [pos[str(u)] for u in c.pool]
    affected = [pos[str(u)] for u in config.affected_units]
    corrected = [i for i in affected if wc[i] > 0]
    skipped = [i for i in affected if wc[i] <= 0]
    if skipped:
        warnings.warn(
            f"SPILLCORRECT: affected units {[labels[i] for i in skipped]} have no "
            "control weight, so they carry no bias; they are not corrected and "
            "are kept out of the donors the others are rebuilt from.",
            UserWarning, stacklevel=3)
    clean = [i for i in pool if i not in affected]
    if not clean:
        raise MlsynthDataError(
            "SPILLCORRECT: every control is affected, so no clean control is left "
            "to rebuild the contaminated ones from. Re-design with the affected "
            "units excluded from the control pool.")
    share = float(wc[corrected].sum())
    if len(corrected) > MAX_CORRECTED or share > MAX_SHARE:
        warnings.warn(
            f"SPILLCORRECT: correcting {len(corrected)} controls holding "
            f"{share:.0%} of the control composite. The corrections were validated "
            f"with one to {MAX_CORRECTED}; for a spillover expected to reach this "
            "many, exclude the neighbours at design time (MAREX's adjacency with "
            "exclude_bordering_donors).",
            UserWarning, stacklevel=3)

    return CorrectionInputs(
        Y=Y, labels=labels, time_labels=times if w.time_labels is not None else None,
        wt=wt, wc=wc, pool=pool, corrected=corrected, skipped=skipped, clean=clean,
        T0=w.T0, fit_end=w.fit_end, blank_periods=w.blank_periods,
        post_periods=w.post_periods)


def _check_reproduces(design, Y, wt, wc, fit_end) -> None:
    """The design's composites, applied to the panel's fitting window, give the
    series the design reported. The fitting window is the only data a design
    reads, so a panel that moved there is not the panel this design is the
    solution for."""
    report = getattr(design, "report", None)
    ts = getattr(report, "time_series", None)
    if ts is None or ts.observed_outcome is None or ts.counterfactual_outcome is None:
        return                                          # pragma: no cover - MAREX reports both
    for name, weights, reported in (("treated", wt, ts.observed_outcome),
                                    ("control", wc, ts.counterfactual_outcome)):
        got = weights @ Y[:, :fit_end]
        want = np.asarray(reported, dtype=float)[:fit_end]
        tol = REPRODUCE_TOL * max(float(np.abs(want).max()), 1.0)
        if np.abs(got - want).max() > tol:
            raise MlsynthDataError(
                f"SPILLCORRECT: the design's {name} composite does not reproduce on "
                "the panel's fitting window (largest gap "
                f"{np.abs(got - want).max():.3g}). The design was fit on other data "
                "there, so it is not this panel's design.")
