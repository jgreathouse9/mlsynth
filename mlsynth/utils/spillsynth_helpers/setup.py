"""Panel preparation and spillover-structure construction for SPILLSYNTH.

Two responsibilities:

1. :func:`prepare_spillsynth_inputs` turns the long-format DataFrame into
   the ``(N, T)`` panel matrix that the Cao-Dowd estimator consumes,
   with row 0 the treated unit, rows ``1 .. p`` the user-declared
   affected units, and the rest the clean controls.
2. The ``build_A_*`` helpers construct the spillover-structure matrix
   ``A`` for each of the three examples in Cao & Dowd (2023, v3
   numbering -- arXiv 1902.07343v3, January 2026):

   * :func:`build_A_per_unit` (Example 1, Section 2.2 -- "Limited range").
     Each declared affected unit gets its own free spillover coefficient.
     ``A`` is ``(N, 1 + p)``. This is Cao-Dowd's *leading* example.
   * :func:`build_A_homogeneous` (Example 2, Section 3.4 --
     "Homogeneous spillovers"). All declared affected units share a
     single spillover coefficient. ``A`` is ``(N, 2)``.
   * :func:`build_A_distance_decay` (Example 3, Section 7.1 --
     "Exponential decay"). Spillover decays as :math:`b \\exp(-d_i)`
     with a known distance per control unit. ``A`` is ``(N, 2)``.

The legacy alias :func:`build_A_example3` is kept for the existing
test scripts; it now resolves to :func:`build_A_per_unit` (v3 Example 1
== v2 Example 3).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SpillSynthInputs


def _resolve_treated_label(df: pd.DataFrame, unitid: str, treat: str) -> Any:
    """Identify the unique treated unit in the long-format panel."""
    treated_rows = df.loc[df[treat] != 0, unitid].unique()
    if len(treated_rows) == 0:
        raise MlsynthDataError(
            "SPILLSYNTH: no treated rows found (every row has treat == 0)."
        )
    if len(treated_rows) > 1:
        raise MlsynthDataError(
            "SPILLSYNTH currently supports a single treated unit. "
            f"Found {len(treated_rows)}: {list(treated_rows)}."
        )
    return treated_rows[0]


def _resolve_intervention_time(
    df: pd.DataFrame, unitid: str, time: str, treat: str, treated_label: Any
) -> Any:
    """First time period at which the treated unit has treat != 0."""
    treated_df = df.loc[df[unitid] == treated_label, [time, treat]]
    on = treated_df.loc[treated_df[treat] != 0, time]
    if on.empty:
        raise MlsynthDataError(
            f"Treated unit {treated_label!r} never has treat != 0."
        )
    return on.min()


def build_A_per_unit(N: int, p: int) -> np.ndarray:
    """Cao-Dowd v3 Example 1 -- per-unit free spillover coefficients.

    Returns an ``(N, 1 + p)`` matrix whose columns are unit basis
    vectors for the treated unit (row 0) and each of the ``p``
    potentially-affected control units (rows ``1 .. p``). All other
    rows are zero, so clean controls have zero spillover.
    """
    if p < 0 or N < 1 + p:
        raise MlsynthDataError(
            f"SPILLSYNTH: cannot build A with N={N}, p={p}."
        )
    A = np.zeros((N, 1 + p))
    A[0, 0] = 1.0
    for k in range(p):
        A[1 + k, 1 + k] = 1.0
    return A


def build_A_homogeneous(N: int, p: int) -> np.ndarray:
    """Cao-Dowd v3 Example 2 -- single shared spillover coefficient ``b``.

    Returns an ``(N, 2)`` matrix with row 0 = ``(1, 0)`` (treated unit,
    treatment effect), rows ``1 .. p`` = ``(0, 1)`` (affected units, all
    sharing the same spillover ``b``), and rows ``p+1 .. N-1`` =
    ``(0, 0)`` (clean controls). The effect vector is
    :math:`A \\gamma = (\\alpha_1, b, \\dots, b, 0, \\dots, 0)^\\prime`.
    """
    if p < 1 or N < 1 + p:
        raise MlsynthDataError(
            f"SPILLSYNTH: Example 2 (homogeneous) needs p >= 1 affected "
            f"units and N >= 1 + p; got N={N}, p={p}."
        )
    A = np.zeros((N, 2))
    A[0, 0] = 1.0
    A[1:1 + p, 1] = 1.0
    return A


def build_A_distance_decay(decay_weights: np.ndarray) -> np.ndarray:
    """Cao-Dowd v3 Example 3 -- exponential decay ``alpha_i = b * exp(-d_i)``.

    Parameters
    ----------
    decay_weights : np.ndarray
        Length-``N - 1`` vector of decay weights :math:`\\exp(-d_i)` for
        control units ``i = 2, \\dots, N`` (in the row order of the
        underlying panel). The treated unit (row 0) is **not** included
        here; its row is hard-coded as ``(1, 0)``.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)``. Row 0 is ``(1, 0)``; row ``i`` (``i >= 1``)
        is ``(0, decay_weights[i - 1])``. The effect vector is
        :math:`A \\gamma = (\\alpha_1, b \\exp(-d_2), \\dots,
        b \\exp(-d_N))^\\prime`.
    """
    decay_weights = np.asarray(decay_weights, dtype=float).ravel()
    if decay_weights.ndim != 1:
        raise MlsynthDataError(
            "SPILLSYNTH: decay_weights must be a 1-D array."
        )
    if not np.all(np.isfinite(decay_weights)):
        raise MlsynthDataError(
            "SPILLSYNTH: decay_weights must be finite (no NaN/inf)."
        )
    if np.any(decay_weights < 0):
        raise MlsynthDataError(
            "SPILLSYNTH: decay_weights must be non-negative."
        )
    if not np.any(decay_weights > 0):
        raise MlsynthDataError(
            "SPILLSYNTH: decay_weights are all zero -- A would have rank 1."
        )
    N = 1 + decay_weights.size
    A = np.zeros((N, 2))
    A[0, 0] = 1.0
    A[1:, 1] = decay_weights
    return A


# Legacy alias: scripts and tests committed before this refactor still
# import ``build_A_example3`` for the per-unit (now v3 Example 1) helper.
build_A_example3 = build_A_per_unit


def _decay_weights_for_panel(
    unit_distances: Dict[Any, float],
    treated_label: Any,
    control_labels: Sequence[Any],
) -> np.ndarray:
    """Map a user-provided ``{label: distance}`` dict to a decay-weight vector.

    Controls **not** present in ``unit_distances`` get weight zero
    (interpreted as "infinitely far -- no spillover"). Treated unit's
    distance, if provided, is ignored (its row is hard-coded
    elsewhere).
    """
    if not isinstance(unit_distances, dict):
        raise MlsynthDataError(
            "SPILLSYNTH: unit_distances must be a dict mapping unit label "
            "to a non-negative scalar distance from the treated unit."
        )
    extras = set(unit_distances).difference([treated_label, *control_labels])
    if extras:
        raise MlsynthDataError(
            f"SPILLSYNTH: unit_distances has labels not in the panel: "
            f"{sorted(extras, key=str)}."
        )
    n_controls = len(control_labels)
    w = np.zeros(n_controls)
    for i, label in enumerate(control_labels):
        d = unit_distances.get(label)
        if d is None:
            continue
        if not np.isfinite(d) or d < 0:
            raise MlsynthDataError(
                f"SPILLSYNTH: distance for {label!r} must be a finite "
                f"non-negative number; got {d!r}."
            )
        w[i] = float(np.exp(-d))
    if not np.any(w > 0):
        raise MlsynthDataError(
            "SPILLSYNTH: every declared decay weight is zero. Provide at "
            "least one control with finite distance via unit_distances."
        )
    return w


SPILLOVER_STRUCTURES = ("per_unit", "homogeneous", "distance_decay")


def prepare_spillsynth_inputs(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    affected_units: Optional[Sequence[Any]] = None,
    spillover_structure: str = "per_unit",
    unit_distances: Optional[Dict[Any, float]] = None,
) -> SpillSynthInputs:
    """Reshape a long-format panel into the inputs SPILLSYNTH expects.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel.
    outcome, treat, unitid, time : str
        Column names.
    affected_units : sequence, optional
        Labels of control units the researcher believes are potentially
        exposed to spillover from the treated unit. Required for
        ``"per_unit"`` (when ``p > 0``) or ``"homogeneous"``; ignored
        for ``"distance_decay"`` (which uses ``unit_distances``).
        The treated unit must NOT appear in this list. If ``None`` with
        ``"per_unit"``, the estimator reduces to vanilla demeaned SCM
        (``p = 0``).
    spillover_structure : {"per_unit", "homogeneous", "distance_decay"}
        Choice of A-matrix construction (Cao-Dowd v3 Examples 1, 2, 3
        respectively).
    unit_distances : dict, optional
        ``{unit_label: distance}`` mapping, used only when
        ``spillover_structure == "distance_decay"``. Controls absent
        from the mapping are treated as infinitely distant (zero
        weight).

    Returns
    -------
    SpillSynthInputs
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(
                f"SPILLSYNTH: required column {col!r} not in df.columns."
            )

    df = df[[unitid, time, outcome, treat]].copy()
    df = df.dropna(subset=[outcome])
    if df.empty:
        raise MlsynthDataError("SPILLSYNTH: panel is empty after dropping NaN outcomes.")

    treated_label = _resolve_treated_label(df, unitid, treat)
    intervention = _resolve_intervention_time(df, unitid, time, treat, treated_label)

    if spillover_structure not in SPILLOVER_STRUCTURES:
        raise MlsynthDataError(
            f"SPILLSYNTH: unknown spillover_structure {spillover_structure!r}. "
            f"Choose from {SPILLOVER_STRUCTURES}."
        )

    affected = tuple(affected_units or ())
    if treated_label in affected:
        raise MlsynthDataError(
            f"SPILLSYNTH: the treated unit {treated_label!r} cannot also "
            "appear in affected_units."
        )
    if len(set(affected)) != len(affected):
        raise MlsynthDataError("SPILLSYNTH: affected_units contains duplicates.")
    units_in_df = set(df[unitid].unique())
    missing = [u for u in affected if u not in units_in_df]
    if missing:
        raise MlsynthDataError(
            f"SPILLSYNTH: affected_units {missing} not present in panel."
        )

    if spillover_structure == "distance_decay" and unit_distances is None:
        raise MlsynthDataError(
            "SPILLSYNTH: spillover_structure='distance_decay' requires "
            "unit_distances={label: distance, ...}."
        )

    if spillover_structure == "homogeneous" and len(affected) == 0:
        raise MlsynthDataError(
            "SPILLSYNTH: spillover_structure='homogeneous' needs at least "
            "one affected unit in affected_units."
        )

    # Order: treated first, affected next, clean controls after (sorted).
    all_controls = sorted([u for u in units_in_df if u != treated_label],
                          key=lambda x: str(x))
    clean = tuple(u for u in all_controls if u not in affected)
    units = [treated_label, *affected, *clean]

    # Pivot to wide (T x N), then transpose to (N, T).
    wide = (df.pivot(index=time, columns=unitid, values=outcome)
              .reindex(columns=units)
              .sort_index())
    if wide.isna().any().any():
        bad_cells = wide.isna().sum().sum()
        raise MlsynthDataError(
            f"SPILLSYNTH: panel has {int(bad_cells)} missing cells after pivot."
        )

    time_labels = wide.index.to_numpy()
    pre_mask = time_labels < intervention
    if not pre_mask.any():
        raise MlsynthDataError(
            "SPILLSYNTH: no pre-treatment periods (intervention at first period)."
        )
    if pre_mask.all():
        raise MlsynthDataError("SPILLSYNTH: no post-treatment periods.")

    Y = wide.to_numpy(dtype=float).T                # (N, T)
    N = Y.shape[0]
    T = Y.shape[1]
    T0 = int(pre_mask.sum())
    T1 = T - T0
    p = len(affected)
    Y_pre = Y[:, :T0]
    Y_post = Y[:, T0:]

    if T0 < 2:
        raise MlsynthDataError(
            f"SPILLSYNTH: need T0 >= 2 pre-periods to estimate SCM weights "
            f"(got T0={T0})."
        )

    if spillover_structure == "per_unit":
        A = build_A_per_unit(N, p)
    elif spillover_structure == "homogeneous":
        A = build_A_homogeneous(N, p)
    elif spillover_structure == "distance_decay":
        control_labels_in_order = units[1:]
        decay = _decay_weights_for_panel(
            unit_distances, treated_label, control_labels_in_order,
        )
        A = build_A_distance_decay(decay)

    return SpillSynthInputs(
        Y=Y, Y_pre=Y_pre, Y_post=Y_post, A=A,
        treated_label=treated_label,
        affected_labels=affected,
        clean_labels=clean,
        time_labels=time_labels,
        pre_time=time_labels[:T0],
        post_time=time_labels[T0:],
        N=N, T=T, T0=T0, T1=T1, p=p,
        spillover_structure=spillover_structure,
    )
