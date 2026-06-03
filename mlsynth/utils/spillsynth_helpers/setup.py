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

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import SpillSynthInputs


def build_A_per_unit(N: int, p: int, n_treated: int = 1) -> np.ndarray:
    """Cao-Dowd v3 Example 1 -- per-unit free spillover coefficients.

    Returns an ``(N, n_treated + p)`` matrix:

    * columns ``0 .. n_treated - 1`` are basis vectors for the treated
      units in rows ``0 .. n_treated - 1`` (each gets its own treatment-
      effect coefficient);
    * columns ``n_treated .. n_treated + p - 1`` are basis vectors for
      the affected control units in rows ``n_treated .. n_treated +
      p - 1`` (each gets its own free spillover);
    * remaining rows (clean controls) are zero.

    Section S.1.2 of Cao-Dowd v3: with ``n_treated > 1`` and the same
    intervention time across all treated, ``gamma_hat`` partitions into
    per-treated-unit treatment effects and per-affected-unit spillover
    effects.
    """
    if p < 0 or n_treated < 1 or N < n_treated + p:
        raise MlsynthDataError(
            f"SPILLSYNTH: cannot build A with N={N}, n_treated={n_treated}, p={p}."
        )
    k = n_treated + p
    A = np.zeros((N, k))
    for i in range(k):
        A[i, i] = 1.0
    return A


def build_A_homogeneous(N: int, p: int, n_treated: int = 1) -> np.ndarray:
    """Cao-Dowd v3 Example 2 -- shared spillover coefficient ``b``.

    Returns an ``(N, n_treated + 1)`` matrix. The first ``n_treated``
    columns are basis vectors for the treated units (each gets its own
    treatment-effect coefficient). The last column is the indicator
    over the affected-unit rows (so all affected units share a single
    spillover ``b``). Clean control rows are zero everywhere.
    """
    if p < 1 or n_treated < 1 or N < n_treated + p:
        raise MlsynthDataError(
            f"SPILLSYNTH: Example 2 (homogeneous) needs p >= 1 affected "
            f"units and N >= n_treated + p; got N={N}, n_treated={n_treated}, p={p}."
        )
    A = np.zeros((N, n_treated + 1))
    for i in range(n_treated):
        A[i, i] = 1.0
    A[n_treated:n_treated + p, n_treated] = 1.0
    return A


def build_A_distance_decay(
    decay_weights: np.ndarray, n_treated: int = 1,
) -> np.ndarray:
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
    if decay_weights.ndim != 1:                              # pragma: no cover
        # Unreachable: .ravel() above always yields a 1-D array. Kept as a
        # defensive assertion in case the normalisation above ever changes.
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
    n_treated = int(n_treated)
    if n_treated < 1:
        raise MlsynthDataError(
            f"SPILLSYNTH: n_treated must be >= 1; got {n_treated}."
        )
    N = n_treated + decay_weights.size
    A = np.zeros((N, n_treated + 1))
    for i in range(n_treated):
        A[i, i] = 1.0
    A[n_treated:, n_treated] = decay_weights
    return A


# Legacy alias: scripts and tests committed before this refactor still
# import ``build_A_example3`` for the per-unit (now v3 Example 1) helper.
build_A_example3 = build_A_per_unit


def _decay_weights_for_panel(
    unit_distances: Dict[Any, float],
    treated_labels: Sequence[Any],
    control_labels: Sequence[Any],
) -> np.ndarray:
    """Map a user-provided ``{label: distance}`` dict to a decay-weight vector.

    Controls **not** present in ``unit_distances`` get weight zero
    (interpreted as "infinitely far -- no spillover"). Distances on any
    of the **treated** units are ignored (those rows are hard-coded
    elsewhere).
    """
    if not isinstance(unit_distances, dict):
        raise MlsynthDataError(
            "SPILLSYNTH: unit_distances must be a dict mapping unit label "
            "to a non-negative scalar distance from the treated unit."
        )
    extras = set(unit_distances).difference([*treated_labels, *control_labels])
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
    covariates: Optional[Sequence[str]] = None,
    covariate_windows: Optional[Dict[Any, Any]] = None,
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
    cov_list = list(covariates or [])
    for col in (outcome, treat, unitid, time, *cov_list):
        if col not in df.columns:
            raise MlsynthDataError(
                f"SPILLSYNTH: required column {col!r} not in df.columns."
            )

    if spillover_structure not in SPILLOVER_STRUCTURES:
        raise MlsynthDataError(
            f"SPILLSYNTH: unknown spillover_structure {spillover_structure!r}. "
            f"Choose from {SPILLOVER_STRUCTURES}."
        )

    # Retain covariate columns (in original df) for the predictor block;
    # the outcome reshape below works on a slimmed copy.
    cov_source = df[[unitid, time, *cov_list]].copy() if cov_list else None
    df = df[[unitid, time, outcome, treat]].copy()
    df = df.dropna(subset=[outcome])
    if df.empty:
        raise MlsynthDataError("SPILLSYNTH: panel is empty after dropping NaN outcomes.")

    # Cohort-aware reshape via mlsynth.utils.datautils.dataprep. The
    # returned dict has different keys depending on how many distinct
    # treated units the panel has -- single-treated returns
    # ``treated_unit_name``/``y``/...; multi-treated returns
    # ``cohorts``/``Ywide``/...
    from ..datautils import dataprep
    prepped = dataprep(
        df, unit_id_column_name=unitid,
        time_period_column_name=time,
        outcome_column_name=outcome,
        treatment_indicator_column_name=treat,
    )

    if "cohorts" in prepped:
        cohorts = prepped["cohorts"]
        if len(cohorts) > 1:
            # Cao-Dowd v3 Section S.1.2 requires a common intervention
            # time across treated units. Staggered adoption (multiple
            # cohorts) is a separate extension and is not part of S.1.2.
            raise MlsynthDataError(
                "SPILLSYNTH: multiple treated units must share a common "
                "intervention time (Cao-Dowd v3 Section S.1.2). The "
                f"panel has {len(cohorts)} treatment-start cohorts: "
                f"{sorted(cohorts.keys(), key=str)}. Staggered adoption "
                "with different start times is not yet supported."
            )
        (start_time, cohort), = cohorts.items()
        treated_labels = tuple(cohort["treated_units"])
        T0 = int(cohort["pre_periods"])
        Ywide = prepped["Ywide"]
        time_labels_arr = np.asarray(prepped["time_labels"])
    else:
        treated_labels = (prepped["treated_unit_name"],)
        T0 = int(prepped["pre_periods"])
        Ywide = prepped["Ywide"]
        time_labels_arr = np.asarray(prepped["time_labels"])

    n_treated = len(treated_labels)

    affected = tuple(affected_units or ())
    treated_set = set(treated_labels)
    overlap = treated_set.intersection(affected)
    if overlap:
        raise MlsynthDataError(
            f"SPILLSYNTH: treated units {sorted(overlap, key=str)} cannot also "
            "appear in affected_units."
        )
    if len(set(affected)) != len(affected):
        raise MlsynthDataError("SPILLSYNTH: affected_units contains duplicates.")
    units_in_df = set(Ywide.columns)
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

    # Order: treated first (in the cohort order), affected next,
    # clean controls after (sorted by string for determinism).
    all_controls_sorted = sorted(
        [u for u in units_in_df if u not in treated_set],
        key=lambda x: str(x),
    )
    clean = tuple(u for u in all_controls_sorted if u not in affected)
    units = [*treated_labels, *affected, *clean]

    Ywide_ordered = Ywide.reindex(columns=units).sort_index()
    if Ywide_ordered.isna().any().any():
        bad_cells = int(Ywide_ordered.isna().sum().sum())
        raise MlsynthDataError(
            f"SPILLSYNTH: panel has {bad_cells} missing cells after pivot."
        )

    Y = Ywide_ordered.to_numpy(dtype=float).T            # (N, T)
    N = Y.shape[0]
    T = Y.shape[1]
    T1 = T - T0
    p = len(affected)
    Y_pre = Y[:, :T0]
    Y_post = Y[:, T0:]

    if T0 < 2:
        raise MlsynthDataError(
            f"SPILLSYNTH: need T0 >= 2 pre-periods to estimate SCM weights "
            f"(got T0={T0})."
        )
    if T1 < 1:                                               # pragma: no cover
        # Unreachable via dataprep: a treated unit always has >= 1 treated
        # (post) period, and a panel with zero pre-periods is rejected by
        # dataprep upstream. Kept as a defensive guard for direct callers.
        raise MlsynthDataError("SPILLSYNTH: no post-treatment periods.")

    if spillover_structure == "per_unit":
        A = build_A_per_unit(N, p, n_treated=n_treated)
    elif spillover_structure == "homogeneous":
        A = build_A_homogeneous(N, p, n_treated=n_treated)
    elif spillover_structure == "distance_decay":  # pragma: no branch
        # spillover_structure is validated to be one of the three known
        # values above, so this elif is always True when reached.
        control_labels_in_order = units[n_treated:]
        decay = _decay_weights_for_panel(
            unit_distances, treated_labels, control_labels_in_order,
        )
        A = build_A_distance_decay(decay, n_treated=n_treated)

    # Optional per-unit predictor block: each covariate averaged over its
    # window, in the same row order as Y (treated, affected, clean). By default
    # a covariate is averaged over the whole pre-treatment period; an entry in
    # ``covariate_windows`` overrides this with an inclusive ``(start, end)``
    # range of time labels (Abadie's special-predictor spec). Covariates with
    # scattered missing cells are aggregated with ``nanmean``.
    windows = dict(covariate_windows or {})
    bad = [c for c in windows if c not in cov_list]
    if bad:
        raise MlsynthDataError(
            f"SPILLSYNTH: covariate_windows keys {bad} are not in covariates."
        )
    predictors = None
    predictor_names: Tuple[Any, ...] = ()
    if cov_source is not None:
        cutoff = np.asarray(Ywide_ordered.index)[T0]
        rows = []
        for u in units:
            sub = cov_source[cov_source[unitid] == u]
            st = sub[time].to_numpy()
            vec = []
            for c in cov_list:
                if c in windows:
                    lo, hi = windows[c]
                    mask = (st >= lo) & (st <= hi)
                else:
                    mask = st < cutoff
                vals = sub[c].to_numpy(dtype=float)[mask]
                if vals.size == 0 or np.all(np.isnan(vals)):
                    raise MlsynthDataError(
                        f"SPILLSYNTH: covariate {c!r} has no observations in its "
                        f"window for unit {u!r}."
                    )
                vec.append(float(np.nanmean(vals)))
            rows.append(vec)
        predictors = np.asarray(rows, dtype=float)
        predictor_names = tuple(cov_list)

    return SpillSynthInputs(
        Y=Y, Y_pre=Y_pre, Y_post=Y_post, A=A,
        treated_label=treated_labels[0],
        treated_labels=treated_labels,
        n_treated=n_treated,
        affected_labels=affected,
        clean_labels=clean,
        time_labels=np.asarray(Ywide_ordered.index),
        pre_time=np.asarray(Ywide_ordered.index)[:T0],
        post_time=np.asarray(Ywide_ordered.index)[T0:],
        N=N, T=T, T0=T0, T1=T1, p=p,
        spillover_structure=spillover_structure,
        predictors=predictors, predictor_names=predictor_names,
    )
