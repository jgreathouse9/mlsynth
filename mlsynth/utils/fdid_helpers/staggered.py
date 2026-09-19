r"""Staggered-adoption Forward Difference-in-Differences.

Li (2023) Web Appendix C extends Forward DID to several treated units: run the
method separately per unit and summarise the results. This module implements
that, plus the three things the extension needs before the summary means
anything.

Donor eligibility
-----------------
The donor pool is the never-treated units, and it is applied at selection as
well as at estimation. Selection reads only ``t < g_i``, so a donor adopting
inside a treated unit's pre-window corrupts the criterion that chose it, not
just the counterfactual it ends up in. ``dataprep``'s cohort mode already
excludes every eventually-treated unit, so one pool serves all cohorts.

How donors are selected
-----------------------
Li's appendix selects per treated unit. That leans hard on each unit's own
pre-window, and Li's Proposition 2.2 needs :math:`T_{1i} \to \infty` with
:math:`\log N / T_{1i} \to 0` for the selection to be consistent -- the
condition staggered adoption strains, because the earliest cohort is handed the
shortest pre-window. Selecting on the cohort mean instead cuts the criterion's
idiosyncratic noise by :math:`1/N_g` at the cost of a cohort-level
parallel-trends requirement, and ``selection="partial"`` interpolates between
the two criteria. This is the Forward DID reading of partially pooled synthetic
control (Ben-Michael, Feller and Rothstein 2022).

Aggregation and inference
-------------------------
The building block is :math:`ATT(g, h)`, aggregated on a balanced event clock
with cohort-size weights (Callaway and Sant'Anna 2021, equation 31). Every
aggregate here is a linear functional of the unobserved parallel-trends
residuals :math:`v_{it}`, so writing unit :math:`i`'s functional as a vector
:math:`c_i` over calendar time, the variance of any weighted aggregate is

.. math::

   \sum_{i,j} \sum_{t,s} c_i(t)\, c_j(s)\, \Omega_{ij}(t - s),

with :math:`\Omega` estimated on the common clean pre-window. Dropping the
cross-unit terms -- combining Li's per-unit standard errors as if independent
-- is anti-conservative, and worsens as treated units accumulate, because the
shared-donor component does not average away.

The overall effect averages the per-horizon estimates, so a covariance that
prices no autocovariance divides its variance by the horizon count as though
the horizons were independent draws. That is why this path prices
autocovariances by default, unlike the single-treated one: with an AR(1)
residual of coefficient 0.8, nominal 95% coverage of the overall ATT was 0.62
under the analytic form and 0.81 under the HAC one, while an uncorrelated
residual gave 0.93 and 0.94. The per-horizon intervals are far less exposed
(0.91 and 0.92 in the same designs) because they average nothing. Truncation
length barely matters over a short pre-window -- 0.81, 0.79, 0.80, 0.81 at
lags 4, 8, 12 and 20 -- since the downward bias in a demeaned stretch's
autocovariances grows about as fast as the signal being added.

References
----------
Li, K. T. (2023). Frontiers: A Simple Forward Difference-in-Differences Method.
Marketing Science, 43(2), 267-279.

Callaway, B., and Sant'Anna, P. H. C. (2021). Difference-in-Differences with
multiple time periods. Journal of Econometrics, 225(2), 200-230.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthEstimationError
from .estimation import forward_selection_path
from .inference import hac_lag
from .structures import (
    FDIDEventStudy,
    FDIDStaggeredInputs,
    FDIDStaggeredResults,
    FDIDUnitFit,
)

#: Smallest pre-window a unit may be fitted on, after any anticipation trim.
MIN_PRE_PERIODS = 2


# ─── covariance ───────────────────────────────────────────────────────────

def residual_cross_covariances(
    residuals: np.ndarray, lag: int = 0
) -> np.ndarray:
    r"""Cross-unit autocovariance matrices :math:`\Gamma_0, \dots, \Gamma_L`.

    Entry ``(i, j)`` of :math:`\Gamma_k` estimates
    :math:`\operatorname{Cov}(v_{it}, v_{j,t-k})` on the supplied stretch.

    Parameters
    ----------
    residuals : np.ndarray
        Parallel-trends residuals over a common clean window, shape
        ``(n_obs, n_units)``.
    lag : int, default 0
        Highest lag to return. Clamped to ``n_obs - 1``.

    Returns
    -------
    np.ndarray
        Shape ``(lag + 1, n_units, n_units)``. The divisor is ``n_obs``,
        matching :func:`~mlsynth.utils.fdid_helpers.inference.residual_autocovariances`,
        which keeps the implied spectral density non-negative.

    Raises
    ------
    ValueError
        If ``residuals`` is empty or ``lag`` is negative.
    """
    V = np.asarray(residuals, dtype=float)
    if V.ndim != 2 or V.size == 0:
        raise ValueError("residuals must be a non-empty (n_obs, n_units) array.")
    if lag < 0:
        raise ValueError(f"lag must be non-negative; got {lag}.")
    n = V.shape[0]
    V = V - V.mean(axis=0)
    top = min(int(lag), n - 1)
    return np.array([V[k:].T @ V[: n - k] / n for k in range(top + 1)])


def aggregate_variance(
    residuals: np.ndarray, contrasts: np.ndarray, lag: int = 0
) -> float:
    r"""Variance of :math:`\sum_i \sum_t C_{it} v_{it}`.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals over the common clean pre-window, shape
        ``(n_obs, n_units)``. Demeaned internally.
    contrasts : np.ndarray
        Coefficient matrix ``C``, shape ``(n_units, T)``. Row ``i`` holds the
        calendar-time weights unit ``i`` contributes to the aggregate,
        including the negative pre-window terms that carry the sampling error
        in its intercept.
    lag : int, default 0
        Highest autocovariance lag to price in. Zero uses only the
        contemporaneous cross-unit covariance.

    Returns
    -------
    float
        The variance, floored at its ``lag = 0`` value. Sample autocovariances
        of a demeaned stretch sum to about :math:`-\gamma_0` over a block, so
        an unguarded lag sum can drive the estimate below the contemporaneous
        value or negative; neither is usable. This is the multivariate form of
        the guard in
        :func:`~mlsynth.utils.fdid_helpers.inference.block_mean_variance`.

    Raises
    ------
    ValueError
        If ``contrasts`` does not have one row per residual column.
    """
    C = np.asarray(contrasts, dtype=float)
    V = np.asarray(residuals, dtype=float)
    if C.ndim != 2 or C.shape[0] != V.shape[1]:
        raise ValueError(
            f"contrasts must be (n_units, T) with n_units = {V.shape[1]}; "
            f"got {C.shape}."
        )

    G = residual_cross_covariances(V, lag)
    var0 = float(np.einsum("it,jt,ij->", C, C, G[0]))
    if lag <= 0 or len(G) == 1:
        return max(var0, 0.0)

    # Split the double sum by the lag k = t - s. The two shifted slices must
    # share one time label, so that einsum pairs period t with period t - k
    # instead of summing the two of them independently: every contrast row
    # sums to zero, so an unpaired form returns nothing at any lag.
    total = var0
    for k in range(1, len(G)):
        total += float(np.einsum("it,jt,ij->", C[:, k:], C[:, :-k], G[k]))
        total += float(np.einsum("it,jt,ji->", C[:, :-k], C[:, k:], G[k]))
    return max(total, var0, 0.0)


# ─── aggregation weights and contrasts ────────────────────────────────────

def event_time_weights(
    adoption_index: np.ndarray, horizon: int, T: int
) -> np.ndarray:
    """Cohort-size weights over the units observed at one event time.

    Equal weight per treated unit among those whose ``g + h`` falls inside the
    panel, which is Callaway and Sant'Anna's cohort-size share when each unit
    counts once.

    Parameters
    ----------
    adoption_index : np.ndarray
        Each treated unit's adoption position.
    horizon : int
        Event time ``h``.
    T : int
        Number of periods.

    Returns
    -------
    np.ndarray
        Weights summing to one, zero for units not observed at ``h``. All
        zeros when no unit is observed.
    """
    g = np.asarray(adoption_index, dtype=int)
    seen = ((g + horizon) < T) & ((g + horizon) >= 0)
    w = seen.astype(float)
    total = w.sum()
    return w / total if total else w


def _contrast_matrix(
    adoption_index: np.ndarray,
    pre_windows: Sequence[int],
    horizons: Sequence[int],
    T: int,
    horizon: Optional[int] = None,
) -> np.ndarray:
    """Coefficient matrix for one horizon, or for the average over horizons.

    Each unit contributes ``+w`` at its own post-treatment period(s) and
    ``-w / T1`` across the pre-window its intercept was taken over, because the
    intercept's sampling error enters every estimate that subtracts it.
    """
    g = np.asarray(adoption_index, dtype=int)
    n = g.size
    C = np.zeros((n, T), dtype=float)
    if horizon is not None:
        w = event_time_weights(g, horizon, T)
        for i in range(n):
            if w[i] == 0.0:
                continue
            C[i, g[i] + horizon] += w[i]
            C[i, : pre_windows[i]] -= w[i] / pre_windows[i]
        return C

    for h in horizons:
        C += _contrast_matrix(g, pre_windows, horizons, T, horizon=h)
    return C / len(horizons)


# ─── per-unit fitting ─────────────────────────────────────────────────────

def _resolve_horizons(
    adoption_index: np.ndarray, T: int, max_horizon: Optional[int]
) -> np.ndarray:
    """Horizons every treated unit supports, so the event study is balanced."""
    balanced = int(np.min(T - 1 - np.asarray(adoption_index, dtype=int)))
    if balanced < 0:  # pragma: no cover - dataprep records an adoption only
        # where the treatment column turns on inside the panel, so g <= T - 1
        # always holds; this guards a caller that builds inputs by hand.
        raise MlsynthEstimationError(
            "At least one treated unit adopts after the panel ends; no "
            "post-treatment horizon is observed for it."
        )
    if max_horizon is None:
        return np.arange(balanced + 1)
    if max_horizon > balanced:
        raise MlsynthEstimationError(
            f"max_horizon={max_horizon} exceeds the largest horizon every "
            f"cohort supports ({balanced}); the event study would be "
            "unbalanced. Lower max_horizon, or drop the late cohorts."
        )
    return np.arange(int(max_horizon) + 1)


def _fit_one_unit(
    y: np.ndarray,
    donors: np.ndarray,
    donor_names: Sequence[Any],
    pre: int,
    selected: Sequence[int],
    path: Optional[np.ndarray],
    horizons: Sequence[int],
    g: int,
    unit_name: Any,
    time_label: Any,
) -> FDIDUnitFit:
    """Difference-in-differences fit for one treated unit and donor subset."""
    idx = list(selected)
    mean_donor = donors[:, idx].mean(axis=1)
    intercept = float((y - mean_donor)[:pre].mean())
    counterfactual = mean_donor + intercept
    gap = y - counterfactual

    resid_pre = gap[:pre]
    rmse = float(np.sqrt(np.mean(resid_pre ** 2)))
    ss_tot = float(np.sum((y[:pre] - y[:pre].mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid_pre ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan

    att_h = {int(h): float(gap[g + h]) for h in horizons}
    names = [donor_names[i] for i in idx]
    return FDIDUnitFit(
        unit_name=unit_name,
        adoption_index=int(g),
        adoption_time=time_label,
        selected_indices=idx,
        selected_names=names,
        donor_weights={nm: 1.0 / len(names) for nm in names},
        intercept=intercept,
        observed=np.asarray(y, dtype=float),
        counterfactual=counterfactual,
        gap=gap,
        pre_periods=int(pre),
        att=float(np.mean(list(att_h.values()))) if att_h else float("nan"),
        att_by_horizon=att_h,
        pre_rmse=rmse,
        r_squared=r2,
        selection_path=path,
    )


def _select_for_units(
    inputs: FDIDStaggeredInputs,
    pre_windows: np.ndarray,
    selection: str,
    pooling_weight: float,
) -> List[Tuple[List[int], np.ndarray]]:
    """Run forward selection for every treated unit under the chosen mode.

    Units adopting at the same time form a cohort and share a pre-window, so
    the pooled criterion is computed once per cohort.
    """
    g = inputs.adoption_index
    out: List[Optional[Tuple[List[int], np.ndarray]]] = [None] * inputs.n_treated

    cohorts: Dict[int, List[int]] = {}
    for i, gi in enumerate(g):
        cohorts.setdefault(int(gi), []).append(i)

    for gi, members in cohorts.items():
        pre = int(pre_windows[members[0]])
        donors_pre = inputs.donor_matrix[:pre]
        cohort_mean_pre = inputs.treated_matrix[:pre, members].mean(axis=1)

        if selection == "pooled":
            order, path = forward_selection_path(donors_pre, [cohort_mean_pre])
            best = int(np.argmax(path))
            for i in members:
                out[i] = (list(order[: best + 1]), path[: best + 1])
            continue

        for i in members:
            own_pre = inputs.treated_matrix[:pre, i]
            if selection == "partial":
                targets = [own_pre, cohort_mean_pre]
                weights = [1.0 - pooling_weight, pooling_weight]
            else:                                   # "unit"
                targets, weights = [own_pre], [1.0]
            order, path = forward_selection_path(donors_pre, targets, weights)
            best = int(np.argmax(path))
            out[i] = (order[: best + 1], path[: best + 1])

    # Every unit belongs to exactly one cohort, so every slot is filled and the
    # list is returned positionally. Filtering it would silently misalign picks
    # with units if that ever stopped holding.
    return out  # type: ignore[return-value]


# ─── pipeline ─────────────────────────────────────────────────────────────

def fit_staggered(
    inputs: FDIDStaggeredInputs,
    selection: str = "unit",
    pooling_weight: float = 0.5,
    anticipation: int = 0,
    max_horizon: Optional[int] = None,
    inference: str = "analytic",
    lrvar_lag: Optional[int] = None,
    alpha: float = 0.05,
) -> FDIDStaggeredResults:
    """Run staggered Forward DID and assemble the typed result.

    Parameters
    ----------
    inputs : FDIDStaggeredInputs
        Preprocessed panel.
    selection : {"unit", "pooled", "partial"}, default "unit"
        Whose pre-treatment fit forward selection optimises.
    pooling_weight : float, default 0.5
        Weight on the cohort criterion when ``selection="partial"``.
    anticipation : int, default 0
        Pre-periods dropped from the end of each unit's pre-window, for
        treatment effects that begin before the recorded adoption date.
    max_horizon : int, optional
        Highest event time to report. Defaults to the largest horizon every
        cohort supports, which keeps the event study balanced.
    inference : {"analytic", "hac"}, default "analytic"
        ``"analytic"`` prices only the contemporaneous cross-unit covariance.
        ``"hac"`` adds autocovariances, for a residual that is serially
        dependent as well as cross-sectionally so.
    lrvar_lag : int, optional
        Truncation lag under ``inference="hac"``. Defaults to
        :func:`~mlsynth.utils.fdid_helpers.inference.hac_lag` over the common
        pre-window and the reported horizon span. Ignored by the analytic
        path, which prices no autocovariance at any lag.
    alpha : float, default 0.05
        One minus the nominal coverage of the reported intervals.

    Returns
    -------
    FDIDStaggeredResults
        Per-unit fits, the balanced event study, and the overall effect.

    Raises
    ------
    MlsynthEstimationError
        If a unit's pre-window is shorter than two periods after the
        anticipation trim, if no donors are available, or if ``max_horizon``
        exceeds what every cohort supports.
    """
    if inputs.n_donors == 0:
        raise MlsynthEstimationError(
            "No never-treated donor units are available; staggered Forward "
            "DID needs a donor pool that is clean over the whole panel."
        )

    g = np.asarray(inputs.adoption_index, dtype=int)
    pre_windows = g - int(anticipation)
    short = [
        (inputs.treated_names[i], int(pre_windows[i]))
        for i in range(inputs.n_treated)
        if pre_windows[i] < MIN_PRE_PERIODS
    ]
    if short:
        detail = ", ".join(f"{nm} ({n} periods)" for nm, n in short)
        raise MlsynthEstimationError(
            f"Insufficient pre-periods after an anticipation trim of "
            f"{anticipation}: {detail}. At least {MIN_PRE_PERIODS} are needed "
            "per treated unit."
        )

    horizons = _resolve_horizons(g, inputs.T, max_horizon)
    picks = _select_for_units(inputs, pre_windows, selection, pooling_weight)

    # The covariance is estimated on the stretch that is clean for every unit.
    common_pre = int(pre_windows.min())
    if inference == "hac":
        lag = (hac_lag(common_pre, len(horizons)) if lrvar_lag is None
               else int(lrvar_lag))
    else:
        lag = 0

    units = [
        _fit_one_unit(
            y=inputs.treated_matrix[:, i],
            donors=inputs.donor_matrix,
            donor_names=inputs.donor_names,
            pre=int(pre_windows[i]),
            selected=picks[i][0],
            path=picks[i][1] if inputs.verbose else None,
            horizons=horizons,
            g=int(g[i]),
            unit_name=inputs.treated_names[i],
            time_label=inputs.time_labels[int(g[i])],
        )
        for i in range(inputs.n_treated)
    ]

    resid = np.column_stack([u.gap[:common_pre] for u in units])
    z = float(norm.ppf(1.0 - alpha / 2.0))

    att, se, lo, hi, n_seen = [], [], [], [], []
    for h in horizons:
        w = event_time_weights(g, int(h), inputs.T)
        att_h = float(sum(w[i] * units[i].att_by_horizon[int(h)]
                          for i in range(inputs.n_treated)))
        C = _contrast_matrix(g, pre_windows, horizons, inputs.T, horizon=int(h))
        se_h = float(np.sqrt(aggregate_variance(resid, C, lag)))
        att.append(att_h)
        se.append(se_h)
        lo.append(att_h - z * se_h)
        hi.append(att_h + z * se_h)
        n_seen.append(int((w > 0).sum()))

    event_study = FDIDEventStudy(
        horizons=np.asarray(horizons, dtype=int),
        att=np.asarray(att, dtype=float),
        se=np.asarray(se, dtype=float),
        ci_lower=np.asarray(lo, dtype=float),
        ci_upper=np.asarray(hi, dtype=float),
        n_units=np.asarray(n_seen, dtype=int),
    )

    overall_att = float(np.mean(att))
    C_all = _contrast_matrix(g, pre_windows, horizons, inputs.T)
    overall_se = float(np.sqrt(aggregate_variance(resid, C_all, lag)))

    return FDIDStaggeredResults(
        inputs=inputs,
        units=units,
        event_study=event_study,
        overall_att=overall_att,
        overall_se=overall_se,
        overall_ci=(overall_att - z * overall_se, overall_att + z * overall_se),
        selection=selection,
        metadata={
            "anticipation": int(anticipation),
            "pooling_weight": float(pooling_weight) if selection == "partial" else None,
            "common_pre_periods": common_pre,
            "inference": inference,
            "lrvar_lag": lag,
            "n_cohorts": int(np.unique(g).size),
        },
    )
