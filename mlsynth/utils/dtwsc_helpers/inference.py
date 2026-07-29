"""Placebo inference for DTWSC, as in Cao & Chadefaux (2025).

The paper's 95 percent band is a placebo distribution, not a sampling
distribution. Every untreated unit is re-analysed as if it had been treated,
with the real treated unit removed from every placebo donor pool; the pool is
then perturbed by dropping one or two further units. The band is the pointwise
quantiles of the resulting gap paths, drawn separately for the warped and
unwarped fits, and the paper's claim is that warping narrows it.

Alongside the band the authors run a paired efficiency test on
:math:`\\log(\\mathrm{MSE}_{\\mathrm{DSC}} / \\mathrm{MSE}_{\\mathrm{SC}})`
across those runs. The runs share units and donor pools, so they are not
independent; the degrees of freedom are deflated by a variance-inflation factor
built from the intra-class correlation of a two-way ANOVA over dataset and
unit. That test asks whether warping helps, which is a different question from
whether the treatment had an effect.

Source: ``code/utility/{Figure_5_1,implement}.R`` of the authors' Political
Analysis replication package (Dataverse / Code Ocean capsule
``31ffbd35-ec68-4f2b-8f01-b4de25adcaa8``). Clean-room re-implementation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError


def placebo_donor_pools(
    n_units: int,
    treated: int,
    n_pairs: int = 100,
) -> List[Tuple[int, ...]]:
    """Build the paper's ``data.list`` -- the perturbed donor pools.

    Returns the untouched pool first, then one pool per dropped donor, then
    pools with two donors dropped, in R ``combn`` (lexicographic) order and
    truncated to ``n_pairs``. The treated unit is absent from all of them:
    every placebo run is fit on donors only, so the real treated unit can never
    leak into a placebo counterfactual.

    Parameters
    ----------
    n_units : int
        Total units in the panel, treated included.
    treated : int
        Index of the treated unit.
    n_pairs : int, default 100
        How many two-drop pools to keep. The paper uses 100.

    Notes
    -----
    Pools that would leave fewer than two donors are skipped -- a synthetic
    control needs something to interpolate between.
    """
    donors = [i for i in range(n_units) if i != treated]
    if len(donors) < 2:
        raise MlsynthEstimationError(
            "DTWSC: placebo inference needs at least 3 units (one treated and "
            f"two donors); the panel has {n_units}."
        )
    pools: List[Tuple[int, ...]] = [tuple(donors)]
    for d in donors:
        rest = tuple(u for u in donors if u != d)
        if len(rest) >= 2:
            pools.append(rest)
    if n_pairs > 0:
        for pair in list(combinations(donors, 2))[:n_pairs]:
            rest = tuple(u for u in donors if u not in pair)
            if len(rest) >= 2:
                pools.append(rest)
    return pools


def icc_corrected_ttest(
    log_ratios: Sequence[float],
    dataset_ids: Sequence[Any],
    unit_ids: Sequence[Any],
) -> Dict[str, float]:
    """The paper's efficiency test on the log MSE ratio.

    Reproduces ``Figure_5_*.R``: a two-way ANOVA of the log ratio on dataset
    and unit supplies the mean squares, from which

    .. math::

       \\mathrm{ICC} = \\frac{\\mathrm{BMS} - \\mathrm{EMS}}
         {\\mathrm{BMS} + (k-1)\\mathrm{EMS} + k(\\mathrm{JMS}-\\mathrm{EMS})/n},
       \\qquad \\mathrm{VIF} = 1 + (k-1)\\,\\mathrm{ICC},

    with ``n`` datasets and ``k`` units. The effective degrees of freedom are
    ``N / VIF``, and the p-value is the two-sided lower-tail probability of the
    ordinary one-sample t statistic under that reduced ``df``. A negative
    ``t`` means the warped fit achieves the lower post-treatment MSE.

    Returns
    -------
    dict
        ``t``, ``p``, ``df``, ``icc``, ``vif``, ``mean_log_ratio``,
        ``mse_reduction`` (the implied proportional MSE saving) and ``n_runs``.
    """
    from scipy import stats

    y = np.asarray(log_ratios, dtype=float)
    d = np.asarray(dataset_ids)
    u = np.asarray(unit_ids)
    ok = np.isfinite(y)
    y, d, u = y[ok], d[ok], u[ok]
    if y.size < 2:
        raise MlsynthEstimationError(
            "DTWSC: the efficiency test needs at least two placebo runs with a "
            f"finite log MSE ratio; got {y.size}."
        )

    bms, jms, ems, n_d, n_u = _two_way_mean_squares(y, d, u)
    if not np.isfinite(ems) or ems <= 0 or n_u < 2:
        icc, vif = 0.0, 1.0
    else:
        denom = bms + (n_u - 1) * ems + n_u * (jms - ems) / max(n_d, 1)
        icc = float((bms - ems) / denom) if denom != 0 else 0.0
        vif = float(1.0 + (n_u - 1) * icc)
    if not np.isfinite(vif) or vif <= 0:  # pragma: no cover - a non-positive
        # VIF needs an ICC below -1/(k-1), which the mean-square identities
        # exclude for the two-way layouts this is called on; kept so a
        # degenerate design cannot produce a negative degrees of freedom.
        vif = 1.0
    df = float(y.size / vif)

    t = float(stats.ttest_1samp(y, 0.0).statistic)
    p = float(stats.t.cdf(t, df=df) * 2.0)
    mean_log_ratio = float(y.mean())
    return {
        "t": t,
        "p": float(min(max(p, 0.0), 1.0)),
        "df": df,
        "icc": float(icc),
        "vif": vif,
        "mean_log_ratio": mean_log_ratio,
        "mse_reduction": float(1.0 - np.exp(mean_log_ratio)),
        "n_runs": int(y.size),
    }


def _two_way_mean_squares(y, factor_a, factor_b):
    """Mean squares of ``y ~ A * B``, in R ``aov`` term order (A, B, A:B).

    Written out rather than delegated so the correspondence with the R code is
    inspectable: R's sequential (Type I) sums of squares for a balanced or
    near-balanced design reduce to the group-mean decomposition below.
    """
    y = np.asarray(y, dtype=float)
    a_levels = np.unique(factor_a)
    b_levels = np.unique(factor_b)
    n_a, n_b = a_levels.size, b_levels.size
    grand = y.mean()

    ss_a = sum(((y[factor_a == a].mean() - grand) ** 2) * (factor_a == a).sum()
               for a in a_levels)
    ss_b = sum(((y[factor_b == b].mean() - grand) ** 2) * (factor_b == b).sum()
               for b in b_levels)
    ss_total = float(((y - grand) ** 2).sum())
    ss_resid = max(ss_total - ss_a - ss_b, 0.0)

    df_a, df_b = n_a - 1, n_b - 1
    df_resid = y.size - 1 - df_a - df_b
    bms = ss_a / df_a if df_a > 0 else np.nan
    jms = ss_b / df_b if df_b > 0 else np.nan
    ems = ss_resid / df_resid if df_resid > 0 else np.nan
    return bms, jms, ems, n_a, n_b


def placebo_band(gaps: np.ndarray, alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Pointwise ``alpha/2`` and ``1 - alpha/2`` quantiles of the placebo gaps.

    ``gaps`` is ``(n_runs, T)``; ``NaN`` entries are ignored per period, so a
    run whose warp ended short still contributes everywhere it is defined.
    """
    g = np.asarray(gaps, dtype=float)
    if g.ndim != 2 or g.shape[0] == 0:
        raise MlsynthEstimationError("DTWSC: no placebo runs to build a band from.")
    with np.errstate(invalid="ignore"):
        lo = np.nanquantile(g, alpha / 2.0, axis=0)
        hi = np.nanquantile(g, 1.0 - alpha / 2.0, axis=0)
    return {"lower": lo, "upper": hi}
