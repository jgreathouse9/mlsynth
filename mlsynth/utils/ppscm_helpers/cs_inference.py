"""Callaway-Sant'Anna influence-function inference for PPSCM's ``callaway_santanna`` mode.

PPSCM reaches the Callaway-Sant'Anna estimator exactly once three conventions
are aligned -- uniform donor weights, a ``g-1`` baseline and a never-treated
donor pool (#465). An equal point estimate does not make an equal interval, so
the interval has to be theirs too, and PPSCM's delete-one jackknife is not.

Under those conventions each cell is a two-period, two-group difference in
differences, and Callaway and Sant'Anna (2021, JoE 225(2):200-230, Sec. 4) give
its influence function in closed form::

    ATT(g, t) = mean_{i in G_g}(Y_it - Y_{i,g-1}) - mean_{i in C}(Y_it - Y_{i,g-1})
    phi_i     =  (Delta_i - mu_treated) / n_g       for i in G_g
    phi_i     = -(Delta_i - mu_control) / n_C       for i in C
    se        = sqrt(sum_i phi_i^2)

so a standard error costs one pass over the panel instead of one refit per unit.

What PPSCM aggregates
---------------------

PPSCM's event study averages cohorts by size at each horizon and then averages
horizons::

    theta_h = sum_{k in K_h} pg_k ATT(g_k, g_k + h) / S_h,   S_h = sum_{k in K_h} pg_k
    theta   = mean_h theta_h

with ``pg_k`` the share of the panel in cohort ``k`` and ``K_h`` the cohorts
observed at horizon ``h``. That is Callaway-Sant'Anna's dynamic aggregation
followed by an average over event time, and it coincides with their simple
aggregation when every cohort is the same size and reaches every horizon.

The group shares ``pg`` are themselves estimated, so the aggregate's variance
carries their sampling error through the delta method::

    d theta_h / d pg_j = (ATT(g_j, g_j + h) - theta_h) / S_h

giving the weight-influence term ``did::aggte`` calls ``wif``::

    psi_h,i = sum_{k in K_h} w_k,h phi_i^(k,h)
            + sum_{k in K_h} (1{G_i = g_k} - pg_k) (ATT(g_k, g_k+h) - theta_h) / (S_h n)
    psi_i   = mean_h psi_h,i
    se      = sqrt(sum_i psi_i^2)

Omitting the second term leaves a standard error that is finite, plausible and
too small, which is why the vendored reference
(``benchmarks/reference/ppscm_cs/reference.py``) pins it and a test compares
against a version with the term deleted.

Bands
-----

The pointwise band is ``theta_h +- z_{alpha/2} se_h``. The uniform band shares
one critical value across horizons, tabulated by Mammen multiplier bootstrap on
the influence functions::

    B_b,h = sum_i v_b,i psi_h,i,        c = quantile_{1-alpha} max_h |B_b,h| / se_h

No refit is involved: the multipliers act on the influence functions the point
estimate already produced. Reading an event-study path ("positive by horizon
three and never back") is a statement about every horizon at once, which the
pointwise band does not cover at the stated level and the uniform band does.

Restrictions
------------

The influence function above is the one for uniform weights against a
never-treated comparison group with a ``g-1`` baseline. Solved SCM weights are
estimated too and contribute their own term; a not-yet-treated pool changes the
comparison group's composition over time. Outside the three conventions this
mode is wrong, not approximate, so the configuration refuses it
(:class:`mlsynth.config_models.PPSCMConfig`) and this module raises on the data
conditions the configuration cannot see.

No refit happens here, which is why #467 does not reach this path: the
influence functions are read off the fit that produced the point estimate,
so there is no second estimator to disagree with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthDataError

#: Mammen (1993) two-point multipliers: mean 0, variance 1, third moment 1 --
#: the same distribution ``did::mboot`` and PPSCM's wild bootstrap draw from.
_KAPPA = (np.sqrt(5.0) + 1.0) / 2.0
_PROB = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))


@dataclass(frozen=True)
class CSInfluenceResult:
    """Everything one pass over the panel produces.

    Attributes
    ----------
    se : float
        Standard error of the aggregate ATT, including the weight-influence term.
    ci : tuple of float
        Wald interval around the estimator's own point estimate at ``1 - alpha``.
    per_time_se, per_time_ci : numpy.ndarray
        Per-horizon standard errors, shape ``(H,)``, and the pointwise band,
        shape ``(H, 2)``. ``NaN`` at horizons no cohort reaches.
    group_time_att, group_time_se : dict
        ``ATT(g, t)`` and its standard error, keyed by the public
        ``(adoption time, time)`` labels -- the cells the aggregate is built from.
    pointwise_band, uniform_band : numpy.ndarray or None
        Both ``(H, 2)``. ``pointwise_band`` repeats ``per_time_ci``; placing them
        on one object is what makes the pair comparable. ``uniform_band`` is
        ``None`` unless a simultaneous band was asked for.
    critical_value : float
        The multiplier behind ``uniform_band``; the normal quantile when no
        simultaneous band was tabulated.
    influence : numpy.ndarray
        Per-unit, per-horizon influence functions, shape ``(n, H)``. The object
        every quantity above is a functional of.
    replicate_paths : numpy.ndarray or None
        Multiplier-bootstrap draws of the per-horizon path, shape
        ``(n_boot, H)``, on the estimator's scale. ``None`` when no draws were
        taken.
    """

    se: float
    ci: Tuple[float, float]
    per_time_se: np.ndarray
    per_time_ci: np.ndarray
    group_time_att: Dict[Tuple[Any, Any], float]
    group_time_se: Dict[Tuple[Any, Any], float]
    pointwise_band: np.ndarray
    uniform_band: Optional[np.ndarray]
    critical_value: float
    influence: np.ndarray
    replicate_paths: Optional[np.ndarray]


def _cohort_structure(fit: Dict[str, Any]):
    """Collapse the fit's groups to Callaway-Sant'Anna cohorts.

    PPSCM groups by treated unit unless ``time_cohort`` is set, while a
    Callaway-Sant'Anna cell is indexed by adoption time. Units sharing an
    adoption time are one cell either way, so the regrouping here is what makes
    the two indexings the same object; donor eligibility depends only on the
    adoption time, so the pools merge without ambiguity.
    """
    adopt_of, members, donors = fit["adopt_of"], fit["members"], fit["donors"]
    units: Dict[int, list] = {}
    pools: Dict[int, np.ndarray] = {}
    for g in fit["groups"]:
        tj = int(adopt_of[g])
        units.setdefault(tj, []).extend(int(i) for i in members[g])
        pools.setdefault(tj, np.asarray(donors[g], dtype=int))
    cohorts = sorted(units)
    return (cohorts,
            [np.asarray(sorted(units[tj]), dtype=int) for tj in cohorts],
            [pools[tj] for tj in cohorts])


def _cell(resid: np.ndarray, mem: np.ndarray, don: np.ndarray, col: int):
    """One cell's estimate and its two influence-function blocks.

    ``resid`` is the fixed-effect-removed matrix for this cohort, so with the
    ``pre_treatment`` baseline ``resid[i, col]`` is ``Y_i,col - Y_i,g-1`` up to a
    time effect common to every unit. That common term drops out of both the
    difference in means and the deviations below, so the cell computed here is
    the raw-outcome difference in differences.
    """
    d_tr = resid[mem, col]
    d_co = resid[don, col]
    mu_t = float(d_tr.mean())
    mu_c = float(d_co.mean())
    return (mu_t - mu_c,
            (d_tr - mu_t) / mem.size,
            -(d_co - mu_c) / don.size)


def influence_function_inference(
    fit: Dict[str, Any], time_labels: np.ndarray, *,
    alpha: float, att_full: float, per_time_full: np.ndarray,
    n_boot: int = 1000, seed: int = 0, cband: bool = False,
    want_paths: bool = False,
) -> CSInfluenceResult:
    """Analytical Callaway-Sant'Anna standard errors for a fitted PPSCM CS mode.

    Parameters
    ----------
    fit : dict
        The dictionary :func:`mlsynth.utils.ppscm_helpers.engine.run_multisynth`
        returns, fitted under the Callaway-Sant'Anna conventions.
    time_labels : numpy.ndarray
        The panel's sorted time labels; the cell dictionaries are keyed by these
        and not by column positions, so a caller reads ``(2014, 2017)`` and not
        ``(9, 12)``.
    alpha : float
        Miscoverage level. Keyword-only.
    att_full, per_time_full : float and numpy.ndarray
        The estimator's own point estimates. The bands are centred on these, so
        the interval belongs to the number the estimator reports.
    n_boot, seed : int
        Multiplier-bootstrap draws and their seed; used only when a simultaneous
        band or replicate paths are asked for.
    cband : bool
        Tabulate the simultaneous critical value and return ``uniform_band``.
    want_paths : bool
        Return the bootstrap draws of the per-horizon path, which a cumulative
        band consumes.

    Returns
    -------
    CSInfluenceResult

    Raises
    ------
    MlsynthDataError
        If a cohort has no admissible donor, or if no cell is estimable.
    """
    res = fit["res"]
    n = int(fit["n"])
    H = int(fit["n_leads"])
    cohorts, members, donors = _cohort_structure(fit)
    K = len(cohorts)

    empty = [time_labels[tj] for tj, don in zip(cohorts, donors) if don.size == 0]
    if empty:
        raise MlsynthDataError(
            "Callaway-Sant'Anna inference needs a non-empty comparison group for "
            f"every cohort; cohort(s) {empty} have no admissible donor. With "
            "donor_pool='never_treated' this means the panel has no "
            "never-treated unit.")

    cohort_of = np.full(n, -1, dtype=int)
    for k, mem in enumerate(members):
        cohort_of[mem] = k
    n1 = np.array([mem.size for mem in members], dtype=float)
    pg = n1 / float(n)

    # ---- cells: ATT(g, t) and its standard error -------------------------- #
    att_cell = np.full((K, H), np.nan)
    se_cell = np.full((K, H), np.nan)
    group_time_att: Dict[Tuple[Any, Any], float] = {}
    group_time_se: Dict[Tuple[Any, Any], float] = {}
    for k, tj in enumerate(cohorts):
        resid = res[tj]
        T = resid.shape[1]
        for h in range(H):
            col = tj + h
            if col >= T:
                continue
            att, inf_t, inf_c = _cell(resid, members[k], donors[k], col)
            att_cell[k, h] = att
            se_cell[k, h] = float(np.sqrt((inf_t ** 2).sum() + (inf_c ** 2).sum()))
            key = (time_labels[tj], time_labels[col])
            group_time_att[key] = att
            group_time_se[key] = se_cell[k, h]
    if not group_time_att:
        raise MlsynthDataError(
            "Callaway-Sant'Anna inference found no estimable (g, t) cell: every "
            "cohort adopts at or after the last observed period.")

    # ---- per-horizon influence functions ---------------------------------- #
    influence = np.full((n, H), np.nan)
    theta = np.full(H, np.nan)
    for h in range(H):
        keep = np.flatnonzero(np.isfinite(att_cell[:, h]))
        if keep.size == 0:
            continue
        S = float(pg[keep].sum())
        w = pg[keep] / S
        a = att_cell[keep, h]
        theta[h] = float(w @ a)
        psi = np.zeros(n)
        for j, k in enumerate(keep):
            _, inf_t, inf_c = _cell(res[cohorts[k]], members[k], donors[k],
                                    cohorts[k] + h)
            psi[members[k]] += w[j] * inf_t
            psi[donors[k]] += w[j] * inf_c
        # The group shares are estimated, so their sampling error enters too.
        indicator = (cohort_of[:, None] == keep[None, :]).astype(float)
        psi += ((indicator - pg[keep][None, :]) @ (a - theta[h])) / (S * n)
        influence[:, h] = psi

    finite = np.isfinite(theta)
    per_time_se = np.where(
        finite, np.sqrt((np.nan_to_num(influence) ** 2).sum(axis=0)), np.nan)
    psi_agg = np.nanmean(influence[:, finite], axis=1)
    se = float(np.sqrt((psi_agg ** 2).sum()))

    z = float(norm.ppf(1.0 - alpha / 2.0))
    per_time_full = np.asarray(per_time_full, dtype=float)
    pointwise = np.column_stack([per_time_full - z * per_time_se,
                                 per_time_full + z * per_time_se])
    ci = (float(att_full) - z * se, float(att_full) + z * se)

    # ---- multiplier bootstrap: simultaneous band and replicate paths ------ #
    critical_value = z
    uniform: Optional[np.ndarray] = None
    paths: Optional[np.ndarray] = None
    if cband or want_paths:
        rng = np.random.default_rng(seed)
        V = np.where(rng.random((int(n_boot), n)) < _PROB, 1.0 - _KAPPA, _KAPPA)
        draws = V @ np.nan_to_num(influence)
        paths = per_time_full[None, :] + draws
        if cband:
            scaled = np.abs(draws[:, finite]) / per_time_se[finite][None, :]
            critical_value = float(np.quantile(np.max(scaled, axis=1), 1.0 - alpha))
            uniform = np.column_stack([
                per_time_full - critical_value * per_time_se,
                per_time_full + critical_value * per_time_se])

    return CSInfluenceResult(
        se=se, ci=ci, per_time_se=per_time_se, per_time_ci=pointwise,
        group_time_att=group_time_att, group_time_se=group_time_se,
        pointwise_band=pointwise, uniform_band=uniform,
        critical_value=critical_value, influence=influence,
        replicate_paths=paths,
    )
