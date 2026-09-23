"""One number for a domain of related outcomes, and a p-value for it.

A multi-outcome study reports an effect per outcome, and the outcomes are in
different units. Tian, Lee & Panchenko (2026, Online Appendix B.3.2) summarize a
domain with the index of Kling, Liebman & Katz (2007): standardize each
outcome's estimated effect by that outcome's cross-sectional SD, average over
the periods of a window, and average across the outcomes,

.. math::

   \\widehat{\\tau}_i(t_1, t_2) = \\frac{1}{K} \\sum_k
       \\frac{1}{\\#^{t_1,t_2}_k} \\sum_{t=t_1}^{t_2}
       \\frac{\\widehat{\\tau}_{it,k}}{\\sigma_k}.

Its p-value is the permutation test of :mod:`~mlsynth.utils.scmo_helpers.inference`
one level up (B.3.3): each unit's aggregate ratio is its post-treatment loss
summed over the outcomes over its pre-treatment loss summed over them, and the
treated unit's rank in that ranking is the p-value,

.. math::

   r_i = \\frac{\\sum_k R^{\\text{post}}_{i,k} / \\sigma_k}
              {\\sum_k R^{\\text{pre}}_{i,k} / \\sigma_k}.

Both are computed from per-unit gaps, so the placebo units that make the test
possible are the same ones the index is read on. The outcomes of a domain need
not share a period grid: the COVID application of the appendix mixes daily,
weekly, monthly and quarterly series, and a window takes from each outcome the
periods it has.

The guard ``eta`` enters both sides of every ratio, as the appendix's footnote
14 states. Their own script does that for the per-outcome and overall tests and
leaves it off the numerator when aggregating inside a window; against a captured
run of it, mlsynth reproduces all 33 window indices, all three domain indices
and all three overall p-values, and differs on three of the 33 window p-values,
each in a window where the treated unit's aggregate statistic is about zero and
the ranking at the bottom is what the guard reshuffles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
from .inference import _ALTERNATIVES, TWO_SIDED, _directional


@dataclass(frozen=True)
class OutcomeGaps:
    """One outcome's contribution to a domain: its per-unit post-treatment gaps.

    Parameters
    ----------
    name : str
        Outcome label, kept so a result can say what it aggregated.
    periods : np.ndarray
        Length ``T_post`` period labels; windows are matched against these, so
        outcomes observed at different frequencies can share a domain.
    gaps : np.ndarray
        ``(N, T_post)`` signed gaps (observed minus counterfactual), one row per
        unit -- the treated unit and every placebo.
    pre_rmspe : np.ndarray
        ``(N,)`` pre-treatment RMSPE per unit, untruncated.
    sigma : float
        The outcome's scale: the average cross-sectional SD over the
        post-treatment periods. Dividing by it is what lets outcomes in
        different units be averaged.
    """

    name: str
    periods: np.ndarray
    gaps: np.ndarray
    pre_rmspe: np.ndarray
    sigma: float


@dataclass(frozen=True)
class DomainAggregate:
    """The aggregate index over a domain and the permutation test for it."""

    outcomes: Tuple[str, ...]
    tau: float                       # over the whole post-treatment period
    p_value: float
    treated_ratio: float
    ratios: np.ndarray               # (N,) aggregate ratio per unit
    windows: Tuple[Tuple[Any, Any], ...] = ()
    tau_by_window: np.ndarray = None    # type: ignore[assignment]
    p_by_window: np.ndarray = None      # type: ignore[assignment]


def outcome_gaps(name: str, inputs, placebo) -> OutcomeGaps:
    """Read one outcome's contribution off a fitted SCMO result.

    ``inputs`` is the prepared panel and ``placebo`` the
    :class:`~mlsynth.utils.scmo_helpers.structures.PlaceboInference` attached to
    the fit (``inference="placebo"``), which carries the per-unit gaps the
    permutation loop produced.
    """
    if placebo is None or placebo.post_gaps is None:
        raise MlsynthConfigError(
            f"outcome {name!r} has no placebo gaps; fit it with inference='placebo'.")
    T0 = inputs.T0
    periods = np.asarray(inputs.time_index.labels)[T0:]
    sigma = float(np.mean(inputs.Y[:, T0:].std(axis=0, ddof=1)))
    return OutcomeGaps(name=name, periods=periods,
                       gaps=np.asarray(placebo.post_gaps, dtype=float),
                       pre_rmspe=np.asarray(placebo.pre_rmspe, dtype=float),
                       sigma=sigma)


def _check(outcomes: Sequence[OutcomeGaps], eta: float, alternative: str) -> int:
    if not outcomes:
        raise MlsynthConfigError("aggregate_domain needs at least one outcome.")
    if alternative not in _ALTERNATIVES:
        raise MlsynthConfigError(
            f"alternative must be one of {_ALTERNATIVES}; got {alternative!r}.")
    if eta < 0:
        raise MlsynthConfigError(f"eta must be non-negative; got {eta}.")
    n_units = int(outcomes[0].gaps.shape[0])
    for o in outcomes:
        if o.gaps.shape[0] != n_units or o.pre_rmspe.shape[0] != n_units:
            raise MlsynthConfigError(
                f"every outcome must cover the same units; {o.name!r} has "
                f"{o.gaps.shape[0]} against {n_units}.")
        if o.gaps.shape[1] != o.periods.shape[0]:
            raise MlsynthConfigError(
                f"outcome {o.name!r} has {o.gaps.shape[1]} gap columns and "
                f"{o.periods.shape[0]} period labels.")
        if not np.isfinite(o.sigma) or o.sigma <= 0:
            raise MlsynthConfigError(
                f"sigma must be positive and finite; outcome {o.name!r} has {o.sigma}.")
    return n_units


def aggregate_domain(
    outcomes: Sequence[OutcomeGaps], *, treated_idx: int,
    windows: Optional[Sequence[Tuple[Any, Any]]] = None,
    eta: float = 0.0, alternative: str = TWO_SIDED,
) -> DomainAggregate:
    """Aggregate a domain's outcomes into one effect and one p-value.

    Parameters
    ----------
    outcomes : sequence of OutcomeGaps
        The domain's outcomes, each with its per-unit gaps and scale.
    treated_idx : int
        Row index of the treated unit, shared by every outcome.
    windows : sequence of (start, end), optional
        Period windows to report the index and its p-value in, inclusive at both
        ends (the appendix's consecutive windows share their endpoints). An
        outcome with no period inside a window sits that window out; a window no
        outcome covers is an error, since it would report an empty average.
    eta : float, default 0
        Guard added to both sides of the ratio, in standardized units -- the
        paper's :math:`0.01`, which is :math:`0.01\\sigma_k` in an outcome's own
        units.
    alternative : {"two-sided", "greater", "less"}
        Which part of the gap the test statistic keeps, as in
        :func:`~mlsynth.utils.scmo_helpers.inference.permutation_inference`.

    Returns
    -------
    DomainAggregate
    """
    outcomes = list(outcomes)
    n_units = _check(outcomes, eta, alternative)

    standardized = [o.gaps / o.sigma for o in outcomes]
    directional = [_directional(g, alternative) for g in standardized]
    pre = np.column_stack([o.pre_rmspe / o.sigma for o in outcomes])        # (N, K)
    post = np.column_stack([np.sqrt(np.mean(d ** 2, axis=1)) for d in directional])

    tau = float(np.mean([g[treated_idx].mean() for g in standardized]))
    ratios = (post.mean(axis=1) + eta) / (pre.mean(axis=1) + eta)
    p_value = float(np.mean(ratios >= ratios[treated_idx]))

    window_list = tuple(tuple(w) for w in (windows or ()))
    taus = np.empty(len(window_list))
    p_windows = np.empty(len(window_list))
    for j, (lo, hi) in enumerate(window_list):
        masks = [(o.periods >= lo) & (o.periods <= hi) for o in outcomes]
        present = [k for k, m in enumerate(masks) if m.any()]
        if not present:
            raise MlsynthConfigError(
                f"window ({lo}, {hi}) selects no period of any outcome.")
        taus[j] = float(np.mean(
            [standardized[k][treated_idx][masks[k]].mean() for k in present]))
        # The numerator averages each present outcome's mean loss inside the
        # window; the denominator is the domain's pre-treatment loss, over every
        # outcome, as the appendix's ratio has it.
        num = np.mean(np.column_stack(
            [directional[k][:, masks[k]].mean(axis=1) for k in present]), axis=1)
        r = (num + eta) / (pre.mean(axis=1) + eta)
        p_windows[j] = float(np.mean(r >= r[treated_idx]))

    return DomainAggregate(
        outcomes=tuple(o.name for o in outcomes), tau=tau, p_value=p_value,
        treated_ratio=float(ratios[treated_idx]), ratios=ratios,
        windows=window_list, tau_by_window=taus, p_by_window=p_windows)
