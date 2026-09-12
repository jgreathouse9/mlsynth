"""Confidence sets by inverting the in-space placebo test.

Firpo, S. & Possebom, V. (2018), "Synthetic Control Method: Inference,
Sensitivity Analysis and Confidence Sets", *Journal of Causal Inference* 6(2),
20160026.

The ordinary placebo test (``inference="placebo"``) answers one hypothesis: that
the effect is zero. This inverts it over a one-parameter family of effect paths
-- constant ``phi``, or linear ``phi * (t - T0)`` -- and reports the parameters
the test does not reject, which is a confidence set for the path, not a
verdict on a single null.

What makes it an inversion and not a quantile
---------------------------------------------

The candidate effect is imposed across the whole panel before any statistic is
recomputed. For each placebo unit ``j`` the path is added to ``j``'s own outcome
and subtracted from the treated unit's column inside ``j``'s donor pool, because
under a non-zero null the treated unit's observed series is not its untreated
one and every donor pool containing it is wrong by exactly that path. Only then
is each unit's post/pre MSPE ratio recomputed and ranked.

Taking quantiles of a placebo distribution computed once at zero is a different
object: the post/pre ratio is not invariant to the hypothesised effect, so the
reference distribution has to move with the null.

Sensitivity to who could have been treated
------------------------------------------

The placebo test assumes every unit was equally likely to receive treatment.
Relaxing that, the rank p-value is reweighted by ``prob = softmax(phi * v)``
over units, with ``v`` a declared 0/1 vector naming the units the design might
have favoured. At ``phi = 0`` this is the uniform test. Sweeping ``phi`` upward
asks how far from uniform assignment the conclusion survives -- the question
``docs/vanillasc.rst`` poses as the Rosenbaum ``Gamma`` and defers as needing a
non-convex program. Declaring the direction instead of optimising over it makes
it closed form, at the cost of a weaker statement: worst case within a nominated
direction, not over a ball.

Donor weights are an input here, exactly as they are in the authors' reference
implementation, which is why this module takes them and does not fit anything.

Cross-validated against the authors' own ``SCM.CS`` to 1.8e-14 over eight
configurations; see ``benchmarks/reference/fp_confidence_sets/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError

EFFECT_CLASSES = ("constant", "linear")


@dataclass(frozen=True)
class PlaceboConfidenceSet:
    """A confidence set for the effect-path parameter.

    ``lower`` and ``upper`` bound the parameter; ``lower_path`` and
    ``upper_path`` are the corresponding per-period effect paths, zero through
    the pre-period, which is what a plot needs.
    """

    lower: float
    upper: float
    point_estimate: float
    kind: str
    alpha: float
    precision: int
    phi: float
    lower_path: np.ndarray
    upper_path: np.ndarray
    pre_periods: int = 0

    @property
    def contains_zero(self) -> bool:
        return bool(self.lower <= 0.0 <= self.upper)

    @property
    def n_post(self) -> int:
        """Post-treatment periods the effect path covers."""
        return int(self.lower_path.size - self.pre_periods)

    @property
    def cumulative(self) -> tuple:
        """The set read as the total post-treatment effect, ``(lower, upper)``.

        Summing the path is a strictly increasing function of the parameter --
        ``c * K`` for the constant class and ``c * K(K+1)/2`` for the linear one,
        over ``K`` post-treatment periods -- so the image of the confidence set
        is the confidence set of the image, at the same level. Inverting the
        test on this scale would return these same two numbers.

        The coverage statement is the family's: this covers the total effect of
        every path in the class the test does not reject, so it is a statement
        about the cumulative effect under the maintained assumption that the
        true path is constant (or linear) in time.
        """
        return (float(self.lower_path.sum()), float(self.upper_path.sum()))

    @property
    def average(self) -> tuple:
        """The set read as the average post-treatment effect, ``(lower, upper)``.

        The cumulative scale divided by the number of post-treatment periods,
        which puts the set on the same scale as the estimator's reported ATT.
        Carries the same caveat as :attr:`cumulative`.
        """
        k = self.n_post
        if k <= 0:  # pragma: no cover - confidence_set refuses an empty window
            raise MlsynthEstimationError("there are no post-treatment periods")
        lo, hi = self.cumulative
        return (lo / k, hi / k)


@dataclass(frozen=True)
class SensitivityRow:
    """One tilt of the assignment distribution.

    ``confidence_set`` is ``None`` when the search did not terminate at this
    tilt, with ``reason`` saying why. A sweep is a diagnostic, so one unbounded
    tilt is recorded and the rest still run.
    """

    phi: float
    confidence_set: Optional[PlaceboConfidenceSet]
    reason: str = ""

    @property
    def contains_zero(self) -> Optional[bool]:
        return None if self.confidence_set is None else self.confidence_set.contains_zero


def effect_path(value: float, n_periods: int, pre_periods: int,
                kind: str) -> np.ndarray:
    """The candidate effect path: zero pre-treatment, then ``kind``.

    ``linear`` counts post-treatment periods from one, so the parameter is the
    per-period slope and the final entry is ``value * (n_periods - pre_periods)``.
    """
    if kind not in EFFECT_CLASSES:
        raise MlsynthEstimationError(
            f"unknown effect class {kind!r}; expected 'constant' or 'linear'")
    path = np.zeros(int(n_periods), dtype=float)
    n_post = int(n_periods) - int(pre_periods)
    if kind == "constant":
        path[pre_periods:] = value
    else:
        path[pre_periods:] = value * np.arange(1, n_post + 1)
    return path


def _check(Y: np.ndarray, W: np.ndarray, treated_index: int,
           pre_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the panel, the weights and the split, and fail with the reason."""
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    if Y.ndim != 2:
        raise MlsynthEstimationError(
            f"outcome matrix must be (periods, units); got shape {Y.shape}")
    n_periods, n_units = Y.shape
    if n_units < 3:
        raise MlsynthEstimationError(
            f"the placebo test needs a donor pool: {n_units} unit(s) in the panel, "
            "and ranking the treated unit needs at least two donors")
    if not (0 <= int(treated_index) < n_units):
        raise MlsynthEstimationError(
            f"treated index {treated_index} is outside the panel's "
            f"{n_units} units")
    if W.shape != (n_units - 1, n_units):
        raise MlsynthEstimationError(
            f"weight matrix shape {W.shape} does not match the panel: expected "
            f"({n_units - 1}, {n_units}), one column of donor weights per unit")
    if not (0 < int(pre_periods) < n_periods):
        raise MlsynthEstimationError(
            f"pre_periods={pre_periods} leaves no post-treatment periods in a "
            f"panel of {n_periods}")
    if not np.isfinite(Y).all() or not np.isfinite(W).all():
        raise MlsynthEstimationError(
            "outcome matrix and weights must be finite")
    return Y, W


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ascending ranks with ties averaged -- R's ``rank`` default.

    The reference ranks with R's default, and ties are not hypothetical: a
    degenerate placebo fit can give two units the same ratio.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=float)
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        totals = np.zeros(counts.shape[0], dtype=float)
        np.add.at(totals, inverse, ranks)
        ranks = (totals / counts)[inverse]
    return ranks


def _mspe_ratios(Y: np.ndarray, W: np.ndarray, treated_index: int,
                 pre_periods: int, path: np.ndarray,
                 impose_on_donor_pool: bool = True) -> np.ndarray:
    """Every unit's post/pre MSPE ratio under the candidate effect path."""
    n_periods, n_units = Y.shape
    ratios = np.empty(n_units, dtype=float)
    for j in range(n_units):
        donors = [k for k in range(n_units) if k != j]
        if j == treated_index:
            y1 = Y[:, treated_index]
            y0 = Y[:, donors]
        else:
            y1 = Y[:, j] + path
            y0 = Y[:, donors]
            if impose_on_donor_pool:
                # the treated unit's position inside j's donor pool, after j
                # itself has been removed
                position = treated_index - 1 if j < treated_index else treated_index
                y0 = y0.copy()
                y0[:, position] -= path
        gaps = y1 - y0 @ W[:, j] - path
        post = float(gaps[pre_periods:] @ gaps[pre_periods:]) / (n_periods - pre_periods)
        pre = float(gaps[:pre_periods] @ gaps[:pre_periods]) / pre_periods
        ratios[j] = np.inf if pre == 0.0 else post / pre
    return ratios


def placebo_pvalue(Y: np.ndarray, W: np.ndarray, treated_index: int,
                   pre_periods: int, path: np.ndarray, phi: float = 0.0,
                   v: Optional[Sequence[float]] = None,
                   _impose_on_donor_pool: bool = True) -> float:
    """The (optionally reweighted) rank p-value for one candidate path.

    With ``phi = 0`` this is the ordinary in-space placebo p-value evaluated
    under ``path``: the share of units whose post/pre MSPE ratio is at least the
    treated unit's.
    """
    Y, W = _check(Y, W, treated_index, pre_periods)
    path = np.asarray(path, dtype=float)
    if path.shape[0] != Y.shape[0]:
        raise MlsynthEstimationError(
            f"effect path has {path.shape[0]} periods, panel has {Y.shape[0]}")
    ratios = _mspe_ratios(Y, W, treated_index, pre_periods, path,
                          _impose_on_donor_pool)
    ranks = _ranks(ratios)
    n_units = Y.shape[1]
    v_arr = np.zeros(n_units) if v is None else np.asarray(v, dtype=float).ravel()
    if v_arr.shape[0] != n_units:
        raise MlsynthEstimationError(
            f"sensitivity vector has {v_arr.shape[0]} entries, panel has "
            f"{n_units} units")
    weights = np.exp(float(phi) * v_arr)
    probabilities = weights / weights.sum()
    return float(probabilities @ (ranks >= ranks[treated_index]).astype(float))


def confidence_set(Y, W, treated_index: int, pre_periods: int, *,
                   kind: str = "linear", alpha: float = 0.05,
                   precision: int = 30, phi: float = 0.0,
                   v: Optional[Sequence[float]] = None) -> PlaceboConfidenceSet:
    """Invert the placebo test into a confidence set for the effect parameter.

    The search starts at the point estimate and walks each bound outward in
    steps of ``(1/2)**power``, stepping back in on a rejection. ``power`` runs
    to ``precision``, halving the bracket at each level.

    Two consequences follow from starting there, and both are the procedure's,
    not this implementation's.

    The point estimate is not guaranteed to survive the test. For the constant
    class it is the mean post-treatment gap and for the linear class the final
    gap spread over the post-periods, and imposing either as the null leaves the
    treated unit a residual that can still rank high enough to reject. When that
    happens the routine reports an empty set, even where the acceptance region
    is non-empty somewhere the search never reaches.

    What is returned is the connected component of ``{c : p(c) > alpha}``
    containing the point estimate. The p-value is a rank, so it is a step
    function of the candidate and its acceptance region is not guaranteed to be
    an interval; on the reference panel it is one component, which
    ``test_the_search_agrees_with_a_brute_force_scan`` checks against a grid.

    Raises
    ------
    MlsynthEstimationError
        If the point estimate itself is rejected (reported as an empty set), if
        a bound runs away without ever being rejected, or if the panel, weights
        or split are inconsistent.
    """
    Y, W = _check(Y, W, treated_index, pre_periods)
    if kind not in EFFECT_CLASSES:
        raise MlsynthEstimationError(
            f"unknown effect class {kind!r}; expected 'constant' or 'linear'")
    n_periods, n_units = Y.shape
    donors = [k for k in range(n_units) if k != treated_index]
    gaps = Y[:, treated_index] - Y[:, donors] @ W[:, treated_index]
    if kind == "constant":
        point = float(gaps[pre_periods:].mean())
    else:
        point = float(gaps[-1]) / (n_periods - pre_periods)
    if point == 0.0:
        raise MlsynthEstimationError(
            "the point estimate is exactly zero, so the search has no direction "
            "to walk; the inversion is not defined for this panel")
    sign = float(np.sign(point))
    upper = lower = point

    def rejects(value: float) -> bool:
        path = effect_path(value, n_periods, pre_periods, kind)
        return placebo_pvalue(Y, W, treated_index, pre_periods, path, phi, v) <= alpha

    first_upper = first_lower = True
    for power in range(0, int(precision) + 1):
        step = 0.5 ** power
        while True:
            if not rejects(upper):
                upper = point * (upper / point + sign * step)
                first_upper = False
            else:
                if first_upper:
                    raise MlsynthEstimationError(
                        f"the confidence set is empty for the {kind!r} effect "
                        f"class at alpha={alpha}: the point estimate itself is "
                        "rejected, so no path in this family survives the test")
                upper = point * (upper / point - sign * step)
                break
            if abs(upper) > 100 * abs(point):
                raise MlsynthEstimationError(
                    f"the upper bound was not found for the {kind!r} class at "
                    f"alpha={alpha}: the search passed 100x the point estimate "
                    "without a rejection, so the set is unbounded at this level")
        while True:
            # mirrored against the upper branch: the lower bound walks away from
            # the point estimate on a non-rejection and steps back in on one
            if not rejects(lower):
                lower = point * (lower / point - sign * step)
                first_lower = False
            else:
                if first_lower:
                    raise MlsynthEstimationError(
                        f"the confidence set is empty for the {kind!r} effect "
                        f"class at alpha={alpha}: the point estimate itself is "
                        "rejected, so no path in this family survives the test")
                lower = point * (lower / point + sign * step)
                break
            if abs(lower) > 100 * abs(point):
                raise MlsynthEstimationError(
                    f"the lower bound was not found for the {kind!r} class at "
                    f"alpha={alpha}: the search passed 100x the point estimate "
                    "without a rejection, so the set is unbounded at this level")

    return PlaceboConfidenceSet(
        lower=float(lower), upper=float(upper), point_estimate=point, kind=kind,
        alpha=float(alpha), precision=int(precision), phi=float(phi),
        lower_path=effect_path(lower, n_periods, pre_periods, kind),
        upper_path=effect_path(upper, n_periods, pre_periods, kind),
        pre_periods=int(pre_periods))


def sensitivity_sweep(Y, W, treated_index: int, pre_periods: int, *,
                      phis: Sequence[float], v: Sequence[float],
                      kind: str = "linear", alpha: float = 0.05,
                      precision: int = 30) -> list:
    """Confidence sets across a sweep of assignment-probability tilts.

    Returns one :class:`SensitivityRow` per ``phi``. A tilt whose search does
    not terminate is recorded with its reason and the sweep continues, because
    the point of the sweep is to see where the conclusion turns over and an
    unbounded tilt is part of that picture.
    """
    rows = []
    for phi in phis:
        try:
            cs = confidence_set(Y, W, treated_index, pre_periods, kind=kind,
                                alpha=alpha, precision=precision, phi=float(phi),
                                v=v)
            rows.append(SensitivityRow(phi=float(phi), confidence_set=cs))
        except MlsynthEstimationError as exc:
            rows.append(SensitivityRow(phi=float(phi), confidence_set=None,
                                       reason=str(exc)))
    return rows


def breakdown_phi(rows: Sequence[SensitivityRow]) -> Optional[float]:
    """The smallest swept tilt at which the set admits zero, if any.

    ``None`` means the conclusion held across every tilt that terminated -- not
    that it holds everywhere, only that the sweep did not find where it fails.
    """
    for row in rows:
        if row.confidence_set is not None and row.confidence_set.contains_zero:
            return row.phi
    return None
