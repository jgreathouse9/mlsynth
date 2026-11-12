"""Calibrated inference for TBR's cumulative effect (#1043).

TBR's interval is the conjugate posterior of the static regression
``y_t = a + b x_t`` with ``x`` the control-group sum. It prices independent
errors around a design fixed in advance. Weekly and daily sales are serially
correlated, and the design search picks the split whose pretest fit came out
best, so on autocorrelated panels the posterior covers 25-50% at a nominal 80%.

The calibration keeps the regression's point prediction and replaces only the
interval. It asks how large the regression's errors actually were on windows
where nothing happened, and mixes that evidence with the posterior.

1. The forecast. From an origin ``s`` the regression is fitted by OLS on
   ``[0, s)`` and predicts the sum of ``y`` over ``[s, s + h)``. With
   ``a = (h, sum of x over the window)`` the centre is ``a' beta_hat`` and the
   scale ``sqrt(S (h + a' (X'X)^-1 a))``, ``S = RSS / (s - 2)``, on ``s - 2``
   degrees of freedom. At ``s = E``, the first post period, this is TBR's own
   posterior predictive (Kerman et al. 2017, eqn 6), algebraically.
2. Blank errors. The last ``B`` pre-periods are held back from the design
   search. Every origin ``s`` in ``[E - B, E - h]`` gives a standardised error
   ``z = (observed sum - centre) / scale`` on a window no design choice saw,
   for the real design and for placebo designs.
3. Distance trend. Errors grow with the distance from the period the design was
   chosen on, and the experiment sits furthest away. ``z^2 = alpha + beta d``
   is fitted by OLS over the pooled errors, the slope is clamped at zero, and
   each error is rescaled to the experiment's distance ``B``.
4. Dirichlet-process predictive (Florens and Rolin 2024, eq. 5.13; Blackwell
   and MacQueen 1973). The counterfactual sum's predictive is the mixture
   ``w0 F_t + (1 - w0) ECDF`` of the posterior t and the empirical
   distribution of ``c + sc z_tilde``, with ``w0 = n0 / (n0 + n_eff)``. The
   interval is the mixture's two quantiles, found by bisection.

Placebo designs are other treatment groups of the real group's size that the
design rules admit, each with a control group matched by the search's own
climb on the design window, so their blank errors carry the same selection the
real design does. Under ``population_tolerance`` the rules carry the tolerance,
so the placebo groups are drawn from the groups within it, the same selection
the real group passed.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from scipy import special

from ...exceptions import MlsynthDataError
from .design.rules import DesignRules
from .design.search import (CONTROL, TREATMENT, UNASSIGNED, _match, _Scorer)

#: The bisection stops once its bracket is narrower than this many posterior
#: scales. The generalised quantile then lies within ``QUANTILE_TOL * scale``
#: below the returned point, about 1e-9 of the interval's width.
QUANTILE_TOL = 1e-9

#: Below this many placebo designs the pooled errors are thin; the fit warns.
MIN_PLACEBOS = 8

#: The trend's value at a blank error's own distance is floored here before it
#: divides, so a fit crossing zero cannot blow an error up.
TREND_FLOOR = 1e-6

#: The fewest periods before the first blank origin: OLS on ``(1, x)`` needs
#: three to leave one degree of freedom for its residual scale.
MIN_DESIGN_PERIODS = 3

#: Placebo groups are enumerated when the candidates number at most this many,
#: and drawn by rejection otherwise.
ENUMERATION_LIMIT = 20000

#: Rejection sampling stops after this many draws per wanted placebo.
DRAWS_PER_PLACEBO = 200

#: A regression whose residual sum of squares is at most this fraction of the
#: total fits the window exactly, and its standardised errors are undefined.
EXACT_FIT_RTOL = 1e-20


# --------------------------------------------------------------------------- #
# the forecast
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdidForecast:
    """The regression's forecast of the cumulative sum, at horizons 1..h."""

    centre: np.ndarray        # predicted sum of y over [s, s + h)
    scale: np.ndarray         # posterior predictive scale of that sum
    df: int                   # s - 2


def adid_forecast(y: np.ndarray, x: np.ndarray, origin: int,
                  horizon: int) -> AdidForecast:
    """OLS of ``y`` on ``(1, x)`` over ``[0, origin)``, projected forward.

    Returns the centre and scale of the predicted sum over ``[origin,
    origin + h)`` for every ``h`` from 1 to ``horizon``. A fit that reproduces
    the window exactly has no residual scale, and its scale is ``nan``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    s = int(origin)
    X = np.column_stack([np.ones(s), x[:s]])
    xtx_inv = np.linalg.pinv(X.T @ X)
    coef = xtx_inv @ (X.T @ y[:s])
    resid = y[:s] - X @ coef
    rss = float(resid @ resid)
    tss = float(np.sum((y[:s] - y[:s].mean()) ** 2))
    S = rss / (s - 2)
    hs = np.arange(1, horizon + 1, dtype=float)
    sx = np.cumsum(x[s:s + horizon])
    centre = coef[0] * hs + coef[1] * sx
    quad = (xtx_inv[0, 0] * hs ** 2 + 2.0 * xtx_inv[0, 1] * hs * sx
            + xtx_inv[1, 1] * sx ** 2)
    scale = np.sqrt(S * (hs + quad))
    if rss <= EXACT_FIT_RTOL * tss or tss == 0.0:
        scale = np.full_like(scale, np.nan)
    return AdidForecast(centre=centre, scale=scale, df=s - 2)


def blank_errors(y: np.ndarray, x: np.ndarray, design_end: int, end: int,
                 horizon: int) -> List[np.ndarray]:
    """Standardised errors at every blank origin, one array per horizon.

    Entry ``h - 1`` holds ``z`` for the origins ``s = design_end ..
    end - h`` in order, so its index is the distance ``s - design_end``.
    """
    sums = np.concatenate([[0.0], np.cumsum(np.asarray(y, dtype=float))])
    out: List[List[float]] = [[] for _ in range(horizon)]
    for s in range(design_end, end):
        reach = min(horizon, end - s)
        f = adid_forecast(y, x, s, reach)
        observed = sums[s + 1:s + reach + 1] - sums[s]
        z = (observed - f.centre) / f.scale
        for h in range(reach):
            out[h].append(float(z[h]))
    return [np.asarray(v) for v in out]


# --------------------------------------------------------------------------- #
# the distance trend
# --------------------------------------------------------------------------- #
def distance_trend(distance: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """OLS of ``z^2`` on the distance, with the slope clamped at zero.

    Clamping keeps the intercept the unconstrained fit gave, which at a
    non-positive slope is at least the mean of ``z^2``, so the trend at the
    experiment is positive whenever any error is non-zero.
    """
    d = np.asarray(distance, dtype=float)
    z2 = np.asarray(z, dtype=float) ** 2
    dc = d - d.mean()
    beta = float(dc @ (z2 - z2.mean()) / (dc @ dc))
    alpha = float(z2.mean() - beta * d.mean())
    return alpha, max(beta, 0.0)


def rescale_to_experiment(z: np.ndarray, distance: np.ndarray, alpha: float,
                          beta: float, target: float) -> np.ndarray:
    """Each error scaled from its own distance to the experiment's."""
    here = np.maximum(alpha + beta * np.asarray(distance, dtype=float),
                      TREND_FLOOR)
    return np.asarray(z, dtype=float) * np.sqrt((alpha + beta * target) / here)


# --------------------------------------------------------------------------- #
# the Dirichlet-process predictive
# --------------------------------------------------------------------------- #
def mixture_cdf(v: float, centre: float, scale: float, df: int,
                points: np.ndarray, w0: float) -> float:
    """``w0 F_t(v) + (1 - w0) ECDF(v)``; ``points`` sorted, equal weights."""
    t = float(special.stdtr(df, (v - centre) / scale))
    if points.size == 0:
        return t
    emp = np.searchsorted(points, v, side="right") / points.size
    return w0 * t + (1.0 - w0) * float(emp)


def mixture_quantile(q: float, centre: float, scale: float, df: int,
                     points: np.ndarray, w0: float,
                     tol: float = QUANTILE_TOL) -> float:
    """The smallest ``v`` with ``mixture_cdf(v) >= q``, to ``tol * scale``.

    The bracket opens one scale below the lower of the t quantile and the
    smallest point, where the CDF is below ``q``, and closes one scale above the
    higher of the two, where it is at least ``q``; the scale of margin absorbs
    the rounding of the t quantile itself. Bisection keeps that invariant and
    stops at half the tolerance, so the returned upper end has reached ``q`` and
    the point ``tol * scale`` below it has not.
    """
    t_q = centre + scale * float(special.stdtrit(df, q))
    lo = min(t_q, points[0] if points.size else t_q) - scale
    hi = max(t_q, points[-1] if points.size else t_q) + scale
    step = 0.5 * tol * scale
    while hi - lo > step:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:          # pragma: no cover - floating point
            break                           # exhausted below the tolerance
        if mixture_cdf(mid, centre, scale, df, points, w0) >= q:
            hi = mid
        else:
            lo = mid
    return hi


def calibrated_interval(centre: float, scale: float, df: int, z: np.ndarray,
                        prior_weight: float, n_eff: int,
                        level: float) -> Tuple[float, float]:
    """The ``level`` interval of the counterfactual sum's mixture predictive."""
    points = np.sort(centre + scale * np.asarray(z, dtype=float))
    w0 = prior_weight / (prior_weight + n_eff)
    lo = mixture_quantile(0.5 * (1.0 - level), centre, scale, df, points, w0)
    hi = mixture_quantile(0.5 * (1.0 + level), centre, scale, df, points, w0)
    return lo, hi


# --------------------------------------------------------------------------- #
# placebo designs
# --------------------------------------------------------------------------- #
def placebo_groups(treatment: Sequence[int],
                   eligibility: Sequence[FrozenSet[str]],
                   rules: Optional[DesignRules], n_placebos: int,
                   seed: int) -> Tuple[List[List[int]], Optional[int]]:
    """Up to ``n_placebos`` treatment groups drawn uniformly, without the real.

    A group has the real group's size, holds every forced geo (eligible for
    treatment alone, so in every design), draws the rest from the
    treatment-eligible geos, and is kept only when ``rules`` admit it. Returns
    the groups, sorted within, and the number of admissible groups other than
    the real one when they were enumerated, or ``None`` when there were too
    many candidates to enumerate and the groups were drawn by rejection.
    """
    real = sorted(int(j) for j in treatment)
    forced = [j for j, r in enumerate(eligibility) if r == frozenset({TREATMENT})]
    free = [j for j, r in enumerate(eligibility)
            if TREATMENT in r and j not in forced]
    m = len(real) - len(forced)
    rng = np.random.default_rng(seed)

    def ok(group: List[int]) -> bool:
        return group != real and (rules is None or rules.admits(group))

    if m < 0 or m > len(free):
        return [], 0
    if math.comb(len(free), m) <= ENUMERATION_LIMIT:
        pool = [g for g in (sorted(forced + list(c))
                            for c in combinations(free, m)) if ok(g)]
        take = rng.choice(len(pool), size=min(n_placebos, len(pool)),
                          replace=False) if pool else []
        return [pool[i] for i in take], len(pool)
    found: List[List[int]] = []
    seen = set()
    for _ in range(DRAWS_PER_PLACEBO * n_placebos):
        g = sorted(forced + [free[i] for i in
                             rng.choice(len(free), size=m, replace=False)])
        if tuple(g) not in seen and ok(g):
            seen.add(tuple(g))
            found.append(g)
            if len(found) == n_placebos:
                break
    return found, None


def match_controls(scorer: _Scorer, treatment: Sequence[int],
                   eligibility: Sequence[FrozenSet[str]],
                   rules: Optional[DesignRules]) -> Optional[List[int]]:
    """The search's matching climb for a fixed treatment group.

    It starts from every control-eligible geo the group leaves free and the
    rules do not ban, toggles the geos that may be held out, and stays inside
    the control band. ``None`` when no control group fits.
    """
    T = list(treatment)
    pool = [j for j, r in enumerate(eligibility) if CONTROL in r]
    banned = frozenset() if rules is None else rules.banned_controls(T)
    start = set(pool) - set(T) - banned
    if not start:
        return None
    toggleable = [j for j in pool if j not in T and j not in banned
                  and UNASSIGNED in eligibility[j]]
    band = (1, None) if rules is None else rules.control_band
    matched = _match(scorer, T, start, toggleable, band)
    return None if matched is None else sorted(matched[0])


# --------------------------------------------------------------------------- #
# the calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlankCalibration:
    """Everything the band and the calibrated MDE read, per horizon 1..H."""

    blank_periods: int
    design_end: int                       # D0 = E - B
    end: int                              # E
    designs: List[Tuple[List[int], List[int]]]   # the real design first
    placebos: List[Tuple[List[int], List[int]]]
    n_admissible: Optional[int]
    dropped: int                          # placebos with undefined errors
    alpha: List[float]
    beta: List[float]
    z: List[np.ndarray] = field(repr=False)       # raw, pooled, per horizon
    z_tilde: List[np.ndarray] = field(repr=False)
    k: List[float] = field(default_factory=list)
    n_eff: List[int] = field(default_factory=list)


def calibrate(W: np.ndarray, treatment: Sequence[int], control: Sequence[int],
              *, blank_periods: int, horizon: int,
              eligibility: Sequence[FrozenSet[str]],
              rules: Optional[DesignRules], objective: str, n_test: int,
              n_placebos: int, seed: int) -> BlankCalibration:
    """Blank errors of the real and placebo designs, trended and pooled.

    ``W`` is the pre-period, periods by geos, ``E`` rows. Placebo control
    groups are matched on ``[0, E - B)`` under ``objective`` and ``n_test``,
    the window and objective the design search uses.
    """
    W = np.asarray(W, dtype=float)
    E, B = W.shape[0], int(blank_periods)
    D0 = design_end(E, B)
    if B - horizon + 1 < 2:
        raise MlsynthDataError(
            f"a horizon of {horizon} periods has {max(B - horizon + 1, 0)} "
            f"blank origin(s) in a blank window of {B}; the distance trend "
            f"needs two. Lengthen blank_periods.")

    def errors(design):
        T, C = design
        return blank_errors(W[:, T].sum(axis=1), W[:, C].sum(axis=1), D0, E,
                            horizon)

    real = (sorted(int(j) for j in treatment), sorted(int(j) for j in control))
    real_z = errors(real)
    if not all(np.all(np.isfinite(v)) for v in real_z):
        raise MlsynthDataError(
            "the real design's regression reproduces a blank window exactly, so "
            "its standardised errors are undefined; the treated sum is an "
            "exact linear function of the control sum.")

    scorer = _Scorer(W[:D0], objective, n_test)
    groups, n_admissible = placebo_groups(treatment, eligibility, rules,
                                          n_placebos, seed)
    candidates = []
    for g in groups:
        c = match_controls(scorer, g, eligibility, rules)
        if c is not None:
            candidates.append((g, c))
    per_design = [real_z]
    placebos = []
    for design in candidates:
        z = errors(design)
        if all(np.all(np.isfinite(v)) for v in z):
            per_design.append(z)
            placebos.append(design)
    dropped = len(candidates) - len(placebos)
    if len(placebos) < MIN_PLACEBOS:
        warnings.warn(
            f"the calibration pools {len(placebos)} placebo design(s), fewer "
            f"than {MIN_PLACEBOS}; the design rules and eligibility admit few "
            f"other treatment groups of this size, so the blank errors are "
            f"thin and the band is less reliable.", UserWarning, stacklevel=3)

    n_designs = len(per_design)
    alpha, beta, zs, zt, ks, n_eff = [], [], [], [], [], []
    for h in range(1, horizon + 1):
        z = np.concatenate([d[h - 1] for d in per_design])
        dist = np.tile(np.arange(B - h + 1, dtype=float), n_designs)
        a, b = distance_trend(dist, z)
        z_tilde = rescale_to_experiment(z, dist, a, b, float(B))
        alpha.append(a)
        beta.append(b)
        zs.append(z)
        zt.append(z_tilde)
        ks.append(float(np.sqrt(np.mean(z_tilde ** 2))))
        n_eff.append(n_designs * max(1, B // h))
    return BlankCalibration(
        blank_periods=B, design_end=D0, end=E, designs=[real] + placebos,
        placebos=placebos, n_admissible=n_admissible, dropped=dropped,
        alpha=alpha, beta=beta, z=zs, z_tilde=zt, k=ks, n_eff=n_eff)


def calibrated_band(blank: BlankCalibration, y: np.ndarray, x: np.ndarray,
                    n_test: int, level: float, prior_weight: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """The calibrated effect interval at every horizon of the test window.

    The centre and scale are the real design's forecast from ``E``, which is
    TBR's posterior; the interval for the counterfactual sum is flipped into one
    for the effect, observed minus counterfactual.
    """
    E = blank.end
    f = adid_forecast(y, x, E, n_test)
    observed = np.cumsum(np.asarray(y, dtype=float)[E:E + n_test])
    lower = np.empty(n_test)
    upper = np.empty(n_test)
    for h in range(n_test):
        lo, hi = calibrated_interval(f.centre[h], f.scale[h], f.df,
                                     blank.z_tilde[h], prior_weight,
                                     blank.n_eff[h], level)
        lower[h] = observed[h] - hi
        upper[h] = observed[h] - lo
    return lower, upper


def design_end(n_pre: int, blank_periods: int) -> int:
    """``D0 = E - B``, refused when too few periods are left to fit from."""
    D0 = n_pre - blank_periods
    if D0 < MIN_DESIGN_PERIODS:
        raise MlsynthDataError(
            f"the pre-period has {n_pre} periods and {blank_periods} are held "
            f"back as blank windows, which leaves {D0} to fit from; the "
            f"calibration needs at least {MIN_DESIGN_PERIODS}. Shorten "
            f"blank_periods or supply a longer pre-period.")
    return D0


def resolve_blank_periods(requested: Optional[int], test_length: int) -> int:
    """``B``: the requested value, or three test lengths; at least two."""
    B = 3 * test_length if requested is None else int(requested)
    if B < 2 * test_length:
        raise MlsynthDataError(
            f"blank_periods is {B} and the test runs {test_length} periods; "
            f"the blank window has to be at least 2 x the test length so the "
            f"longest horizon has blank origins to pool over.")
    return B
