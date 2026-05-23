"""Program- and arm-level power / MDE analysis for a PANGEO design.

Once a PANGEO design is frozen, the minimum detectable effect (MDE) after
``X`` post-treatment periods is a closed-form function of the *pre-period
parallelism the design achieved* -- which is exactly what the MILP
minimised, so power and the design objective are the same quantity.

For a supergeo pair the no-effect gap
:math:`g_t = \\bar Y^T_t - \\bar Y^C_t` sits on its parallel-trends line
:math:`\\delta_p = \\overline{g}_{\\text{pre}}`; its per-period residual
variance is

.. math::

   \\sigma_p^2 = \\text{ss\\_res}_p / (T_0 - 1)
   \\qquad (\\text{ss\\_res}_p = \\text{pair.gap\\_variance}),

the noise an ``X``-period difference-in-differences ATT must overcome. The
estimator :math:`\\hat\\tau_p = \\overline{g}_{\\text{post}} - \\delta_p`
has variance

.. math::

   \\operatorname{Var}(\\hat\\tau_p)
   = \\sigma_p^2\\,\\big[f(X,\\rho) + f(T_0,\\rho)\\big],

where :math:`f(n,\\rho) = \\operatorname{Var}(\\text{mean of } n
\\text{ serially-correlated periods})/\\sigma^2` is the variance-inflation
factor of an AR(1) process. Consecutive weeks are correlated, so ``X`` post
weeks are worth far fewer than ``X`` independent draws -- the trap a naive
i.i.d. power calculation falls into. :math:`\\rho` is estimated from the
pooled pre-period gap residuals of the chosen pairs.

The **program** ATT is the treated-size-weighted average of the pair ATTs;
its MDE is the headline number a program owner reports. Per-arm curves are
also returned. Pairs are treated as independent across the program, so the
arm count multiplies the effective sample size -- which is *why* pooling to
the program level detects far smaller effects than any one small arm could.
Cross-pair common shocks within an arm are ignored (a mild optimism; a
placebo-in-time engine would absorb them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class MDEPoint:
    """Minimum detectable effect at one post-period horizon.

    Attributes
    ----------
    post_periods : int
        Number of post-treatment periods (``X``) assumed.
    mde_absolute : float
        MDE on the per-unit outcome scale.
    mde_pct : float
        MDE as a percentage of the baseline outcome level.
    se : float
        Standard error of the ``X``-period ATT under the design.
    """

    post_periods: int
    mde_absolute: float
    mde_pct: float
    se: float


@dataclass(frozen=True)
class PowerCurve:
    """MDE-vs-horizon curve at one aggregation level (program or arm).

    Attributes
    ----------
    level : str
        ``"program"`` or an arm label.
    baseline : float
        Treated-group baseline outcome level used to express ``mde_pct``.
    n_treated : int
        Number of treated geos contributing.
    n_pairs : int
        Number of supergeo pairs contributing.
    points : list of MDEPoint
        One entry per post-period horizon.
    """

    level: str
    baseline: float
    n_treated: int
    n_pairs: int
    points: List[MDEPoint]

    def mde_pct_by_horizon(self) -> Dict[int, float]:
        """``{post_periods: mde_pct}`` for quick lookup."""
        return {pt.post_periods: pt.mde_pct for pt in self.points}


@dataclass(frozen=True)
class PangeoPower:
    """Power / MDE analysis attached to :class:`PangeoResults`.

    Attributes
    ----------
    program : PowerCurve
        Headline program-level MDE curve (pooled across all arms).
    arms : dict
        ``{arm_label: PowerCurve}`` -- per-arm MDE curves.
    alpha : float
        Two-sided significance level assumed.
    power_target : float
        Target power the MDEs are computed at (default 0.80).
    post_periods : list of int
        Horizons evaluated.
    serial_correlation : float
        Pooled lag-1 (AR(1)) autocorrelation of the gap residuals used to
        inflate the variance for serial dependence.
    """

    program: PowerCurve
    arms: Dict[Any, PowerCurve]
    alpha: float
    power_target: float
    post_periods: List[int]
    serial_correlation: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Tidy table of MDE (% of baseline) by horizon: program + each arm."""
        data: Dict[str, Any] = {
            "post_periods": list(self.post_periods),
            "program_mde_pct": [pt.mde_pct for pt in self.program.points],
            "program_mde_abs": [pt.mde_absolute for pt in self.program.points],
            "program_se": [pt.se for pt in self.program.points],
        }
        for arm, curve in self.arms.items():
            data[f"arm_{arm}_mde_pct"] = [pt.mde_pct for pt in curve.points]
        return pd.DataFrame(data).set_index("post_periods")

    def power_for_effect(
        self, effect_pct: float, post_periods: int, level: str = "program",
    ) -> float:
        """Power to detect a ``effect_pct`` % effect at horizon ``post_periods``.

        Inverts the MDE relation for a given true effect size (two-sided
        Gaussian approximation).
        """
        curve = self.program if level == "program" else self.arms[level]
        pt = next((p for p in curve.points if p.post_periods == post_periods),
                  None)
        if pt is None or pt.se <= 0:
            return float("nan")
        effect_abs = abs(effect_pct) / 100.0 * curve.baseline
        z = effect_abs / pt.se
        z_a = norm.ppf(1.0 - self.alpha / 2.0)
        return float(norm.cdf(z - z_a) + norm.cdf(-z - z_a))


def _variance_inflation(n: int, rho: float) -> float:
    """``Var(mean of n serially-correlated periods) / sigma^2``.

    For an AR(1) process with lag-1 correlation ``rho`` this is
    ``(1 + 2 * sum_{k=1}^{n-1} (1 - k/n) rho^k) / n`` (``= 1/n`` when
    ``rho = 0``).
    """
    if n <= 0:
        return 0.0
    if n == 1 or abs(rho) < 1e-12:
        return 1.0 / n
    k = np.arange(1, n)
    s = float(np.sum((1.0 - k / n) * rho ** k))
    return (1.0 + 2.0 * s) / n


def _pooled_ar1_rho(residual_series: Sequence[np.ndarray]) -> float:
    """Pooled lag-1 autocorrelation across the pairs' gap residuals."""
    num = 0.0
    den = 0.0
    for r in residual_series:
        r = np.asarray(r, dtype=float)
        if r.size < 2:
            continue
        num += float(r[:-1] @ r[1:])
        den += float(r @ r)
    if den <= 1e-12:
        return 0.0
    return float(np.clip(num / den, -0.99, 0.99))


def _build_curve(
    level: str, pairs: List[dict], post_periods: Sequence[int],
    rho: float, mult: float,
) -> PowerCurve:
    n_treated = int(sum(p["n_t"] for p in pairs))
    if n_treated == 0:
        return PowerCurve(level, float("nan"), 0, 0,
                          [MDEPoint(int(X), float("nan"), float("nan"),
                                    float("nan")) for X in post_periods])
    w = np.array([p["n_t"] for p in pairs], dtype=float)
    w = w / w.sum()
    sigma2 = np.array([p["sigma2"] for p in pairs], dtype=float)
    T0 = np.array([p["T0"] for p in pairs], dtype=float)
    baselines = np.array([p["baseline"] for p in pairs], dtype=float)
    baseline = float(w @ baselines)

    pre_factor = np.array([_variance_inflation(int(t), rho) for t in T0])
    points: List[MDEPoint] = []
    for X in post_periods:
        post_factor = _variance_inflation(int(X), rho)
        var_p = sigma2 * (post_factor + pre_factor)
        var = float(np.sum(w ** 2 * var_p))
        se = float(np.sqrt(var))
        mde_abs = mult * se
        mde_pct = (mde_abs / baseline * 100.0) if abs(baseline) > 1e-12 \
            else float("nan")
        points.append(MDEPoint(int(X), mde_abs, mde_pct, se))
    return PowerCurve(level, baseline, n_treated, len(pairs), points)


def compute_pangeo_power(
    arm_designs: Dict[Any, Any],
    *,
    post_periods: Optional[Sequence[int]] = None,
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> PangeoPower:
    """Program- and arm-level MDE curves for a frozen PANGEO design.

    Parameters
    ----------
    arm_designs : dict
        ``{arm_label: ArmDesign}`` from a completed design.
    post_periods : sequence of int, optional
        Horizons to evaluate (default ``range(2, 13)`` = 2..12).
    alpha : float
        Two-sided significance level (default 0.05).
    power_target : float
        Target power (default 0.80).
    """
    if post_periods is None:
        post_periods = list(range(2, 13))
    post_periods = [int(X) for X in post_periods]
    mult = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power_target))

    residual_series: List[np.ndarray] = []
    by_arm: Dict[Any, List[dict]] = {}
    for arm, d in arm_designs.items():
        rows: List[dict] = []
        for p in d.pairs:
            gap = np.asarray(p.treatment_mean) - np.asarray(p.control_mean)
            resid = gap - gap.mean()
            residual_series.append(resid)
            T0 = gap.size
            rows.append({
                "arm": arm,
                "sigma2": float(p.gap_variance) / max(T0 - 1, 1),
                "n_t": len(p.treatment),
                "baseline": float(np.asarray(p.treatment_mean).mean()),
                "T0": T0,
            })
        by_arm[arm] = rows

    rho = _pooled_ar1_rho(residual_series)
    all_pairs = [r for rows in by_arm.values() for r in rows]

    program = _build_curve("program", all_pairs, post_periods, rho, mult)
    arms = {arm: _build_curve(str(arm), rows, post_periods, rho, mult)
            for arm, rows in by_arm.items()}

    return PangeoPower(
        program=program,
        arms=arms,
        alpha=alpha,
        power_target=power_target,
        post_periods=post_periods,
        serial_correlation=rho,
        metadata={
            "multiplier": mult,
            "variance_model": "AR(1) serial-correlation-corrected DiD; "
                              "treated-size-weighted program pooling; "
                              "pairs independent across program.",
        },
    )
