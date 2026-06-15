"""Cross-estimator design comparison on a common fit-vs-power plane.

GEOLIFT and SYNDES differ in their optimisers but share a grammar: each emits a
design that reduces to a unit-level contrast ``c = w_treated - w_control`` (both
sides summing to one) over the same panel. From ``c`` two comparable numbers
follow -- the pre-period fit RMSE ``sqrt(mean((Y_pre @ c)**2))`` and, by
injecting a known effect at a fixed horizon, a minimum detectable effect.

Scoring every design from either method through one shared simulation harness --
same horizon, same effect grid, same moving-block permutation null, same
baseline -- puts them on identical axes, so the per-method Pareto frontiers can
be overlaid and compared directly. This is the point of the module: the frontier
gap then reflects the *designs*, not two different power methodologies.

The MDE simulation rests on one fact true for any normalised design: adding an
effect ``tau`` to the treated units' outcomes shifts the contrast mean
``Y_t @ c`` by exactly ``tau`` (because the treated weights sum to one). So the
alternative distribution of the length-``h`` block mean is the null distribution
shifted by ``tau``, and the power at ``tau`` is the share of the (empirical,
moving-block) null whose shifted magnitude clears the two-sided critical value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesignSpec:
    """A design from any estimator, in the common cross-method currency.

    Parameters
    ----------
    method : str
        Originating method, e.g. ``"SYNDES"`` or ``"GEOLIFT"``.
    label : str
        Human-readable design label (shown on the plot).
    contrast : dict
        ``{unit_label: weight}`` for ``w_treated - w_control``; missing units are
        zero. Treated weights are positive and sum to one, control weights
        negative and sum to minus one.
    treated : list
        Treated-unit labels (used for the baseline and the label).
    """

    method: str
    label: str
    contrast: Dict[Any, float]
    treated: List[Any]


def simulated_mde(
    pre_contrast: np.ndarray,
    *,
    horizon: int,
    effects_abs: Sequence[float],
    alpha: float = 0.10,
    power_target: float = 0.80,
) -> float:
    """Minimum detectable effect from a contrast's moving-block permutation null.

    Parameters
    ----------
    pre_contrast : np.ndarray
        The pre-period contrast series ``g_t = Y_t @ c``, shape ``(T0,)``.
    horizon : int
        Post-period horizon ``h`` the test averages over.
    effects_abs : sequence of float
        Ascending grid of candidate effect sizes, in outcome units.
    alpha, power_target : float
        Two-sided significance level and target power.

    Returns
    -------
    float
        Smallest grid effect reaching ``power_target``, or ``inf`` if none does.
    """
    g = np.asarray(pre_contrast, dtype=float)
    g = g - g.mean()                      # center under the sharp null
    T = g.size
    if T < horizon + 1:
        return float("inf")
    # Empirical null of the length-h block mean: overlapping pre-period blocks.
    blocks = np.array([g[i:i + horizon].mean()
                       for i in range(T - horizon + 1)], dtype=float)
    crit = float(np.quantile(np.abs(blocks), 1.0 - alpha))
    for tau in np.sort(np.asarray(effects_abs, dtype=float)):
        # Under an additive effect tau the statistic shifts by tau; power is the
        # share of the shifted null whose magnitude clears the critical value.
        power = float(np.mean(np.abs(blocks + tau) > crit))
        if power >= power_target:
            return float(tau)
    return float("inf")


def _pareto_mask(fit: np.ndarray, mde: np.ndarray) -> np.ndarray:
    """Boolean non-dominated mask on (fit downwards, mde downwards)."""
    fit = np.asarray(fit, dtype=float)
    mde = np.asarray(mde, dtype=float)
    n = fit.size
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (fit[j] <= fit[i] and mde[j] <= mde[i]
                    and (fit[j] < fit[i] or mde[j] < mde[i])):
                keep[i] = False
                break
    return keep


def compare_pareto(
    specs: Sequence[DesignSpec],
    Ywide: pd.DataFrame,
    n_pre: int,
    *,
    horizon: int = 5,
    effects_pct: Optional[Sequence[float]] = None,
    alpha: float = 0.10,
    power_target: float = 0.80,
) -> pd.DataFrame:
    """Score designs from any method(s) on the shared fit-vs-power plane.

    Parameters
    ----------
    specs : sequence of DesignSpec
        Designs to compare (typically from :func:`from_syndes` and
        :func:`from_geolift`, run on the same panel).
    Ywide : pd.DataFrame
        Outcome panel, rows = time, columns = unit labels (the canonical order
        every contrast is aligned to).
    n_pre : int
        Number of leading rows that are pre-treatment.
    horizon : int, default 5
        Post-period horizon for the MDE simulation.
    effects_pct : sequence of float, optional
        Ascending grid of effect sizes as a percent of the treated baseline.
        Defaults to ``0.25 .. 30`` in 0.25 steps -- the common grid both methods
        are simulated over.
    alpha, power_target : float
        Two-sided significance level and target power.

    Returns
    -------
    pd.DataFrame
        One row per design: ``method``, ``label``, ``treated``, ``fit_rmse``,
        ``mde_pct`` (at ``horizon``), and a per-method ``pareto`` flag.
    """
    if effects_pct is None:
        effects_pct = np.arange(0.25, 30.25, 0.25)
    cols = list(Ywide.columns)
    pre = np.asarray(Ywide.values[:n_pre], dtype=float)
    effects_pct = np.asarray(effects_pct, dtype=float)

    rows: List[Dict[str, Any]] = []
    for s in specs:
        c = np.array([s.contrast.get(u, 0.0) for u in cols], dtype=float)
        g = pre @ c
        fit_rmse = float(np.sqrt(np.mean(g ** 2)))
        treated_pos = [cols.index(u) for u in s.treated if u in cols]
        base = float(np.mean(pre[:, treated_pos])) if treated_pos else float("nan")
        if not np.isfinite(base) or abs(base) < 1e-12:
            mde_pct = float("inf")
        else:
            eff_abs = (effects_pct / 100.0) * abs(base)
            mde_abs = simulated_mde(g, horizon=horizon, effects_abs=eff_abs,
                                    alpha=alpha, power_target=power_target)
            mde_pct = (100.0 * mde_abs / abs(base)
                       if np.isfinite(mde_abs) else float("inf"))
        rows.append({
            "method": s.method, "label": s.label, "treated": list(s.treated),
            "fit_rmse": fit_rmse, "mde_pct": mde_pct,
        })

    df = pd.DataFrame(rows)
    df["pareto"] = False
    for _, idx in df.groupby("method").groups.items():
        sub = df.loc[idx]
        mask = _pareto_mask(sub["fit_rmse"].to_numpy(),
                            np.where(np.isfinite(sub["mde_pct"]),
                                     sub["mde_pct"], np.inf))
        df.loc[idx, "pareto"] = mask
    return df


def from_syndes(res) -> List[DesignSpec]:
    """Build DesignSpecs from a fitted SYNDES result (its solution pool).

    Falls back to the single returned design when no pool exists (``top_K == 1``).
    """
    labels = list(np.asarray(res.inputs.unit_index.labels))

    def _spec(design, markets, tag):
        c = np.asarray(design.contrast_weights, dtype=float).reshape(-1)
        contrast = {labels[j]: float(c[j]) for j in range(len(labels))}
        return DesignSpec("SYNDES", tag, contrast, list(markets))

    pool = getattr(res, "pool", None)
    if pool:
        return [
            _spec(e["design"], e["markets"],
                  f"S{i + 1}:{'+'.join(map(str, sorted(e['markets'])))}")
            for i, e in enumerate(pool)
        ]
    d = res.design
    return [_spec(d, list(np.asarray(d.selected_unit_labels)), "S1")]


def from_geolift(res) -> List[DesignSpec]:
    """Build DesignSpecs from a fitted GEOLIFT result (its candidate designs).

    The treated side is the equal-weighted (``1/K``) average of the selected
    markets, matching SYNDES's normalised treated weights, so both methods'
    contrasts share the ``sum(w_treated) = 1`` convention.
    """
    specs: List[DesignSpec] = []
    for cand in res.search.candidates:
        treated = sorted(cand.candidate)
        K = len(treated)
        contrast: Dict[Any, float] = {u: 1.0 / K for u in treated}
        for u, w in cand.weights.donor_weights.items():
            contrast[u] = contrast.get(u, 0.0) - float(w)
        specs.append(DesignSpec(
            "GEOLIFT", f"G:{'+'.join(map(str, treated))}", contrast, treated,
        ))
    return specs


def plot_compare_pareto(frame: pd.DataFrame, ax=None):
    """Overlay each method's fit-vs-power Pareto frontier on shared axes.

    Renders in the in-house mlsynth style. Returns the axis drawn on.
    """
    import matplotlib.pyplot as plt

    from .plotting import mlsynth_style

    _METHOD_COLORS = {"SYNDES": "blue", "GEOLIFT": "red"}
    _palette = ("green", "purple", "orange", "brown")

    def _draw(ax):
        for k, (method, sub) in enumerate(frame.groupby("method")):
            color = _METHOD_COLORS.get(method, _palette[k % len(_palette)])
            fit = sub["fit_rmse"].to_numpy()
            mde = np.where(np.isfinite(sub["mde_pct"]), sub["mde_pct"], np.nan)
            par = sub["pareto"].to_numpy()
            ax.scatter(fit[~par], mde[~par], color=color, alpha=0.35, s=45,
                       zorder=2)
            order = np.argsort(fit[par])
            ax.plot(fit[par][order], mde[par][order], "-o", color=color,
                    label=f"{method} frontier", zorder=3)
        ax.set_xlabel("pre-period RMSE (treated vs weighted control)")
        ax.set_ylabel("simulated MDE %  (lower is better)")
        ax.set_title("GEOLIFT vs SYNDES: fit vs power at a common horizon")
        ax.legend()
        return ax

    if ax is not None:
        return _draw(ax)
    with mlsynth_style():
        _, ax = plt.subplots(figsize=(8, 5))
        _draw(ax)
        ax.figure.tight_layout()
        plt.show()
    return ax
