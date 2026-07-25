"""Plotting for the CAST estimator: the per-period aggregate effect with its
confidence band, and the entrywise interval spread across treated units."""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthPlottingError


def plot_cast(results, show: bool = True):
    """Draw (1) the aggregate treated effect per period with its confidence
    band and (2) a scatter of the entrywise effects. Returns the figure."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        raise MlsynthPlottingError(f"matplotlib unavailable: {exc}") from exc

    from scipy.stats import norm

    inp = results.inputs
    periods = list(results.period_effects)
    if not periods:                                # pragma: no cover - guarded upstream
        raise MlsynthPlottingError("CAST: no treated periods to plot.")
    eff = np.array([results.period_effects[p][0] for p in periods], dtype=float)
    se = np.array([results.period_effects[p][1] for p in periods], dtype=float)
    z = norm.ppf(1 - results.metadata.get("alpha", 0.05) / 2)
    x = np.arange(len(periods))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.axhline(0, color="0.5", lw=1, ls=(0, (4, 3)))
    ax.fill_between(x, eff - z * se, eff + z * se, color="#1d4e89", alpha=0.20,
                    label=f"{int((1 - results.metadata.get('alpha', 0.05)) * 100)}% CI")
    ax.plot(x, eff, color="#1d4e89", lw=2.0, marker="o", ms=4, label="aggregate effect")
    ax.set_xticks(x); ax.set_xticklabels([str(p) for p in periods], fontsize=7, rotation=45)
    ax.set_xlabel("period"); ax.set_ylabel("effect")
    ax.set_title(f"CAST: treated-average effect (rank {results.rank})", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    mask = inp.treated_mask
    for j, p in enumerate(inp.time_labels):
        rows = np.where(mask[:, j])[0]
        if rows.size:
            ax.scatter(np.full(rows.size, j), results.effects_panel[rows, j],
                       s=9, alpha=0.55, color="#c1121f", edgecolors="none")
    ax.axhline(0, color="0.5", lw=1, ls=(0, (4, 3)))
    ax.set_xticks(np.arange(len(inp.time_labels)))
    ax.set_xticklabels([str(p) for p in inp.time_labels], fontsize=7, rotation=45)
    ax.set_xlabel("period"); ax.set_ylabel("per-unit effect")
    ax.set_title("Entrywise treatment effects", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    if show:                                       # pragma: no cover
        plt.show()
    return fig
