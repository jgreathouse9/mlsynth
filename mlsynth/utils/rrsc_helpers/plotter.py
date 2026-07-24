"""Plotting for the RRSC estimator: the treated unit's observed-vs-counterfactual
path and the per-unit interference map (a forest plot of direct/interference
effects with confidence intervals)."""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthPlottingError


def plot_rrsc(results, show: bool = True):
    """Draw (1) observed vs counterfactual for the treated unit and (2) the
    interference forest plot. Returns the Matplotlib figure."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:                               # pragma: no cover
        raise MlsynthPlottingError(f"matplotlib unavailable: {exc}") from exc

    inp = results.inputs
    treated = str(inp.treated_name)
    t = np.asarray(inp.time_labels)
    T0 = inp.T0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # (1) observed vs counterfactual (treated)
    ax = axes[0]
    obs = np.asarray(inp.treated_outcome, dtype=float)
    cf = np.asarray(results.counterfactual_panel[0], dtype=float)
    if T0 < len(t):
        ax.axvspan(t[T0], t[-1], color="0.92", zorder=0)
        ax.axvline(t[T0], color="0.5", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot(t, obs, color="#c1121f", lw=2.0, label="observed", zorder=3)
    ax.plot(t, cf, color="0.25", lw=1.8, ls=(0, (5, 2)),
            label=r"counterfactual $\hat\mu_i+\hat\lambda_i^\top\hat f_t$", zorder=3)
    comm = results.communality.get(treated, float("nan"))
    ax.set_title(f"{treated}: direct effect {results.direct_effect:+.1f}"
                 f"  (communality {comm:.2f})", fontsize=10)
    ax.set_xlabel("time"); ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="best")

    # (2) interference forest plot
    ax = axes[1]
    rows = sorted(results.interference_map, key=lambda r: r["effect"])
    y = np.arange(len(rows))
    level = results.metadata.get("alpha", 0.05)
    for i, row in enumerate(rows):
        sig = (row["pvalue"] is not None and np.isfinite(row["pvalue"])
               and row["pvalue"] < level)
        col = "#1d4e89" if row["unit"] == treated else ("#c1121f" if sig else "0.6")
        lo, hi = row["ci_lower"], row["ci_upper"]
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], color=col, lw=1.4, zorder=2)
        ax.plot(row["effect"], i, "o", color=col, ms=5, zorder=3)
    ax.axvline(0, color="0.5", lw=1, ls=(0, (4, 3)))
    ax.set_yticks(y); ax.set_yticklabels([r["unit"] for r in rows], fontsize=7)
    ax.set_xlabel("effect (direct / interference)")
    ax.set_title(f"Interference map ({results.regime}, r={results.n_factors})", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="x")

    fig.tight_layout()
    if show:                                               # pragma: no cover
        plt.show()
    return fig
