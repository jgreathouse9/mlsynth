"""Specification curves that retain failed and warning-bearing attempts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_specification_curve(summary: pd.DataFrame) -> Figure:
    """Return a specification-curve Figure displaying every attempted design."""
    fig, ax = plt.subplots()
    ordered = summary.sort_values("att", kind="stable", na_position="last")
    for i, row in enumerate(ordered.itertuples()):
        if row.status == "failed":
            ax.text(
                i,
                0.02,
                "failed",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                rotation=90,
            )
        else:
            ax.scatter(i, row.att, marker="x" if row.status == "warning" else "o")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(range(len(ordered)), ordered.spec_id, rotation=90)
    ax.set_ylabel("ATT")
    ax.set_xlabel("Specification (x: fit with warnings; failed: no estimate)")
    fig.tight_layout()
    return fig


def plot_validation(summary: pd.DataFrame, limit: float | None) -> Figure:
    """Show effect/prediction comparisons without assigning values to failures."""
    fig, ax = plt.subplots()
    finite = np.isfinite(summary.att) & np.isfinite(summary.validation_rmse)
    for position, row in enumerate(summary.loc[finite].itertuples()):
        warning = row.status == "warning" or row.validation_status == "warning"
        ax.scatter(row.validation_rmse, row.att, marker="x" if warning else "o")
        ax.annotate(
            row.spec_id,
            (row.validation_rmse, row.att),
            xytext=(5, 8 if position % 2 == 0 else 22),
            textcoords="offset points",
        )
    missing = summary.loc[~finite, "spec_id"].tolist()
    if missing:
        ax.text(
            0,
            1.02,
            "ATT or validation unavailable: " + ", ".join(missing),
            transform=ax.transAxes,
            va="bottom",
            wrap=True,
        )
    if limit is not None:
        ax.axvline(limit, color="grey", linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Pre-treatment validation RMSE")
    ax.set_ylabel("ATT")
    fig.tight_layout()
    return fig
