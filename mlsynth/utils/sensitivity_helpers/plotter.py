"""Specification curves that retain failed and warning-bearing attempts."""

import matplotlib.pyplot as plt
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
