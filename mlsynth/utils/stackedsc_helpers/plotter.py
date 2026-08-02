"""Plotting helper for STACKEDSC: the stacked event study.

The per-unit paths are drawn behind the aggregate rather than omitted. With a
donor pool small relative to the pre-window the individual weight vectors are
not identified even where their weighted mean is, so the spread is a property
of the design worth seeing, not clutter to hide.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..plotting import mlsynth_style
from .structures import STACKEDSCResults


def plot_stackedsc(
    results: STACKEDSCResults,
    title: str = "Stacked-in-event-time synthetic control",
    save: Union[bool, str, dict] = False,
    show_units: bool = True,
    max_units: int = 200,
) -> None:
    """Render the weighted event-time effect path.

    Parameters
    ----------
    results : STACKEDSCResults
        A fitted result.
    title : str
        Figure title.
    save : bool or str or dict
        Falsy to skip saving; a string is used as the filename.
    show_units : bool
        Draw the per-unit paths behind the aggregate.
    max_units : int
        Cap on how many per-unit paths are drawn, so a panel with hundreds of
        treated units stays legible. The cap is stated in the legend rather
        than applied silently.
    """
    import matplotlib.pyplot as plt

    es = results.event_study
    e = np.asarray(es.horizons, float)
    tau = np.asarray(es.tau, float)
    units = list(results.per_unit.values())
    unit_label = ("percent of base level" if results.design.normalized
                  else "outcome units")

    with mlsynth_style():
        fig, ax = plt.subplots(figsize=(7.5, 5))
        if show_units and units:
            drawn = units[:max_units]
            for k, u in enumerate(drawn):
                ax.plot(np.asarray(u.horizons, float),
                        np.asarray(u.tau, float),
                        color="#9aa5b1", linewidth=0.6, alpha=0.35,
                        zorder=1,
                        label=(f"per treated unit "
                               f"({len(drawn)} of {len(units)} shown)"
                               if k == 0 else None))
        ax.plot(e, tau, marker="o", color="#1428A0", linewidth=2.0, zorder=3,
                label=(f"weighted mean (ATT "
                       f"{results.effects.att:.3f})"))
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        # The intervention lands between e = -1 and e = 0, so the rule goes
        # between them rather than on a period that is itself observed.
        ax.axvline(-0.5, color="grey", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Event time $e$")
        ax.set_ylabel(rf"$\widehat{{\tau}}_e$  ({unit_label})")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        if save:
            fname = save if isinstance(save, str) else "stackedsc_event_study.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
