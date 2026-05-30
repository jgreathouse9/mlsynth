"""Treated-vs-counterfactual plot for MUSC.

Renders the treated unit's observed outcome alongside the MUSC (and
optionally SC) counterfactual, with a vertical line at the
intervention period -- the same shape Bottmer et al. (2024) show as
their Figure 3 California-smoking comparison.
"""

from __future__ import annotations

from typing import Optional, Union

from ...exceptions import MlsynthPlottingError
from .structures import MUSC, SC, MUSCResults


def plot_musc(
    results: MUSCResults,
    *,
    outcome: str = "outcome",
    time: str = "time",
    treated_color: str = "black",
    counterfactual_color: str = "tab:red",
    show_sc_baseline: bool = True,
    save: Union[bool, str, dict] = False,
) -> None:
    """Plot the treated outcome against the MUSC counterfactual.

    Parameters
    ----------
    results : MUSCResults
        Fitted MUSC results.
    outcome, time : str
        Axis labels.
    treated_color : str
        Colour of the treated outcome line.
    counterfactual_color : str
        Colour of the MUSC counterfactual; the SC baseline is drawn
        with the same hue at reduced opacity when
        ``show_sc_baseline`` is True.
    show_sc_baseline : bool, default True
        Overlay the standard-SC counterfactual for comparison.
    save : bool, str, or dict
        Falsy disables saving. A truthy value saves to ``save`` (if a
        path) or to a default path; ``dict`` is forwarded as
        ``Figure.savefig`` kwargs.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                                  # noqa: BLE001
        raise MlsynthPlottingError(
            "matplotlib is required for MUSC plotting; install it or "
            "set display_graphs=False."
        ) from exc

    inputs = results.inputs
    times = inputs.time_index.labels
    y_treated = inputs.Y[inputs.treated_idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, y_treated, color=treated_color, linewidth=2.4,
             label=f"Treated ({inputs.treated_label})")
    ax.plot(times, results.fits[MUSC].counterfactual,
             color=counterfactual_color, linewidth=2.0,
             label="MUSC counterfactual")
    if show_sc_baseline and SC in results.fits:
        ax.plot(times, results.fits[SC].counterfactual,
                 color=counterfactual_color, linewidth=1.4, alpha=0.45,
                 linestyle="--", label="SC counterfactual")
    intervention = times[inputs.T0]
    ax.axvline(intervention, color="grey", linestyle=":", linewidth=1.2)
    ax.set_xlabel(time)
    ax.set_ylabel(outcome)
    ax.legend(loc="best", frameon=False)
    ax.set_title(
        f"MUSC: ATT = {results.att:+.3f}  "
        f"(95% CI {results.att_ci[0]:+.3f}, {results.att_ci[1]:+.3f})"
    )

    if save:
        if isinstance(save, dict):
            fig.savefig(**save)
        else:
            fname = save if isinstance(save, str) else "musc_plot.png"
            fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.show()
