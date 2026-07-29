"""Plotting for DTWSC."""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthPlottingError


def plot_dtwsc(results, plot_config=None) -> None:
    """Observed against warped-synthetic path, with the intervention marked."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:                # pragma: no cover - no matplotlib
        raise MlsynthPlottingError(f"DTWSC: plotting unavailable: {exc}") from exc

    try:
        ts = results.time_series
        x = (np.asarray(ts.time_periods) if ts.time_periods is not None
             else np.arange(np.asarray(ts.observed_outcome).size))
        pc = plot_config
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, np.asarray(ts.observed_outcome, dtype=float),
                color=(pc.observed_color if pc else "black"),
                lw=2, label="Observed")
        cf_colors = (pc.counterfactual_colors if pc else None) or ["red"]
        ax.plot(x, np.asarray(ts.counterfactual_outcome, dtype=float),
                color=cf_colors[0], lw=1.6, ls="--",
                label="DTWSC counterfactual")
        if ts.intervention_time is not None:
            ax.axvline(ts.intervention_time, color="grey", ls=":", lw=1.2)
        ax.set_xlabel((pc.xlabel if pc and pc.xlabel else "Time"))
        ax.set_ylabel((pc.ylabel if pc and pc.ylabel else "Outcome"))
        ax.set_title(f"DTWSC -- {results.metadata.get('treated_unit', 'treated')}")
        ax.legend(frameon=False)
        fig.tight_layout()
        if pc is not None and pc.save:
            fig.savefig(pc.save if isinstance(pc.save, str) else "dtwsc.png",
                        dpi=150)
        plt.show()
    except MlsynthPlottingError:            # pragma: no cover - re-raise
        raise
    except Exception as exc:
        raise MlsynthPlottingError(f"DTWSC plotting failed: {exc}") from exc
