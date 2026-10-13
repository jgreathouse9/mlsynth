"""Plotting for DMLFM."""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthPlottingError


def plot_dmlfm(results, cfg) -> None:
    """Observed against the posterior-mean counterfactual, with a credible band.

    One panel per treated unit, each with its own adoption line, so a staggered
    fit reads as a column of event plots.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a hard dependency
        raise MlsynthPlottingError(f"matplotlib unavailable: {exc}") from exc

    try:
        ts = results.time_series
        t = np.asarray(ts.time_periods, float).ravel()
        obs = np.atleast_2d(np.asarray(ts.observed_outcome, float).T)   # (n_tr, T)
        cf = np.atleast_2d(np.asarray(ts.counterfactual_outcome, float).T)
        draws = results.additional_outputs["counterfactual_draws"]      # (n_tr, T, ndraw)
        adoption = np.asarray(results.additional_outputs["adoption_index"], int)
        names = results.method_details.parameters["treated_units"]
        alpha = 1.0 - float(results.inference.confidence_level)
        lo = np.quantile(draws, alpha / 2, axis=2)
        hi = np.quantile(draws, 1 - alpha / 2, axis=2)

        n_tr = obs.shape[0]
        fig, axes = plt.subplots(n_tr, 1, figsize=(8, 5 if n_tr == 1 else 3.2 * n_tr),
                                 squeeze=False, sharex=True)
        for k in range(n_tr):
            ax = axes[k, 0]
            ax.fill_between(t, lo[k], hi[k], color=cfg.counterfactual_color[0],
                            alpha=0.25,
                            label=f"{int(100 * (1 - alpha))}% credible band")
            ax.plot(t, obs[k], color=cfg.treated_color, lw=2.2, label=cfg.outcome)
            ax.plot(t, cf[k], color=cfg.counterfactual_color[0], lw=2.0, ls="--",
                    label="DMLFM counterfactual")
            cut = int(adoption[k])
            if 0 < cut < len(t):
                ax.axvline(t[cut] - 0.5, color="0.4", ls=":", lw=1.4)
            ax.set_ylabel(cfg.outcome)
            ax.grid(alpha=0.25)
            if n_tr > 1:
                ax.set_title(str(names[k]), fontsize=10, loc="left")
            if k == 0:
                ax.legend(fontsize=9)
        axes[-1, 0].set_xlabel(cfg.time)
        fig.tight_layout()
        if cfg.save:
            fig.savefig(cfg.save if isinstance(cfg.save, str) else "dmlfm.png", dpi=150)
        else:
            plt.show()
        plt.close(fig)
    except MlsynthPlottingError:
        raise
    except Exception as exc:
        raise MlsynthPlottingError(f"DMLFM plotting failed: {exc}") from exc
