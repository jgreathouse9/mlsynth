"""Plotting for DMLFM."""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthPlottingError


def plot_dmlfm(results, cfg) -> None:
    """Observed against the posterior-mean counterfactual, with a credible band."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a hard dependency
        raise MlsynthPlottingError(f"matplotlib unavailable: {exc}") from exc

    try:
        ts = results.time_series
        t = np.asarray(ts.time_periods, float).ravel()
        obs = np.asarray(ts.observed_outcome, float).ravel()
        cf = np.asarray(ts.counterfactual_outcome, float).ravel()
        draws = results.additional_outputs["counterfactual_draws"]
        alpha = 1.0 - float(results.inference.confidence_level)
        lo = np.quantile(draws, alpha / 2, axis=1)
        hi = np.quantile(draws, 1 - alpha / 2, axis=1)
        cut = int(results.fit_diagnostics.pre_periods)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(t, lo, hi, color=cfg.counterfactual_color[0], alpha=0.25,
                        label=f"{int(100 * (1 - alpha))}% credible band")
        ax.plot(t, obs, color=cfg.treated_color, lw=2.2, label=cfg.outcome)
        ax.plot(t, cf, color=cfg.counterfactual_color[0], lw=2.0, ls="--",
                label="DMLFM counterfactual")
        if 0 < cut < len(t):
            ax.axvline(t[cut] - 0.5 if len(t) > cut else t[-1], color="0.4",
                       ls=":", lw=1.4)
        ax.set_xlabel(cfg.time)
        ax.set_ylabel(cfg.outcome)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
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
