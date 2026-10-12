"""Exposure-augmented synthetic control (EASC).

A thin orchestration over :mod:`mlsynth.utils.easc_helpers`. Spruk (2026)
studies an observed common shock -- a commodity price, an exchange rate -- that
moves with the latent factors before treatment and departs from them
afterwards. Pre-treatment fit then says almost nothing about the synthetic
unit's exposure to the shock, and the post-treatment gap absorbs the
difference. EASC reports the diagnostics that detect this without looking at
the treated unit's post-treatment outcomes, corrects the synthetic control's
exposure imbalance with a cross-donor regression of outcome changes on a
pre-determined exposure index, and gives bias-aware intervals with a breakdown
value.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..exceptions import MlsynthEstimationError, MlsynthPlottingError
from ..utils.datautils import balance
from ..utils.easc_helpers import EASCConfig, EASCResults, plot_easc, prepare_easc_inputs, run_easc


class EASC:
    """Exposure-augmented synthetic control.

    Parameters
    ----------
    config : EASCConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``outcome``,
        ``treat``, ``unitid``, ``time``, ``display_graphs``, ``save``, colors),
        EASC reads ``exposure`` and ``exposure_window`` (the exposure index),
        ``shock`` (the observed common shock), ``method``, ``n_factors``,
        ``n_subperiods``, ``base_period``, ``frontier_points``, ``phases``,
        ``inference``, ``alpha``, ``m_grid``, ``headline_m`` and
        ``correction_se``.

    Examples
    --------
    >>> from mlsynth import EASC                                    # doctest: +SKIP
    >>> res = EASC({"df": panel, "outcome": "lgdppc", "treat": "sanctions",
    ...             "unitid": "iso", "time": "year", "exposure": "oilrent",
    ...             "exposure_window": (1996, 2011), "shock": "log_brent",
    ...             "display_graphs": False}).fit()                  # doctest: +SKIP
    >>> res.att, res.diagnostics.max_partial_leverage_post          # doctest: +SKIP

    References
    ----------
    Spruk, R. (2026). Synthetic Control under Out-of-Span Common Shocks:
    Diagnosis, Exposure Balance and Correction. arXiv:2610.05535.
    """

    def __init__(self, config: Union[EASCConfig, dict]) -> None:
        if isinstance(config, dict):
            config = EASCConfig(**config)
        self.config = config

    def fit(self) -> EASCResults:
        """Diagnose, correct, and return standardized results.

        Returns
        -------
        EASCResults
            The headline fit in the standardized sub-models, with all four fits,
            the diagnostics, the frontier and the bias-aware inference as typed
            fields.
        """
        cfg = self.config
        balance(cfg.df, cfg.unitid, cfg.time)
        inputs = prepare_easc_inputs(
            cfg.df, unitid=cfg.unitid, time=cfg.time, outcome=cfg.outcome, treat=cfg.treat,
            exposure=cfg.exposure, shock=cfg.shock, exposure_window=cfg.exposure_window,
            base_period=cfg.base_period)
        try:
            est = run_easc(
                inputs, n_factors=cfg.n_factors, n_subperiods=cfg.n_subperiods,
                frontier_points=cfg.frontier_points, phases=cfg.phases, inference=cfg.inference,
                alpha=cfg.alpha, m_grid=cfg.m_grid, headline_m=cfg.headline_m,
                correction_se=cfg.correction_se)
        except np.linalg.LinAlgError as exc:
            raise MlsynthEstimationError(f"EASC: estimation failed: {exc}") from exc

        plot_settings = cfg.resolved_plot()
        results = EASCResults(
            inputs=inputs, method=cfg.method, fits=est.fits, diagnostics=est.diagnostics,
            frontier=est.frontier, bias_aware=est.inference, plot_config=plot_settings,
            metadata={"n_subperiods": cfg.n_subperiods})

        if plot_settings.display:
            try:
                import matplotlib.pyplot as plt

                fig = plot_easc(results)
                if plot_settings.save:
                    name = plot_settings.save if isinstance(plot_settings.save, str) else "easc.png"
                    fig.savefig(name, bbox_inches="tight")
                plt.show()
            except Exception as exc:
                raise MlsynthPlottingError(f"EASC plotting failed: {exc}") from exc
        return results
