"""DTWSC: Dynamic Synthetic Control with speed warping (Cao & Chadefaux 2025).

Cao, J. & Chadefaux, T. (2025). *"Dynamic Synthetic Controls: Accounting for
Varying Speeds in Comparative Case Studies."* Political Analysis 33:18-31.

Classical synthetic control lines donors up with the treated unit period by
period, which quietly assumes every unit responds to a common shock at the same
rate. Regions, countries and firms rarely do. When a donor runs slower, its
boom arrives two years late, and matching it to the treated unit's calendar
compares a peak against a trough -- so the pre-treatment fit degrades and the
estimated effect absorbs the mismatch.

DTWSC removes that mismatch before fitting anything. Dynamic time warping
learns, from the pre-treatment window alone, how much of the treated unit's
clock each donor period accounts for; the donor series is then resampled onto
the treated unit's clock. A second pass carries those speeds past the
intervention by matching each post-treatment window to the pre-treatment
stretch it most resembles -- so the post-treatment outcome never sets its own
speed. Speed differences the donor brought with it are removed; any change in
speed the intervention caused survives, and shows up in the effect.

The synthetic control itself is then ordinary: simplex weights fit to the
warped donors over the pre-period.

Note: unrelated to :class:`mlsynth.DSC` (Gunsilius's *distributional* synthetic
control) and to :class:`mlsynth.DSCAR` (dynamic SC for auto-regressive
processes), which share the "dynamic"/"DSC" naming but are different methods.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import DTWSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.dtwsc_helpers.pipeline import run_dtwsc
from ..utils.dtwsc_helpers.plotter import plot_dtwsc
from ..utils.dtwsc_helpers.setup import prepare_dtwsc_inputs
from ..utils.dtwsc_helpers.structures import DTWSCResults


class DTWSC:
    """Dynamic Synthetic Control estimator.

    Parameters
    ----------
    config : DTWSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.DTWSCConfig`.

    Returns
    -------
    DTWSCResults
        Typed container with the ATT, the warped-donor counterfactual path,
        the simplex donor weights, and the learned per-donor speeds.

    Examples
    --------
    >>> from mlsynth import DTWSC
    >>> res = DTWSC({"df": panel, "outcome": "gdp", "treat": "treated",
    ...              "unitid": "region", "time": "year",
    ...              "display_graphs": False}).fit()   # doctest: +SKIP
    >>> res.att                                        # doctest: +SKIP
    """

    def __init__(self, config: Union[DTWSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = DTWSCConfig(**config)
            except MlsynthConfigError:
                raise
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid DTWSC configuration: {exc}"
                ) from exc
        self.config: DTWSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> DTWSCResults:
        """Warp the donors, fit the synthetic control, and return the results."""
        try:
            inputs = prepare_dtwsc_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
                covariates=self.config.covariates,
            )
            cfg = self.config
            results = run_dtwsc(
                inputs,
                k=cfg.k, warp=cfg.warp, smooth=cfg.smooth,
                filter_width=cfg.filter_width, poly_order=cfg.poly_order,
                buffer=cfg.buffer, n_burn=cfg.n_burn, ma=cfg.ma,
                default_margin=cfg.default_margin, n_q=cfg.n_q, n_r=cfg.n_r,
                dist_quant=cfg.dist_quant, n_iqr=cfg.n_iqr,
                step_pattern1=cfg.step_pattern1,
                step_pattern2=cfg.step_pattern2,
                match_method=cfg.match_method,
                inference=cfg.inference, placebo_pairs=cfg.placebo_pairs,
                alpha=cfg.alpha, mse_window=cfg.mse_window,
                sc_backend=cfg.sc_backend, covariates=cfg.covariates,
                covariate_windows=cfg.covariate_windows,
                fit_window=cfg.fit_window,
            )
            pc = cfg.resolved_plot()
            if pc.xlabel is None:
                pc.xlabel = self.time
            if pc.ylabel is None:
                pc.ylabel = self.outcome
            object.__setattr__(results, "plot_config", pc)
            if self.display_graphs:
                plot_dtwsc(results, pc)
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError,
                MlsynthPlottingError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"DTWSC estimation failed: {exc}") from exc
