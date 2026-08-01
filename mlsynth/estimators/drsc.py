"""DRSC -- conditional distributional treatment effects (Wied 2026).

Wied, D. (2026). *"A Synthetic Control Approach to Conditional Distributional
Treatment Effects."* arXiv:2606.09625.

Where :class:`mlsynth.DSC` (Gunsilius 2023) matches *unconditional* quantile
functions, DRSC targets the conditional distribution
:math:`F(y \\mid x)`. It models it semiparametrically as
:math:`\\Lambda(x'\\theta(y))` -- a distribution regression -- and imposes
parallel trends on the parameter function :math:`\\theta` rather than on the
CDF. Two things follow from putting the assumption in parameter space: the
counterfactual stays inside the model class, and the weight problem is a
quadratic program with one linear constraint, hence closed-form.

Why condition at all
--------------------

When the outcome depends on individual characteristics -- wages on education
and experience -- an effect concentrated in one part of the covariate space can
be invisible in the marginal distribution. The paper's simulation makes this
concrete: an effect entering along a single mean-zero covariate leaves the
marginal almost unchanged, so an unconditional method has essentially no power
while the conditional test rejects.

Data requirements
-----------------

Micro-level panel data: one row per ``(unit, time, individual)``, with
covariates on every row. Ingestion does not use ``dataprep`` -- that pivots to
one outcome per cell -- and follows :class:`mlsynth.DSC`'s micro-panel setup
instead.

Two properties that surprise people
-----------------------------------

The asymptotics are in the *within-cell* sample size, with the donor count and
the number of periods fixed. One pre-period suffices for consistency; more
pre-periods buy precision in the weights, not in the DR parameters. If your
cells are small and your panel is long, this is the wrong estimator.

Weights are constrained only to sum to one, so negative weights are ordinary
(20 of 42 in the paper's application) and mean the treated unit sits outside
the donor hull. The Gram matrix is correspondingly ill-conditioned, and the
inputs must be float64 -- see :doc:`../drsc` for the measured cost of not
doing that.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..exceptions import (MlsynthConfigError, MlsynthDataError,
                          MlsynthEstimationError)
from ..utils.drsc_helpers.config import DRSCConfig
from ..utils.drsc_helpers.pipeline import run_drsc
from ..utils.drsc_helpers.plotter import plot_drsc
from ..utils.drsc_helpers.setup import prepare_drsc_inputs
from ..utils.drsc_helpers.structures import DRSCResults


class DRSC:
    """Distribution-regression synthetic control.

    Parameters
    ----------
    config : DRSCConfig or dict
        See :class:`mlsynth.utils.drsc_helpers.config.DRSCConfig`.

    Examples
    --------
    >>> from mlsynth import DRSC                              # doctest: +SKIP
    >>> res = DRSC({                                          # doctest: +SKIP
    ...     "df": micro_panel, "outcome": "logwage", "treat": "treat",
    ...     "unitid": "state", "time": "policy_year",
    ...     "covariates": ["educ_std", "exper_std", "exper_std_sq"],
    ...     "evaluation_points": {
    ...         "low_skill_young": {"educ_std": -0.98, "exper_std": -1.17,
    ...                             "exper_std_sq": 1.37}},
    ... }).fit()
    >>> res.conditional_effects["low_skill_young"].p_value   # doctest: +SKIP
    """

    def __init__(self, config: Union[DRSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = DRSCConfig(**config)
            except MlsynthConfigError:
                raise
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid DRSC configuration: {exc}") from exc
        self.config: DRSCConfig = config

    def fit(self) -> DRSCResults:
        """Estimate the conditional distributional treatment effect."""
        cfg = self.config
        try:
            inputs = prepare_drsc_inputs(
                df=cfg.df, outcome=cfg.outcome, treat=cfg.treat,
                unitid=cfg.unitid, time=cfg.time, covariates=cfg.covariates)
            results = run_drsc(
                inputs=inputs, evaluation_points=cfg.evaluation_points,
                link=cfg.link, n_grid=cfg.n_grid, grid_lo=cfg.grid_lo,
                grid_hi=cfg.grid_hi, focus_region=cfg.focus_region,
                ridge=cfg.ridge, alpha=cfg.alpha, n_gp_draws=cfg.n_gp_draws,
                seed=cfg.seed)
            if cfg.display_graphs:
                plot_drsc(results, outcome_label=cfg.outcome,
                          focus_region=cfg.focus_region, save=cfg.save)
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - translation guard
            raise MlsynthEstimationError(f"DRSC estimation failed: {exc}") from exc
