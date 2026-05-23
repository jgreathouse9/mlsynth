"""CTSC: Continuous-Treatment Synthetic Control (Powell 2022).

Powell, D. (2022). *"Synthetic Control Estimation Beyond Comparative Case
Studies: Does the Minimum Wage Reduce Employment?"* Journal of Business &
Economic Statistics 40(3):1302-1314.

The synthetic control method was built for a single treated unit adopting
a binary policy. CTSC generalises it to settings with **continuous and/or
multi-valued treatments** where there is no clean treated / never-treated
split -- every unit has a time-varying treatment (e.g. every U.S. state
has a minimum wage that changes over time, so the comparative-case-study
synthetic control cannot be applied).

CTSC builds a synthetic control for *every* unit out of the others'
untreated outcomes and jointly estimates a unit-specific treatment-slope
vector :math:`\\alpha_i` together with the synthetic-control weights. The
reported effect is the population-weighted average marginal effect
:math:`\\alpha^{AE} = \\sum_i \\pi_i \\alpha_i`. Because the outcome model
allows interactive fixed effects (factor structure) and unit-specific
slopes, CTSC is consistent where two-way fixed-effects regressions are
badly biased when the treatment is correlated with unobserved factors.

The paper names the estimator "GSC"; mlsynth uses **CTSC** to avoid
collision with Xu (2017)'s differently constructed Generalized Synthetic
Control (``gsynth``).

Implementation note
-------------------

The paper minimises the joint objective (eq. 5) with Nelder-Mead over all
:math:`nK + n(n-1)` parameters. mlsynth exploits the **biconvex**
structure -- weighted linear least squares in the slopes for fixed
weights, and per-unit simplex-constrained least squares in the weights
for fixed slopes -- and solves it by block coordinate descent, which
optimises the same objective far more stably. The bundled simulation
module reproduces the paper's Table 1 (Models 1-4) as a calibration
check (CTSC mean bias :math:`\\approx 0` vs two-way FE bias
:math:`\\approx 0.85`).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import CTSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.ctsc_helpers.pipeline import run_ctsc
from ..utils.ctsc_helpers.setup import prepare_ctsc_inputs
from ..utils.ctsc_helpers.structures import CTSCResults


class CTSC:
    """Continuous-Treatment Synthetic Control estimator.

    Parameters
    ----------
    config : CTSCConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.CTSCConfig`.
    """

    def __init__(self, config: Union[CTSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = CTSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid CTSC configuration: {exc}"
                ) from exc
        self.config: CTSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treatment_vars = config.treatment_vars
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.population_col = config.population_col
        self.use_fit_weights: bool = config.use_fit_weights
        self.inference: bool = config.inference
        self.n_draws: int = config.n_draws
        self.random_state: int = config.random_state

    def fit(self) -> CTSCResults:
        """Run CTSC and return :class:`CTSCResults`."""
        try:
            inputs = prepare_ctsc_inputs(
                df=self.df,
                outcome=self.outcome,
                treatment_vars=self.treatment_vars,
                unitid=self.unitid,
                time=self.time,
                population_col=self.population_col,
            )
            return run_ctsc(
                inputs=inputs,
                use_fit_weights=self.use_fit_weights,
                inference=self.inference,
                n_draws=self.n_draws,
                random_state=self.random_state,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"CTSC estimation failed: {exc}"
            ) from exc
