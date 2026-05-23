"""ISCM: Imperfect Synthetic Controls (Powell 2026).

Powell, D. (2026). *"Imperfect Synthetic Controls."* Journal of Applied
Econometrics 41(3):253-264.

The synthetic control method assumes a *perfect* synthetic control
exists -- the treated unit lies inside the convex hull of the donors and
its pre-period path is matched exactly. With transitory shocks this is
implausible: an exact fit cannot hold even in expectation. ISCM relaxes
the assumption by constructing synthetic controls for **every** unit and
identifying the treatment effect even when the treated unit is *outside*
the convex hull. The intuition (paper eq. 6): a treated unit that fits no
donor combination can still appear *as a donor* for control units, and
those units' post-treatment residuals then carry information about the
effect.

ISCM also introduces a data-driven fit metric :math:`a_i` that
asymptotically excludes units lacking a valid synthetic control --
removing the researcher's eyeball judgment of pre-period fit -- and an
Ibragimov-Muller inference procedure that remains valid with a very
small donor pool, where permutation tests cannot reach standard
significance thresholds.

This implementation follows Powell's applied procedure: synthetic
controls for all units are obtained by the traditional SCM, the
:math:`a_i` fit weights are formed from the pre-period moment conditions,
the ATT is the :math:`a_i`-weighted least-squares effect (eq. 8 / 15),
and inference is the sign-flip randomization test of eq. 16. It does not
run the optional continuously-updating GMM refinement of the weights
(paper Section 3.2-3.4); the SCM-initialised weights are the documented
starting point of that procedure.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import ISCMConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.iscm_helpers.pipeline import run_iscm
from ..utils.iscm_helpers.plotter import plot_iscm
from ..utils.iscm_helpers.setup import prepare_iscm_inputs
from ..utils.iscm_helpers.structures import ISCMResults


class ISCM:
    """Imperfect Synthetic Controls estimator.

    Parameters
    ----------
    config : ISCMConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.ISCMConfig`.
    """

    def __init__(self, config: Union[ISCMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = ISCMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid ISCM configuration: {exc}"
                ) from exc
        self.config: ISCMConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.inference: bool = config.inference
        self.null_value: float = config.null_value
        self.alpha: float = config.alpha
        self.n_draws: int = config.n_draws
        self.random_state: int = config.random_state
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> ISCMResults:
        """Run ISCM and return :class:`ISCMResults`."""
        try:
            inputs = prepare_iscm_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
            )
            results = run_iscm(
                inputs=inputs,
                inference=self.inference,
                null_value=self.null_value,
                alpha_level=self.alpha,
                n_draws=self.n_draws,
                random_state=self.random_state,
            )
            if self.display_graphs:
                cf_color = self.counterfactual_color
                if isinstance(cf_color, (list, tuple)):
                    cf_color = cf_color[0] if cf_color else "red"
                plot_iscm(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=cf_color,
                    save=self.save,
                    unit_label=self.unitid,
                    effect_label=f"Treatment effect on {self.outcome}",
                )
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"ISCM estimation failed: {exc}"
            ) from exc
