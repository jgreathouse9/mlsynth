"""MOSC: many-outcomes synthetic control.

Implements:

    Wang, Y., Schein, A., Shou, J., & Blei, D. M. "A Many-outcomes Perspective
    on the Synthetic Control Method."

Synthetic control is usually justified by assuming the untreated outcomes follow
a linear factor model. MOSC replaces that with an argument from negative control
outcomes: everything observed before the intervention, and everything observed
for the untreated units, is known in advance to be unaffected by it. If some
latent per-unit variable renders a unit's outcomes conditionally independent, and
enough negative controls are observed to pin that variable down, then adjusting
for it identifies the effect -- with no linearity anywhere.

What that buys in practice is the freedom to choose the likelihood. MOSC fits a
probabilistic factor model to the pre-intervention panel, takes each unit's
latent loadings as the estimated confounding structure, and adjusts for them in a
downstream outcome regression. For a count outcome the model can be gamma-Poisson
where every other factor estimator in this library assumes a Gaussian one.

MOSC has no donor weights. The counterfactual is a regression prediction from the
treated unit's own loadings, borrowing strength across every unit at once, so the
usual synthetic-control question of which donors were used has no answer here --
which the result states instead of leaving blank.

See ``mlsynth.utils.mosc_helpers`` for the algorithmic pieces, and
``benchmarks/reference/mosc_spike/`` for the three places this implementation
departs from the paper on purpose.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import MOSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.datautils import balance
from ..utils.mosc_helpers.pipeline import run_mosc
from ..utils.mosc_helpers.setup import prepare_mosc_inputs
from ..utils.mosc_helpers.structures import MOSCInference, MOSCResults


class MOSC:
    """Many-outcomes synthetic control (Wang, Schein, Shou & Blei).

    Parameters
    ----------
    config : MOSCConfig or dict
        Configuration. See :class:`mlsynth.config_models.MOSCConfig`.

    Returns
    -------
    MOSCResults
        An ``EffectResult`` carrying the posterior-mean counterfactual with a
        credible band, the ATT and its credible interval, the draws of the
        estimated confounding structure, and the diagnostics that say whether the
        factor model's assumptions hold on this panel.

    Examples
    --------
    >>> from mlsynth import MOSC
    >>> cfg = {"df": panel, "outcome": "cases", "treat": "reopened",
    ...        "unitid": "county", "time": "date", "factor_model": "gap"}
    >>> res = MOSC(cfg).fit()                          # doctest: +SKIP
    >>> res.att, res.diagnostics.pearson_dispersion    # doctest: +SKIP
    """

    def __init__(self, config: Union[MOSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MOSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid MOSC configuration: {exc}") from exc
        self.config: MOSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> MOSCResults:
        """Fit the factor model, adjust for its loadings, and report the effect."""
        cfg = self.config
        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_mosc_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat,
                n_factors=cfg.n_factors, factor_model=cfg.factor_model,
                outcome_scale=cfg.outcome_scale)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error preparing MOSC inputs: {exc}") from exc

        posterior, diagnostics = run_mosc(inputs, cfg)

        results = MOSCResults(
            inputs=inputs, posterior=posterior, diagnostics=diagnostics,
            inference_detail=_summarise(inputs, posterior, cfg.ci_alpha))
        if self.display_graphs:
            results.plot()
        return results


def _summarise(inputs, posterior, ci_alpha: float) -> MOSCInference:
    """Posterior summaries, with the effect on equation 43's sign.

    The paper defines the effect as the observed path minus the counterfactual.
    The authors' code computes the reverse, which on their null result is
    invisible and on any real effect returns the wrong sign; this follows the
    equation.
    """
    observed = np.asarray(inputs.y_target, dtype=float)
    pre, total = inputs.pre_periods, inputs.total_periods
    draws = np.asarray(posterior.counterfactual, dtype=float)   # (S, T_post)

    lower_q, upper_q = 100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)
    post_mean = draws.mean(axis=0)

    counterfactual = observed.copy()
    counterfactual[pre:] = post_mean
    lower = observed.copy()
    lower[pre:] = np.percentile(draws, lower_q, axis=0)
    upper = observed.copy()
    upper[pre:] = np.percentile(draws, upper_q, axis=0)

    att_samples = (observed[None, pre:total] - draws).mean(axis=1)
    return MOSCInference(
        counterfactual_mean=counterfactual,
        counterfactual_lower=lower,
        counterfactual_upper=upper,
        att_mean=float(att_samples.mean()),
        att_lower=float(np.percentile(att_samples, lower_q)),
        att_upper=float(np.percentile(att_samples, upper_q)),
        att_samples=att_samples,
        ci_alpha=float(ci_alpha),
    )
