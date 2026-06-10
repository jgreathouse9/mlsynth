"""Sequential Synthetic Difference-in-Differences (Sequential SDiD) estimator.

Implements:

    Arkhangelsky, D., & Samkov, A. (2025). "Sequential Synthetic Difference
    in Differences." arXiv:2404.00164v2.

The estimator targets event-study designs with staggered treatment adoption
and remains robust when the parallel-trends assumption is violated by
interactive fixed effects. It operates on cohort-level aggregates rather
than unit-level data, sequentially imputes treated outcomes with their
estimated counterfactuals, and uses unconstrained-sum weights with a
population-share-scaled L2 penalty.

Output is a typed :class:`mlsynth.utils.seq_sdid_helpers.structures.SeqSDIDResults`
container exposing:

    * ``cohort_effects``      cohort-by-horizon point estimates
                              ``tau_hat_{a, k}^SSDiD``
    * ``event_study``         the pooled horizon-k effects
                              ``tau_hat_k^SSDiD(mu)`` with bootstrap CIs
    * ``inference``           bootstrap configuration summary
    * ``raw_event_study``     the non-bootstrap point-estimate vector
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    SequentialSDIDConfig,
    TimeSeriesResults,
    WeightsResults,
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.seq_sdid_helpers.algorithm import (
    pooled_event_study,
    run_sequential_sdid,
)
from ..utils.seq_sdid_helpers.inference import (
    bayesian_bootstrap_event_study,
    wald_intervals,
)
from ..utils.seq_sdid_helpers.plotter import plot_seq_sdid
from ..utils.seq_sdid_helpers.setup import prepare_seq_sdid_inputs
from ..utils.seq_sdid_helpers.structures import (
    SeqSDIDEventStudy,
    SeqSDIDInference,
    SeqSDIDResults,
)

# A very large eta value used as a numerical surrogate for the analytical
# eta -> infinity limit. With this value the optimizer's quadratic penalty
# dominates the fit term so the solution lies essentially on the kernel of
# the constraint, which gives the Remark 2.2 stacked-DiD weights.
_LARGE_ETA: float = 1e12


class SequentialSDID:
    """Sequential Synthetic Difference-in-Differences estimator.

    Parameters
    ----------
    config : SequentialSDIDConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.SequentialSDIDConfig`.

    Returns
    -------
    SeqSDIDResults
        Typed container with cohort-by-horizon effects, the pooled
        event-study trajectory, and Bayesian-bootstrap SE / CI.

    Notes
    -----
    The two-way fixed-effects representation underlying canonical SDiD
    requires parallel trends; Sequential SDiD relaxes this by modelling
    interactive fixed effects directly. The theoretical guarantees in the
    paper require that adoption cohorts be relatively large; on
    single-treated-unit panels the algorithm still runs but the formal
    efficiency results don't apply.

    Each treated cohort's counterfactual is balanced against the cohorts that
    adopt *after* it (plus any never-treated cohort). A cohort therefore needs
    at least two such donor cohorts to balance even a rank-one interactive
    fixed effect -- with a single donor its effect collapses to an unbalanced
    DiD and is biased under interactive fixed effects, a bias that also
    cascades backward through the sequential imputation. The latest cohorts are
    the most exposed; ``fit()`` emits a :class:`UserWarning` naming any
    donor-starved cohort and the largest ``a_max`` that keeps every estimated
    cohort balanced.

    References
    ----------
    Arkhangelsky, D., & Samkov, A. (2025). "Sequential Synthetic Difference
    in Differences." arXiv:2404.00164v2.

    Examples
    --------
    >>> import pandas as pd                                                  # doctest: +SKIP
    >>> from mlsynth import SequentialSDID                                   # doctest: +SKIP
    >>> df = pd.read_csv("...")                                              # doctest: +SKIP
    >>> res = SequentialSDID({                                               # doctest: +SKIP
    ...     "df": df, "outcome": "y", "treat": "treated",
    ...     "unitid": "unit", "time": "year",
    ...     "n_bootstrap": 200, "eta": 0.0, "display_graphs": False,
    ... }).fit()
    >>> res.event_study.tau                                                  # doctest: +SKIP
    """

    def __init__(self, config: Union[SequentialSDIDConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SequentialSDIDConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SequentialSDID configuration: {exc}"
                ) from exc

        self.config: SequentialSDIDConfig = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.eta: float = config.eta
        self.mode: str = config.mode
        self.K = config.K
        self.a_min = config.a_min
        self.a_max = config.a_max
        self.n_bootstrap: int = config.n_bootstrap
        self.alpha: float = config.alpha
        self.seed: int = config.seed

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SeqSDIDResults:
        """Run Algorithm 1 + bootstrap inference and return the typed result."""

        try:
            inputs = prepare_seq_sdid_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                a_min=self.a_min,
                a_max=self.a_max,
                K=self.K,
            )

            effective_eta = (
                _LARGE_ETA if self.mode == "sdid_imputation" else float(self.eta)
            )

            _, cohort_effects = run_sequential_sdid(
                Y_agg=inputs.Y_agg,
                pi=inputs.pi,
                cohort_periods=inputs.cohort_periods,
                treated_cohort_indices=inputs.treated_cohort_indices,
                a_min=inputs.a_min,
                a_max=inputs.a_max,
                K=inputs.K,
                eta=effective_eta,
            )

            tau_hat = pooled_event_study(
                cohort_effects=cohort_effects,
                pi=inputs.pi,
                cohort_periods=inputs.cohort_periods,
                a_min=inputs.a_min,
                a_max=inputs.a_max,
                K=inputs.K,
            )

            if self.n_bootstrap > 0:
                bootstrap_draws = bayesian_bootstrap_event_study(
                    df=self.df,
                    outcome=self.outcome,
                    treat=self.treat,
                    unitid=self.unitid,
                    time=self.time,
                    inputs=inputs,
                    eta=effective_eta,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
                se, ci = wald_intervals(tau_hat, bootstrap_draws, alpha=self.alpha)
            else:
                bootstrap_draws = np.zeros((0, inputs.K + 1))
                se = np.full(inputs.K + 1, np.nan)
                ci = np.full((inputs.K + 1, 2), np.nan)

            event_study = SeqSDIDEventStudy(
                horizons=np.arange(inputs.K + 1),
                tau=tau_hat,
                se=se,
                ci=ci,
                bootstrap_draws=bootstrap_draws,
                alpha=self.alpha,
            )
            inference = SeqSDIDInference(
                n_bootstrap=self.n_bootstrap,
                method="bayesian_bootstrap",
                seed=self.seed,
            )
            horizons = np.arange(inputs.K + 1)
            tau_arr = np.asarray(tau_hat, dtype=float)
            att = float(np.nanmean(tau_arr)) if tau_arr.size else float("nan")
            # Standardized event-time series: gap = pooled horizon effect,
            # counterfactual = no-effect baseline (0), observed = effect.
            cf = np.zeros_like(tau_arr)
            results = SeqSDIDResults(
                inputs=inputs,
                cohort_effects=cohort_effects,
                event_study=event_study,
                inference_detail=inference,
                eta=effective_eta,
                mode=self.mode,
                raw_event_study=tau_hat.copy(),
                effects=EffectsResults(att=None if np.isnan(att) else att),
                time_series=TimeSeriesResults(
                    observed_outcome=tau_arr,
                    counterfactual_outcome=cf,
                    estimated_gap=tau_arr,
                    time_periods=horizons,
                    intervention_time=0,
                ),
                weights=WeightsResults(
                    summary_stats={"constraint": "SSDiD unit + time weights "
                                   "(per cohort x horizon)"}),
                fit_diagnostics=FitDiagnosticsResults(),
                inference=InferenceResults(
                    method=inference.method, details=inference),
                method_details=MethodDetailsResults(
                    method_name=f"SequentialSDID ({self.mode})", is_recommended=True),
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"Sequential SDiD estimation failed: {exc}"
            ) from exc

        if self.display_graphs:
            try:
                plot_seq_sdid(results, save=self.save)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"Sequential SDiD plotting failed: {exc}"
                ) from exc

        return results
