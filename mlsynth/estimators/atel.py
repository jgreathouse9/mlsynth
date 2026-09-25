"""Average Treatment Effect Localization (ATEL).

Lee, R.-C. (2026). *"Average Treatment Effect Localization: Projection Methods
in Synthetic Control."* Econometric Theory.

ATEL answers a question the ATT does not. A policy evaluated over a long
post-period mixes the policy's effect with whatever the treated unit did in
response to it, so an average over the whole post-period can describe the
adaptation as much as the intervention. ATEL instead reports

.. math::

   \\alpha = \\frac{1}{T_1} \\sum_{t > T_0}
            \\bigl( Y^I_{1t} - Y^N_{1t} \\bigr)\\,
            K_h\\!\\left( \\frac{t - T_0}{T_1} \\right),

a kernel-weighted average localized at the adoption date, so early post-periods
count for more than late ones.

The counterfactual comes from a factor model whose loadings vary with time:

1. A sieve basis in the observed covariates gives diversified weights.
2. The factors are cross-sectional donor averages against those weights (Fan and
   Liao 2022), so no eigendecomposition is taken and the donor panel's own
   spectrum does not enter.
3. The treated unit's loading is fit by local linear regression on the
   pre-period, with a one-sided kernel at the boundary, and carried into the
   post-period.
4. The post-period gap is averaged against the localization kernel, with
   Theorem 1's variance for the interval.

Where this sits next to :class:`~mlsynth.FMA`: FMA takes its factors by
principal components and holds the loading fixed over time, which recovers a
time-average of a loading that moves. ATEL estimates the movement.

Two parameters are required and not selected from the data. The factor count is
required because the reference implementation's information criterion returns an
endpoint of its candidate grid on the paper's own panel. The bandwidth may be
cross-validated, and the estimator warns when that search also lands on an
endpoint of its grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config_models import (
    ATELConfig,
    EffectsResults,
    InferenceResults,
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.atel_helpers.pipeline import run_atel
from ..utils.atel_helpers.plotter import plot_atel
from ..utils.atel_helpers.setup import prepare_atel_inputs
from ..utils.atel_helpers.structures import ATELResults
from ..utils.results_helpers import build_effect_submodels, make_weights_results

__all__ = ["ATEL"]


class ATEL:
    """Average Treatment Effect Localization for a single treated unit.

    Parameters
    ----------
    config : ATELConfig
        Validated configuration. See :class:`~mlsynth.config_models.ATELConfig`
        for the parameters, in particular why ``n_factors`` is required and why
        it must be a multiple of the covariate count.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth import ATEL
    >>> from mlsynth.config_models import ATELConfig
    >>> df = pd.read_csv("basedata/fdi_oecd_brexit.csv")   # doctest: +SKIP
    >>> config = ATELConfig(                               # doctest: +SKIP
    ...     df=df, outcome="fdi", treat="treated",
    ...     unitid="country", time="year",
    ...     covariates=["log_gdp", "log_gdp_percap"],
    ...     n_factors=2, display_graphs=False,
    ... )
    >>> results = ATEL(config).fit()                       # doctest: +SKIP
    >>> results.atel                                       # doctest: +SKIP
    """

    def __init__(self, config: ATELConfig) -> None:
        if isinstance(config, dict):
            config = ATELConfig(**config)
        if not isinstance(config, ATELConfig):
            raise MlsynthConfigError(
                f"ATEL needs an ATELConfig; got {type(config).__name__}."
            )
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.covariates = list(config.covariates)
        self.n_factors: int = int(config.n_factors)
        self.basis: str = config.basis
        self.bandwidth = config.bandwidth
        self.alpha: float = float(config.alpha)

    def fit(self) -> ATELResults:
        """Run the ATEL pipeline end to end.

        Returns
        -------
        ATELResults

        Raises
        ------
        MlsynthDataError
            If the panel is unbalanced, a covariate has a missing cell, the
            pre-period is shorter than twice the factor count, or the
            localization window is empty.
        MlsynthEstimationError
            If a consumed weight block is identically zero.
        """
        inputs = prepare_atel_inputs(
            df=self.df,
            outcome=self.outcome,
            treat=self.treat,
            unitid=self.unitid,
            time=self.time,
            covariates=self.covariates,
            n_factors=self.n_factors,
        )

        estimates, diagnostics = run_atel(
            inputs,
            n_factors=self.n_factors,
            basis=self.basis,
            bandwidth=self.bandwidth,
            alpha=self.alpha,
        )

        inference = InferenceResults(
            p_value=estimates["p_value"],
            ci_lower=estimates["ci"][0],
            ci_upper=estimates["ci"][1],
            standard_error=estimates["standard_error"],
            confidence_level=1.0 - self.alpha,
            method="asymptotic (Theorem 1), for the localized estimate",
            details={
                "estimand": "ATEL",
                "bandwidth": estimates["bandwidth"],
                "post_window": diagnostics["post_window"],
            },
        )

        donor_labels = list(inputs.unit_labels[1:])
        weights_container = make_weights_results(
            {
                name: float(w)
                for name, w in zip(donor_labels, estimates["localized_donor_weights"])
            },
            constraint=(
                "implied by the diversified projection and the fitted loading, "
                "averaged against the localization kernel; unconstrained in sign "
                "and not normalized"
            ),
            extra={"kernel_mass": diagnostics["kernel_mass"]},
        )

        # ``att`` stays the unweighted post-period mean the two paths imply;
        # ATEL is the kernel-weighted estimand and is carried beside it, so
        # neither number stands in for the other.
        submodels = build_effect_submodels(
            observed_outcome=np.asarray(estimates["observed"], dtype=float),
            counterfactual_outcome=np.asarray(estimates["counterfactual"], dtype=float),
            n_pre_periods=int(inputs.n_pre),
            n_post_periods=int(inputs.n_post),
            time_periods=np.asarray(inputs.time_labels),
            inference=inference,
            weights=weights_container,
            method_name="ATEL",
            effects_overrides={"additional_effects": {"atel": estimates["atel"]}},
            intervention_time=(
                inputs.time_labels[inputs.n_pre]
                if inputs.n_pre < inputs.n_periods
                else inputs.time_labels[-1]
            ),
        )

        results = ATELResults(
            **submodels,
            atel=estimates["atel"],
            bandwidth=estimates["bandwidth"],
            n_factors=self.n_factors,
            factors=estimates["factors"],
            loadings=estimates["loadings"],
            kernel_weights=estimates["kernel_weights"],
            implied_donor_weights=estimates["implied_donor_weights"],
            pointwise_standard_errors=estimates["pointwise_standard_errors"],
            inputs=inputs,
            diagnostics=diagnostics,
        )

        plot_settings = self.config.resolved_plot()
        if plot_settings.display:
            try:
                import matplotlib.pyplot as plt

                fig = plot_atel(results)
                if plot_settings.save:
                    name = (
                        plot_settings.save
                        if isinstance(plot_settings.save, str)
                        else "atel.png"
                    )
                    fig.savefig(name, bbox_inches="tight")
                plt.show()
            except Exception as exc:  # pragma: no cover - display path
                raise MlsynthPlottingError(f"ATEL plotting failed: {exc}") from exc

        return results
