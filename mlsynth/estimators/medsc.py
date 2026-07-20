"""MEDSC: mediation-analysis synthetic control (Mellace & Pasquini 2022).

Decomposes the synthetic-control treatment effect on a panel outcome into a
direct effect and an indirect effect that runs through a single mediator series.

* Total effect -- an ordinary synthetic control on the treated outcome
  (:math:`\\hat\\tau^{tot}_t = Y_{1t} - \\hat Y^{0,M0}_{1t}`); the mediator does
  not enter.
* Direct effect -- a cross-world synthetic control that, in each post period,
  also matches the treated unit's post-treatment mediator path up to that period
  (:math:`\\hat\\tau^{dir}_t = Y_{1t} - \\hat Y^{0,M1}_{1t}`). This holds the
  mediator at its treated (post-intervention) value, so what remains is the
  effect not routed through the mediator.
* Indirect effect -- total minus direct, i.e. everything the intervention does
  by moving the mediator.

The direct fit may draw on a wider donor pool than the total fit, so the
cross-world control can span the treated unit's post-treatment mediator values.
Without covariates the fit matches the pre-treatment outcome path (the
specification under which the paper's Prop 99 decomposition reproduces); with
covariates the predictor weights are cross-validated by the bilevel mscmt
search.

Note: unrelated to :class:`mlsynth.MASC` (Kellogg's Matching-and-Synthetic-
Control), which shares the paper's original acronym but is a different method.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.medsc_helpers.config import MEDSCConfig
from ..utils.medsc_helpers.pipeline import run_medsc_core
from ..utils.medsc_helpers.plotter import plot_medsc
from ..utils.medsc_helpers.setup import prepare_medsc_inputs
from ..utils.medsc_helpers.structures import MEDSCResults

try:  # pydantic v2 / v1 compatibility for the error type
    from pydantic import ValidationError
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore


class MEDSC:
    """Mediation-analysis synthetic control estimator.

    Parameters
    ----------
    config : MEDSCConfig or dict
        Configuration. See
        :class:`mlsynth.utils.medsc_helpers.config.MEDSCConfig`.

    Examples
    --------
    >>> from mlsynth import MEDSC
    >>> cfg = {"df": panel, "outcome": "cigsale", "mediator": "price",
    ...        "treat": "treated", "unitid": "state", "time": "year",
    ...        "direct_donors": wide_pool, "total_donors": narrow_pool}
    >>> res = MEDSC(cfg).fit()
    >>> res.att, res.att_direct, res.att_indirect          # doctest: +SKIP
    """

    def __init__(self, config: Union[MEDSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MEDSCConfig(**config)
            except ValidationError as exc:  # pragma: no cover - passthrough
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, MEDSCConfig):
            raise MlsynthConfigError(
                "config must be a MEDSCConfig or a dict of its fields.")
        self.config = config

    def fit(self) -> MEDSCResults:
        """Run the MEDSC decomposition and return standardized results."""
        try:
            cfg = self.config
            inputs = prepare_medsc_inputs(
                df=cfg.df, outcome=cfg.outcome, mediator=cfg.mediator,
                treat=cfg.treat, unitid=cfg.unitid, time=cfg.time,
                total_donors=cfg.total_donors, direct_donors=cfg.direct_donors,
                covariates=cfg.covariates,
            )
            dec, inf_p, inf_details, metadata = run_medsc_core(inputs, cfg)

            labels = np.asarray(inputs.time_labels)
            T0, T = inputs.T0, inputs.T
            results = MEDSCResults(
                inputs=inputs,
                decomposition=dec,
                metadata=metadata,
                effects=EffectsResults(
                    att=dec.att_total,
                    additional_effects={
                        "att_direct": dec.att_direct,
                        "att_indirect": dec.att_indirect,
                    },
                ),
                time_series=TimeSeriesResults(
                    observed_outcome=np.asarray(inputs.treated_outcome, dtype=float),
                    counterfactual_outcome=np.asarray(
                        dec.counterfactual_total, dtype=float),
                    estimated_gap=np.asarray(dec.total, dtype=float),
                    time_periods=labels,
                    intervention_time=(labels[T0] if T0 < T else None),
                ),
                weights=WeightsResults(
                    donor_weights=dec.total_weights,
                    summary_stats={
                        "direct_weights_final": dec.direct_weights_final},
                ),
                fit_diagnostics=FitDiagnosticsResults(
                    rmse_pre=dec.pre_rmse_total),
                inference=InferenceResults(
                    p_value=inf_p,
                    method="placebo" if cfg.inference else None,
                    details=inf_details or None,
                ),
                method_details=MethodDetailsResults(
                    method_name="MEDSC", is_recommended=True,
                    parameters_used={
                        "backend": metadata["backend"],
                        "n_total_donors": metadata["n_total_donors"],
                        "n_direct_donors": metadata["n_direct_donors"],
                    }),
            )

            if cfg.display_graphs:
                try:
                    plot_medsc(results)
                except Exception as exc:  # pragma: no cover - defensive translation
                    raise MlsynthPlottingError(
                        f"MEDSC plotting failed: {exc}") from exc

            return results

        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
            MlsynthPlottingError,
        ):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(
                f"MEDSC estimation failed: {exc}") from exc
