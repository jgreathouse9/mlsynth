"""ROLLDID: rolling-transformation difference-in-differences (Lee & Wooldridge).

A clean-room implementation (MIT) of the rolling-DiD / small-N exact-inference
method: collapse the panel to one cross-sectional observation per unit by a
pre-treatment rolling transformation, then read the ATT off a cross-sectional
regression on a treatment indicator. Common timing and staggered adoption,
demean / detrend transforms, exact-t / HC3 / randomization inference.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import ValidationError

from ..config_models import EffectsResults, InferenceResults, TimeSeriesResults
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.rolldid_helpers import (
    ROLLDIDConfig,
    ROLLDIDResults,
    estimate,
    plot_rolldid,
    rolldid_setup,
)


class ROLLDID:
    """Rolling-transformation difference-in-differences.

    Parameters
    ----------
    config : ROLLDIDConfig or dict
        See
        :class:`mlsynth.utils.rolldid_helpers.config.ROLLDIDConfig` — adds
        ``rolling`` (``demean``/``detrend``), ``inference``
        (``exact``/``hc3``/``ri``), ``alpha``, ``ri_reps``, ``seed`` to the base
        ``df`` / ``outcome`` / ``treat`` / ``unitid`` / ``time`` fields.

    Examples
    --------
    >>> from mlsynth import ROLLDID
    >>> res = ROLLDID({"df": panel, "outcome": "y", "treat": "w",
    ...                "unitid": "id", "time": "t", "rolling": "detrend"}).fit()  # doctest: +SKIP
    >>> res.effects.att, res.inference.p_value                                    # doctest: +SKIP
    """

    def __init__(self, config: Union[ROLLDIDConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = ROLLDIDConfig(**config)
            except ValidationError as exc:  # pragma: no cover - passthrough
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, ROLLDIDConfig):
            raise MlsynthConfigError(
                "config must be a ROLLDIDConfig or a dict of its fields.")
        self.config = config
        self._result: Optional[ROLLDIDResults] = None

    def fit(self) -> ROLLDIDResults:
        """Run the rolling-DiD estimate and return a standardized result."""
        cfg = self.config
        try:
            prep = rolldid_setup(cfg.df, cfg.unitid, cfg.time, cfg.outcome, cfg.treat)
            out = estimate(prep, mode=cfg.rolling, inference=cfg.inference,
                           alpha=cfg.alpha, ri_reps=cfg.ri_reps, seed=cfg.seed)
            agg = out["aggregate"]
            ts = out.get("time_series")
            time_series = None
            if ts is not None:
                time_series = TimeSeriesResults(
                    observed_outcome=ts["observed"],
                    counterfactual_outcome=ts["counterfactual"],
                    estimated_gap=ts["gap"],
                    time_periods=ts["time_periods"],
                    intervention_time=ts["intervention_time"],
                )
            result = ROLLDIDResults(
                effects=EffectsResults(att=agg["att"]),
                inference=InferenceResults(
                    p_value=agg["p_value"], standard_error=agg["se"],
                    ci_lower=agg["ci_lower"], ci_upper=agg["ci_upper"],
                    confidence_level=1.0 - cfg.alpha, method=agg["method"]),
                time_series=time_series,
                transformation=out["transformation"], inference_type=cfg.inference,
                design=out["design"], n_treated=out["n_treated"],
                n_control=out["n_control"], per_period=out["per_period"],
                per_cohort=out["per_cohort"],
            )
            if cfg.display_graphs:
                plot_rolldid(result, show=True)
            self._result = result
            return result
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"ROLLDID failed: {exc}") from exc
