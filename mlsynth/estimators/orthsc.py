"""ORTHSC: the Orthogonalized Synthetic Control (Fry 2026).

An IV synthetic control whose ATT estimate is Neyman-orthogonalized with respect
to the control weights -- a partially identified, simplex-constrained,
high-dimensional nuisance. Because the moment conditions are orthogonal to those
weights, the ATT is asymptotically normal (and insensitive to which weight
vector in the identified set is chosen), and a fixed-smoothing Series-HAC
variance with a Sun (2013) bandwidth gives a t-test that controls size without a
consistent variance estimate.

The weights are identified using, as instruments, the outcomes of untreated
units excluded from the control pool (Fry's IV moment conditions). The caller
names those instrument units in the config.
"""
from __future__ import annotations

from typing import Union

from ..config_models import BaseEstimatorResults
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.orthsc_helpers.config import OrthSCConfig
from ..utils.orthsc_helpers.pipeline import run_orthsc

try:  # pydantic v2 / v1 compatibility for the error type
    from pydantic import ValidationError
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore


class ORTHSC:
    """Orthogonalized Synthetic Control estimator.

    Parameters
    ----------
    config : OrthSCConfig or dict
        Configuration. See
        :class:`mlsynth.utils.orthsc_helpers.config.OrthSCConfig`.

    Examples
    --------
    >>> from mlsynth import ORTHSC
    >>> res = ORTHSC({"df": panel, "outcome": "y", "treat": "treated",
    ...               "unitid": "country", "time": "year",
    ...               "instruments": ["Finland", "Germany"]}).fit()
    >>> res.att                          # doctest: +SKIP
    """

    def __init__(self, config: Union[OrthSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = OrthSCConfig(**config)
            except ValidationError as exc:  # pragma: no cover - passthrough
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, OrthSCConfig):
            raise MlsynthConfigError(
                "config must be an OrthSCConfig or a dict of its fields.")
        self.config = config

    def fit(self) -> BaseEstimatorResults:
        """Estimate the orthogonalized ATT and return standardized results."""
        try:
            return run_orthsc(self.config)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"ORTHSC estimation failed: {exc}") from exc
