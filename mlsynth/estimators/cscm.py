"""CSCM: a flexible synthetic control for count and other non-negative outcomes.

Bonander (2021, *Epidemiology*). The donor weights keep the non-negativity
constraint -- so the counterfactual of a non-negative outcome stays
non-negative -- but drop the adding-up (sum-to-one) constraint, penalising
departures from the classic simplex weights by ``lambda ||w - w_scm||^2``
(solved exactly as non-negative least squares). This buys strictly better
pre-treatment balance than ordinary SCM while remaining valid for counts.

The predictor-importance matrix ``V`` weights each balance feature by a
leave-one-out Poisson ridge (``v_method="poisson_ridge"``) or equally
(``"uniform"``). The effect is reported as a rate ratio with a cross-fitted,
bias-corrected t-interval (Chernozhukov, Wuthrich & Zhu 2021).
"""

from __future__ import annotations

from typing import Union

from ..config_models import BaseEstimatorResults
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.cscm_helpers.config import CSCMConfig
from ..utils.cscm_helpers.pipeline import run_cscm

try:  # pydantic v2 / v1 compatibility for the error type
    from pydantic import ValidationError
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore


class CSCM:
    """Flexible count synthetic control estimator (Bonander 2021).

    Parameters
    ----------
    config : CSCMConfig or dict
        Configuration. See
        :class:`mlsynth.utils.cscm_helpers.config.CSCMConfig`.

    Examples
    --------
    >>> from mlsynth import CSCM
    >>> cfg = {"df": panel, "outcome": "deaths", "treat": "policy",
    ...        "unitid": "country", "time": "year", "K": 2}
    >>> res = CSCM(cfg).fit()
    >>> res.effects.additional_effects["rate_ratio"]   # doctest: +SKIP
    """

    def __init__(self, config: Union[CSCMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = CSCMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, CSCMConfig):
            raise MlsynthConfigError(
                "config must be a CSCMConfig or a dict of its fields."
            )
        self.config = config

    def fit(self) -> BaseEstimatorResults:
        """Estimate the flexible count SC and return standardized results."""
        try:
            return run_cscm(self.config)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"CSCM estimation failed: {exc}") from exc
