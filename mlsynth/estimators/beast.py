"""BEAST: immunized doubly-robust synthetic control (Bléhaut et al. 2021).

A synthetic control whose donor weights come from an ℓ₁-penalised covariate
*balancing* (exponential tilting) rather than an outcome-fitting simplex QP, and
whose ATT is *immunized* by a Lasso outcome regression, making it doubly robust
and asymptotically normal with an analytic standard error. Best suited to a
sparse, informative covariate regime; a balance-validity guard rejects the
over-saturated high-dimensional case the method is not built for.

Reference: Bléhaut, D'Haultfœuille, L'Hour & Tsybakov (2021), *"An alternative to
synthetic control for models with many covariates under sparsity"*
(arXiv 2005.12225); authors' code ``jeremylhour/alternative-synthetic-control-sparsity``.
"""
from __future__ import annotations

from typing import Union

from ..config_models import BaseEstimatorResults
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.beast_helpers.config import BEASTConfig
from ..utils.beast_helpers.pipeline import run_beast
from ..utils.beast_helpers.plotter import plot_beast

try:  # pydantic v2 / v1 compatibility for the error type
    from pydantic import ValidationError
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore


class BEAST:
    """Immunized doubly-robust synthetic control estimator.

    Parameters
    ----------
    config : BEASTConfig or dict
        Configuration. See
        :class:`mlsynth.utils.beast_helpers.config.BEASTConfig`.

    Examples
    --------
    >>> from mlsynth import BEAST
    >>> res = BEAST({"df": panel, "outcome": "cigsale", "treat": "treated",
    ...              "unitid": "state", "time": "year",
    ...              "covariates": ["loginc", "p_cig", "pct15-24", "pc_beer"],
    ...              "outcome_lags": [1975, 1980, 1988]}).fit()
    >>> res.att                          # doctest: +SKIP
    """

    def __init__(self, config: Union[BEASTConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = BEASTConfig(**config)
            except ValidationError as exc:  # pragma: no cover - passthrough
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, BEASTConfig):
            raise MlsynthConfigError(
                "config must be a BEASTConfig or a dict of its fields.")
        self.config = config

    def fit(self) -> BaseEstimatorResults:
        """Estimate the immunized ATT and return standardized results."""
        try:
            results = run_beast(self.config)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"BEAST estimation failed: {exc}") from exc

        if self.config.display_graphs:
            plot_beast(self.config, results)
        return results
