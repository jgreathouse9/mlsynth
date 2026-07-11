"""DPSC: differentially private synthetic control (Rho, Cummings & Misra 2023).

A ridge synthetic control whose released counterfactual satisfies
``epsilon``-differential privacy over the donor pool, via differentially private
empirical risk minimization. The regression coefficients are learned privately
(output perturbation, Algorithm 2, or objective perturbation, Algorithm 3), and
the post-intervention donor matrix is privatized before the counterfactual is
formed -- so publishing the synthetic control leaks a provably bounded amount
about any single donor.

Use it when the donor pool is sensitive (patient records, proprietary
firm-level series) and the counterfactual must be released externally. On small
donor pools the privacy noise is large; the method is favorable when the donor
pool is big and the pre-period long.

Reference: Rho, Cummings & Misra (2023), *"Differentially Private Synthetic
Control"*, AISTATS. Authors' code: ``srho1/dpsc``.
"""
from __future__ import annotations

from typing import Union

from ..config_models import BaseEstimatorResults
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.dpsc_helpers.config import DPSCConfig
from ..utils.dpsc_helpers.pipeline import run_dpsc
from ..utils.dpsc_helpers.plotter import plot_dpsc

try:  # pydantic v2 / v1 compatibility for the error type
    from pydantic import ValidationError
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore


class DPSC:
    """Differentially private synthetic control estimator.

    Parameters
    ----------
    config : DPSCConfig or dict
        Configuration. See
        :class:`mlsynth.utils.dpsc_helpers.config.DPSCConfig`.

    Returns
    -------
    BaseEstimatorResults
        The released private counterfactual and ATT, with the privacy noise
        quantified as the standard error / interval, and the non-private ridge
        ATT recorded for reference in ``method_details``.

    Examples
    --------
    >>> from mlsynth import DPSC                                            # doctest: +SKIP
    >>> res = DPSC({"df": panel, "outcome": "cigsale", "treat": "treat",
    ...             "unitid": "state", "time": "year",
    ...             "mechanism": "objective", "epsilon1": 1.0, "epsilon2": 1.0,
    ...             "display_graphs": False}).fit()
    >>> res.effects.att                                                     # doctest: +SKIP
    """

    def __init__(self, config: Union[DPSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = DPSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(str(exc)) from exc
        self.config: DPSCConfig = config

    def fit(self) -> BaseEstimatorResults:
        """Fit DPSC and return the standardized results container."""
        try:
            results = run_dpsc(self.config)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"DPSC estimation failed: {exc}") from exc

        if self.config.display_graphs:
            try:
                plot_dpsc(self.config, results)
            except MlsynthPlottingError:  # pragma: no cover - defensive re-raise
                raise
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"DPSC plotting failed: {exc}") from exc

        return results
