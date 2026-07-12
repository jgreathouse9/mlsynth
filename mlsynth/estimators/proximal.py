"""Proximal Inference (PROXIMAL) estimator.

Implements:

    Shi, X., Li, K., Miao, W., Hu, M., & Tchetgen Tchetgen, E. (2023).
    "Theory for Identification and Inference with Synthetic Controls: A
    Proximal Causal Inference Framework." arXiv:2108.13935.

    Liu, J., Tchetgen Tchetgen, E. J., & Varjao, C. (2023). "Proximal
    Causal Inference for Synthetic Control with Surrogates."
    arXiv:2308.09527.

PROXIMAL treats donor outcomes as negative controls instrumented by donor
proxies, and optionally adds surrogate outcomes instrumented by surrogate
proxies. It runs up to three methods on the same panel:

1. **PI** -- Proximal Inference (donors only).
2. **PIS** -- Proximal Inference with surrogates (full-sample two-stage).
3. **PIPost** -- post-treatment-only surrogate variant.

PIS and PIPost run only when surrogate units are configured. Every method
closes with a GMM sandwich variance for the ATT (HAC/Bartlett middle).

See ``mlsynth.utils.proximal_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import PROXIMALConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.proximal_helpers.orchestration import run_proximal
from ..utils.proximal_helpers.plotter import plot_proximal
from ..utils.proximal_helpers.setup import prepare_proximal_inputs
from ..utils.proximal_helpers.structures import (
    PI,
    PIPOST,
    PIPW,
    PIS,
    SPSC,
    DR,
    DR_OID,
    PIOID,
    PROXIMALResults,
)


class PROXIMAL:
    """Proximal Inference (PROXIMAL) estimator.

    Parameters
    ----------
    config : PROXIMALConfig or dict
        Configuration object. See :class:`mlsynth.config_models.PROXIMALConfig`.

    Returns
    -------
    PROXIMALResults
        Container with the PI fit (always) and the PIS / PIPost fits when
        surrogates are configured, plus convenience accessors forwarding
        to the headline PI method.
    """

    def __init__(self, config: Union[PROXIMALConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = PROXIMALConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid PROXIMAL configuration: {exc}") from exc

        self.config: PROXIMALConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color
        self.surrogates: List[Union[str, int]] = config.surrogates
        self.donors: List[Union[str, int]] = config.donors
        self.vars: Dict[str, List[str]] = config.vars
        self.methods: List[str] = config.methods
        self.spsc_detrend: bool = config.spsc_detrend
        self.spsc_lambda = config.spsc_lambda
        self.spsc_spline_df: int = config.spsc_spline_df
        self.spsc_basis_degree: int = config.spsc_basis_degree
        self.spsc_att_degree: int = config.spsc_att_degree
        self.spsc_detrend_basis: str = config.spsc_detrend_basis
        self.spsc_detrend_degree: int = config.spsc_detrend_degree
        self.spsc_conformal: bool = config.spsc_conformal
        self.spsc_conformal_periods = config.spsc_conformal_periods
        self.outcome_instruments: List[Union[str, int]] = config.outcome_instruments
        self.treatment_instruments: List[Union[str, int]] = config.treatment_instruments
        self.dr_oid_ridge: float = config.dr_oid_ridge
        self.dr_oid_n_starts: int = config.dr_oid_n_starts
        self.pioid_hac_lag: int = config.pioid_hac_lag

    def fit(self) -> PROXIMALResults:
        """Run the proximal pipeline and return a :class:`PROXIMALResults`."""

        try:
            balance(self.df, self.unitid, self.time)

            inputs = prepare_proximal_inputs(
                df=self.df,
                outcome=self.outcome,
                unitid=self.unitid,
                time=self.time,
                treat=self.treat,
                donors=self.donors,
                surrogates=self.surrogates,
                vars=self.vars,
                methods=self.methods,
                spsc_detrend=self.spsc_detrend,
                spsc_lambda=self.spsc_lambda,
                spsc_spline_df=self.spsc_spline_df,
                spsc_basis_degree=self.spsc_basis_degree,
                spsc_att_degree=self.spsc_att_degree,
                spsc_detrend_basis=self.spsc_detrend_basis,
                spsc_detrend_degree=self.spsc_detrend_degree,
                spsc_conformal=self.spsc_conformal,
                spsc_conformal_periods=self.spsc_conformal_periods,
                outcome_instruments=self.outcome_instruments,
                treatment_instruments=self.treatment_instruments,
                dr_oid_ridge=self.dr_oid_ridge,
                dr_oid_n_starts=self.dr_oid_n_starts,
                pioid_hac_lag=self.pioid_hac_lag,
            )

            fits = run_proximal(inputs)

            results = PROXIMALResults(
                inputs=inputs,
                pi=fits.get(PI),
                pis=fits.get(PIS),
                pipost=fits.get(PIPOST),
                spsc=fits.get(SPSC),
                dr=fits.get(DR),
                pipw=fits.get(PIPW),
                dr_oid=fits.get(DR_OID),
                pioid=fits.get(PIOID),
                selected_variant=self.methods[0],
                metadata={
                    "methods": list(self.methods),
                    "has_surrogates": inputs.has_surrogates,
                    "bandwidth": inputs.bandwidth,
                    "n_donors": inputs.n_donors,
                },
            )

        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except (ValueError, KeyError, IndexError) as exc:
            raise MlsynthEstimationError(
                f"Proximal estimation failed: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise MlsynthEstimationError(
                f"An unexpected error occurred during Proximal fit: {type(exc).__name__}: {exc}"
            ) from exc

        if self.display_graphs:
            try:
                plot_proximal(results)
            except (MlsynthPlottingError, MlsynthDataError) as exc:
                warnings.warn(f"Plotting failed in Proximal estimator: {exc}", UserWarning)
            except Exception as exc:
                warnings.warn(
                    f"An unexpected error occurred during Proximal plotting: {type(exc).__name__}: {exc}",
                    UserWarning,
                )

        return results
