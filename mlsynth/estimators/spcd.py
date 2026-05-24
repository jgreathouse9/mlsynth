"""Synthetic Principal Component Design (SPCD) estimator.

This module implements:

    Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
    "Synthetic Principal Component Design: Fast Covariate Balancing
    with Synthetic Controls." arXiv:2211.15241v1.

Algorithm flow (see ``mlsynth.utils.spcd_helpers``):

    formulation.py         : Eq. (2)         M = Y Y^T + alpha I + lambda 1 1^T
    spectral_init.py       : Algs 1 & 2      y^0 = sgn(smallest eigvec of M)
    iteration_spcd.py      : Eqs. (4)/(7)    SPCD update     (variant="spcd")
    iteration_norm_spcd.py : Eqs. (5)/(8)    NormSPCD update (variant="norm_spcd")
    weights_empirical.py   : Eq. (9)         Algorithm 2 final step (weights="empirical")
    weights_exact.py       : Eq. (6)         Algorithm 1 final step (weights="exact")
    treatment_effect.py    : Algs 1 & 2 footer (minority flip, synthetic paths)

Theoretical artifacts (not implemented as separate code paths):

    Algorithm 3 (Appx 3.2, p.20) : abstract GPW used in the proof of Theorem 3
    Algorithm 4 (Appx 3.2, p.20) : abstract Normalized GPW used in the proof
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SPCDConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.spcd_helpers.orchestration import solve_spcd_with_holdout
from ..utils.spcd_helpers.results_assembly import build_summary
from ..utils.spcd_helpers.plotter import plot_spcd_design
from ..utils.spcd_helpers.setup import prepare_spcd_inputs
from ..utils.spcd_helpers.structures import SPCDMultiArmResults, SPCDResults


class SPCD:
    """Synthetic Principal Component Design (SPCD) estimator.

    Selects a treated group and constructs synthetic control weights
    simultaneously by solving the optimal experiment-design problem of
    Lu, Li, Ying, Blanchet (2022) via a (normalized) generalized power
    method with spectral initialization.

    Parameters
    ----------
    config : SPCDConfig or dict
        Configuration object defining data, iteration variant, and
        weight-step choice.

    Attributes
    ----------
    config : SPCDConfig
        Parsed configuration object.
    df : pandas.DataFrame
        Input panel data.
    outcome : str
        Name of outcome column.
    unitid : str
        Unit identifier column.
    time : str
        Time identifier column.
    variant : {"spcd", "norm_spcd"}
        Iteration-box choice. ``"spcd"`` uses Eq. (4)/(7);
        ``"norm_spcd"`` uses Eq. (5)/(8).
    weights : {"empirical", "exact"}
        Final-weight-step choice. ``"empirical"`` uses Eq. (9), the
        paper's experimental default; ``"exact"`` solves Eq. (6) via
        cvxpy.
    alpha_ridge, lam_balance, beta : float or None
        Hyperparameters of Eq. (2) and the iteration update. When
        ``None``, each is auto-estimated from the spectrum of
        ``Y_pre.T @ Y_pre``.
    max_iter : int
        Maximum iterations for the SPCD/NormSPCD while loop.
    T0, post_col : optional
        Pre/post split specification (mirrors SYNDES's interface).
    solver : optional
        Passed to cvxpy when ``weights="exact"``.
    display_graph : bool
        Whether to plot synthetic treated/control series after fit.
    verbose : bool
        Solver verbosity flag.

    Returns
    -------
    SPCDResults
        Container with the optimized design, attached preprocessed
        inputs, and synthetic-path arrays.
    """

    def __init__(self, config: Union[SPCDConfig, dict]) -> None:
        """Initialize SPCD from a Pydantic config or compatible dictionary."""

        if isinstance(config, dict):
            try:
                config = SPCDConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid SPCD configuration: {exc}") from exc

        self.config = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.variant: str = config.variant
        self.weights: str = config.weights
        self.alpha_ridge: Optional[float] = config.alpha_ridge
        self.lam_balance: Optional[float] = config.lam_balance
        self.beta: Optional[float] = config.beta
        self.max_iter: int = config.max_iter

        self.T0: Optional[int] = config.T0
        self.post_col: Optional[str] = config.post_col
        self.arm: Optional[str] = config.arm

        self.solver: Any = config.solver
        self.display_graph: bool = config.display_graph
        self.verbose: bool = config.verbose

        # Inference and power analysis controls (see SPCDConfig docs).
        self.enable_inference: bool = config.enable_inference
        self.holdout_frac_E: float = config.holdout_frac_E
        self.inference_alpha: float = config.inference_alpha
        self.power_target: float = config.power_target
        self.mde_n_sims: int = config.mde_n_sims
        self.mde_n_trials: int = config.mde_n_trials
        self.mde_horizon_grid = config.mde_horizon_grid
        self.inference_seed: int = config.inference_seed
        self.min_blank_size: int = config.min_blank_size

    def _fit_single(self, df: pd.DataFrame) -> SPCDResults:
        """Run the SPCD pipeline on one (sub-)panel and return its design."""
        balance(df, self.unitid, self.time)
        inputs = prepare_spcd_inputs(
            df=df,
            outcome=self.outcome,
            unitid=self.unitid,
            time=self.time,
            T0=self.T0,
            post_col=self.post_col,
        )
        design, conformal, power = solve_spcd_with_holdout(
            inputs=inputs,
            variant=self.variant,
            weights=self.weights,
            alpha=self.alpha_ridge,
            lam=self.lam_balance,
            beta=self.beta,
            max_iter=self.max_iter,
            solver=self.solver,
            verbose=self.verbose,
            enable_inference=self.enable_inference,
            holdout_frac_E=self.holdout_frac_E,
            inference_alpha=self.inference_alpha,
            power_target=self.power_target,
            mde_n_sims=self.mde_n_sims,
            mde_n_trials=self.mde_n_trials,
            mde_horizon_grid=self.mde_horizon_grid,
            inference_seed=self.inference_seed,
            min_blank_size=self.min_blank_size,
        )
        summary = build_summary(
            design=design, inputs=inputs, conformal=conformal, power=power
        )
        return SPCDResults(
            design=design, inputs=inputs, summary=summary,
            conformal=conformal, power=power,
        )

    def fit(self) -> Union[SPCDResults, SPCDMultiArmResults]:
        """Run the SPCD pipeline and return the design.

        Returns
        -------
        SPCDResults or SPCDMultiArmResults
            A single design when no ``arm`` column is configured; otherwise
            one independent SPCD design per arm, keyed by arm label.
        """

        try:
            if self.arm is None:
                results = self._fit_single(self.df)
                if self.display_graph:
                    try:
                        plot_spcd_design(results)
                    except Exception as exc:
                        raise MlsynthPlottingError(
                            f"SPCD plotting failed: {exc}") from exc
                return results

            # Multi-arm: solve the SPCD design independently within each arm.
            if self.arm not in self.df.columns:
                raise MlsynthDataError(
                    f"Arm column {self.arm!r} not found in the data.")
            if self.df.groupby(self.unitid)[self.arm].nunique().max() > 1:
                raise MlsynthDataError(
                    "The arm column varies within a unit over time.")

            arm_designs = {
                arm_label: self._fit_single(sub.copy())
                for arm_label, sub in self.df.groupby(self.arm, sort=True)
            }
            results = SPCDMultiArmResults(arm_designs=arm_designs, arm=self.arm)

            if self.display_graph:
                for arm_result in arm_designs.values():
                    try:
                        plot_spcd_design(arm_result)
                    except Exception as exc:
                        raise MlsynthPlottingError(
                            f"SPCD plotting failed: {exc}") from exc

            return results

        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
            MlsynthPlottingError,
        ):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"SPCD estimation failed: {exc}") from exc
