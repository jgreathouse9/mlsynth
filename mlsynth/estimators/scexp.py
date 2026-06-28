"""MAREX: Synthetic Controls for Experimental Design (Abadie & Zhao 2026).

MAREX *designs* an experiment on aggregate units (e.g. markets): using only
pre-experimental data it selects which units to treat (treated weights ``w``)
and which untreated units form the synthetic control (control weights ``v``),
on the simplex and disjoint (a unit is treated or a control, never both). The
synthetic treated and synthetic control units are built to reproduce population
predictor means, so their post-period difference estimates the average
treatment effect. Optional clustering treats one (or a few) units per cluster;
optional blank-period placebo inference yields p-values and confidence bands.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import MAREXConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.marex_helpers.orchestration import solve_marex
from ..utils.marex_helpers.plotter import plot_marex
from ..utils.marex_helpers.setup import prepare_marex_panel
from ..utils.marex_helpers.structures import MAREXResults


class MAREX:
    """Synthetic-control experimental design estimator (Abadie & Zhao 2026).

    Parameters
    ----------
    config : MAREXConfig or dict
        Configuration object. See :class:`mlsynth.config_models.MAREXConfig`.

    Returns
    -------
    MAREXResults
        Per-cluster and aggregate treated/control weights, synthetic series,
        the selected treated units, and (optionally) placebo inference.
    """

    def __init__(self, config: Union[MAREXConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MAREXConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid MAREX configuration: {exc}") from exc

        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time

        self.T0 = config.T0
        self.post_col = config.post_col
        self.cluster = config.cluster
        self.design: str = config.design

        self.beta: float = config.beta
        self.lambda1: float = config.lambda1
        self.lambda2: float = config.lambda2
        self.xi: float = config.xi
        self.lambda1_unit: float = config.lambda1_unit
        self.lambda2_unit: float = config.lambda2_unit
        self.costs = config.costs
        self.budget = config.budget
        self.covariates = config.covariates
        self.covariate_weight: float = config.covariate_weight
        self.standardize: bool = config.standardize

        self.blank_periods: int = config.blank_periods
        self.m_eq = config.m_eq
        self.m_min = config.m_min
        self.m_max = config.m_max
        self.exclusive: bool = config.exclusive
        self.solver = config.solver
        self.verbose: bool = config.verbose
        self.inference: bool = config.inference
        self.T_post = config.T_post
        self.relaxed: bool = getattr(config, "relaxed", False)
        self.display_graph: bool = config.display_graph
        # geographic design restrictions
        self.to_be_treated = config.to_be_treated
        self.not_to_be_treated = config.not_to_be_treated
        self.adjacency = config.adjacency
        self.spillover_threshold = config.spillover_threshold
        self.exclude_bordering_donors = config.exclude_bordering_donors
        self.size_col = config.size_col
        self.min_size = config.min_size
        self.max_size = config.max_size
        self.cluster_col = config.cluster_col
        self.stratum_col = config.stratum_col
        self.min_per_stratum = config.min_per_stratum
        self.max_per_stratum = config.max_per_stratum

        if self.cluster and self.cluster not in self.df.columns:
            raise MlsynthDataError(f"Cluster column '{self.cluster}' not found in DataFrame.")

    def fit(self) -> MAREXResults:
        """Run the MAREX design and return :class:`MAREXResults`."""
        balance(self.df, self.unitid, self.time)

        if self.m_eq is not None and (self.m_min is not None or self.m_max is not None):
            raise MlsynthConfigError(
                "Cannot specify both 'm_eq' and 'm_min/m_max'. Choose an exact "
                "count or a range of treated units per cluster."
            )
        if self.m_eq is None and self.m_min is None and self.m_max is None:
            raise MlsynthConfigError(
                "You must specify either 'm_eq' or at least one of 'm_min'/'m_max'."
            )

        try:
            panel = prepare_marex_panel(
                df=self.df, outcome=self.outcome, unitid=self.unitid, time=self.time,
                cluster=self.cluster, T0=self.T0, inference=self.inference,
                blank_periods=self.blank_periods, T_post=self.T_post,
                covariates=self.covariates,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except ValueError as exc:
            raise MlsynthConfigError(str(exc)) from exc
        except Exception as exc:
            raise MlsynthDataError(f"Error preparing MAREX inputs: {exc}") from exc

        # Build the geographic restriction bundle against the panel's unit
        # IndexSet -- the single source of truth for who is who.
        from ..utils.marex_helpers.restrictions import build_restrictions
        restrictions = build_restrictions(
            self.df, self.unitid, panel.unit_index,
            to_be_treated=self.to_be_treated,
            not_to_be_treated=self.not_to_be_treated,
            cluster_col=self.cluster_col,
            adjacency=self.adjacency, spillover_threshold=self.spillover_threshold,
            stratum_col=self.stratum_col,
            min_per_stratum=self.min_per_stratum,
            max_per_stratum=self.max_per_stratum,
            size_col=self.size_col, min_size=self.min_size, max_size=self.max_size,
            exclude_bordering_donors=self.exclude_bordering_donors,
        )
        restrictions = None if restrictions.is_empty else restrictions

        try:
            import cvxpy as cp
            results = solve_marex(
                Y_full=panel.Y_full, T0=panel.T0, clusters=panel.clusters,
                design=self.design, blank_periods=panel.blank_periods,
                m_eq=self.m_eq, m_min=self.m_min, m_max=self.m_max,
                exclusive=self.exclusive, beta=self.beta,
                lambda1=self.lambda1, lambda2=self.lambda2, xi=self.xi,
                lambda1_unit=self.lambda1_unit, lambda2_unit=self.lambda2_unit,
                costs=self.costs, budget=self.budget,
                covariates=panel.covariates, covariate_names=panel.covariate_names,
                covariate_weight=self.covariate_weight,
                standardize=self.standardize,
                solver=self.solver or cp.SCIP, verbose=self.verbose,
                relaxed=self.relaxed, inference=self.inference,
                unit_index=panel.unit_index, time_index=panel.time_index,
                restrictions=restrictions,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"MAREX design failed: {exc}") from exc

        # Solution pool + MDE power + recommendation (SYNDES parity): when
        # top_K > 1, enumerate distinct designs via no-good cuts (exact MIQP),
        # score each on a Newey-West MDE power curve, and attach the menu plus a
        # composite recommendation. The primary result above is unchanged.
        if self.config.top_K and self.config.top_K > 1:
            try:
                import cvxpy as cp
                from ..utils.marex_helpers.optimization import solve_design_pool
                from ..utils.marex_helpers.pool import build_marex_pool
                from ..utils.marex_helpers.select import recommend_marex
                # Hold out a blank tail of the pre-period so each pooled design
                # can be ranked on its OUT-OF-SAMPLE (blank) fit, not in-sample
                # overfit. Reuse the configured blank window if any, else the
                # 30% pre-tail convention shared by the MAREX-family estimators.
                pool_blank = (panel.blank_periods if panel.blank_periods > 0
                              else max(1, int(0.3 * panel.T0)))
                raws = solve_design_pool(
                    Y_full=panel.Y_full, T0=panel.T0, clusters=panel.clusters,
                    top_K=self.config.top_K, blank_periods=pool_blank,
                    m_eq=self.m_eq, m_min=self.m_min, m_max=self.m_max,
                    exclusive=self.exclusive, design=self.design, beta=self.beta,
                    lambda1=self.lambda1, lambda2=self.lambda2, xi=self.xi,
                    lambda1_unit=self.lambda1_unit, lambda2_unit=self.lambda2_unit,
                    costs=self.costs, budget=self.budget, covariates=panel.covariates,
                    covariate_weight=self.covariate_weight, standardize=self.standardize,
                    solver=self.solver or cp.SCIP, verbose=self.verbose,
                )
                pool = build_marex_pool(
                    raws, alpha=self.config.alpha, power=self.config.power_target,
                    n_post=self.config.T_post or 1)
                if pool:
                    rec = recommend_marex(
                        pool, power_weight=self.config.power_weight,
                        fit_weight=self.config.fit_weight,
                        max_shortlist=self.config.max_shortlist)
                    results = results.model_copy(
                        update={"pool": pool, "recommendation": rec})
            except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
                raise
            except Exception as exc:
                raise MlsynthEstimationError(f"MAREX pool failed: {exc}") from exc

        if self.display_graph:
            try:
                plot_marex(results, plot_type="treatment")
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(f"MAREX plotting failed: {exc}") from exc

        return results
