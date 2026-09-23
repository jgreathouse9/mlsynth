"""LPCA: Local Principal Component Analysis (Feng 2024).

Feng, Y. (2024). *"Optimal Estimation of Large-Dimensional Nonlinear Factor
Models."* Working paper; arXiv:2311.07243.

Every factor-model estimator in mlsynth writes the untreated outcome as
:math:`y_{it}(0) = f_t' \\alpha_i + u_{it}` -- the outcome matrix is low-rank
because each unit's path is a *linear* combination of a few common factors.
LPCA drops the linearity. It supposes only

.. math::

   y_{it}(0) = \\eta_t(\\alpha_i) + u_{it},

where the latent variable :math:`\\alpha_i` is low-dimensional but the response
:math:`\\eta_t` is an unknown, possibly nonlinear function. Such a matrix is
generally full rank, so nuclear-norm completion and principal components fitted
to the whole panel both misread it.

The way through is that a smooth surface looks linear up close. If two units
have similar :math:`\\alpha`, a first-order expansion makes their outcomes an
approximately linear factor model, so PCA applies *within a neighbourhood* even
though it fails globally. Since :math:`\\alpha_i` is unobserved, the
neighbourhood is built by matching on the outcomes themselves:

1. Split the periods in two. On the first block, compute the pseudo-max
   distance between units and take each unit's ``K`` nearest neighbours. The
   split matters: searching for neighbours uses the noise, so matching and
   fitting on the same periods would correlate the factors with the errors.
2. On the second block, take a truncated SVD of the neighbour submatrix and
   read off the treated unit's row. That row is the estimate of
   :math:`\\eta_t(\\alpha_i)` -- the counterfactual.

The treated unit's post-treatment cells are set to the period mean before any
of this, and Theorem 6.1 is the statement that doing so perturbs the estimate
by a vanishing amount when the post-period is short relative to the panel.

LPCA wants a wide panel: neighbours are only close in :math:`\\alpha` when many
units are available to choose among, and the paper's theory is asymptotic in
both dimensions. It is the wrong tool for a classic thirty-donor study, where
``K = round(N^(2/3))`` leaves a handful of neighbours and half the periods go
to matching.

The paper reports no standard errors, confidence intervals or p-values, and
Theorem 6.1 is a uniform max-norm rate. This estimator therefore returns a
point counterfactual and no inference.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.lpca_helpers.config import LPCAConfig
from ..utils.lpca_helpers.pipeline import run_lpca
from ..utils.lpca_helpers.setup import prepare_lpca_inputs, resolve_neighbours
from ..utils.lpca_helpers.structures import LPCAResults


class LPCA:
    """Local Principal Component Analysis estimator.

    Parameters
    ----------
    config : LPCAConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.LPCAConfig`.
    """

    def __init__(self, config: Union[LPCAConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = LPCAConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid LPCA configuration: {exc}"
                ) from exc
        self.config: LPCAConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.n_neighbors = config.n_neighbors
        self.match_periods = config.match_periods
        self.match_fraction: float = config.match_fraction
        self.distance: str = config.distance
        self.max_components: int = config.max_components
        self.demean: bool = config.demean
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> LPCAResults:
        """Run local PCA and return :class:`LPCAResults`."""
        try:
            inputs = prepare_lpca_inputs(
                self.df, self.outcome, self.treat, self.unitid, self.time,
                match_periods=self.match_periods,
                match_fraction=self.match_fraction,
            )
            results = run_lpca(
                inputs,
                n_neighbours=resolve_neighbours(inputs.N, self.n_neighbors),
                max_components=self.max_components,
                distance=self.distance,
                demean=self.demean,
            )
        except (MlsynthConfigError, MlsynthDataError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"LPCA estimation failed: {exc}") from exc

        plot_config = self.config.resolved_plot()
        if plot_config.xlabel is None:
            plot_config.xlabel = self.time
        if plot_config.ylabel is None:
            plot_config.ylabel = self.outcome
        object.__setattr__(results, "plot_config", plot_config)
        if self.display_graphs:
            results.plot()
        return results
