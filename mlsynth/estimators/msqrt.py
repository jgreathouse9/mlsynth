"""MSQRT: Multivariate Square-root Lasso Synthetic Control (Shen et al. 2025).

Shen, Z., Song, X. & Abadie, A. (2025). *"Efficiently Learning Synthetic
Control Models for High-dimensional Disaggregated Data."* arXiv:2510.22828.

MSQRT targets the regime where the panel is *disaggregated* -- many fine-grained
units (often with the number of donors :math:`n` comparable to or exceeding the
number of pre-treatment periods :math:`T_0`) and several units treated at the
same time (a block design). Rather than fitting each treated unit separately,
it stacks all treated units into a single matrix regression
:math:`Y = X\\Theta + E` and estimates the donor-weight matrix
:math:`\\Theta` by *Multivariate Square-root Lasso*,

.. math::

   \\widehat\\Theta \\;=\\; \\arg\\min_{\\Theta}\\;
     \\tfrac{1}{\\sqrt{T_0}}\\,\\lVert Y - X\\Theta \\rVert_{*}
     \\;+\\; \\lambda \\sum_{i,j} \\lvert \\Theta_{ij}\\rvert ,

where :math:`\\lVert\\cdot\\rVert_{*}` is the nuclear norm. The nuclear-norm
("square-root") loss is *pivotal*: the optimal penalty level does not depend on
the unknown noise variance, so a single :math:`\\lambda` regularises the whole
weight matrix while the element-wise :math:`\\ell_1` term performs donor
selection. Borrowing strength across treated units in one joint problem is what
makes the high-dimensional, multiple-treated case tractable and efficient.

The treatment effect is the mean post-treatment gap (observed minus synthetic)
over the treated cells -- an ATT. mlsynth selects :math:`\\lambda` by
rolling-origin cross-validation on the pre-period and, optionally, attaches
CFPT/scpi prediction intervals (Cattaneo, Feng, Palomba & Titiunik 2025) for
all four predictands -- TSUS, TAUS, TSUA and the overall ATT (TAUA) -- plus
simultaneous bands. For MSQRT only the out-of-sample (post-treatment noise)
error is modelled; see :mod:`mlsynth.utils.scpi_helpers`.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import MSQRTConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.msqrt_helpers.pipeline import run_msqrt
from ..utils.msqrt_helpers.plotter import plot_msqrt
from ..utils.msqrt_helpers.setup import prepare_msqrt_inputs
from ..utils.msqrt_helpers.structures import MSQRTResults


class MSQRT:
    """Multivariate Square-root Lasso Synthetic Control estimator.

    Parameters
    ----------
    config : MSQRTConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.MSQRTConfig`.
    """

    def __init__(self, config: Union[MSQRTConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MSQRTConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid MSQRT configuration: {exc}"
                ) from exc
        self.config: MSQRTConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.lambda_ = config.lambda_
        self.n_lambda: int = config.n_lambda
        self.cv_initial_train = config.cv_initial_train
        self.cv_val_window = config.cv_val_window
        self.cv_step = config.cv_step
        self.cv_folds = config.cv_folds
        self.inference: bool = config.inference
        self.alpha: float = config.alpha
        self.time_dependence: str = config.time_dependence
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> MSQRTResults:
        """Run MSQRT and return :class:`MSQRTResults`."""
        try:
            inputs = prepare_msqrt_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
            )
            results = run_msqrt(
                inputs=inputs,
                lambda_=self.lambda_,
                n_lambda=self.n_lambda,
                cv_initial_train=self.cv_initial_train,
                cv_val_window=self.cv_val_window,
                cv_step=self.cv_step,
                cv_folds=self.cv_folds,
                inference=self.inference,
                alpha=self.alpha,
                time_dependence=self.time_dependence,
            )
            if self.display_graphs:
                plot_msqrt(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=self.counterfactual_color,
                    save=self.save,
                    time_axis_label=self.time,
                    treatment_label=self.treat,
                    unit_label=self.unitid,
                    outcome_label=self.outcome,
                )
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"MSQRT estimation failed: {exc}"
            ) from exc
