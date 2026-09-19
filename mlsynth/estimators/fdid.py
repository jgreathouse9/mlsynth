"""Forward Difference-in-Differences (FDID) estimator.

Implements the forward-selection difference-in-differences method of
Li (2023), *Frontiers: A Simple Forward Difference-in-Differences
Method*, Marketing Science. FDID greedily grows the control group one
donor at a time, keeping the subset that maximises pre-treatment fit,
and reports both the forward-selected estimate (``FDID``) and the
textbook all-donor difference-in-differences benchmark (``DID``), each
with Li (2023) analytical standard errors. Set ``inference="hac"`` for a
standard error robust to a serially correlated parallel-trends residual,
which Li's formula does not price in.

A panel with several treated units is staggered, and the fit routes to the
Web Appendix C extension: one Forward DID per treated unit, aggregated on a
balanced event clock. See :mod:`mlsynth.utils.fdid_helpers.staggered` for
what that adds beyond the appendix -- donor eligibility enforced at
selection, cohort-pooled selection, and a covariance that prices the
dependence between units drawing on one donor pool. That surface is
experimental.

The estimator is a thin orchestration layer over
:mod:`mlsynth.utils.fdid_helpers`: it validates configuration, prepares
the panel, runs forward selection, assembles a typed
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDResults`, and
optionally plots the counterfactuals.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..exceptions import MlsynthDataError, MlsynthEstimationError
from ..utils.fdid_helpers.config import FDIDConfig
from ..utils.fdid_helpers import (
    FDIDResults,
    FDIDStaggeredResults,
    assemble_fdid_results,
    fit_staggered,
    forward_did_select,
    prepare_fdid_inputs,
    prepare_panel,
    prepare_staggered_inputs,
)


class FDID:
    """Forward Difference-in-Differences (FDID) estimator.

    Parameters
    ----------
    config : FDIDConfig or dict
        Validated configuration (or a compatible dictionary). See
        :class:`mlsynth.utils.fdid_helpers.config.FDIDConfig` for the available
        fields (``df``, ``outcome``, ``treat``, ``unitid``, ``time``,
        ``display_graphs``, ``save``, ``counterfactual_color``,
        ``treated_color``, ``verbose``, ``inference``, ``lrvar_lag``), plus
        ``selection`` / ``pooling_weight`` / ``anticipation`` /
        ``max_horizon``, which apply on a staggered panel.

    References
    ----------
    Li, K. T. (2023). Frontiers: A Simple Forward Difference-in-Differences
    Method. Marketing Science, 43(2), 267-279.
    https://doi.org/10.1287/mksc.2022.0212

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth import FDID
    >>> url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
    >>> data = pd.read_csv(url)
    >>> config = {
    ...     "df": data,
    ...     "outcome": data.columns[2],
    ...     "treat": data.columns[-1],
    ...     "unitid": data.columns[0],
    ...     "time": data.columns[1],
    ...     "display_graphs": False,
    ... }
    >>> results = FDID(config).fit()
    >>> round(results.att, 3)  # doctest: +SKIP
    """

    def __init__(self, config: Union[FDIDConfig, dict]) -> None:
        if isinstance(config, dict):
            config = FDIDConfig(**config)
        self.config = config
        self.df: pd.DataFrame = config.df
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.outcome: str = config.outcome
        self.treated: str = config.treat
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, dict] = config.save
        self.verbose: bool = config.verbose
        self.inference: str = config.inference
        self.lrvar_lag = config.lrvar_lag

    def fit(self) -> Union[FDIDResults, FDIDStaggeredResults]:
        """Run forward selection and return the typed FDID results.

        Returns
        -------
        FDIDResults or FDIDStaggeredResults
            With one treated unit, a container exposing the forward-selected
            ``fdid`` fit (primary) and the all-donor ``did`` benchmark, plus
            convenience aliases (``att``, ``att_se``, ``counterfactual``,
            ``gap``, ``donor_weights``).

            With several treated units the panel is staggered, so the result
            carries one fit per treated unit, a balanced event study, and an
            overall effect whose standard error prices the dependence between
            units drawing on a shared donor pool. That surface is
            experimental; see
            :mod:`mlsynth.utils.fdid_helpers.staggered`.

        Raises
        ------
        MlsynthDataError
            If panel balancing or data preparation fails.
        MlsynthEstimationError
            If there are too few pre-periods or forward selection fails.
        """
        prepped = prepare_panel(
            df=self.df,
            outcome=self.outcome,
            treat=self.treated,
            unitid=self.unitid,
            time=self.time,
        )
        if "cohorts" in prepped:
            return self._fit_staggered(prepped)

        inputs = prepare_fdid_inputs(
            df=self.df,
            outcome=self.outcome,
            treat=self.treated,
            unitid=self.unitid,
            time=self.time,
            verbose=self.verbose,
        )

        try:
            selector_output = forward_did_select(
                inputs.y,
                inputs.donor_matrix,
                inputs.pre_periods,
                donor_names=list(inputs.donor_names),
                verbose=self.verbose,
                inference=self.inference,
                lrvar_lag=self.lrvar_lag,
            )
        except Exception as e:  # noqa: BLE001 - surface as estimation failure
            raise MlsynthEstimationError(
                f"Unexpected error during FDID/DID estimation: {str(e)}"
            ) from e

        results = assemble_fdid_results(selector_output, inputs)

        # Attach plotting context so result.plot() is self-contained and styled
        # from the (possibly nested) config. Fill axis labels from the column
        # names only when the user has not set them.
        pc = self.config.resolved_plot()
        if pc.xlabel is None:
            pc.xlabel = self.time
        if pc.ylabel is None:
            pc.ylabel = self.outcome
        object.__setattr__(results, "plot_config", pc)
        if results.time_series is not None:
            pre = int(inputs.pre_periods)
            labels = list(inputs.time_labels)
            results.time_series.intervention_time = (
                labels[pre] if pre < len(labels) else labels[-1]
            )

        if self.display_graphs:
            results.plot()

        return results

    def _fit_staggered(self, prepped: dict) -> FDIDStaggeredResults:
        """Fit every treated unit of a staggered panel and aggregate.

        ``dataprep`` hands back cohorts when the treatment column marks more
        than one unit. Li's Web Appendix C is the starting point -- one Forward
        DID per treated unit -- and the additions live in
        :func:`mlsynth.utils.fdid_helpers.staggered.fit_staggered`.
        """
        inputs = prepare_staggered_inputs(prepped, verbose=self.verbose)
        # The aggregate averages over event horizons, so the analytic variance
        # divides by the horizon count as though the horizons were independent
        # draws. Under an AR(1) residual with coefficient 0.8 that took nominal
        # 95% coverage of the overall ATT to 0.62, against 0.81 for the HAC
        # form, which costs nothing when the residual is in fact uncorrelated
        # (0.94 against 0.93). So the staggered path prices autocovariances
        # unless the caller asked for the analytic form by name.
        inference = (self.inference if "inference" in self.config.model_fields_set
                     else "hac")
        try:
            results = fit_staggered(
                inputs,
                selection=self.config.selection,
                pooling_weight=self.config.resolved_pooling_weight,
                anticipation=self.config.anticipation,
                max_horizon=self.config.max_horizon,
                inference=inference,
                lrvar_lag=self.lrvar_lag,
            )
        except (MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as e:  # noqa: BLE001 - surface as estimation failure
            raise MlsynthEstimationError(
                f"Unexpected error during staggered FDID estimation: {str(e)}"
            ) from e

        # Leave the axis labels unset: the event study's axes are an event
        # clock and a treatment effect, neither of which is the time or
        # outcome column, and the plotter names them.
        object.__setattr__(results, "plot_config", self.config.resolved_plot())

        if self.display_graphs:
            results.plot()

        return results
