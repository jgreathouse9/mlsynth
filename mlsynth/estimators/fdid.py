"""Forward Difference-in-Differences (FDID) estimator.

Implements the forward-selection difference-in-differences method of
Li (2023), *Frontiers: A Simple Forward Difference-in-Differences
Method*, Marketing Science. FDID greedily grows the control group one
donor at a time, keeping the subset that maximises pre-treatment fit,
and reports both the forward-selected estimate (``FDID``) and the
textbook all-donor difference-in-differences benchmark (``DID``), each
with Li (2023) analytical standard errors.

The estimator is a thin orchestration layer over
:mod:`mlsynth.utils.fdid_helpers`: it validates configuration, prepares
the panel, runs forward selection, assembles a typed
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDResults`, and
optionally plots the counterfactuals.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..exceptions import MlsynthEstimationError
from ..utils.fdid_helpers.config import FDIDConfig
from ..utils.fdid_helpers import (
    FDIDResults,
    assemble_fdid_results,
    forward_did_select,
    plot_fdid,
    prepare_fdid_inputs,
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
        ``treated_color``, ``verbose``).

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

    def fit(self) -> FDIDResults:
        """Run forward selection and return the typed FDID results.

        Returns
        -------
        FDIDResults
            Container exposing the forward-selected ``fdid`` fit (primary)
            and the all-donor ``did`` benchmark, plus convenience aliases
            (``att``, ``att_se``, ``counterfactual``, ``gap``,
            ``donor_weights``).

        Raises
        ------
        MlsynthDataError
            If panel balancing or data preparation fails.
        MlsynthEstimationError
            If there are too few pre-periods or forward selection fails.
        """
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
            )
        except Exception as e:  # noqa: BLE001 - surface as estimation failure
            raise MlsynthEstimationError(
                f"Unexpected error during FDID/DID estimation: {str(e)}"
            ) from e

        results = assemble_fdid_results(selector_output, inputs)

        if self.display_graphs:
            plot_fdid(
                results,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treat=self.treated,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color,
                save=self.save,
            )

        return results
