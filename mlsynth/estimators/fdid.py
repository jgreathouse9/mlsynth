from dataclasses import dataclass
from typing import Dict, Optional, Union, List
import pandas as pd
from pydantic import ValidationError

from ..utils.resultutils import DID_org
from ..utils.estutils import fast_DID_selector
from ..utils.datautils import balance, dataprep
from ..utils.resultutils import plot_estimates
from ..exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError
)
from ..config_models import FDIDConfig


@dataclass(frozen=True)
class FDIDOutput:
    """
    Container for the results of FDID and DID estimation.

    Attributes
    ----------
    results : Dict[str, BaseEstimatorResults]
        Dictionary containing the estimation results. Expected keys are:
        - "FDID": Results for Forward Difference-in-Differences.
        - "DID": Results for standard Difference-in-Differences.
    prepped_data : dict
        Dictionary containing the prepared panel data and matrices used
        internally during estimation.
    """
    results: Dict[str, 'BaseEstimatorResults']
    prepped_data: dict

class FDID:
    """
    Forward Difference-in-Differences (FDID) estimator.

    Implements FDID, which performs forward selection of control units to construct
    the counterfactual for a treated unit, alongside standard DID estimation. Optionally,
    it can plot the observed and counterfactual outcomes.

    Parameters
    ----------
    config : FDIDConfig or dict
        Configuration object or dictionary containing all necessary parameters:
        - df : pd.DataFrame
            Panel data containing outcome, treatment indicator, unit, and time.
        - outcome : str
            Column name of the outcome variable.
        - treat : str
            Column name of the binary treatment indicator.
        - unitid : str
            Column name of the unit identifier.
        - time : str
            Column name of the time period.
        - display_graphs : bool, default=True
            Whether to display plots after fitting.
        - save : Union[bool, str], default=False
            Controls whether plots are saved.
        - counterfactual_color : str or list of str, default="red"
            Color for the counterfactual line(s) in plots.
        - treated_color : str, default="black"
            Color for the treated unit line in plots.

    Attributes
    ----------
    df : pd.DataFrame
        Input panel data.
    unitid : str
        Name of the unit identifier column.
    time : str
        Name of the time period column.
    outcome : str
        Name of the outcome variable column.
    treated : str
        Name of the treatment indicator column.
    counterfactual_color : str or list of str
        Color(s) for counterfactual series in plots.
    treated_color : str
        Color for the treated unit in plots.
    display_graphs : bool
        Whether to display plots after fitting.
    save : Union[bool, dict]
        Plot-saving configuration.
    config : FDIDConfig
        Original configuration object.

    References
    ----------
    Li, K. T. (2023). Frontiers: A Simple Forward Difference-in-Differences Method.
    Marketing Science, 43(2), 267-279. https://doi.org/10.1287/mksc.2022.0212

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
    ...     "display_graphs": True,
    ...     "save": False,
    ...     "counterfactual_color": ["red", "blue"]
    ... }
    >>> results = FDID(config).fit()
    """

    def __init__(self, config: FDIDConfig) -> None:
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

    def fit(self) -> FDIDOutput:
        """
        Fit the FDID and standard DID models.
    
        This method performs the following steps:
        1. Balances the panel data to ensure consistent unit-time structure.
        2. Prepares the outcome and donor matrices for estimation.
        3. Runs forward selection to identify the best donor units for FDID.
        4. Constructs FDID and DID results using `DID_org`.
        5. Optionally plots the observed and counterfactual outcomes.
    
        Returns
        -------
        FDIDOutput
            Dataclass containing:
            - `results`: A dictionary with "FDID" and "DID" keys containing
              the estimation results.
            - `prepped_data`: Dictionary of prepared data used internally.
    
        Raises
        ------
        MlsynthDataError
            Raised if panel data balancing or data preparation fails.
        MlsynthEstimationError
            Raised if FDID forward selection fails, results cannot be constructed,
            or an unexpected error occurs during estimation.
        MlsynthPlottingError
            Raised if plotting fails (warning issued instead of exception).
        """
        # Step 1: Balance
        try:
            balance(self.df, self.unitid, self.time)
        except Exception as e:
            raise MlsynthDataError(f"Error balancing panel data: {str(e)}") from e

        # Step 2: Prepare matrices
        try:
            prepared_data = dataprep(self.df, self.unitid, self.time, self.outcome, self.treated)

            # Must have at least two pre-periods to compute first differences reliably.
            if prepared_data['pre_periods'] is None or prepared_data['pre_periods'] < 2:
                raise MlsynthEstimationError("Insufficient pre-periods for estimation.")

        except MlsynthEstimationError:
            # <-- IMPORTANT: re-raise untouched
            raise
        except Exception as e:
            raise MlsynthDataError(f"Error preparing data matrices: {str(e)}") from e
    
        # Step 3: Forward selection
        try:
            results = fast_DID_selector(prepared_data["y"],
                                        prepared_data["donor_matrix"],
                                        prepared_data["pre_periods"],
                                        donor_names=prepared_data["donor_names"],
                                        verbose=self.verbose)

            gathered = {
                key: DID_org(
                    result_dict=results[key],  # pass the nested FDID or DID dict
                    preppeddict=prepared_data,
                    method_name=key
                )
                for key in ["FDID", "DID"]
                if key in results
            }

        except ValidationError as e:
            raise MlsynthEstimationError(f"Error creating results structure: {str(e)}") from e
        except Exception as e:
            raise MlsynthEstimationError(f"Unexpected error during FDID/DID construction: {str(e)}") from e
    
        # Step 5: Plotting
        if self.display_graphs:
            try:
                plot_estimates(
                    processed_data_dict=prepared_data,
                    time_axis_label=self.time,
                    unit_identifier_column_name=self.unitid,
                    outcome_variable_label=self.outcome,
                    treatment_name_label=self.treated,
                    treated_unit_name=prepared_data["treated_unit_name"],
                    observed_outcome_series=prepared_data["y"],
                    counterfactual_series_list=[
                        gathered['FDID'].time_series.counterfactual_outcome.flatten(),gathered['DID'].time_series.counterfactual_outcome.flatten()
                    ],
                    estimation_method_name="FDID",
                    counterfactual_names=[
                        f"FDID {prepared_data['treated_unit_name']}",
                        f"DID {prepared_data['treated_unit_name']}"
                    ],
                    treated_series_color=self.treated_color,
                    save_plot_config=self.save,
                    counterfactual_series_colors=self.counterfactual_color
                )
            except (MlsynthPlottingError, MlsynthDataError) as e:
                import warnings
                warnings.warn(f"Plotting failed: {str(e)}", UserWarning)
            except Exception as e:
                import warnings
                warnings.warn(f"Unexpected plotting error: {str(e)}", UserWarning)
    
        return FDIDOutput(results=gathered, prepped_data=prepared_data)
