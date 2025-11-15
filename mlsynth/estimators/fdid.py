import numpy as np
import pandas as pd # Added for type hinting
from scipy.stats import norm
from typing import Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass

from ..utils.selector_helpers import DID_selector
from ..utils.estutils import DID

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import plot_estimates
from ..config_models import ( # Import the Pydantic models
    FDIDConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)
from ..exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, MlsynthPlottingError
from pydantic import ValidationError


class FDID:
    """
    Implements the Forward Difference-in-Differences (FDID) estimator.

    This class computes estimates for three related methods:
    1.  Forward Difference-in-Differences (FDID): Uses a forward selection
        procedure to choose a subset of control units to construct the
        counterfactual for the treated unit.
    2.  Standard Difference-in-Differences (DID): Uses all available control
        units.
    3.  Augmented Difference-in-Differences (ADID): Also uses all available
        control units but incorporates them differently in the regression.

    The `fit` method returns results for all three estimators. Configuration is
    managed via a Pydantic model. Key instance attributes are derived from this
    configuration object. Refer to the `__init__` method for a more detailed parameter
    description.

    References
    ----------
    Li, K. T. (2023). "Frontiers: A Simple Forward Difference-in-Differences Method."
    *Marketing Science* 43(2):267-279.

    Li, K. T. & Van den Bulte, C. (2023). "Augmented Difference-in-Differences."
    *Marketing Science* 42:4, 746-767.

    Examples
    --------
    >>> from mlsynth import FDID
    >>> from mlsynth.config_models import FDIDConfig
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data for demonstration
    >>> data = pd.DataFrame({
    ...     'unit': np.repeat(np.arange(1, 4), 10), # 3 units
    ...     'time': np.tile(np.arange(1, 11), 3),   # 10 time periods
    ...     'outcome': np.random.rand(30) + np.repeat(np.arange(0,3),10)*0.5,
    ...     'treated_unit_1': ((np.repeat(np.arange(1, 4), 10) == 1) & \
    ...                        (np.tile(np.arange(1, 11), 3) >= 6)).astype(int)
    ... })
    >>> fdid_config = FDIDConfig(
    ...     df=data,
    ...     outcome='outcome',
    ...     treat='treated_unit_1',
    ...     unitid='unit',
    ...     time='time',
    ...     display_graphs=False # Typically True, False for non-interactive examples
    ... )
    >>> estimator = FDID(config=fdid_config)
    >>> # Results can be obtained by calling estimator.fit()
    >>> # results_list = estimator.fit() # doctest: +SKIP
    """

    def __init__(self, config: FDIDConfig) -> None:
        """
        Initializes the FDID estimator with a configuration object.

        The configuration object `config` bundles all parameters for the estimator.
        `FDIDConfig` inherits from `BaseEstimatorConfig`.

        Parameters
        ----------
        config : FDIDConfig
            A Pydantic model instance containing the configuration parameters.
            The fields are inherited from `BaseEstimatorConfig`:
            df : pd.DataFrame
                The input panel data. Must contain columns for outcome, treatment
                indicator, unit identifier, and time identifier.
            outcome : str
                Name of the outcome variable column in `df`.
            treat : str
                Name of the binary treatment indicator column in `df`.
            unitid : str
                Name of the unit identifier (e.g., country, individual ID) column in `df`.
            time : str
                Name of the time period column in `df`.
            display_graphs : bool, default=True
                Whether to display plots of the results after fitting.
            save : Union[bool, str], default=False
                If False, plots are not saved. If True, plots are saved with default names.
                If a string, it's used as a prefix for saved plot filenames.
                (Note: The internal `plot_estimates` function might handle more complex
                types like Dict for `save`, but `FDIDConfig` defines it as `Union[bool, str]`).
            counterfactual_color : str, default="red"
                Color for the counterfactual line(s) in plots.
                (Note: The internal `plot_estimates` function might handle a list of colors,
                but `FDIDConfig` defines it as `str`).
            treated_color : str, default="black"
                Color for the treated unit line in plots.
        """
        if isinstance(config, dict):
            config = FDIDConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.outcome: str = config.outcome
        self.treated: str = config.treat # Maps to 'treat' from config
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, Dict[str, str]] = config.save # Kept Union for flexibility

    def fdid_to_pydantic(self, result_dict: Dict, method_name: str = "FDID") -> BaseEstimatorResults:
        """
        Convert a FDID/DID result dictionary to a BaseEstimatorResults Pydantic model.

        Parameters
        ----------
        result_dict : dict
            The DID/FDID output dictionary.
        method_name : str, optional
            Name of the method (default is "FDID").

        Returns
        -------
        BaseEstimatorResults
            Standardized Pydantic model with all results mapped.
        """
        # --- Effects ---
        effects_data = result_dict.get("effects", {})
        effects_model = EffectsResults(
            att=effects_data.get("ATT"),
            att_percent=effects_data.get("Percent ATT"),
            additional_effects={k: v for k, v in effects_data.items() if k not in ["ATT", "Percent ATT"]}
        )

        # --- Fit diagnostics ---
        fit_model = FitDiagnosticsResults(
            r_squared_pre=result_dict.get("R2"),
            rmse_pre=result_dict.get("rmse")
        )

        # --- Time series ---
        fit_vectors = result_dict.get("fit_vectors", {})
        observed = fit_vectors.get("Observed Unit")
        counterfactual = fit_vectors.get("Counterfactual")
        gap = fit_vectors.get("Gap")

        time_series_model = TimeSeriesResults(
            observed_outcome=observed.flatten() if observed is not None else None,
            counterfactual_outcome=counterfactual.flatten() if counterfactual is not None else None,
            estimated_gap=gap[:, 0] if gap is not None else None,
            time_periods=np.arange(len(observed)) if observed is not None else None
        )

        # --- Weights ---
        selected_names = result_dict.get("selected_names", set())
        donor_weights = {name: 1.0 / len(selected_names) for name in selected_names} if selected_names else None

        weights_model = WeightsResults(
            donor_weights=donor_weights,
            summary_stats={
                "num_selected": len(selected_names),
                "total_weight": sum(donor_weights.values()) if donor_weights else None
            } if donor_weights else None
        )

        # --- Inference ---
        inference_data = result_dict.get("Inference Results", {})
        inference_model = InferenceResults(
            p_value=inference_data.get("P-Value"),
            ci_lower=inference_data.get("95 LB"),
            ci_upper=inference_data.get("95 UB"),
            standard_error=inference_data.get("SE"),
            details={k: v for k, v in inference_data.items() if k not in ["P-Value", "95 LB", "95 UB", "SE"]}
        )

        # --- Method details ---
        method_model = MethodDetailsResults(
            method_name=method_name,
            is_recommended=True,
            parameters_used={"selected_names": list(selected_names)}
        )

        # --- Assemble BaseEstimatorResults ---
        base_results = BaseEstimatorResults(
            effects=effects_model,
            fit_diagnostics=fit_model,
            time_series=time_series_model,
            weights=weights_model,
            inference=inference_model,
            method_details=method_model,
            raw_results=result_dict
        )

        return base_results

    @dataclass(frozen=True)
    class FDIDOutput:
        results: Dict[str, 'BaseEstimatorResults']
        prepped_data: dict


    def fit(self) -> FDIDOutput:
        """
        Fits FDID and standard DID models, constructs results objects, optionally plots outcomes,
        and returns a dataclass with results and prepped data.
        """
        # Initialize outputs
        FDID_Result: Optional[BaseEstimatorResults] = None
        DID_Result: Optional[BaseEstimatorResults] = None
        prepared_data: Optional[dict] = None

        try:
            # Step 1: Balance the panel data
            balance(self.df, self.unitid, self.time)

            # Step 2: Prepare data into matrix format
            prepared_data = dataprep(self.df, self.unitid, self.time, self.outcome, self.treated)

            # Step 3: Forward selection for FDID
            selectorresults = DID_selector(
                treated_outcome=prepared_data["y"],
                control_outcomes=prepared_data["donor_matrix"],
                T0=prepared_data["pre_periods"],
                control_names=prepared_data["donor_names"]
            )

            # --- FDID ---
            if selectorresults['best_candidate_info']:
                best_index = next(iter(selectorresults['best_candidate_info']))
                FDID_Design = selectorresults['best_candidate_info'][best_index]

                FDID_Result = BaseEstimatorResults(
                    effects=EffectsResults(
                        att=FDID_Design['effects']['ATT'],
                        att_percent=FDID_Design['effects']['Percent ATT'],
                        additional_effects={'SATT': FDID_Design['effects']['SATT']}
                    ),
                    fit_diagnostics=FitDiagnosticsResults(
                        r_squared_pre=FDID_Design['R2'],
                        rmse_pre=FDID_Design['rmse']
                    ),
                    time_series=TimeSeriesResults(
                        observed_outcome=FDID_Design['fit_vectors']['Observed Unit'].flatten(),
                        counterfactual_outcome=FDID_Design['fit_vectors']['Counterfactual'].flatten(),
                        estimated_gap=FDID_Design['fit_vectors']['Gap'][:, 0],
                        time_periods=np.arange(len(FDID_Design['fit_vectors']['Observed Unit']))
                    ),
                    weights=WeightsResults(
                        donor_weights={name: 1.0 / len(FDID_Design['selected_names']) for name in FDID_Design['selected_names']},
                        summary_stats={'num_selected': len(FDID_Design['selected_names']), 'total_weight': 1.0}
                    ),
                    inference=InferenceResults(
                        p_value=FDID_Design['Inference Results']['P-Value'],
                        ci_lower=FDID_Design['Inference Results']['95 LB'],
                        ci_upper=FDID_Design['Inference Results']['95 UB'],
                        standard_error=FDID_Design['Inference Results']['SE'],
                        details={
                            'Intercept': FDID_Design['Inference Results']['Intercept'],
                            'Width': FDID_Design['Inference Results']['Width']
                        }
                    ),
                    method_details=MethodDetailsResults(
                        method_name="FDID",
                        is_recommended=True,
                        parameters_used={'selected_names': list(FDID_Design['selected_names'])}
                    )
                )
            else:
                raise MlsynthEstimationError("No FDID candidates found during forward selection.")

            # --- DID ---
            last_candidate_group = selectorresults['candidate_dict'][max(selectorresults['candidate_dict'].keys())]
            last_candidate_index = max(last_candidate_group, key=lambda j: last_candidate_group[j]['R2'])
            DID_Design = last_candidate_group[last_candidate_index]

            # Convert to Pydantic result
            DID_Result = self.fdid_to_pydantic(DID_Design, method_name="DID")

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError) as e:
            raise e
        except ValidationError as e:
            raise MlsynthEstimationError(f"Error creating results structure: {str(e)}") from e
        except Exception as e:
            raise MlsynthEstimationError(f"Unexpected error during FDID/DID estimation: {str(e)}") from e

        # --- Plotting ---
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
                        FDID_Result.time_series.counterfactual_outcome.flatten(),
                        DID_Result.time_series.counterfactual_outcome.flatten()
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
                print(f"Warning: Plotting failed - {str(e)}")
            except Exception as e:
                print(f"Warning: Unexpected plotting error - {str(e)}")

        # --- Return ---
        results_dict = {"FDID": FDID_Result, "DID": DID_Result}
        return self.FDIDOutput(results=results_dict, prepped_data=prepared_data)
