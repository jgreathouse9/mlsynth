import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Union, Optional
import cvxpy as cp  # For catching solver errors
from pydantic import ValidationError  # For catching Pydantic errors if models are created internally
from dataclasses import dataclass

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates
from ..utils.inferutils import quantileconformal_intervals
from ..utils.estutils import fsSCM
from ..exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import (
    FSCMConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class FSCM:
    """
    Estimates Average Treatment Effect on the Treated (ATT) using the Forward Selected Synthetic Control Method (FSCM).

    This approach, based on Cerulli (2024), optimally selects a subset of donor
    units using a forward selection algorithm. It begins by identifying the
    single best-fitting donor unit. Then, it iteratively adds other donor units
    to the pool if their inclusion improves the predictive fit (minimizes MSE)
    for the treated unit's pre-treatment outcomes. The final weights for the
    selected donor pool and the counterfactual outcome series are derived using
    a constrained optimization procedure (SIMPLEX).

    Attributes
    ----------
    config : FSCMConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    save : Union[bool, str], default False
        If False, plots are not saved. If True, plots are saved with default names.
        If a string, it's used as the directory path to save plots.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    counterfactual_color : str, default "red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    treated_color : str, default "black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)

    Methods
    -------
    fit()
        Fits the FSCM model and returns the standardized results.
    evaluate_donor(donor_index, donor_columns, y_pre, T0)
        Evaluates the Mean Squared Error (MSE) for a single potential donor.
    fSCM(y_pre, Y0, T0)
        Performs the core Forward Selected Synthetic Control Method optimization.

    References
    ----------
    Cerulli, Giovanni. "Optimal initial donor selection for the synthetic control method."
    *Economics Letters*, 244 (2024): 111976. https://doi.org/10.1016/j.econlet.2024.111976

    Examples
    --------
    >>> from mlsynth import FSCM
    >>> from mlsynth.config_models import FSCMConfig
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
    >>> fscm_config = FSCMConfig(
    ...     df=data,
    ...     outcome='outcome',
    ...     treat='treated_unit_1',
    ...     unitid='unit',
    ...     time='time',
    ...     display_graphs=False # Typically True, False for non-interactive examples
    ... )
    >>> estimator = FSCM(config=fscm_config)
    >>> # Results can be obtained by calling estimator.fit()
    >>> # results = estimator.fit() # doctest: +SKIP
    """

    def __init__(self, config: FSCMConfig) -> None:  # Changed to FSCMConfig
        """
        Initializes the FSCM estimator with a configuration object.

        Parameters
        ----------
        config : FSCMConfig
            A Pydantic model instance containing all configuration parameters
            for the FSCM estimator. `FSCMConfig` inherits from `BaseEstimatorConfig`.

        References
        ----------
        Shi, Zhentao, and Jingyi Huang. 2023.
        "Forward-selected panel data approach for program evaluation."
        Journal of Econometrics 234 (2): 512–535.
        https://doi.org/10.1016/j.jeconom.2021.04.009

        Cerulli, Giovanni. 2024.
        “Optimal initial donor selection for the synthetic control method.”
        Economics Letters, 244: 111976.
        https://doi.org/10.1016/j.econlet.2024.111976

        Ben-Michael, Eli, Avi Feller, and Jesse Rothstein. 2021.
        "The Augmented Synthetic Control Method."
        Journal of the American Statistical Association 116 (536): 1789–1803.
        https://doi.org/10.1080/01621459.2021.1929245

            Core panel data structure:
            --------------------------
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

            Plotting and display:
            ---------------------
            display_graphs : bool, default=True
                Whether to display plots of the results after fitting.
            save : Union[bool, str], default=False
                If False, plots are not saved. If True, plots are saved with default names.
                If a string, it's used as a prefix for saved plot filenames.
            counterfactual_color : str, default="red"
                Color for the counterfactual line(s) in plots.
            treated_color : str, default="black"
                Color for the treated unit line in plots.

            Forward selection configuration:
            --------------------------------
            use_augmented : bool, default=False
                Whether to refine weights after selection using affine hull optimization.
            selection_fraction : float, default=1.0
                Fraction of the donor pool to use in forward selection. Reducing this can
                substantially reduce runtime in high-dimensional settings.
            full_selection : bool, default=False
                Whether to force forward selection to run through all possible donor subsets,
                even after mBIC fails to improve. Can be computationally expensive.

            Affine refinement tuning (used only if `use_augmented=True`):
            -------------------------------------------------------------
            bo_n_iter : int, default=25
                Number of iterations to run in Bayesian optimization.
            bo_initial_evals : int, default=5
                Number of initial random evaluations before surrogate modeling begins.
        """

        if isinstance(config, dict):
            config = FSCMConfig(**config)  # convert dict to config object
        self.config = config  # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color  # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save  # Align with BaseEstimatorConfig
        # FSCM-specific config
        self.use_augmented: bool = config.use_augmented
        self.full_selection: bool = config.full_selection
        self.selection_fraction: float = config.selection_fraction

        # Affine tuning parameters for augmented FSCM
        self.bo_n_iter: int = config.bo_n_iter
        self.bo_initial_evals: int = config.bo_initial_evals

    def _build_master_results(self,
            weight_dicts: Dict[str, any],
            prepped_data: Dict[str, any],
            resultfscm: Dict[str, any],
            donor_names: list[str]
    ) -> BaseEstimatorResults:
        """
        Construct a standardized master BaseEstimatorResults object
        from weight vectors/dicts and prepped data for all SCM methods.
        """
        Y = prepped_data['donor_matrix']
        y = prepped_data['y']
        T_pre = prepped_data['pre_periods']
        T_post = prepped_data['post_periods']

        sub_results: Dict[str, BaseEstimatorResults] = {}

        for method_name, w in weight_dicts.items():
            # Convert numpy array to dict if needed
            if isinstance(w, np.ndarray):
                w = {name: float(val) for name, val in zip(donor_names, w)}

            # Compute counterfactual
            weights_array = np.array(list(w.values()))
            y_hat = Y @ weights_array

            # Compute effects
            eff_dict, fit_dict, vectors_info = effects.calculate(y, y_hat, T_pre, T_post)

            # Wrap effects into standardized EffectsResults
            effects_res = EffectsResults(
                att=eff_dict.get('ATT'),
                att_percent=eff_dict.get('Percent ATT'),
                att_std_err=eff_dict.get('ATT_SE'),
                additional_effects={k: v for k, v in eff_dict.items() if k not in ['ATT', 'Percent ATT', 'ATT_SE']}
            )

            # Wrap fit dictionary into FitDiagnosticsResults
            fit_res = FitDiagnosticsResults(
                rmse_pre=fit_dict.get('T0 RMSE'),
                rmse_post=fit_dict.get('T1 RMSE'),
                r_squared_pre=fit_dict.get('R-Squared'),
                additional_metrics={k: v for k, v in fit_dict.items() if k not in ['T0 RMSE', 'T1 RMSE', 'R-Squared']}
            )

            # Wrap time series
            ts_res = TimeSeriesResults(
                observed_outcome=vectors_info.get('Observed Unit'),
                counterfactual_outcome=vectors_info.get('Counterfactual'),
                estimated_gap=vectors_info.get('Gap'),
                time_periods=vectors_info.get('Time Periods')
            )

            # Additional outputs for FSCM / augmented SCM
            additional_outputs = {}
            if method_name == "Forward SCM":
                additional_outputs['candidate_dict'] = resultfscm['Forward SCM']['candidate_dict']
            if method_name == "Forward Augmented SCM":
                additional_outputs['validation_results'] = resultfscm['Forward Aumented SCM'].get('validation_results')
                additional_outputs['lambda'] = resultfscm['Forward Aumented SCM'].get('lambda')

            # Wrap weights
            weights_res = WeightsResults(donor_weights=w)

            # Construct BaseEstimatorResults for this method
            sub_results[method_name] = BaseEstimatorResults(
                effects=effects_res,
                fit_diagnostics=fit_res,
                time_series=ts_res,
                weights=weights_res,
                method_details={"name": method_name},
                additional_outputs=additional_outputs,
                raw_results=resultfscm.get(method_name)
            )

        # Construct master results object
        master_results = BaseEstimatorResults(
            sub_method_results=sub_results,
            additional_outputs=prepped_data,
            raw_results=resultfscm
        )

        return master_results

    def fit(self) -> BaseEstimatorResults:  # Main method to fit the FSCM estimator
        try:
            # Step 1: Validate data balance (ensures each unit has the same time periods)
            balance(self.df, self.unitid, self.time)  # This can raise MlsynthDataError

            # Step 2: Prepare data using the dataprep utility
            # This separates data into treated/control, pre/post periods, and formats it.
            prepared_data_dict: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )  # This can raise MlsynthDataError or MlsynthConfigError
            # Step 3: Perform essential checks on the output of dataprep
            required_keys = ["donor_matrix", "pre_periods", "y", "donor_names", "treated_unit_name"]
            for key in required_keys:
                if key not in prepared_data_dict or prepared_data_dict[key] is None:
                    raise MlsynthEstimationError(f"Essential key '{key}' missing or None in dataprep output.")

            if not isinstance(prepared_data_dict["pre_periods"], int) or prepared_data_dict["pre_periods"] <= 0:
                raise MlsynthEstimationError(
                    f"Invalid 'pre_periods' ({prepared_data_dict['pre_periods']}) from dataprep.")

            # Extract pre-treatment outcome data
            all_donors_outcomes_matrix_pre_treatment = prepared_data_dict["donor_matrix"][
                                                       : prepared_data_dict["pre_periods"]]
            treated_outcome_pre_treatment_vector = prepared_data_dict["y"][: prepared_data_dict["pre_periods"]]
            num_pre_treatment_periods = prepared_data_dict["pre_periods"]

            if all_donors_outcomes_matrix_pre_treatment.shape[0] != num_pre_treatment_periods or\
                    treated_outcome_pre_treatment_vector.shape[0] != num_pre_treatment_periods:
                raise MlsynthEstimationError(
                    "Mismatch in pre-treatment period lengths between donor matrix and treated vector.")

            if all_donors_outcomes_matrix_pre_treatment.shape[1] == 0:
                raise MlsynthEstimationError("No donor units available after data preparation.")

            resultfscm = fsSCM(prepared_data_dict["y"],
                               prepared_data_dict["donor_matrix"],
                               num_pre_treatment_periods,
                               prepared_data_dict["donor_names"], selection_fraction=self.selection_fraction)

            weight_dicts = {
                "Convex SCM": resultfscm['Convex SCM']['weight_dict'],
                "Forward Augmented SCM": resultfscm['Forward Aumented SCM']['weight_dict'],
                "Forward SCM": resultfscm['Forward SCM']['weights']
            }

            master_results = self._build_master_results(weight_dicts, prepared_data_dict, resultfscm, prepared_data_dict["donor_names"])

            # Determine which counterfactuals to plot based on L2 similarity
            convex_cf = master_results.sub_method_results['Convex SCM'].time_series.counterfactual_outcome
            fwd_cf = master_results.sub_method_results['Forward SCM'].time_series.counterfactual_outcome
            aug_cf = master_results.sub_method_results['Forward Augmented SCM'].time_series.counterfactual_outcome

            # Flatten in advance
            convex_cf = np.asarray(convex_cf).reshape(-1)
            fwd_cf = np.asarray(fwd_cf).reshape(-1)
            aug_cf = np.asarray(aug_cf).reshape(-1)

            # Compute L2 distances
            d_conv_fwd = np.linalg.norm(convex_cf - fwd_cf)
            d_fwd_aug = np.linalg.norm(aug_cf - fwd_cf)

            # --- Case selection ---
            if d_conv_fwd < 1:
                # Convex ~ Forward → Forward SCM suffices
                counterfactual_vector_for_plot = [fwd_cf]
                cflist = ["Forward SCM"]

            elif d_fwd_aug < 1:
                # Augmented ~ Forward → Plot Convex + Forward
                counterfactual_vector_for_plot = [convex_cf, fwd_cf]
                cflist = ["Convex SCM", "Forward SCM"]

            else:
                # Everything is meaningfully different → Plot all three
                counterfactual_vector_for_plot = [convex_cf, fwd_cf, aug_cf]
                cflist = ["Convex SCM", "Forward SCM", "Forward Augmented SCM"]

        # Step 10: Handle specific and general exceptions during the fitting process.
        except (MlsynthDataError, MlsynthConfigError) as e:  # Propagate custom Mlsynth errors directly.
            raise e
        except KeyError as e:  # Handle errors due to missing keys in data structures.
            raise MlsynthEstimationError(f"Missing expected key in data structures: {e}") from e
        except IndexError as e:  # Handle errors due to invalid indexing.
            raise MlsynthEstimationError(
                f"Index out of bounds, likely during donor selection or data processing: {e}") from e
        except ValueError as e:  # Catch other ValueErrors (e.g., from np.dot if shapes mismatch).
            raise MlsynthEstimationError(f"ValueError during FSCM estimation: {e}") from e
        except Exception as e:  # Catch-all for any other unexpected errors.
            raise MlsynthEstimationError(f"An unexpected error occurred during FSCM fitting: {e}") from e

        # Step 11: Display graphs if requested by the user configuration.
        try:

            if self.display_graphs:

                plot_estimates(
                    processed_data_dict=prepared_data_dict,
                    time_axis_label=self.time,
                    unit_identifier_column_name=self.unitid,
                    outcome_variable_label=self.outcome,
                    treatment_name_label=self.treat,
                    treated_unit_name=prepared_data_dict["treated_unit_name"],
                    observed_outcome_series=prepared_data_dict["y"],  # Observed outcome vector.
                    counterfactual_series_list=counterfactual_vector_for_plot,  # List of counterfactual vectors.
                    estimation_method_name="FSCM",
                    counterfactual_names=cflist,  # Names for legend.
                    treated_series_color=self.treated_color,
                    counterfactual_series_colors=self.counterfactual_color,
                    save_plot_config=self.save)

        except MlsynthPlottingError as e:  # Handle specific plotting errors defined in Mlsynth.
            print(f"Warning: Plotting failed with MlsynthPlottingError: {e}")
        except MlsynthDataError as e:  # Handle data-related errors that might occur during plotting.
            print(f"Warning: Plotting failed due to data issues: {e}")
        except Exception as e:  # Catch-all for any other unexpected errors during plotting.
            print(f"Warning: An unexpected error occurred during plotting: {e}")

        @dataclass(frozen=True)
        class FSCMOutput:
            results: dict
            prepped_data: dict

        return FSCMOutput(
            results=master_results.sub_method_results,
            prepped_data=master_results.additional_outputs
        )
