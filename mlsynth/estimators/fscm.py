import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Union, Optional
import cvxpy as cp  # For catching solver errors
from pydantic import ValidationError  # For catching Pydantic errors if models are created internally
from dataclasses import dataclass

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates, _build_fscm_results
from ..utils.estutils import fsSCM
from ..exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import (
    FSCMConfig,
    BaseEstimatorResults
)


class FSCM:
    class FSCM:
        """
        Estimates the Average Treatment Effect on the Treated (ATT) using the Forward Selected Synthetic Control Method (FSCM).

        FSCM constructs a donor pool iteratively via forward selection. At each iteration,
        it evaluates all candidate donor units using an **inner loss function** (pre-treatment Mean Squared Error, MSE)
        and selects the donor that optimally improves fit. The **outer optimization** builds the subset of donors that
        minimizes the total pre-treatment MSE.

        Inner and outer loss definitions:

        .. math::
            \mathbf{w} \in \mathcal{W}_{\text{conv}}(\widehat{U}) =
            \Big\{ \mathbf{w} \in \mathbb{R}_+^{|\widehat{U}|} : \sum_{j \in \widehat{U}} w_j = 1 \Big\}

        .. math::
            \ell_{\text{FSCM}}(\widehat{U}) =
            \min_{\mathbf{w} \in \mathcal{W}_{\text{conv}}(\widehat{U})}
            \Big\| \mathbf{y}_1 - \mathbf{Y}_{\widehat{U}} \mathbf{w} \Big\|_2^2

        .. math::
            \widehat{U}^\ast_{\text{FSCM}} =
            \operatorname*{argmin}_{\widehat{U} \subseteq \mathcal{N}_0}
            \ell_{\text{FSCM}}(\widehat{U})

        Attributes
        ----------
        config : FSCMConfig
            Configuration object for the estimator.
        df : pd.DataFrame
            Input panel data.
        outcome : str
            Name of the outcome variable column.
        treat : str
            Name of the treatment indicator column.
        unitid : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        display_graphs : bool
            Whether to display plots after fitting.
        save : Union[bool, str]
            Controls saving of plots; can be False, True, or a path prefix.
        counterfactual_color : str or List[str]
            Color(s) for counterfactual lines in plots.
        treated_color : str
            Color for the treated unit line in plots.
        selection_fraction : float
            Fraction of the donor pool to consider in forward selection.
        full_selection : bool
            Whether to run forward selection through all donors regardless of improvements.

        Methods
        -------
        fit()
            Fits the FSCM estimator using forward selection and constrained optimization,
            returning standardized results and prepped data.

        References
        ----------
        Cerulli, Giovanni. 2024.
            "Optimal initial donor selection for the synthetic control method."
            Economics Letters, 244: 111976. https://doi.org/10.1016/j.econlet.2024.111976
        Shi, Zhentao, and Jingyi Huang. 2023.
            "Forward-selected panel data approach for program evaluation."
            Journal of Econometrics 234 (2): 512–535. https://doi.org/10.1016/j.jeconom.2021.04.009
        Ben-Michael, Eli, Avi Feller, and Jesse Rothstein. 2021.
            "The Augmented Synthetic Control Method."
            Journal of the American Statistical Association 116 (536): 1789–1803. https://doi.org/10.1080/01621459.2021.1929245

        Examples
        --------
        >>> from mlsynth import FSCM
        >>> from mlsynth.config_models import FSCMConfig
        >>> import pandas as pd, numpy as np
        >>> data = pd.DataFrame({
        ...     'unit': np.repeat(np.arange(1,4), 10),
        ...     'time': np.tile(np.arange(1,11), 3),
        ...     'outcome': np.random.rand(30) + np.repeat(np.arange(0,3),10)*0.5,
        ...     'treated_unit_1': ((np.repeat(np.arange(1,4),10)==1) & (np.tile(np.arange(1,11),3)>=6)).astype(int)
        ... })
        >>> config = FSCMConfig(df=data, outcome='outcome', treat='treated_unit_1', unitid='unit', time='time', display_graphs=False)
        >>> estimator = FSCM(config=config)
        >>> results = estimator.fit()  # doctest: +SKIP
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
            selection_fraction : float, default=1.0
                Fraction of the donor pool to use in forward selection. Reducing this can
                substantially reduce runtime in high-dimensional settings.
            full_selection : bool, default=False
                Whether to force forward selection to run through all possible donor subsets,
                even after mBIC fails to improve. Can be computationally expensive.
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
        self.full_selection: bool = config.full_selection
        self.selection_fraction: float = config.selection_fraction

    def fit(self) -> BaseEstimatorResults:  # Main method to fit the FSCM estimator

        """
        Performs Forward Selected Synthetic Control estimation.

        Steps:
        1. Validate panel balance and prepare the data.
        2. Compute the inner loss (pre-treatment MSE) for each candidate donor.
        3. Iteratively select donors that minimize the inner loss (forward selection).
        4. For the selected donor subset, solve a constrained optimization problem
           to obtain the final weights and counterfactual outcomes.
        5. Optionally plot the treated vs counterfactual outcomes.

        Returns
        -------
        FSCMOutput
            A dataclass containing:
            - results: dict of sub-method results, including Forward SCM, Forward Augmented SCM, and Convex SCM.
            - prepped_data: dict of preprocessed inputs used in estimation.
        """



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

            master_results = _build_fscm_results(weight_dicts, prepared_data_dict, resultfscm,
                                                      prepared_data_dict["donor_names"])

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
