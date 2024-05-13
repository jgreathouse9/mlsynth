"""
mlsynth
==========

This module provides a convenient apporach to estimating various synthetic
control estimators

Classes:
-------
- FMA: Implements the Factor Model Approach as discussed in 

Li, K. T., & Sonnier, G. P. (2023). 
Statistical Inference for the Factor Model Approach to Estimate 
Causal Effects in Quasi-Experimental Settings. 
Journal of Marketing Research, 60(3), 449–472. 
https://doi.org/10.1177/00222437221137533.

- PCR: Implements Principal Component Regression as discussed in

Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). 
On robustness of principal component regression. 
J. Am. Stat. Assoc., 116(536), 1731-1745.
https://doi.org/10.1080/01621459.2021.1928513 
 
- TSSC: Two-Step Synthetic Control Metho

This implements the vanilla SCM, as well as 3 forms of SCM which
imposes different constraints on the objective function, namely,
whether to have summation or intercept constraints.

Li, K. T., & Shankar, V. (2023).
A two-step synthetic control approach for estimating causal effects of
marketing events. Management Science, in press(0), null.
https://doi.org/10.1287/mnsc.2023.4878

- AUGDID: Implements Augmented DID as discussed in
Li, K. T., & Bulte, C. V. d. (2022).
Augmented difference-in-differences.
Marketing Science, in press,
https://doi.org/10.1287/mksc.2022.1406

- fsPDA: Implements the Forward Selection Algorithm as discussed in
Shi, Z., & Huang, J. (2023).
Forward-selected panel data approach for program evaluation.
J. Econom., 234(2), 512-535.
https://doi.org/10.1016/j.jeconom.2021.04.009

- PCASC: Implements Robust Principal Component Analysis Synthetic Control
as discussed in "Essays on Machine Learning Mehtods in Economics"
by Mani Bayani.

- FDID: Implements Forward DID and AUGDID by

Li, K. T. (2024). Frontiers: A simple forward difference-in-differences method.
Marketing Science, 43(2), 267-279. https://doi.org/10.1287/mksc.2022.0212

Li, K. T., & Van den Bulte, C. (2023). Augmented difference-in-differences.
Marketing Science, 42(4), 746-767. https://doi.org/10.1287/mksc.2022.1406

Each Class below takes the following:

    Parameters:
    ----------
    - df: pandas.DataFrame
        Any dataframe the user specifies.

    - treat: str
        The column name in df representing the treatment variable.
        Must be 0 or 1, with only one treatment unit (for now).

    - time: str
        The column name in df representing the time variable.

    - outcome: str
        The column name in df representing the outcome variable.

    - unitid: str
        The column name in df representing the unit identifier.
        The string identifier, specifically.

    - figsize: tuple, optional
        The size of the figure (width, height) in inches.
        Default is (10, 8).

    - graph_style: str, optional
        The style of the graph.
        Default is 'default'.

    - grid: bool, optional
        Whether to display grid lines on the graph.
        Default is False.

    - counterfactual_color: str, optional
        The color of the counterfactual line on the graph.
        Default is 'blue'.

    - treated_color: str, optional
        The color of the treated line on the graph.
        Default is 'black'.

    - filetype: str, optional
        The file type for saving the graph. Default is None.
        Users may specify pdf, png, or any other file Python accepts.

    - display_graphs: bool, optional
        Whether to display the generated graphs.
        Default is True. Note, most of the above options
        are only relvant if you choose to display graphs.
  
    Returned Objects:
    ----------
    - Each class returns a results_df, comprised of the real value,
    the predicted counterfactual, the time values themselves as well
    as the difference which defines our treatment effects.
    
    - As of now, only MSC returns a weights dictionary. Including a
    dictionary for PCR which doesn't assign sparsity to weights would
    make a dictionary less meaningful.
    
    - All classes return a statistics_dict, which contatain (at least)
    the ATT (both absolute and percentage) as well as the T0 RMSE.
    In the future, confidence intervals and additional diagnostics
    will be included.
    
"""

# To Do list:
# 1: At Minimum, standardize plotting
# via using a utilities .py file.
# This includes the observed vs predicted plots
# and the gap plot.


# 2: Standardize the reshaping of the data.
# With a few exceptions, the way we reshape
# all these datasets is the exact same and
# pretty much does not change aross all the methods
# we have. This inludes the standardizing of notations aross
# methods. - Done!

# 3: Inlude static methods for ATT and other stats that are reported
# aross all methods (the fit Fict, ATTs Dict, etc) - Done!

# Wish list:

# 2: Add plaCebo tests where appliCable and relevant (e.g, in time)

# 3: Extend at least 1 estimator to staggered adoption

import warnings
warnings.filterwarnings("ignore", category=Warning, module="cvxpy")
import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
import warnings
from numpy.linalg import inv
#from sklearn.linear_model import LassoCV
from scipy.stats import norm
import scipy.stats as stats
from typing import Union
import scipy as sp
from scipy.optimize import lsq_linear
from sklearn.decomposition import PCA
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cvxpy as cp
from scipy.optimize import fmin_slsqp, minimize
from toolz import reduce, partial
import matplotlib.ticker as ticker
from mlsynth.utils.datautils import prepare_data, balance
from mlsynth.utils.resultutils import effects, plot_estimates
from mlsynth.utils.estutils import Opt, pcr, TSEST
from mlsynth.utils.inferutils import step2


class TSSC:
    def __init__(self, df, unitid, time, outcome, treat,
                 figsize=(12, 6),
                 graph_style="default",
                 grid=True,
                 counterfactual_color="red",
                 treated_color="black",
                 filetype="png",
                 display_graphs=True, draws=1000,
                 ):
        self.df = df
        self.draws = draws
        self.unitid = unitid
        self.time = time
        self.outcome = outcome
        self.treated = treat
        self.figsize = figsize
        self.graph_style = graph_style
        self.grid = grid
        self.counterfactual_color = counterfactual_color
        self.treated_color = treated_color

        self.filetype = filetype
        self.display_graphs = display_graphs

    def fit(self):

        nb =self.draws
        treated_unit_name, Ywide, y, donor_names, Xbar, t, t1, t2 = prepare_data(self.df, self.unitid, self.time,
                                                                                 self.outcome, self.treated)

        x = np.concatenate((np.ones((t, 1)), Xbar), axis=1)

        result = TSEST(Xbar, y, t1, nb, donor_names,t2)

        n = x.shape[1]

        b_MSC_c = next((method_dict["MSCc"]["WeightV"] for method_dict in result if "MSCc" in method_dict), None)
        b_SC = next((method_dict["SC"]["WeightV"] for method_dict in result if "SC" in method_dict), None)
        b_MSC_a = next((method_dict["MSCa"]["WeightV"] for method_dict in result if "MSCa" in method_dict), None)
        b_MSC_b = next((method_dict["MSCb"]["WeightV"] for method_dict in result if "MSCb" in method_dict), None)

        recommended_model = step2(np.array([[0] + list(np.ones(n - 1))]),
                                  np.array([[1] + list(np.zeros(n - 1))]),
                                  np.array([[0] + list(np.ones(n - 1)),
                                            [1] + list(np.zeros(n - 1))]),
                                  b_MSC_c, 1, 0, np.array([[1], [0]]), t1, x[:t1, :], y[:t1],
                                  nb, x.shape[1], np.zeros((n, nb)))

        recommended_variable = next(
            (method_dict[recommended_model]["Vectors"]["Counterfactual"] for method_dict in result if
             recommended_model in method_dict), None)
        ATT, RMSE = next(
            ((method_dict[recommended_model]["Effects"]["ATT"], method_dict[recommended_model]["Fit"]["T0 RMSE"]) for
             method_dict in result if recommended_model in method_dict), (None, None))

        plot_estimates(self.df, self.time, self.unitid, self.outcome, self.treated,
                       treated_unit_name, y, recommended_variable, method=recommended_model,
                       treatedcolor=self.treated_color, counterfactualcolor=self.counterfactual_color,
                       rmse=RMSE, att=ATT)

        return result






class FDID:
    def __init__(self, df, unitid, time, outcome, treat,
                 figsize=(12, 6),
                 graph_style="default",
                 grid=True,
                 counterfactual_color="red",
                 treated_color="black",
                 filetype="png",
                 display_graphs=True,
                 placebo=None
                 ):
        self.df = df
        self.unitid = unitid
        self.time = time
        self.outcome = outcome
        self.treated = treat
        self.figsize = figsize
        self.graph_style = graph_style
        self.grid = grid
        self.counterfactual_color = counterfactual_color
        self.treated_color = treated_color

        self.filetype = filetype
        self.display_graphs = display_graphs

        # Check for the "placebo" option
        if placebo is not None:
            self.validate_placebo_option(placebo)
            self.placebo_option = placebo

    def validate_placebo_option(self, placebo):
        # Check if the provided placebo option is a dictionary
        if not isinstance(placebo, dict):
            raise ValueError("The 'placebo' option must be a dictionary.")

        # Check for the first key in the dictionary
        first_key = next(iter(placebo), None)
        if first_key not in ["Time", "Space"]:
            raise ValueError(
                "The first key in the 'placebo' option must be either 'Time' or 'Space'.")

        # If the first key is "Time", check if the associated value is a list of positive integers
        if first_key == "Time":
            values = placebo[first_key]
            if not (isinstance(values, list) and all(isinstance(num, int) and num > 0 for num in values)):
                raise ValueError(
                    "If the first key in the 'placebo' option is 'Time', the associated value must be a list of positive integers.")

    def DID(self, y, datax, t1, itp=0):
        t = len(y)


        x1, x2 = np.mean(datax[:t1], axis=1).reshape(-1,
                                                     1), np.mean(datax[t1:t], axis=1).reshape(-1, 1)
        b_DID = np.mean(y[:t1] - x1, axis=0)  # DID intercept estimator

        y1_DID = b_DID + x1  # DID in-sample-fit
        y2_DID = b_DID + x2  # DID out-of-sample prediction
        y_DID = np.vstack((y1_DID, y2_DID))  # Stack y1_DID and y2_DID vertically

        y1_DID, y2_DID = y_DID[:t1], y_DID[t1:t]

       # if hasattr(self, 'placebo_option'):
           # t1 = self.realt1
        if itp > 0:
            t1+itp

        ATT_DID = np.mean(y[t1:t] - y_DID[t1:t])
        ATT_DID_percentage = 100 * ATT_DID / np.mean(y_DID[t1:t])

        # DID R-square

        R2_DID = 1 - (np.mean((y[:t1] - y_DID[:t1]) ** 2)) / (
            np.mean((y[:t1] - np.mean(y[:t1])) ** 2)
        )

        # Estimated DID residual

        u1_DID = y[:t1] - y_DID[:t1]

        # \hat \Sigma_{1,DID} and \hat \Sigma_{2,DID}
        t2 = t - t1

        Omega_1_hat_DID = (t2 / t1) * np.mean(u1_DID**2)
        Omega_2_hat_DID = np.mean(u1_DID**2)

        # \hat Sigma_{DID}

        std_Omega_hat_DID = np.sqrt(Omega_1_hat_DID + Omega_2_hat_DID)

        # Standardized ATT_DID

        ATT_std_DID = np.sqrt(t2) * ATT_DID / std_Omega_hat_DID

        # P-value for H0: ATT=0

        p_value_DID = 2 * (1 - norm.cdf(np.abs(ATT_std_DID)))

        # P-value for 1-sided test

        p_value_one_sided = 1 - norm.cdf(ATT_std_DID)

        # 95% Confidence Interval for DID ATT estimate

        z_critical = norm.ppf(0.975)  # 1.96 for a two-tailed test
        CI_95_DID_left = ATT_DID - z_critical * std_Omega_hat_DID / np.sqrt(t2)
        CI_95_DID_right = ATT_DID + z_critical * std_Omega_hat_DID / np.sqrt(t2)
        CI_95_DID_width = [
            CI_95_DID_left,
            CI_95_DID_right,
            CI_95_DID_right - CI_95_DID_left,
        ]
        if itp > 0:
            t1-itp

        # Metrics of fit subdictionary
        Fit_dict = {
            "T0 RMSE": round(np.std(y[:t1] - y_DID[:t1]), 3),
            "R-Squared": round(R2_DID, 3),
            "Pre-Periods": t1
        }

        # ATTS subdictionary
        ATTS = {
            "ATT": round(ATT_DID, 3),
            "Percent ATT": round(ATT_DID_percentage, 3),
            "SATT": round(ATT_std_DID, 3),
        }

        # Inference subdictionary
        Inference = {
            "P-Value": round(p_value_DID, 3),
            "95 LB": round(CI_95_DID_left, 3),
            "95 UB": round(CI_95_DID_right, 3),
            "Width": CI_95_DID_right - CI_95_DID_left,
            "Intercept":  np.round(b_DID, 3)
        }

        gap = y - y_DID

        second_column = np.arange(gap.shape[0]) - t1+1

        gap_matrix = np.column_stack((gap, second_column))

        # Vectors subdictionary
        Vectors = {
            "Observed Unit": np.round(y, 3),
            "Counterfactual": np.round(y_DID, 3),
            "Gap": np.round(gap_matrix, 3)
        }

        # Main dictionary
        DID_dict = {
            "Effects": ATTS,
            "Vectors": Vectors,
            "Fit": Fit_dict,
            "Inference": Inference
        }

        return DID_dict

    def AUGDID(self, datax, t, t1, t2, y, y1, y2):

        const = np.ones(t)      # t by 1 vector of ones (for intercept)
        # add an intercept to control unit data matrix, t by N (N=11)
        x = np.column_stack([const, datax])
        x1 = x[:t1, :]          # control units' pretreatment data matrix, t1 by N
        x2 = x[t1:, :]          # control units' pretreatment data matrix, t2 by N

        # ATT estimation by ADID method
        x10 = datax[:t1, :]
        x20 = datax[t1:, :]
        x1_ADID = np.column_stack([np.ones(x10.shape[0]), np.mean(x10, axis=1)])
        x2_ADID = np.column_stack([np.ones(x20.shape[0]), np.mean(x20, axis=1)])

        b_ADID = np.linalg.inv(x1_ADID.T @ x1_ADID) @ (x1_ADID.T @ y1)  # ADID estimate of delta

        y1_ADID = x1_ADID @ b_ADID  # t1 by 1 vector of ADID in-sample fit
        y2_ADID = x2_ADID @ b_ADID  # t2 by 1 vector of ADID prediction

        # t by 1 vector of ADID fit/prediction
        y_ADID = np.concatenate([y1_ADID, y2_ADID]).reshape(-1, 1)

        ATT = np.mean(y2 - y2_ADID)  # ATT by ADID
        ATT_per = 100 * ATT / np.mean(y2_ADID)  # ATT in percentage by ADID

        e1_ADID = (
            y1 - y1_ADID
        )  # t1 by 1 vector of treatment unit's (pre-treatment) residuals
        sigma2_ADID = np.mean(e1_ADID**2)  # \hat sigma^2_e

        eta_ADID = np.mean(x2, axis=0).reshape(-1, 1)
        psi_ADID = x1.T @ x1 / t1

        Omega_1_ADID = (sigma2_ADID * eta_ADID.T) @ np.linalg.pinv(psi_ADID) @ eta_ADID
        Omega_2_ADID = sigma2_ADID

        Omega_ADID = (t2 / t1) * Omega_1_ADID + Omega_2_ADID  # Variance

        ATT_std = np.sqrt(t2) * ATT / np.sqrt(Omega_ADID)

        quantile = norm.ppf(0.975)

        CI_95_DID_left = ATT - quantile * np.sqrt(sigma2_ADID) / np.sqrt(t2)
        CI_95_DID_right = ATT + quantile * np.sqrt(sigma2_ADID) / np.sqrt(t2)

        RMSE = np.sqrt(np.mean((y1 - y1_ADID) ** 2))
        RMSEPost = np.sqrt(np.mean((y2 - y2_ADID) ** 2))

        R2_ADID = 1 - (np.mean((y1 - y1_ADID) ** 2)) / np.mean((y1 - np.mean(y1)) ** 2)

        # P-value for H0: ATT=0

        p_value_aDID = 2 * (1 - norm.cdf(np.abs(ATT_std)))

        CI_95_DID_width = [
            CI_95_DID_left,
            CI_95_DID_right,
            CI_95_DID_right - CI_95_DID_left,
        ]

        # Metrics of fit subdictionary
        Fit_dict = {
            "T0 RMSE": round(np.std(y[:t1] - y_ADID[:t1]), 3),
            "R-Squared": round(R2_ADID, 3),
            "T0": len(y[:t1])
        }

        # ATTS subdictionary
        ATTS = {
            "ATT": round(ATT, 3),
            "Percent ATT": round(ATT_per, 3),
            "SATT": round(ATT_std.item(), 3),
        }

        # Inference subdictionary
        Inference = {
            "P-Value": round(p_value_aDID.item(), 3),
            "95 LB": round(CI_95_DID_left.item(), 3),
            "95 UB": round(CI_95_DID_right.item(), 3),
            "Width": CI_95_DID_right - CI_95_DID_left
        }
        gap = y - y_ADID

        second_column = np.arange(gap.shape[0]) - t1+1

        gap_matrix = np.column_stack((gap, second_column))

        # Vectors subdictionary
        Vectors = {
            "Observed Unit": np.round(y, 3),
            "Counterfactual": np.round(y_ADID, 3),
            "Gap": np.round(gap_matrix, 3)
        }

        # Main dictionary
        ADID_dict = {
            "Effects": ATTS,
            "Vectors": Vectors,
            "Fit": Fit_dict,
            "Inference": Inference
        }

        return ADID_dict, y_ADID

    def est(self, control, t, t1, t2, y, y1, y2, datax):

        FDID_dict = self.DID(y.reshape(-1, 1), control, t1)

        y_FDID = FDID_dict['Vectors']['Counterfactual']

        DID_dict = self.DID(y.reshape(-1, 1), datax, t1)

        AUGDID_dict, y_ADID = self.AUGDID(datax, t, t1, t2, y, y1, y2)
        time_points = np.arange(1, len(y) + 1)

        # Calculate the ratio of widths for DID and AUGDID compared to FDID
        ratio_DID = DID_dict["Inference"]["Width"] / FDID_dict["Inference"]["Width"]
        ratio_AUGDID = AUGDID_dict["Inference"]["Width"] / FDID_dict["Inference"]["Width"]

        # Add the new elements to the Inference dictionaries
        DID_dict["Inference"]["WidthRFDID"] = ratio_DID
        AUGDID_dict["Inference"]["WidthRFDID"] = ratio_AUGDID

        return FDID_dict, DID_dict, AUGDID_dict, y_FDID

    def selector(self, no_control, t1, t, y, y1, y2, datax, control_ID, df):


        R2 = np.zeros(no_control)
        R2final = np.zeros(no_control)
        control_ID_adjusted = np.array(control_ID) - 1
        select_c = np.zeros(no_control, dtype=int)

        for j in range(no_control):
            ResultDict = self.DID(y.reshape(-1, 1), datax[:t, j].reshape(-1, 1), t1)
            R2[j] = ResultDict["Fit"]["R-Squared"]
        R2final[0] = np.max(R2)
        first_c = np.argmax(R2)
        select_c[0] = control_ID_adjusted[first_c]

        for k in range(2, no_control + 1):
            left = np.setdiff1d(control_ID_adjusted, select_c[: k - 1])
            control_left = datax[:, left]
            R2 = np.zeros(len(left))

            for jj in range(len(left)):
                combined_control = np.concatenate(
                    (
                        datax[:t1, np.concatenate((select_c[: k - 1], [left[jj]]))],
                        datax[t1:t, np.concatenate((select_c[: k - 1], [left[jj]]))]
                    ),
                    axis=0
                )
                ResultDict = self.DID(y.reshape(-1, 1), combined_control, t1)
                R2[jj] = ResultDict["Fit"]["R-Squared"]

            R2final[k - 1] = np.max(R2)
            select = left[np.argmax(R2)]
            select_c[k - 1] = select

        return select_c, R2final

    def fit(self):
        Ywide = self.df.pivot_table(
            values=self.outcome, index=self.time, columns=self.unitid, sort=False
        )

        treated_unit_name = self.df.loc[self.df[self.treated] == 1, self.unitid].values[0]
        Ywide = Ywide[[treated_unit_name] +
                      [col for col in Ywide.columns if col != treated_unit_name]]
        y = Ywide[treated_unit_name].values.reshape(-1, 1)
        donor_df = self.df[self.df[self.unitid] != treated_unit_name]
        donor_names = donor_df[self.unitid].unique()
        datax = Ywide[donor_names].values
        no_control = datax.shape[1]
        control_ID = np.arange(1, no_control + 1)  # 1 row vector from 1 to no_control

        t = np.shape(y)[0]
        assert t > 5, "You have less than 5 total periods."

        results = []  # List to store results

        self.realt1 = len(
            self.df[
                (self.df[self.unitid] == treated_unit_name)
                & (self.df[self.treated] == 0)
            ]
        )

        if hasattr(self, 'placebo_option'):
            print("Placebo Option in fit method:", self.placebo_option)
            placebo_list = self.placebo_option.get("Time")
            placebo_list = [0] + self.placebo_option.get("Time")
            for i, itp in enumerate(placebo_list):
                t1 = len(
                    self.df[
                        (self.df[self.unitid] == treated_unit_name)
                        & (self.df[self.treated] == 0)
                    ]
                ) - itp

                t2 = t - t1

                y1 = np.ravel(y[:t1])
                y2 = np.ravel(y[-t2:])

                control_order, R2_final = self.selector(
                    no_control, t1, t, y, y1, y2, datax, control_ID, Ywide
                )
                selected_control_indices = control_order[:R2_final.argmax() + 1]

                copy_wide_copy = Ywide.iloc[:, 1:].copy()
                selected_controls = [copy_wide_copy.columns[i] for i in selected_control_indices]

                control = datax[:, control_order[: R2_final.argmax() + 1]]

                FDID_dict, DID_dict, AUGDID_dict, y_FDID = self.est(
                    control, t, t1, t - t1, y, y1, y2, datax
                )

                placebo_results = []  # Initialize an empty list for each placebo iteration
                FDID_dict['Selected Units'] = selected_controls
                placebo_results.append({"FDID": FDID_dict})

                # Append the normal DID dictionary to the list
                placebo_results.append({"DID": DID_dict})
                placebo_results.append({"AUGDID": AUGDID_dict})

                def round_dict_values(input_dict, decimal_places=3):
                    rounded_dict = {}
                    for key, value in input_dict.items():
                        if isinstance(value, dict):
                            # Recursively round nested dictionaries
                            rounded_dict[key] = round_dict_values(value, decimal_places)
                        elif isinstance(value, (int, float, np.float64)):
                            # Round numeric values
                            rounded_dict[key] = round(value, decimal_places)
                        else:
                            rounded_dict[key] = value
                    return rounded_dict

                # Round all values in the placebo_results list of dictionaries
                placebo_results = [
                    round_dict_values(result) for result in placebo_results
                ]

                # Add the placebo results to the overall results list with a labeled key
                results.append({"Placebo" + str(i): placebo_results})

        else:

            t1 = len(
                self.df[
                    (self.df[self.unitid] == treated_unit_name)
                    & (self.df[self.treated] == 0)
                ]
            )

            t2 = t - t1
            y1 = np.ravel(y[:t1])
            y2 = np.ravel(y[-t2:])

            control_order, R2_final = self.selector(
                no_control, t1, t, y, y1, y2, datax, control_ID, Ywide
            )
            selected_control_indices = control_order[:R2_final.argmax() + 1]
            copy_wide_copy = Ywide.iloc[:, 1:].copy()
            selected_controls = [copy_wide_copy.columns[i] for i in selected_control_indices]

            control = datax[:, control_order[: R2_final.argmax() + 1]]

            FDID_dict, DID_dict, AUGDID_dict, y_FDID = self.est(
                control, t, t1, t - t1, y, y1, y2, datax
            )

            estimators_results = []

            # Calculate the weight
            weight = 1 / len(selected_controls)

            # Create the dictionary
            unit_weights = {unit: weight for unit in selected_controls}

            FDID_dict['Weights'] = unit_weights


            estimators_results.append({"FDID": FDID_dict})

            # Append the normal DID dictionary to the list
            estimators_results.append({"DID": DID_dict})
            estimators_results.append({"AUGDID": AUGDID_dict})

            def round_dict_values(input_dict, decimal_places=3):
                rounded_dict = {}
                for key, value in input_dict.items():
                    if isinstance(value, dict):
                        # Recursively round nested dictionaries
                        rounded_dict[key] = round_dict_values(value, decimal_places)
                    elif isinstance(value, (int, float, np.float64)):
                        # Round numeric values
                        rounded_dict[key] = round(value, decimal_places)
                    else:
                        rounded_dict[key] = value
                return rounded_dict

            # Round all values in the estimators_results list of dictionaries
            estimators_results = [
                round_dict_values(result) for result in estimators_results
            ]

            # Add the estimators results to the overall results list
            # results.append({"Estimators": estimators_results})

            if self.display_graphs:
                time_axis = self.df[self.df[self.unitid] == treated_unit_name][self.time].values
                intervention_point = self.df.loc[self.df[self.treated] == 1, self.time].min()
                n = np.arange(1, t+1)
                fig = plt.figure(figsize=self.figsize)

                plt.plot(
                    n,
                    y,
                    color=self.treated_color,
                    label="Observed {}".format(treated_unit_name),
                    linewidth=2
                )
                plt.plot(
                    n[: t1],
                    y_FDID[: t1],
                    color=self.counterfactual_color,
                    linewidth=2,
                    label="FDID {}".format(treated_unit_name),
                )
                plt.plot(
                    n[t1-1:],
                    y_FDID[t1-1:],
                    color=self.counterfactual_color,
                    linestyle="--",
                    linewidth=2,
                )
                plt.axvline(
                    x=t1+1,
                    color="grey",
                    linestyle="--", linewidth=1.5,
                    label=self.treated + ", " + str(intervention_point),
                )
                upb = max(max(y), max(y_FDID))
                lpb = min(0.5 * min(min(y), min(y_FDID)), 1 * min(min(y), min(y_FDID)))

                error_value = FDID_dict["Fit"]["T0 RMSE"]
                ATT = FDID_dict["Effects"]["ATT"]

                # Set y-axis limits
                # plt.ylim(lpb, upb)
                plt.xlabel(self.time)
                plt.ylabel(self.outcome)
                plt.title(fr'Forward DID, $\bar{{\tau}}$ = {ATT:.3f}, RMSE = {error_value:.3f}')
                plt.grid(self.grid)
                plt.legend()
                plt.show()

        if hasattr(self, 'placebo_option'):
            return results
        else:
            return estimators_results

