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

# Edit this dicstring in the future!!

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




class PCR:
    """
    Implements Principal Component Regression.

    Here, we estimate the low rank structure of the donor matrix using singular
    value thresholding, and then learning the pre-intervention period with
    OLS regression. Then, we predict the post-intervention counterfacutal
    using the learnt weights.
    """

    def __init__(
        self,
        df,
        treat,
        time,
        outcome,
        unitid,
        figsize=(12, 6),
        graph_style="default",
        grid=True,
        counterfactual_color="red",
        treated_color="black",
        filetype="png",
        display_graphs=True,
        objective="OLS", placebo=None, save=False
    ):
        self.df = df
        self.objective = objective
        self.save = save
        self.treat = treat
        self.time = time
        self.outcome = outcome
        self.unitid = unitid
        self.figsize = figsize
        self.graph_style = graph_style
        self.grid = grid
        self.counterfactual_color = counterfactual_color
        self.treated_color = treated_color

        self.filetype = filetype
        self.display_graphs = display_graphs

        self.placebo = placebo

    def fit(self):

        def get_PlaceboGaps(Y0, t, t1, t2, pcr, objective, donor_names, effects):
            PlaceboGaps = np.empty((t, 0))  # Initialize an empty matrix to store Gap columns

            for i in range(Y0.shape[1]):
                ypla = Y0[:, i].copy()  # Copy the ith column of Y0
                Y0pla = np.delete(Y0, i, axis=1)  # Remove the ith column from Y0
                weights2 = pcr(Y0pla[:t1], ypla, objective, donor_names)
                attdictplacebo, fitdictplacebo, Vectorsplacebo = effects.calculate(ypla, np.dot(Y0pla, weights2),
                                                                                   t1, t2)
                PlaceboGaps = np.concatenate((PlaceboGaps, Vectorsplacebo["Gap"][:, 0][:, np.newaxis]),
                                             axis=1)  # Concatenate the first column of Gap to PlaceboGaps
                Y0 = np.insert(Y0pla, i, ypla, axis=1)  # Insert the modified column back into Y0

            return PlaceboGaps

        # Extracts our data characteristics
        treated_unit_name, Ywide, y, donor_names, Y0, t, t1, t2 = prepare_data(self.df, self.unitid, self.time,
                                                                                 self.outcome, self.treat)

        weights = pcr(Y0[:t1], y, self.objective, donor_names)
        attdict, fitdict, Vectors = effects.calculate(y, np.dot(Y0, weights), t1, t2)

        if self.placebo=="Unit":
            gapreal = Vectors["Gap"][:, 0][:, np.newaxis]
            Y0pla = get_PlaceboGaps(Y0, t, t1, t2, pcr, self.objective, donor_names, effects)
            # Plotting
            plt.figure(figsize=(10, 6))
            for i in range(Y0pla.shape[1]):
                plt.plot(range(t), Y0pla[:, i], color="grey", alpha=.45, linewidth=1)

            plt.plot(range(t), gapreal, color="black", linewidth=2.5)
            plt.axvline(x=t1, color='r', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('Treatment Effect')
            plt.title('In-Space Placebo Tests')
            plt.grid(True)
            if self.save == True:
                plt.savefig("Placebo" + "PCR" + treated_unit_name + ".png")
            else:
                plt.show()

            gapmat = np.concatenate((gapreal, Y0pla), axis=1)

            PreRMSEs = np.sqrt(np.mean(gapmat[:t1] ** 2, axis=0))

            PostRMSEs = np.sqrt(np.mean(gapmat[t1:] ** 2, axis=0))

            RMSEs = np.vstack((PreRMSEs, PostRMSEs,PostRMSEs/PreRMSEs)).T

            info = {"Gaps": gapmat, "RMSEs": RMSEs}

            return info
        else:
            plot_estimates(self.df, self.time, self.unitid, self.outcome, self.treat,
                           treated_unit_name, y, np.dot(Y0, weights), method="PCR",
                           treatedcolor=self.treated_color, counterfactualcolor=self.counterfactual_color,
                           rmse=fitdict["T0 RMSE"], att=attdict["ATT"], save=self.save)

            weights_dict = {}
            for unit, weight in zip(Ywide.columns[Ywide.columns != treated_unit_name], weights):
                weights_dict[unit] = round(weight, 4)

            PCR_dict = {
                "Effects": attdict,
                "Vectors": Vectors,
                "Fit": fitdict,
                "Weights": weights_dict
            }
            return PCR_dict






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




class PCASC:
    def __init__(
        self,
        df,
        treat,
        time,
        outcome,
        unitid,
        method="RPCA",  # Add an argument for specifying the method
        figsize=(12, 6),
        graph_style="default",
        grid=True,
        counterfactual_color="red",
        treated_color="black",
        filetype=".png",
        display_graphs=True,
        diagnostics=False,
        save=False, vallamb=1
    ):
        self.df = df
        self.outcome = outcome
        self.treat = treat
        self.unitid = unitid
        self.time = time
        self.weighted_units_dict = None
        self.statistics_dict = None
        self.result_df = None
        self.counterfactual_color = counterfactual_color
        self.treated_color = treated_color
        self.graph_style = graph_style
        self.grid = grid
        self.figsize = figsize
        self.filetype = filetype
        self.display_graphs = display_graphs
        self.diagnostics = diagnostics
        self.save = save
        self.method = method  # Store the method choice
        self.vallamb = vallamb

    def default_mu(self, data_mat, sigma=10):
        return 1.0 / ((2 * np.max(data_mat.shape) ** 0.5 * sigma))
        # return ((2 * np.max(data_mat.shape) ** .5 * sigma))

    def term_criteria(self, L, S, L_prev, S_prev, tol=1e-7):
        diff = (
                np.linalg.norm(L - L_prev, ord="fro") ** 2 + np.linalg.norm(S - S_prev, ord="fro") ** 2
        )
        if diff < tol:
            return True, diff
        else:
            return False, diff

    def shrinkage(self, mat: np.ndarray, thresh: Union[np.ndarray, float]) -> np.ndarray:
        return np.sign(mat) * np.maximum(np.abs(mat) - thresh, np.zeros(mat.shape))

    def sv_thresholding(self, mat: np.ndarray, thresh: float) -> np.ndarray:
        U, s, V = sp.linalg.svd(mat, full_matrices=False)
        s = self.shrinkage(s, thresh)
        return U @ np.diag(s) @ V

    def decompose(self, data_mat, mu, max_iter=1e5, tol=1e-7, verbose=False):
        n, m = data_mat.shape
        lamda = 1.0 / (max(n, m)) ** 0.5
        mu_inv = mu ** (-1)
        S = np.zeros(data_mat.shape)
        L = np.zeros(data_mat.shape)

        L_prev = L
        S_prev = S

        it = 0
        while (
                not self.term_criteria(L, S, L_prev, S_prev, tol=tol)[0] and it < max_iter
        ) or it == 0:
            L_prev = L
            S_prev = S

            L = self.sv_thresholding(data_mat - S, mu_inv)
            S = self.shrinkage(data_mat - L, lamda * mu_inv)
            it += 1

        if verbose:
            print(
                f"Iteration: {it}, diff: {term_criteria(L, S, L_prev, S_prev, tol=tol)[1]}, terminating alg."
            )

        return L, S

    def HQF(self, M_noise, maxiter=1000, ip=2, lam_1=0.001):

        rak = universal_rank(M_noise)

        # https://doi.org/10.1016/j.sigpro.2022.108816
        m, n = M_noise.shape
        U = np.random.rand(m, rak)
        Ip = 2
        inn = 0
        RMSE = []
        RRMSE = []
        peaksnr = []
        lam_2 = lam_1

        for iter in range(Ip):
            PINV_U = np.linalg.pinv(U)
            V = np.dot(PINV_U, M_noise)
            PIMV_V = np.linalg.pinv(V)
            U = np.dot(M_noise, PIMV_V)

        X = np.dot(U, V)
        T = M_noise - X
        t_m_n = T.ravel()
        scale = 10 * 1.4815 * np.median(np.abs(t_m_n - np.median(t_m_n)))
        sigma = np.full((m, n), scale)  # Use np.full to create a 2D array
        ONE_1 = np.ones((m, n))
        ONE_1[np.abs(T - np.median(t_m_n)) - sigma < 0] = 0
        N = T * ONE_1
        U_p = U
        V_p = V

        for iter in range(maxiter):
            D = M_noise - N
            U = np.dot((D.dot(V.T) - lam_1 * U_p), np.linalg.inv(V.dot(V.T) - lam_1))
            V = np.linalg.inv(U.T.dot(U) - lam_2).dot(U.T.dot(D) - lam_2 * V_p)
            U_p = U
            V_p = V
            X = U.dot(V)
            T = M_noise - X
            t_m_n = T.ravel()
            scale = min([scale, ip * 1.4815 * np.median(np.abs(t_m_n - np.median(t_m_n)))])
            sigma = np.ones((m, n)) * scale
            ONE_1 = np.ones((m, n))
            ONE_1[np.abs(T - np.median(t_m_n)) - sigma < 0] = 0
            N = T * ONE_1
            RMSE.append(np.linalg.norm(M_noise - X, 'fro') / np.sqrt(m * n))

            if iter != 0:
                step_MSE = RMSE[iter - 1] - RMSE[iter]
                if step_MSE < 0.000001:
                    inn += 1
                if inn > 1:
                    break

        Out_X = X
        return Out_X, N

        # Count the number of singular values above the threshold
        num_components = np.sum(S > threshold)

        return num_components

    def fit(self):
        """
        Main workhorse for estimating the effects- likely can be
        improved by making functions for reshaping the data.
        """

        self.df = self.df.copy()
        timevar = self.time
        unit = self.unitid
        balance(self.df, unit, timevar)

        treated_unit_name = self.df.loc[self.df[self.treat] == 1, self.unitid].iloc[0]

        t1 = len(
            self.df[
                (self.df[self.treat] == 0) & (self.df[self.unitid] == treated_unit_name)
            ]
        )
        if t1 < 3:
            raise ValueError("You need at least 3 pre-periods")
        self.df.reset_index(drop=True, inplace=True)

        trainframe = self.df.pivot_table(
            index=self.unitid, columns=self.time, values=self.outcome, sort=False
        )

        X = trainframe.iloc[:, 0:t1]

        optimal_clusters, cluster_x, numvals = fpca(X)

        kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, init='k-means++', algorithm='elkan')
        cluster_labels = kmeans.fit_predict(cluster_x)

        trainframe["cluster"] = cluster_labels
        treat_cluster = trainframe.loc[
            trainframe.index == treated_unit_name, "cluster"
        ].iloc[0]

        self.clustered_units = trainframe[trainframe["cluster"] == treat_cluster].copy()
        # self.clustered_units.reset_index(drop=True, inplace=True)

        self.clustered_units.drop("cluster", axis=1, inplace=True)

        treated_row = self.clustered_units.index.get_loc(treated_unit_name)
        # Makes our wide data into a matrix
        Y = self.clustered_units.iloc[:, 0:].to_numpy()

        y = Y[treated_row]
        # Our observed values
        Y0 = np.delete(Y, (treated_row), axis=0)

        Nco =np.shape(Y0)[0]

        L = RPCA(Y0, self.vallamb)


        if self.diagnostics:
            timeind = [col for col in self.clustered_units.columns]

            reference_time = timeind[t1]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.grid(True)
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.3)
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax1.plot(timeind, L.transpose(), color='#1F4788', alpha=.8)
            ax1.set_xlabel(self.time)
            ax1.set_ylabel(self.outcome)
            ax1.set_title('Low-Rank Component')
            ax1.axvline(x=reference_time, color="#2E211B", linestyle="-",
                        label=str(reference_time), linewidth=1.5)
            ax2.plot(timeind, S.transpose(), color='#1F4788', alpha=.4)
            ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax2.set_xlabel(self.time)
            ax2.axvline(x=reference_time, color="#2E211B", linestyle="-",
                        label=str(reference_time), linewidth=1.5)
            ax2.set_title('Noise Component')
            ax2.set_ylabel('Difference from Target Unit')
            ax2.grid(True)
            plt.show()

        beta_value = Opt.SCopt(Nco, y[:t1], t1, L[:, 0:t1].T, model="CONVEX")
        y_RPCA = np.dot(L.T, beta_value)

        t2 = len(y)-t1

        beta_value = np.round(beta_value, 3)

        # Extract unit names excluding treated unit name
        unit_names = [index for index in self.clustered_units.index if index != treated_unit_name]

        # Create a dictionary of non-zero weights
        weights_dict = {key: value for key, value in zip(unit_names, beta_value) if value > 0}

        attdict, fitdict, Vectors = effects.calculate(y, y_RPCA, t1, t2)

        # List of regionnames you want to include in the subframe
        selected_regions = self.clustered_units.index

        # Create the subframe using the isin() method
        clustered_units_long = self.df[self.df[self.unitid].isin(self.clustered_units.index)]

        stats = { "Number of Clusters": len(set(cluster_labels)),
            "Donors in Cluster": len(self.clustered_units) - 1,
                       "ClusterFrame": clustered_units_long,
                       "Treated Unit": treated_unit_name,
                       "Treatment Name": self.treat,
                       "Outcome": self.outcome}

        # Main dictionary
        RPCA_dict = {
            "Effects": attdict,
            "Vectors": Vectors,
            "Fit": fitdict,
            "stats": stats,
            "Weights": weights_dict
        }
        plot_estimates(self.df, self.time, self.unitid, self.outcome, self.treat,
                       treated_unit_name, y, y_RPCA, method="RPCA-SYNTH",
                       treatedcolor=self.treated_color, counterfactualcolor=self.counterfactual_color, rmse=fitdict["T0 RMSE"], att=attdict["ATT"], save=self.save)

        return RPCA_dict

