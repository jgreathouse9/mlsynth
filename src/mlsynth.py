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
# methods.

# 3: Inlude static methods for ATT and other stats that are reported
# aross all methods (the fit Fict, ATTs Dict, etc).

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



class TSSC:
    def __init__(self, df, unitid, time, outcome, treat,
                 figsize=(12, 6),
                 graph_style="default",
                 grid=True,
                 counterfactual_color="red",
                 treated_color="black",
                 filetype="png",
                 display_graphs=True
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

    def fit(self):

        treated_unit_name = self.df.loc[self.df[self.treated] == 1, self.unitid].values[0]

        # Pivot the DataFrame
        Ywide = self.df.pivot(index=self.time, columns=self.unitid, values=self.outcome)



        y = Ywide[treated_unit_name].values
        donor_df = self.df[self.df[self.unitid] != treated_unit_name]
        donor_names = donor_df[self.unitid].unique()
        Xbar = Ywide[donor_names].values

        nb =10000
        t = y.shape[0]
        t1 = len(
                    self.df[
                        (self.df[self.unitid] == treated_unit_name)
                        & (self.df[self.treated] == 0)
                    ]
                )
        t2 = t - t1

        y1 = y[:t1]
        y2 = y[t1 + 1:t]
        control = Xbar

        const = np.ones((t, 1))
        x = np.concatenate((const, Xbar), axis=1)
        x1 = x[:t1, :]
        x2 = x[t1 + 1:t, :]


        n = x.shape[1]
        bm_MSC_c = np.zeros((n, nb))

        Rt = np.array([[0] + list(np.ones(n - 1)), [1] + list(np.zeros(n - 1))])

        qt = np.array([[1], [0]])
        R1t = np.array([[0] + list(np.ones(n - 1))])

        R2t = np.array([[1] + list(np.zeros(n - 1))])

        q1t = 1
        q2t = 0

        def solve_optimization(x, y, t1, donornames):

            # Define a list to store dictionaries for each method
            fit_dicts_list = []
            fit_results = {}
            # List of methods to loop over
            methods = ["SC", "MSC_b",  "MSC_a", "MSC_c"]

            for method in methods:
                # Determine if intercept should be included
                if method == "MSC_a":
                    x_ = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
                    is_ones_vector = np.all(x_[:, 0] == 1)

                    # Assert to check if it's true
                    assert is_ones_vector, "The first column is not a vector of 1s."
                    donornames = ["Intercept"] + list(donornames)
                elif method in ["MSC_c"]:
                    x_ = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
                    is_ones_vector = np.all(x_[:, 0] == 1)

                    # Assert to check if it's true
                    assert is_ones_vector, "The first column is not a vector of 1s."
                elif method not in ["SC", "MSC_b"]:
                    x_ = x  # No intercept added
                else:
                    x_ = x
                t2 = len(y) - t1
                # Define optimization variables
                b_cvx = cp.Variable(x_.shape[1])

                # Define objective function
                objective = cp.Minimize(cp.norm(x_[:t1] @ b_cvx - y[:t1], 2))

                # Define constraints based on method
                constraints = []
                if method == "SC":
                    constraints.append(cp.sum(b_cvx) == 1)  # Sum constraint for all coefficients
                    constraints.append(b_cvx >= 0)  # Non-negativity constraint for all coefficients
                elif method == "MSC_a":
                    constraints.append(cp.sum(b_cvx[1:]) == 1)  # Exclude the intercept from the sum constraint
                    constraints.append(b_cvx[1:] >= 0)  # Non-negativity constraint for coefficients excluding intercept
                    constraints.append(b_cvx[0] == 0)  # No constraint on the intercept
                elif method == "MSC_b":
                    constraints.append(b_cvx >= 0)  # Non-negativity constraint for all coefficients
                elif method == "MSC_c":
                    constraints.append(b_cvx[1:] >= 0)  # Non-negativity constraint for coefficients excluding intercept
                    constraints.append(b_cvx[0] == 0)  # No constraint on the intercept

                # Solve the problem
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.CLARABEL)

                # Extract the solution
                b = b_cvx.value

                weights_dict = {donor: weight for donor, weight in zip(donornames, np.round(b,4))  if weight > 0.001}

                # Calculate the counterfactual outcome
                y_counterfactual = x_.dot(b)

                # Calculate the ATT
                att = np.mean(y[t1:] - y_counterfactual[t1:])
                att_percentage = 100 * att / np.mean(y_counterfactual[t1:])

                # Calculate omega hat values
                u1_rpc = y[:t1] - y_counterfactual[:t1]
                omega_1_hat_rpc = (t2 / t1) * np.mean(u1_rpc ** 2)
                omega_2_hat_predicted = np.mean(u1_rpc ** 2)
                std_omega_hat_predicted = np.sqrt(omega_1_hat_rpc + omega_2_hat_predicted)

                # Calculate the SATT
                att_std_predicted = np.sqrt(t2) * att / std_omega_hat_predicted

                m = t1

                zz1 = np.concatenate((x_[:t1], y[:t1].reshape(-1,1)), axis=1)
                # Confidence Intervals
                sigma_MSC = np.sqrt(np.mean(( y[:t1] -  y_counterfactual[:t1]) ** 2))

                sigma2_v = np.mean((y[t1:]  - y_counterfactual[t1:] - np.mean(y[t1:] - y_counterfactual[t1:])) ** 2)  # \hat \Sigma_v

                e1_star = np.sqrt(sigma2_v) * np.random.randn(t2, nb)  # e_{1t}^* iid N(0, \Sigma^2_v)
                A_star = np.zeros(nb)

                x2 = x_[t1 + 1: t, :]

                np.random.seed(1476)

                for g in range(nb):

                    np.random.shuffle(zz1)  # the random generator does not repeat the same number

                    zm = zz1[:m, :]  # randomly select m rows from z1_{T_1 by (N+1)}
                    xm = zm[:, :-1]  # it uses the last m observation
                    ym = np.dot(xm, b) + sigma_MSC * np.random.randn(m)

                    # Define optimization variables
                    b_cvx2 = cp.Variable(xm.shape[1])

                    # Define objective function
                    objective = cp.Minimize(cp.norm(xm @ b_cvx2 - ym, 2))

                    constraints2 = []
                    if method == "SC":
                        constraints2.append(cp.sum(b_cvx2) == 1)  # Sum constraint for all coefficients
                        constraints2.append(b_cvx2 >= 0)  # Non-negativity constraint for all coefficients
                    elif method == "MSC_a":
                        constraints2.append(cp.sum(b_cvx2[1:]) == 1)  # Exclude the intercept from the sum constraint
                        constraints2.append(
                            b_cvx2[1:] >= 0)  # Non-negativity constraint for coefficients excluding intercept
                        constraints2.append(b_cvx2[0] == 0)  # No constraint on the intercept
                    elif method == "MSC_b":
                        constraints2.append(b_cvx2 >= 0)  # Non-negativity constraint for all coefficients
                    elif method == "MSC_c":
                        constraints2.append(
                            b_cvx2[1:] >= 0)  # Non-negativity constraint for coefficients excluding intercept
                        constraints2.append(b_cvx2[0] == 0)  # No constraint on the intercept

                    # Define the problem
                    problem = cp.Problem(objective, constraints2)

                    # Solve the problem
                    problem.solve(solver=cp.CLARABEL)

                    # Extract the solution
                    bm = b_cvx2.value

                    A1_star = -np.mean(np.dot(x2, (bm - b))) * np.sqrt((t2 * m) / t1)  # A_1^*
                    A2_star = np.sqrt(t2) * np.mean(e1_star[:, g])
                    A_star[g] = A1_star + A2_star  # A^* = A_1^* + A_2^*

                # end of the subsampling-bootstrap loop
                ATT_order = np.sort(A_star / np.sqrt(t2))  # sort subsampling ATT by ascending order

                c005 = ATT_order[int(0.005 * nb)]  # compute critical values
                c025 = ATT_order[int(0.025 * nb)]
                c05 = ATT_order[int(0.05 * nb)]
                c10 = ATT_order[int(0.10 * nb)]
                c25 = ATT_order[int(0.25 * nb)]
                c75 = ATT_order[int(0.75 * nb)]
                c90 = ATT_order[int(0.90 * nb)]
                c95 = ATT_order[int(0.95 * nb)]
                c975 = ATT_order[int(0.975 * nb)]
                c995 = ATT_order[int(0.995 * nb)]
                cr_min = att - ATT_order[nb - 1]  # ATT_order[nb] is the maximum A^* value
                cr_max = att - ATT_order[0]  # ATT_order[1] is the maximum A^* value

                # 95% confidence interval of ATT is [cr_025, cr_975]
                # 90% confidence interval of ATT is [cr_05, cr_95], etc.
                cr_005_995 = [att - c995, att - c005]

                cr_025_0975 = [att - c975, att - c025]

                # Create effects dictionary
                effects_dict = {
                    "ATT": round(att, 3),
                    "Percent ATT": round(att_percentage, 3),
                    "SATT": round(att_std_predicted, 3),
                    "95% CI": [round(val, 3) for val in cr_025_0975]
                }

                gap = y - y_counterfactual

                # Create a second column representing time periods
                second_column = np.arange(gap.shape[0]) - t1 + 1

                # Stack the gap and second column horizontally to create the gap matrix
                gap_matrix = np.column_stack((gap, second_column))

                # Create vectors dictionary for the current method
                vectors_dict = {
                    "Observed Unit": np.round(y.reshape(-1, 1), 3),
                    "Counterfactual": np.round(y_counterfactual.reshape(-1, 1), 3),
                    "Gap": np.round(gap_matrix, 3)
                }

                t0_rmse = np.sqrt(np.mean((y[:t1] - y_counterfactual[:t1]) ** 2))
                t1_rmse = np.sqrt(np.mean((y[t1:] - y_counterfactual[t1:]) ** 2))

                # Create Fit_dict for the specific method
                fit_dict = {
                    "Fit": {
                        "T0 RMSE":  round(t0_rmse,3),
                        "T1 RMSE":  round(t1_rmse,3),
                        "Pre-Periods": t1,
                        "Post-Periods": len(y[t1:])
                    },
                    "Effects": effects_dict,
                    "Vectors": vectors_dict,
                    "WeightV": np.round(b, 3),
                    "Weights": weights_dict
                }

                # Append fit_dict to the list
                fit_dicts_list.append({method: fit_dict})

            return fit_dicts_list


        result = solve_optimization(Xbar, y, t1, donor_names)

        b_MSC_c = next((method_dict["MSC_c"]["WeightV"] for method_dict in result if "MSC_c" in method_dict), None)
        b_SC = next((method_dict["SC"]["WeightV"] for method_dict in result if "SC" in method_dict), None)
        b_MSC_a = next((method_dict["MSC_a"]["WeightV"] for method_dict in result if "MSC_a" in method_dict), None)
        b_MSC_b = next((method_dict["MSC_b"]["WeightV"] for method_dict in result if "MSC_b" in method_dict), None)

        d1t = np.dot(R1t, b_MSC_c) - q1t
        d2t = np.dot(R2t, b_MSC_c) - q2t
        dt = np.dot(Rt, b_MSC_c) - qt
        test1 = t1 * np.dot(d1t.T, d1t)
        test2 = t1 * np.dot(d2t.T, d2t)

        z1 = np.hstack((x1, y1.reshape(-1, 1)))
        bm_MSC_c_sum0 = np.zeros(nb)
        V_hatI = np.zeros((2, 2))
        d1t_s = np.zeros(nb)
        d2t_s = np.zeros(nb)
        test1_s = np.zeros(nb)
        test2_s = np.zeros(nb)

        for g in range(nb):
            m = t1
            zm = z1[np.random.choice(z1.shape[0], m, replace=True), :]
            ym = zm[:, -1]
            xm = zm[:, :-1]

            lb = np.zeros(n)
            lb[0] = -np.inf
            bm_MSC_c[:, g] = lsq_linear(xm, ym, bounds=(lb, np.inf), method='trf', lsmr_tol='auto').x

            bm_MSC_c_g = bm_MSC_c[:, g]
            bm_MSC_c_sum0[g] = np.sum(bm_MSC_c_g[1:])

            dt_s = np.dot(Rt, bm_MSC_c_g) - qt
            dt_ss = np.dot(Rt, (bm_MSC_c_g - b_MSC_c))
            V_hatI += (m / nb) * np.outer(dt_ss, dt_ss)

            d1t_s[g] = np.dot(R1t, (bm_MSC_c_g - b_MSC_c))[0]
            d2t_s[g] = np.dot(R2t, (bm_MSC_c_g - b_MSC_c))[0]

            test1_s[g] = m * np.dot(d1t_s[g].T, d1t_s[g])
            test2_s[g] = m * np.dot(d2t_s[g].T, d2t_s[g])

        V_hat = np.linalg.inv(V_hatI)

        Js_test = np.zeros(nb)
        for ggg in range(nb):
            ds = np.dot(Rt, (bm_MSC_c[:, ggg] - b_MSC_c))
            Js_test[ggg] = m * np.dot(ds.T, np.dot(V_hat, ds))

        dt = np.dot(Rt, b_MSC_c.reshape(-1, 1)) - qt
        J_test = t1 * np.dot(dt.T, np.dot(V_hat, dt))

        # Calculate p-values
        pJ = np.mean(J_test < Js_test)  # p-value for joint hypothesis H0. If fail to reject, use Original SC in Step 2. If reject, then look at p1
        p1 = np.mean(test1 < test1_s)   # p-value for single restriction hypothesis test of sum to one H0a. If fail to reject, use MSCa in Step 2. If reject, then look at p2
        p2 = np.mean(test2 < test2_s)   # p-value for single restriction hypothesis test of zero intercept H0b. If fail to reject, use MSCb in step 2. Otherwise, use MSC in step 2.
        # Check p-values and recommend SCM model
        if pJ >= 0.05:
            recommended_model = "MSC_c"
        elif pJ < 0.05 and p1 >= 0.05 and p2 >= 0.05:
            recommended_model = "MSC_a"
        elif pJ < 0.05 and (p1 < 0.05 or p2 < 0.05):
            recommended_model = "MSC_b"
        else:
            recommended_model = "No recommendation due to inconclusive p-values"

        print("Recommended SCM model:", recommended_model)

        recommended_variable = next(
            (method_dict[recommended_model]["Vectors"]["Counterfactual"] for method_dict in result if
             recommended_model in method_dict), None)

        intervention_point = self.df.loc[
            self.df[self.treated] == 1, self.time
        ].min()

        time_axis = self.df[self.df[self.unitid] == treated_unit_name][
            self.time
        ].values

        # Plotting recommended SCM model
        plt.figure(figsize=(10, 6))  # Adjust size as needed

        # Plot the recommended model
        plt.plot(time_axis, recommended_variable, label=f'{recommended_model}  {treated_unit_name}', linestyle='-', color=self.counterfactual_color, linewidth=2)
        plt.plot(time_axis, y, label='Observed '+treated_unit_name, color=self.treated_color, linewidth=2)
        plt.xlabel(self.time)
        plt.ylabel(self.outcome)
        plt.title('Two-Step Synthetic Control')
        plt.axvline(
            x=intervention_point,
            color="black",
            linestyle="--",
            linewidth=2,
            label=self.treated + ", " + str(intervention_point)
        )
        plt.legend()  # Show legend
        plt.show()

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

