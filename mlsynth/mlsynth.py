import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import inv
from scipy.stats import norm
import scipy.stats as stats
import cvxpy as cp
import warnings
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
from mlsynth.utils.helperutils import prenorm
from mlsynth.utils.datautils import balance, dataprep, proxy_dataprep, clean_surrogates2
from mlsynth.utils.resultutils import effects, plot_estimates
from mlsynth.utils.estutils import Opt, pcr, TSEST, pda, pi, pi_surrogate, pi_surrogate_post, get_theta, get_sigmasq, SRCest, RPCASYNTH, SMOweights
from mlsynth.utils.inferutils import step2, ag_conformal
from mlsynth.utils.selectorsutils import fpca
from mlsynth.utils.denoiseutils import (
    RPCA,
    spectral_rank,
    RPCA_HQF,
    DC_PR_with_suggested_rank,
    standardize,
    nbpiid,
    demean_matrix,
)


class TSSC:
    def __init__(self, config):
        """
        Generate estimates for SIMPLEX, MSCa, MSCb, and MSCc methods.

        Parameters
        ----------

        config : dict

            A dictionary containing the necessary parameters. The following keys are expected:

            df : pandas.DataFrame

                Input dataset. At minimum, the user must have one column for the string or numeric unit identifier, one column for time, another column for the numeric outcome, and, finally, a column that is a dummy variable, equal to 1 when the unit is treated, else 0.

            treat : str

                Column name identifying the treated unit (must be a 0 or 1 dummy).

            time : str

                Column name for the time variable (must be numeric).

            outcome : str

                Column name for the outcome variable.

            unitid : str

                Column name identifying the units.

            counterfactual_color : str, optional

                Color for the counterfactual line in the plots, by default "red".

            treated_color : str, optional
                Color for the treated line in the plots, by default "black".

            display_graphs : bool, optional

                Whether to display the plots, by default True.

            save : bool or dict, optional

                Whether to save the generated plots. Default is False.

                If a dictionary, keys can include:
                    - 'filename' : Custom file name (without extension).
                    - 'extension' : File format (e.g., 'png', 'pdf').
                    - 'directory' : Directory to save the plot.

            draws : int, optional
                Number of subsample replications, by default 500.

        Returns
        -------

        dict
            A dictionary with the following keys:

            'SIMPLEX' : dict

                Estimates and inference from the SIMPLEX method.

            'MSCa' : dict

                Estimates and inference from the MSCa method.

            'MSCb' : dict

                Estimates and inference from the MSCb method.

            'MSCc' : dict

                Estimates and inference from the MSCc method.
        """

        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)
        self.draws = config.get("draws", 500)

    def fit(self):

        balance(self.df, self.unitid, self.time)

        nb = self.draws
        prepped = dataprep(
            self.df, self.unitid, self.time, self.outcome, self.treat
        )

        x = np.concatenate(
            (np.ones((prepped["total_periods"], 1)), prepped["donor_matrix"]),
            axis=1,
        )

        result = TSEST(
            prepped["donor_matrix"],
            prepped["y"],
            prepped["pre_periods"],
            nb,
            prepped["donor_names"],
            prepped["post_periods"],
        )

        n = x.shape[1]

        b_MSC_c = next(
            (
                method_dict["MSCc"]["WeightV"]
                for method_dict in result
                if "MSCc" in method_dict
            ),
            None,
        )
        b_SC = next(
            (
                method_dict["SIMPLEX"]["WeightV"]
                for method_dict in result
                if "SIMPLEX" in method_dict
            ),
            None,
        )
        b_MSC_a = next(
            (
                method_dict["MSCa"]["WeightV"]
                for method_dict in result
                if "MSCa" in method_dict
            ),
            None,
        )
        b_MSC_b = next(
            (
                method_dict["MSCb"]["WeightV"]
                for method_dict in result
                if "MSCb" in method_dict
            ),
            None,
        )

        recommended_model = step2(
            np.array([[0] + list(np.ones(n - 1))]),
            np.array([[1] + list(np.zeros(n - 1))]),
            np.array(
                [[0] + list(np.ones(n - 1)), [1] + list(np.zeros(n - 1))]
            ),
            b_MSC_c,
            1,
            0,
            np.array([[1], [0]]),
            prepped["pre_periods"],
            x[: prepped["pre_periods"], :],
            prepped["y"][: prepped["pre_periods"]],
            nb,
            x.shape[1],
            np.zeros((n, nb)),
        )

        recommended_variable = next(
            (
                method_dict[recommended_model]["Vectors"]["Counterfactual"]
                for method_dict in result
                if recommended_model in method_dict
            ),
            None,
        )
        ATT, RMSE = next(
            (
                (
                    method_dict[recommended_model]["Effects"]["ATT"],
                    method_dict[recommended_model]["Fit"]["T0 RMSE"],
                )
                for method_dict in result
                if recommended_model in method_dict
            ),
            (None, None),
        )

        # Call the function
        if self.display_graphs:
            plot_estimates(
                df=prepped,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=[recommended_variable],
                counterfactual_names=[recommended_model],
                method="TSSC",
                treatedcolor=self.treated_color,
                counterfactualcolors=self.counterfactual_color,
                save=self.save
            )

        return result


class FMA:
    def __init__(self, config):
        """
        Compute estimates and inference using the Factor Model Approach (FMA).

        Implements the Factor Model Approach.

        Parameters
        ----------
        config : dict

            A dictionary containing the necessary parameters. The following keys are expected:

            df : pandas.DataFrame

                Input dataset. At minimum, the user must have one column for the string or numeric unit identifier, one column for time, another column for the numeric outcome, and, finally, a column that is a dummy variable, equal to 1 when the unit is treated, else 0.

            treat : str

                Column name identifying the treated unit.

            time : str

                Column name for the time variable.

            outcome : str

                Column name for the outcome variable.

            unitid : str

                Column name identifying the units.

            counterfactual_color : str, optional

                Color for the counterfactual line in the plots, by default "red".

            treated_color : str, optional

                Color for the treated line in the plots, by default "black".

            display_graphs : bool, optional

                Whether to display the plots, by default True.

            save : bool or dict, optional

                Whether to save the generated plots. Default is False.

                If a dictionary, keys can include:

                    - 'filename' : Custom file name (without extension).
                    - 'extension' : File format (e.g., 'png', 'pdf').
                    - 'directory' : Directory to save the plot.

            criti : int, optional
                A value to indicate whether the data is assumed to be stationary or nonstationary.
                If criti = 11, nonstationarity is assumed; if criti = 10, stationarity is assumed. Default is 11.

            DEMEAN : int, optional

                A value that determines how the data is processed:
                - If DEMEAN = 1, the data is demeaned.
                - If DEMEAN = 2, the data is standardized.
                Default is 1.

        Returns
        -------
        dict
            A dictionary containing the following keys:

            'Effects' : dict
                Estimated treatment effects for the treated unit over time.

            'Fit' : dict
                Goodness-of-fit metrics for the model.

            'Vectors' : dict
                Observed, counterfactual, and treatment effect vectors.

            'Inference' : dict
                Inference results, including confidence intervals and p-values.

        References
        ----------
        Li, K. T. & Sonnier, G. P. (2023). "Statistical Inference for the Factor Model Approach to Estimate Causal Effects in Quasi-Experimental Settings." *Journal of Marketing Research*, Volume 60, Issue 3.
        """

        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)
        self.criti = config.get("criti", 11)
        self.DEMEAN = config.get("DEMEAN", 1)

    def fit(self):

        balance(self.df, self.unitid, self.time)

        prepped = dataprep(
            self.df, self.unitid, self.time, self.outcome, self.treat
        )
        t = prepped["total_periods"]
        t1 = prepped["pre_periods"]

        t2 = prepped["post_periods"]
        datax = prepped["donor_matrix"]
        y1 = prepped["y"][: prepped["pre_periods"]]
        y2 = prepped["y"][prepped["post_periods"]:]
        y = prepped["y"]
        N_co = datax.shape[1]
        n_cutoff = 70
        m_N = max(0, n_cutoff - N_co)
        m_T = max(0, n_cutoff - t)
        const = np.ones((t, 1))
        x = np.hstack((const, datax))
        x1 = x[:t1, :]
        x2 = x[t1:, :]
        X = demean_matrix(datax)  # For Xu's method
        XX = np.dot(X, X.T)
        eigval, Fhat0 = np.linalg.eigh(XX)
        Fhat0 = Fhat0[:, ::-1]
        t1_one = np.ones((t1, 1))
        t2_one = np.ones((t2, 1))
        rmax = 10
        DEMEAN = self.DEMEAN
        MSE = np.zeros(rmax)
        for r in range(1, rmax + 1):
            Fhat_1 = np.hstack((t1_one, Fhat0[:t1, :r]))
            u1_hat_xu = np.zeros(t)
            for s in range(t1):
                Fhat_1_xu = Fhat_1.copy()
                Fhat_1_xu = np.delete(Fhat_1_xu, s, axis=0)
                y1_xu = np.delete(y1, s)
                lambda_hat_xu = (
                    inv(np.dot(Fhat_1_xu.T, Fhat_1_xu))
                    .dot(Fhat_1_xu.T)
                    .dot(y1_xu)
                )
                u1_hat_xu[s] = y1[s] - np.dot(Fhat_1[s, :], lambda_hat_xu)
            MSE[r - 1] = np.mean(u1_hat_xu**2)
        mse_min, nfactor_xu = np.min(MSE), np.argmin(MSE) + 1

        if self.criti == 11:
            nfactor_Bai, _, _ = nbpiid(X, rmax, self.criti, DEMEAN, m_N, m_T)
            nfactor = min(nfactor_Bai, nfactor_xu)
        elif self.criti == 10:
            nfactor_MBN, _, _ = nbpiid(X, rmax, self.criti, DEMEAN, m_N, m_T)
            nfactor = min(nfactor_MBN, nfactor_xu)
        F_hat1 = np.hstack((t1_one, Fhat0[:t1, :nfactor]))
        F_hat2 = np.hstack((t2_one, Fhat0[t1:t, :nfactor]))

        b_hat = inv(np.dot(F_hat1.T, F_hat1)).dot(F_hat1.T).dot(y1)
        y1_hat = np.dot(F_hat1, b_hat)
        y2_hat = np.dot(F_hat2, b_hat)
        y_hat = np.concatenate((y1_hat, y2_hat))
        attdict, fitdict, Vectors = effects.calculate(
            prepped["y"],
            y_hat,
            prepped["pre_periods"],
            prepped["post_periods"],
        )
        e1_hat = y1 - np.dot(F_hat1, b_hat)

        # Create the dictionary of statistics
        sigma2_e_hat = np.mean(e1_hat**2)  # \hat sigma^2_e
        hat_eta = np.mean(F_hat2, axis=0)[:, np.newaxis]
        e1_2 = e1_hat**2
        E_FF = np.dot(F_hat1.T, F_hat1) / t1
        hat_Psi = (
            np.linalg.inv(E_FF)
            .dot(np.dot(F_hat1.T, np.diag(e1_2)).dot(F_hat1) / t1)
            .dot(np.linalg.inv(E_FF))
        )
        Omega_1_hat = (t2 / t1) * np.dot(np.dot(hat_eta.T, hat_Psi), hat_eta)
        Omega_2_hat = sigma2_e_hat
        # variance of sqrt(t2)(\hat \Delta_1 - \Delta_1)
        Omega_hat = Omega_1_hat + Omega_2_hat

        AA = np.sqrt(Omega_hat) / np.sqrt(t2)

        def calculate_cr(perc, att, AA):
            return att - norm.ppf(perc) * AA

        # ATT and AA values
        ATT = attdict["ATT"]

        # Calculate the bounds in one line each
        cr005, cr995 = calculate_cr(0.005, ATT, AA), calculate_cr(
            0.995, ATT, AA
        )
        cr025, cr975 = calculate_cr(0.025, ATT, AA), calculate_cr(
            0.975, ATT, AA
        )
        cr05, cr95 = calculate_cr(0.05, ATT, AA), calculate_cr(0.95, ATT, AA)

        # 95% CI and 90% CI
        c90FM = [cr05, cr95, cr95 - cr05]
        c95FM = [cr025, cr975, cr975 - cr025]

        confidence_interval = (c95FM[1].item(), c95FM[2].item())
        t_stat = ATT / AA  # t-statistic
        # Calculate p-value using the t-statistic
        p_value = 2 * (1 - norm.cdf(abs(t_stat)))

        Inference = {
            "SE": AA,
            "tstat": t_stat,
            "95% CI": confidence_interval,
            "p_value": p_value,
        }
        if self.display_graphs:

            plot_estimates(
               prepped,
                self.time,
                self.unitid,
                self.outcome,
                self.treat,
                prepped["treated_unit_name"],
                prepped["y"],
                [y_hat],
                method="FMA",
                treatedcolor=self.treated_color,
                counterfactualcolors=["red"],
                counterfactual_names=[f"FMA {prepped['treated_unit_name']}"],
                save=self.save,
            )

        return {
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors,
            "Inference": Inference,
        }


class PDA:
    def __init__(self, config):
        """
        Implements the Panel Data Approach (PDA).

        Parameters
        ----------
        config : dict

            A dictionary containing the necessary parameters. The following keys are expected:

            df : pandas.DataFrame

                Input dataset. At minimum, the user must have one column for the string or numeric unit identifier,
                one column for time, another column for the numeric outcome, and, finally, a column that is a dummy
                variable, equal to 1 when the unit is treated, else 0.

            treat : str

                Column name identifying the treated unit.

            time : str

                Column name for the time variable.

            outcome : str

                Column name for the outcome variable.

            unitid : str

                Column name identifying the units.

            counterfactual_color : str, optional

                Color for the counterfactual line in the plots, by default "red".

            treated_color : str, optional

                Color for the treated line in the plots, by default "black".

            display_graphs : bool, optional

                Whether to display the plots, by default True.

            save : bool or dict, optional

                Whether to save the generated plots. Default is False.

                If a dictionary, keys can include:

                    - 'filename' : Custom file name (without extension).
                    - 'extension' : File format (e.g., 'png', 'pdf').
                    - 'directory' : Directory to save the plot.

            method : str, optional

                Type of PDA to use, either:
                - "LASSO" (L1 Penalty),
                - "l2" (L2-Relaxation),
                - "fs" (Forward Selection), default.

            tau : float, optional

                A user-specified treatment effect value, default is None.

        Returns
        -------
        dict
            A dictionary containing the estimated treatment effects, fit metrics,
            and inference results depending on the selected PDA method.

        References
        ----------
        Shi, Z. & Huang, J. (2023). "Forward-selected panel data approach for program evaluation."
        *Journal of Econometrics*, Volume 234, Issue 2, Pages 512-535.
        DOI: https://doi.org/10.1016/j.jeconom.2021.04.009

        Li, Kathleen T., and David R. Bell. "Estimation of Average Treatment Effects with Panel Data:
        Asymptotic Theory and Implementation." *Journal of Econometrics* 197, no. 1 (March, 2017): 65-75.
        DOI: https://doi.org/10.1016/j.jeconom.2016.01.011

        Shi, Z. & Wang, Y. (2024). "L2-relaxation for Economic Prediction."
        DOI: https://doi.org/10.13140/RG.2.2.11670.97609.
        """

        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)
        self.method = config.get("method", "fs")
        self.tau = config.get("tau", None) 

        if self.tau is not None and not isinstance(self.tau, (int, float)):
            raise ValueError("tau must be a numeric value.")

    def fit(self):

        balance(self.df, self.unitid, self.time)

        prepped = dataprep(
            self.df, self.unitid, self.time, self.outcome, self.treat
        )

        pdaest = pda(prepped, len(prepped["donor_names"]), method=self.method,tau=self.tau)
        attdict, fitdict, Vectors = effects.calculate(
            prepped["y"],
            pdaest["Vectors"]["Counterfactual"],
            prepped["pre_periods"],
            prepped["post_periods"],
        )
        est_method = pdaest.get("method")

        counterfactual_name = f'{est_method} {prepped["treated_unit_name"]}'

        if self.display_graphs:

            plot_estimates(
                df=prepped,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=[pdaest["Vectors"]["Counterfactual"]],
                counterfactual_names=[counterfactual_name],
                method=est_method,
                treatedcolor=self.treated_color,
                counterfactualcolors=self.counterfactual_color,
                save=self.save,
            )

        return pdaest


class FDID:
    def __init__(self, config):
        """
        Compute Forward DID, Augmented DID, and standard DID estimates.

        Parameters
        ----------
        config : dict

            Dictionary containing the configuration options. The following keys are expected:

            df : pandas.DataFrame

                Input dataset. At minimum, the user must have one column for the string or numeric unit identifier, one column for time, another column for the numeric outcome, and, finally, a column that is a dummy variable, equal to 1 when the unit is treated, else 0.

            unitid : str

                Column name for unit IDs.

            time : str

                Column name for time periods.

            outcome : str

                Column name for outcomes.

            treat : str

                Column name for the treatment indicator.

            counterfactual_color : str, optional

                Color for the counterfactual lines, by default "red".

            treated_color : str, optional

                Color for the treated lines, by default "black".

            display_graphs : bool, optional

                Whether to display the graphs, by default True.

            save : bool or dict, optional
                Whether to save the generated plots. Default is False.
                If a dictionary, keys can include:
                    - 'filename' : Custom file name (without extension).
                    - 'extension' : File format (e.g., 'png', 'pdf').
                    - 'directory' : Directory to save the plot.

        Returns
        -------
        dict

            A dictionary containing the following keys for each method (FDID, ADID, DID):

            'Effects' : dict

                ATTs: ATT, percent ATT, Standardized Effect Size

            'Fit' : dict

                Goodness-of-fit metrics for the model: R-Squared, Pre-RMSE

            'Inference' : dict

                Inference results, including 95% confidence intervals and p-values.

        References
        ----------
        Li, K. T. (2023). "Frontiers: A Simple Forward Difference-in-Differences Method."
        *Marketing Science* 43(2):267-279.

        Li, K. T. & Van den Bulte, C. (2023). "Augmented Difference-in-Differences."
        *Marketing Science* 42:4, 746-767.
        """

        # Required parameters
        self.df = config.get("df")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.outcome = config.get("outcome")
        self.treated = config.get("treat")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)

    def DID(self, y, datax, t1, itp=0):
        t = len(y)

        x1, x2 = np.mean(datax[:t1], axis=1).reshape(-1, 1), np.mean(
            datax[t1:t], axis=1
        ).reshape(-1, 1)
        b_DID = np.mean(y[:t1] - x1, axis=0)  # DID intercept estimator

        y1_DID = b_DID + x1  # DID in-sample-fit
        y2_DID = b_DID + x2  # DID out-of-sample prediction
        y_DID = np.vstack(
            (y1_DID, y2_DID)
        )  # Stack y1_DID and y2_DID vertically

        y1_DID, y2_DID = y_DID[:t1], y_DID[t1:t]

        # if hasattr(self, 'placebo_option'):
        # t1 = self.realt1
        if itp > 0:
            t1 + itp

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

        # print(f'({t2} / {t1}) * {np.mean(u1_DID ** 2)}')
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

        # print(z_critical * std_Omega_hat_DID / np.sqrt(t2))

        CI_95_DID_left = ATT_DID - z_critical * std_Omega_hat_DID / np.sqrt(t2)
        # print(f"{ATT_DID} - {z_critical} * {std_Omega_hat_DID} / np.sqrt({t2})")
        CI_95_DID_right = ATT_DID + z_critical * std_Omega_hat_DID / np.sqrt(
            t2
        )
        CI_95_DID_width = [
            CI_95_DID_left,
            CI_95_DID_right,
            CI_95_DID_right - CI_95_DID_left,
        ]
        if itp > 0:
            t1 - itp

        # Metrics of fit subdictionary
        Fit_dict = {
            "T0 RMSE": round(np.std(y[:t1] - y_DID[:t1]), 3),
            "R-Squared": round(R2_DID, 3),
            "Pre-Periods": t1,
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
            "SE": std_Omega_hat_DID / np.sqrt(t2),
            "Intercept": np.round(b_DID, 3),
        }

        gap = y - y_DID

        second_column = np.arange(gap.shape[0]) - t1 + 1

        gap_matrix = np.column_stack((gap, second_column))

        # Vectors subdictionary
        Vectors = {
            "Observed Unit": np.round(y, 3),
            "Counterfactual": np.round(y_DID, 3),
            "Gap": np.round(gap_matrix, 3),
        }

        # Main dictionary
        DID_dict = {
            "Effects": ATTS,
            "Vectors": Vectors,
            "Fit": Fit_dict,
            "Inference": Inference,
        }

        return DID_dict

    def AUGDID(self, datax, t, t1, t2, y, y1, y2):

        const = np.ones(t)  # t by 1 vector of ones (for intercept)
        # add an intercept to control unit data matrix, t by N (N=11)
        x = np.column_stack([const, datax])
        x1 = x[:t1, :]  # control units' pretreatment data matrix, t1 by N
        x2 = x[t1:, :]  # control units' pretreatment data matrix, t2 by N

        # ATT estimation by ADID method
        x10 = datax[:t1, :]
        x20 = datax[t1:, :]
        x1_ADID = np.column_stack(
            [np.ones(x10.shape[0]), np.mean(x10, axis=1)]
        )
        x2_ADID = np.column_stack(
            [np.ones(x20.shape[0]), np.mean(x20, axis=1)]
        )

        b_ADID = np.linalg.inv(x1_ADID.T @ x1_ADID) @ (
            x1_ADID.T @ y1
        )  # ADID estimate of delta

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

        Omega_1_ADID = (
            (sigma2_ADID * eta_ADID.T) @ np.linalg.pinv(psi_ADID) @ eta_ADID
        )
        Omega_2_ADID = sigma2_ADID

        Omega_ADID = (t2 / t1) * Omega_1_ADID + Omega_2_ADID  # Variance

        ATT_std = np.sqrt(t2) * ATT / np.sqrt(Omega_ADID)

        quantile = norm.ppf(0.975)

        CI_95_DID_left = ATT - quantile * np.sqrt(sigma2_ADID) / np.sqrt(t2)
        CI_95_DID_right = ATT + quantile * np.sqrt(sigma2_ADID) / np.sqrt(t2)

        RMSE = np.sqrt(np.mean((y1 - y1_ADID) ** 2))
        RMSEPost = np.sqrt(np.mean((y2 - y2_ADID) ** 2))

        R2_ADID = 1 - (np.mean((y1 - y1_ADID) ** 2)) / np.mean(
            (y1 - np.mean(y1)) ** 2
        )

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
            "T0": len(y[:t1]),
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
            "Width": CI_95_DID_right - CI_95_DID_left,
        }
        gap = y - y_ADID

        second_column = np.arange(gap.shape[0]) - t1 + 1

        gap_matrix = np.column_stack((gap, second_column))

        # Vectors subdictionary
        Vectors = {
            "Observed Unit": np.round(y, 3),
            "Counterfactual": np.round(y_ADID, 3),
            "Gap": np.round(gap_matrix, 3),
        }

        # Main dictionary
        ADID_dict = {
            "Effects": ATTS,
            "Vectors": Vectors,
            "Fit": Fit_dict,
            "Inference": Inference,
        }

        return ADID_dict, y_ADID

    def est(self, control, t, t1, t2, y, y1, y2, datax):

        FDID_dict = self.DID(y.reshape(-1, 1), control, t1)

        y_FDID = FDID_dict["Vectors"]["Counterfactual"]

        DID_dict = self.DID(y.reshape(-1, 1), datax, t1)

        AUGDID_dict, y_ADID = self.AUGDID(datax, t, t1, t2, y, y1, y2)
        time_points = np.arange(1, len(y) + 1)

        # Calculate the ratio of widths for DID and AUGDID compared to FDID
        ratio_DID = (
            DID_dict["Inference"]["Width"] / FDID_dict["Inference"]["Width"]
        )
        ratio_AUGDID = (
            AUGDID_dict["Inference"]["Width"] / FDID_dict["Inference"]["Width"]
        )

        # Add the new elements to the Inference dictionaries
        DID_dict["Inference"]["WidthRFDID"] = ratio_DID
        AUGDID_dict["Inference"]["WidthRFDID"] = ratio_AUGDID

        return FDID_dict, DID_dict, AUGDID_dict, y_FDID

    def selector(self, no_control, t1, t, y, datax, control_ID):
        # Preallocate R-squared arrays and select array
        R2final = np.zeros(no_control)
        control_ID_adjusted = np.array(control_ID) - 1
        select_c = np.zeros(no_control, dtype=int)

        # Reshape y just once
        y_reshaped = y.reshape(-1, 1)

        # Initialize the selected control matrix
        selected_controls = np.empty((t, 0))

        for k in range(no_control):
            # Find the best control to add at the current step
            left = np.setdiff1d(control_ID_adjusted, select_c[:k])
            R2 = np.zeros(len(left))

            for jj, control_idx in enumerate(left):
                # Combine previously selected controls with the current
                # candidate
                combined_control = np.hstack(
                    (
                        selected_controls[:t, :],
                        datax[:, control_idx].reshape(-1, 1),
                    )
                )

                # Estimate DiD and compute R-squared for this combination
                ResultDict = self.DID(y_reshaped, combined_control, t1)
                R2[jj] = ResultDict["Fit"]["R-Squared"]

            # Update R-squared and the best control
            R2final[k] = np.max(R2)

            best_new_control = left[np.argmax(R2)]
            select_c[k] = best_new_control

            # Update the selected controls matrix with the new control
            selected_controls = np.hstack(
                (selected_controls, datax[:, best_new_control].reshape(-1, 1))
            )
        # Get the index of the model with the highest R2
        best_model_idx = R2final.argmax()
        Uhat = datax[:, select_c[: best_model_idx + 1]]

        return select_c[: best_model_idx + 1], R2final, Uhat

    def fit(self):

        balance(self.df, self.unitid, self.time)

        prepped = dataprep(
            self.df, self.unitid, self.time, self.outcome, self.treated
        )

        y1 = np.ravel(prepped["y"][: prepped["pre_periods"]])
        y2 = np.ravel(prepped["y"][-prepped["post_periods"]:])

        selected, R2_final, Uhat = self.selector(
            prepped["donor_matrix"].shape[1],
            prepped["pre_periods"],
            prepped["total_periods"],
            prepped["y"],
            prepped["donor_matrix"],
            np.arange(1, prepped["donor_matrix"].shape[1] + 1),
        )

        selected_controls = (
            prepped["donor_names"].take(selected.tolist()).tolist()
        )

        FDID_dict, DID_dict, AUGDID_dict, y_FDID = self.est(
            Uhat,
            prepped["total_periods"],
            prepped["pre_periods"],
            prepped["post_periods"],
            prepped["y"],
            y1,
            y2,
            prepped["donor_matrix"],
        )

        estimators_results = []

        # Calculate the weight
        weight = 1 / len(selected_controls)

        # Create the dictionary
        unit_weights = {unit: weight for unit in selected_controls}

        FDID_dict["Weights"] = unit_weights

        estimators_results.append({"FDID": FDID_dict})

        # Append the normal DID dictionary to the list
        estimators_results.append({"DID": DID_dict})
        estimators_results.append({"AUGDID": AUGDID_dict})

        def round_dict_values(input_dict, decimal_places=3):
            rounded_dict = {}
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    # Recursively round nested dictionaries
                    rounded_dict[key] = round_dict_values(
                        value, decimal_places
                    )
                elif isinstance(value, (int, float, np.float64)):
                    # Round numeric values
                    rounded_dict[key] = round(value, decimal_places)
                else:
                    rounded_dict[key] = value
            return rounded_dict

        DID_counterfactual = DID_dict["Vectors"]["Counterfactual"]

        if self.display_graphs:

            plot_estimates(
                prepped,
                self.time,
                self.unitid,
                self.outcome,
                self.treated,
                prepped["treated_unit_name"],
                prepped["y"],
                [y_FDID, DID_counterfactual],
                method="FDID",
                counterfactual_names=["FDID " + prepped["treated_unit_name"], "DID " + prepped["treated_unit_name"]],
                treatedcolor=self.treated_color,
                save=self.save,
                counterfactualcolors=self.counterfactual_color,
            )

        return estimators_results


class GSC:
    def __init__(self, config):
        """
        Compute estimates and inference using the Generalized Synthetic Control (GSC) method.

        This function implements the Generalized Synthetic Control method as described in
        Costa et al. (2023). It returns a dictionary containing the estimated effects,
        model fit, factor vectors, and inference results, including t-statistics,
        standard errors, and 95% confidence intervals.

        Parameters
        ----------
        config : dict

            A dictionary containing the necessary parameters. The following keys are expected:

            df : pandas.DataFrame

                Input dataset. At minimum, the user must have one column for the string or numeric unit identifier, one column for time, another column for the numeric outcome, and, finally, a column that is a dummy variable, equal to 1 when the unit is treated, else 0.

            treat : str
                Column name identifying the treated unit.
            time : str
                Column name for the time variable.
            outcome : str
                Column name for the outcome variable.
            unitid : str
                Column name identifying the units.
            counterfactual_color : str, optional
                Color for the counterfactual line in the plots, by default "red".
            treated_color : str, optional
                Color for the treated line in the plots, by default "black".
            display_graphs : bool, optional
                Whether to display the plots, by default True.
            save : bool, optional
                Whether to save the generated plots, by default False.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            'Effects' : dict
                Estimated treatment effects for the treated unit over time.
            'Fit' : dict
                Goodness-of-fit metrics for the model.
            'Vectors' : dict
                Factor model vectors, including factor loadings and common factors.
            'Inference' : dict
                Inference results, including:
                - 't_stat' : t-statistics for the estimated effects.
                - 'se' : Standard errors of the estimates.
                - '95% ci' : 95% confidence intervals for the estimated effects.

        References
        ----------
        Costa, L., Farias, V. F., Foncea, P., Gan, J. (D.), Garg, A., Montenegro, I. R.,
        Pathak, K., Peng, T., & Popovic, D. (2023). "Generalized Synthetic Control for TestOps at ABI:
        Models, Algorithms, and Infrastructure." *INFORMS Journal on Applied Analytics* 53(5):336-349.
        """

        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)

    def fit(self):

        balance(self.df, self.unitid, self.time)

        prepped = dataprep(
            self.df, self.unitid, self.time, self.outcome, self.treat
        )
        treatedunit = set(self.df.loc[self.df[self.treat] == 1, self.unitid])

        Ywide = prepped["Ywide"].T

        # Filter the row indices of Ywide using treatedunit
        row_indices_of_treatedunit = Ywide.index[
            Ywide.index == prepped["treated_unit_name"]
        ]

        # Get the row index numbers corresponding to the treated units
        treatrow = [
            Ywide.index.get_loc(idx) for idx in row_indices_of_treatedunit
        ][0]

        Z = np.zeros_like(Ywide.to_numpy())

        n, p = prepped["donor_matrix"][: prepped["pre_periods"]].shape
        k = (min(n, p) - 1) // 2

        # Y0_rank, Topt_gauss, rank = adaptiveHardThresholding(prepped["donor_matrix"][:prepped["pre_periods"]], k, strategy='i')

        Z[treatrow, -prepped["post_periods"]:] = 1
        result = DC_PR_with_suggested_rank(
            Ywide.to_numpy(), Z, suggest_r=rank, method="non-convex"
        )

        if self.display_graphs:
            plot_estimates(
                df=prepped,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=[result["Vectors"]["Counterfactual"]],
                counterfactual_names=["GSC"],
                method="GSC",
                treatedcolor=self.treated_color,
                counterfactualcolors=[self.counterfactual_color],
            )

        return result



class CLUSTERSC:
    def __init__(self, config):
        """
        This function provides ATT estimates and weights using Robust PCA Synthetic Control (RPCA SCM) and/or Principal Component Regression (PCR).

        Parameters
        ----------
        config : dict
            A dictionary containing the necessary parameters. The following keys are expected:

            df : pandas.DataFrame
                Input dataset. At minimum, the user must have one column for the string or numeric unit identifier, one column for time, another column for the numeric outcome, and, finally, a column that is a dummy variable, equal to 1 when the unit is treated, else 0.

            treat : str
                Column name identifying the treated unit.

            time : str
                Column name for the time variable.

            outcome : str
                Column name for the outcome variable.

            unitid : str
                Column name identifying the units.

            cluster : bool, optional
                Whether to apply clustering for PCR. Default is True.

            objective : str, optional
                Constraint for PCR. Default is "OLS", but user may specify "SIMPLEX".

            method : str, optional
                Specifies which estimation method(s) to run. Options are:
                    - "PCR": Run Principal Component Regression only.
                    - "RPCA": Run Robust PCA Synthetic Control only.
                    - "BOTH": Run both PCR and RPCA methods.
                Default is "PCR".

            counterfactual_color : str or list of str, optional
                Color for the counterfactual line(s) in the plots. If a string, the same color is used for all methods. If a list, it must contain as many color strings as there are counterfactuals returned by the estimator (e.g., two colors if both PCR and RPCA are run). Default is "red".

            treated_color : str, optional
                Color for the treated line in the plots. Default is "black".

            display_graphs : bool, optional
                Whether to display the plots. Default is True.

            save : bool or dict, optional
                Whether to save the generated plots. Default is False.
                If a dictionary, keys can include:
                    - 'filename' : Custom file name (without extension).
                    - 'extension' : File format (e.g., 'png', 'pdf').
                    - 'directory' : Directory to save the plot.

            Frequentist : bool, optional
                If true, use Frequentist Robust SCM.
                If False, uses Amjad's Bayesian method.
                Defaults to True.

            Robust : str, optional
                Specifies the robust method to use. If "PCP", Principal Component Pursuit (PCP) is used for Robust PCA. If "HQF", non-convex half-quadratic regularization is applied. Defaults to "PCP".

        Returns
        -------
        dict
            A dictionary containing results for the specified method(s), with the following keys:

            'Weights' : list of dict
                A two-element list where:
                    - The first element is a dictionary mapping unit IDs to their assigned weights.
                    - The second element is a dictionary of unit-weight pairs where the weights are strictly positive.

            'Effects' : dict
                Estimated treatment effects for the treated unit over time.

            'Vectors' : dict
                Observed, predicted, and treatment effects for the selected methods.

        References
        ----------
        Amjad, M., Shah, D., & Shen, D. (2018). "Robust synthetic control."
        *Journal of Machine Learning Research*, 19(22), 1-51.

        Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). "On Robustness of Principal Component Regression."
        *Journal of the American Statistical Association*, 116(536), 1731–45.

        Bayani, M. (2022). "Essays on Machine Learning Methods in Economics." Chapter 1.
        *CUNY Academic Works*.

        Wang, Zhi-Yong, Xiao Peng Li, Hing Cheung So, and Zhaofeng Liu. (2023). "Robust PCA via non-convex half-quadratic regularization."
        *Signal Processing*, 204, 108816.
        """


        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)
        self.objective = config.get("objective", "OLS")
        self.cluster = config.get("cluster", True)
        self.Frequentist = config.get("Frequentist", True)
        self.ROB = config.get("Robust", "PCP")
        self.method = config.get("method", "PCR").upper()

    def fit(self):
        balance(self.df, self.unitid, self.time)

        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treat)

        results = {}

        match self.method.upper():
            case "PCR":
                result = pcr(
                    prepped["donor_matrix"],
                    prepped["y"],
                    self.objective,
                    prepped["donor_names"],
                    prepped["donor_matrix"],
                    pre=prepped["pre_periods"],
                    cluster=self.cluster,
                    Frequentist=self.Frequentist
                )

                attdict, fitdict, Vectors = effects.calculate(
                    prepped["y"], result['cf_mean'], prepped["pre_periods"], prepped["post_periods"]
                )

                results["PCR"] = {
                    "Effects": attdict,
                    "Fit": fitdict,
                    "Vectors": Vectors,
                    "Weights": [result["weights"], {k: round(v, 3) for k, v in result["weights"].items() if v > 0}]
                }

            case "RPCA":
                results["RPCA"] = RPCASYNTH(self.df, self.__dict__, prepped)

            case "BOTH":
                # PCR block
                result = pcr(
                    prepped["donor_matrix"],
                    prepped["y"],
                    self.objective,
                    prepped["donor_names"],
                    prepped["donor_matrix"],
                    pre=prepped["pre_periods"],
                    cluster=self.cluster,
                    Frequentist=self.Frequentist
                )

                attdict, fitdict, Vectors = effects.calculate(
                    prepped["y"], result['cf_mean'], prepped["pre_periods"], prepped["post_periods"]
                )

                results["PCR"] = {
                    "Effects": attdict,
                    "Fit": fitdict,
                    "Vectors": Vectors,
                    "Weights": result["weights"]
                }

                # RPCA block
                results["RPCA"] = RPCASYNTH(self.df, self.__dict__, prepped)

            case _:
                raise ValueError("method must be 'PCR', 'RPCA', or 'BOTH'")

        # Assuming results is a dictionary that contains either PCR or RPCA results
        all_counterfactuals = {}

        # Iterate over all methods in the results dictionary
        for method, method_data in results.items():
            # Check if the "Vectors" key exists and contains "Counterfactual"
            if 'Vectors' in method_data and 'Counterfactual' in method_data['Vectors']:
                # Extract the counterfactual for this method
                counterfactual = method_data['Vectors']['Counterfactual']
                all_counterfactuals[method] = counterfactual

        # Determine counterfactual names based on the method and Frequentist flag
        if self.method.upper() == "PCR":
            if self.Frequentist:
                counterfactual_names = ["RSC"]  # Only RSC for Frequentist True
            else:
                counterfactual_names = ["Bayesian RSC"]  # Only Bayesian RSC for Frequentist False
        elif self.method.upper() == "RPCA":
            counterfactual_names = ["RPCA Synth"]  # Only RPCA Synth for RPCA method
        else:
            # Handle case for both methods (PCR and RPCA)
            if self.Frequentist:
                counterfactual_names = ["RSC", "RPCA Synth"]
            else:
                counterfactual_names = ["Bayesian RSC", "RPCA Synth"]

        # Initialize an empty list for the vectors to supply
        vectors_to_supply = []

        # Check if 'PCR' is in all_counterfactuals and append if it exists
        if 'PCR' in all_counterfactuals:
            vectors_to_supply.append(all_counterfactuals['PCR'])

        # Check if 'RPCA' is in all_counterfactuals and append if it exists
        if 'RPCA' in all_counterfactuals:
            vectors_to_supply.append(all_counterfactuals['RPCA'])

        # Call the function
        if self.display_graphs:
            plot_estimates(
                df=prepped,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=vectors_to_supply,
                counterfactual_names=counterfactual_names,  # Use the dynamic counterfactual names
                method="CLUSTERSC",
                treatedcolor="black",
                counterfactualcolors=self.counterfactual_color,
                save=self.save
            )

        return results


class PROXIMAL:
    """
    A class for implementing the Proximal Inference framework with surrogates, proxies, and donor units.

    to estimate causal impacts in the context of synthetic control methods.

    Parameters
    ----------
    config : dict

        A dictionary containing configuration options:

        - "df" : pandas.DataFrame

            Input dataset. At minimum, the user must have one column for the string or numeric unit identifier, one column for time, another column for the numeric outcome, and, finally, a column that is a dummy variable, equal to 1 when the unit is treated, else 0.

        - "outcome" : str

            The name of the outcome variable.

        - "treat" : str

            The name of the treatment indicator variable.

        - "unitid" : str

            The name of the unit identifier column.

        - "time" : str

            The name of the time variable.

        - "counterfactual_color" : str, optional, default="red"

            The color used for counterfactual estimates in the plots.

        - "treated_color" : str, optional, default="black"

            The color used for the treated unit in the plots.

        - "display_graphs" : bool, optional, default=True

            Whether to display the resulting plots.

        - "save" : bool or dict, optional

            Whether to save the generated plots. Default is False.

            If a dictionary, keys can include:

                - 'filename' : Custom file name (without extension).
                - 'extension' : File format (e.g., 'png', 'pdf').
                - 'directory' : Directory to save the plot.

        - "surrogates" : list, optional, default=[]

            A list of surrogate unit identifiers (string or numeric) used in the analysis.

        - "vars" : list of str, optional, default=[]

            A list of proxy variables, where:

              * The first element corresponds to proxies for the donors.
              * The second element corresponds to proxies for the surrogates.

        - "donors" : list of str, optional, default=[]

            A list of donor units to construct the counterfactual.

    Returns
    -------

    dict

        A dictionary with keys "PI", "PIS", and "PIPost", each containing the estimated effects,
        model fit, and vectors for the corresponding approach.

    References
    ----------
    Xu Shi, Kendrick Li, Wang Miao, Mengtong Hu, and Eric Tchetgen Tchetgen.
    "Theory for identification and Inference with Synthetic Controls: A Proximal Causal Inference Framework."
    arXiv preprint arXiv:2108.13935, 2023. URL: https://arxiv.org/abs/2108.13935.

    Jizhou Liu, Eric J. Tchetgen Tchetgen, and Carlos Varjão.
    "Proximal Causal Inference for Synthetic Control with Surrogates."
    arXiv preprint arXiv:2308.09527, 2023. URL: https://arxiv.org/abs/2308.09527.
    """

    def __init__(self, config):
        # Load configuration
        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", ["grey", "red", "blue"])
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)

        
        self.surrogates = config.get("surrogates", [])
        self.donors = config.get("donors", [])
        self.vars = config.get("vars", [])

    def fit(self):
        # Ensure the required lists are not empty
        if not self.donors:
            raise ValueError("List of donors cannot be empty.")

        # Placeholder for the balance method implementation
        balance(self.df, self.unitid, self.time)

        prepped = dataprep(
            self.df, self.unitid, self.time, self.outcome, self.treat
        )

        # Filter self.donors to include only valid columns in Ywide
        valid_donors = [donor for donor in self.donors if donor in  prepped['Ywide'].columns]

        # Extract only the valid columns
        W =  prepped['Ywide'][valid_donors].to_numpy()

        donorprox = self.df.pivot(index=self.time, columns=self.unitid, values=self.vars["donorproxies"][0])

        Z0 = donorprox[valid_donors].to_numpy()

        X, Z1 = proxy_dataprep(self.df, surrogate_units=self.surrogates, proxy_vars=self.vars, id_col=self.unitid, time_col=self.time, T=prepped["total_periods"])

        h = int(np.floor(4 * (prepped["post_periods"] / 100) ** (2 / 9)))

        y_PI, alpha, se_tau = pi(prepped["y"], W, Z0, prepped["pre_periods"], prepped["post_periods"], prepped["total_periods"], h)

        PIattdict, PIfitdict, PIVectors = effects.calculate(prepped["y"], y_PI, prepped["pre_periods"],
                                                         prepped["post_periods"])

        PIdict = {"Effects": PIattdict, "Fit": PIfitdict, "Vectors": PIVectors}


        if len(self.surrogates) > 1:
            # Surrogate-based estimations
            tau, taut, alpha, se_tau = pi_surrogate(prepped["y"], W, Z0, Z1,
                                                    clean_surrogates2(X, Z0, W, prepped["pre_periods"]),
                                                    prepped["pre_periods"], prepped["post_periods"],
                                                    prepped["total_periods"], h)
            y_PIS = prepped["y"] - taut
            PISattdict, PISfitdict, PISVectors = effects.calculate(prepped["y"], y_PIS, prepped["pre_periods"],
                                                                   prepped["post_periods"])
            PISdict = {"Effects": PISattdict, "Fit": PISfitdict, "Vectors": PISVectors}

            # Post-surrogate estimators
            tau, taut, alpha, se_tau = pi_surrogate_post(prepped["y"], W, Z0, Z1,
                                                         clean_surrogates2(X, Z0, W, prepped["pre_periods"]),
                                                         prepped["pre_periods"], prepped["post_periods"],
                                                         prepped["total_periods"], h)
            y_PIPost = prepped["y"] - taut
            PIPostattdict, PIPostfitdict, PIPostVectors = effects.calculate(prepped["y"], y_PIPost,
                                                                            prepped["pre_periods"],
                                                                            prepped["post_periods"])
            PIPostdict = {"Effects": PIPostattdict, "Fit": PIPostfitdict, "Vectors": PIPostVectors}

            # Add surrogate results to dictionary
            ProximalDict = {"PI": PIdict, "PIS": PISdict, "PIPost": PIPostdict}

            # Plotting: Include PI, PI-S, PI-P
            counterfactuals = [y_PI, y_PIS, y_PIPost]
            counterfactual_names = ["Proximal Inference", "Proximal Surrogates", "Proximal Post"]
        else:
            # If no surrogates, return only PI results
            ProximalDict = {"PI": PIdict}

            # Plotting: Only PI
            counterfactuals = [y_PI]
            counterfactual_names = ["Proximal Inference"]

        if self.display_graphs:
            plot_estimates(
                prepped,
                self.time,
                self.unitid,
                self.outcome,
                self.treat,
                prepped["treated_unit_name"],
                prepped["y"],
                counterfactuals,
                method="PI",
                counterfactual_names=counterfactual_names,
                treatedcolor=self.treated_color,
                save=self.save,
                counterfactualcolors=self.counterfactual_color,
            )


        return ProximalDict


class FSCM:
    def __init__(self, config):
        """
        This function provides ATT estimates using the Forward Selected Synthetic Control Method (FSCM).

        This approach optimally selects a subset of donor units using forward selection, beginning with the best-fitting single donor and adding additional units only if they improve predictive fit. The final weights and counterfactual are derived using a constrained optimization procedure (SIMPLEX) over the selected donor pool.

        Parameters
        ----------
        config : dict
            A dictionary containing the necessary parameters. The following keys are expected:

            df : pandas.DataFrame
                Input dataset. Must contain at least four columns: one identifying the unit, one for time, one for the outcome, and one indicating treatment status (binary).

            treat : str
                Column name indicating the treated unit (treated unit is the one with a 1 in the treatment column).

            time : str
                Column name for the time variable.

            outcome : str
                Column name for the numeric outcome variable.

            unitid : str
                Column name identifying unit labels.

            counterfactual_color : str or list of str, optional
                Color used in the counterfactual plot line. If multiple methods are plotted, a list of colors can be supplied. Default is "red".

            treated_color : str, optional
                Color used for the treated unit plot line. Default is "black".

            display_graphs : bool, optional
                Whether to display the ATT graph. Default is True.

            save : bool or dict, optional
                If True, saves the plot to the current directory.
                If a dictionary, keys may include:
                    - 'filename': Custom name for the file (without extension)
                    - 'extension': Format like 'png', 'pdf'
                    - 'directory': Directory path to save the file

        Returns
        -------
        dict
            A dictionary containing the following keys:

            'Effects' : dict
                Contains average treatment effect estimates over time, including pre- and post-treatment periods.

            'Fit' : dict
                Goodness-of-fit metrics over the pre-treatment period.

            'Vectors' : dict
                Observed outcomes, counterfactual estimates, and treatment effects (difference between treated and synthetic) over time.

            'Weights' : list
                A list of two elements:
                    - A dictionary mapping selected donor unit names to their corresponding weights (rounded to 3 decimal places).
                    - A dictionary with summary metrics:
                        * 'Cardinality of Positive Donors': Number of selected donor units with non-negligible weights (> 0.001).
                        * 'Cardinality of Selected Donor Pool': Total number of donors considered in the final model.

            '_prepped' : dict
                Internal dictionary with intermediate data used to compute the estimate (e.g., treated vector, donor matrix, donor names).

        References
        ----------
        Cerulli, Giovanni. "Optimal initial donor selection for the synthetic control method."
        *Economics Letters*, 244 (2024): 111976. https://doi.org/10.1016/j.econlet.2024.111976
        """

        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)

    def evaluate_donor(self, donor_index, donor_columns, y_pre, T0):
        """Evaluate the MSE for a given donor index using the SCM optimization."""
        donor = donor_columns[donor_index]
        prob = Opt.SCopt(1, y_pre, T0, donor, model="SIMPLEX")
        mse = prob.solution.opt_val
        return donor_index, mse

    def fSCM(self, y_pre, Y0, T0):
        """Returns the optimal donor indices, their corresponding weights, and optimal RMSE using Synthetic Control Method (SCM)."""
        best_mse = float("inf")
        best_set = None
        donor_columns = [Y0[:, i].reshape(-1, 1) for i in range(Y0.shape[1])]  # Precompute donor columns

        # Find the best single donor in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            donor_mse_results = list(
                executor.map(lambda i: self.evaluate_donor(i, donor_columns, y_pre, T0), range(Y0.shape[1])))

        donor_mse_results.sort(key=lambda x: x[1])  # Sort by MSE
        best_set = [donor_mse_results[0][0]]  # Store the best donor index
        best_mse = donor_mse_results[0][1]

        # Greedy donor addition
        all_donors = set(range(Y0.shape[1]))
        remaining_donors = all_donors - set(best_set)

        best_combination = best_set
        best_combination_mse = best_mse

        for j in remaining_donors:
            current_set = best_combination + [j]
            Y0_subset = Y0[:, current_set]

            prob = Opt.SCopt(len(current_set), y_pre, T0, Y0_subset, model="SIMPLEX")
            mse = np.sqrt((prob.solution.opt_val ** 2) / T0)

            if mse < best_combination_mse:
                best_combination_mse = mse
                best_combination = current_set

        optimal_indices = best_combination
        optimal_Y0 = Y0[:, optimal_indices]
        prob = Opt.SCopt(len(optimal_indices), y_pre, T0, optimal_Y0, model="SIMPLEX")

        first_key = list(prob.solution.primal_vars.keys())[0]
        weights = prob.solution.primal_vars[first_key]

        optimal_rmse = best_combination_mse

        return optimal_indices, weights, optimal_rmse, optimal_Y0, y_pre

    def fit(self):
        balance(self.df, self.unitid, self.time)

        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treat)

        Y0 = prepped["donor_matrix"][:prepped["pre_periods"]]  # Pre-period donor matrix
        y_pre = prepped["y"][:prepped["pre_periods"]]  # Pre-period treated unit
        T0 = prepped["pre_periods"]

        optimal_indices, rounded_weights, optimal_rmse, optimal_Y0, y_pre = self.fSCM(y_pre, Y0, T0)

        Y0_selected = prepped["donor_matrix"][:, optimal_indices]

        counterfactual = np.dot(Y0_selected, rounded_weights)
        # Extract donor names from prepped["donor_names"]
        donor_names = prepped["donor_names"]

        # Select the donor names corresponding to the optimal indices
        selected_donor_names = [donor_names[i] for i in optimal_indices]

        # Extract the corresponding weights for the optimal indices
        donor_weights =  {selected_donor_names[i]: round(rounded_weights[i], 3) for i in range(len(optimal_indices))}


        attdict, fitdict, Vectors = effects.calculate(prepped["y"], counterfactual, prepped["pre_periods"],
                                                       prepped["post_periods"])



        # Call the function
        if self.display_graphs:
            plot_estimates(
                df=prepped,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=[counterfactual],
                counterfactual_names=[f"FSC {prepped['treated_unit_name']}"],
                method="FSC",
                treatedcolor="black",
                counterfactualcolors=self.counterfactual_color,
                save=self.save
            )

        return {
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors,
            "Weights": [
                donor_weights,
                {
                    "Cardinality of Positive Donors": np.sum(np.abs(rounded_weights) > 0.001),
                    "Cardinality of Selected Donor Pool": np.shape(optimal_Y0)[1]
                }
            ],
            "_prepped": prepped
        }


class SRC:
    def __init__(self, config):
        """
        Implements the Synthetic Regressing Control (SRC) method for estimating treatment effects.

        This method, introduced by Zhu (2023), estimates a counterfactual trajectory for a treated unit by regressing its outcome on a weighted combination of donor units in the post-treatment period, with adjustments to ensure predictive fit over the pre-period.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration for the estimator. The following keys are expected:

            df : pandas.DataFrame
                Input dataset. Must include columns for unit, time, outcome, and treatment indicator.

            outcome : str
                Column name of the outcome variable.

            treat : str
                Column name of the binary treatment indicator (1 if treated, 0 otherwise).

            unitid : str
                Column name for unit identifiers.

            time : str
                Column name for the time variable.

            counterfactual_color : str or list of str, optional
                Color for the counterfactual trajectory line in the output graph. Can be a single color or a list if plotting multiple counterfactuals. Default is "red".

            treated_color : str, optional
                Color for the treated unit’s trajectory line in the graph. Default is "black".

            display_graphs : bool, optional
                If True, plots the estimated counterfactual against the observed treated trajectory. Default is True.

            save : bool or dict, optional
                If True, saves the generated plot using default settings.
                If a dictionary, may include:
                    - 'filename': Custom filename (without extension)
                    - 'extension': Format like 'png', 'pdf'
                    - 'directory': Directory path to save the plot

        Returns
        -------
        dict
            Dictionary with the following keys:

            "Counterfactual" : np.ndarray
                Estimated counterfactual trajectory for the treated unit.

            "Weights" : dict
                Dictionary mapping donor unit names to their estimated weights.

            "ATT" : dict
                Average treatment effect estimates for the post-treatment period.

            "Fit" : dict
                Goodness-of-fit metrics computed over the pre-treatment period.

            "Vectors" : dict
                Contains:
                    - "Treated": The observed outcome for the treated unit.
                    - "Counterfactual": The estimated counterfactual.
                    - "Effect": The estimated treatment effect vector.

        References
        ----------
        Zhu, Rong J. B. "Synthetic Regressing Control Method." arXiv preprint arXiv:2306.02584 (2023).
        https://arxiv.org/abs/2306.02584
        """

        # Assigning the configuration dictionary parameters
        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)

    def fit(self):
        """This method prepares the data and runs the SRC estimation process."""

        # Pre-process the data using dataprep function
        prepped = dataprep(self.df,
                           self.unitid,
                           self.time,
                           self.outcome,
                           self.treat)

        y1 = prepped["y"]
        Y0 = prepped["donor_matrix"]

        y_SRC, w_hat, theta_hat = SRCest(y1, Y0, prepped["post_periods"])

        donor_weights = {prepped["donor_names"][i]: round(w_hat[i], 3) for i in range(len(w_hat))}

        # Calculate the ATT and other effects using the `effects` module (ensure it's defined)
        attdict, fitdict, Vectors = effects.calculate(prepped["y"], y_SRC, prepped["pre_periods"],
                                                       prepped["post_periods"])

        # Call the function to plot the estimates
        if self.display_graphs:
            plot_estimates(
                df=prepped,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=[y_SRC],
                counterfactual_names=[f"SRC {prepped['treated_unit_name']}"],
                method="SRC",
                treatedcolor=self.treated_color,
                counterfactualcolors=self.counterfactual_color,
                save=self.save
            )

        # Return the results as a dictionary
        return {
            "Counterfactual": y_SRC,
            "Weights": donor_weights,
            "ATT": attdict,
            "Fit": fitdict,
            "Vectors": Vectors,
        }


class SCMO:
    """
    SCMO: Synthetic Control with Multiple Outcomes

    Implements synthetic control estimators for settings with one treated unit and multiple
    auxiliary outcomes. Supports two methods: TLP (Tian, Lee, and Panchenko) and SBMF (Sun et al.),
    as well as model averaging between the two. Optional conformal prediction intervals are available
    for treatment effect inference.

    Parameters
    ----------
    config : dict
        Configuration dictionary specifying the data, model, and visualization behavior.

        Required keys:
            - df : pandas.DataFrame
                Long-form panel dataset. Each row represents a unit-time observation.

            - outcome : str
                Name of the main outcome variable to be used for treatment effect estimation.

            - treat : str
                Name of the treatment indicator column. Should be 1 for the treated unit post-intervention, 0 otherwise.

            - unitid : str
                Name of the column identifying the unit (e.g., city, region).

            - time : str
                Name of the time variable column (e.g., week, year).

        Optional keys:
            - addout : str or list of str, default = []
                One or more auxiliary outcome variables to be included in outcome stacking.

            - method : str, default = 'TLP'
                Estimation method to use. One of:
                    'TLP' — Two-layer projection estimator (Tian et al.)
                    'SBMF' — Sparse balancing matrix factorization (Sun et al.)
                    'both' — Model averaging between TLP and SBMF

            - display_graphs : bool, default = True
                If True, displays a plot of the treated unit and its synthetic counterfactual over time.

            - save : bool or dict, default = False
                If True, saves the plot with default settings (PNG, working directory).
                If a dict, the following keys are supported:
                    - 'filename': str, custom name for the plot (without extension)
                    - 'extension': str, format to save in (e.g., 'png', 'pdf')
                    - 'directory': str, path to the directory to save the file

            - counterfactual_color : str, default = 'red'
                Color for the synthetic control trajectory in plots.

            - treated_color : str, default = 'black'
                Color for the treated unit trajectory in plots.

    Returns
    -------
    results : dict
        Dictionary of estimation results. Structure depends on the value of `method`.

        If method is 'TLP' or 'SBMF', the dictionary has the following keys:

            - 'Weights': numpy.ndarray
                Donor weights estimated for the treated unit.

            - 'Effects': dict
                Treatment effect statistics:
                    - 'ATT': Average treatment effect on the treated (scalar)
                    - 'Percent ATT': ATT as a percentage of counterfactual mean
                    - 'SATT': Standardized ATT, normalized by estimated variance
                    - 'TTE': Total treatment effect (sum of post-period differences)

            - 'Fit': dict
                Pre- and post-treatment fit diagnostics:
                    - 'T0 RMSE': Root mean squared error for pre-treatment periods
                    - 'T1 RMSE': Standard deviation of treatment effect in post-treatment periods
                    - 'R-Squared': Fit quality over pre-treatment window
                    - 'Pre-Periods': Number of pre-treatment periods
                    - 'Post-Periods': Number of post-treatment periods

            - 'Vectors': dict
                Time series vectors (all numpy arrays):
                    - 'Observed Unit': Observed outcome of treated unit
                    - 'Counterfactual': Synthetic control prediction for treated unit
                    - 'Gap': 2D array where first column is observed - counterfactual,
                      and second column is time relative to intervention

            - 'Conformal Prediction': dict
                Prediction intervals using agnostic conformal inference. Keys are:
                    - 'Lower Bound': Lower bound of pointwise prediction interval
                    - 'Upper Bound': Upper bound of pointwise prediction interval

        If method is 'both', the dictionary contains:

            - 'Weights': numpy.ndarray
                Donor weights obtained by optimally averaging the TLP and SBMF predictions.

            - 'Lambdas': dict
                Averaging weights assigned to each method:
                    - 'TLP': Weight placed on TLP model
                    - 'SBMF': Weight placed on SBMF model

            - 'Effects': dict
                Treatment effect statistics computed from the model-averaged counterfactual.

            - 'Fit': dict
                Fit diagnostics for the model-averaged counterfactual.

            - 'Vectors': dict
                Time series vectors based on the averaged counterfactual:
                    - 'Observed Unit', 'Counterfactual', and 'Gap' as above.

            - 'Conformal Prediction': dict
                Prediction intervals applied to the averaged counterfactual.

        Per-model effects, fits, or trajectories are not returned when using model averaging.
    """


    def __init__(self, config):
        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.treat = config.get("treat")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")
        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)
        self.addout = config.get("addout", [])
        self.method = config.get("method", "tlp").upper()

        assert self.method in ["TLP", "SBMF", "BOTH"], "Method must be 'TLP', 'SBMF', or 'both'"

    def fit(self):
        """Prepares data and fits the synthetic control weights for one or both estimators."""

        # Handle addout gracefully
        if isinstance(self.addout, str):
            outcome_list = [self.outcome, self.addout]
        elif isinstance(self.addout, list):
            outcome_list = [self.outcome] + self.addout
        else:
            outcome_list = [self.outcome]

        # Prep all outcomes
        results = [
            dataprep(self.df, self.unitid, self.time, outcome, self.treat)
            for outcome in outcome_list
        ]


        T0 = results[0]['pre_periods']
        post = results[0]['post_periods']

        # Always prep primary outcome for final plotting/effects
        base = dataprep(self.df, self.unitid, self.time, self.outcome, self.treat)
        Y0, y = base["donor_matrix"], base["y"]

        estimators = {}

        for method in (["TLP", "SBMF"] if self.method == "BOTH" else [self.method]):
            if method == "TLP":
                y_stack = np.concatenate([prenorm(r["y"][:T0]) for r in results], axis=0)
                Y0_stack = np.concatenate([prenorm(r["donor_matrix"][:T0]) for r in results], axis=0)

            elif method == "SBMF":
                # Use raw data, no need to demean or average across outcomes
                y_stack = np.concatenate([r["y"][:T0] for r in results], axis=0)
                Y0_stack = np.concatenate([r["donor_matrix"][:T0] for r in results], axis=0)
            else:
                raise ValueError("Unknown method.")

            assert np.all(np.isfinite(y_stack)), "y_stack contains non-numeric values (NaN or inf)"
            assert np.all(np.isfinite(Y0_stack)), "Y0_stack contains non-numeric values (NaN or inf)"

            # Solve SCM with MSCa for SBMF, SIMPLEX otherwise
            model_type = "MSCa" if method == "SBMF" else "SIMPLEX"
            prob = Opt.SCopt(
                Y0_stack.shape[1], y_stack, T0 * len(outcome_list), Y0_stack, model=model_type
            )
            first_key = list(prob.solution.primal_vars.keys())[0]
            weights = prob.solution.primal_vars[first_key]
            # Assign donor weights dictionary
            if method == "SBMF":
                donor_weights = {
                    base["donor_names"][i]: round(weights[i + 1], 3)
                    for i in range(len(base["donor_names"]))
                }
            else:
                donor_weights = {
                    base["donor_names"][i]: round(weights[i], 3)
                    for i in range(len(base["donor_names"]))
                }

            # Construct counterfactual for main outcome
            if method == "SBMF":
                # No need to reapply demeaning; just use the original counterfactual computation
                y_SCMO = np.dot(Y0, weights[1:]) + weights[0]  # Use weights for SBMF
            else:
                y_SCMO = np.dot(Y0, weights)  # Use weights for SIMPLEX (same calculation here)

            # Compute effects and store
            attdict, fitdict, Vectors = effects.calculate(y, y_SCMO, T0, post)

            # Compute prediction intervals using agnostic conformal inference
            alpha = 0.1  # Set alpha for confidence level (e.g., 90% intervals)
            lower, upper = ag_conformal(y[:T0], y_SCMO[:T0], y_SCMO[T0:], alpha=alpha)

            # Create a 2-column matrix for the prediction intervals (lower, upper)
            prediction_intervals_matrix = np.vstack([lower, upper]).T

            # Add results to estimators with Prediction Intervals in the Vectors sub-dictionary
            estimators[method] = {
                "weights": donor_weights,
                "Effects": attdict,
                "Fit": fitdict,
                "Vectors": {
                    **Vectors,
                    "Agnostic Prediction Intervals": prediction_intervals_matrix,  # Add the matrix
                },
            }

        if len(estimators) > 1:
            # Proceed with model averaging
            cf_vectors = {}
            for method_name, info in estimators.items():
                cf = info["Vectors"]["Counterfactual"]
                w = np.array(list(info["weights"].values()))

                if method_name == "SBMF":
                    # SBMF: intercept is included in weights[0], prepend it
                    w_full = np.insert(w, 0, weights[0])  # Already handled above
                elif method_name == "TLP":
                    # TLP: no intercept, add dummy 0 at beginning
                    w_full = np.insert(w, 0, 0.0)
                else:
                    raise ValueError(f"Unknown method '{method_name}' for model averaging.")

                cf_vectors[method_name] = {
                    "cf": cf,
                    "weights": w_full
                }
            MAres = Opt.SCopt(len(estimators), y[:T0], T0, Y0, model="MA",  model_weights=cf_vectors, donor_names=base["donor_names"])
        else:
            print("Only one estimator — skipping model averaging.")

        treated_unit_name = base["treated_unit_name"]

        if self.method != "BOTH":
            cfvecs = [est["Vectors"]["Counterfactual"] for est in estimators.values()]
            cflist = [f"{key} {treated_unit_name}" for key in estimators.keys()]
        elif self.method == "BOTH":
            cfvecs = [MAres['Counterfactual']]
            # Compute prediction intervals using agnostic conformal inference
            alpha = 0.1  # Set alpha for confidence level (e.g., 90% intervals)
            lower, upper = ag_conformal(y[:T0], MAres['Counterfactual'][:T0], MAres['Counterfactual'][T0:], alpha=alpha)

            # Create a 2-column matrix for the prediction intervals (lower, upper)
            prediction_intervals_matrix = np.vstack([lower, upper]).T
            cflist = [f"MA {treated_unit_name}"]
            MAattdict, MAfitdict, MAVectors = effects.calculate(y, MAres['Counterfactual'], T0, post)
            ma_weights = {
                base["donor_names"][i]: round(MAres["w_MA"][i], 3)
                for i in range(len(base["donor_names"]))
            }

            estimators = {"Effects": MAattdict,
                          "Fit": MAfitdict,
                          "Vectors": MAVectors,
                          "Conformal Prediction": prediction_intervals_matrix,
                          "Weights": [ma_weights, {k: v for k, v in ma_weights.items() if v > 0}],
                          "Lambdas": MAres["Lambdas"]}

        # Call the function to plot the estimates
        if self.display_graphs:
            plot_estimates(
                df=base,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=base["treated_unit_name"],
                y=base["y"],
                cf_list=cfvecs,
                counterfactual_names=cflist,
                method="SRC",
                treatedcolor=self.treated_color,
                counterfactualcolors=self.counterfactual_color,
                save=self.save,
                uncvectors=prediction_intervals_matrix
            )
        return estimators


class SI:
    """
    SI: Synthetic Interventions

    Estimates counterfactual outcomes under alternative treatments using Principal Component Regression (PCR).
    For each treatment in `inters`, it computes the counterfactual outcomes for a focal treated unit, as if
    that unit had received the alternative treatment instead.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

        Required keys:
            - df : pandas.DataFrame
                Long-form panel dataset.
            - outcome : str
                Name of the outcome variable.
            - unitid : str
                Column identifying units (e.g., "state", "region").
            - time : str
                Column identifying time periods.
            - inters : list of str
                List of binary treatment indicator columns used to define donor groups.
            - treat : str
                Name of the treatment indicator column for the focal unit.

        Optional keys:
            - display_graphs : bool, default=True
                Whether to show the outcome and counterfactual trajectory.
            - save : bool or dict, default=False
                Whether to save the plot and how to configure it.
            - counterfactual_color : str, default="red"
                Color for counterfactual lines in the plot.
            - treated_color : str, default="black"
                Color for the treated unit's observed line.
    """

    def __init__(self, config):
        self.df = config.get("df")
        self.outcome = config.get("outcome")
        self.unitid = config.get("unitid")
        self.time = config.get("time")
        self.inters = config.get("inters")
        self.treat = config.get("treat")

        # Validate inputs
        assert self.inters and isinstance(self.inters, list), "'inters' must be a non-empty list."
        for inter in self.inters:
            assert inter in self.df.columns, f"Intervention column '{inter}' not found in dataframe."

        self.display_graphs = config.get("display_graphs", True)
        self.save = config.get("save", False)
        self.counterfactual_color = config.get("counterfactual_color", "red")
        self.treated_color = config.get("treated_color", "black")

    def fit(self):
        # Ensure panel is balanced
        balance(self.df, self.unitid, self.time)

        # Prepare the data for analysis... reshaping
        
        prepped = dataprep(self.df, self.unitid, self.time, self.outcome, self.treat)

        # Dictionary: {intervention name → set of units assigned to that intervention}
        intervention_sets = {
            col: set(self.df.loc[self.df[col] == 1, self.unitid])
            for col in self.inters
        }

        counterfactuals = []
        counterfactual_names = []
        SIresults = {}  # Store results by intervention name

        for alt_treat, donor_units in intervention_sets.items():
            # Select only donor columns present in Ywide
            donor_cols = prepped['Ywide'].columns.intersection(donor_units)
            Y_donor = prepped['Ywide'][donor_cols].to_numpy()

            # Estimate PCR weights and counterfactual
            result = pcr(
                Y_donor,
                prepped["y"],
                "OLS", # Maybe have OLS be an option?
                donor_cols,
                Y_donor,
                pre=prepped["pre_periods"], # Number of pre periods is set by the analyst
                cluster=False,
                Frequentist=True
            )

            cf = result[list(result.keys())[-1]]  # Extract counterfactual vector

            weight_dict = dict(zip(donor_cols, result[list(result.keys())[0]]))  # Map weights to donor unit names

            # Compute ATT, fit diagnostics, and relevant vectors
            attdict, fitdict, vectors = effects.calculate(
                prepped["y"],
                cf,
                prepped["pre_periods"],
                prepped["post_periods"],
            )

            # Store results for this alternative treatment
            SIresults[alt_treat] = {
                "Effects": attdict,
                "Fit": fitdict,
                "Vectors": vectors, "Weights": weight_dict
            }

            counterfactuals.append(cf)
            counterfactual_names.append(f"{alt_treat} {prepped['treated_unit_name']}")

        # If you wish to plot...
        if self.display_graphs:
            plot_estimates(
                df=prepped,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname="Synthetic Treatments Start",
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=counterfactuals,
                counterfactual_names=counterfactual_names,
                method="SI",
                treatedcolor=self.treated_color,
                counterfactualcolors=self.counterfactual_color,
                save=self.save
            )

        return SIresults
