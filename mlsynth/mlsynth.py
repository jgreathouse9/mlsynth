import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import inv
from scipy.stats import norm
import scipy.stats as stats
from sklearn.cluster import KMeans
from screenot.ScreeNOT import adaptiveHardThresholding
from mlsynth.utils.datautils import prepare_data, balance, dataprep
from mlsynth.utils.resultutils import effects, plot_estimates
from mlsynth.utils.estutils import Opt, pcr, TSEST, pda
from mlsynth.utils.inferutils import step2
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

        TSSC computes the two step SCM as described in:

        https://doi.org/10.1287/mnsc.2023.4878

        Parameters
        ----------
        config : dict
            A dictionary containing the necessary parameters. The following keys are expected:
            df : pandas.DataFrame
                User-specified dataframe containing the data.
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
            save : bool, optional
                Whether to save the generated plots, by default False.
            draws : int, optional
                Number of subsample replications, by default 500.

        Returns
        -------
        dict
            A dictionary with the following keys:
            'SIMPLEX' : float
                Estimates and inference from the SIMPLEX method.
            'MSCa' : float
                Estimates and inference from the MSCa method.
            'MSCb' : float
                Estimates and inference from the MSCb method.
            'MSCc' : float
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

        nb =self.draws
        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treat)

        x = np.concatenate((np.ones((prepped['total_periods'], 1)), prepped['donor_matrix']), axis=1)

        result = TSEST(prepped['donor_matrix'], prepped['y'], prepped['pre_periods'], nb, prepped['donor_names'],prepped['post_periods'])

        n = x.shape[1]

        b_MSC_c = next((method_dict["MSCc"]["WeightV"] for method_dict in result if "MSCc" in method_dict), None)
        b_SC = next((method_dict["SIMPLEX"]["WeightV"] for method_dict in result if "SIMPLEX" in method_dict), None)
        b_MSC_a = next((method_dict["MSCa"]["WeightV"] for method_dict in result if "MSCa" in method_dict), None)
        b_MSC_b = next((method_dict["MSCb"]["WeightV"] for method_dict in result if "MSCb" in method_dict), None)

        recommended_model = step2(np.array([[0] + list(np.ones(n - 1))]),
                                  np.array([[1] + list(np.zeros(n - 1))]),
                                  np.array([[0] + list(np.ones(n - 1)),
                                            [1] + list(np.zeros(n - 1))]),
                                  b_MSC_c, 1, 0, np.array([[1], [0]]), prepped['pre_periods'], x[:prepped['pre_periods'], :], prepped['y'][:prepped['pre_periods']],
                                  nb, x.shape[1], np.zeros((n, nb)))

        recommended_variable = next(
            (method_dict[recommended_model]["Vectors"]["Counterfactual"] for method_dict in result if
             recommended_model in method_dict), None)
        ATT, RMSE = next(
            ((method_dict[recommended_model]["Effects"]["ATT"], method_dict[recommended_model]["Fit"]["T0 RMSE"]) for
             method_dict in result if recommended_model in method_dict), (None, None))
        
        if self.display_graphs:
        
            plot_estimates(
                df=self.df,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=[recommended_variable],
                counterfactual_names=[recommended_model],
                method=f'{recommended_model}',
                treatedcolor=self.treated_color,
                counterfactualcolors=[self.counterfactual_color]
            )

        return result

class FMA:
    def __init__(self, config):
        """
        Compute estimates and inference using the Factor Model Approach (FMA).

        This function implements the Factor Model Approach described in
        Li and Sonnier (2023) and returns a dictionary containing the estimated
        effects, model fit, factor vectors, and inference results.

        Parameters
        ----------
        config : dict
            A dictionary containing the necessary parameters. The following keys are expected:
            
            df : pandas.DataFrame
                DataFrame containing the data.
                
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
                    - 'display' : If False, prevents the plot from being displayed.
            
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
        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treat)
        t = prepped["total_periods"]
        t1 = prepped["pre_periods"]
        
        t2 = prepped["post_periods"]
        datax = prepped["donor_matrix"]
        y1 = prepped["y"][:prepped["pre_periods"]]
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
        X = demean_matrix(datax) # For Xu's method
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
                    inv(np.dot(Fhat_1_xu.T, Fhat_1_xu)).dot(Fhat_1_xu.T).dot(y1_xu)
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
        attdict, fitdict, Vectors = effects.calculate(prepped["y"], y_hat, prepped["pre_periods"], prepped["post_periods"])
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
        cr005, cr995 = calculate_cr(0.005, ATT, AA), calculate_cr(0.995, ATT, AA)
        cr025, cr975 = calculate_cr(0.025, ATT, AA), calculate_cr(0.975, ATT, AA)
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
            "p_value": p_value
        }
        
        plot_estimates(self.df, self.time, self.unitid, self.outcome, self.treat,
                       prepped["treated_unit_name"], prepped["y"], [y_hat], method="FMA",
                       treatedcolor=self.treated_color, 
                       counterfactualcolors=["red"],
                       counterfactual_names=[f"FMA {prepped['treated_unit_name']}"], save=self.save)
        
        
        return {"Effects": attdict, "Fit": fitdict, "Vectors": Vectors, "Inference": Inference}


class PDA:
    def __init__(self, config):
        """
        Causal inference via the Panel Data Approach (PDA).

        This class implements PDA as described in
        Shi and Huang (2023), Hsiao et al. (2012), and Shi and Wang (2024).
        It supports different methods including LASSO, forward selection (fs),
        and L2-relaxation.

        Parameters
        ----------
        config : dict
            A dictionary containing the necessary parameters. The following keys are expected:
            df : pandas.DataFrame
                DataFrame containing the data.
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
                    - 'display' : If False, prevents the plot from being displayed.
            method : str, optional
                Type of PDA to use, either "LASSO", "fs" (forward selection), or "l2" (L2-relaxation),
                by default "fs" (forward selection).

        Returns
        -------
        dict
            A dictionary containing the estimated treatment effects, fit metrics,
            and inference results depending on the selected PDA method.

        References
        ----------
        Shi, Z. & Huang, J. (2023). "Forward-selected panel data approach for program evaluation."
        *Journal of Econometrics*, Volume 234, Issue 2, Pages 512-535.

        Hsiao, C., Ching, H. S., & Wan, S. K. (2012). "A Panel Data Approach for Program Evaluation:
        Measuring the Benefits of Political and Economic Integration of Hong Kong with Mainland China."
        *J. Appl. Econ.*, 27:705-740.

        Shi, Z. & Wang, Y. (2024). "L2-relaxation for Economic Prediction."
        DOI: 10.13140/RG.2.2.11670.97609.
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

    def fit(self):
        # Preprocess the data
        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treat)

        pdaest = pda(prepped,  len(prepped["donor_names"]), method=self.method)
        attdict, fitdict, Vectors = effects.calculate(prepped["y"], pdaest['Vectors']['Counterfactual'], prepped["pre_periods"],
                                                         prepped["post_periods"])
        est_method = pdaest.get("method")

        counterfactual_name = f'{est_method} {prepped["treated_unit_name"]}'

        # Call the function
        plot_estimates(
            df=self.df,
            time=self.time,
            unitid=self.unitid,
            outcome=self.outcome,
            treatmentname=self.treat,
            treated_unit_name=prepped["treated_unit_name"],
            y=prepped["y"],
            cf_list=[pdaest['Vectors']['Counterfactual']],
            counterfactual_names=[counterfactual_name],
            method=est_method,
            treatedcolor=self.treated_color,
            counterfactualcolors=[self.counterfactual_color],
            save=self.save
        )

        return pdaest


class FDID:
    def __init__(self, config):
        """
        Compute Forward DID, Augmented DID, and standard DID estimates.

        This function implements the Forward Difference-in-Differences (FDID) method,
        the Augmented Difference-in-Differences (ADID) method, and the standard Difference-in-Differences (DID) method,
        as described in Li (2023) and Li & Van den Bulte (2023).
        It returns estimates for all three methods, including effects, model fit, and inference results.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration options. The following keys are expected:
            
            df : pandas.DataFrame
                Dataframe containing the data.
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
                    - 'display' : If False, prevents the plot from being displayed.

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
        self.save= config.get("save", False)


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

        #print(f'({t2} / {t1}) * {np.mean(u1_DID ** 2)}')
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

        #print(z_critical * std_Omega_hat_DID / np.sqrt(t2))

        CI_95_DID_left = ATT_DID - z_critical * std_Omega_hat_DID / np.sqrt(t2)
        #print(f"{ATT_DID} - {z_critical} * {std_Omega_hat_DID} / np.sqrt({t2})")
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
            "SE": std_Omega_hat_DID,
            "Intercept":  np.round(b_DID, 3),
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
                # Combine previously selected controls with the current candidate
                combined_control = np.hstack((selected_controls[:t, :], datax[:, control_idx].reshape(-1, 1)))

                # Estimate DiD and compute R-squared for this combination
                ResultDict = self.DID(y_reshaped, combined_control, t1)
                R2[jj] = ResultDict["Fit"]["R-Squared"]

            # Update R-squared and the best control
            R2final[k] = np.max(R2)

            best_new_control = left[np.argmax(R2)]
            select_c[k] = best_new_control


            # Update the selected controls matrix with the new control
            selected_controls = np.hstack((selected_controls, datax[:, best_new_control].reshape(-1, 1)))
        # Get the index of the model with the highest R2
        best_model_idx = R2final.argmax()
        Uhat = datax[:, select_c[:best_model_idx + 1]]

        return select_c[:best_model_idx + 1], R2final, Uhat

    def fit(self):
        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treated)

        y1 = np.ravel(prepped["y"][:prepped["pre_periods"]])
        y2 = np.ravel(prepped["y"][-prepped["post_periods"]:])

        selected, R2_final, Uhat = self.selector(
            prepped["donor_matrix"].shape[1], prepped["pre_periods"], prepped["total_periods"], prepped["y"], prepped["donor_matrix"],  np.arange(1, prepped["donor_matrix"].shape[1] + 1)
        )

        selected_controls = prepped["donor_names"].take(selected.tolist()).tolist()

        FDID_dict, DID_dict, AUGDID_dict, y_FDID = self.est(
            Uhat, prepped["total_periods"], prepped["pre_periods"], prepped["post_periods"], prepped["y"], y1, y2, prepped["donor_matrix"]
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

        if self.display_graphs:

            plot_estimates(self.df, self.time, self.unitid, self.outcome, self.treated,
                           prepped["treated_unit_name"], prepped["y"], [y_FDID], method="FDID", counterfactual_names=["FDID "+prepped["treated_unit_name"]],
                           treatedcolor=self.treated_color, save=self.save, counterfactualcolors=[self.counterfactual_color])

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
                DataFrame containing the data.
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
        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treat)
        treatedunit = set(self.df.loc[self.df[self.treat] == 1, self.unitid])
        
        Ywide = prepped["Ywide"].T

        # Filter the row indices of Ywide using treatedunit
        row_indices_of_treatedunit = Ywide.index[Ywide.index == prepped["treated_unit_name"]]

        # Get the row index numbers corresponding to the treated units
        treatrow = [Ywide.index.get_loc(idx) for idx in row_indices_of_treatedunit][0]

        Z = np.zeros_like(Ywide.to_numpy())

        n, p = prepped["donor_matrix"][:prepped["pre_periods"]].shape
        k = (min(n, p) - 1) // 2

        Y0_rank, Topt_gauss, rank = adaptiveHardThresholding(prepped["donor_matrix"][:prepped["pre_periods"]], k, strategy='i')

        Z[treatrow, -prepped["post_periods"]:] = 1
        result = DC_PR_with_suggested_rank(Ywide.to_numpy(), Z, suggest_r=rank, method='non-convex')

        if self.display_graphs:
            plot_estimates(
                df=self.df,
                time=self.time,
                unitid=self.unitid,
                outcome=self.outcome,
                treatmentname=self.treat,
                treated_unit_name=prepped["treated_unit_name"],
                y=prepped["y"],
                cf_list=[result['Vectors']['Counterfactual']],
                counterfactual_names=["GSC"],
                method=r'$\ell_2$ relaxation',
                treatedcolor=self.treated_color,
                counterfactualcolors=[self.counterfactual_color]
            )
        

        return result


class CLUSTERSC:
    def __init__(self, config):
        """
        This function provides ATT estimates using two robust methods: Robust PCA Synthetic Control (RPCA SCM)
        and Principal Component Regression (PCR).

        Parameters
        ----------
        config : dict
            A dictionary containing the necessary parameters. The following keys are expected:
            df : pandas.DataFrame
                DataFrame containing the data.
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
                Objective function for PCR. Options are "SIMPLEX" and "OLS".
            counterfactual_color : str, optional
                Color for the counterfactual line in the plots. Default is "red".
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
                    - 'display' : If False, prevents the plot from being displayed.
                
        Returns
        -------
        dict
            A dictionary containing results for both RPCA-SC and PCR methods, with the following keys:
            'Weights' : dict
                Weights assigned to control units in the synthetic control model.
            'Effects' : dict
                Estimated treatment effects for the treated unit over time.
            'Vectors' : dict
                Observed, predicted, and treatment effects for both methods.

        References
        ----------
        Amjad, M., Shah, D., & Shen, D. (2018). "Robust synthetic control."
        *Journal of Machine Learning Research*, 19(22), 1-51.

        Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). "On Robustness of Principal Component Regression."
        *Journal of the American Statistical Association*, 116(536), 1731–45.

        Bayani, M. (2022). "Essays on Machine Learning Methods in Economics." Chapter 1.
        *CUNY Academic Works*.
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


    def fit(self):
        # Preprocess the data
        prepped = dataprep(self.df,
                           self.unitid, self.time,
                           self.outcome, self.treat)

        # Run PCR with the cluster parameter
        RSCweight, synth = pcr(
            prepped["donor_matrix"],
            prepped["y"],
            self.objective,
            prepped["donor_names"],
            prepped["donor_matrix"],
            pre = prepped["pre_periods"],
            cluster=self.cluster  # Use the cluster parameter
        )
        RSCweight = {key: round(value, 3) for key, value in RSCweight.items()}

        # Calculate effects
        attdict, fitdict, Vectors = effects.calculate(
            prepped["y"], synth, prepped["pre_periods"], prepped["post_periods"]
        )

        RSCdict = {"Effects": attdict, "Fit": fitdict, "Vectors": Vectors, "Weights": RSCweight}

        # Pivot the DataFrame to wide format
        trainframe = self.df.pivot_table(
            index=self.unitid, columns=self.time, values=self.outcome, sort=False
        )

        # Extract pre-treatment period data
        X = trainframe.iloc[:, :prepped["pre_periods"]]

        # Perform functional PCA and clustering
        optimal_clusters, cluster_x, numvals = fpca(X)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, init="k-means++", algorithm="elkan")
        trainframe["cluster"] = kmeans.fit_predict(cluster_x)

        # Identify treated unit's cluster and filter corresponding units
        treat_cluster = trainframe.at[prepped["treated_unit_name"], "cluster"]
        clustered_units = trainframe[trainframe["cluster"] == treat_cluster].drop("cluster", axis=1)

        # Extract treated unit row and control group matrix
        treated_row_idx = clustered_units.index.get_loc(prepped["treated_unit_name"])
        Y = clustered_units.to_numpy()
        y = Y[treated_row_idx]
        Y0 = np.delete(Y, treated_row_idx, axis=0)

        # Perform RPCA on the control group matrix
        L = RPCA(Y0)

        m, n = Y0.shape
        lambda_1 = 1 / np.sqrt(max(m, n))

        #L = RPCA_HQF(Y0, rank, maxiter=3000, ip=2, lam_1=lambda_1)

        # Optimize synthetic control weights using pre-period data
        beta_value = Opt.SCopt(
            len(Y0),
            prepped["y"][:prepped["pre_periods"]],
            prepped["pre_periods"],
            L[:, :prepped["pre_periods"]].T,
            model="MSCb"
        )

        # Calculate synthetic control predictions
        y_RPCA = np.dot(L.T, beta_value)
        beta_value = np.round(beta_value, 3)

        # Create a dictionary of non-zero weights
        unit_names = [name for name in clustered_units.index if name != prepped["treated_unit_name"]]
        Rweights_dict = {name: weight for name, weight in zip(unit_names, beta_value)}

        Rattdict, Rfitdict, RVectors = effects.calculate(prepped["y"], y_RPCA, prepped["pre_periods"], prepped["post_periods"])

        RPCAdict = {"Effects": Rattdict, "Fit": Rfitdict, "Vectors": RVectors, "Weights": Rweights_dict}

        # Call the function
        plot_estimates(
            df=self.df,
            time=self.time,
            unitid=self.unitid,
            outcome=self.outcome,
            treatmentname=self.treat,
            treated_unit_name=prepped["treated_unit_name"],
            y=prepped["y"],
            cf_list=[y_RPCA, synth],
            counterfactual_names=["RPCA Synth", "Robust Synthetic Control"],
            method="CLUSTERSC",
            treatedcolor="black",
            counterfactualcolors=["blue", "red"],
            save=self.save
        )

        ClustSCdict = {"RSC": RSCdict, "RPCASC": RPCAdict}

        return ClustSCdict

