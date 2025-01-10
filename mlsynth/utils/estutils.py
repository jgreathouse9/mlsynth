import numpy as np
import cvxpy as cp
from mlsynth.utils.denoiseutils import universal_rank, svt
from mlsynth.utils.resultutils import effects
from mlsynth.utils.selectorsutils import determine_optimal_clusters, SVDCluster, PDAfs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from functools import partial
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from screenot.ScreeNOT import adaptiveHardThresholding
from sklearn.linear_model import LassoCV
from scipy.stats import t as t_dist


def compute_hac_variance(treatment_effects, truncation_lag):
    """
    Compute the HAC long-run variance estimator with truncation lag h.

    Args:
        treatment_effects (np.ndarray): Array of estimated treatment effects, \(\hat{\Delta}_{t, \tau}\).
        truncation_lag (int): The truncation lag \(h\).

    Returns:
        float: HAC variance estimate.
    """
    T2 = len(treatment_effects)
    mean_effect = np.mean(treatment_effects)
    residuals = treatment_effects - mean_effect

    # Compute the HAC variance
    variance = np.var(residuals, ddof=1)  # Start with the sample variance
    for lag in range(1, truncation_lag + 1):
        weight = 1 - lag / (truncation_lag + 1)  # Bartlett kernel weight
        covariance = np.cov(residuals[:-lag], residuals[lag:], bias=True)[0, 1]
        variance += 2 * weight * covariance

    return variance


def compute_t_stat_and_ci(att, treatment_effects, truncation_lag, confidence_level=0.95):
    """
    Compute the t-statistic and confidence interval for ATT.

    Args:
        att (float): Average treatment effect on the treated (ATT).
        treatment_effects (np.ndarray): Array of estimated treatment effects, \(\hat{\Delta}_{t, \tau}\).
        truncation_lag (int): The truncation lag \(h\).
        confidence_level (float): Desired confidence level (default is 0.95).

    Returns:
        tuple: t-statistic, (lower CI bound, upper CI bound).
    """
    T2 = len(treatment_effects)
    hac_variance = compute_hac_variance(treatment_effects, truncation_lag)
    standard_error = np.sqrt(hac_variance / T2)

    # t-statistic
    t_stat = att / standard_error

    # 95% confidence interval
    z_low, z_high = zconfint(att, standard_error, alpha=1 - confidence_level)

    return t_stat, (z_low, z_high)


# Define the L2-relaxation estimator function
import cvxpy as cp
import numpy as np


def l2_relax(time_periods, treated_unit, X, tau):
    """
    Implements the L2-relaxation estimator using cvxpy, for time-series panel data, including an intercept.

    Parameters:
        time_periods (int): Number of initial time periods to use for estimation.
        treated_unit (numpy.ndarray): Outcome vector for the treated unit (n x 1).
        X (numpy.ndarray): Design matrix with rows as time periods and columns as control units (n x p).
        tau (float): Tuning parameter for the sup-norm constraint.

    Returns:
        numpy.ndarray: Estimated coefficients (p x 1).
        float: Estimated intercept.
        numpy.ndarray: Predicted counterfactual outcomes for the treated unit.
    """
    # Subset data for the first `time_periods` rows
    treated_unit_subset = treated_unit[:time_periods]
    X_subset = X[:time_periods, :]

    # Add a column of ones to X_subset for the intercept
    X_with_intercept = np.hstack((np.ones((time_periods, 1)), X_subset))

    # Calculate sample covariance components
    n, p = X_with_intercept.shape
    Sigma = (X_with_intercept.T @ X_with_intercept) / n
    eta = (X_with_intercept.T @ treated_unit_subset) / n

    # Define the variable (beta includes intercept as the first element)
    beta = cp.Variable(p)

    # Objective: minimize 1/2 * ||beta||_2^2
    objective = cp.Minimize(0.5 * cp.norm(beta, 2) ** 2)

    # Constraint: ||eta - Sigma @ beta||_inf <= tau
    constraint = [cp.norm(eta - Sigma @ beta, "inf") <= tau]

    # Problem formulation
    problem = cp.Problem(objective, constraint)

    # Solve the problem
    problem.solve(solver=cp.CLARABEL)

    # Check if optimization was successful
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status {problem.status}")

    # Extract intercept and coefficients
    intercept = beta.value[0]
    coefficients = beta.value[1:]

    # Predict counterfactual outcomes for the treated unit
    counterfactuals = X_with_intercept @ beta.value + intercept

    return coefficients, intercept, counterfactuals


def cross_validate_tau(treated_unit, X, tau_init, num_tau=1000):
    """
    Cross-validation for L2-relaxation to select the optimal tau by splitting
    the training data into two halves, including an intercept.

    Parameters:
        treated_unit (numpy.ndarray): Outcome vector for the treated unit (n x 1).
        X (numpy.ndarray): Design matrix with rows as time periods and columns as control units (n x p).
        tau_init (float): Upper bound for tau values (derived from initial computation).
        num_tau (int): Number of tau values to evaluate (default: 1000).

    Returns:
        float: Optimal tau value.
        float: Minimum out-of-sample MSE.
    """

    def cvmapper(tau):
        """
        Computes the MSE for a given tau by training on the first half and validating on the second half.

        Parameters:
            tau (float): Regularization parameter for L2-relaxation.

        Returns:
            float: MSE on the validation set.
        """
        # Train L2-relaxation on the first half of the data
        coefficients, intercept, _ = l2_relax(half_n, treated_train, X_train, tau)

        # Add intercept to the predictions on the validation set
        predictions = X_val @ coefficients + intercept

        # Compute Mean Squared Error (MSE) on the validation set
        mse = np.mean((treated_val - predictions) ** 2)
        return mse


    n = len(treated_unit)
    half_n = n // 2

    # Split data into first and second halves
    treated_train = treated_unit[:half_n]
    X_train = X[:half_n, :]
    treated_val = treated_unit[half_n:]
    X_val = X[half_n:, :]

    # Add a column of ones to X_train and X_val for the intercept
    X_train = np.hstack((np.ones((half_n, 1)), X_train))
    X_val = np.hstack((np.ones((n - half_n, 1)), X_val))

    # Generate tau values in logspace from a small positive number to tau_init
    tau_values = np.logspace(-4, np.log10(tau_init), num=num_tau)

    # Use map to compute MSE for each tau
    mse_errors = list(map(cvmapper, tau_values))

    # Find the tau with the minimum validation error
    optimal_tau_idx = np.argmin(mse_errors)
    optimal_tau = tau_values[optimal_tau_idx]
    min_mse = mse_errors[optimal_tau_idx]

    return optimal_tau, min_mse


def ci_bootstrap(b, Nco, x, y, t1, nb, att, method, y_counterfactual):
    """
    Perform subsampling bootstrap for TSSC

    Args:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        t1 (int): Number of observations for the treatment group.
        nb (int): Number of bootstrap samples.

    Returns:
        numpy.ndarray: Array of treatment effects estimates.
    """

    t = len(y_counterfactual)
    m = t1-5

    t2 =  t-t1

    zz1 = np.concatenate((x[:t1], y[:t1].reshape(-1, 1)), axis=1)
    # Confidence Intervals
    sigma_MSC = np.sqrt(np.mean((y[:t1] - y_counterfactual[:t1]) ** 2))

    sigma2_v = np.mean((y[t1:] - y_counterfactual[t1:] - np.mean(y[t1:] - y_counterfactual[t1:])) ** 2)  # \hat \Sigma_v

    e1_star = np.sqrt(sigma2_v) * np.random.randn(t2, nb)  # e_{1t}^* iid N(0, \Sigma^2_v)
    A_star = np.zeros(nb)

    x2 = x[t1 + 1: t, :]

    np.random.seed(1476)

    for g in range(nb):

        np.random.shuffle(zz1)  # the random generator does not repeat the same number

        zm = zz1[:m, :]  # randomly select m rows from z1_{T_1 by (N+1)}
        xm = zm[:, :-1]  # it uses the last m observation
        ym = np.dot(xm, b) + sigma_MSC * np.random.randn(m)

        if method in ["MSCa"]:
            xm = xm[:, 1:]

        if method in ["MSCc"]:
            xm = xm[:, 1:]

        bm = Opt.SCopt(Nco, ym[:t1], t1, xm[:t1], model=method)

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

    return cr_025_0975

def TSEST(x, y, t1, nb, donornames, t2):
    # Define a list to store dictionaries for each method
    fit_dicts_list = []
    fit_results = {}
    # List of methods to loop over
    methods = ["SIMPLEX", "MSCb", "MSCa", "MSCc"]

    for method in methods:

        if method in ["MSCc"]:
            x = x[:, 1:]

        Nco = x.shape[1]

        b = Opt.SCopt(Nco, y[:t1], t1, x[:t1], model=method)

        weights_dict = {donor: weight for donor, weight in zip(donornames, np.round(b, 4)) if weight > 0.001}

        if method in ["MSCa"]:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        if method in ["MSCc"]:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        # Calculate the counterfactual outcome
        y_counterfactual = x.dot(b)

        attdict, fitdict, Vectors = effects.calculate(y, y_counterfactual, t1, t2)

        att = attdict["ATT"]

        cis = ci_bootstrap(b, Nco, x, y, t1, nb, att, method, y_counterfactual)

        # Create Fit_dict for the specific method
        fit_dict = {"Fit": fitdict,
                    "Effects": attdict,
                    "95% CI": cis,
                    "Vectors": Vectors,
                    "WeightV": np.round(b, 3),
                    "Weights": weights_dict
                    }

        # Append fit_dict to the list
        fit_dicts_list.append({method: fit_dict})

    return fit_dicts_list


def pcr(X, y, objective, donor_names, xfull, pre=10, cluster=False):
    #n, p = X[:pre].shape
    #k = (min(n, p) - 1) // 2
    #Y0_rank, Topt_gauss, rank = adaptiveHardThresholding(X[:pre], k, strategy='0')

    # Perform SVT on the original donor matrix
    Y0_rank, n2, u_rank, s_rank, v_rank = svt(X[:pre])

    if cluster:
        X_sub, selected_donor_names, indices = SVDCluster(X[:pre], y, donor_names)
        Y0_rank, n2, u_rank, s_rank, v_rank = svt(X_sub[:pre])
        # Estimate synthetic control weights
        weights = Opt.SCopt(Y0_rank.shape[1], y[:pre], X_sub.shape[0], Y0_rank, model=objective, donor_names=donor_names)

        weights_dict = {selected_donor_names[i]: weights[i] for i in range(len(weights))}

        return weights_dict, np.dot(X[:, indices], weights)
    else:
        # Default to the full low-rank approximation
        #Y0_rank = np.dot(u_rank, np.dot(np.diag(s_rank), v_rank))
        weights = Opt.SCopt(n2, y[:pre], X.shape[0], Y0_rank, model=objective, donor_names=donor_names)

        weights_dict = {donor_names[i]: weights[i] for i in range(len(weights))}
        return weights_dict, np.dot(X, weights)


class Opt:
    @staticmethod
    def SCopt(Nco, y, t1, X, model="MSCb", donor_names=None):

        # Check if matrix dimensions allow multiplication
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of columns in X must be equal to the number of rows in y.")

        if model in ["MSCa", "MSCc"]:
            Nco += 1
            if donor_names:
                donor_names = ["Intercept"] + list(donor_names)

        # Define the optimization variable
        beta = cp.Variable(Nco)

        # Preallocate memory for constraints
        constraints = [beta >= 0] if model != "OLS" else []

        # If intercept added, append the intercept column to X
        if model in ["MSCa", "MSCc"]:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        # Define the objective function
        objective = cp.Minimize(cp.norm(y[:t1] - X @ beta, 2))

        # Define the constraints based on the selected method
        if model == "SIMPLEX":
            constraints.append(cp.sum(beta) == 1)  # Sum constraint for all coefficients
        elif model == "MSCa":
            constraints.append(cp.sum(beta[1:]) == 1)  # Sum constraint for coefficients excluding intercept
        elif model == "MSCc":
            constraints.append(beta[1:] >= 0)

        prob = cp.Problem(objective, constraints)

        # Solve the problem
        result = prob.solve(solver=cp.CLARABEL)

        return beta.value


def pda(prepped, N, method="fs"):
    """
    Estimate the counterfactual outcomes using either 'fs' or 'LASSO' method.

    Parameters:
        prepped (dict): A dictionary containing the preprocessed data with keys:
                        - 'y' : Outcome variable
                        - 'donor_matrix' : Donor unit matrix
                        - 'donor_names' : Donor names (Index)
                        - 'pre_periods' : Number of pre-treatment periods
                        - 'total_periods' : Total number of periods
        N (int): Number of donor units
        method (str): Either 'fs' for fs-based synthetic control or 'LASSO' for Lasso regularization

    Returns:
        dict: Results containing donor names, coefficients, and counterfactual outcomes.
    """

    if method == "fs":
        # Step 1: Select donor units using PDAfs method
        result = PDAfs(prepped["y"], prepped["donor_matrix"], prepped["pre_periods"],
                       prepped["total_periods"], N)

        # Step 2: Extract donor names using selected donor indices
        donor_names_selected = prepped["donor_names"].values[result["selected_donors"]].tolist()

        # Step 3: Map donor names to their coefficients (skip intercept)
        donor_coefficients = {
            donor: round(coef, 2)  # Round coefficients for readability (skip intercept)
            for donor, coef in zip(donor_names_selected, result["model_coefficients"][1:])
        }

        # Step 4: Extract donor outcomes (both pre-treatment and post-treatment)
        donor_outcomes = prepped["donor_matrix"][:, result["selected_donors"]]

        # Step 5: Calculate counterfactual using the model coefficients (excluding intercept)
        intercept = result["model_coefficients"][0]
        weighted_donors = donor_outcomes.dot(result["model_coefficients"][1:])  # Dot product with coefficients
        y_fsPDA = intercept + weighted_donors  # Counterfactual outcome

        d_hat = prepped["y"][prepped["pre_periods"]:]- y_fsPDA[prepped["pre_periods"]:]

        lrvar_lag = int(np.floor(4 * (prepped["post_periods"] / 100) ** (2 / 9)))

        # Assuming d_hat, lrvar_lag, and T2 are defined earlier
        if lrvar_lag > 0:

            gamma_d = acf(d_hat, nlags=lrvar_lag, fft=False)

            # Calculate weights
            w = 1 - (np.arange(1, lrvar_lag + 1)) / (lrvar_lag + 1)

            # Compute long-run variance
            lrvar_d = gamma_d[0] + 2 * np.sum(w * gamma_d[1:])
        else:
            # Use variance if no lags
            lrvar_d = np.var(d_hat)

        # Compute standard error of the average treatment effect (ATE)
        lrse_ATE = np.sqrt(lrvar_d / prepped["post_periods"])

        # Calculate the Z statistic
        Z = np.mean(d_hat) / lrse_ATE

        # Calculate the p-value
        p_value = 2 * (1 - norm.cdf(np.abs(Z)))

        # Compute the 95% confidence interval
        quantile = norm.ppf(1 - 0.05 / 2)  # Two-tailed quantile for 95% CI
        CI_left = np.mean(d_hat) - quantile * lrse_ATE
        CI_right = np.mean(d_hat) + quantile * lrse_ATE
        CI = (CI_left, CI_right)

        Inference = {
            "t_stat": Z,
            "SE": lrse_ATE,
            "95% CI": CI,
            "p_value": p_value
        }

        attdict, fitdict, Vectors = effects.calculate(prepped["y"], y_fsPDA, prepped["pre_periods"],
                                                      prepped["post_periods"])

        return {
            "method": "fs",
            "Betas": donor_coefficients,
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors,
            "Inference": Inference
        }

    elif method == "LASSO":
        # Step 1: Calculate SST (total sum of squares)
        SST = np.sum((prepped["y"][:prepped["pre_periods"]] - np.mean(prepped["y"][:prepped["pre_periods"]])) ** 2)

        # Step 2: Perform LassoCV for cross-validated Lasso regularization
        las = LassoCV(cv=prepped["pre_periods"])
        las.fit(prepped["donor_matrix"][:prepped["pre_periods"], :], prepped["y"][:prepped["pre_periods"]])
        y_LHCW = las.predict(prepped["donor_matrix"])

        # Step 3: Get non-zero coefficients and their corresponding column names
        non_zero_coef_indices = np.where(las.coef_ != 0)[0]
        non_zero_coef_columns = prepped["donor_names"][non_zero_coef_indices]

        non_zero_coef_dict = dict(zip(non_zero_coef_columns, np.round(las.coef_[non_zero_coef_indices], 3)))

        # Step 4: Compute ATT using the effects.calculate function
        attdict, fitdict, Vectors = effects.calculate(prepped["y"], y_LHCW, prepped["pre_periods"],
                                                      prepped["post_periods"])

        # Step 5: Compute inference metrics (CI, t-stat, SE)
        def compute_inference_metrics(y1, y_pred, datax, t1, t2, ATT, alpha=0.05):
            """
            Compute the 95% confidence interval, test statistic, and standard error for ATT.

            Parameters:
                y1 (array): Observed pre-treatment outcomes for the treated unit.
                y_pred (array): Predicted pre-treatment outcomes for the treated unit.
                datax (array): Donor covariate matrix (pre-treatment).
                t1 (int): Number of pre-treatment periods.
                t2 (int): Number of post-treatment periods.
                ATT (float): Average Treatment Effect on the Treated.
                alpha (float): Significance level (default is 0.05 for a 95% confidence interval).

            Returns:
                dict: A dictionary containing the confidence interval (CI), test statistic (t-stat), and standard error (SE).
            """
            # Compute residuals and variance
            e1 = y1 - y_pred[:t1]
            sigma2 = np.mean(e1 ** 2)  # Residual variance estimate

            # Covariate-based variance components
            eta = np.mean(datax[:t1, :], axis=0).reshape(-1, 1)  # Mean covariates
            psi = datax[:t1, :].T @ datax[:t1, :] / t1  # Covariance structure

            # Variance components
            Omega_1 = (sigma2 * eta.T) @ np.linalg.inv(psi) @ eta
            Omega_2 = sigma2
            Omega = (t2 / t1) * Omega_1 + Omega_2  # Total variance

            # Standard error and t-statistic
            SE = np.sqrt(Omega / t2).item()
            t_stat = np.sqrt(t2) * ATT / np.sqrt(Omega).item()

            # Confidence interval
            quantile = norm.ppf(1 - alpha / 2)  # Two-tailed quantile
            CI_left = ATT - quantile * SE
            CI_right = ATT + quantile * SE
            CI = (CI_left, CI_right)

            return {"CI": CI, "t_stat": t_stat, "SE": SE}

        results = compute_inference_metrics(prepped["y"][:prepped["pre_periods"]], y_LHCW,
                                            prepped["donor_matrix"], prepped["pre_periods"],
                                            prepped["post_periods"], attdict['ATT'])

        return {
            "method": "LASSO",
            "non_zero_coef_dict": non_zero_coef_dict,
            "Inference": results,
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors
        }
    
    
    elif method == "l2":
        # To do: upper bound of sup norm
        # Learn the first 40 percent of the data best using the range of tau
        # then predict with that

        def compute_t_stat_and_ci(att, treatment_effects, truncation_lag, confidence_level=0.95):
            """
            Compute the t-statistic, standard error, p-value, and confidence interval for ATT.

            Args:
                att (float): Average treatment effect on the treated (ATT).
                treatment_effects (np.ndarray): Array of estimated treatment effects, (Delta_{t, tau}).
                truncation_lag (int): The truncation lag h.
                confidence_level (float): Desired confidence level (default is 0.95).

            Returns:
                dict: Dictionary containing t-statistic, standard error, p-value, and confidence interval.
            """
            T2 = len(treatment_effects)
            hac_variance = compute_hac_variance(treatment_effects, truncation_lag)
            standard_error = np.sqrt(hac_variance / T2)

            # t-statistic
            t_stat = np.sqrt(T2) * att / standard_error

            # p-value from the t-distribution (two-tailed)
            p_value = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=T2 - 1))

            # Compute the critical t-value for the given confidence level (two-tailed)
            t_critical = t_dist.ppf(1 - (1 - confidence_level) / 2, df=T2 - 1)

            # Confidence interval based on t-statistic, standard error, and critical t-value
            margin_of_error = t_critical * standard_error
            ci_low = att - margin_of_error
            ci_high = att + margin_of_error

            # Return results as a dictionary
            return {
                "t_stat": t_stat,
                "standard_error": standard_error,
                "p_value": p_value,
                "confidence_interval": (ci_low, ci_high)
            }

        # Compute initial components
        n, p = prepped["donor_matrix"][:prepped["pre_periods"], :].shape
        Sigma = (prepped["donor_matrix"][:prepped["pre_periods"], :].T @ prepped["donor_matrix"][:prepped["pre_periods"], :]) / n

        eta = (prepped["donor_matrix"][:prepped["pre_periods"], :].T @ prepped["y"][:prepped["pre_periods"]]) / n

        tau1 = np.linalg.norm(eta, ord=np.inf)

        optimal_tau, min_mse = cross_validate_tau(prepped["y"][:prepped["pre_periods"]], prepped["donor_matrix"][:prepped["pre_periods"]], 2)

        # Step 2: Re-fit the model using the optimal tau
        beta_hat, intercept, _ = l2_relax(prepped["pre_periods"], prepped["y"], prepped["donor_matrix"], optimal_tau)

        yl2 = prepped["donor_matrix"] @ beta_hat+ intercept


        
        attdict, fitdict, Vectors = effects.calculate(prepped["y"], yl2, prepped["pre_periods"],
                                                      prepped["post_periods"])

        h = int(np.floor(4 * (prepped["post_periods"] / 100) ** (2 / 9)))

        inference_results = compute_t_stat_and_ci(attdict["ATT"], Vectors["Gap"][-prepped["post_periods"]:, 0], h)

        # Step 3: Map donor names to their coefficients
        donor_coefficients = {
            donor: round(coef, 4)
            for donor, coef in zip(prepped["donor_names"].values, beta_hat)
        }

        return {
            "method": r'l2 relaxation',
            "optimal_tau": optimal_tau,
            "Betas": donor_coefficients,
            "Inference": inference_results,
            "Effects": attdict, 
            "Fit": fitdict, 
            "Vectors": Vectors
        }

    else:
        raise ValueError("Invalid method specified. Choose either 'fs', 'LASSO', or 'l2'.")

