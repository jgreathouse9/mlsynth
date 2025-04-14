import numpy as np
import cvxpy as cp
from mlsynth.utils.denoiseutils import (
    universal_rank, svt, RPCA,
    spectral_rank,
    RPCA_HQF)
from mlsynth.utils.resultutils import effects
from mlsynth.utils.selectorsutils import (
    determine_optimal_clusters,
    SVDCluster,
    PDAfs,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from functools import partial
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from screenot.ScreeNOT import adaptiveHardThresholding
from sklearn.linear_model import LassoCV
from scipy.stats import t as t_dist
from mlsynth.utils.bayesutils import BayesSCM
from mlsynth.utils.selectorsutils import fpca

def pi2(Y, W, Z0, T0, t1, T, lag, Cw=None, Cy=None):
    """
    Proximal inference for treatment effect estimation using GMM.

    Parameters:
    - Y: Outcome vector (T x 1)
    - W: Design matrix (T x dimW)
    - Z0: Proxy matrix (T x dimZ0)
    - T0: Number of pre-treatment periods
    - t1: Number of post-treatment periods
    - T: Total time periods
    - lag: HAC lag length for variance estimation
    - Cw, Cy: Optional covariates to augment W and Z0

    Returns:
    - y_PI: Predicted outcomes using proximal inference
    - alpha: Estimated coefficients for the design matrix
    - se_tau: Standard error of the treatment effect
    """
    if W.shape[1] != Z0.shape[1]:
        raise RuntimeError("Design matrix W and proxy matrix Z0 must have the same number of columns.")
    
    # Step 1: Augment W and Z0 with covariates if provided
    if Cw is not None and Cy is not None:
        Z0 = np.column_stack((Z0, Cy, Cw))
        W = np.column_stack((W, Cy, Cw))
    
    # Step 2: Solve for alpha and tau using CVXPY
    dimW = W.shape[1]
    alpha = cp.Variable(dimW)
    tau = cp.Variable(1)

    # Residuals
    residual1 = Z0[:T0].T @ (Y[:T0] - W[:T0] @ alpha)  # Pre-treatment residuals
    residual2 = (Y - tau - W @ alpha)  # Post-treatment residuals
    
    # Mask the pre-treatment and post-treatment residuals
    pre_treatment_mask = np.concatenate([np.ones(T0), np.zeros(T - T0)])
    post_treatment_mask = np.concatenate([np.zeros(T0), np.ones(T - T0)])
    
    residual2_masked = cp.multiply(post_treatment_mask, residual2)
    residual1_masked = cp.multiply(pre_treatment_mask[:T0], residual1)  # Only for Z0 residuals
    
    # Combine residuals
    U = cp.vstack([residual1_masked, residual2_masked])

    # Minimize quadratic form for GMM
    Omega_inv = np.eye(U.shape[0])  # Replace with your HAC Omega_inv
    objective = cp.Minimize(cp.quad_form(U, Omega_inv))
    problem = cp.Problem(objective)
    problem.solve()

    # Extract solution
    alpha_val = alpha.value
    tau_val = tau.value

    # Step 3: Predict outcomes and calculate treatment effect
    y_PI = W @ alpha_val
    taut = Y - y_PI
    tau_mean = np.mean(taut[T0:T0 + t1])

    # Step 4: Compute variance and standard error of tau
    dimZ0 = Z0.shape[1]
    G = np.zeros((dimZ0 + 1, dimW + 1))
    G[:dimZ0, :dimW] = (Z0[:T0].T @ W[:T0]) / T
    G[-1, :dimW] = np.sum(W[T0:T0 + t1], axis=0) / T
    G[-1, -1] = t1 / T

    Omega = hac(U.value, lag)  # Compute HAC Omega using residuals
    Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
    var_tau = Cov[-1, -1]
    se_tau = np.sqrt(var_tau / T)

    return y_PI, alpha_val[:W.shape[1]], se_tau


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


def compute_t_stat_and_ci(
    att, treatment_effects, truncation_lag, confidence_level=0.95
):
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



def l2_relax(time_periods, treated_unit, X, tau):
    """
    Implements the L2-relaxation estimator using cvxpy for time-series panel data,.

    where the intercept is estimated separately after learning the coefficients.

    Parameters:
        time_periods (int): Number of initial time periods to use for estimation.
        treated_unit (numpy.ndarray): Outcome vector for the treated unit (T x 1).
        X (numpy.ndarray): Design matrix with rows as time periods and columns as control units (T x N).
        tau (float): Tuning parameter for the sup-norm constraint.

    Returns:
        numpy.ndarray: Estimated coefficients (p x 1).
        float: Estimated intercept.
        numpy.ndarray: Predicted counterfactual outcomes for the treated unit.
    """
    # Subset data for the first `time_periods` rows
    treated_unit_subset = treated_unit[:time_periods]
    X_subset = X[:time_periods, :]

    # Calculate sample covariance components
    n, p = X_subset.shape
    Sigma = (X_subset.T @ X_subset) / n
    eta = (X_subset.T @ treated_unit_subset) / n

    # Define the variable for coefficients
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

    # Extract the estimated coefficients
    coefficients = beta.value

    # Estimate the intercept separately
    intercept = np.mean(treated_unit_subset) - np.mean(X_subset @ coefficients)

    # Predict counterfactual outcomes for the treated unit
    counterfactuals = X @ coefficients + intercept

    return coefficients, intercept, counterfactuals


def cross_validate_tau(treated_unit, X, tau_init, num_tau=1000):
    """
    Cross-validation for L2-relaxation to select the optimal tau by splitting.

    the training data into two halves, without including an intercept.

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
        coefficients, _, _ = l2_relax(half_n, treated_train, X_train, tau)

        # Predict on the validation set without an intercept
        predictions = X_val @ coefficients

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
    Perform subsampling bootstrap for TSSC.

    Args:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        t1 (int): Number of observations for the treatment group.
        nb (int): Number of bootstrap samples.

    Returns:
        numpy.ndarray: Array of treatment effects estimates.
    """

    t = len(y_counterfactual)
    m = t1 - 5

    t2 = t - t1

    zz1 = np.concatenate((x[:t1], y[:t1].reshape(-1, 1)), axis=1)
    # Confidence Intervals
    sigma_MSC = np.sqrt(np.mean((y[:t1] - y_counterfactual[:t1]) ** 2))

    sigma2_v = np.mean(
        (
            y[t1:]
            - y_counterfactual[t1:]
            - np.mean(y[t1:] - y_counterfactual[t1:])
        )
        ** 2
    )  # \hat \Sigma_v

    e1_star = np.sqrt(sigma2_v) * np.random.randn(
        t2, nb
    )  # e_{1t}^* iid N(0, \Sigma^2_v)
    A_star = np.zeros(nb)

    x2 = x[t1 + 1 : t, :]

    np.random.seed(1476)

    for g in range(nb):

        np.random.shuffle(
            zz1
        )  # the random generator does not repeat the same number

        zm = zz1[:m, :]  # randomly select m rows from z1_{T_1 by (N+1)}
        xm = zm[:, :-1]  # it uses the last m observation
        ym = np.dot(xm, b) + sigma_MSC * np.random.randn(m)

        if method in ["MSCa"]:
            xm = xm[:, 1:]

        if method in ["MSCc"]:
            xm = xm[:, 1:]

        prob = Opt.SCopt(Nco, ym[:t1], t1, xm[:t1], model=method)

        bm = prob.solution.primal_vars[next(iter(prob.solution.primal_vars))]

        A1_star = -np.mean(np.dot(x2, (bm - b))) * np.sqrt(
            (t2 * m) / t1
        )  # A_1^*
        A2_star = np.sqrt(t2) * np.mean(e1_star[:, g])
        A_star[g] = A1_star + A2_star  # A^* = A_1^* + A_2^*

    # end of the subsampling-bootstrap loop
    ATT_order = np.sort(
        A_star / np.sqrt(t2)
    )  # sort subsampling ATT by ascending order

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

        prob = Opt.SCopt(Nco, y[:t1], t1, x[:t1], model=method)

        weightest = prob.solution.primal_vars[next(iter(prob.solution.primal_vars))]

        weights_dict = {
            donor: weight
            for donor, weight in zip(donornames, np.round(weightest, 4))
            if weight > 0.001
        }

        if method in ["MSCa"]:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        if method in ["MSCc"]:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        # Calculate the counterfactual outcome
        y_counterfactual = x.dot(weightest)

        attdict, fitdict, Vectors = effects.calculate(
            y, y_counterfactual, t1, t2
        )

        att = attdict["ATT"]

        cis = ci_bootstrap(weightest, Nco, x, y, t1, nb, att, method, y_counterfactual)

        # Create Fit_dict for the specific method
        fit_dict = {
            "Fit": fitdict,
            "Effects": attdict,
            "95% CI": cis,
            "Vectors": Vectors,
            "WeightV": np.round(weightest, 3),
            "Weights": weights_dict,
        }

        # Append fit_dict to the list
        fit_dicts_list.append({method: fit_dict})

    return fit_dicts_list


def pcr(X, y, objective, donor_names, xfull, pre=10, cluster=False, Frequentist=False):
    # Perform SVT on the original donor matrix
    Y0_rank, n2, u_rank, s_rank, v_rank = svt(X[:pre])

    if cluster:
        X_sub, selected_donor_names, indices = SVDCluster(X[:pre], y, donor_names)
        Y0_rank, n2, u_rank, s_rank, v_rank = svt(X_sub[:pre])

        # Frequentist Synthetic Control
        if Frequentist:
            prob = Opt.SCopt(Y0_rank.shape[1], y[:pre], X_sub.shape[0], Y0_rank, model=objective,
                                donor_names=donor_names)

            weights = prob.solution.primal_vars[next(iter(prob.solution.primal_vars))]


            weights_dict = {selected_donor_names[i]: weights[i] for i in range(len(weights))}

            return {"weights": weights_dict, "cf_mean": np.dot(X[:, indices], weights)}

        # Bayesian Synthetic Control
        else:
            alpha = 1.0
            beta_D, Sigma_D, Y_pred, Y_var = BayesSCM(Y0_rank, y[:pre], np.var(y[:pre], ddof=1), alpha)
            weights_dict = {selected_donor_names[i]: beta_D[i] for i in range(len(beta_D))}
            num_samples = 1000
            samples = np.random.multivariate_normal(beta_D, Sigma_D, size=num_samples)

            # Compute counterfactual predictions for each sample
            cf_samples = X[:, indices] @ samples.T

            # Compute credible intervals
            lower_bound = np.percentile(cf_samples, 2.5, axis=1)
            upper_bound = np.percentile(cf_samples, 97.5, axis=1)

            return {
                "weights": weights_dict,
                "cf_mean": np.median(cf_samples, axis=1),
                "credible_interval": (lower_bound, upper_bound)
            }

    else:
        if Frequentist:
            prob = Opt.SCopt(n2, y[:pre], X.shape[0], Y0_rank, model=objective, donor_names=donor_names)

            weights = prob.solution.primal_vars[next(iter(prob.solution.primal_vars))]
            return {"weights": weights, "cf_mean": np.dot(X, weights)}

        else:
            alpha = 1.0
            beta_D, Sigma_D, Y_pred, Y_var = BayesSCM(Y0_rank, y[:pre], np.var(y[:pre], ddof=1), alpha)
            weights_dict = {donor_names[i]: beta_D[i] for i in range(len(beta_D))}
            num_samples = 4000
            samples = np.random.multivariate_normal(beta_D, Sigma_D, size=num_samples)

            # Compute counterfactual predictions for each sample
            cf_samples = X @ samples.T

            # Compute credible intervals
            lower_bound = np.percentile(cf_samples, 2.5, axis=1)
            upper_bound = np.percentile(cf_samples, 97.5, axis=1)

            return {
                "weights": weights_dict,
                "cf_mean": np.median(cf_samples, axis=1),
                "credible_interval": (lower_bound, upper_bound)
            }


class Opt:
    @staticmethod
    def SCopt(Nco, y, t1, X, model="MSCb", donor_names=None, model_weights=None):
        """
        Parameters:
        - Nco: Number of control units (donors)
        - y: Target vector
        - t1: Number of pre-treatment periods
        - X: Donor matrix (features)
        - model: Optimization method ('MSCa', 'MSCb', 'MSCc', 'SIMPLEX', 'OLS', 'MA')
        - donor_names: Optional names of donor units
        - model_weights: List of Nco-length weight vectors (required if model="MA")

        Returns:
        - Optimized problem object
        """
        # Check if matrix dimensions allow multiplication
        if model != "MA" and X.shape[0] != y.shape[0]:
            raise ValueError("For non-MA models, X and y must have matching row counts.")

        if model in ["MSCa", "MSCc"]:
            Nco += 1
            if donor_names:
                donor_names = ["Intercept"] + list(donor_names)

        if model == "MA":
            if not isinstance(model_weights, dict):
                raise ValueError("For 'MA', model_weights must be a dictionary of {model_name: weight_vector}")

            model_names = list(model_weights.keys())
            M = len(model_names)

            # Stack counterfactuals as columns: each column is Y0 @ w_model
            cf_matrix = np.column_stack([model_weights[name]["cf"] for name in model_names])

            lambda_vars = cp.Variable(M, nonneg=True)
            objective = cp.Minimize(cp.norm(y[:t1] - cf_matrix[:t1] @ lambda_vars, 2))
            constraints = [cp.sum(lambda_vars) == 1]
            prob = cp.Problem(objective, constraints)
            prob.solve()

            lambda_values = lambda_vars.value

            lambda_dict = {model_names[i]: float(lambda_values[i]) for i in range(M)}

            # Compute final model-averaged donor weights
            final_weights = sum(lambda_values[i] * model_weights[model_names[i]]["weights"] for i in range(M))

            cf_pred = np.dot(cf_matrix, lambda_values)  # Counterfactual as a weighted sum

            # Return dict of {model_name: lambda_value}
            return {"Lambdas": lambda_dict, "w_MA": final_weights, "Counterfactual": cf_pred}

        else:
            # Define the optimization variable
            beta = cp.Variable(Nco)

            # Constraints setup
            constraints = [beta >= 0] if model != "OLS" else []

            # If intercept added, append the intercept column to X
            if model in ["MSCa", "MSCc"]:
                X = np.c_[np.ones((X.shape[0], 1)), X]

            # Define the objective function
            objective = cp.Minimize(cp.norm(y[:t1] - X @ beta, 2))

            # Constraints for different models
            if model == "SIMPLEX":
                constraints.append(cp.sum(beta) == 1)
            elif model == "MSCa":
                constraints.append(cp.sum(beta[1:]) == 1)
            elif model == "MSCc":
                constraints.append(beta[1:] >= 0)

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.CLARABEL)

        return prob


def pda(prepped, N, method="fs",tau=None):
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
        result = PDAfs(
            prepped["y"],
            prepped["donor_matrix"],
            prepped["pre_periods"],
            prepped["total_periods"],
            N,
        )

        # Step 2: Extract donor names using selected donor indices
        donor_names_selected = (
            prepped["donor_names"].values[result["selected_donors"]].tolist()
        )

        # Step 3: Map donor names to their coefficients (skip intercept)
        donor_coefficients = {
            donor: round(
                coef, 2
            )  # Round coefficients for readability (skip intercept)
            for donor, coef in zip(
                donor_names_selected, result["model_coefficients"][1:]
            )
        }

        # Step 4: Extract donor outcomes (both pre-treatment and post-treatment)
        donor_outcomes = prepped["donor_matrix"][:, result["selected_donors"]]

        # Step 5: Calculate counterfactual using the model coefficients (excluding intercept)
        intercept = result["model_coefficients"][0]
        weighted_donors = donor_outcomes.dot(
            result["model_coefficients"][1:]
        )  # Dot product with coefficients
        y_fsPDA = intercept + weighted_donors  # Counterfactual outcome

        d_hat = (
            prepped["y"][prepped["pre_periods"] :]
            - y_fsPDA[prepped["pre_periods"] :]
        )

        lrvar_lag = int(
            np.floor(4 * (prepped["post_periods"] / 100) ** (2 / 9))
        )

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
            "p_value": p_value,
        }

        attdict, fitdict, Vectors = effects.calculate(
            prepped["y"],
            y_fsPDA,
            prepped["pre_periods"],
            prepped["post_periods"],
        )

        return {
            "method": "fs",
            "Betas": donor_coefficients,
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors,
            "Inference": Inference,
        }

    elif method == "LASSO":
        # Step 1: Calculate SST (total sum of squares)
        SST = np.sum(
            (
                prepped["y"][: prepped["pre_periods"]]
                - np.mean(prepped["y"][: prepped["pre_periods"]])
            )
            ** 2
        )

        # Step 2: Perform LassoCV for cross-validated Lasso regularization
        las = LassoCV(cv=prepped["pre_periods"])
        las.fit(
            prepped["donor_matrix"][: prepped["pre_periods"], :],
            prepped["y"][: prepped["pre_periods"]],
        )
        y_LHCW = las.predict(prepped["donor_matrix"])

        # Step 3: Get non-zero coefficients and their corresponding column names
        non_zero_coef_indices = np.where(las.coef_ != 0)[0]
        non_zero_coef_columns = prepped["donor_names"][non_zero_coef_indices]

        non_zero_coef_dict = dict(
            zip(
                non_zero_coef_columns,
                np.round(las.coef_[non_zero_coef_indices], 3),
            )
        )

        # Step 4: Compute ATT using the effects.calculate function
        attdict, fitdict, Vectors = effects.calculate(
            prepped["y"],
            y_LHCW,
            prepped["pre_periods"],
            prepped["post_periods"],
        )

        # Step 5: Compute inference metrics (CI, t-stat, SE)
        def compute_inference_metrics(
            y1, y_pred, datax, t1, t2, ATT, alpha=0.05
        ):
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
            sigma2 = np.mean(e1**2)  # Residual variance estimate

            # Covariate-based variance components
            eta = np.mean(datax[:t1, :], axis=0).reshape(
                -1, 1
            )  # Mean covariates
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

        results = compute_inference_metrics(
            prepped["y"][: prepped["pre_periods"]],
            y_LHCW,
            prepped["donor_matrix"],
            prepped["pre_periods"],
            prepped["post_periods"],
            attdict["ATT"],
        )

        return {
            "method": "LASSO",
            "non_zero_coef_dict": non_zero_coef_dict,
            "Inference": results,
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors,
        }

    elif method == "l2":
        # To do: upper bound of sup norm
        # Learn the first 40 percent of the data best using the range of tau
        # then predict with that

        def compute_t_stat_and_ci(
            att, treatment_effects, truncation_lag, confidence_level=0.95
        ):
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
            hac_variance = compute_hac_variance(
                treatment_effects, truncation_lag
            )
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
                "confidence_interval": (ci_low, ci_high),
            }

        # Compute initial components
        n, p = prepped["donor_matrix"][: prepped["pre_periods"], :].shape
        Sigma = (
            prepped["donor_matrix"][: prepped["pre_periods"], :].T
            @ prepped["donor_matrix"][: prepped["pre_periods"], :]
        ) / n

        eta = (
            prepped["donor_matrix"][: prepped["pre_periods"], :].T
            @ prepped["y"][: prepped["pre_periods"]]
        ) / n

        tau1 = 1.5


        if tau is not None:
            # Use the user-specified tau
            tau_to_use = tau
        else:
            # Perform cross-validation to determine the optimal tau
            tau_to_use, min_mse = cross_validate_tau(
                prepped["y"][: prepped["pre_periods"]],
                prepped["donor_matrix"][: prepped["pre_periods"]],
                tau1,
            )

        # Fit the model using the selected tau (either user-specified or cross-validated)
        beta_hat, intercept, _ = l2_relax(
            prepped["pre_periods"],
            prepped["y"],
            prepped["donor_matrix"],
            tau_to_use,
        )

        yl2 = prepped["donor_matrix"] @ beta_hat + intercept

        attdict, fitdict, Vectors = effects.calculate(
            prepped["y"], yl2, prepped["pre_periods"], prepped["post_periods"]
        )

        h = int(np.floor(4 * (prepped["post_periods"] / 100) ** (2 / 9)))

        inference_results = compute_t_stat_and_ci(
            attdict["ATT"], Vectors["Gap"][-prepped["post_periods"] :, 0], h
        )

        # Step 3: Map donor names to their coefficients
        donor_coefficients = {
            donor: round(coef, 4)
            for donor, coef in zip(prepped["donor_names"].values, beta_hat)
        }

        return {
            "method": r"l2 relaxation",
            "optimal_tau": tau_to_use,
            "Betas": donor_coefficients,
            "Inference": inference_results,
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors,
            "Intercept": intercept
        }

    else:
        raise ValueError(
            "Invalid method specified. Choose either 'fs', 'LASSO', or 'l2'."
        )



def bartlett(i, J):
    if np.abs(i) <= J:
        return 1 - np.abs(i) / (J + 1)
    else:
        return 0


def hac(G, J, kernel=bartlett):
    T, K = G.shape
    omega = np.zeros((K, K))
    for j in range(-min(J, T - 1), min(J, T - 1) + 1):  # Limit lags to valid range
        k = kernel(j, J)
        if j >= 0:
            idx0, idx1 = 0, T - j - 1
            omega += k * (G[idx0:idx1 + 1].T @ G[idx0 + j:idx1 + 1 + j]) / T
        else:
            idx0, idx1 = -j, T - 1
            omega += k * (G[idx0:idx1 + 1].T @ G[idx0 + j:idx1 + 1 + j]) / T
    return omega


def pi(Y, W, Z0, T0, t1, T, lag, Cw=None, Cy=None):
    if W.shape[1] == Z0.shape[1]:
        if Cw is not None and Cy is not None:
            Z0 = np.column_stack((Z0, Cy, Cw))
            W = np.column_stack((W, Cy, Cw))

        Z0W = Z0[:T0].T @ W[:T0]
        Z0Y = Z0[:T0].T @ Y[:T0]
        alpha = np.linalg.solve(Z0W, Z0Y)
        y_PI = W.dot(alpha)
        taut = Y - W.dot(alpha)
        tau = np.mean(taut[T0:T0 + t1])


        # Inference with GMM
        U0 = Z0.T * (Y - W.dot(alpha))
        U1 = Y - tau - W.dot(alpha)
        U0[:, T0:] *= 0
        U1[:T0] *= 0
        U = np.column_stack((U0.T, U1))

        dimZ0, dimW = Z0.shape[1], W.shape[1]
        G = np.zeros((U.shape[1], U.shape[1]))
        G[:dimZ0, :dimW] = Z0W / T
        G[-1, :dimW] = np.sum(W[T0:T0 + t1], axis=0) / T
        G[-1, -1] = t1 / T

        Omega = hac(U, lag)
        Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
        var_tau = Cov[-1, -1]
        se_tau = np.sqrt(var_tau / T)
    else:
        raise RuntimeError("Not implemented yet.")

    return y_PI, alpha[:W.shape[1]], se_tau


def pi_surrogate(Y, W, Z0, Z1, X, T0, t1, T, lag, Cw=None, Cy=None, Cx=None):
    if W.shape[1] == Z0.shape[1] and X.shape[1] == Z1.shape[1]:
        if Cw is not None and Cy is not None and Cx is not None:
            Z0 = np.column_stack((Z0, Cy, Cw))
            W = np.column_stack((W, Cy, Cw))
            Z1 = np.column_stack((Z1, Cx))
            X = np.column_stack((X, Cx))

        Z0W = Z0[:T0].T @ W[:T0]
        Z0Y = Z0[:T0].T @ Y[:T0]
        alpha = np.linalg.solve(Z0W, Z0Y)
        tauhat = Y[T0:] - W[T0:].dot(alpha)

        Z1X = Z1[T0:].T @ X[T0:]
        Z1tau = Z1[T0:].T @ tauhat
        gamma = np.linalg.solve(Z1X, Z1tau)
        taut = X.dot(gamma)
        taut[:T0] = (Y - W.dot(alpha))[:T0]
        tau = np.mean(taut[T0:T0 + t1])

        # Inference with GMM
        U0 = Z0.T * (Y - W.dot(alpha))
        U1 = Z1.T * (Y - W.dot(alpha) - X.dot(gamma))
        U2 = X.dot(gamma) - tau
        U0[:, T0:] *= 0
        U1[:, :T0] *= 0
        U2[:T0] *= 0
        U = np.column_stack((U0.T, U1.T, U2))

        dimZ0, dimZ1, dimW, dimX = Z0.shape[1], Z1.shape[1], W.shape[1], X.shape[1]
        G = np.zeros((U.shape[1], U.shape[1]))
        G[:dimZ0, :dimW] = Z0W / T
        G[dimZ0:dimZ0 + dimZ1, :dimW] = Z1[T0:].T @ W[T0:] / T
        G[dimZ0:dimZ0 + dimZ1, dimW:dimW + dimX] = Z1[T0:].T @ X[T0:] / T
        G[-1, dimW:dimW + dimX] = -np.sum(X[T0:T0 + t1], axis=0) / T
        G[-1, -1] = t1 / T

        Omega = hac(U, lag)
        Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
        var_tau = Cov[-1, -1]
        se_tau = np.sqrt(var_tau / T)
    else:
        raise RuntimeError("Not implemented yet.")

    return tau, taut, alpha[:W.shape[1]], se_tau

def pi_surrogate_post(Y, W, Z0, Z1, X, T0, T1, lag, Cw=None, Cy=None, Cx=None):
    """
    Computes the treatment effect using post-treatment surrogate variables.

    and instruments with GMM-based inference.

    Parameters:
    Y : np.ndarray
        Outcome variable (T x 1)
    W : np.ndarray
        Covariates (T x dim(W))
    Z0 : np.ndarray
        Pre-treatment instruments (T x dim(Z0))
    Z1 : np.ndarray
        Surrogate variables (T x dim(Z1))
    X : np.ndarray
        Post-treatment covariates (T x dim(X))
    T0 : int
        Time period at which treatment starts
    T1 : int
        Length of the post-treatment period
    lag : int
        Lag parameter for HAC covariance estimation
    Cw : np.ndarray, optional
        Additional covariates for W (T x dim(Cw))
    Cy : np.ndarray, optional
        Additional covariates for Y (T x dim(Cy))
    Cx : np.ndarray, optional
        Additional covariates for X (T x dim(Cx))

    Returns:
    tau : float
        Estimated average treatment effect
    taut : np.ndarray
        Time-varying treatment effect estimates
    params : np.ndarray
        Estimated coefficients for W and X
    se_tau : float
        Standard error of the estimated treatment effect
    """
    # Check dimension compatibility
    if W.shape[1] == Z0.shape[1] and X.shape[1] == Z1.shape[1]:
        # Add additional covariates if provided
        if Cw is not None and Cy is not None and Cx is not None:
            Z = np.column_stack((Z0, Cy, Cw, Z1, Cx))
            WX = np.column_stack((W, Cy, Cw, X, Cx))
            X_ext = np.column_stack((X, Cx))
        else:
            Z = np.column_stack((Z0, Z1))
            WX = np.column_stack((W, X))
            X_ext = X

        # Solve for parameters
        ZWX = Z[T0:].T @ WX[T0:]
        ZY = Z[T0:].T @ Y[T0:]
        params = np.linalg.solve(ZWX, ZY)
        gamma = params[-X_ext.shape[1]:]
        taut = X_ext.dot(gamma)
        tau = np.mean(taut[T0:])

        # Inference with GMM
        U0 = (Z.T * (Y - WX.dot(params)))[:, T0:]
        U1 = X_ext[T0:].dot(gamma) - tau
        U = np.column_stack((U0.T, U1))

        # Construct G matrix
        G = np.zeros((U.shape[1], U.shape[1]))
        G[:Z.shape[1], :WX.shape[1]] = ZWX / T1
        G[-1, -X_ext.shape[1] - 1:-1] = -np.sum(X_ext[T0:], axis=0) / T1
        G[-1, -1] = 1
        Omega = hac(U, lag)
        Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
        var_tau = Cov[-1, -1]
        se_tau = np.sqrt(var_tau / T1)
    else:
        raise RuntimeError("Not implemented yet.")

    return tau, taut, params[:W.shape[1]], se_tau


def get_theta(y1_pre, Y0_pre):
    y1_demeaned = y1_pre - np.mean(y1_pre)
    Y0_demeaned = Y0_pre - np.mean(Y0_pre, axis=0)

    numerators = Y0_demeaned.T @ y1_demeaned
    denominators = np.sum(Y0_demeaned ** 2, axis=0)
    theta_hat = numerators / (denominators)

    Y_theta = Y0_pre * theta_hat
    return theta_hat, Y_theta

# -------------------------------
# Step 2: Estimate Noise Variance
# -------------------------------

def get_sigmasq(y1_pre, Y0_pre):
    T0 = len(y1_pre)
    Q = np.eye(T0) - np.ones((T0, T0)) / T0

    G = Y0_pre.T @ Q @ Y0_pre
    diag_G = np.diag(G)
    Z = Y0_pre @ np.diag(1 / (diag_G)) @ Y0_pre.T

    projection = Z @ Q @ y1_pre
    residual = Q @ y1_pre - Q @ projection
    sigma2 = np.linalg.norm(residual) ** 2
    return sigma2

def __SRC_opt(y1_pre, Y0_pre, Y0_post, Y_theta, theta_hat, sigma2):
    """
    Perform SRC weight optimization and prediction.

    Parameters:
    - y1_pre: (T0,) vector, pre-treatment treated unit outcomes
    - Y0_pre: (T0, J) matrix, pre-treatment donor outcomes
    - Y0_post: (T1, J) matrix, post-treatment donor outcomes
    - Y_theta: (T0, J) donor matrix adjusted by alignment coefficients
    - theta_hat: (J,) alignment coefficients
    - sigma2: estimated noise variance
    - post_periods: list of post-treatment indices

    Returns:
    - y1_hat_pre: (T0,) in-sample predicted treated outcomes
    - y1_hat_post: (T1,) post-treatment counterfactuals
    - w_hat: (J,) optimal donor weights
    - theta_hat: (J,) alignment coefficients (just passed through)
    """

    J = Y0_pre.shape[1]
    w = cp.Variable(J, nonneg=True)

    loss = cp.sum_squares(y1_pre - Y_theta @ w)
    penalty = 2 * sigma2 * cp.sum(w)
    objective = cp.Minimize(loss + penalty)

    constraints = [cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    w_hat = w.value

    y1_bar = np.mean(y1_pre)
    yj_bar = np.mean(Y0_pre, axis=0)
    y1_hat_pre = Y0_pre @ (w_hat * theta_hat)
    y1_hat_post = y1_bar + (Y0_post - yj_bar) @ (w_hat * theta_hat)

    return y1_hat_pre, y1_hat_post, w_hat, theta_hat

def SRCest(y1, Y0, post_periods):
    T, J = Y0.shape
    T0 = T-post_periods
    y1_pre = y1[:T0]
    Y0_pre = Y0[:T0]

    theta_hat, Y_theta = get_theta(y1_pre, Y0_pre)
    sigma2 = get_sigmasq(y1_pre, Y0_pre)
    Y0_post = Y0[-post_periods:]

    y1_hat_pre, y1_hat_post, w_hat, theta_hat = __SRC_opt(
        y1_pre, Y0_pre, Y0_post, Y_theta, theta_hat, sigma2)
    # In-sample prediction
    y1_hat_pre = Y0_pre @ (w_hat * theta_hat)

    return np.concatenate([y1_hat_pre, y1_hat_post]), w_hat, theta_hat

def RPCASYNTH(df, config, prepped):

    unitid = config["unitid"]
    time = config["time"]
    outcome = config["outcome"]
    treated_unit_name = prepped["treated_unit_name"]
    pre_periods = prepped["pre_periods"]
    post_periods = prepped["post_periods"]
    ROB = config.get("ROB", "PCP")  # default to PCP

    # Pivot to wide format
    trainframe = df.pivot_table(index=unitid, columns=time, values=outcome, sort=False)

    # Extract pre-treatment outcomes
    X = trainframe.iloc[:, :pre_periods]

    # Functional PCA clustering
    optimal_clusters, cluster_x, numvals = fpca(X)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, init="k-means++", algorithm="elkan")
    trainframe["cluster"] = kmeans.fit_predict(cluster_x)

    # Get donor pool with same cluster
    treat_cluster = trainframe.at[treated_unit_name, "cluster"]
    clustered_units = trainframe[trainframe["cluster"] == treat_cluster].drop("cluster", axis=1)

    # Extract arrays
    treated_row_idx = clustered_units.index.get_loc(treated_unit_name)
    Y = clustered_units.to_numpy()
    y = Y[treated_row_idx]
    Y0 = np.delete(Y, treated_row_idx, axis=0)

    # Robust PCA
    if ROB == "PCP":
        L = RPCA(Y0)
    elif ROB == "HQF":
        m, n = Y0.shape
        u, s, v = np.linalg.svd(Y0, full_matrices=False)
        t = 0.999
        lambda_1 = 1 / np.sqrt(max(m, n))
        L = RPCA_HQF(Y0, spectral_rank(s, t=t), maxiter=1000, ip=1, lam_1=lambda_1)
    else:
        warnings.warn(f"Invalid robust method '{ROB}'. Defaulting to 'PCP'.", UserWarning)
        L = RPCA(Y0)

    # Solve synthetic control weights
    rpcaw = Opt.SCopt(
        len(Y0),
        prepped["y"][:pre_periods],
        pre_periods,
        L[:, :pre_periods].T,
        model="MSCb"
    )
    beta_value = rpcaw.solution.primal_vars[next(iter(rpcaw.solution.primal_vars))]
    y_RPCA = np.dot(L.T, beta_value)
    beta_value = np.round(beta_value, 3)

    # Return weights
    unit_names = [name for name in clustered_units.index if name != treated_unit_name]
    weights = {name: weight for name, weight in zip(unit_names, beta_value)}

    # Effects + Vectors
    Rattdict, Rfitdict, RVectors = effects.calculate(prepped["y"], y_RPCA, pre_periods, post_periods)

    return {
        "name": "RPCA",
        "weights": weights,
        "Effects": Rattdict,
        "Fit": Rfitdict,
        "Vectors": RVectors,
    }

import numpy as np
import cvxpy as cp

def SMOweights(data, method='concatenated', T0=None):
    """
    Estimate SCM weights using either concatenated (TLP) or averaged (SBMF) objective,
    assuming donors are shape (T, N) — i.e., rows = time periods, columns = units.

    Parameters:
        data (dict): {
            'Target': list of 1D numpy arrays, each of shape (T,),
            'Donors': list of 2D numpy arrays, each of shape (T, N)
        }
        method (str): 'concatenated' or 'average'
        T0 (int): number of pre-treatment periods

    Returns:
        np.ndarray: Optimal weight vector of shape (N,)
    """
    target_list = data.get("Target")
    donor_list = data.get("Donors")

    assert isinstance(target_list, list) and isinstance(donor_list, list), "Inputs must be lists."
    assert len(target_list) == len(donor_list), "Target and Donor lists must be same length."

    K = len(target_list)
    T_full, N = donor_list[0].shape

    if T0 is None:
        T0 = T_full

    # Truncate to pre-treatment period
    target_list = [y[:T0] for y in target_list]
    donor_list = [Y[:T0, :] for Y in donor_list]

    w = cp.Variable(N)

    if method == "concatenated":
        # Stack rows (time) from each outcome
        y_stack = np.concatenate(target_list, axis=0)  # shape (K*T0,)
        Y_stack = np.vstack(donor_list)               # shape (K*T0, N)
        objective = cp.Minimize(cp.sum_squares(Y_stack @ w - y_stack))

    elif method == "average":
        y_demeaned = [y - np.mean(y) for y in target_list]
        Y_demeaned = [Y - np.mean(Y, axis=0, keepdims=True) for Y in donor_list]  # demean across time (rows)

        y_avg = sum(y_demeaned) / K                     # (T0,)
        Y_avg = sum(Y_demeaned) / K                     # (T0, N)
        objective = cp.Minimize(cp.sum_squares(Y_avg @ w - y_avg))

    else:
        raise ValueError("Method must be 'concatenated' or 'average'")

    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    fgfg

    return w.value
