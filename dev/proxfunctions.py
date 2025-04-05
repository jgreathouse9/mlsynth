import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mlsynth.mlsynth import dataprep
import pandas as pd

def SRC(target, donors, n_periods, T0):
    """
    Perform synthetic control estimation and return the donor weights and out-of-sample estimates.

    Args:
        target (numpy.ndarray): Target pre-treatment vector (n_periods,)
        donors (numpy.ndarray): Donor matrix with outcomes for potential control units (n_periods, n_donors)
        n_periods (int): The number of time periods (including pre-treatment and treatment periods)
        T0 (int): The number of pre-treatment periods

    Returns:
        numpy.ndarray: Optimal donor weights (n_donors,)
        numpy.ndarray: Counterfactual outcome including level difference (both pre-treatment and post-treatment)
    """
    # Slice target and donor matrices to include only the first T0 periods
    Y0 = donors.copy()
    target = target[:T0]  # Slice the target vector
    donors = donors[:T0, :]  # Slice the donor matrix

    # Demeaning the target and donor matrix
    Qy_1 = np.subtract(target, target.mean(axis=0))  # Demeaned target
    QY_0 = np.subtract(donors, donors.mean(axis=0))  # Demeaned donor matrix

    # Inverting the diagonal (this part remains unchanged)
    inv_diag = np.diag(1.0 / np.diag(QY_0.T @ QY_0))

    # In-sample fit (counterfactual outcomes in demeaned space)
    y_1_hat = QY_0 @ inv_diag @ QY_0.T @ Qy_1  # Regressing

    # Residuals from demeaned estimates above (sigma_hat_squared)
    sigma_hat_squared = np.sum(np.subtract(Qy_1, y_1_hat) ** 2) / T0

    # Define the weights variable for the optimization
    n_donors = donors.shape[1]
    w = cp.Variable(n_donors, nonneg=True)

    # Regularization term (penalty on the weights)
    penalty_term = 2 * sigma_hat_squared * np.ones(n_donors)

    # Define the objective function (penalized least squares)
    objective = cp.Minimize(cp.norm(target - donors @ w, p=2) + penalty_term.T @ cp.abs(w))

    # Constraints (weights sum to 1)
    constraints = [cp.sum(w) == 1]

    # Set up and solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    # Calculate counterfactual for both pre- and post-treatment periods, including level difference
    counterfactual = np.concatenate([
        (donors @ w.value) + (np.mean(target) - np.mean(donors @ w.value)),  # Pre-treatment
        (Y0[T0:, :] @ w.value) + (np.mean(target) - np.mean(donors @ w.value))  # Post-treatment
    ])

    # Return the optimal donor weights and the counterfactual (pre and post-treatment)
    return w.value, counterfactual

# Example usagea
url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
data = pd.read_csv(url)

prepped = dataprep(data, data.columns[0], data.columns[1], data.columns[2], data.columns[-1])

y1 = prepped["y"]
Y0 = prepped["donor_matrix"]
T0 = prepped["pre_periods"]

optimal_weights, counterfactual = SRC(y1, Y0, len(y1), T0)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y1, label="California", color='black')
plt.plot(counterfactual, label="Synthetic California", color='blue')
plt.legend()
plt.title("Prop 99")
plt.show()
