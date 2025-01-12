# Estimates Synthetic Regression Control sans covariates: https://doi.org/10.48550/arXiv.2306.02584


import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

def SRC(y1, Y0, theta_hat, sigma_squared):
    """
    Parameters:
    y1 (numpy.ndarray): Pre-treatment outcomes for the treated unit, shape (T0, 1).
    Y0 (numpy.ndarray): Pre-treatment outcomes for control units, shape (T0, J).
    theta_hat (numpy.ndarray): Regression coefficients for control units, shape (J,).
    sigma_squared (float): Variance estimate from the unit regressions.

    Returns:
    numpy.ndarray: Optimal weights (w_opt), shape (J,).
    """
    J = Y0.shape[1]  # Number of control units

    # Step 1: Define the CVXPY variable for weights
    w = cp.Variable(J, nonneg=True)
    intercept = cp.Variable(1)

    # Step 2: Define the objective function
    Y0_scaled = Y0 * theta_hat  # Element-wise multiplication (T0, J)
    objective = cp.Minimize(cp.norm(y1.flatten() -  (Y0 * theta_hat) @ w - intercept, 2)**2 + (2 * sigma_squared * cp.sum(w)))

    # Step 3: Define the constraints and solve the problem
    constraints = [cp.sum(w) == 1]  # Sum of weights equals 1
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Step 4: Return the optimal weights
    return w.value, intercept.value


def compute_theta_and_sigma(y1, Y0):
    """
    Compute regression coefficients (theta) and variance estimate (sigma^2)
    for the Synthetic Regressing Control (SRC) method.

    Parameters:
    y1 (numpy.ndarray): Pre-treatment outcomes for the treated unit, shape (T0, 1).
    Y0 (numpy.ndarray): Pre-treatment outcomes for control units, shape (T0, J).

    Returns:
    tuple: (theta_hat, sigma_squared)
        - theta_hat (numpy.ndarray): Regression coefficients, shape (J,).
        - sigma_squared (float): Variance estimate.
    """
    T0, J = Y0.shape

    # Step 1: Run OLS for each control unit to estimate theta_j
    theta_hat = np.zeros(J)
    y1_centered = y1 - np.mean(y1)
    for j in range(J):
        y_j = Y0[:, j]
        y_j_centered = y_j - np.mean(y_j)
        theta_hat[j] = np.dot(y_j_centered.T, y1_centered[:, 0]) / np.dot(y_j_centered.T, y_j_centered)

    # Step 2: Compute variance estimate sigma^2
    Q = np.eye(T0) - (1 / T0) * np.ones((T0, T0))  # Centering matrix
    sigma_squared = np.linalg.norm(Q @ y1 - Q @ Y0 @ np.diag(theta_hat) @ np.ones(J)) ** 2

    return theta_hat, sigma_squared

df = pd.read_csv('https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv')


Ywide = df.pivot(index='year', columns='regionname', values='gdpcap')

y = Ywide['Basque'].to_numpy()
Y0frame = Ywide.drop(Ywide.columns[3], axis=1)

Y0 = Y0frame.to_numpy()

pre = 19

theta_hat, sigma_squared = compute_theta_and_sigma( y[:pre].reshape(-1,1), Y0[:pre])

# Call the SRC method
w_opt, intercept= SRC(y[:pre].flatten(), Y0[:pre], theta_hat, sigma_squared)

print("Optimal weights (rounded):", np.round(w_opt, 3))
y_hat = np.dot(Y0, w_opt) + intercept  # Predicted outcomes for the pre-treatment period

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1,len(y)+1), y, label='Basque', marker='o')
plt.plot(np.arange(1,len(y)+1), y_hat, label='SRC Basque', marker='D')

# Step 3: Add labels, legend, and title
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('Observed vs Predicted GDP per Capita (Pre-treatment Period)')
plt.legend()
plt.grid(True)
plt.show()


