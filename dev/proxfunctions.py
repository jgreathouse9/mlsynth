import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# -------------------------------
# Data Simulation
# -------------------------------

def simulate_latent_factor_data(T=150, J=90, r=2, rho=0.9, treatment_time=120, tau=5.0, seed=1292):
    if seed is not None:
        np.random.seed(seed)

    F = np.zeros((T, r))
    F[0] = np.random.normal(size=r)
    for t in range(1, T):
        F[t] = rho * F[t - 1] + np.random.normal(size=r)

    Lambda = np.random.normal(size=(J + 1, r))  # includes treated unit
    noise = np.random.normal(scale=0.5, size=(T, J + 1))
    Y = F @ Lambda.T + noise + 60
    y1 = Y[:, 0].copy()
    Y0 = Y[:, 1:]

    y1[treatment_time:] += tau
    post_periods = list(range(treatment_time, T))
    return y1, Y0, post_periods

# -------------------------------
# Step 1: Alignment Coefficients
# -------------------------------

def compute_alignment_coefficients(y1_pre, Y0_pre):
    y1_demeaned = y1_pre - np.mean(y1_pre)
    Y0_demeaned = Y0_pre - np.mean(Y0_pre, axis=0)

    numerators = Y0_demeaned.T @ y1_demeaned
    denominators = np.sum(Y0_demeaned ** 2, axis=0)
    theta_hat = numerators / (denominators + 1e-8)

    Y_theta = Y0_pre * theta_hat
    return theta_hat, Y_theta

# -------------------------------
# Step 2: Estimate Noise Variance
# -------------------------------

def estimate_noise_variance(y1_pre, Y0_pre):
    T0 = len(y1_pre)
    Q = np.eye(T0) - np.ones((T0, T0)) / T0

    G = Y0_pre.T @ Q @ Y0_pre
    diag_G = np.diag(G)
    Z = Y0_pre @ np.diag(1 / (diag_G + 1e-8)) @ Y0_pre.T

    projection = Z @ Q @ y1_pre
    residual = Q @ y1_pre - Q @ projection
    sigma2 = np.linalg.norm(residual) ** 2
    return sigma2

def SRC_opt(y1_pre, Y0_pre, Y0_post, Y_theta, theta_hat, sigma2, post_periods):
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
    prob.solve(solver=cp.CLARABEL)

    w_hat = w.value

    y1_bar = np.mean(y1_pre)
    yj_bar = np.mean(Y0_pre, axis=0)
    y1_hat_pre = Y0_pre @ (w_hat * theta_hat)
    y1_hat_post = y1_bar + (Y0_post - yj_bar) @ (w_hat * theta_hat)

    return y1_hat_pre, y1_hat_post, w_hat, theta_hat

# -------------------------------
# SRC Estimator
# -------------------------------

def SRC(y1, Y0, post_periods, weight_constraint='simplex'):
    T, J = Y0.shape
    T0 = max(set(range(T)) - set(post_periods))
    y1_pre = y1[:T0]
    Y0_pre = Y0[:T0]

    theta_hat, Y_theta = compute_alignment_coefficients(y1_pre, Y0_pre)
    sigma2 = estimate_noise_variance(y1_pre, Y0_pre)
    Y0_post = Y0[post_periods]

    y1_hat_pre, y1_hat_post, w_hat, theta_hat = SRC_opt(
        y1_pre, Y0_pre, Y0_post, Y_theta, theta_hat, sigma2, post_periods
    )
    # In-sample prediction
    y1_hat_pre = Y0_pre @ (w_hat * theta_hat)
    return y1_hat_pre, y1_hat_post, w_hat, theta_hat

# -------------------------------
# Run + Plot
# -------------------------------

y1, Y0, post_periods = simulate_latent_factor_data(seed=42)
y1_hat_pre, y1_hat_post, w_hat, theta_hat = SRC(y1, Y0, post_periods)

# Combine predictions
T0 = post_periods[0]
y1_hat_full = np.concatenate([y1_hat_pre, y1_hat_post])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y1, label='Observed Treated', linewidth=2)
plt.plot(y1_hat_full, label='SRC Prediction', linestyle='--')
plt.axvline(T0, color='k', linestyle=':', label='Treatment Begins')
plt.title("Synthetic Regularized Control: Observed vs. Predicted")
plt.xlabel("Time")
plt.ylabel("Outcome")
plt.legend()
plt.tight_layout()
plt.show()
