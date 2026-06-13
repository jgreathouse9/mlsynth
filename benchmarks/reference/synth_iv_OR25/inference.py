import numpy as np
from scipy.stats import norm

#=====================
# Variance estimation
#=====================
def variance_estimation(U, V, Y_pre_target, Y_post_donors): 
    
    # get dimensions
    T0 = len(Y_pre_target)
    (T1, N) = Y_post_donors.shape 
    k = U.shape[1] 

    # estimator 1 
    df1 = T0 - k 
    residual = (np.eye(T0) - U@U.T) @ Y_pre_target 
    var_hat1 = np.linalg.norm(residual, 2)**2 / df1
    sigma_hat1 = np.sqrt(var_hat1)

    # estimator 2
    df2 = T1 * (N - k) 
    residual = (np.eye(N) - V@V.T) @ Y_post_donors.T 
    var_hat2 = np.linalg.norm(residual, 'fro')**2 / df2 
    sigma_hat2 = np.sqrt(var_hat2)

    # weighted estimator
    df = df1 + df2 
    var_hat = (df2/df)*var_hat1 + (df1/df)*var_hat2
    sigma_hat = np.sqrt(var_hat)

    return (sigma_hat, sigma_hat1, sigma_hat2)

#=====================
# Confidence interval
#=====================
def confidenceInterval(theta_hat, w_hat, sigma_hat, T1, alpha):
    z_critical = norm.ppf(1 - alpha / 2) 
    delta = (z_critical*sigma_hat*np.linalg.norm(w_hat,2)) / np.sqrt(T1)
    return (theta_hat-delta, theta_hat+delta) 

#=====================
# Confidence interval
#=====================
def predictionInterval(theta_hat, w_hat, sigma_hat, T1, alpha):
    z_critical = norm.ppf(1 - alpha / 2) 
    delta = z_critical * sigma_hat * np.sqrt(1 + np.linalg.norm(w_hat,2)**2) / np.sqrt(T1)
    return (theta_hat-delta, theta_hat+delta) 