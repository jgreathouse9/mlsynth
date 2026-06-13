import random
import numpy as np 
import scipy 
import cvxpy as cp

#===================
# Matrix operations
#===================

def donoho_rank(s, ratio): 
    """
    Retain all singular values above optimal threshold as per Donoho & Gavin '14:
    https://arxiv.org/pdf/1305.5870.pdf
    """ 
    omega = 0.56*ratio**3 - 0.95*ratio**2 + 1.43 + 1.82*ratio
    t = omega * np.median(s) 
    rank = max(len(s[s>t]), 1)
    return rank 

def spectral_rank(s, thresh=None):
    if thresh==1.0: 
        rank = len(s)
    else: 
        total_energy = (s**2).cumsum() / (s**2).sum()
        rank = list((total_energy>thresh)).index(True) + 1
    return rank

def HSVT(Z, max_rank=None, thresh=None):
    """
    Perform Hard Singular Value Thresholding (HSVT).

    Parameters:
        Z (np.ndarray): Input matrix of shape (n_samples, n_features).
        max_rank (int, optional): Number of principal components to retain.
        thresh (float, optional): Threshold for spectral decay to determine rank.

    Returns:
        X_hat (np.ndarray): Low-rank approximation of Z.
        rank: rank of X_hat.
    """
    (u, s, v) = np.linalg.svd(Z, full_matrices=False)
    if max_rank is not None: 
        rank = max_rank 
    elif thresh is not None: 
        rank = spectral_rank(s, thresh=thresh)
    else: 
        (m, n) = Z.shape 
        rank = donoho_rank(s, ratio=m/n)
    s_rank = s[:rank]
    u_rank = u[:, :rank]
    v_rank = v[:rank, :] 
    X_hat = (u_rank * s_rank) @ v_rank
    return (X_hat, u_rank, v_rank.T)

# PCR
def PCR(Z, y, max_rank=None, thresh=None):
    # de-noise X
    (u, s, v) = np.linalg.svd(Z, full_matrices=False)
    if max_rank is not None: 
        rank = max_rank 
    if thresh is not None: 
        rank = spectral_rank(s, thresh=thresh)
    else: 
        (n, p) = Z.shape 
        rank = donoho_rank(s, ratio=n/p)
    s_rank = s[:rank]
    u_rank = u[:, :rank]
    v_rank = v[:rank, :] 
    beta = ((v_rank.T/s_rank) @ u_rank.T) @ y
    return (beta, rank)  

# OLS
def OLS(X, y):
    return np.linalg.pinv(X) @ y 

#=================
# SELECTING OMEGA
#=================

#------------------
# Random selection
#------------------

def random_selection(N, k, seed=None):
    """
    Randomly select k column indices (donor units) from Y_pre without modifying the original matrix.

    Parameters:
    - Y_pre (np.ndarray): Pre-treatment outcome matrix of shape (T_pre, N)
    - k (int): Number of columns to select for Omega
    - seed (int or None): Seed for reproducibility

    Returns:
    - omega_indices (list): List of selected column indices
    """
    np.random.seed(seed)

    if k > N:
        raise ValueError(f"Cannot select {k} columns from a matrix with only {N} columns.")

    omega_indices = np.random.choice(N, size=k, replace=False).tolist()
    return omega_indices

#-----------------
# Greedy strategy
#-----------------

def greedy_rank_preserving_selection(Y, k, seed=None):
    """
    Greedy selection of k columns from Y such that each newly added column
    increases the matrix rank, starting from a randomized column order.

    Parameters:
    - Y (np.ndarray): Input matrix of shape (T, N)
    - k (int): Number of columns to select
    - seed (int, optional): Random seed for reproducibility

    Returns:
    - selected_indices (list): Indices of selected columns
    """
    T, N = Y.shape
    selected_indices = []

    # Set random seed
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(N)

    # Initialize with the first non-zero column in the randomized order
    for j in permuted_indices:
        if np.linalg.norm(Y[:, j]) > 1e-10:
            selected_indices.append(j)
            current_matrix = Y[:, [j]]
            break
    else:
        raise ValueError("All columns in the matrix are zero.")

    # Continue greedy selection
    while len(selected_indices) < k:
        for j in permuted_indices:
            if j in selected_indices:
                continue
            candidate_matrix = np.hstack((current_matrix, Y[:, [j]]))
            if np.linalg.matrix_rank(candidate_matrix) > np.linalg.matrix_rank(current_matrix):
                selected_indices.append(j)
                current_matrix = candidate_matrix
                break  # Greedily add the first one that increases rank
        else:
            break  # No further improvement possible

    return selected_indices


def hybrid_qr_leverage_selection(Y, k_total, qr_fraction=0.5, rank=None, seed=None):
    """
    Hybrid selection using QR with column pivoting + leverage score sampling.

    Args:
        Y (np.ndarray): Pre-treatment outcome matrix (T, N).
        k_total (int): Total number of donor units to select.
        qr_fraction (float): Fraction of k_total to select via QR pivoting.
        rank (int or None): Truncation rank for leverage scores (defaults to k_total if None).
        seed (int or None): Random seed for reproducibility.

    Returns:
        selected_indices (List[int]): Combined indices from QR and leverage sampling.
    """
    np.random.seed(seed)
    T, N = Y.shape
    k_qr = int(np.floor(qr_fraction * k_total))
    k_lev = k_total - k_qr

    # --- Step 1: QR pivoting ---
    _, _, pivot_indices = scipy.linalg.qr(Y, pivoting=True)
    qr_selected = list(pivot_indices[:k_qr])

    # --- Step 2: Leverage score sampling ---
    if rank is None:
        rank = k_total
    Yt = Y.T  # shape (N, T)
    Uc, _, _ = scipy.linalg.svd(Yt, full_matrices=False)
    Uc_k = Uc[:, :rank]
    col_leverage_scores = np.sum(Uc_k**2, axis=1)
    col_leverage_scores /= np.sum(col_leverage_scores)

    # Avoid resampling columns already picked by QR
    remaining_indices = list(set(range(N)) - set(qr_selected))
    remaining_probs = col_leverage_scores[remaining_indices]
    remaining_probs /= remaining_probs.sum()

    leverage_selected = np.random.choice(remaining_indices, size=k_lev, replace=False, p=remaining_probs)

    # --- Combine and return sorted list ---
    combined_indices = sorted(qr_selected + list(leverage_selected))
    return combined_indices



# def greedy_rank_preserving_selection(Y, k):
#     """
#     Greedy selection of k columns from Y such that each newly added column
#     increases the matrix rank.

#     Parameters:
#     - Y (np.ndarray): Input matrix of shape (T, N)
#     - k (int): Number of columns to select

#     Returns:
#     - selected_indices (list): Indices of selected columns
#     """
#     T, N = Y.shape
#     selected_indices = []

#     # Initialize with the first non-zero column
#     for j in range(N):
#         if np.linalg.norm(Y[:, j]) > 1e-10:
#             selected_indices.append(j)
#             current_matrix = Y[:, [j]]
#             break
#     else:
#         raise ValueError("All columns in the matrix are zero.")

#     while len(selected_indices) < k:
#         best_col = None
#         for j in range(N):
#             if j in selected_indices:
#                 continue
#             candidate_matrix = np.hstack((current_matrix, Y[:, [j]]))
#             if np.linalg.matrix_rank(candidate_matrix) > np.linalg.matrix_rank(current_matrix):
#                 best_col = j
#                 current_matrix = candidate_matrix
#                 break  # Greedily add the first one that increases rank
#         if best_col is None:
#             break  # No further improvement possible
#         selected_indices.append(best_col)

#     return selected_indices

#-----------------------------------------
# QR Decomposition with leverage sampling
#-----------------------------------------
def leverage_score_selection(Y, k, n_select, seed=None):
    """
    Select columns using QR decomposition and leverage score sampling.

    Args:
        Y (np.ndarray): Pre-treatment outcome matrix of shape (T, N).
        k (int): Truncation rank for QR/SVD decomposition.
        n_select (int): Number of columns to select into Ω.
        seed (int or None): Seed for reproducibility

    Returns:
        selected_indices (List[int]): Indices of selected donor units (columns of Y).
    """
    np.random.seed(seed)
    
    T, N = Y.shape

    # Step 1: Truncated SVD (used to compute leverage scores)
    U, S, Vt = scipy.linalg.svd(Y, full_matrices=False)
    U_k = U[:, :k]  # Top-k left singular vectors

    # Step 2: Compute leverage scores for columns
    leverage_scores = np.sum(U_k**2, axis=1)  # Leverage scores per row (time)
    leverage_scores = leverage_scores / np.sum(leverage_scores)

    # Transpose Y to compute leverage scores for columns
    Yt = Y.T  # Now shape is (N, T)
    Uc, _, _ = scipy.linalg.svd(Yt, full_matrices=False)
    Uc_k = Uc[:, :k]
    col_leverage_scores = np.sum(Uc_k**2, axis=1)
    col_leverage_scores = col_leverage_scores / np.sum(col_leverage_scores)

    # Step 3: Sample columns with probability ∝ leverage scores
    selected_indices = np.random.choice(N, size=n_select, replace=False, p=col_leverage_scores)

    return sorted(selected_indices)

#---------------------------------------
# QR Decomposition with column pivoting
#---------------------------------------
def qr_column_pivoting_selection(Y, k):
    """
    Select donor units using QR decomposition with column pivoting.

    Args:
        Y (np.ndarray): Pre-treatment outcome matrix of shape (T, N).
        k (int): Number of columns to select.

    Returns:
        selected_indices (List[int]): Indices of selected donor units (columns of Y).
    """
    # QR with column pivoting
    Q, R, pivot_indices = scipy.linalg.qr(Y, pivoting=True)
    
    # Select first k pivoted columns
    selected_indices = sorted(pivot_indices[:k])
    
    return selected_indices

#-------
# Lasso 
#-------
from sklearn.linear_model import LassoCV

def lasso_selection(X, y, alpha=None, cv=5):
    """
    Select donor columns using Lasso regression.

    Args:
        Y (np.ndarray): Matrix of shape (T, N), columns are units (including treated).
        target_idx (int): Index of treated unit (whose outcomes we try to reconstruct).
        alpha (float or None): Regularization parameter. If None, selected by CV.
        cv (int): Number of folds for cross-validation (if alpha is None).

    Returns:
        selected_indices (List[int]): Indices of donor units selected by Lasso.
    """
    (T, N) = X.shape
    donor_indices = [i for i in range(N)]

    if alpha is None:
        lasso = LassoCV(cv=cv, fit_intercept=False).fit(X, y)
    else:
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=alpha, fit_intercept=False).fit(X, y)

    # Get non-zero coefficients
    selected_mask = np.abs(lasso.coef_) > 1e-8
    selected_indices = [donor_indices[i] for i, sel in enumerate(selected_mask) if sel]
    print(len(selected_indices))

    return selected_indices

#-------------
# Group Lasso 
#-------------
from sklearn.model_selection import KFold

def group_lasso_selection(Y_donors, Y_target, lambdas=None, K=5, tol=1e-6, verbose=False):
    """
    Select Omega using group lasso with K-fold cross-validation to choose lambda.

    Args:
        Y_target: np.array of shape (T,), target pre-treatment outcomes.
        Y_donors: np.array of shape (T, N), each column is a donor unit's outcomes.
        lambdas: list or np.array of lambda values to try. If None, uses logspace.
        K: int, number of folds.
        tol: float, threshold to select non-zero weights.
        verbose: bool, whether to print CV progress.

    Returns:
        selected_indices: list of selected donor indices.
    """
    T, N = Y_donors.shape
    if lambdas is None:
        lambdas = np.logspace(-4, 1, 20)  # adjust range as needed

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    cv_errors = []

    for lam in lambdas:
        errors = []

        for train_idx, test_idx in kf.split(np.arange(T)):
            Y_train = Y_target[train_idx]
            X_train = Y_donors[train_idx, :]

            Y_test = Y_target[test_idx]
            X_test = Y_donors[test_idx, :]

            W = cp.Variable(N)
            residual = X_train @ W - Y_train
            objective = cp.Minimize(0.5 * cp.sum_squares(residual) + lam * cp.norm1(W))
            prob = cp.Problem(objective)
            prob.solve()

            if W.value is not None:
                pred = X_test @ W.value
                error = np.mean((Y_test - pred)**2)
            else:
                error = np.inf

            errors.append(error)

        mean_cv_error = np.mean(errors)
        cv_errors.append(mean_cv_error)
        if verbose:
            print(f"Lambda: {lam:.5f}, CV Error: {mean_cv_error:.5f}")

    # Choose best lambda
    best_idx = np.argmin(cv_errors)
    best_lambda = lambdas[best_idx]

    # Fit final model on full data using best lambda
    W_final = cp.Variable(N)
    residual_final = Y_donors @ W_final - Y_target
    final_obj = cp.Minimize(0.5 * cp.sum_squares(residual_final) + best_lambda * cp.norm1(W_final))
    final_prob = cp.Problem(final_obj)
    final_prob.solve()

    weights = W_final.value
    selected_indices = [j for j in range(N) if np.abs(weights[j]) > tol]
    print(len(selected_indices))

    return selected_indices


# def select_omega_group_lasso(Y_target, Y_donors, lambda_reg=0.1, tol=1e-6):
#     """
#     Selects donor units (columns of Y_donors) using group lasso.
    
#     Args:
#         Y_target: np.array of shape (T,), the pre-treatment outcomes for unit i.
#         Y_donors: np.array of shape (T, N), each column is a donor's pre-treatment outcomes.
#         lambda_reg: float, regularization strength for group lasso.
#         tol: float, threshold for selecting nonzero groups.
    
#     Returns:
#         omega: list of indices corresponding to selected donor units.
#     """
#     T, N = Y_donors.shape

#     # Create N vector variables (one per unit)
#     W = cp.Variable((N,))  # single coefficient per unit (or use blocks for richer models)

#     # Define objective with group L1 penalty (just L1 since each group is scalar here)
#     residual = Y_donors @ W - Y_target
#     obj = cp.Minimize(0.5 * cp.sum_squares(residual) + lambda_reg * cp.norm1(W))

#     # Solve problem
#     problem = cp.Problem(obj)
#     problem.solve()

#     # Get selected unit indices
#     w_value = W.value
#     omega = [j for j in range(N) if np.abs(w_value[j]) > tol]

#     return omega, w_value

    