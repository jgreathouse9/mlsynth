import numpy as np
from scipy.stats import norm
from mlsynth.utils.resultutils import effects


def universal_rank(s, ratio):
    omega = 0.56 * ratio**3 - 0.95 * ratio**2 + 1.43 + 1.82 * ratio
    t = omega * np.median(s)
    rank = max(len(s[s > t]), 1)
    return rank


def spectral_rank(s, t=0.95):
    if t == 1.0:
        rank = len(s)
    else:
        total_energy = (s**2).cumsum() / (s**2).sum()
        rank = list((total_energy > t)).index(True) + 1
    return rank


def svt(X, max_rank=None, t=None):
    """
    Perform Singular Value Thresholding (SVT) on a matrix.

    Args:
        X (numpy.ndarray): Input matrix.
        max_rank (int, optional): Maximum rank for truncation.
        t (float, optional): Threshold for determining rank.

    Returns:
        Y0_rank (numpy.ndarray): Low-rank approximation of the input matrix.
        u_rank (numpy.ndarray): Truncated left singular vectors (U).
        s_rank (numpy.ndarray): Truncated singular values (diagonal).
        v_rank (numpy.ndarray): Truncated right singular vectors (V^T).
    """
    (n1, n2) = X.shape
    (u, s, v) = np.linalg.svd(X, full_matrices=False)
    ratio = min(n1, n2) / max(n1, n2)
    rank = universal_rank(s, ratio=ratio)  # Assumes a spectral_rank method

    # Truncate SVD components
    s_rank = s[:rank]
    u_rank = u[:, :rank]
    v_rank = v[:rank, :]

    # Low-rank approximation
    Y0_rank = np.dot(u_rank, np.dot(np.diag(s_rank), v_rank))

    # Optional projections (if needed for other calculations)
    Hu = u_rank @ u_rank.T
    Hv = v_rank.T @ v_rank
    Hu_perp = np.eye(n1) - Hu
    Hv_perp = np.eye(n2) - Hv

    # Return relevant components
    return Y0_rank, n2, u_rank, s_rank, v_rank


def shrink(X, tau):
    """
    Gets our optimal singular values (shrinks) to optimal number,
    'soft impute'
    """
    # 'Soft' Impute
    Y = np.abs(X) - tau
    return np.sign(X) * np.maximum(Y, np.zeros_like(Y))


def SVT(X, tau):
    """
    Does the Singular Value thresholding (the matrix operation)
    """
    # Does Singular Value Thresholding
    U, S, VT = np.linalg.svd(X, full_matrices=0)
    out = U @ np.diag(shrink(S, tau)) @ VT
    return out


def RPCA(X):
    """
    Gets our low-rank representation into
    Low rank structure, common patterns and noise
    """

    n1, n2 = X.shape
    mu = n1 * n2 / (4 * np.sum(np.abs(X.reshape(-1))))

    thresh = 10 ** (-9) * np.linalg.norm(X)

    S = np.zeros_like(X)  # Individual effects
    Y = np.zeros_like(X)  # Encourages convergence (updates)
    L = np.zeros_like(X)  # Low Rank structure

    count = 0
    lambd = 1 / np.sqrt(np.maximum(n1, n2))
    while (np.linalg.norm(X - L - S) > thresh) and (count < 1000):
        L = SVT(X - S + (1 / mu) * Y, 1 / mu)
        S = shrink(X - L + (1 / mu) * Y, lambd / mu)
        Y = Y + mu * (X - L - S)
        count += 1
    return L


def debias(M, tau, Z, l):
    u, s, vh = svd_fast(M)
    r = np.sum(s / np.cumsum(s) >= 1e-6)
    u = u[:, :r]
    vh = vh[:r, :]

    PTperpZ = np.zeros_like(Z)
    for k in np.arange(Z.shape[2]):
        PTperpZ[:, :, k] = remove_tangent_space_component(u, vh, Z[:, :, k])

    D = np.zeros((Z.shape[2], Z.shape[2]))
    for k in np.arange(Z.shape[2]):
        for m in np.arange(k, Z.shape[2]):
            D[k, m] = np.sum(PTperpZ[:, :, k] * PTperpZ[:, :, m])
            D[m, k] = D[k, m]

    Delta = np.array(
        [l * np.sum(Z[:, :, k] * (u.dot(vh))) for k in range(Z.shape[2])]
    )

    tau_delta = np.linalg.pinv(D) @ Delta
    tau_debias = tau - tau_delta

    PTZ = Z - PTperpZ
    M_debias = (
        M + l * u.dot(vh) + np.sum(PTZ * tau_delta.reshape(1, 1, -1), axis=2)
    )
    return M_debias, tau_debias


def svd_fast(M):
    is_swap = False
    if M.shape[0] > M.shape[1]:
        is_swap = True
        M = M.T

    A = M @ M.T  # this will speed up the calculation when M is asymmetric
    u, ss, uh = np.linalg.svd(A, full_matrices=False)
    ss[ss < 1e-7] = 0
    s = np.sqrt(ss)
    sinv = 1.0 / (s + 1e-7 * (s < 1e-7))
    vh = sinv.reshape(M.shape[0], 1) * (uh @ M)

    if is_swap:
        return vh.T, s, u.T
    else:
        return u, s, vh


## least-squares solved via single SVD
def SVD(M, r):
    """
    input matrix M, approximating with rank r
    """
    u, s, vh = svd_fast(M)
    s[r:] = 0
    return (u * s).dot(vh)


def SVD_soft(X, l):
    u, s, vh = svd_fast(X)
    s_threshold = np.maximum(0, s - l)
    return (u * s_threshold).dot(vh)


def DC_PR_with_l(O, Z, l, initial_tau=None, eps=1e-6):
    """
    De-biased Convex Panel Regression with the regularizer l.

    Parameters
    -------------
    O : 2d float numpy array
        Observation matrix.
    Z : a list of 2d float numpy array or a single 2d/3d float numpy array
        Intervention matrices. If Z is a list, then each element of the list is a 2d numpy array. If Z is a single 2d numpy array, then Z is a single intervention matrix. If Z is a 3d numpy array, then Z is a collection of intervention matrices with the last dimension being the index of interventions.
    l : float
        Regularizer for the nuclear norm.
    intial_tau : (num_treat,) float numpy array
        Initial value(s) for tau.
    eps : float
        Convergence threshold.

    Returns
    -------------
    M : 2d float numpy array
        Estimated matrix.
    tau : (num_treat,) float numpy array
        Estimated treatment effects.
    """
    Z = transform_to_3D(Z)  ## Z is (n1 x n2 x num_treat) numpy array
    if initial_tau is None:
        tau = np.zeros(Z.shape[2])
    else:
        tau = initial_tau

    small_index, X, Xinv = prepare_OLS(Z)

    for T in range(2000):
        #### SVD to find low-rank M
        M = SVD_soft(O - np.tensordot(Z, tau, axes=([2], [0])), l)
        #### OLS to get tau
        y = (O - M)[small_index]  # select non-zero entries
        tau_new = Xinv @ (X.T @ y)
        #### Check convergence
        if np.linalg.norm(tau_new - tau) < eps * np.linalg.norm(tau):
            return M, tau
        tau = tau_new
    return M, tau


def non_convex_PR(O, Z, r, initial_tau=None, eps=1e-6):
    """
    Non-Convex Panel Regression with the rank r

    Parameters
    -------------
    O : 2d float numpy array
        Observation matrix.
    Z : a list of 2d float numpy array or a single 2d/3d float numpy array
        Intervention matrices. If Z is a list, then each element of the list is a 2d numpy array. If Z is a single 2d numpy array, then Z is a single intervention matrix. If Z is a 3d numpy array, then Z is a collection of intervention matrices with the last dimension being the index of interventions.
    r : int
        rank constraint for the baseline matrix.
    intial_tau : (num_treat,) float numpy array
        Initial value(s) for tau.
    eps : float
        Convergence threshold.

    Returns
    -------------
    M : 2d float numpy array
        Estimated baseline matrix.
    tau : (num_treat,) float numpy array
        Estimated treatment effects.
    """
    Z = transform_to_3D(Z)  ## Z is (n1 x n2 x num_treat) numpy array
    if initial_tau is None:
        tau = np.zeros(Z.shape[2])
    else:
        tau = initial_tau

    small_index, X, Xinv = prepare_OLS(Z)

    for T in range(2000):
        #### SVD to find low-rank M
        M = SVD(
            O - np.tensordot(Z, tau, axes=([2], [0])), r
        )  # hard truncation
        #### OLS to get tau
        y = (O - M)[small_index]  # select non-zero entries
        tau_new = Xinv @ (X.T @ y)
        #### Check convergence
        if np.linalg.norm(tau_new - tau) < eps * np.linalg.norm(tau):
            return M, tau
        tau = tau_new
    return M, tau


def panel_regression_CI(M, Z, E):
    """
    Compute the confidence interval of taus using the first-order approximation.

    Parameters:
    -------------
    M: the (approximate) baseline matrix
    Z: a list of intervention matrices
    E: the (approximate) noise matrix

    Returns
    -----------
    CI: a kxk matrix that charaterizes the asymptotic covariance matrix of treatment estimation from non-convex panel regression,
        where k is the number of treatments
    """
    u, s, vh = svd_fast(M)
    r = np.sum(s / np.cumsum(s) >= 1e-6)
    u = u[:, :r]
    vh = vh[:r, :]

    X = np.zeros((Z.shape[0] * Z.shape[1], Z.shape[2]))
    for k in np.arange(Z.shape[2]):
        X[:, k] = remove_tangent_space_component(u, vh, Z[:, :, k]).reshape(-1)

    A = np.linalg.inv(X.T @ X) @ X.T
    CI = (A * np.reshape(E**2, -1)) @ A.T
    return CI


def remove_tangent_space_component(u, vh, Z):
    """
    Remove the projection of Z (a single treatment) onto the tangent space of M in memory-aware manner
    """

    # We conduct some checks for extremely wide or extremely long matrices, which may result in OOM errors with
    # naïve operation sequencing.  If BOTH dimensions are extremely large, there may still be an OOM error, but this
    # case is quite rare.
    treatment_matrix_shape = Z.shape
    if max(treatment_matrix_shape) > 1e4:

        if treatment_matrix_shape[0] > treatment_matrix_shape[1]:
            first_factor = Z - u.dot(u.T.dot(Z))
            second_factor = np.eye(vh.shape[1]) - vh.T.dot(vh)
        else:
            first_factor = np.eye(u.shape[0]) - u.dot(u.T)
            second_factor = Z - (Z.dot(vh.T)).dot(vh)

        PTperpZ = first_factor.dot(second_factor)

    else:
        PTperpZ = (
            (np.eye(u.shape[0]) - u.dot(u.T))
            .dot(Z)
            .dot(np.eye(vh.shape[1]) - vh.T.dot(vh))
        )

    return PTperpZ


def transform_to_3D(Z):
    """
    Z is a list of 2D numpy arrays or a single 2D/3D numpy array
    convert Z to a 3D numpy array with the last dimension being the index of interventions
    """
    if isinstance(Z, list):  # if Z is a list of numpy arrays
        Z = np.stack(Z, axis=2)
    elif Z.ndim == 2:  # if a single Z
        Z = Z.reshape(Z.shape[0], Z.shape[1], 1)
    return Z.astype(float)


def prepare_OLS(Z):
    ### Select non-zero entries for OLS (optmizing sparsity of Zs)
    small_index = np.sum(np.abs(Z) > 1e-9, axis=2) > 0
    X = Z[small_index, :].astype(float)  # small X
    ## X.shape = (#non_zero entries of Zs, num_treat)
    Xinv = np.linalg.inv(X.T @ X)
    return small_index, X, Xinv


def solve_tau(O, Z):
    small_index, X, Xinv = prepare_OLS(Z)
    y = O[small_index]  # select non-zero entries
    tau = Xinv @ (X.T @ y)
    return tau


def DC_PR_auto_rank(O, Z, spectrum_cut=0.002, method="auto"):
    treatedrow = [i for i, row in enumerate(Z) if 1 in row][0]
    s = np.linalg.svd(O, full_matrices=False, compute_uv=False)
    suggest_r = np.sum(np.cumsum(s**2) / np.sum(s**2) <= 1 - spectrum_cut)
    M, tau, std = DC_PR_with_suggested_rank(
        O, Z, suggest_r=suggest_r, method="convex"
    )
    y_DCR = M[treatedrow, :]
    observed = O[treatedrow, :]

    t1 = Z[treatedrow, :].shape[0] - np.count_nonzero(Z[treatedrow, :])

    preRMSE = round(np.std(observed[:t1] - y_DCR[:t1]), 3)
    Debiased_Dict = {
        "Vectors": {
            "Treated Unit": np.round(observed.reshape(-1, 1), 3),
            "Counterfactual": np.round(y_DCR.reshape(-1, 1), 3),
        },
        "Effects": {
            "ATT": round(tau, 3),
            "Percent ATT": round(100 * tau / np.mean(y_DCR[t1:]), 3),
        },
        "CIs": std,
        "RMSE": round(preRMSE, 3),
    }
    return Debiased_Dict


def DC_PR_with_suggested_rank(O, Z, suggest_r=1, method="auto"):
    """
    De-biased Convex Panel Regression with the suggested rank. Gradually decrease the nuclear-norm regularizer l until the rank of the next-iterated estimator exceeds r.

    :param O: observation matrix
    :param Z: intervention matrix

    """
    treatedrow = [i for i, row in enumerate(Z) if 1 in row][0]
    Z = transform_to_3D(Z)  ## Z is (n1 x n2 x num_treat) numpy array
    ## determine pre_tau
    pre_tau = solve_tau(O, Z)

    if method == "convex" or method == "auto":
        ## determine l
        coef = 1.1
        _, s, _ = svd_fast(O - np.tensordot(Z, pre_tau, axes=([2], [0])))
        l = s[1] * coef
        ##inital pre_M and pre_tau for current l
        pre_M, pre_tau = DC_PR_with_l(O, Z, l, initial_tau=pre_tau)
        l = l / coef
        while True:
            M, tau = DC_PR_with_l(O, Z, l, initial_tau=pre_tau)
            if np.linalg.matrix_rank(M) > suggest_r:
                M_debias, tau_debias = debias(pre_M, pre_tau, Z, l * coef)
                M = SVD(M_debias, suggest_r)
                tau = tau_debias
                break
            pre_M = M
            pre_tau = tau
            l = l / coef
    if method == "non-convex":
        M, tau = non_convex_PR(O, Z, suggest_r, initial_tau=pre_tau)

    if method == "auto":
        M1, tau1 = non_convex_PR(O, Z, suggest_r, initial_tau=solve_tau(O, Z))
        if np.linalg.matrix_rank(M) != suggest_r or np.linalg.norm(
            O - M - np.tensordot(Z, tau, axes=([2], [0]))
        ) > np.linalg.norm(O - M1 - np.tensordot(Z, tau1, axes=([2], [0]))):
            M = M1
            tau = tau1

    CI = panel_regression_CI(
        M, Z, O - M - np.tensordot(Z, tau, axes=([2], [0]))
    )
    SE = np.sqrt(np.diag(CI))

    # Compute the 95% CI
    alpha = 0.05
    z_score = norm.ppf(1 - alpha / 2)

    # Construct the CI_dict with single numbers
    CI_dict = {
        "Lower Bound": round((tau - z_score * SE)[0], 3),
        "Upper Bound": round((tau + z_score * SE)[-1], 3),
        "SE": SE,
        "tstat": z_score,
    }

    y_DCR = M[treatedrow, :]
    observed = O[treatedrow, :]

    t1 = Z[treatedrow, :].shape[0] - np.count_nonzero(Z[treatedrow, :])
    t2 = len(y_DCR) - t1

    attdict, fitdict, Vectors = effects.calculate(observed, y_DCR, t1, t2)

    return {
        "Effects": attdict,
        "Fit": fitdict,
        "Vectors": Vectors,
        "Inference": CI_dict,
    }


def RPCA_HQF(M_noise, rak, maxiter, ip, lam_1):
    """
    Robust PCA via Non-convex Half-quadratic Regularization.

    Parameters:
        M_noise (ndarray): m x n observed matrix corrupted by noise.
        rak (int): Rank of the object matrix.
        maxiter (int): Maximum number of iterations.
        ip (float): Controls the sparse.
        lam_1 (float): Regularization parameter.

    Returns:
        Out_X (ndarray): Denoised matrix.
        RMSE (list): Root mean square error at each iteration.
        peaksnr (list): Placeholder for peak signal-to-noise ratio (not implemented).
        U (ndarray): Left singular matrix.
        V (ndarray): Right singular matrix.
        RRMSE (list): Placeholder for relative root mean square error (not implemented).
        N (ndarray): Sparse noise matrix.
    """
    m, n = M_noise.shape
    np.random.seed(42)
    U = np.random.rand(m, rak)
    Ip = 3
    in_count = 0
    RMSE = []
    peaksnr = []
    RRMSE = []
    lam_2 = lam_1

    # PF initialization
    for _ in range(Ip):
        PINV_U = np.linalg.pinv(U)
        V = PINV_U @ M_noise
        PIMV_V = np.linalg.pinv(V)
        U = M_noise @ PIMV_V

    X = U @ V
    T = M_noise - X
    t_m_n = T.flatten()
    scale = 10 * 1.4815 * np.median(np.abs(t_m_n - np.median(t_m_n)))
    sigma = np.ones((m, n)) * scale

    ONE_1 = np.ones((m, n))
    ONE_1[np.abs(T - np.median(t_m_n)) - sigma < 0] = 0
    N = T * ONE_1

    U_p = U.copy()
    V_p = V.copy()

    for iter in range(maxiter):
        D = M_noise - N
        U = (D @ V.T - lam_1 * U_p) @ np.linalg.inv(
            V @ V.T - lam_1 * np.eye(rak)
        )
        V = np.linalg.inv(U.T @ U - lam_2 * np.eye(rak)) @ (
            U.T @ D - lam_2 * V_p
        )

        U_p = U.copy()
        V_p = V.copy()
        X = U @ V

        T = M_noise - X
        t_m_n = T.flatten()
        scale = min(
            scale, ip * 1.4815 * np.median(np.abs(t_m_n - np.median(t_m_n)))
        )

        sigma = np.ones((m, n)) * scale
        ONE_1 = np.ones((m, n))
        ONE_1[np.abs(T - np.median(t_m_n)) - sigma < 0] = 0
        N = T * ONE_1

        RMSE.append(np.linalg.norm(M_noise - X, "fro") / np.sqrt(m * n))

        if iter != 0:
            step_MSE = RMSE[iter - 1] - RMSE[iter]
            if step_MSE < 1e-6:
                in_count += 1
            if in_count > 1:
                break

    Out_X = X
    return Out_X


def demean_matrix(matrix):
    return matrix - np.mean(matrix, axis=0)


def nbpiid(x, kmax, jj, demean_flag, m_N, m_T):
    T = x.shape[0]
    N = x.shape[1]
    NT = N * T
    NT1 = N + T
    CT = np.zeros(kmax)
    ii = np.arange(1, kmax + 1)

    if jj == 1:
        CT = np.log(NT / NT1) * ii * NT1 / NT
    elif jj == 10:
        CT = ((N + m_N) * (T + m_T) / NT) * np.log(NT / NT1) * ii * NT1 / NT
    elif jj == 11:
        CT = (
            (N * T / NT)
            * (T / (4 * np.log(np.log(T))))
            * np.log(NT / NT1)
            * ii
            * NT1
            / NT
        )
    elif jj == 2:
        CT = (NT1 / NT) * np.log(min(N, T)) * ii
    elif jj == 3:
        GCT = min(N, T)
        CT = ii * np.log(GCT) / GCT
    elif jj == 4:
        CT = 2 * ii / T
    elif jj == 5:
        CT = np.log(T) * ii / T
    elif jj == 6:
        CT = 2 * ii * NT1 / NT
    elif jj == 7:
        CT = np.log(NT) * ii * NT1 / NT
    if demean_flag == 2:
        X = standardize(x)
    elif demean_flag == 1:
        X = demean_matrix(x)
    else:
        X = x
    IC1 = np.zeros((len(CT), kmax + 1))
    Sigma = np.zeros(kmax + 1)
    XX = np.dot(X, X.T)
    eigval, Fhat0 = np.linalg.eigh(XX)
    Fhat0 = Fhat0[:, ::-1]

    for i in range(kmax, 0, -1):
        Fhat = Fhat0[:, :i]
        lambda_hat = np.dot(Fhat.T, X)
        chat = np.dot(Fhat, lambda_hat)
        ehat = X - chat
        Sigma[i - 1] = np.mean(np.sum(ehat * ehat / T, axis=0))
        IC1[:, i - 1] = Sigma[i - 1] + CT[i - 1] * Sigma[kmax - 1]
    Sigma[kmax] = np.mean(np.sum(X * X / T, axis=0))
    IC1[:, kmax] = Sigma[kmax]

    ic1 = np.argmin(IC1, axis=1)
    ic1 = ic1 * (ic1 <= kmax)
    Fhat = Fhat0[:, : ic1[0]]
    lambda_hat = np.dot(Fhat.T, X)
    chat = np.dot(Fhat, lambda_hat)

    return ic1[0], chat, Fhat


def standardize(matrix):
    return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
