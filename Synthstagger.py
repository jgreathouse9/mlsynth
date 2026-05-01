import numpy as np
import cvxpy as cp


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_treatment_info(W):
    T, N = W.shape
    treated = []
    T0 = {}

    for j in range(N):
        idx = np.where(W[:, j] == 1)[0]
        if len(idx) > 0:
            treated.append(j)
            T0[j] = idx[0]

    return treated, T0


def get_donors(W):
    return np.where(W.sum(axis=0) == 0)[0]


# --------------------------------------------------
# Separate SCM (for ν estimation)
# --------------------------------------------------

def fit_single_scm(y, Y0):
    n0 = Y0.shape[1]

    w = cp.Variable(n0)
    alpha = cp.Variable()

    resid = y - Y0 @ w - alpha

    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(resid)),
        [w >= 0, cp.sum(w) == 1]
    )
    prob.solve()

    return np.mean(resid.value**2)


# --------------------------------------------------
# Estimate ν
# --------------------------------------------------

def estimate_nu(Y, W):
    T, N = Y.shape

    treated, T0 = get_treatment_info(W)
    donors = get_donors(W)

    N1 = len(treated)

    q_j = []
    L_j = []

    # --- separate fits ---
    for j in treated:
        t0 = T0[j]
        y_j = Y[:t0, j]
        Y0_j = Y[:t0, donors]

        mse = fit_single_scm(y_j, Y0_j)

        q_j.append(mse)
        L_j.append(t0)

    q_j = np.array(q_j)
    L_j = np.array(L_j)

    # --- pooled fit ---
    T_union = max(T0.values())

    mask = np.zeros((T_union, N1))
    Y_stack = np.zeros((T_union, N1))

    for idx, j in enumerate(treated):
        t0 = T0[j]
        mask[:t0, idx] = 1
        Y_stack[:t0, idx] = Y[:t0, j]

    counts = mask.sum(axis=1)
    valid = counts > 0

    y_bar = np.zeros(T_union)
    y_bar[valid] = (Y_stack[valid].sum(axis=1)) / counts[valid]

    Y0_pool = Y[:T_union, donors]

    w = cp.Variable(len(donors))
    alpha = cp.Variable()

    resid = y_bar[valid] - Y0_pool[valid] @ w - alpha

    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(resid)),
        [w >= 0, cp.sum(w) == 1]
    )
    prob.solve()

    q_pool = np.mean(resid.value**2)

    # --- ν formula ---
    denom = np.mean(np.sqrt(L_j) * q_j)
    num = np.sqrt(T_union) * q_pool

    nu_hat = num / denom if denom > 0 else 1.0
    nu_hat = min(nu_hat, 1.0)

    return nu_hat


# --------------------------------------------------
# Main: matrix partially pooled SCM
# --------------------------------------------------

def partially_pooled_scm(Y, W, lam=1e-3):
    T, N = Y.shape

    treated, T0 = get_treatment_info(W)
    donors = get_donors(W)

    Y1 = Y[:, treated]
    Y0 = Y[:, donors]

    T, N1 = Y1.shape
    N0 = Y0.shape[1]

    # --- estimate ν ---
    nu = estimate_nu(Y, W)

    # --- mask ---
    M = np.zeros((T, N1))
    for idx, j in enumerate(treated):
        M[:T0[j], idx] = 1

    c = M.sum(axis=1)
    c_safe = np.where(c == 0, 1, c)

    # --- variables ---
    Theta = cp.Variable((N0, N1))
    alpha = cp.Variable(N1)

    # --- predictions ---
    Y_hat = Y0 @ Theta + np.ones((T,1)) @ cp.reshape(alpha, (1, N1))

    R = Y1 - Y_hat

    # --------------------------
    # Separate term
    # --------------------------
    sep = cp.sum_squares(cp.multiply(M, R)) / N1

    # --------------------------
    # Pooled term
    # --------------------------
    pooled_terms = []
    for t in range(T):
        if c[t] == 0:
            continue

        r_t = (1 / c_safe[t]) * cp.sum(cp.multiply(M[t, :], R[t, :]))
        pooled_terms.append(r_t**2)

    pool = cp.sum(pooled_terms)

    # --------------------------
    # Objective
    # --------------------------
    ridge = cp.sum_squares(Theta)

    obj = (1 - nu) * sep + nu * pool + lam * ridge

    constraints = [
        Theta >= 0,
        cp.sum(Theta, axis=0) == 1
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()

    # --- outputs ---
    Theta_val = Theta.value
    alpha_val = alpha.value

    # counterfactuals
    Y_hat_full = Y.copy()
    Y_hat_full[:, treated] = Y0 @ Theta_val + np.ones((T,1)) @ alpha_val.reshape(1,-1)

    return {
        "Theta": Theta_val,
        "alpha": alpha_val,
        "Y_hat": Y_hat_full,
        "nu": nu
  }
