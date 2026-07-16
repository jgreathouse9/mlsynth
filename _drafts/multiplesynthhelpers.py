import numpy as np
import cvxpy as cp

# ================================================================
# --- Cohort Filtering Helper
# ================================================================
def select_cohorts_by_postperiods(L_list, Y_treated_pre_list, T_total, k_min):
    """
    Select cohorts that have at least k_min post-treatment periods.

    Parameters
    ----------
    L_list : list of int
        Number of pre-treatment periods for each cohort.
    Y_treated_pre_list : list of np.ndarray
        Pre-treatment outcome matrices for each cohort.
    T_total : int
        Total number of time periods (from donor panel).
    k_min : int
        Minimum required number of post-treatment periods.

    Returns
    -------
    filtered_Y : list of np.ndarray
        Treated pre-outcome matrices for cohorts that meet the post-period requirement.
    filtered_L : list of int
        Corresponding pre-period lengths.
    kept_indices : list of int
        Indices of cohorts that were kept.
    dropped_indices : list of int
        Indices of cohorts that were dropped.
    """
    kept_indices, dropped_indices = [], []

    for i, L_c in enumerate(L_list):
        post_periods = T_total - L_c
        if post_periods >= k_min:
            kept_indices.append(i)
        else:
            dropped_indices.append(i)

    filtered_Y = [Y_treated_pre_list[i] for i in kept_indices]
    filtered_L = [L_list[i] for i in kept_indices]

    return filtered_Y, filtered_L, kept_indices, dropped_indices


def filter_cohorts_with_postperiods(Y_treated_pre_list, L_list, Y_donor_full, k_min):
    T_total = Y_donor_full.shape[0]
    return select_cohorts_by_postperiods(L_list, Y_treated_pre_list, T_total, k_min)




# ================================================================
# --- Optional Mask Construction
# ================================================================
def build_preperiod_mask(Y_treated_pre_list, L_list):
    """
    Construct a binary mask for the full pre-treatment matrix alignment.

    The mask has shape (L_max, sum(N_1c)) and contains 1 where
    a treated unit has data and 0 otherwise.
    """
    L_max = max(L_list)
    N_total = sum(Y.shape[1] for Y in Y_treated_pre_list)
    mask = np.zeros((L_max, N_total))
    start = 0
    for L_c, Y_c in zip(L_list, Y_treated_pre_list):
        n_treated_c = Y_c.shape[1]
        mask[-L_c:, start:start+n_treated_c] = 1
        start += n_treated_c
    return mask


# ================================================================
# --- Imbalance Components
# ================================================================
def compute_masked_imbalance(weights, Y_treated_pre_list, L_list, Y_donor_pre, mask):
    """
    Compute pooled pre-treatment imbalance using the full panel + mask.
    Missing entries (mask=0) are ignored in the squared loss.
    """
    L_max = Y_donor_pre.shape[0]
    N_treated_total = sum(Y.shape[1] for Y in Y_treated_pre_list)

    # Stack treated pre outcomes aligned to L_max
    Y_treated_full = np.zeros((L_max, N_treated_total))
    start = 0
    for L_c, Y_c in zip(L_list, Y_treated_pre_list):
        n_treated_c = Y_c.shape[1]
        Y_treated_full[-L_c:, start:start+n_treated_c] = Y_c
        start += n_treated_c

    donor_pred = Y_donor_pre @ weights
    diff = cp.multiply(mask, (Y_treated_full - donor_pred))
    return cp.sum_squares(diff) / cp.sum(mask)


def compute_truncated_imbalance(weights, Y_treated_pre_list, L_list, Y_donor_pre):
    """
    Compute pooled imbalance under truncation (use L_min).
    """
    L_min = min(L_list)
    Y_treated_trunc = [Y[-L_min:, :] for Y in Y_treated_pre_list]
    Y_treated_trunc = np.hstack(Y_treated_trunc)
    Y_donor_trunc = Y_donor_pre[-L_min:, :]
    diff = Y_treated_trunc - Y_donor_trunc @ weights
    return cp.sum_squares(diff) / (L_min * Y_treated_trunc.shape[1])


def compute_unit_imbalance(weights, Y_treated_pre_list, L_list, Y_donor_pre, mode="truncate", mask=None):
    """
    Compute average of unit-specific imbalances.
    """
    sep_terms = []
    start = 0
    L_max = Y_donor_pre.shape[0]

    for j, (Y_c, L_c) in enumerate(zip(Y_treated_pre_list, L_list)):
        n_treated_c = Y_c.shape[1]
        end = start + n_treated_c
        if mode == "mask":
            for k in range(n_treated_c):
                diff = cp.multiply(mask[-L_c:, start+k],
                                   Y_c[:, k] - (Y_donor_pre[-L_c:, :] @ weights[:, start+k]))
                sep_terms.append(cp.sum_squares(diff) / L_c)
        else:
            for k in range(n_treated_c):
                diff = Y_c[-min(L_list):, k] - (Y_donor_pre[-min(L_list):, :] @ weights[:, start+k])
                sep_terms.append(cp.sum_squares(diff) / min(L_list))
        start = end

    return cp.sum(sep_terms) / len(sep_terms)


# ================================================================
# --- Objective + Constraints
# ================================================================
def build_objective(weights, Y_treated_pre_list, L_list, Y_donor_pre, nu, lambda_reg, mode="truncate", mask=None):
    """
    Build partially pooled SCM objective under truncation or masking.
    """
    if mode == "mask":
        pooled = compute_masked_imbalance(weights, Y_treated_pre_list, L_list, Y_donor_pre, mask)
        unit = compute_unit_imbalance(weights, Y_treated_pre_list, L_list, Y_donor_pre, mode="mask", mask=mask)
    else:
        pooled = compute_truncated_imbalance(weights, Y_treated_pre_list, L_list, Y_donor_pre)
        unit = compute_unit_imbalance(weights, Y_treated_pre_list, L_list, Y_donor_pre, mode="truncate")

    return cp.Minimize(nu * pooled + (1 - nu) * unit + lambda_reg * cp.sum_squares(weights))


def build_constraints(weights):
    """
    Each treated unit's donor weights must sum to one and be nonnegative.
    """
    return [cp.sum(weights[:, i]) == 1 for i in range(weights.shape[1])]


# ================================================================
# --- Main Entry Function
# ================================================================
def partial_scm_weights(
    Y_treated_pre_list,
    L_list,
    Y_donor_pre,
    nu=0.5,
    lambda_reg=1e-4,
    k_min=None,
    Y_donor_full=None,
    mode="truncate"
):
    """
    Estimate partially pooled SCM weights with optional masking or truncation.

    Parameters
    ----------
    mode : {'truncate', 'mask'}
        - 'truncate': use common L_min pre-period length.
        - 'mask': keep all data, ignore missing entries.
    """
    # Filter by post-periods if requested
    if k_min is not None and Y_donor_full is not None:
        Y_treated_pre_list, L_list, kept, dropped = filter_cohorts_with_postperiods(
            Y_treated_pre_list, L_list, Y_donor_full, k_min
        )
    else:
        kept, dropped = list(range(len(Y_treated_pre_list))), []

    # Prepare dimensions
    N_donor = Y_donor_pre.shape[1]
    N_treated_total = sum(Y.shape[1] for Y in Y_treated_pre_list)
    weights = cp.Variable((N_donor, N_treated_total), nonneg=True)

    # Build mask if needed
    mask = build_preperiod_mask(Y_treated_pre_list, L_list) if mode == "mask" else None

    # Build and solve
    objective = build_objective(weights, Y_treated_pre_list, L_list, Y_donor_pre, nu, lambda_reg, mode, mask)
    constraints = build_constraints(weights)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    return weights.value, kept, dropped, mask
