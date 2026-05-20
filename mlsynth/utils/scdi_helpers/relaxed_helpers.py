#relaxed_SC_global_help

import numpy as np
import cvxpy as cp
from typing import Dict


def solve_weights_global(Y, D, lam=0.0):
    T, N = Y.shape

    treated = np.where(D == 1)[0]
    control = np.where(D == 0)[0]

    Y_T = Y[:, treated]
    Y_C = Y[:, control]

    w_T = cp.Variable(len(treated))
    w_C = cp.Variable(len(control))

    treated_mean = Y_T @ w_T
    control_mean = Y_C @ w_C

    objective = cp.Minimize(
        cp.sum_squares(treated_mean - control_mean)
        + lam * (cp.sum_squares(w_T) + cp.sum_squares(w_C))
    )

    constraints = [
        w_T >= 0,
        w_C >= 0,
        cp.sum(w_T) == 1,
        cp.sum(w_C) == 1,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    w = np.zeros(N)
    w[treated] = w_T.value
    w[control] = w_C.value

    return w



def init_D(Y: np.ndarray, K: int) -> np.ndarray:
    """
    Greedy subset selection that minimizes reconstruction error
    of the full panel using selected columns (span-based criterion).
    """
    T, N = Y.shape
    Yc = Y - Y.mean(axis=1, keepdims=True)

    selected = []
    remaining = list(range(N))

    def reconstruction_error(cols):
        if len(cols) == 0:
            return np.inf
        B = Yc[:, cols]
        proj = B @ np.linalg.pinv(B) @ Yc
        return np.linalg.norm(Yc - proj, ord="fro")

    for _ in range(K):
        best_i = None
        best_score = np.inf

        for i in remaining:
            score = reconstruction_error(selected + [i])
            if score < best_score:
                best_score = score
                best_i = i

        selected.append(best_i)
        remaining.remove(best_i)

    D_init = np.zeros(N)
    D_init[selected] = 1
    return D_init


# ============================================================
# STATUS HANDLING
# ============================================================

_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


# ============================================================
# VALIDATION
# ============================================================

def _validate_inputs(Y: np.ndarray, K: int) -> None:
    if Y.ndim != 2:
        raise MlsynthDataError("Y must be a T x N matrix.")

    if Y.shape[0] < 2:
        raise MlsynthConfigError("At least two time periods required.")

    if K <= 0:
        raise MlsynthConfigError("K must be positive.")

    if K > Y.shape[1]:
        raise MlsynthConfigError("K cannot exceed number of units.")


def estimate_lambda(Y: np.ndarray) -> float:
    """Heuristic regularization: average cross-sectional variance."""
    return float(np.mean(np.var(Y, axis=0, ddof=1)))


# ============================================================
# WEIGHT + ASSIGNMENT EXTRACTION
# ============================================================

def _extract_weights(
    D: np.ndarray,
    w: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Unified extraction of treated/control weights."""

    treated_raw = D * w
    control_raw = (1 - D) * w

    treated_weights = treated_raw / (treated_raw.sum() + 1e-12)
    control_weights = control_raw / (control_raw.sum() + 1e-12)

    contrast_weights = (2 * D - 1) * w

    return {
        "treated_weights": treated_weights,
        "control_weights": control_weights,
        "contrast_weights": contrast_weights,
    }


# ============================================================
# OBJECTIVE (ENERGY FUNCTION)
# ============================================================

def energy(Y: np.ndarray, D: np.ndarray, w: np.ndarray, lam: float) -> float:
    treated = (D == 1)
    control = (D == 0)

    mu_T = Y[:, treated] @ w[treated]
    mu_C = Y[:, control] @ w[control]

    return (
        np.mean((mu_T - mu_C) ** 2)
        + lam * (np.sum(w[treated] ** 2) + np.sum(w[control] ** 2))
    )



def rmse_synthetic_gap(Y: np.ndarray, D: np.ndarray, w: np.ndarray) -> float:
    treated = (D == 1)
    control = (D == 0)

    mu_T = Y[:, treated] @ w[treated]
    mu_C = Y[:, control] @ w[control]

    return float(np.sqrt(np.mean((mu_T - mu_C) ** 2)))

# ============================================================
# TEMPERATURE SCHEDULE
# ============================================================

def temperature_schedule(
    it: int,
    Y: np.ndarray,
    delta_history=None,
    T0: float = None,
    decay: float = 0.97,
    target_accept: float = 0.4,
):
    if T0 is None:
        T0 = np.std(Y)

    # Phase 1: warm-up (pure annealing)
    if it < 5:
        return T0 * (decay ** it)

    # Phase 2: adaptive control (main improvement)
    if delta_history is not None and len(delta_history) > 20:
        deltas = np.array(delta_history[-50:])
        uphill = deltas[deltas > 0]

        if len(uphill) > 5:
            scale = np.median(uphill)
            T_adapt = scale / (np.log(1 / target_accept + 1e-8))
            return max(T_adapt, 1e-8)

    # fallback
    return T0 * (decay ** it)


# ============================================================
# ANNEALED SWAP PROPOSAL
# ============================================================

def propose_swap(D: np.ndarray, T: float, max_m: int = 5):
    treated = np.where(D == 1)[0]
    control = np.where(D == 0)[0]

    # map temperature → swap size
    m = int(1 + (T / (T + 1e-8)) * (max_m - 1))
    m = min(m, len(treated), len(control))

    i_idx = np.random.choice(treated, size=m, replace=False)
    j_idx = np.random.choice(control, size=m, replace=False)

    D_new = D.copy()
    D_new[i_idx] = 0
    D_new[j_idx] = 1

    return D_new, (i_idx, j_idx)


# ============================================================
# ANNEALING D-STEP (PURE METROPOLIS MOVE GENERATOR)
# ============================================================
def d_step_annealed(Y, D, w, K, T, lam, n_proposals=None):

    N = len(D)
    n_proposals = N if n_proposals is None else n_proposals

    log = {
        "n_proposals": 0,
        "n_accepted": 0,
        "n_uphill": 0,
        "n_uphill_accepted": 0,
        "delta_history": []
    }

    base_E = energy(Y, D, w, lam)

    best_D = D.copy()
    best_w = w.copy()
    best_E = base_E

    for _ in range(n_proposals):

        D_cand, _ = propose_swap(D,T,max_m=5)

        w_cand = solve_weights_global(Y, D_cand)
        E_cand = energy(Y, D_cand, w_cand, lam)

        delta = E_cand - base_E

        log["n_proposals"] += 1
        log["delta_history"].append(delta)

        if delta > 0:
            log["n_uphill"] += 1

        accept = (delta <= 0) or (np.random.rand() < np.exp(-delta / max(T, 1e-8)))

        if accept:
            D, w = D_cand, w_cand
            base_E = E_cand

            log["n_accepted"] += 1

            if delta > 0:
                log["n_uphill_accepted"] += 1

            if E_cand < best_E:
                best_E = E_cand
                best_D = D.copy()
                best_w = w.copy()

    return best_D, best_w, log




def synthetic_paths(Y: np.ndarray, D: np.ndarray, w: np.ndarray):
    treated = (D == 1)
    control = (D == 0)

    mu_T = Y[:, treated] @ w[treated]
    mu_C = Y[:, control] @ w[control]

    return mu_T, mu_C, (mu_T - mu_C)

