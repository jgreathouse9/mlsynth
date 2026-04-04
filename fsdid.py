import numpy as np
import cvxpy as cp
from typing import List, Tuple, Optional, Dict




def conformal_interval(effects: np.ndarray, B_idx: np.ndarray, post_idx: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Compute conformal prediction intervals based on blank/pre-treatment periods,
    and measure coverage of post-treatment effects.
    """
    # 1. Residuals from blank/pre-treatment periods
    residuals = effects[B_idx]

    # 2. Quantile of absolute residuals
    q = np.quantile(np.abs(residuals), 1 - alpha)

    # 3. Interval around 0 (null effect)
    lower = -q
    upper = q

    # 4. Coverage: fraction of post-period effects within this interval
    coverage = np.mean((effects[post_idx] >= lower) & (effects[post_idx] <= upper))

    # 5. Width for reporting
    width = upper - lower

    return {'lower': lower, 'upper': upper, 'width': width, 'coverage': coverage}




# -------------------- SIMPLEX PROJECTION --------------------
def project_to_simplex(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


# -------------------- FAST SIMPLEX QP --------------------
def solve_qp_simplex(Q: np.ndarray, max_iter: int = 200, lr: float = 0.01) -> np.ndarray:
    m = Q.shape[0]
    w = np.ones(m) / m
    for _ in range(max_iter):
        grad = 2 * Q @ w
        w -= lr * grad
        w = project_to_simplex(w)
    return w


# -------------------- BRANCH AND BOUND STAGE 1 (NOW RESPECTS CANDIDATE MASK) --------------------
def branch_and_bound_topK(
        G: np.ndarray,
        candidate_idx: np.ndarray,  # NEW: only search within these indices
        m: int = 5,
        top_K: int = 20,
        top_P: int = 10
) -> List[Tuple[float, List[int], np.ndarray]]:
    N = G.shape[0]
    top_tuples = []

    # Stage 0: compute 1-unit losses only among candidates
    unit_losses = []
    for i in candidate_idx:
        w = np.array([1.0])
        loss = float(G[i, i])
        unit_losses.append((loss, [i], w))

    unit_losses.sort(key=lambda x: x[0])
    seeds = unit_losses[:top_P]

    # Branching function - only add from candidate_idx
    def expand_tuple(loss_so_far: float, indices: List[int]):
        if len(indices) == m:
            Q = G[np.ix_(indices, indices)]
            w = solve_qp_simplex(Q)
            total_loss = float(w @ Q @ w)
            top_tuples.append((total_loss, indices[:], w))
            top_tuples.sort(key=lambda x: x[0])
            if len(top_tuples) > top_K:
                top_tuples.pop(-1)
            return

        start = max(indices[-1] + 1, np.min(candidate_idx)) if indices else np.min(candidate_idx)
        for j_idx in range(np.searchsorted(candidate_idx, start), len(candidate_idx)):
            j = candidate_idx[j_idx]
            if j in indices:
                continue
            partial_Q = G[np.ix_(indices + [j], indices + [j])]
            lb = np.sum(np.diag(partial_Q))
            if len(top_tuples) >= top_K and lb >= top_tuples[-1][0]:
                continue  # prune
            expand_tuple(loss_so_far + lb, indices + [j])

    # Expand seeds
    for _, idx, _ in seeds:
        expand_tuple(0.0, idx)

    return top_tuples


# -------------------- TIME SPLIT --------------------
def split_periods(T0: int, T: int, frac_E: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    TE = int(T0 * frac_E)
    E_idx = np.arange(TE)
    B_idx = np.arange(TE, T0)
    post_idx = np.arange(T0, T)
    return E_idx, B_idx, post_idx


# -------------------- STANDARDIZATION --------------------
def build_X_tilde(X: np.ndarray, f: np.ndarray, idx: np.ndarray, J: int) -> np.ndarray:
    """Standardize X using weighted mean (f) and std dev, only over the first J columns (outcomes Y)"""
    X_sub = X[idx, :]
    mu = X_sub[:, :J] @ f.reshape(-1, 1)  # weighted mean over Y units only
    sigma = np.std(X_sub, axis=1, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (X_sub - mu) / sigma


# -------------------- CONTROL QP --------------------
def solve_control_qp(
        X_E: np.ndarray,
        treated_vec: np.ndarray,
        treated_idx: List[int],
        lambda_penalty: float = 0.1
) -> Optional[np.ndarray]:
    _, N = X_E.shape
    v = cp.Variable(N)
    match_term = cp.sum_squares(X_E @ v - treated_vec)
    penalty_term = lambda_penalty * cp.sum([
        v[j] * cp.sum_squares(X_E[:, j] - treated_vec)
        for j in range(N)
    ])
    objective = match_term + penalty_term
    constraints = [v >= 0, cp.sum(v) == 1]
    for j in treated_idx:
        constraints.append(v[j] == 0)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    if v.value is None:
        return None
    return np.asarray(v.value).flatten()


# -------------------- METRICS --------------------
def compute_nmse(X: np.ndarray, w: np.ndarray, target: np.ndarray, idx: np.ndarray, treated_idx: list) -> float:
    # Use correct slicing for treated units
    synth = X[idx][:, treated_idx] @ w
    tgt = target[idx]
    denom = np.var(X[idx, :len(target)], axis=1) + 1e-8  # variance over Y units
    return float(np.mean(((synth - tgt) ** 2) / denom))


def compute_effect_series(X: np.ndarray, treated_idx: list, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    return X[:, treated_idx] @ w - X @ v


# -------------------- PERMUTATION TEST --------------------
def permutation_test(
        effects: np.ndarray,
        B_idx: np.ndarray,
        post_idx: np.ndarray,
        n_perm: int = 500
) -> Dict[str, float]:
    placebo = effects[B_idx]
    post = effects[post_idx]
    stat_post = float(np.mean(np.abs(post)))
    combined = np.concatenate([placebo, post])
    T_b = len(B_idx)
    count = 0
    rng = np.random.default_rng()
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        fake_post = perm[T_b:]
        if np.mean(np.abs(fake_post)) >= stat_post:
            count += 1
    return {"stat_post": stat_post, "p_value": count / n_perm}


import numpy as np
from typing import Optional, Dict, List

def fast_synthetic_experiment(
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
        f: Optional[np.ndarray] = None,
        candidate_mask: Optional[np.ndarray] = None,
        m: int = 5,
        lambda_penalty: float = 0.1,
        top_K: int = 20,
        n_permutations: int = 500,
        frac_E: float = 0.7,
        top_P: int = 10,
        verbose: bool = True
) -> Dict:

    # Combine Y and Z if Z exists
    X = np.concatenate([Y, Z], axis=1) if Z is not None else Y.copy()
    T, N = X.shape
    if f is None:
        f = np.ones(N) / N

    if candidate_mask is None:
        candidate_mask = np.ones(N, dtype=bool)
    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) < m:
        raise ValueError(f"Not enough candidate units: {len(candidate_idx)} < m={m}")

    # Split periods
    T0 = int(0.8 * T)
    TE = int(T0 * frac_E)
    E_idx = np.arange(TE)
    B_idx = np.arange(TE, T0)
    post_idx = np.arange(T0, T)

    # Standardize X over E_idx
    X_sub = X[E_idx, :]
    mu = X_sub[:, :Y.shape[1]] @ f.reshape(-1, 1)
    sigma = np.std(X_sub, axis=1, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    X_E = (X_sub - mu) / sigma

    # Gram matrix
    G = X_E.T @ X_E

    # Stage 1: Branch-and-Bound (top-K treated combinations)
    candidates = branch_and_bound_topK(G, candidate_idx, m=m, top_K=top_K, top_P=top_P)
    if verbose:
        print(f"Stage 1 completed: {len(candidates)} top treated tuples selected from {len(candidate_idx)} candidates.")

    # Stage 2: Control weights + blank validation
    results = []
    target = X[:, :Y.shape[1]] @ f
    for loss, idx, w in candidates:
        treated_vec = X_E[:, idx] @ w
        v = solve_control_qp(X_E, treated_vec, idx, lambda_penalty)
        if v is None:
            continue

        nmse_B = compute_nmse(X, w, target, B_idx, treated_idx=idx)
        effects = X[:, idx] @ w - X @ v

        # --- Conformal interval based on blank periods ---
        residuals = effects[B_idx]
        q = np.quantile(np.abs(residuals), 0.95)  # 95% quantile
        lower = -q
        upper = q
        coverage = np.mean((effects[post_idx] >= lower) & (effects[post_idx] <= upper))
        width = upper - lower

        result_dict = {
            "treated_idx": idx.copy(),
            "treated_weights": w.copy(),
            "control_weights": v.copy(),
            "loss_E": float(loss),
            "nmse_B": float(nmse_B),
            "effects": effects,
            "conformal_lower": lower,
            "conformal_upper": upper,
            "conformal_width": width,
            "conformal_coverage": coverage
        }
        results.append(result_dict)

    results.sort(key=lambda x: x["nmse_B"])

    # Permutation test
    for r in results:
        r.update(permutation_test(r['effects'], B_idx, post_idx, n_permutations))

    if verbose and results:
        best = results[0]
        print("\n=== Best Design ===")
        print(f"Treated units       : {best['treated_idx']}")
        print(f"NMSE (blank periods): {best['nmse_B']:.6f}")
        print(f"Conformal coverage  : {best['conformal_coverage']:.4f}")
        print(f"p-value (permutation): {best.get('p_value', 'N/A'):.4f}")

    # Add scalar conformal bounds to best
    if results:
        best_res = results[0]
        best_res['conformal_lower_mean'] = best_res['conformal_lower']
        best_res['conformal_upper_mean'] = best_res['conformal_upper']
        best_res['conformal_width_mean'] = best_res['conformal_width']
        best_res['conformal_coverage'] = best_res['conformal_coverage']

    return {
        "results": results,
        "best": results[0] if results else None,
        "E_idx": E_idx,
        "B_idx": B_idx,
        "post_idx": post_idx,
        "X": X,
        "f": f,
        "candidate_idx": candidate_idx,
        "T": T,
        "T0": T0
    }


import numpy as np
from typing import Tuple, Dict

def simulate_synthetic_experiment_data(
    J: int = 15,
    T: int = 30,
    T0: int = 25,
    R: int = 7,
    F: int = 5,
    sigma: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Correct simulation: Y (T, J), Z (T, R)"""
    np.random.seed(seed)

    f = np.ones(J) / J  # unit weights

    # === Generate Z directly as (T, R) ===
    Z = np.random.uniform(0, 1, size=(T, R))  # shape (T, R)

    # Unit loadings for unobservables
    mu = np.random.uniform(0, 1, size=(F, J))       # (F, J)

    # Time-varying factors
    delta_t = np.sort(np.random.uniform(0, 20, T))[::-1]
    upsilon_t = np.sort(np.random.uniform(0, 20, T))[::-1]

    gamma_t = np.random.uniform(0, 10, size=(F, T))   # (F, T)
    lambda_t = np.random.uniform(0, 10, size=(R, T))  # (R, T)

    # Generate outcomes
    Y = np.zeros((T, J))
    for t in range(T):
        for j in range(J):
            obs_part = Z[t, :] @ lambda_t[:, t]  # use Z row at time t
            unobs_part = mu[:, j] @ gamma_t[:, t]
            common = delta_t[t] + upsilon_t[t]

            epsilon = np.random.normal(0, sigma)
            xi = np.random.normal(0, sigma)

            Y[t, j] = obs_part + unobs_part + common + epsilon + xi

    # Candidate mask: first 5 units eligible for treatment
    candidate_mask = np.zeros(J, dtype=bool)
    candidate_mask[:20] = True

    # ==================== SAFETY ASSERTIONS ====================
    assert Y.shape == (T, J), f"Y should be ({T}, {J}), got {Y.shape}"
    assert Z.shape == (T, R), f"Z should be ({T}, {R}), got {Z.shape}"
    assert f.shape == (J,), f"f should be ({J},), got {f.shape}"
    assert candidate_mask.shape == (J,), f"candidate_mask should be ({J},), got {candidate_mask.shape}"
    assert np.sum(candidate_mask) >= 1, "At least one candidate unit required"
    assert np.all(np.isfinite(Y)) and np.all(np.isfinite(Z)), "NaN or Inf in data"

    #print(f"✅ Simulation successful | Y: {Y.shape} | Z: {Z.shape} | Candidates: {np.sum(candidate_mask)}")

    info = {
        "J": J,
        "T": T,
        "T0": T0,
        "R": R,
        "F": F,
        "sigma": sigma,
        "f": f,
        "candidate_mask": candidate_mask,
        "description": "Corrected factor model simulation"
    }

    return Y, Z, f, candidate_mask, info







from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import multiprocessing as mp
import numpy as np
import pandas as pd


# -------------------- PROCESS CHECK --------------------
def is_main_process():
    return mp.current_process().name == "MainProcess"


# -------------------- SINGLE SIMULATION --------------------
def run_single_sim(sim, m, lam, J, T, R, F, sigma, top_K, n_permutations):
    seed = 1000 + sim

    Y, Z, f, candidate_mask, info = simulate_synthetic_experiment_data(
        J=J, T=T, R=R, F=F, sigma=sigma, seed=seed
    )

    res_dict = fast_synthetic_experiment(
        Y=Y,
        Z=Z,
        f=f,
        candidate_mask=candidate_mask,
        m=m,
        lambda_penalty=lam,
        top_K=top_K,
        n_permutations=n_permutations,
        verbose=False   # 🔑 ensures no inner printing
    )

    best = res_dict['best']
    if best is None:
        return None

    post_mean = float(np.mean(best['effects'][res_dict['post_idx']]))
    prop_treated = len(best['treated_idx']) / candidate_mask.sum()

    return {
        "sim": sim,
        "m": m,
        "lambda_penalty": lam,
        "loss_E": best['loss_E'],
        "nmse_B": best['nmse_B'],
        "post_effect_mean": post_mean,
        "p_value": best.get('p_value', np.nan),
        "treated_units": best['treated_idx'],
        "prop_treated": prop_treated,
        "conformal_lower_mean": best['conformal_lower_mean'],
        "conformal_upper_mean": best['conformal_upper_mean'],
        "conformal_width_mean": best['conformal_width_mean'],
        "conformal_coverage": best['conformal_coverage']
    }


# -------------------- MONTE CARLO SWEEP --------------------
def monte_carlo_sweep_parallel(
        n_sim=1000,
        J=200,
        T=120,
        R=7,
        F=5,
        sigma=1.0,
        lambda_list=[0.01, 0.1, 0.5],
        m_list=[2, 4, 10],
        top_K=15,
        n_permutations=1000,
        n_jobs=-1,
        verbose=True,
        output_file="mc_sweep_results.csv"
):
    tasks = [(sim, m, lam)
             for m in m_list
             for lam in lambda_list
             for sim in range(n_sim)]

    if verbose:
        print(f"Running {len(tasks)} simulations in parallel...")

    # 🔑 SINGLE CLEAN PROGRESS BAR (Windows-safe)
    pbar = tqdm(
        total=len(tasks),
        desc="Monte Carlo",
        leave=True,
        disable=True
    )

    with tqdm_joblib(pbar):
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            verbose=0
        )(
            delayed(run_single_sim)(
                sim, m, lam, J, T, R, F, sigma, top_K, n_permutations
            )
            for sim, m, lam in tasks
        )

    results = [r for r in results if r is not None]

    if verbose:
        print(f"Completed {len(results)}/{len(tasks)} successful simulations.")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

    if verbose:
        print(f"Simulation results saved to {output_file}")

    return df_results


# -------------------- EXAMPLE RUN --------------------

lambda_list = [0.01, 0.1, 0.5]
m_list = [2, 4, 9]

df_mc_sweep = monte_carlo_sweep_parallel(
    n_sim=10,
    lambda_list=lambda_list,
    m_list=m_list,
    verbose=True,
    output_file="mc_sweep_results.csv"
)
