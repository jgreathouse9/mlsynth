"""
synthetic_design.py

One-way global synthetic experimental design using the closed-form
reduced objective from Theorem 1 of Doudchenko et al. (2021).

The control weights are analytically eliminated via Theorem 1, leaving
a MIP purely over binary treatment indicators D_i. The reduced objective:

    J(I) = sigma^2 * (1/K + 1/(N-K) + (a_bar_I - a_bar_Ibar)^2 / (sigma^2 + V^2_Ibar))

is a ratio of quadratics in D, solved via Dinkelbach iteration —
each subproblem is a standard MIQP handled by cvxpy.
"""

import numpy as np
import cvxpy as cp
from typing import Optional


# --------------------------------------------------------------------------- #
# Step 1 — Estimate sigma^2                                                   #
# --------------------------------------------------------------------------- #

def estimate_sigma2(Y: np.ndarray) -> float:
    """
    Estimate sigma^2 as the average per-unit sample variance across
    pre-treatment time periods (Section 6 of the paper).

    Parameters
    ----------
    Y : np.ndarray, shape (T, N)

    Returns
    -------
    float
        Estimated sigma^2.
    """
    return float(np.mean(np.var(Y, axis=0, ddof=1)))


# --------------------------------------------------------------------------- #
# Step 2 — Initialise a feasible binary assignment                            #
# --------------------------------------------------------------------------- #

def initialise_assignment(N: int, K: int, seed: int = 0) -> np.ndarray:
    """
    Return a random feasible binary assignment vector D of length N
    with exactly K ones.

    Parameters
    ----------
    N : int
        Total number of units.
    K : int
        Number of treated units.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (N,), dtype int
    """
    rng = np.random.default_rng(seed)
    D = np.zeros(N, dtype=int)
    D[rng.choice(N, K, replace=False)] = 1
    return D


# --------------------------------------------------------------------------- #
# Step 3 — Evaluate the reduced objective J(I) for a given assignment        #
# --------------------------------------------------------------------------- #

def evaluate_objective(D_val: np.ndarray, a: np.ndarray, sigma2: float) -> float:
    """
    Evaluate the ratio term of the reduced objective from Theorem 1:

        f(D) / g(D) = (a_bar_I - a_bar_Ibar)^2 / (sigma^2 + V^2_Ibar)

    Parameters
    ----------
    D_val : np.ndarray, shape (N,), binary int
    a     : np.ndarray, shape (N,)
        Time-averaged outcome per unit.
    sigma2 : float

    Returns
    -------
    float
    """
    treated = D_val == 1
    control = ~treated
    a_I   = a[treated].mean()
    a_Ic  = a[control].mean()
    V2_Ic = np.sum((a[control] - a_Ic) ** 2)
    return float((a_I - a_Ic) ** 2 / (sigma2 + V2_Ic))


# --------------------------------------------------------------------------- #
# Step 4 — Build the cvxpy MIQP subproblem expressions                       #
# --------------------------------------------------------------------------- #

def build_subproblem_expressions(
    a: np.ndarray,
    K: int,
    sigma2: float,
) -> tuple[cp.Variable, cp.Expression, cp.Expression, list]:
    """
    Construct the cvxpy variable, f(D), g(D), and base constraints
    for the Dinkelbach MIQP subproblem.

    The Dinkelbach subproblem at iteration k is:
        min  f(D) - eta_k * g(D)
        s.t. D binary, sum(D) = K

    where:
        f(D) = (a_bar_I - a_bar_Ibar)^2          [numerator, quadratic]
        g(D) = sigma^2 + V^2_Ibar                [denominator, quadratic]

    V^2_Ibar is expanded algebraically to avoid auxiliary variables:
        V^2_Ibar = sum_i (1-D_i)*a_i^2  -  (1/(N-K)) * (sum_i (1-D_i)*a_i)^2

    Parameters
    ----------
    a      : np.ndarray, shape (N,)
    K      : int
    sigma2 : float

    Returns
    -------
    D_var       : cp.Variable  — binary decision variable, shape (N,)
    f_expr      : cp.Expression — numerator quadratic
    g_expr      : cp.Expression — denominator quadratic
    constraints : list          — base constraints [sum(D)==K]
    """
    N = len(a)
    D_var = cp.Variable(N, boolean=True)

    a_bar_I  = (1.0 / K)       * (a @ D_var)
    a_bar_Ic = (1.0 / (N - K)) * (a @ (1 - D_var))
    diff     = a_bar_I - a_bar_Ic

    # V^2_Ibar expanded as quadratic in D (no auxiliary variables needed)
    sum_a2_ctrl = (a ** 2) @ (1 - D_var)
    sum_a_ctrl  = a        @ (1 - D_var)
    V2_Ic       = sum_a2_ctrl - (1.0 / (N - K)) * cp.square(sum_a_ctrl)

    f_expr = cp.square(diff)
    g_expr = sigma2 + V2_Ic

    constraints = [cp.sum(D_var) == K]

    return D_var, f_expr, g_expr, constraints


# --------------------------------------------------------------------------- #
# Step 5 — Solve a single Dinkelbach MIQP subproblem                         #
# --------------------------------------------------------------------------- #

def solve_subproblem(
    D_var: cp.Variable,
    f_expr: cp.Expression,
    g_expr: cp.Expression,
    constraints: list,
    eta: float,
) -> tuple[np.ndarray, float]:
    """
    Solve one Dinkelbach subproblem:
        min  f(D) - eta * g(D)
        s.t. constraints

    Parameters
    ----------
    D_var       : cp.Variable
    f_expr      : cp.Expression
    g_expr      : cp.Expression
    constraints : list
    eta         : float  — current Dinkelbach parameter

    Returns
    -------
    D_new       : np.ndarray, shape (N,), binary int — optimal assignment
    sub_value   : float — optimal subproblem objective value
    """
    problem = cp.Problem(cp.Minimize(f_expr - eta * g_expr), constraints)

    for solver in [cp.SCIP, cp.GUROBI, cp.CPLEX, cp.MOSEK, cp.GLPK_MI]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ("optimal", "optimal_inaccurate"):
                break
        except (cp.SolverError, AttributeError):
            continue
    else:
        raise RuntimeError(
            "No MIP solver found or problem infeasible. "
            "Install SCIP via: pip install pyscipopt"
        )

    D_new = np.round(D_var.value).astype(int)
    return D_new, float(problem.value)


# --------------------------------------------------------------------------- #
# Step 6 — Dinkelbach iteration loop                                          #
# --------------------------------------------------------------------------- #

def dinkelbach_loop(
    D_init: np.ndarray,
    a: np.ndarray,
    sigma2: float,
    K: int,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """
    Run Dinkelbach's algorithm to minimise f(D)/g(D) over binary D.

    At each iteration:
        1. Solve:  min f(D) - eta * g(D)   [MIQP via cvxpy]
        2. Update: eta <- f(D*) / g(D*)
        3. Converge when subproblem optimal value < tol

    Dinkelbach's theorem guarantees superlinear convergence.

    Parameters
    ----------
    D_init   : np.ndarray, shape (N,) — initial feasible assignment
    a        : np.ndarray, shape (N,) — time-averaged outcomes
    sigma2   : float
    K        : int
    max_iter : int
    tol      : float — convergence tolerance on subproblem value

    Returns
    -------
    best_D   : np.ndarray, shape (N,), binary int
    best_obj : float — best reduced objective value f/g achieved
    """
    N = len(a)
    D_var, f_expr, g_expr, constraints = build_subproblem_expressions(a, K, sigma2)

    best_D   = D_init.copy()
    best_obj = evaluate_objective(D_init, a, sigma2)
    eta      = best_obj

    for iteration in range(max_iter):
        D_new, sub_value = solve_subproblem(D_var, f_expr, g_expr, constraints, eta)
        eta_new = evaluate_objective(D_new, a, sigma2)

        if eta_new < best_obj:
            best_obj = eta_new
            best_D   = D_new.copy()

        if abs(sub_value) < tol:
            break

        eta = eta_new

    return best_D, best_obj


# --------------------------------------------------------------------------- #
# Step 7 — Compute closed-form control weights                                #
# --------------------------------------------------------------------------- #

def compute_closed_form_weights(
    D_val: np.ndarray,
    a: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """
    Compute the closed-form optimal control weights from Theorem 1:

        w*_l = 1/(N-K)  -  (a_bar_I - a_bar_Ibar)(a_bar_Ibar - a_l) / (sigma^2 + V^2_Ibar)

    for l in control set; 0 for treated units.

    Parameters
    ----------
    D_val  : np.ndarray, shape (N,), binary int
    a      : np.ndarray, shape (N,) — time-averaged outcomes
    sigma2 : float

    Returns
    -------
    w : np.ndarray, shape (N,)
        Optimal control weights (0 for treated units).
    """
    N       = len(a)
    treated = D_val == 1
    control = ~treated

    a_I   = a[treated].mean()
    a_Ic  = a[control].mean()
    V2_Ic = np.sum((a[control] - a_Ic) ** 2)
    denom = sigma2 + V2_Ic
    n_c   = control.sum()

    w = np.zeros(N)
    w[control] = (1.0 / n_c) - (a_I - a_Ic) * (a_Ic - a[control]) / denom
    return w


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #

def one_way_global_design(
    Y: np.ndarray,
    K: int,
    sigma2: Optional[float] = None,
    seed: int = 0,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> dict:
    """
    One-way global synthetic experimental design.

    Jointly selects K treated units and closed-form control weights to
    minimise the Theorem 1 reduced objective via Dinkelbach iteration.

    Parameters
    ----------
    Y      : np.ndarray, shape (T, N)
        Panel outcome matrix. Rows = time periods, columns = units.
    K      : int
        Number of units to assign to treatment.
    sigma2 : float or None
        Ridge penalty / noise variance. Defaults to average per-unit
        sample variance across pre-treatment periods (Section 6).
    seed     : int   — random seed for initialisation
    max_iter : int   — max Dinkelbach iterations
    tol      : float — convergence tolerance

    Returns
    -------
    dict with keys
        "D"           : np.ndarray (N,) binary treatment indicators
        "w_control"   : np.ndarray (N,) closed-form control weights
        "treated"     : list of treated unit indices
        "control"     : list of control unit indices
        "objective"   : full objective value J(I) including constants
    """
    T, N = Y.shape

    if K < 1 or K >= N:
        raise ValueError(f"K must satisfy 1 <= K < N, got K={K}, N={N}")

    # Step 1 — sigma^2
    if sigma2 is None:
        sigma2 = estimate_sigma2(Y)

    # Time-averaged outcome per unit (T=1 analog from Section 4)
    a = Y.mean(axis=0)

    # Step 2 — initialise
    D_init = initialise_assignment(N, K, seed=seed)

    # Steps 3–6 — Dinkelbach loop
    best_D, ratio_obj = dinkelbach_loop(
        D_init, a, sigma2, K, max_iter=max_iter, tol=tol
    )

    # Step 7 — closed-form weights
    w = compute_closed_form_weights(best_D, a, sigma2)

    treated_idx = [i for i in range(N) if best_D[i] == 1]
    control_idx = [i for i in range(N) if best_D[i] == 0]

    full_obj = sigma2 * (1.0 / K + 1.0 / (N - K) + ratio_obj)

    return {
        "D":         best_D,
        "w_control": w,
        "treated":   treated_idx,
        "control":   control_idx,
        "objective": full_obj,
    }


# =========================================================================== #
#  PERMUTATION INFERENCE & POWER ANALYSIS                                     #
#  Following Chernozhukov, Wüthrich & Zhu (2021) as described in             #
#  Section A.4 of Doudchenko et al. (2021).                                  #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Inference Step 1 — Estimate ATET given an outcome matrix and design        #
# --------------------------------------------------------------------------- #

def estimate_atet(
    Y_post: np.ndarray,
    D: np.ndarray,
    w_control: np.ndarray,
) -> float:
    """
    Estimate the average treatment effect on the treated (ATET) as the
    difference between the uniform treated mean and the weighted control mean,
    averaged across post-treatment periods.

        ATET = mean_t [ (1/K) * sum_{i treated} Y_it
                        - sum_{i control} w_i * Y_it ]

    Parameters
    ----------
    Y_post    : np.ndarray, shape (S_post, N)
        Outcome matrix over post-treatment periods only.
    D         : np.ndarray, shape (N,), binary int
    w_control : np.ndarray, shape (N,)
        Pre-computed control weights (0 for treated units).

    Returns
    -------
    float
    """
    treated = D == 1
    control = ~treated
    K = treated.sum()

    treated_mean = Y_post[:, treated].mean(axis=1)           # shape (S_post,)
    control_mean = Y_post[:, control] @ w_control[control]   # shape (S_post,)
    return float((treated_mean - control_mean).mean())


# --------------------------------------------------------------------------- #
# Inference Step 2 — Compute test statistic U(Y)                             #
# --------------------------------------------------------------------------- #

def test_statistic(atet: float, n_treat_periods: int) -> float:
    """
    Compute the CWZ test statistic:

        U(Y) = |ATET| / sqrt(n_treat_periods)

    Dividing by sqrt(S_treat) normalises for the number of post-treatment
    periods, making the statistic comparable across permuted samples that
    may have different numbers of "treatment" periods.

    Parameters
    ----------
    atet            : float
    n_treat_periods : int

    Returns
    -------
    float
    """
    return abs(atet) / np.sqrt(n_treat_periods)


# --------------------------------------------------------------------------- #
# Inference Step 3 — Generate permuted time indices                          #
# --------------------------------------------------------------------------- #

def generate_permutations(
    S: int,
    scheme: str = "iid",
    seed: int = 0,
) -> list[np.ndarray]:
    """
    Generate all permuted orderings of S time period indices.

    Two schemes from CWZ / Section A.4:

    "iid"          — all S! permutations (approximated by S random draws
                     without replacement for tractability when S is large).
                     Requires i.i.d. exchangeability of Yt(0) across time.

    "moving_block" — the S cyclic shifts of [0, 1, ..., S-1].
                     Requires only stationarity (weaker assumption).
                     Produces exactly S unique permutations.

    Parameters
    ----------
    S      : int    — total number of time periods (pre + post)
    scheme : str    — "iid" or "moving_block"
    seed   : int    — random seed (used only for iid scheme)

    Returns
    -------
    list of np.ndarray, each of shape (S,)
        Each array is a permuted ordering of range(S).
    """
    base = np.arange(S)

    if scheme == "moving_block":
        # Exactly S cyclic shifts — deterministic, no seed needed
        return [np.roll(base, shift) for shift in range(S)]

    elif scheme == "iid":
        # S random permutations without replacement
        rng = np.random.default_rng(seed)
        return [rng.permutation(S) for _ in range(S)]

    else:
        raise ValueError(f"scheme must be 'iid' or 'moving_block', got '{scheme}'")


# --------------------------------------------------------------------------- #
# Inference Step 4 — Build permuted bootstrap distribution                   #
# --------------------------------------------------------------------------- #

def bootstrap_distribution(
    Y_full: np.ndarray,
    D: np.ndarray,
    w_control: np.ndarray,
    n_treat_periods: int,
    scheme: str = "iid",
    seed: int = 0,
) -> np.ndarray:
    """
    Construct the permutation distribution of the test statistic U(Y)
    under the sharp null hypothesis of zero treatment effects.

    For each permuted ordering of time periods:
        1. Reorder Y according to the permutation
        2. Treat the last n_treat_periods as "post-treatment"
        3. Estimate ATET and compute U on the permuted data

    Parameters
    ----------
    Y_full          : np.ndarray, shape (S, N)
        Full outcome matrix (all pre + post periods, NO treatment added).
        Under the sharp null, Y_full is the observed control potential outcome.
    D               : np.ndarray, shape (N,), binary int
    w_control       : np.ndarray, shape (N,)
    n_treat_periods : int   — number of post-treatment periods
    scheme          : str   — "iid" or "moving_block"
    seed            : int

    Returns
    -------
    np.ndarray, shape (n_permutations,)
        Bootstrap distribution of U under the sharp null.
    """
    S = Y_full.shape[0]
    permutations = generate_permutations(S, scheme=scheme, seed=seed)
    null_stats = []

    for perm in permutations:
        Y_perm      = Y_full[perm, :]
        Y_post_perm = Y_perm[-n_treat_periods:, :]
        atet_perm   = estimate_atet(Y_post_perm, D, w_control)
        null_stats.append(test_statistic(atet_perm, n_treat_periods))

    return np.array(null_stats)


# --------------------------------------------------------------------------- #
# Inference Step 5 — Sharp null hypothesis test                              #
# --------------------------------------------------------------------------- #

def permutation_test(
    Y_pre: np.ndarray,
    Y_post: np.ndarray,
    D: np.ndarray,
    w_control: np.ndarray,
    alpha: float = 0.10,
    scheme: str = "iid",
    seed: int = 0,
) -> dict:
    """
    Test the sharp null hypothesis H0: no treatment effect on any treated
    unit in any treatment period, using the CWZ permutation procedure.

    Parameters
    ----------
    Y_pre     : np.ndarray, shape (T, N)   — pre-treatment outcomes
    Y_post    : np.ndarray, shape (S_post, N) — post-treatment outcomes
                (observed, including any true treatment effect)
    D         : np.ndarray, shape (N,), binary int
    w_control : np.ndarray, shape (N,)
    alpha     : float — significance level (e.g. 0.10 for 90% test)
    scheme    : str   — "iid" or "moving_block"
    seed      : int

    Returns
    -------
    dict with keys
        "atet"        : float — observed ATET estimate
        "statistic"   : float — observed test statistic U
        "threshold"   : float — (1-alpha) quantile of null distribution
        "p_value"     : float — permutation p-value
        "reject"      : bool  — True if H0 rejected at level alpha
        "null_dist"   : np.ndarray — full bootstrap null distribution
    """
    n_treat_periods = Y_post.shape[0]

    # Observed test statistic on actual data
    atet_obs = estimate_atet(Y_post, D, w_control)
    U_obs    = test_statistic(atet_obs, n_treat_periods)

    # Null distribution: permute over all time periods (pre + post)
    # Under H0, Y_post has no treatment effect so we can concatenate safely
    Y_full   = np.vstack([Y_pre, Y_post])
    null_dist = bootstrap_distribution(
        Y_full, D, w_control, n_treat_periods, scheme=scheme, seed=seed
    )

    threshold = np.quantile(null_dist, 1.0 - alpha)
    p_value   = float(np.mean(null_dist >= U_obs))

    return {
        "atet":      atet_obs,
        "statistic": U_obs,
        "threshold": threshold,
        "p_value":   p_value,
        "reject":    bool(U_obs > threshold),
        "null_dist": null_dist,
    }


# --------------------------------------------------------------------------- #
# Power Step 1 — Inject a synthetic treatment effect into post data          #
# --------------------------------------------------------------------------- #

def inject_treatment_effect(
    Y_post_clean: np.ndarray,
    D: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Add a homogeneous additive treatment effect tau to all treated units
    in all post-treatment periods.

        Y_post_observed[t, i] = Y_post_clean[t, i] + tau  if D[i] == 1

    Parameters
    ----------
    Y_post_clean : np.ndarray, shape (S_post, N)
        Post-treatment outcomes under the null (no treatment effect).
    D            : np.ndarray, shape (N,), binary int
    tau          : float — true ATET to inject

    Returns
    -------
    np.ndarray, shape (S_post, N)
    """
    Y_obs = Y_post_clean.copy()
    Y_obs[:, D == 1] += tau
    return Y_obs


# --------------------------------------------------------------------------- #
# Power Step 2 — Estimate rejection rate at a single tau value               #
# --------------------------------------------------------------------------- #

def rejection_rate_at_tau(
    Y_pre: np.ndarray,
    Y_post_clean: np.ndarray,
    D: np.ndarray,
    w_control: np.ndarray,
    tau: float,
    n_simulations: int = 100,
    alpha: float = 0.10,
    scheme: str = "iid",
    seed: int = 0,
) -> float:
    """
    Estimate the probability of rejecting H0 when the true ATET = tau,
    by repeating the permutation test across n_simulations bootstrap
    replications of the post-treatment data.

    Since we have fixed pre-treatment data and a single post-treatment
    sample, we approximate the sampling distribution by re-drawing
    the null distribution with different seeds across simulations.

    Parameters
    ----------
    Y_pre         : np.ndarray, shape (T, N)
    Y_post_clean  : np.ndarray, shape (S_post, N)
        Post-treatment outcomes with NO treatment effect.
    D             : np.ndarray, shape (N,), binary int
    w_control     : np.ndarray, shape (N,)
    tau           : float — true treatment effect to inject
    n_simulations : int   — number of simulation replications
    alpha         : float — significance level
    scheme        : str   — "iid" or "moving_block"
    seed          : int   — base random seed

    Returns
    -------
    float — estimated rejection probability at this tau
    """
    rejections = 0

    for sim in range(n_simulations):
        Y_post_obs = inject_treatment_effect(Y_post_clean, D, tau)
        result = permutation_test(
            Y_pre, Y_post_obs, D, w_control,
            alpha=alpha, scheme=scheme, seed=seed + sim,
        )
        if result["reject"]:
            rejections += 1

    return rejections / n_simulations


# --------------------------------------------------------------------------- #
# Power Step 3 — Compute full power curve across a grid of tau values        #
# --------------------------------------------------------------------------- #

def power_curve(
    Y_pre: np.ndarray,
    Y_post_clean: np.ndarray,
    D: np.ndarray,
    w_control: np.ndarray,
    tau_grid: np.ndarray,
    n_simulations: int = 100,
    alpha: float = 0.10,
    scheme: str = "iid",
    seed: int = 0,
) -> dict:
    """
    Compute the power curve: rejection probability as a function of
    the true ATET, across a user-supplied grid of tau values.

    tau = 0 gives the empirical test size (should be ~= alpha for a
    well-calibrated test).

    Parameters
    ----------
    Y_pre         : np.ndarray, shape (T, N)
    Y_post_clean  : np.ndarray, shape (S_post, N)
        Post-treatment outcomes with NO treatment effect.
    D             : np.ndarray, shape (N,), binary int
    w_control     : np.ndarray, shape (N,)
    tau_grid      : np.ndarray, shape (n_tau,)
        Grid of true ATET values to evaluate power at.
        Should include 0 to verify test size.
    n_simulations : int   — simulations per tau value
    alpha         : float — significance level
    scheme        : str   — "iid" or "moving_block"
    seed          : int   — base random seed

    Returns
    -------
    dict with keys
        "tau_grid"        : np.ndarray — input tau grid
        "rejection_rates" : np.ndarray — power at each tau
        "test_size"       : float      — rejection rate at tau=0
        "alpha"           : float      — nominal significance level
        "scheme"          : str
    """
    rejection_rates = []

    for tau in tau_grid:
        rate = rejection_rate_at_tau(
            Y_pre, Y_post_clean, D, w_control,
            tau=tau,
            n_simulations=n_simulations,
            alpha=alpha,
            scheme=scheme,
            seed=seed,
        )
        rejection_rates.append(rate)

    rejection_rates = np.array(rejection_rates)

    # Test size = rejection rate under the null (tau = 0)
    tau0_idx  = np.argmin(np.abs(tau_grid))
    test_size = rejection_rates[tau0_idx]

    return {
        "tau_grid":        tau_grid,
        "rejection_rates": rejection_rates,
        "test_size":       test_size,
        "alpha":           alpha,
        "scheme":          scheme,
    }


# --------------------------------------------------------------------------- #
# Quick smoke test                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    T      = 30   # pre-treatment periods
    S_post = 5    # post-treatment periods
    N      = 20   # units
    K      = 4    # treated units

    # Simulate panel data
    Y_all   = rng.standard_normal((T + S_post, N))
    Y_pre   = Y_all[:T, :]
    Y_post  = Y_all[T:, :]

    # --- Design ---
    design = one_way_global_design(Y_pre, K=K)
    print(f"Treated units : {design['treated']}")
    print(f"Objective     : {design['objective']:.6f}")

    # --- Single permutation test (tau = 0.05 injected) ---
    Y_post_obs = inject_treatment_effect(Y_post, design["D"], tau=0.05)
    test = permutation_test(
        Y_pre, Y_post_obs,
        design["D"], design["w_control"],
        alpha=0.10, scheme="iid",
    )
    print(f"\nPermutation test (tau=0.05):")
    print(f"  ATET estimate : {test['atet']:.4f}")
    print(f"  Test statistic: {test['statistic']:.4f}")
    print(f"  Threshold     : {test['threshold']:.4f}")
    print(f"  p-value       : {test['p_value']:.4f}")
    print(f"  Reject H0     : {test['reject']}")

    # --- Power curve ---
    tau_grid = np.linspace(0, 0.10, 8)
    curve    = power_curve(
        Y_pre, Y_post,
        design["D"], design["w_control"],
        tau_grid=tau_grid,
        n_simulations=40,
        alpha=0.10,
        scheme="iid",
    )
    print(f"\nPower curve (iid permutations, alpha=0.10):")
    print(f"  {'tau':>8}  {'power':>8}")
    for tau, power in zip(curve["tau_grid"], curve["rejection_rates"]):
        print(f"  {tau:8.4f}  {power:8.3f}")
    print(f"\n  Empirical test size at tau=0: {curve['test_size']:.3f}")
