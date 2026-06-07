"""Path-B replication of Vives-i-Bastida (2022) "Predictor Selection for
Synthetic Controls", Section 4 / Figures 1-2 (arXiv:2203.11576), using the
grouped linear-factor design of the companion paper Abadie & Vives-i-Bastida
(2022, arXiv:2203.06279) extended with covariates.

DGP:  Y_it = delta_t + theta_t' Z_i + lambda_t' mu_i + eps_it
  * J+1 = 21 units in 7 groups of 3, one-hot factor loadings mu_i (each unit
    loads only on its group's factor);
  * lambda_t AR(1), rho = 0.5, standard-Gaussian innovations; delta_t = 100;
    eps ~ N(0, 0.25^2);
  * covariates Z = [Z^1 useful (theta != 0), Z^2 nuisance (theta = 0)] ~ U[0,1];
  * treated unit's useful predictors = 1/2(Z_2 + Z_3) and it shares units 2,3's
    group -> the oracle synthetic control is w_2 = w_3 = 1/2;
  * design matrix adds 10 lagged outcomes (20 predictors total);
  * theta_t is time-varying N(0,1) on the useful predictors (its scale is not
    specified in either paper; filled in the spirit of the theta_t Z_i term).
  Two regimes: k1=k2=5 (balanced) and k1=1,k2=9 (nuisance-heavy). True effect 0.

Three estimators, mapped onto mlsynth's SparseSC machinery:
  * SCM      -- fixed Mahalanobis V = (X0' X0)^{-1} (no optimisation);
  * SCM l=0  -- V minimises the validation-block fit, no penalty (ADH 2015);
  * Sparse   -- the same with L1 penalty + CV-selected lambda.

Result (B = 60 draws, this script):
  k1=5,k2=5 : post-MSE  SCM 1.78 / l=0 0.132 / Sparse 0.153 ; Vnoise l=0 5.27 / Sparse 0.16
  k1=1,k2=9 : post-MSE  SCM 1.99 / l=0 0.164 / Sparse 0.141 ; Vnoise l=0 0.85 / Sparse 0.21
  -> SCM worst; Sparse robust across regimes and beats the unpenalised method
     under heavy nuisance; Sparse prunes the nuisance-predictor weights to ~0.

Run:  python examples/sparsesc_vives_simulation.py [B]      (default B=60)
Parallel over draws with BLAS pinned to 1 thread/worker.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from mlsynth.utils.sparse_sc_helpers.setup import prepare_sparse_sc_inputs
from mlsynth.utils.sparse_sc_helpers.optimization import sweep_lambda, recover_w
from mlsynth.utils.fscm_helpers.bilevel.simplex import simplex_lstsq

def scm_maha_w(X1, X0):
    """Standard SCM with the paper's fixed V = (X0' X0)^{-1} (full Mahalanobis).
    Solve min_w (X1-X0 w)'V(X1-X0 w) on the simplex via V=LL', A=L'X0, b=L'X1.
    """
    K = X0.shape[0]
    G = X0 @ X0.T; G = 0.5 * (G + G.T)
    G += (1e-8 * np.trace(G) / max(K, 1)) * np.eye(K)
    V = np.linalg.inv(G); V = 0.5 * (V + V.T)
    L = np.linalg.cholesky(V + 1e-12 * np.eye(K))
    return simplex_lstsq(L.T @ X0, L.T @ X1)

N1, T, T0, Tv, F = 21, 30, 20, 10, 7
LAGS = list(range(1, 11))
SP_GRID = np.concatenate([[0.0], np.logspace(-4, 0, 20)])
B = int(sys.argv[1]) if len(sys.argv) > 1 else 60
# SCM = fixed Mahalanobis V (special); the other two optimise V (validation fit +/- L1)
SWEEP = {"SCM l=0": ("validation", [0.0]), "Sparse": ("validation", SP_GRID)}
METHODS = ["SCM", "SCM l=0", "Sparse"]
SETTINGS = [(5, 5), (1, 9)]

def metrics(w, inp, idx23):
    T0t, T0tr = inp.T0_total, inp.T0_train
    gpre = inp.Y1[:T0t] - inp.Y0[:T0t] @ w
    gval = inp.Y1[T0tr:T0t] - inp.Y0[T0tr:T0t] @ w
    gpost = inp.Y1[T0t:] - inp.Y0[T0t:] @ w
    return [float(np.mean(gpost ** 2)), float(np.mean(gpost)),
            float(np.sqrt(np.mean(gpre ** 2))), float(np.sqrt(np.mean(gval ** 2))),
            float(w[idx23].sum())]

def simulate(k1, k2, seed):
    rng = np.random.default_rng(seed); k = k1 + k2
    Z = rng.uniform(0, 1, (N1, k))
    Z[0, :k1] = 0.5 * (Z[1, :k1] + Z[2, :k1])
    group = np.arange(N1) // 3; mu = np.eye(F)[group]
    lam = np.zeros((T, F)); lam[0] = rng.normal(0, 1, F)
    for t in range(1, T):
        lam[t] = 0.5 * lam[t - 1] + rng.normal(0, 1, F)
    theta = np.zeros((T, k)); theta[:, :k1] = rng.normal(0, 1, (T, k1))  # time-varying, useful only
    eps = rng.normal(0, 0.25, (N1, T))
    Y = 100.0 + (Z @ theta.T) + (mu @ lam.T) + eps   # useful Z drive outcome time-variation
    return Y, Z

def make_df(Y, Z):
    k = Z.shape[1]; cov = [f"z{m}" for m in range(k)]; rows = []
    for i in range(N1):
        for t in range(T):
            treat = 1 if (i == 0 and (t + 1) > T0) else 0
            rows.append([f"u{i:02d}", t + 1, Y[i, t], treat] + list(Z[i]))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treat"] + cov), cov

def fit_sweep(inp, window, grid, idx23, k1, k):
    bv, *_ = sweep_lambda(inp.X1, inp.X0, inp.Y1, inp.Y0, inp.T0_total, inp.T0_train,
                          lambda_grid=grid, outer_loss_window=window,
                          use_analytical_grad=False, warm_start=False, multi_start=1)
    w = recover_w(bv, inp.X1, inp.X0)
    pn = list(inp.predictor_names)
    vu = np.mean([bv[pn.index(f"z{m}")] for m in range(k1)])           # useful covariates
    vn = np.mean([bv[pn.index(f"z{m}")] for m in range(k1, k)])        # useless covariates
    return metrics(w, inp, idx23) + [float(vu), float(vn)]

def run_draw(b):
    out = {}
    for (k1, k2) in SETTINGS:
        k = k1 + k2
        Y, Z = simulate(k1, k2, 1000 * k1 + b)
        df, cov = make_df(Y, Z)
        inp = prepare_sparse_sc_inputs(df, outcome="y", treat="treat", unitid="unit",
              time="time", covariates=cov, outcome_lag_periods=LAGS,
              T0_train=10, standardize=True)
        names = list(inp.donor_names); idx23 = [names.index("u01"), names.index("u02")]
        # SCM: fixed Mahalanobis V (no diagonal v -> Vuse/Vnoise N/A)
        w_scm = scm_maha_w(inp.X1, inp.X0)
        out[(k1, k2, "SCM")] = metrics(w_scm, inp, idx23) + [float("nan"), float("nan")]
        for m, (win, grid) in SWEEP.items():
            out[(k1, k2, m)] = fit_sweep(inp, win, grid, idx23, k1, k)
    return out

if __name__ == "__main__":
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        draws = list(ex.map(run_draw, range(B)))
    dt = time.perf_counter() - t0
    print(f"B={B} draws in {dt:.0f}s ({4} workers)\n")
    for (k1, k2) in SETTINGS:
        print(f"k1={k1}, k2={k2}:")
        print(f"  {'method':8} {'postMSE':>8} {'|bias|':>7} {'preRMSE':>8} "
              f"{'valRMSE':>8} {'w2+w3':>7} {'Vuse':>6} {'Vnoise':>7}")
        for m in METHODS:
            a = np.array([d[(k1, k2, m)] for d in draws])     # (B,7)
            print(f"  {m:8} {a[:,0].mean():>8.3f} {abs(a[:,1].mean()):>7.3f} "
                  f"{a[:,2].mean():>8.3f} {a[:,3].mean():>8.3f} {a[:,4].mean():>7.3f} "
                  f"{a[:,5].mean():>6.3f} {a[:,6].mean():>7.3f}")
        print()
