"""Does DiD's bias survive redrawing the factors?

Appendix A.f of the paper says the common factors ``lambda`` and the
loading coefficients ``gamma`` "were independently drawn once" and are held
"constant across all simulations", and prints the realised matrices (eq. 26
and eq. 27). The shipped driver
(``Simulation/3_Simulation_CSC_IDiD_FDiD.R``) does the opposite: it draws
fresh seeds inside the replication loop, so every replication gets a new
lambda and gamma.

The distinction decides the paper's headline. Under selection on the loadings,
DiD's bias for one panel is driven by
``E[mu | D = 1] - E[mu | D = 0]`` contracted with ``lambda_T - lambda_pre``.
With lambda held fixed, that contraction has a fixed sign and accumulates over
replications. With lambda redrawn, it changes sign across replications and
averages away, so the mean estimation error the paper tabulates goes to zero
even though each individual panel is biased.

This script runs both protocols on the same DGP and reports the average
estimation error and its absolute counterpart, so a bias that averages to
zero is still visible as dispersion.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from baselines import idid, psc_weights_all, twfe_did
from core import csc_counterfactual, csc_weights
from dgp import simulate

# eq. (26): the paper's realised (T x F) common factors, and eq. (27): the
# realised (K x F) loading coefficients.
LAMBDA_PAPER = np.array(
    [
        [1.79, 2.44, 2.49, 2.31],
        [3.27, 2.11, 2.09, 1.55],
        [4.08, 2.52, 2.16, 3.58],
        [0.65, 2.00, 5.41, 1.98],
        [3.43, 2.22, 3.13, 2.99],
        [3.51, 3.06, 2.51, 2.06],
        [2.43, 3.96, 2.56, 4.10],
        [2.45, 2.89, 3.46, 2.52],
    ]
)
GAMMA_PAPER = np.array(
    [
        [-1.48, -0.32, -0.78, 0.51],
        [1.58, -0.63, 0.01, -0.29],
        [-0.96, -0.11, -0.15, 0.22],
        [-0.92, 0.43, -0.70, 2.01],
        [-2.00, -0.78, 1.19, 1.01],
        [-0.27, -1.29, 0.34, -0.30],
    ]
)


def one(args):
    seed, cfg = args
    p = simulate(rng=np.random.default_rng(seed), **cfg)
    tr, do = p.treated_idx, p.donor_idx
    if tr.size < 2 or do.size < 2:
        return None
    T0 = p.Y.shape[1] - 1
    Y1_pre, Y0_pre, X1 = p.Y[tr, :T0], p.Y[do, :T0], p.X_dummies[tr]
    Y1_post, Y0_post = p.Y[tr, T0:], p.Y[do, T0:]

    out = {}
    try:
        fit = csc_weights(Y1_pre, Y0_pre, X1, ridge=1e-2)
    except Exception:  # noqa: BLE001
        return None
    out["csc"] = float(np.mean(Y1_post - csc_counterfactual(fit, Y0_post))) - p.tau
    out["fdid"] = twfe_did(p.Y, p.D)[0] - p.tau
    out["idid"] = idid(p.Y, p.D, p.mu, p.lam, p.mu_mean, p.lambda_mean)[0] - p.tau
    W, _ = psc_weights_all(Y1_pre, Y0_pre)
    out["psc"] = float(np.mean(Y1_post - W @ Y0_post)) - p.tau
    return out


def run(name, cfg, reps, workers):
    jobs = [(3000 + 7919 * i, cfg) for i in range(reps)]
    with mp.Pool(workers) as pool:
        res = [r for r in pool.map(one, jobs, chunksize=2) if r]
    print(f"  {name}  ({len(res)} panels)")
    for m in ("csc", "fdid", "psc", "idid"):
        v = np.array([r[m] for r in res])
        print(
            f"    {m:<5} mean err {v.mean():+6.2f}   mean |err| {np.abs(v).mean():5.2f}"
            f"   sd {v.std(ddof=1):5.2f}"
        )


if __name__ == "__main__":
    base = dict(N=100, T=5, num_categories=5, num_factors=2, prob_tr=0.15)
    reps, workers = 150, max(1, mp.cpu_count() - 2)
    print("true tau = 1.0; CSC solved with the ridge tie-break\n")
    print("factors redrawn every replication (what the shipped driver does)")
    run("redrawn", base, reps, workers)
    print("\nfactors held at the paper's printed draw (what Appendix A.f says)")
    run(
        "fixed",
        dict(base, fixed_lam=LAMBDA_PAPER, fixed_Gamma=GAMMA_PAPER),
        reps,
        workers,
    )
