"""What a pre-committed design keeps when it stops assuming the spillovers away.

Wei (2026) identifies treatment effects under unknown interference without
naming the contaminated donors. The device is that synthetic-control weights
are not unique: every convex weight vector ``v`` over the donor pool yields

    | theta - v'gamma - tau_hat(v) |  <=  M * d(v)

with ``theta`` the direct effect, ``gamma`` the vector of spillovers on the
control markets, ``tau_hat(v)`` the post-period estimate that ``v`` produces,
and ``d(v)`` that weight's pre-treatment discrepancy. Each weight is one
inequality in the same unknowns, so the weights jointly discipline a set.

The tension with an experimental design is that a design commits to one ``v``
before the experiment runs, which is its whole value, while this identification
consumes the set of them. The resolution this arm tests is that both objects
are pre-period: the weights and their discrepancies are fixed at ``T0``, before
any post-period outcome exists. A design can therefore commit to one ``v`` for
running the experiment and keep the rest for the identification afterwards,
with no specification search, because nothing in the set was chosen after
seeing a result.

With a box restriction on the spillovers the identified set for ``theta`` is a
pair of linear programs. Sampling weights instead of representing the continuum
exactly gives an OUTER approximation: fewer inequalities means a larger set, so
every width here is conservative and the true set is at least this tight.

    python partial_id.py 40 results/partial_id.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from dgps import DGPS
from run import design

N_RANDOM = 60          # interior weights drawn per replication
M_GRID = (1.0, 2.0, 4.0)
B_MULT = (1.0, 2.0)    # spillover bound, as a multiple of the true largest one
N_CONTAM = 3           # control markets carrying a spillover


def weight_sample(n_controls, v_star, rng):
    """The committed weight, every donor vertex, and interior draws."""
    rows = [v_star, np.ones(n_controls) / n_controls]
    rows += list(np.eye(n_controls))
    rows += list(rng.dirichlet(np.ones(n_controls), size=N_RANDOM))
    return np.vstack(rows)


def identified_set(V, tau, d, M, B):
    """[theta_min, theta_max] from the weight inequalities and |gamma| <= B.

    Variables are ``(theta, gamma)``. Each weight contributes two rows:
    ``theta - v'gamma <= tau + M d`` and ``-theta + v'gamma <= -tau + M d``.
    """
    K, m = V.shape
    A = np.vstack([np.column_stack([np.ones(K), -V]),
                   np.column_stack([-np.ones(K), V])])
    b = np.concatenate([tau + M * d, -tau + M * d])
    bounds = [(None, None)] + [(-B, B)] * m
    c = np.zeros(m + 1)
    c[0] = 1.0
    lo = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    hi = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not (lo.success and hi.success):
        return np.nan, np.nan
    return float(lo.x[0]), float(hi.x[0])


def replication(name, seed):
    rng = np.random.default_rng(10_000 + seed)
    YN, _, T0 = DGPS[name](seed)
    T, J = YN.shape
    w, v_full = design(YN, T0, max(2, J // 6))
    treated = np.flatnonzero(w > 1e-8)
    controls = np.flatnonzero(v_full > 1e-12)
    m = len(controls)
    if m < 4:
        raise ValueError(f"only {m} control markets carry weight")
    v_star = v_full[controls] / v_full[controls].sum()

    scale = float(np.median(YN[:T0].std(axis=0)))
    theta_true = scale
    gamma = np.zeros(m)
    hit = rng.choice(m, size=min(N_CONTAM, m - 1), replace=False)
    gamma[hit] = rng.normal(0.0, 0.5 * scale, size=len(hit))

    post = slice(T0, T)
    Y = YN.copy()
    Y[post, treated] += theta_true
    for i, j in enumerate(controls):
        Y[post, j] += gamma[i]

    treated_agg = Y @ w
    V = weight_sample(m, v_star, rng)
    ctrl = Y[:, controls]
    gap = treated_agg[:, None] - ctrl @ V.T                  # (T, K)
    d = np.sqrt((gap[:T0] ** 2).mean(axis=0))
    tau = gap[post].mean(axis=0)

    star = np.zeros((1, m)); star[0] = v_star
    d_star = np.sqrt((( treated_agg[:T0] - ctrl[:T0] @ v_star) ** 2).mean())
    tau_star = float((treated_agg[post] - ctrl[post] @ v_star).mean())

    rows = []
    for M in M_GRID:
        for bm in B_MULT:
            B = bm * float(np.abs(gamma).max()) if np.abs(gamma).max() > 0 else bm * scale
            lo_f, hi_f = identified_set(V, tau, d, M, B)
            lo_s, hi_s = identified_set(star, np.array([tau_star]),
                                        np.array([d_star]), M, B)
            rows.append(dict(
                dgp=name, seed=seed, M=M, b_mult=bm, B=B, n_controls=m,
                n_weights=V.shape[0], theta_true=theta_true, tau_star=tau_star,
                d_star=float(d_star), scale=scale,
                full_lo=lo_f, full_hi=hi_f, full_width=hi_f - lo_f,
                single_lo=lo_s, single_hi=hi_s, single_width=hi_s - lo_s,
                full_covers=bool(lo_f <= theta_true <= hi_f),
                single_covers=bool(lo_s <= theta_true <= hi_s),
                naive_err=tau_star - theta_true))
    return rows


def main(reps, out):
    warnings.filterwarnings("ignore")
    rows, fails = [], {}
    for name in DGPS:
        for seed in range(reps):
            try:
                rows += replication(name, seed)
            except Exception as exc:                       # noqa: BLE001
                fails[name] = fails.get(name, 0) + 1
                if fails[name] <= 2:
                    print(f"{name} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        print(f"done {name} ({fails.get(name, 0)} failures)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}; failures={fails}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
