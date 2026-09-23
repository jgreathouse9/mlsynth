"""Section 5 coverage claim: GP vs segmented regression.

DGPs transcribed from gpits/code/05_simulations.R make_dgp(). Coverage is
mean(|effect - tau0| <= half-width) over post-periods and replications, the
reference's own criterion.

C-ARIMA and CausalImpact are omitted: their R packages need CRAN, which this
environment cannot reach. Segmented regression is the ITS workhorse and the
arm the paper's coverage argument is really against.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from gpits_port import gp_its

NOISE_SD = np.sqrt(0.05)
TAU0 = 0.6
H = 12
SCENARIOS = ("kernel_smooth", "nonlinear_trend", "trend_seasonal")


def kgpl(s, t, b, p):
    d = np.subtract.outer(s, t)
    return (np.exp(-(d**2) / b)
            + np.exp(-2 * np.sin(np.pi * np.abs(d) / p) ** 2 / (b / 2))
            + np.outer(s, t))


def make_dgp(scenario, n_pre, rng):
    t = np.arange(1, n_pre + H + 1, dtype=float)
    pre = np.arange(1, n_pre + 1, dtype=float)
    z = (t - pre.mean()) / pre.std(ddof=1)
    p_dgp = 12 / pre.std(ddof=1)

    beta = rng.uniform(0.8, 1.3)
    A = rng.uniform(0.7, 1.0)
    ph = rng.uniform(0, 2 * np.pi)
    ab = rng.uniform(-0.3, 0.3)
    mu = rng.uniform(8, n_pre - 4) if n_pre > 12 else rng.uniform(4, n_pre)
    wd = rng.uniform(4, 8)
    bump = ab * np.exp(-((t - mu) ** 2) / (2 * wd**2))
    sea = A * np.sin(2 * np.pi * t / 12 + ph)

    if scenario == "kernel_smooth":
        kn = rng.uniform(z.min(), z.max(), 10)
        f = kgpl(z, kn, 2.0, p_dgp) @ rng.standard_normal(10)
        g = 1.5 * (f - f[:n_pre].mean()) / f[:n_pre].std(ddof=1)
    elif scenario == "nonlinear_trend":
        gg = 3.0 * beta * (1 - np.exp(-np.maximum(t - 2, 0) / (n_pre * 0.22)))
        g = (gg - gg[:n_pre].mean()) / gg[:n_pre].std(ddof=1) + 0.7 * sea + bump
    else:
        g = (1.3 * beta * np.sin(2 * np.pi * t / 12 + ph)
             + 0.9 * beta * np.tanh(0.9 * z) + bump)

    y = g + rng.normal(0, NOISE_SD, n_pre + H)
    y[n_pre:] += TAU0
    return y


def est_gp(y, n_pre):
    months = np.array([f"{((i) % 12) + 1:02d}" for i in range(len(y))])
    r = gp_its(y, months, n_pre)
    from scipy.stats import norm
    return r["tau_t"], norm.ppf(0.975) * r["tau_t_se"]


def est_segmented(y, n_pre, degree=1):
    """Linear pre-trend + annual harmonic, OLS prediction interval."""
    t = np.arange(1, len(y) + 1, dtype=float)

    def design(x):
        cols = [np.ones_like(x)] + [x**d for d in range(1, degree + 1)]
        cols += [np.sin(2 * np.pi * x / 12), np.cos(2 * np.pi * x / 12)]
        return np.column_stack(cols)

    Xp, Xf = design(t[:n_pre]), design(t[n_pre:])
    beta, *_ = np.linalg.lstsq(Xp, y[:n_pre], rcond=None)
    resid = y[:n_pre] - Xp @ beta
    dof = n_pre - Xp.shape[1]
    s2 = resid @ resid / dof
    XtXi = np.linalg.pinv(Xp.T @ Xp)
    lev = np.einsum("ij,jk,ik->i", Xf, XtXi, Xf)
    se = np.sqrt(s2 * (1.0 + lev))
    from scipy.stats import t as tdist
    return y[n_pre:] - Xf @ beta, tdist.ppf(0.975, dof) * se


def _fingerprint():
    """sha256 of the two files that determine these numbers."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    for name in ("gpits_port.py", "run_coverage.py"):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()[:16]


def main(n_reps=200, n_grid=(12, 36, 60, 96, 120), seed=1):
    out = []
    for scen in SCENARIOS:
        for n_pre in n_grid:
            rng = np.random.default_rng(abs(hash((scen, n_pre, seed))) % 2**32)
            acc = {k: {"cov": [], "err": [], "hw": []} for k in ("GP", "Segmented")}
            for _ in range(n_reps):
                y = make_dgp(scen, n_pre, rng)
                for name, fn in (("GP", est_gp), ("Segmented", est_segmented)):
                    try:
                        eff, hw = fn(y, n_pre)
                    except Exception:
                        continue
                    acc[name]["cov"].append(np.abs(eff - TAU0) <= hw)
                    acc[name]["err"].append(eff - TAU0)
                    acc[name]["hw"].append(hw)
            row = dict(scenario=scen, n_pre=n_pre, n_reps=n_reps)
            for name in acc:
                cov = np.concatenate(acc[name]["cov"]) if acc[name]["cov"] else np.array([])
                err = np.concatenate(acc[name]["err"]) if acc[name]["err"] else np.array([])
                row[f"{name}_coverage"] = float(cov.mean()) if cov.size else None
                row[f"{name}_rmse"] = float(np.sqrt((err**2).mean())) if err.size else None
                hw = np.concatenate(acc[name]["hw"]) if acc[name]["hw"] else np.array([])
                row[f"{name}_halfwidth"] = float(hw.mean()) if hw.size else None
            out.append(row)
            print(f"{scen:<16} n_pre={n_pre:>4}  "
                  f"GP cov={row['GP_coverage']:.3f} hw={row['GP_halfwidth']:.2f} "
                  f"rmse={row['GP_rmse']:.2f}   |   "
                  f"Seg cov={row['Segmented_coverage']:.3f} "
                  f"hw={row['Segmented_halfwidth']:.2f} "
                  f"rmse={row['Segmented_rmse']:.2f}", flush=True)
    # Stamp run identity into the artifact. A smoke run and a production run
    # write the same filename, so the file must say which one it is; verify.py
    # refuses any artifact whose meta disagrees with manifest.json.
    doc = {"meta": {"n_reps": n_reps, "n_grid": list(n_grid), "seed": seed,
                    "tau0": TAU0, "H": H, "scenarios": list(SCENARIOS),
                    "code_sha256_16": _fingerprint()},
           "rows": out}
    json.dump(doc, open("coverage_results.json", "w"), indent=2)
    print(f"\nwrote coverage_results.json  (n_reps={n_reps}, "
          f"code {doc['meta']['code_sha256_16']})")


if __name__ == "__main__":
    main(n_reps=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
