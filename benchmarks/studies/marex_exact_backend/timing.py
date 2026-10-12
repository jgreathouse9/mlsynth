"""Clean timing of MAREX's two design backends, as shipped.

``solve_design`` (SCIP's MIQP, cvxpy's compilation included, which is what
MAREX pays) against ``solve_design_exact`` (the search), on the Walmart design
of ``docs/marex.rst`` and on 18 simulated one-cluster panels. ``panel`` is the
panel of the prototype study of this search (``marex_exact_search``), copied so
this study stands alone.

Single-threaded on both sides: BLAS threads are pinned before NumPy loads, and
SCIP's LP solver is single-threaded. The search is timed as the median of seven
after a warm-up; SCIP three times where a solve takes seconds and once where it
takes minutes. The load average is logged per case, and every case checks that
the two backends agree.

From the repository root::

    python benchmarks/studies/marex_exact_backend/timing.py \
        benchmarks/studies/marex_exact_backend/results/timing.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, platform
import cvxpy as cp
import numpy as np
import pandas as pd
from mlsynth.utils.marex_helpers.optimization import solve_design
from mlsynth.utils.marex_helpers.exact import solve_design_exact

SIZES = ((12, 3), (16, 4), (20, 5), (25, 5), (30, 5), (40, 5))
SEEDS = (3, 11, 42)


def panel(J=12, T=24, T0=18, seed=3, n_good=5, hi=4.0):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    load = rng.uniform(0.4, 1.6, size=(2, J))
    load[:, :n_good] = load.mean(axis=1, keepdims=True)
    scale = np.full(J, hi)
    scale[:n_good] = 0.05
    Y = f @ load + rng.normal(size=(T, J)) * scale + 20.0
    return pd.DataFrame(Y.T, index=[f"u{j}" for j in range(J)]), T0


def med_time(fn, reps, warm=True):
    if warm:
        fn()
    runs, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs)), out


def case(label, Y, T0, clusters, kw, scip_reps):
    load0 = os.getloadavg()[0]
    ts, rs = med_time(lambda: solve_design_exact(Y, T0, clusters, **kw), 7)
    tm, rm = med_time(lambda: solve_design(Y, T0, clusters, solver=cp.SCIP, **kw),
                      scip_reps, warm=False)
    os_, om = rs["objective"], rm["objective"]
    tr_s = sorted(np.flatnonzero(rs["w_opt"].sum(axis=1) > 1e-8).tolist())
    tr_m = sorted(np.flatnonzero(rm["w_opt"].sum(axis=1) > 1e-8).tolist())
    ok = os_ <= om + 1e-6 * (1 + abs(om)) and abs(os_ - om) <= 1e-5 * (1 + abs(om))
    row = dict(case=label, search_secs=ts, mip_secs=tm, speedup=tm / ts,
               search_obj=os_, mip_obj=om, same_treated=tr_s == tr_m, match=ok,
               load_before=load0, load_after=os.getloadavg()[0])
    print(f"{label:24s} search {ts:8.4f}s  MIQP {tm:8.2f}s  x{tm / ts:8.1f}  "
          f"obj {os_:.10g} vs {om:.10g}  same set {tr_s == tr_m}  match {ok}  "
          f"load {load0:.2f}->{row['load_after']:.2f}", flush=True)
    return row


def main(out):
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"cvxpy {cp.__version__}, threads pinned to 1", flush=True)
    rows = []
    df = pd.read_csv("basedata/walmart_weekly_sales.csv")
    Yw = df.pivot(index="store", columns="week", values="sales")
    rows.append(case("walmart J=45 m=2", Yw, 128, np.zeros(len(Yw), dtype=int),
                     dict(blank_periods=28, m_eq=2, standardize=True), 1))
    for J, m in SIZES:
        for seed in SEEDS:
            Y, T0 = panel(J=J, seed=seed)
            rows.append(case(f"J={J} m={m} seed={seed}", Y, T0,
                             np.zeros(J, dtype=int), dict(m_eq=m),
                             3 if J <= 20 else 1))
    fr = pd.DataFrame(rows)
    fr.to_csv(out, index=False)
    print("\nper size, geometric mean speedup over seeds:")
    fr["size"] = fr["case"].str.extract(r"^(J=\d+ m=\d+)")
    for size, g in fr.dropna(subset=["size"]).groupby("size", sort=False):
        print(f"  {size}: search median {g['search_secs'].median():.4f}s, "
              f"MIQP median {g['mip_secs'].median():.2f}s, "
              f"x{np.exp(np.log(g['speedup']).mean()):.1f}, all match {bool(g['match'].all())}")


if __name__ == "__main__":
    main(sys.argv[1])
