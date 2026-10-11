"""Clean timing: the exact search against SCIP's MIQP, alone on the machine.

Every earlier time in this study came from runs beside other work. This one is
meant to be quotable, so it controls what it can:

* single-threaded on both sides -- SCIP's LP solver is single-threaded, and
  the BLAS threads behind NumPy are pinned to one before NumPy is imported, so
  the comparison is one core against one core;
* a warm-up run of the search, then the median of seven; SCIP three times
  where a solve takes seconds, once where it takes minutes and timer noise is
  negligible, with the node count required to agree across repeats;
* SCIP timed both ways: wall time of ``prob.solve`` on a freshly built cvxpy
  problem, which is what MAREX pays, and SCIP's own solving time, which leaves
  out cvxpy's compilation;
* the one-minute load average recorded before and after every case, as the
  evidence that nothing else was running;
* every instance checked: the search's objective has to match SCIP's.

    python timing.py results/timing.csv
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panels import fit_matrices, program  # noqa: E402
from search import enumerate_design, exact_design  # noqa: E402

SIZES = ((12, 3), (16, 4), (20, 5), (25, 5), (30, 5), (40, 5))
SEEDS = (3, 11, 42)
SEARCH_REPEATS = 7
ENUMERATE_UP_TO = 20


def scip_repeats(J: int) -> int:
    return 3 if J <= 20 else 1


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:                                   # pragma: no cover
        pass
    return platform.processor() or "unknown"


def time_search(B, A, m):
    exact_design(B, A, m)                             # warm-up
    runs = []
    for _ in range(SEARCH_REPEATS):
        t0 = time.perf_counter()
        res = exact_design(B, A, m)
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs)), res, runs


def time_enumeration(B, A, m):
    runs, res = [], None
    for _ in range(3):
        t0 = time.perf_counter()
        res = enumerate_design(B, A, m)
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs)), res


def time_scip(J, m, seed):
    walls, solving, nodes, value, gap = [], [], set(), None, None
    for _ in range(scip_repeats(J)):
        prob = program(J, m, seed)                    # fresh: cvxpy recompiles
        t0 = time.perf_counter()
        prob.solve(solver=cp.SCIP)
        walls.append(time.perf_counter() - t0)
        model = prob.solver_stats.extra_stats["model"]
        solving.append(model.getSolvingTime())
        nodes.add(model.getNNodes())
        value, gap = float(prob.value), model.getGap()
    return dict(scip_wall=float(np.median(walls)),
                scip_solving=float(np.median(solving)),
                scip_nodes=sorted(nodes), scip_objective=value, scip_gap=gap,
                scip_repeats=len(walls))


def main(out: str) -> None:
    env = (f"cpu: {cpu_model()}\ncores visible: {os.cpu_count()}\n"
           f"python {platform.python_version()}, numpy {np.__version__}, "
           f"cvxpy {cp.__version__}\nthreads pinned to 1 "
           f"(OMP/OPENBLAS/MKL/VECLIB/NUMEXPR)\n")
    print(env, flush=True)
    rows, raw = [], []
    for J, m in SIZES:
        for seed in SEEDS:
            load_before = os.getloadavg()[0]
            B, A = fit_matrices(J, seed)
            search_secs, res, runs = time_search(B, A, m)
            raw += [dict(J=J, m=m, seed=seed, method="search", secs=s) for s in runs]
            rec = dict(J=J, m=m, seed=seed, candidates=res["candidates"],
                       search_secs=search_secs,
                       search_objective=res["objective"],
                       search_treated=res["treated"],
                       exact_g=res["exact_g"], exact_h=res["exact_h"],
                       bound_secs=res["stage_secs"]["bound"])
            if J <= ENUMERATE_UP_TO:
                enum_secs, enum = time_enumeration(B, A, m)
                rec["enumerate_secs"] = enum_secs
                rec["enumerate_match"] = abs(enum["objective"] - res["objective"]) <= 1e-9
            rec.update(time_scip(J, m, seed))
            rec["match"] = (abs(rec["search_objective"] - rec["scip_objective"])
                            <= 1e-4 * (1 + abs(rec["scip_objective"])))
            rec["nodes_deterministic"] = len(rec["scip_nodes"]) == 1
            rec["load_before"] = load_before
            rec["load_after"] = os.getloadavg()[0]
            rec["speedup_wall"] = rec["scip_wall"] / rec["search_secs"]
            rec["speedup_solving"] = rec["scip_solving"] / rec["search_secs"]
            rows.append(rec)
            print(f"J={J:2d} m={m} seed={seed:2d}  search {search_secs:7.3f}s"
                  f"  SCIP {rec['scip_wall']:8.2f}s wall / {rec['scip_solving']:8.2f}s"
                  f" solving  nodes={rec['scip_nodes']}  match={rec['match']}"
                  f"  load {load_before:.2f}->{rec['load_after']:.2f}", flush=True)
            if not rec["match"]:
                print("  MISMATCH", rec["search_objective"], rec["scip_objective"],
                      flush=True)
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)
    pd.DataFrame(raw).to_csv(out.replace(".csv", "_raw.csv"), index=False)
    with open(out.replace(".csv", "_env.txt"), "w") as fh:
        fh.write(env)

    print("\nper size, geometric mean over seeds:")
    for (J, m), grp in frame.groupby(["J", "m"]):
        gw = float(np.exp(np.log(grp["speedup_wall"]).mean()))
        gs = float(np.exp(np.log(grp["speedup_solving"]).mean()))
        print(f"  J={J:2d} m={m}: search {grp['search_secs'].median():.3f}s  "
              f"SCIP {grp['scip_wall'].median():.2f}s  "
              f"speedup {gw:7.1f}x wall, {gs:7.1f}x vs SCIP solving time  "
              f"all match: {bool(grp['match'].all())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/timing.csv")
