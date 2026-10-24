"""Runner for the #847 study: one JSON line per solve, appended under ``results/``.

From the repository root::

    python -m benchmarks.studies.bnb_external.run syndes noise 40 8 ours
    BIQCRUNCH=/path/to/problems/generic/biqcrunch \\
        python -m benchmarks.studies.bnb_external.run syndes noise 40 8 biqcrunch 1800
    python -m benchmarks.studies.bnb_external.run syndes noise 40 8 scip 1800
    python -m benchmarks.studies.bnb_external.run marex geolift 8 mip 1800

SYNDES solvers: ``ours`` (the library's two-way branch and bound, no work
limit), ``biqcrunch`` (its defaults and its own heuristics), ``seeded`` (given
the value the swap search's seeds reach, less one, as ours starts from them),
``root`` (the root bound only, with and without the product constraints) and
``scip`` (the library's ``backend="mip"`` on the whole two-way design). MAREX
backends: ``exact`` and ``mip``, through ``MAREX`` itself. The last argument is
the time limit in seconds (CPU for BiqCrunch, SCIP's own for the MIQPs).

BLAS runs on one thread, so every solver uses one core.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

#: Where the runner appends its rows.
RESULTS = Path(__file__).resolve().parent / "results"
#: Scratch space for BiqCrunch's input and output files.
WORK = Path(os.environ.get("BNB_EXTERNAL_WORK", "/tmp/bnb_external"))
SOLVERS = ("ours", "biqcrunch", "seeded", "root", "scip")
BACKENDS = ("exact", "mip")


def _problem(name: str, N: int):
    from benchmarks.studies.syndes_global_bnb.run import PANELS
    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    Y = PANELS[name](N)
    problem = TwoWayProblem.from_panel(Y)
    return Y, problem, BoundEngine.from_problem(problem)


def _core(R: np.ndarray, S) -> float:
    sigma = -np.ones(R.shape[0])
    sigma[list(S)] = 1.0
    return float(sigma @ R @ sigma)


def ours(name: str, N: int, K: int) -> dict:
    from mlsynth.utils.syndes_helpers import branch
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    _, problem, engine = _problem(name, N)
    free = list(range(N))
    c0, t0 = time.process_time(), time.perf_counter()
    U, _, _ = branch.lift(*branch.node_parts(engine.R, [], [], free), 2 * K - N, target=None)
    root_cpu = time.process_time() - c0
    c0, t0 = time.process_time(), time.perf_counter()
    out = branch.branch_and_bound(problem, engine, K, SubsetFilter(), work_limit=np.inf)
    cpu, wall = time.process_time() - c0, time.perf_counter() - t0
    S = [int(i) for i in out.treated]
    return {"solver": "ours", "treated": S, "value": out.value, "lower": out.lower,
            "certified": out.certified, "core": _core(engine.R, S),
            "root_core_bound": float(U), "root_cpu": root_cpu, "nodes": out.nodes,
            "lifts": out.lifts, "trials": out.trials, "priced": out.priced,
            "work": out.work, "cpu": cpu, "wall": wall}


def biqcrunch(name: str, N: int, K: int, solver: str, time_limit: float) -> dict:
    from mlsynth.utils.syndes_helpers.branch import RESTARTS, TwoWayLeaves
    from mlsynth.utils.syndes_helpers.gram import maxcut_candidates

    from . import biqcrunch as bc

    binary = os.environ.get("BIQCRUNCH")
    if not binary or not Path(binary).exists():
        raise SystemExit("set BIQCRUNCH to BiqCrunch's built problems/generic/biqcrunch")
    _, problem, engine = _problem(name, N)
    inst = bc.Instance.build(engine.R, K)
    work = WORK / f"{name}_{N}_{K}_{solver}"
    row = {"solver": solver, "scale": inst.scale, "rounding_error": inst.rounding_error()}
    if solver == "root":
        for products in (True, False):
            out = bc.run(inst, binary=binary, workdir=work / str(products),
                         products=products, root=True, time_limit=time_limit)
            key = "products" if products else "plain"
            row[f"root_core_bound_{key}"] = inst.core_value(out["bound"])
            row[f"root_cpu_{key}"] = out["cpu"]
        return row
    soln = None
    if solver == "seeded":
        best = 0
        for S in maxcut_candidates(engine, K, n_restarts=RESTARTS, seed=0):
            z = np.zeros(N, dtype=np.int64)
            z[S] = 1
            best = max(best, inst.integer_value(z))
        soln = best - 1
        row["seed_core"] = inst.core_value(best)
    out = bc.run(inst, binary=binary, workdir=work, time_limit=time_limit, soln_value=soln)
    row.update({k: out[k] for k in ("nodes", "cpu", "wall", "stopped", "gap_percent")})
    row["core_bound"] = inst.core_value(out["bound"])
    row["core_value"] = None if out["value"] is None else inst.core_value(out["value"])
    if out["solution"] is not None:
        S = out["solution"]
        row.update(treated=S, core=_core(engine.R, S),
                   value=float(TwoWayLeaves(problem, engine).value(np.array(S))))
    return row


def scip(name: str, N: int, K: int, time_limit: float) -> dict:
    from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design

    Y, _, engine = _problem(name, N)
    c0, t0 = time.process_time(), time.perf_counter()
    d = solve_synthetic_design(Y, K, mode="global_2way", time_limit=time_limit)
    cpu, wall = time.process_time() - c0, time.perf_counter() - t0
    S = sorted(int(i) for i in d.selected_unit_indices)
    out = d.solve_outcome
    return {"solver": "scip", "treated": S, "value": float(d.objective_value),
            "core": _core(engine.R, S), "status": out.status,
            "stopped": out.stopped_at_limit, "lower": out.dual_bound, "gap": out.gap,
            "cpu": cpu, "wall": wall}


def marex(name: str, m: int, backend: str, time_limit: float) -> dict:
    import pandas as pd

    from benchmarks.studies.marex_bnb.panels import ROOT
    from mlsynth import MAREX

    if name == "geolift":
        df = pd.read_csv(ROOT / "basedata" / "geolift_market_data.csv")
        cfg = {"df": df, "outcome": "Y", "unitid": "location", "time": "date", "T0": 90}
    else:
        from benchmarks.cases.marex_walmart import _BASE, _CFG
        df = pd.read_csv(_BASE / "walmart_weekly_sales_covariates.csv")
        cfg = {"df": df, **{k: v for k, v in _CFG.items() if k != "inference"}}
    cfg.update(m_eq=m, backend=backend, inference=False, display_graph=False)
    if backend == "mip":
        cfg["time_limit"] = time_limit
    c0, t0 = time.process_time(), time.perf_counter()
    res = MAREX(cfg).fit()
    cpu, wall = time.process_time() - c0, time.perf_counter() - t0
    return {"backend": backend, "objective": float(res.globres.objective),
            "treated": sorted(map(str, res.assignment["treated"])), "cpu": cpu,
            "wall": wall}


def main(argv=None) -> int:
    from threadpoolctl import threadpool_limits

    args = list(sys.argv[1:] if argv is None else argv)
    kind, name = args[0], args[1]
    RESULTS.mkdir(exist_ok=True)
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "syndes":
            N, K, solver = int(args[2]), int(args[3]), args[4]
            limit = float(args[5]) if len(args) > 5 else 1800.0
            if solver not in SOLVERS:
                raise ValueError(f"solver must be one of {SOLVERS}")
            if solver == "ours":
                row = ours(name, N, K)
            elif solver == "scip":
                row = scip(name, N, K, limit)
            else:
                row = biqcrunch(name, N, K, solver, limit)
            row.update(panel=name, N=N, K=K, time_limit=limit, load=os.getloadavg()[0])
            path = RESULTS / "syndes.jsonl"
        elif kind == "marex":
            m, backend = int(args[2]), args[3]
            limit = float(args[4]) if len(args) > 4 else 1800.0
            if backend not in BACKENDS:
                raise ValueError(f"backend must be one of {BACKENDS}")
            row = dict(marex(name, m, backend, limit), panel=name, m=m,
                       time_limit=limit, load=os.getloadavg()[0])
            path = RESULTS / "marex.jsonl"
        else:
            raise ValueError(f"unknown kind {kind!r}")
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())
