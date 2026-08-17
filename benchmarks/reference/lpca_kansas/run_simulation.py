"""Path B: Table 1 of Feng (2024), Section 5.

    python benchmarks/reference/lpca_kansas/run_simulation.py [replications]

Writes ``simulation_results.json`` and prints the comparison against the
published table. The paper runs 2 000 replications; the default here is smaller
and the Monte Carlo standard error of every cell is reported alongside it, so
the comparison stays honest about its own precision.
"""

from __future__ import annotations

import os

# One BLAS thread per worker. Each replication is thousands of small
# eigendecompositions, which multithreaded BLAS cannot speed up but will happily
# oversubscribe: four workers times four threads on four cores thrashes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json      # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation import MODELS, N, P, k_sequence, one_replication  # noqa: E402

DEFAULT_REPLICATIONS = 300
ROOT_SEED = 20240731

# Feng (2024) Table 1: {model: {column: (MAE, q10, q50, q90)}}. Identical to the
# November 2023 version's table.
PAPER = {
    "Model 1": {49: (0.680, 0.075, 0.077, 0.076), 99: (0.937, 0.068, 0.078, 0.068),
                149: (2.203, 0.091, 0.119, 0.088), "gpca": (1.148, 0.111, 0.095, 0.112)},
    "Model 2": {49: (0.599, 0.067, 0.066, 0.071), 99: (0.636, 0.055, 0.054, 0.054),
                149: (0.706, 0.051, 0.052, 0.051), "gpca": (0.870, 0.111, 0.097, 0.112)},
    "Model 3": {49: (0.475, 0.035, 0.038, 0.034), 99: (0.461, 0.032, 0.038, 0.032),
                149: (0.461, 0.039, 0.043, 0.039), "gpca": (0.470, 0.047, 0.047, 0.046)},
}
ROWS = ("MAE", "q_alpha=.1", "q_alpha=.5", "q_alpha=.9")


def _task(job):
    model_index, seed = job
    return model_index, one_replication(MODELS[model_index],
                                        np.random.default_rng(seed))


def main(replications: int = DEFAULT_REPLICATIONS) -> int:
    seeds = np.random.SeedSequence(ROOT_SEED).spawn(len(MODELS) * replications)
    jobs = [(m, seeds[m * replications + r].generate_state(4))
            for m in range(len(MODELS)) for r in range(replications)]

    started = time.time()
    collected = {m: [] for m in range(len(MODELS))}
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        for done, (model_index, result) in enumerate(
                pool.imap_unordered(_task, jobs, chunksize=1), start=1):
            collected[model_index].append(result)
            if done % 50 == 0 or done == len(jobs):
                rate = (time.time() - started) / done
                print(f"  {done}/{len(jobs)} replications "
                      f"({rate:.1f} s each, {rate * (len(jobs) - done) / 60:.0f} min left)",
                      flush=True)

    out = {"replications": replications, "n": N, "p": P, "root_seed": ROOT_SEED,
           "K_grid": k_sequence(), "models": {}}
    for index, model in enumerate(MODELS):
        per_column = {}
        for column in k_sequence() + ["gpca"]:
            stacked = np.array([rep[column] for rep in collected[index]])
            per_column[str(column)] = {
                "mean": [float(v) for v in stacked.mean(axis=0)],
                "mc_se": [float(v) for v in
                          stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked))],
                "paper": list(PAPER[model["name"]][column]),
            }
        out["models"][model["name"]] = per_column

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "simulation_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    _report(out)
    print(f"\nwrote {path}  ({(time.time() - started) / 60:.1f} min)")
    return 0


def _report(out):
    columns = [str(k) for k in out["K_grid"]] + ["gpca"]
    print(f"\nTable 1, {out['replications']} replications "
          f"(paper: 2 000). Cells are `ported (mc se) | paper`.\n")
    head = "".join(f"{('K=' + c if c != 'gpca' else 'GPCA'):>22s}" for c in columns)
    print(f"{'':14s}{head}")
    for name, per_column in out["models"].items():
        print(name)
        for row, label in enumerate(ROWS):
            cells = ""
            for column in columns:
                cell = per_column[column]
                cells += (f"{cell['mean'][row]:8.3f} "
                          f"({cell['mc_se'][row]:.3f}) "
                          f"|{cell['paper'][row]:6.3f}")
            print(f"  {label:12s}{cells}")


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else
                          DEFAULT_REPLICATIONS))
