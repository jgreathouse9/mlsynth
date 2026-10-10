"""Wall time for the baseline against the handler taking the row, on a quiet machine.

``damage.py`` records time, but it ran beside other solves and its time column
is contaminated by contention. Iterations, cuts, nodes and the root bound are
deterministic given the model and its parameters, so those stand; wall time
is not, and it is measured here alone. Each (program, ordering) is solved
under both configurations back to back, in alternating order, three times,
and the median of the three is kept.

    python timing.py results/timing.csv
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import pandas as pd
from pyscipopt import Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causes import CONFIGS
from damage import write_cip

PROGRAMS = [(20, 5, seed) for seed in (3, 11, 42)] + [(16, 4, 3)]
ORDERS = (None, 1, 2, 3)
PAIR = ("baseline", "no_aggregation")
REPEATS = 3


def once(path: str, params: dict, order) -> float:
    model = Model()
    model.hideOutput()
    model.readProblem(path)
    for key, val in params.items():
        model.setParam(key, val)
    if order is not None:
        model.setParam("randomization/permutationseed", order)
        model.setParam("randomization/permuteconss", True)
        model.setParam("randomization/permutevars", True)
    t0 = time.perf_counter()
    model.optimize()
    return time.perf_counter() - t0


def main(out: str) -> None:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for J, m, seed in PROGRAMS:
            path = write_cip(J, m, seed, tmp)
            for order in ORDERS:
                times = {name: [] for name in PAIR}
                for rep in range(REPEATS):
                    sequence = PAIR if rep % 2 == 0 else PAIR[::-1]
                    for name in sequence:
                        times[name].append(once(path, CONFIGS[name], order))
                rec = dict(J=J, m=m, seed=seed,
                           order=-1 if order is None else order)
                for name in PAIR:
                    rec[f"{name}_secs"] = sorted(times[name])[REPEATS // 2]
                rec["ratio"] = rec["no_aggregation_secs"] / rec["baseline_secs"]
                rows.append(rec)
                print(f"J={J} seed={seed} order={rec['order']:2d}  "
                      f"baseline={rec['baseline_secs']:.3f}s  "
                      f"soc={rec['no_aggregation_secs']:.3f}s  "
                      f"ratio={rec['ratio']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)
    import numpy as np
    g = float(np.exp(np.log(frame["ratio"]).mean()))
    print(f"\ngeometric mean ratio (soc / baseline): {g:.3f}   "
          f"faster in {int((frame['ratio'] < 1).sum())}/{len(frame)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/timing.csv")
