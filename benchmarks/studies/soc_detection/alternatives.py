"""A narrower lever than either correction: let the convex handler take sums.

Both corrections in ``causes.py`` are global switches. ``presolving/donotaggr``
stops aggregation for every row of the model and ``expr/pow/expandmaxexponent``
stops expanding every square of a sum, so either one changes far more than the
two cone rows. The collapsed row, ``-4 + 4u + sum s_i^2 <= 0``, is a convex
function of a sum, and it reaches ``nlhdlr_default`` partly because
``nlhdlr/convex/detectsum`` is off by default: the convex handler declines any
expression whose root is a sum. This measures what turning that on does, on the
same 54 (program, ordering) pairs as ``damage.py``, and compares the result to
the stored baseline and to the configuration where the SOC handler takes the
row. Iterations, cuts, nodes and the root bound are deterministic given the
model, its parameters and the ordering, so pairing against the stored rows is
valid; wall time is not recorded here (see ``timing.py``).

    python alternatives.py results/damage.csv results/alternatives.csv
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damage import ORDERS, PANELS, solve, write_cip

ALTERNATIVES = {"convex_sum": {"nlhdlr/convex/detectsum": True}}
KEY = ["J", "m", "seed", "order"]


def main(damage_csv: str, out: str) -> None:
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        for J, m, seed in PANELS:
            path = write_cip(J, m, seed, tmp)
            for order in ORDERS:
                for name, params in ALTERNATIVES.items():
                    rec = dict(J=J, m=m, seed=seed,
                               order=-1 if order is None else order,
                               config=name)
                    rec.update(solve(path, params, order))
                    records.append(rec)
    alt = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    alt.to_csv(out, index=False)

    dmg = pd.read_csv(damage_csv)
    base = dmg[dmg["config"] == "baseline"].set_index(KEY)
    soc = dmg[dmg["config"] == "no_aggregation"].set_index(KEY)
    a = alt.set_index(KEY).loc[base.index]
    spread = (a["objective"] - base["objective"]).abs().max()
    print(f"pairs: {len(a)}   largest objective gap to baseline: {spread:.2e}")

    def g(x, y, col):
        r = x[col].clip(lower=1) / y[col].clip(lower=1)
        return float(np.exp(np.log(r).mean())), int((r < 1).sum())

    for col in ("lp_iters", "nonlinear_cuts", "nodes"):
        gb, wb = g(a, base, col)
        gs, ws = g(a, soc.loc[base.index], col)
        print(f"{col:15s} convex_sum / baseline = {gb:.3f} (wins {wb}/{len(a)})"
              f"   convex_sum / soc-takes-row = {gs:.3f} (wins {ws}/{len(a)})")
    higher = int(((a["root_bound"] - base["root_bound"]) > 1e-9).sum())
    print(f"root bound higher than baseline in {higher}/{len(a)}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 2
           else ("results/damage.csv", "results/alternatives.csv")))
