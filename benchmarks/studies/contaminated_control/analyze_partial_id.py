"""Tables for the partial-identification arm.

    python analyze_partial_id.py results/partial_id.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main(path: str) -> None:
    d = pd.read_csv(path)
    d["feasible"] = d.full_width.notna()

    print("=" * 88)
    print("Feasibility. An infeasible program is a result: no direct effect and")
    print("spillover vector satisfies every weight's inequality, so the envelope")
    print("and the spillover bound together are refuted by the data.")
    print("=" * 88)
    print(d.pivot_table(index="dgp", columns="M", values="feasible",
                        aggfunc="mean").round(3).to_string())

    print("\n" + "=" * 88)
    print("Coverage of the true effect, among the programs that are feasible")
    print("=" * 88)
    f = d[d.feasible]
    print(f.pivot_table(index="dgp", columns="M", values="full_covers",
                        aggfunc="mean").round(3).to_string())
    print(f"\n  overall, feasible only: full {f.full_covers.mean():.3f}   "
          f"single {f.single_covers.mean():.3f}")

    print("\n" + "=" * 88)
    print("What the weight set buys: width with one committed weight against")
    print("width with the whole pre-period weight sample")
    print("=" * 88)
    f = f.copy()
    f["reduction"] = 1.0 - f.full_width / f.single_width
    print(f.pivot_table(index="dgp", columns="M",
                        values="reduction", aggfunc="mean").round(3).to_string())
    print("\n  by envelope, pooled across DGPs:")
    print(f.groupby("M").agg(single=("single_width", "mean"),
                             full=("full_width", "mean"),
                             reduction=("reduction", "mean"),
                             best=("reduction", "max")).round(3).to_string())

    print("\n" + "=" * 88)
    print("Width against the committed estimate's own error")
    print("=" * 88)
    print(f.groupby("dgp").agg(theta=("theta_true", "mean"),
                               abs_naive_err=("naive_err", lambda s: s.abs().mean()),
                               single=("single_width", "mean"),
                               full=("full_width", "mean")).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
