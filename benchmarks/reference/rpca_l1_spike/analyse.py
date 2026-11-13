"""Summarise sim.csv: median post-period RMSE by noise x outlier pattern, the
share of cells each method wins, and the certificate rate of the l1 fit."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METHODS = ["oracle", "pcp_1000", "pcp_conv", "hqf", "hsvt_usvt", "hsvt_true", "l1_true", "l1_usvt"]


def main():
    df = pd.read_csv(HERE / "sim.csv")
    fit = df[df.method.isin(METHODS)]
    med = fit.pivot_table(index=["sigma", "pattern"], columns="method", values="post_rmse",
                          aggfunc="median")[METHODS]
    print("median post-period RMSE against the untreated signal\n")
    print(med.round(3).to_string())

    # head to head per panel, excluding the oracle
    comp = [m for m in METHODS if m != "oracle"]
    wide = fit.pivot_table(index=["seed", "sigma", "pattern"], columns="method", values="post_rmse")[comp]
    wins = wide.idxmin(axis=1).value_counts(normalize=True).reindex(comp).fillna(0)
    print("\nshare of panels each method has the lowest post RMSE\n")
    print(wins.round(3).to_string())
    l1_vs = {m: float((wide["l1_true"] < wide[m]).mean()) for m in comp if m != "l1_true"}
    l1u_vs = {m: float((wide["l1_usvt"] < wide[m]).mean()) for m in comp if m != "l1_usvt"}

    cert = df[df.method == "l1_true_cert"]
    cert_rate = cert.groupby(["sigma", "pattern"])["pre_rmse"].mean()
    print("\nshare of panels where the rank-3 l1 fit certifies as critical\n")
    print(cert_rate.round(2).to_string())
    rank = df[df.method == "usvt_rank"].groupby(["sigma", "pattern"])["post_rmse"].agg(
        lambda s: float(np.mean(s == 3)))
    print("\nshare of panels where USVT picks the true rank 3\n")
    print(rank.round(2).to_string())
    lerr = fit.pivot_table(index=["sigma", "pattern"], columns="method", values="L_err",
                           aggfunc="median")[METHODS[1:]]
    print("\nmedian relative error of the denoised donor matrix\n")
    print(lerr.round(3).to_string())

    out = {"median_post_rmse": {f"{s}|{p}": row.round(4).to_dict() for (s, p), row in med.iterrows()},
           "win_share": wins.round(4).to_dict(),
           "l1_true_beats": {k: round(v, 3) for k, v in l1_vs.items()},
           "l1_usvt_beats": {k: round(v, 3) for k, v in l1u_vs.items()},
           "l1_true_certified_share": {f"{s}|{p}": round(float(v), 3) for (s, p), v in cert_rate.items()},
           "usvt_picks_true_rank": {f"{s}|{p}": round(float(v), 3) for (s, p), v in rank.items()},
           "panels": int(len(wide))}
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nl1_true beats:", out["l1_true_beats"])
    print("l1_usvt beats:", out["l1_usvt_beats"])


if __name__ == "__main__":
    main()
