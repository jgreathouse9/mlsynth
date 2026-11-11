"""Summarise the spike's CSVs into results.json and print markdown tables."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def table(df, metric, order=None):
    sub = df[df.metric == metric]
    piv = sub.pivot_table(index="method", columns="setting", values="value", aggfunc="mean")
    if order:
        piv = piv.reindex([m for m in order if m in piv.index])
    return piv


def md(piv, digits=3):
    cols = list(piv.columns)
    lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in piv.iterrows():
        lines.append(f"| {idx} | " + " | ".join(f"{v:.{digits}f}" for v in row.values) + " |")
    return "\n".join(lines)


def main():
    out = {}
    order = ["pmf", "pmf_default", "pmfcf_mletr", "sc_means", "mle_sc", "mle_mcnnm",
             "mom_mcnnm", "momcf_pmftr", "pmfcf_momtr", "dsc", "truth"]
    for name, metrics in [("table1", ["outcome_mae", "eta1_effect_mae", "cf_eta1_mae"]),
                          ("table2", ["kl_mae", "cf_eta1_mae", "cf_mean_mae", "kl_mean"]),
                          ("table2_swap", ["kl_mae", "tr_eta1_mae"])]:
        path = HERE / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        out[name] = {"reps": int(df.rep.nunique())}
        for metric in metrics:
            piv = table(df, metric, order)
            out[name][metric] = piv.round(4).to_dict()
            print(f"\n### {name}: {metric} (reps={df.rep.nunique()})\n\n{md(piv)}")
    path = HERE / "multinomial.csv"
    if path.exists():
        df = pd.read_csv(path)
        out["multinomial"] = {"reps": int(df.rep.nunique())}
        morder = ["efsc", "efsc_cf_obs_tr", "logit_mcnnm", "compsc_halfm", "compsc", "truth"]
        for metric in ["cf_prob_mae", "cell_effect_mae", "avg_effect_abs_err",
                       "medicaid_avg_effect", "kl_mae", "zero_share_cells"]:
            piv = table(df, metric, morder)
            out["multinomial"][metric] = piv.round(4).to_dict()
            print(f"\n### multinomial: {metric} (reps={df.rep.nunique()})\n\n{md(piv, 4)}")
    for path in sorted(HERE.glob("placebo_*.csv")):
        df = pd.read_csv(path)
        scen = path.stem.replace("placebo_", "")
        p = df.p.to_numpy()
        res = {"panels": int(len(df)), "reject_at_0.05": float(np.mean(p <= 0.05)),
               "reject_at_0.10": float(np.mean(p <= 0.10)), "mean_p": float(p.mean()),
               "observed_mean": float(df.observed.mean()),
               "placebo_mean": float(df.placebo_mean.mean()),
               "share_observed_above_all_placebos": float(np.mean(df.observed > df.placebo_max))}
        out[f"placebo_{scen}"] = res
        print(f"\n### placebo {scen}\n\n" + "\n".join(f"- {k}: {v:.4g}" if isinstance(v, float)
                                                       else f"- {k}: {v}" for k, v in res.items()))
    (HERE / "results.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
