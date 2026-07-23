"""Build reference.json for the microsynth_baltimore case from the R config-A
capture (ref_configA.csv). Post-period synthetic Control + treated totals per
model, plus scalar summaries. Run after benchmarks/R/microsynth_baltimore.R."""
import json, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
PRE = {"Central": 27, "Eastern": 62, "Southwestern": 42, "Western": 62}
ref = pd.read_csv(HERE / "ref_configA.csv")
ref = ref[ref.config == "A"]
control, treat, values = {}, {}, {}
for (dist, panel, oc), g in ref.groupby(["district", "panel", "outcome"]):
    g = g.sort_values("period")
    post = g[g.period > PRE[dist]]
    key = f"{dist}/{panel}/{oc}"
    control[key] = [round(float(x), 4) for x in post.control.tolist()]
    treat[key] = [round(float(x), 4) for x in post.treat.tolist()]
    values[f"{key}/cum_control"] = round(float(post.control.sum()), 4)
    values[f"{key}/cum_treat"] = round(float(post.treat.sum()), 4)
out = {"values": values, "control": control, "treat": treat}
(HERE / "reference.json").write_text(json.dumps(out, indent=1))
print(f"wrote reference.json: {len(control)} models, "
      f"{sum(len(v) for v in control.values())} post-period control cells")
