"""google/trimmed_match's own geo pairing on its example design data, run in the reference
environment by run.sh:  run.sh pairs.py {response|default}

TrimmedMatchGeoXDesign with the design colab's windows (design 2020-01-01 to 2020-12-29, evaluation
2020-12-02 to 2020-12-29, cross validation on: pairs are formed on the 336 days before the evaluation
window); create_geo_pairs; the full pairing (no pairs excluded) printed as JSON. ``response`` matches on
the response alone, the paper objective's analogue; ``default`` uses the package's default metrics
(response, and spend with weight 0.01)."""
import json
import sys

import pandas as pd
from trimmed_match.design import trimmed_match_design as tmd
from trimmed_match.design.common_classes import GeoXType, TimeWindow

clone, metrics = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"{clone}/trimmed_match/example_datasets/example_data_for_design.csv",
                 parse_dates=["date"])
design = tmd.TrimmedMatchGeoXDesign(
    GeoXType.HOLD_BACK, pretest_data=df,
    time_window_for_design=TimeWindow(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-29")),
    time_window_for_eval=TimeWindow(pd.Timestamp("2020-12-02"), pd.Timestamp("2020-12-29")),
    response="response", spend_proxy="cost",
    matching_metrics={"response": 1.0} if metrics == "response" else None)
design.create_geo_pairs(use_cross_validation=True)
full = design.pairs[0]
json.dump({"pairs": [[int(a), int(b)] for a, b in zip(full.geo1, full.geo2)],
           "metrics": metrics, "n_geos": int(df.geo.nunique())}, sys.stdout)
