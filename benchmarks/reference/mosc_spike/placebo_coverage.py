"""Placebo coverage of MOSC's interval on the authors' own control panels.

Every team here kept fans out of its stadium for the whole 2020 season, so the
effect is zero by construction and any interval excluding zero is a false
positive. The paper states that the coverage of its bootstrap is evaluated in
Section 5; that evaluation does not appear, so this supplies it.

Three interval forms are computed from one set of replicates, since the choice
between them is empirical and this is the evidence that settled it. Measured:
percentile 9/10, basic (reflected) 7/10, mean-shift recentring 6/10, against a
posterior band of 4/10. The estimator ships the percentile form.

The one panel no form covers is Minnesota, whose point estimate misses by 21
percent of the outcome. That is a counterfactual that is wrong, not an interval
that is narrow, and no inference procedure repairs it.

    python benchmarks/reference/mosc_spike/placebo_coverage.py --panels <dir>

Panels come from ``extract_panels.py``; the ten here are the control teams among
the authors' 31, at the ranks their own notebook selected.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from mlsynth import MOSC

import argparse
from pathlib import Path

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--panels", type=Path, required=True,
                help="directory of parquet panels written by extract_panels.py")
S = str(_p.parse_args().panels)
K = {"ny_giants": 10, "washington": 5, "new_orleans": 20, "seattle": 5,
     "new_england": 9, "minnesota": 5, "detroit": 10, "ny_jets": 10, "chicago": 20,
     "green_bay": 15}
A = 0.05

rows = []
for slug, k in K.items():
    long = pd.read_parquet(f"{S}/{slug}.parquet").rename(columns={"county": "unit", "date": "day"})
    r = MOSC(dict(df=long, outcome="cases", treat="stadium_open", unitid="unit", time="day",
                  factor_model="ppca", outcome_scale="difference", n_factors=k,
                  n_samples=100, n_warmup=100, n_bootstrap=150, inference="bootstrap",
                  seed=0, display_graphs=False)).fit()
    obs = np.asarray(r.inputs.y_target, float); pre = r.inputs.pre_periods
    theta = r.att
    reps = np.asarray(r.posterior.bootstrap_counterfactual, float)
    att_star = (obs[None, pre:] - reps).mean(axis=1)          # replicate ATTs
    lo_q, hi_q = np.percentile(att_star, [100*A/2, 100*(1-A/2)])

    forms = {
        "percentile":  (lo_q, hi_q),
        "basic":       (2*theta - hi_q, 2*theta - lo_q),
        "mean-shift":  (theta + (lo_q - att_star.mean()), theta + (hi_q - att_star.mean())),
    }
    rows.append({"team": slug, "att": theta,
                 **{n: (lo <= 0 <= hi) for n, (lo, hi) in forms.items()},
                 "miss_pct": abs(theta) / obs[-1]})

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
print("\ncoverage of the nominal 95% interval, 10 placebo panels")
for form in ("percentile", "basic", "mean-shift"):
    print(f"  {form:12} {df[form].sum()}/{len(df)} = {df[form].mean():.0%}")
big = df[df.miss_pct > 0.02]
print(f"\npanels whose point estimate misses by >2% of the outcome: {len(big)}")
print(f"  of those, covered by the best form: {big[['percentile','basic','mean-shift']].max(axis=1).sum()}/{len(big)}")
