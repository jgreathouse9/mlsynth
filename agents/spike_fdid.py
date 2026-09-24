"""Would FDID's set have rescued greedy's k=2 hole on Prop 99?"""
import numpy as np, pandas as pd
from mlsynth import FDID
from mlsynth.utils.fscm_helpers.setup import prepare_fscm_inputs
from mlsynth.utils.fscm_helpers.estimation import _fit_weights, _outcome_rmspe

df = pd.read_csv('basedata/P99data.csv')
df['treat'] = ((df['state']=='California')&(df['year']>=1989)).astype(int)
inp = prepare_fscm_inputs(df, unitid='state', time='year', outcome='cigsale', treat='treat')
labels = [str(x) for x in inp.donor_labels]; full = slice(0, inp.T0)

def score(names):
    idx = [labels.index(n) for n in names]
    return _outcome_rmspe(inp, idx, _fit_weights(inp, idx, full, Pt=None,Pd=None,v=None), full)

from mlsynth.config_models import FDIDConfig
r = FDID(FDIDConfig(df=df, unitid='state', time='year', outcome='cigsale',
                    treat='treat', display_graphs=False)).fit()
fd = r[0] if isinstance(r, list) else r
picked = None
for attr in ("weights",):
    w = getattr(fd, attr, None)
    if w is not None and getattr(w, "donor_weights", None):
        picked = [k for k, v in w.donor_weights.items() if abs(v) > 1e-9]
print(f"FDID selected {len(picked)} donors: {sorted(picked)}")
print()
print(f"{'set':<46}{'size':>5}{'FSCM in-sample RMSPE':>23}")
print("-"*74)
rows = [("greedy k=2   {Montana, Nevada}", ["Montana","Nevada"]),
        ("exhaustive k=2  {Nevada, New Mexico}", ["Nevada","New Mexico"]),
        ("greedy k=3   {Montana, Nevada, Utah}", ["Montana","Nevada","Utah"])]
if picked:
    rows.append((f"FDID's set (k={len(picked)})", sorted(picked)))
    for k in (2, 3):
        if len(picked) >= k:
            rows.append((f"FDID's first {k} by |weight|", sorted(picked)[:k]))
for label, names in rows:
    try: print(f"{label:<46}{len(names):>5}{score(names):>23.8f}")
    except Exception as e: print(f"{label:<46}{len(names):>5}{'n/a':>23}  ({e})")
