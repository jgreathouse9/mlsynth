"""Is FSCM's greedy order leaving anything on the table at the chosen size?"""
import itertools, time, numpy as np, pandas as pd
from mlsynth.utils.fscm_helpers.setup import prepare_fscm_inputs
from mlsynth.utils.fscm_helpers.estimation import _fit_weights, _outcome_rmspe, _forward_select
from mlsynth.utils.fscm_helpers.config import FSCMConfig

def load(name):
    if name == "prop99":
        df = pd.read_csv('basedata/P99data.csv'); u,t,o,tr,yr='state','year','cigsale','California',1989
    else:
        df = pd.read_csv('basedata/basque_data.csv'); df = df[df['regionname']!='Spain (Espana)']
        u,t,o,tr,yr='regionname','year','gdpcap','Basque Country (Pais Vasco)',1975
    df['treat'] = ((df[u]==tr)&(df[t]>=yr)).astype(int)
    return prepare_fscm_inputs(df, unitid=u, time=t, outcome=o, treat="treat")

for name in ("prop99", "basque"):
    inp = load(name)
    J, T0 = inp.n_donors, inp.T0
    full = slice(0, T0)
    origins = np.arange(max(2, T0 // 2), T0)
    sel, path = _forward_select(inp, origins, J, Pt=None, Pd=None, v=None)
    k = path.optimal_size
    labels = [str(x) for x in inp.donor_labels]
    print(f"\n=== {name}: J={J} T0={T0}  greedy chose k={k} ===")

    for size in range(1, min(k, 3) + 1):
        greedy = sorted([str(x) for x in path.order[:size]])
        gw = _fit_weights(inp, [labels.index(g) for g in greedy], full, Pt=None,Pd=None,v=None)
        gscore = _outcome_rmspe(inp, [labels.index(g) for g in greedy], gw, full)
        t0=time.perf_counter(); best, bscore, n = None, np.inf, 0
        for combo in itertools.combinations(range(J), size):
            c = list(combo)
            w = _fit_weights(inp, c, full, Pt=None, Pd=None, v=None)
            s = _outcome_rmspe(inp, c, w, full); n += 1
            if s < bscore: best, bscore = c, s
        el = time.perf_counter()-t0
        bl = sorted(labels[j] for j in best)
        same = bl == greedy
        print(f"  k={size}: greedy {gscore:.8f}  exhaustive {bscore:.8f}  "
              f"gap {gscore-bscore:+.3e}  {'SAME SET' if same else 'DIFFERENT'}  "
              f"({n} subsets, {el:.1f}s)")
        if not same:
            print(f"      greedy      {greedy}")
            print(f"      exhaustive  {bl}")
