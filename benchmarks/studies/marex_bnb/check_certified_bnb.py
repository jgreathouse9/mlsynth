"""Run the certified branch and bound at 12 markets against brute force.

Every panel of ``bnb12`` plus ``factor_twin``, the factor panel with one market
duplicated so that two treated sets tie exactly. Each run is checked against
the enumeration: the reported ``lower`` must not exceed the certified minimum,
the incumbent's certified ``hi`` must not be below its set's certified lower
value, and every bounded node is audited against its subtree's certified
minimum (the ``audit`` column counts violations and must be zero).

From the repository root::

    python -m benchmarks.studies.marex_bnb.check_certified_bnb
"""
import os, itertools, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
from fractions import Fraction
from benchmarks.studies.marex_bnb.bnb12 import panels
from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.certified_bnb import search
from mlsynth.utils.marex_helpers.exact import side_fit

m = 3

def brute(B, A):
    n = B.shape[1]; pan = Panel(B - A[:, None]); out = {}
    for S in itertools.combinations(range(n), m):
        C = [j for j in range(n) if j not in S]
        g, w = side_fit(B, A, list(S)); h, v = side_fit(B, A, C)
        out[S] = (g + h, pan.side_upper(list(S), w) + pan.side_upper(C, v),
                  pan.side_lower(list(S), w) + pan.side_lower(C, v))
    return out

cases = dict(panels())
B, A = cases["factor"]
Bt = B.copy(); Bt[:, 11] = Bt[:, 4]                 # two identical markets: exact ties
cases["factor_twin"] = (Bt, Bt.mean(axis=1))
print(f"{'panel':12s} {'lift':9s} {'nodes':>5s} {'pruned':>6s} {'dnn':>4s} {'sets':>4s} "
      f"{'s':>6s}  {'set':12s} {'brute':12s} {'gap/f*':>9s}  unresolved  audit")
for name, (B, A) in cases.items():
    br = brute(B, A)
    fstar = min(v[0] for v in br.values())
    best = sorted(S for S, v in br.items() if v[0] <= fstar * (1 + 1e-12))
    for use in (None, "trace", "indicator"):
        bad = [0]
        def audit(I, O, lower):
            sub = [v for S, v in br.items() if set(I) <= set(S) and not set(O) & set(S)]
            if lower > min(v[1] for v in sub):
                bad[0] += 1
        t0 = time.perf_counter()
        res = search(B, A, m, lift=use, audit=audit)
        dt = time.perf_counter() - t0
        st = res["stats"]
        # what the certificate claims, checked against brute force
        assert res["lower"] <= min(v[1] for v in br.values())
        assert br[res["set"]][2] <= res["hi"]
        print(f"{name:12s} {str(use):9s} {st['nodes']:5d} {st['pruned']:6d} {st['dnn']:4d} {st['sets']:4d} "
              f"{dt:6.1f}  {str(res['set']):12s} {str(best):12s} {float(res['gap'])/fstar:9.1e}  "
              f"{res['unresolved']}  {bad[0]}", flush=True)
