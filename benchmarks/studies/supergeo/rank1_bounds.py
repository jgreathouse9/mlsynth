"""Root bound over the exact optimum: subset-row cuts only against every rank-1 family.

    python -m benchmarks.studies.supergeo.rank1_bounds"""
import time
from . import rank1, scale

OPT = {(12, 0): 2.241053888, (16, 0): 0.0215906, (16, 1): 0.286779, (16, 2): 0.268188,
       (20, 0): 0.075935, (20, 1): 0.0867097, (20, 2): 0.0827753, (40, 0): 0.0088722102117969}
SR3 = [(3, (1, 1, 1), 2)]

if __name__ == "__main__":
    for (k, r), opt in OPT.items():
        z = list(scale.subsets(k, r + 1))[r]
        line = f"K={k} subset {r}:"
        for label, fams in (("SR3 only", SR3), ("all rank-1 families", rank1.FAMILIES)):
            t = time.perf_counter(); out = rank1.root_bound(z, families=fams, rounds=40, per_round=50, max_cuts=3000)
            h = out["hist"]
            line += f"  {label}: {h[0][1] / opt:.3f} -> {h[-1][1] / opt:.3f} ({h[-1][0]} cuts, {time.perf_counter() - t:.0f}s);"
        print(line, flush=True)
