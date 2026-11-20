"""Branch and price against branch, cut and price on GeoLift subsets of K markets.

    python -m benchmarks.studies.supergeo.scale K [REPS] [TIME_LIMIT]

Every optimum is checked: by brute force for K <= 12, by the exact certificate of
:mod:`.certify_exact` above that."""
import pathlib, sys, time
import numpy as np, pandas as pd
from . import bnp, bcp, certify_exact, test_bcp

GEOLIFT = pathlib.Path(__file__).resolve().parents[3] / "basedata" / "geolift_test_data.csv"


def subsets(k, reps):
    df = pd.read_csv(GEOLIFT); dates = sorted(df.date.unique())
    pre = df[df.date.isin(dates[-56:-28])].groupby("location").Y.sum().to_numpy(float)
    rng = np.random.default_rng(2026)
    for _ in range(reps):
        p = pre[sorted(rng.choice(len(pre), k, replace=False))] if k < len(pre) else pre
        yield p / np.median(np.diff(np.sort(p)))


def main(k, reps=5, limit=600.0):
    for r, z in enumerate(subsets(k, reps)):
        if k <= 12:
            ref, how = test_bcp.brute(z), "brute force"
        else:
            ref, how = certify_exact.certify(z, verbose=False)["opt"], "exact certificate"
        for name, solver in (("B&P", lambda: bnp.BnP(z, verbose=False)), ("BCP", lambda: bcp.BCP(z, verbose=False))):
            out = solver().solve(time_limit=limit)
            ok = abs(out["opt"] - ref) <= 1e-9 * max(1.0, abs(ref))
            extra = f", group/pair branches {out['group_branches']}/{out['pair_branches']}, cuts {out['cuts']}" if name == "BCP" else ""
            print(f"K={k} subset {r} {name}: {out['status']}, nodes {out['nodes']}, {out['t']:.1f}s, root bound/opt "
                  f"{out['root_bound'] / ref if ref > 0 else float('nan'):.3f}{extra}, optimum {'matches' if ok else 'DIFFERS from'} {how}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5, float(sys.argv[3]) if len(sys.argv) > 3 else 600.0)
