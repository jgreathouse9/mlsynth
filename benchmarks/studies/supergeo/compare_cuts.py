"""Branch, cut and price: subset-row cuts, every rank-1 family, and every rank-1 family with cut management.

    python -m benchmarks.studies.supergeo.compare_cuts K [REPS] [TIME_LIMIT]"""
import sys
from . import bcp, rank1, scale, certify_exact


def main(k, reps=3, limit=600.0):
    for r, z in enumerate(scale.subsets(k, reps)):
        ref = certify_exact.certify(z, verbose=False)["opt"]
        for label, kw in (("SR3", {}), ("rank-1", {"families": rank1.FAMILIES}),
                          ("rank-1 managed", {"families": rank1.FAMILIES, "purge_after": 5, "deep_depth": 1, "max_active": 250})):
            out = bcp.BCP(z, verbose=False, **kw).solve(time_limit=limit, heuristic_seconds=2.0)
            ok = abs(out["opt"] - ref) <= 1e-9 * max(1.0, abs(ref))
            print(f"K={k} subset {r} {label}: {out['status']}, nodes {out['nodes']}, {out['t']:.1f}s, root bound/opt "
                  f"{out['root_bound'] / ref:.3f}, group/pair {out['group_branches']}/{out['pair_branches']}, cuts {out['cuts']}, "
                  f"purged {out.get('purged', 0)}, max pool {out.get('max_pool', 0)}, optimum {'matches' if ok else 'DIFFERS from'} the exact certificate", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3, float(sys.argv[3]) if len(sys.argv) > 3 else 600.0)
