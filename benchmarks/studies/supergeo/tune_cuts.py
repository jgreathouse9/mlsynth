"""Tune the rank-1 cut management (depth limit, purge age) on 20-market subsets.

    python -m benchmarks.studies.supergeo.tune_cuts tune      # grid on subsets 0-2
    python -m benchmarks.studies.supergeo.tune_cuts validate D1 P1 D2 P2 ...   # held-out subsets 3-5

Depth -1 means every depth; purge 0 means never. Every optimum is checked against
the exact certificate. One JSON line per run."""
import json, sys
from . import bcp, rank1, scale, certify_exact

K = 20


def run(z, ref, depth, purge, sr3=False):
    kw = {} if sr3 else {"families": rank1.FAMILIES, "max_active": 250,
                         "deep_depth": None if depth < 0 else depth, "purge_after": None if purge == 0 else purge}
    out = bcp.BCP(z, verbose=False, **kw).solve(time_limit=900, heuristic_seconds=2.0)
    return dict(depth=None if sr3 else depth, purge=None if sr3 else purge, arm="SR3" if sr3 else "rank-1",
                status=out["status"], nodes=out["nodes"], t=round(out["t"], 2), max_pool=out.get("max_pool", 0),
                ok=abs(out["opt"] - ref) <= 1e-9 * max(1.0, abs(ref)))


def main(mode, args):
    subs = list(scale.subsets(K, 6))
    idx = [0, 1, 2] if mode == "tune" else [3, 4, 5]
    refs = {i: certify_exact.certify(subs[i], verbose=False)["opt"] for i in idx}
    if mode == "tune":
        grid = [(d, p) for d in (0, 1, 2, 3, -1) for p in (2, 5, 10, 0)]
        arms = [("SR3", None)] + [("rank-1", g) for g in grid]
    else:
        pairs = [(int(args[i]), int(args[i + 1])) for i in range(0, len(args), 2)]
        arms = [("SR3", None)] + [("rank-1", g) for g in pairs]
    for i in idx:
        for arm, g in arms:
            rec = run(subs[i], refs[i], *(g or (0, 0)), sr3=(arm == "SR3"))
            rec["subset"] = i
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
