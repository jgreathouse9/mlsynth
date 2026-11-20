"""Why branch, cut and price is slow at 50 markets: the measurements of the root-cause analysis.

    python -m benchmarks.studies.supergeo.rca50 profile INSTANCE ARM NODES
    python -m benchmarks.studies.supergeo.rca50 incumbent INSTANCE ARM START [TIME_LIMIT]
    python -m benchmarks.studies.supergeo.rca50 heuristics INSTANCE

INSTANCE and ARM as in :mod:`.fifty`. ``profile`` splits the time of the first NODES
nodes into pricing, master LPs, separation, branching and the rest (cProfile,
cumulative time of each component's entry point). ``incumbent`` runs the search
from START: ``lns`` (pairs plus repair for 10 s, what :mod:`.fifty` used) or ``opt``
(the certified optimum in ``fifty_optima.json``). ``heuristics`` prices the
starting designs available against the certified optimum. One JSON line each."""
import cProfile, json, pathlib, pstats, sys, time
import numpy as np
from . import bcp, bnp, dive, fifty, incumbent as inc_mod, lns, rank1, sr

OPTIMA = pathlib.Path(__file__).with_name("fifty_optima.json")
PARTS = {"pricing": [(bcp, "_scan_bcp"), (bcp, "_scan_bcp_r1")],
         "master LP": [(bcp, "_master"), (bcp, "_master_r1")],
         "separation": [(rank1, "separate"), (sr, "_separate")],
         "branching": [(bcp.BCP, "_branch")], "purge": [(bcp.BCP, "_purge")],
         "column filter": [(bnp, "compat_py")]}


def _instrument():
    """Wrap each component's entry point with a timer (numba kernels are invisible to cProfile)."""
    acc = {k: 0.0 for k in PARTS}; calls = {k: 0 for k in PARTS}
    for label, targets in PARTS.items():
        for owner, name in targets:
            f = getattr(owner, name)
            def wrap(*a, _f=f, _l=label, **k):
                t = time.perf_counter()
                try:
                    return _f(*a, **k)
                finally:
                    acc[_l] += time.perf_counter() - t; calls[_l] += 1
            setattr(owner, name, wrap)
    return acc, calls


def kwargs(arm):
    return bcp.managed() if arm == "managed" else {}


def lns_start(z, seconds=10.0):
    o = list(np.argsort(z)); n = len(z)
    start = [tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])] if n % 2 else [tuple(o[i:i + 2]) for i in range(0, n, 2)]
    design, _, _ = lns.lns_dp(z, start, seconds=seconds, verbose=False)
    return [tuple(sorted(int(v) for v in g)) for g in design]


def cost(z, design):
    return sum(dive._sc(z, tuple(g)) for g in design)


def profile(name, arm, nodes):
    z = fifty.instance(name); start = lns_start(z)
    acc, calls = _instrument()
    b = bcp.BCP(z, verbose=False, **kwargs(arm))
    pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable()
    out = b.solve(incumbent=start, time_limit=3600, node_limit=nodes)
    pr.disable(); total = time.perf_counter() - t0
    pstats.Stats(pr).sort_stats("tottime").dump_stats(str(pathlib.Path.cwd() / f"rca50_{name}_{arm}.prof"))
    rec = dict(mode="profile", instance=name, arm=arm, nodes=out["nodes"], lps=out["lps"], total=round(total, 2),
               per_node=round(total / max(out["nodes"], 1), 3), cuts=out["cuts"], calls=calls,
               **{k: round(v, 2) for k, v in acc.items()})
    # the column filter runs inside the master-LP loop, not inside a timed part
    rec["rest"] = round(total - sum(acc[k] for k in ("pricing", "master LP", "separation", "branching", "column filter")), 2)
    print(json.dumps(rec), flush=True)


def run_from(name, arm, start, limit):
    z = fifty.instance(name)
    design = lns_start(z) if start == "lns" else [tuple(g) for g in json.loads(OPTIMA.read_text())[name]["design"]]
    out = bcp.BCP(z, verbose=False, **kwargs(arm)).solve(incumbent=design, time_limit=limit)
    print(json.dumps(dict(mode="incumbent", instance=name, arm=arm, start=start, start_cost=cost(z, design) / 1.0,
                          status=out["status"], nodes=out["nodes"], t=round(out["t"], 1), bound=out["bound"],
                          opt=out["opt"], root_bound=out["root_bound"])), flush=True)


def heuristics(name):
    z = fifty.instance(name); opt = json.loads(OPTIMA.read_text())[name]["opt"]
    # the certified optimum is in original units; fifty.instance is in those units too
    rows = {}
    t = time.perf_counter(); rows["pairs + repair 10 s"] = (cost(z, lns_start(z, 10.0)), time.perf_counter() - t)
    t = time.perf_counter(); g = inc_mod.grasp(z, seconds=10.0, seed=0); rows["GRASP 10 s"] = (inc_mod.cost(z, g), time.perf_counter() - t)
    for k, (c, s) in rows.items():
        print(json.dumps(dict(mode="heuristics", instance=name, heuristic=k, cost=c, over_opt=c / opt, seconds=round(s, 1))), flush=True)


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "profile":
        profile(args[0], args[1], int(args[2]))
    elif mode == "incumbent":
        run_from(args[0], args[1], args[2], float(args[3]) if len(args) > 3 else 1800.0)
    elif mode == "heuristics":
        heuristics(args[0])
    else:
        raise SystemExit(__doc__)
