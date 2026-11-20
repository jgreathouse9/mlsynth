"""Starting heuristics against known optima.

    python -m benchmarks.studies.supergeo.heuristics

Optima: the exact certificate (:mod:`.certify_exact`) for GeoLift subsets and all
40 markets; the best known design (0.003882, steered repair) at N = 210, which
is not proven optimal."""
import time
import numpy as np
from . import sg, dive, lns, incumbent, certify_exact, scale


def root_pool(z, lo=2, hi=4, add=2000):
    """Columns of the root LP after column generation (rescaled data, as the search sees it)."""
    import math
    n = len(z); z = np.asarray(z, float).copy()
    cols = sg.initial_pool(z, lo, hi); score = {c: dive._sc(z, c) for c in cols}
    while True:
        lp, pi, x = sg.master_lp(cols, np.array([score[c] for c in cols]), n)
        if 0 < lp < 1e-2 * n:
            k = int(round(math.log(n / lp, 4))); z = z * 2.0 ** k; score = {c: dive._sc(z, c) for c in score}; continue
        rows, s_, rc, m = sg._scan(z, pi, lo, hi, -1e-9 * max(1.0, lp), 5_000_000)
        new = 0
        for t in np.argsort(rc)[:add]:
            c = tuple(int(v) for v in rows[t] if v >= 0)
            if c not in score: score[c] = float(s_[t]); cols.append(c); new += 1
        if new == 0:
            return cols


def current(z):
    o = list(np.argsort(z)); n = len(z)
    start = ([tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])]) if n % 2 else [tuple(o[i:i + 2]) for i in range(0, n, 2)]
    d, _, _ = lns.lns_dp(np.asarray(z, float), start, seconds=2.0, verbose=False)
    return d


def run(name, z, opt):
    t = time.perf_counter(); pool = root_pool(z); t_pool = time.perf_counter() - t
    res = {}
    for label, f in (("current: pairs + repair 2 s", lambda: current(z)),
                     ("greedy", lambda: incumbent.greedy(z)),
                     ("grasp 2 s", lambda: incumbent.grasp(z, seconds=2.0)),
                     ("grasp 2 s + LP-guided MIP 1 s", lambda: incumbent.lp_round(z, pool, incumbent.grasp(z, seconds=2.0), seconds=1.0))):
        t = time.perf_counter(); d = f(); dt = time.perf_counter() - t
        res[label] = (incumbent.cost(z, d) / opt, dt)
    print(f"{name} (optimum {opt:.6g}, root pool {len(pool)} columns in {t_pool:.1f}s): " +
          "; ".join(f"{k} {v[0]:.4g}x in {v[1]:.1f}s" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    for k, reps in ((12, 5), (16, 3), (20, 3)):
        for r, z in enumerate(scale.subsets(k, reps)):
            run(f"K={k} subset {r}", z, certify_exact.certify(z, verbose=False)["opt"])
    z = next(scale.subsets(40, 1)); run("K=40 all GeoLift", z, 0.0088722102117969)
    rng = np.random.default_rng(0); pre = np.exp(rng.normal(0, 1.3, 210)) * 1000
    run("N=210 simulated (best known, not proven)", pre / np.median(np.diff(np.sort(pre))), 0.003882)
