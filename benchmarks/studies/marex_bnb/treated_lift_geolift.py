"""The treated-side lift on GeoLift, against the full indicator lift.

``treated_lift`` drops the control weights from the indicator lift, so its PSD
block has ``|I | F| + |F| - 1`` rows in place of about ``3n``. The node bound
is its certified value plus the Wolfe-dual certificate of ``h(O | F)``, the
control fit over every market a completion could use as a control. The control
fit is under 1% of ``f`` at GeoLift's optimum, so the loss should be small.

The first part repeats ``certified_dnn_geolift``'s twelve nodes for four
treated markets, whose full-lift values (certified) are listed beside. The
second compares the roots for six and eight treated markets with the full
lift's (272,312 and 165,254, ``root_gap_by_m``).

From the repository root::

    python -m benchmarks.studies.marex_bnb.treated_lift_geolift
"""
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

from benchmarks.studies.marex_bnb.bnb12 import fit
from benchmarks.studies.marex_bnb.certify import Panel, down
from benchmarks.studies.marex_bnb.certified_dnn import lift_bound, treated_lift
from benchmarks.studies.marex_bnb.panels import geolift, leaves
from mlsynth.utils.marex_helpers.exact import side_fit

FULL = {"root": 0.9915, "lex d1 q=0": 0.9877, "lex d1 q=5": 0.9999,
        "lex d1 q=10": 0.9999, "lex d1 q=15": 0.9967, "lex d1 q=20": 0.9980,
        "lex d1 q=25": 0.9985, "lex d1 q=30": 1.0000, "lex d1 q=35": 1.0000,
        "out: 6 non-optimal": 0.9956, "out: 1 optimal": 0.9027,
        "in 1 optimal, out 1 optimal": 0.9972}
FULL_TIME = {"root": 252.2, "lex d1 q=0": 205.7, "lex d1 q=5": 145.0,
             "lex d1 q=10": 69.5, "lex d1 q=15": 40.6, "lex d1 q=20": 24.9,
             "lex d1 q=25": 15.4, "lex d1 q=30": 6.1, "lex d1 q=35": 2.2,
             "out: 6 non-optimal": 139.8, "out: 1 optimal": 170.3,
             "in 1 optimal, out 1 optimal": 169.0}


def node_bound(pan, B, A, m, I, O, F):
    t0 = time.perf_counter()
    lb, approx, status = lift_bound(pan, treated_lift(B.shape[1], m, I, O, F))
    cols = sorted(O + F)
    h_lo = pan.side_lower(cols, side_fit(B, A, cols)[1])
    return lb + down(h_lo), status, time.perf_counter() - t0


def main():
    B, A = geolift()
    n = B.shape[1]
    pan = Panel(B - A[:, None])
    sets, g, h = leaves("geolift_m4")
    f = g + h
    fstar = float(f.min())
    member = np.zeros((len(sets), n), bool)
    member[np.arange(len(sets))[:, None], sets] = True
    single = np.array([fit(B, A, [j]) for j in range(n)])
    order = np.argsort(single, kind="stable")
    nodes = [("root", [], [], list(range(n)))]
    for q in range(0, 36, 5):
        nodes.append((f"lex d1 q={q}", [int(order[q])],
                      [int(order[p]) for p in range(q)],
                      [int(order[p]) for p in range(q + 1, n)]))
    S_star = sets[np.argmin(f)].tolist()
    out = sorted(set(range(n)) - set(S_star))
    nodes.append(("out: 6 non-optimal", [], out[:6], [j for j in range(n) if j not in out[:6]]))
    nodes.append(("out: 1 optimal", [], [S_star[0]], [j for j in range(n) if j != S_star[0]]))
    nodes.append(("in 1 optimal, out 1 optimal", [S_star[0]], [S_star[1]],
                  [j for j in range(n) if j not in S_star[:2]]))
    print(f"GeoLift, {n} markets, 4 treated, f* = {fstar:.10g}")
    print(f"{'node':30s} {'sets':>6s} {'need':>6s} {'treated':>8s} {'full':>7s} "
          f"{'seconds':>8s} {'full s':>7s}")
    for lab, I, O, F in nodes:
        mask = member[:, I].all(1) if I else np.ones(len(sets), bool)
        if O:
            mask &= ~member[:, O].any(1)
        sub = float(f[mask].min())
        lb, status, secs = node_bound(pan, B, A, 4, I, O, F)
        print(f"{lab:30s} {int(mask.sum()):6d} {fstar / sub:6.3f} {lb / sub:8.4f} "
              f"{FULL[lab]:7.4f} {secs:8.1f} {FULL_TIME[lab]:7.1f}  "
              f"{'PRUNE' if lb >= fstar else ''} {status}", flush=True)
    for m, full_lb, best in ((6, 272312, 297926), (8, 165254, 191758)):
        lb, status, secs = node_bound(pan, B, A, m, [], [], list(range(n)))
        print(f"root m={m}: treated {lb:.6g} ({(best - lb) / best:.4f} gap)  full {full_lb} "
              f"({(best - full_lb) / best:.4f} gap)  {secs:.1f} s  {status}", flush=True)


if __name__ == "__main__":
    main()
