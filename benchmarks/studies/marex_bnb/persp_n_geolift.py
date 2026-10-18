"""persp+N on GeoLift's twelve nodes, against the treated and the full lift.

The nodes, the subtree minima and the "need" column are those of
``treated_lift_geolift`` (four treated markets): a node is pruned when its
bound reaches the optimum ``f*``, which needs ``bound / subtree minimum`` at
least ``need``. The treated and full lifts' certified values and times are
copied from ``results/treated_lift_geolift.log`` and
``results/certified_indicator_geolift.log``. persp+N's are the solver's, and
the last column is the relative distance from the solver's value down to the
certified bound (``persp_n.certified``). Then the roots for six and eight
treated markets.

From the repository root::

    python -m benchmarks.studies.marex_bnb.persp_n_geolift
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

from benchmarks.studies.marex_bnb.bnb12 import fit
from benchmarks.studies.marex_bnb.certify import Panel, down
from benchmarks.studies.marex_bnb.panels import geolift, leaves
from benchmarks.studies.marex_bnb.persp_n import bound, certified
from benchmarks.studies.marex_bnb.treated_lift_geolift import FULL, FULL_TIME
from mlsynth.utils.marex_helpers.exact import side_fit

TREATED = {"root": (0.9811, 35.1), "lex d1 q=0": (0.9739, 32.0),
           "lex d1 q=5": (0.9786, 17.6), "lex d1 q=10": (0.9924, 9.6),
           "lex d1 q=15": (0.9799, 4.2), "lex d1 q=20": (0.9866, 1.5),
           "lex d1 q=25": (0.9810, 0.6), "lex d1 q=30": (0.9953, 0.1),
           "lex d1 q=35": (0.9999, 0.0), "out: 6 non-optimal": (0.9741, 17.2),
           "out: 1 optimal": (0.8774, 26.3),
           "in 1 optimal, out 1 optimal": (0.9856, 25.8)}


def nodes(B, A, sets, f):
    n = B.shape[1]
    single = np.array([fit(B, A, [j]) for j in range(n)])
    order = np.argsort(single, kind="stable")
    out = [("root", [], [], list(range(n)))]
    for q in range(0, 36, 5):
        out.append((f"lex d1 q={q}", [int(order[q])], [int(order[p]) for p in range(q)],
                    [int(order[p]) for p in range(q + 1, n)]))
    S_star = sets[np.argmin(f)].tolist()
    rest = sorted(set(range(n)) - set(S_star))
    out.append(("out: 6 non-optimal", [], rest[:6], [j for j in range(n) if j not in rest[:6]]))
    out.append(("out: 1 optimal", [], [S_star[0]], [j for j in range(n) if j != S_star[0]]))
    out.append(("in 1 optimal, out 1 optimal", [S_star[0]], [S_star[1]],
                [j for j in range(n) if j not in S_star[:2]]))
    return out


def main(sides):
    B, A = geolift()
    n = B.shape[1]
    Bc = B - A[:, None]
    pan = Panel(Bc)
    sets, g, h = leaves("geolift_m4")
    f = g + h
    fstar = float(f.min())
    member = np.zeros((len(sets), n), bool)
    member[np.arange(len(sets))[:, None], sets] = True
    print(f"GeoLift, {n} markets, 4 treated, f* = {fstar:.10g}; persp+N side(s): {sides}")
    print(f"{'node':30s} {'sets':>6s} {'need':>6s} {'persp+N':>8s} {'s':>6s} "
          f"{'treated':>8s} {'s':>5s} {'full':>7s} {'s':>6s} {'cert gap':>9s}")
    for lab, I, O, F in nodes(B, A, sets, f):
        mask = member[:, I].all(1) if I else np.ones(len(sets), bool)
        if O:
            mask &= ~member[:, O].any(1)
        sub = float(f[mask].min())
        val, status, secs = bound(Bc, 4, I, O, F, side=sides)
        cert = certified(pan, 4, I, O, F, side=sides)[0]
        if sides == "treated":
            cols = sorted(O + F)
            ctl = down(pan.side_lower(cols, side_fit(B, A, cols)[1]))
            val, cert = val + ctl, cert + ctl
        tr, tr_s = TREATED[lab]
        print(f"{lab:30s} {int(mask.sum()):6d} {fstar / sub:6.3f} {val / sub:8.4f} "
              f"{secs:6.1f} {tr:8.4f} {tr_s:5.1f} {FULL[lab]:7.4f} {FULL_TIME[lab]:6.1f} "
              f"{(val - cert) / val:9.1e}  {'PRUNE' if val >= fstar else ''} {status}",
              flush=True)
    for m, treated_lb, full_lb, best in ((6, 259334, 272312, 297926.2374),
                                         (8, 141842, 165254, 181943.3763)):
        val, status, secs = bound(Bc, m, [], [], list(range(n)), side=sides)
        cert = certified(pan, m, [], [], list(range(n)), side=sides)[0]
        if sides == "treated":
            cols = list(range(n))
            ctl = down(pan.side_lower(cols, side_fit(B, A, cols)[1]))
            val, cert = val + ctl, cert + ctl
        print(f"root m={m}: persp+N {val:.6g} (gap {(best - val) / best:.4f}, {secs:.1f} s, "
              f"cert gap {(val - cert) / val:.1e})  treated {treated_lb} "
              f"(gap {(best - treated_lb) / best:.4f})  full {full_lb} "
              f"(gap {(best - full_lb) / best:.4f})  {status}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "both")
