"""ADMM against the interior-point solve, on the indicator lift at 12 markets.

For each panel and node of ``compare_lifts``, the indicator lift without cuts
is solved by Clarabel and certified (``lift_bound``), then by ``FaceADMM`` from
a cold start and certified exactly at its last iterate. The lines report the
two certified bounds over the subtree minimum, the ADMM iterations and the
time of each. Any certified bound above the subtree's certified minimum is
counted in ``above`` and must be zero.

From the repository root::

    python -m benchmarks.studies.marex_bnb.check_admm
"""
import itertools
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
from fractions import Fraction

from benchmarks.studies.marex_bnb.admm import FaceADMM
from benchmarks.studies.marex_bnb.bnb12 import panels
from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.certified_dnn import indicator_lift, lift_bound
from benchmarks.studies.marex_bnb.compare_lifts import nodes_of
from mlsynth.utils.marex_helpers.exact import side_fit

M = 3


def main(n_nodes=25, **kw):
    rng = np.random.default_rng(1)
    for name, (B, A) in panels().items():
        n = B.shape[1]
        pan = Panel(B - A[:, None])
        val, ub = {}, {}
        for S in itertools.combinations(range(n), M):
            C = [j for j in range(n) if j not in S]
            g, w = side_fit(B, A, list(S))
            h, v = side_fit(B, A, C)
            val[S] = g + h
            ub[S] = pan.side_upper(list(S), w) + pan.side_upper(C, v)
        nodes = nodes_of(n, rng)
        nodes = [nodes[0]] + [nodes[i] for i in
                              rng.choice(np.arange(1, len(nodes)), n_nodes - 1, replace=False)]
        R, above, t_ipm, t_admm, its = [], 0, 0.0, 0.0, []
        for I, O, F in nodes:
            sub = [S for S in val if set(I) <= set(S) and not set(O) & set(S)]
            smin = min(val[S] for S in sub)
            lift = indicator_lift(n, M, I, O, F, cuts=False)
            t0 = time.perf_counter()
            ipm = lift_bound(pan, lift)[0]
            t_ipm += time.perf_counter() - t0
            t0 = time.perf_counter()
            solver = FaceADMM(pan, lift)
            state, _, k = solver.solve(**kw)
            lb = solver.bound(state["U"], state["rho"])
            t_admm += time.perf_counter() - t0
            its.append(k)
            cert_min = min(ub[S] for S in sub)
            above += (Fraction(lb) > cert_min) + (Fraction(ipm) > cert_min)
            R.append((ipm / smin, lb / smin))
        R = np.array(R)
        print(f"{name:10s} root ipm {R[0, 0]:.4f} admm {R[0, 1]:.4f} | median ipm "
              f"{np.median(R[:, 0]):.4f} admm {np.median(R[:, 1]):.4f} | worst admm-ipm "
              f"{np.min(R[:, 1] - R[:, 0]):+.4f} | above {above} | iters median "
              f"{int(np.median(its))} max {max(its)} | s/node ipm "
              f"{t_ipm / len(nodes):.3f} admm {t_admm / len(nodes):.3f}", flush=True)


if __name__ == "__main__":
    main()
