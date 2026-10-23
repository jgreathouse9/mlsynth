"""Issue #770's leads, per candidate: how tight, how costly, how many fits they leave.

Every bound here sits between the treated fit ``g(S)``, which the search already
computes, and the coupled fit, so only candidates with ``g(S)`` under the cutoff
reach it. The cutoff is the optimum ``search.py`` recorded plus the search's
margin, which is the count the issue asks for: the candidates a bound leaves to
fit once the incumbent is optimal.

* lead 1, ``value - gap``: the coupled objective at the treated-only start less
  its Frank-Wolfe gap. Costs one control fit.
* lead 1, dual: :func:`bound.fenchel_bound` at the same start, with
  ``lam = 2 (A - t)`` and ``mu = 2 beta (t - c)``. Same cost.
* lead 3, Wolfe: ``g + beta * dist_lb^2`` with :func:`bound.hull_distance_bound`,
  stopped once it clears the cutoff.
* lead 3, exact: the same with the exact distance from Clarabel, the ceiling of
  any distance bound.

At ``beta = 100`` the sample is 1,000 of the candidates the closed-form bound
leaves; at ``beta = 1`` it is all of them. Shares are of the candidate's exact
objective. The Wolfe bound stops once it clears, so its share is a floor.

From the repository root, after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/leads.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.studies.marex_coupling_bound import bound, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

SAMPLE = 1000
SEED = 3


def exact_distance2(B, S, C):
    w = cp.Variable(len(S), nonneg=True)
    v = cp.Variable(len(C), nonneg=True)
    d = B[:, S] @ w - B[:, C] @ v
    cp.Problem(cp.Minimize(cp.sum_squares(d)),
               [cp.sum(w) == 1, cp.sum(v) == 1]).solve(solver=cp.CLARABEL)
    r = B[:, S] @ w.value - B[:, C] @ v.value
    return max(float(r @ r), 0.0)


def measure(B, A, beta, pair_d2):
    n = B.shape[1]
    fstar = panel.optimum(beta)
    cut = fstar + exact.PRUNE_MARGIN * (1.0 + abs(fstar))
    sets = np.array(list(itertools.combinations(range(n), panel.M)))
    lb = exact.treated_fit_bound(B, A, sets)
    survivors = np.flatnonzero(lb < cut)
    rng = np.random.default_rng(SEED)
    idx = (survivors if len(survivors) <= SAMPLE
           else rng.choice(survivors, SAMPLE, replace=False))
    rows = []
    for k in idx:
        S = [int(s) for s in sets[k]]
        C = [j for j in range(n) if j not in S]
        t0 = time.perf_counter()
        g, wg = exact.side_fit(B, A, S)
        t_g = time.perf_counter() - t0
        if g >= cut:
            continue
        wg = np.asarray(wg, float)
        Bs, Bc = B[:, S], B[:, C]
        t0 = time.perf_counter()
        value, gap, _, c = exact._coupled_point(Bs, Bc, A, beta, wg, None)
        t = Bs @ wg
        lead1 = value - gap
        lead1_dual = bound.fenchel_bound(B, A, S, C, 2 * (A - t),
                                         2 * beta * (t - c), beta)
        t_lead1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        d, cycles, stop = bound.hull_distance_bound(B, S, C, (cut - g) / beta,
                                                    pair_d2=pair_d2)
        t_wolfe = time.perf_counter() - t0
        t0 = time.perf_counter()
        d2 = exact_distance2(B, S, C)
        t_exact = time.perf_counter() - t0
        t0 = time.perf_counter()
        exact.coupled_fit(B, A, S, C, beta, w0=wg, cutoff=cut)
        t_fit = time.perf_counter() - t0
        f = exact.coupled_fit(B, A, S, C, beta, w0=wg)[0]
        rows.append(dict(f=f, lead1=lead1, lead1_dual=lead1_dual,
                         wolfe=g + beta * max(d, 0.0) ** 2,
                         exact_distance=g + beta * d2, cycles=cycles, stop=stop,
                         t_g=t_g, t_lead1=t_lead1, t_wolfe=t_wolfe,
                         t_exact=t_exact, t_fit=t_fit))
    col = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    scale = len(survivors) / len(idx)
    q = lambda x: [round(float(np.quantile(x, a)), 3) for a in (0.1, 0.5, 0.9)]
    ms = lambda x: round(float(np.mean(x)) * 1e3, 3)
    out = {"optimum": fstar, "closed_form_survivors": int(len(survivors)),
           "sampled": int(len(idx)), "reach_coupled_fit": len(rows),
           "projected_reach": int(round(len(rows) * scale)), "leads": {}}
    for name in ("lead1", "lead1_dual", "wolfe", "exact_distance"):
        left = int((col[name] < cut).sum())
        out["leads"][name] = {
            "share_of_f_q10_q50_q90": q(col[name] / col["f"]),
            "never_above_f": bool(np.all(col[name] <= col["f"] * (1 + 1e-9))),
            "left_to_fit_sampled": left,
            "left_to_fit_projected": int(round(left * scale))}
    out["ms_per_candidate"] = {
        "treated_fit": ms(col["t_g"]), "lead1_both": ms(col["t_lead1"]),
        "wolfe": ms(col["t_wolfe"]), "exact_distance_clarabel": ms(col["t_exact"]),
        "cutoff_coupled_fit": ms(col["t_fit"])}
    out["wolfe_cycles_q50_q90_max"] = [int(np.quantile(col["cycles"], 0.5)),
                                       int(np.quantile(col["cycles"], 0.9)),
                                       int(col["cycles"].max())]
    out["wolfe_stops"] = {s: int((col["stop"] == s).sum())
                          for s in sorted(set(col["stop"]))}
    return out


def main():
    B, A = panel.geolift()
    pair_d2 = bound.squared_pair_distances(B)
    out = {str(beta): measure(B, A, beta, pair_d2) for beta in panel.BETAS}
    (panel.RESULTS / "leads.json").write_text(json.dumps(out, indent=1) + "\n")
    for beta, r in out.items():
        print(f"beta={beta}: {r['reach_coupled_fit']} of {r['sampled']} sampled "
              f"reach the coupled fit")
        for name, s in r["leads"].items():
            print(f"  {name:15s} median share {s['share_of_f_q10_q50_q90'][1]:7.3f}  "
                  f"left to fit {s['left_to_fit_projected']:6d}  "
                  f"never above f {s['never_above_f']}")
        print(f"  ms per candidate {r['ms_per_candidate']}")


if __name__ == "__main__":
    main()
