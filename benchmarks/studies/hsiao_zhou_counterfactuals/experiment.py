"""Run one Hsiao & Zhou (2019) cell and print it beside the paper's row.

    python experiment.py --dgp dgp6 --n-co 30 --T 40 --reps 100 --r-known

``--n-co`` is the number of controls, which is what the paper's tables label
``N``, and ``--T`` is the total length, so ``40`` means ``T0 = 30`` and
``60`` means ``T0 = 50``. Ten post-treatment periods throughout, as Section 6
sets them.

Neither DGP6 nor DGP7 carries a treatment effect, so the treated unit's
observed series is its own counterfactual and every criterion is a prediction
error. That is what makes the paper's tables comparable across methods.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .methods import (average_counterfactuals, cce_counterfactual, criteria,
                      pca_counterfactual, pda_counterfactual, select_r_by_cv,
                      simulate)

#: Tables 6 and 7, the MAB and "MSE" rows, keyed (dgp, n_co, T). The row
#: labelled MSE is a root mean squared error; see the README.
PAPER = {
    ("dgp6", 30, 40): {"PCA": (1.717, 2.271), "CCE": (3.728, 9.187),
                       "CPDA": (1.193, 1.571), "PDA": (1.181, 1.552),
                       "PDAX": (1.234, 1.624), "MA": (1.368, 2.290),
                       "MB": (1.461, 2.530)},
    ("dgp6", 30, 60): {"PCA": (1.712, 2.351), "CCE": (0.693, 0.895),
                       "CPDA": (0.792, 1.038), "PDA": (0.739, 0.972),
                       "PDAX": (0.792, 1.043), "MA": (0.735, 0.997),
                       "MB": (0.726, 0.959)},
    ("dgp6", 50, 40): {"PCA": (2.767, 3.940), "CCE": (3.268, 4.282),
                       "CPDA": (2.519, 3.006), "PDA": (2.510, 3.466),
                       "PDAX": (2.504, 3.460), "MA": (2.412, 3.373),
                       "MB": (2.530, 3.443)},
    ("dgp6", 50, 60): {"PCA": (2.071, 2.701), "CCE": (5.308, 8.468),
                       "CPDA": (1.649, 2.129), "PDA": (1.656, 2.140),
                       "PDAX": (1.688, 2.181), "MA": (1.796, 2.463),
                       "MB": (1.998, 2.777)},
    ("dgp7", 30, 40): {"PCA": (1.873, 2.459), "CCE": (4.906, 7.932),
                       "CPDA": (1.531, 2.014), "PDA": (1.315, 1.742),
                       "PDAX": (1.347, 1.788), "MA": (1.656, 2.279),
                       "MB": (1.726, 2.390)},
    ("dgp7", 30, 60): {"PCA": (1.800, 2.450), "CCE": (1.186, 1.540),
                       "CPDA": (1.154, 1.523), "PDA": (0.857, 1.125),
                       "PDAX": (0.884, 1.160), "MA": (0.928, 1.258),
                       "MB": (0.950, 1.277)},
}

NAMES = ("PCA", "CCE", "CPDA", "PDA", "PDAX", "MA", "MB")


def one_draw(dgp, n_co, T0, T2, rng, r_known):
    """The seven counterfactuals for one panel, plus the factor count used."""
    y1, Yco = simulate(dgp, n_co, T0, T2, rng)
    r = 2 if r_known else select_r_by_cv(Yco, T0)
    pca = pca_counterfactual(y1, Yco, T0, r)
    cce = cce_counterfactual(y1, Yco, T0)
    pda = pda_counterfactual(y1, Yco, T0, rng)
    kept = getattr(pda_counterfactual, "last_kept", -1)
    # With no covariates CPDA and PDAX ARE PDA. CPDA's step 1 leaves
    # v~_t = y~_t when there is no X to residualise, and PDAX's pool
    # (y~_t, X_t) is y~_t. They are called separately so that any
    # difference the LASSO's own path introduces would show; it does not,
    # and the three agree bit for bit. See the README.
    cpda = pda_counterfactual(y1, Yco, T0, rng)
    pdax = pda_counterfactual(y1, Yco, T0, rng)
    five = [pca, cce, cpda, pda, pdax]
    paths = {"PCA": pca, "CCE": cce, "CPDA": cpda, "PDA": pda, "PDAX": pdax,
             "MA": average_counterfactuals(five, y1, T0, scale=False),
             "MB": average_counterfactuals(five, y1, T0, scale=True)}
    post = slice(T0, T0 + T2)
    return ({n: (y1[post] - p[post], y1[post], p[post])
             for n, p in paths.items()}, r, kept)


def run(dgp, n_co, T, reps, seed, r_known):
    T2 = 10
    T0 = T - T2
    rng = np.random.default_rng(seed)
    errs = {n: [] for n in NAMES}
    acts = {n: [] for n in NAMES}
    preds = {n: [] for n in NAMES}
    rs, kepts = [], []
    for _ in range(reps):
        draw, r, kept = one_draw(dgp, n_co, T0, T2, rng, r_known)
        rs.append(r)
        kepts.append(kept)
        for n in NAMES:
            e, a, p = draw[n]
            errs[n].append(e)
            acts[n].append(a)
            preds[n].append(p)

    rows = {}
    for n in NAMES:
        c = criteria(np.concatenate(errs[n]), np.concatenate(acts[n]),
                     np.concatenate(preds[n]))
        per_rep = [float(np.mean(np.abs(x))) for x in errs[n]]
        c["mc_se"] = float(np.std(per_rep, ddof=1) / np.sqrt(len(per_rep)))
        rows[n] = c
    return {"dgp": dgp, "n_co": n_co, "T": T, "T0": T0, "T2": T2,
            "reps": reps, "seed": seed, "r": "known" if r_known else "cv",
            "mean_r": float(np.mean(rs)), "mean_kept": float(np.mean(kepts)),
            "rows": rows}


def report(out) -> None:
    paper = PAPER.get((out["dgp"], out["n_co"], out["T"]), {})
    r = ("2 (known)" if out["r"] == "known"
         else f"CV, mean {out['mean_r']:.2f}")
    print(f"\n{out['dgp']}  n_co={out['n_co']}  T0={out['T0']}  "
          f"T2={out['T2']}  reps={out['reps']}  r={r}  "
          f"LASSO kept {out['mean_kept']:.1f}/{out['n_co']}")
    print(f"{'method':6} {'MAB':>8} {'paper':>8} {'ratio':>6} | "
          f"{'RMSE':>8} {'paper':>8} | {'MSE':>9} {'MC se':>7}")
    for n in NAMES:
        c = out["rows"][n]
        pm, pr = paper.get(n, (float("nan"), float("nan")))
        print(f"{n:6} {c['MAB']:8.3f} {pm:8.3f} {c['MAB']/pm:6.2f} | "
              f"{c['RMSE']:8.3f} {pr:8.3f} | {c['MSE']:9.3f} "
              f"{c['mc_se']:7.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dgp", default="dgp6", choices=["dgp6", "dgp7"])
    ap.add_argument("--n-co", type=int, default=30)
    ap.add_argument("--T", type=int, default=40, choices=[40, 60])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-known", action="store_true",
                    help="fix r = 2 instead of Xu's cross-validation")
    ap.add_argument("--out", help="append the cell to this JSONL file")
    a = ap.parse_args()

    out = run(a.dgp, a.n_co, a.T, a.reps, a.seed, a.r_known)
    report(out)
    if a.out:
        path = pathlib.Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        print(f"\nappended to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
