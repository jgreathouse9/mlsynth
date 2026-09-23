"""SDM on the authors' own demo panel, against exact constrained least squares.

The demo panel ships in the authors' replication package and is not committed
here (see README, Provenance). Point the script at it:

    python benchmarks/reference/sdm_kyokawata/run_demo.py --data /path/to/data_demo_F.txt

Writes ``results.json`` beside this file. The comparison arms are the four
SC-class variants of Li and Shankar (2023), solved exactly, which is what
``mlsynth.TSSC`` fits: SC (simplex), MSCa (simplex plus free intercept), MSCb
(non-negative, no intercept, no adding-up) and MSCc (non-negative plus free
intercept). MSCc is the objective SDM's iteration converges toward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdm_port import eq14, sdm_fit  # noqa: E402

HERE = Path(__file__).resolve().parent
TREATED = "Fukui"
INTERVENTION = 2015
SCALE = 1e5  # conditioning for the cvxpy solves; outcomes are ~3e5


def load_panel(path):
    rows = [l.split("\t") for l in Path(path).read_text().strip().split("\n")][1:]
    units = sorted({r[1] for r in rows})
    years = sorted({int(r[0]) for r in rows})
    Y = np.full((len(years), len(units)), np.nan)
    yi = {y: i for i, y in enumerate(years)}
    ui = {u: i for i, u in enumerate(units)}
    for r in rows:
        Y[yi[int(r[0])], ui[r[1]]] = float(r[2])
    return Y, units, years


def simplex_fits(yp, Xp, ones):
    """SC and MSCa. Returns ``None`` when cvxpy is unavailable or fails."""
    try:
        import cvxpy as cp
    except ImportError:
        return None
    out = {}
    w = cp.Variable(Xp.shape[1])
    cp.Problem(
        cp.Minimize(cp.sum_squares(yp - Xp @ w)), [w >= 0, cp.sum(w) == 1]
    ).solve(solver=cp.SCS, eps=1e-10, max_iters=200_000)
    if w.value is None:
        return None
    out["SC"] = (None, np.asarray(w.value))
    D = np.hstack([ones, Xp])
    v = cp.Variable(D.shape[1])
    cp.Problem(
        cp.Minimize(cp.sum_squares(yp - D @ v)), [v[1:] >= 0, cp.sum(v[1:]) == 1]
    ).solve(solver=cp.SCS, eps=1e-10, max_iters=200_000)
    if v.value is None:
        return None
    out["MSCa"] = (float(v.value[0]), np.asarray(v.value[1:]))
    return out


def tssc_mscc(path, T0):
    """MSCc as ``mlsynth.TSSC`` fits it. Returns ``None`` if unavailable."""
    try:
        import pandas as pd

        from mlsynth import TSSC
    except ImportError:
        return None
    df = pd.read_csv(path, sep="\t")
    df["treated"] = (
        (df["pref"] == TREATED) & (df["year"] >= INTERVENTION)
    ).astype(int)
    summary = TSSC(
        {
            "df": df,
            "outcome": "Y",
            "treat": "treated",
            "unitid": "pref",
            "time": "year",
            "method": "MSCc",
            "inference": False,
        }
    ).fit().summary
    y = np.asarray(summary.time_series.observed_outcome, dtype=float)
    cf = np.asarray(summary.time_series.counterfactual_outcome, dtype=float)
    w = summary.weights.donor_weights
    return dict(
        sum_w=float(sum(v for v in w.values() if abs(v) > 1e-6)),
        intercept=None,
        pre_mspe=float(np.mean((y[:T0] - cf[:T0]) ** 2)),
        att=float(summary.effects.att),
        active_donors=int(sum(1 for v in w.values() if abs(v) > 1e-6)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=os.environ.get("SDM_DEMO_DATA"),
        help="path to data_demo_F.txt (or set SDM_DEMO_DATA)",
    )
    args = ap.parse_args()
    if not args.data or not Path(args.data).exists():
        sys.exit(
            "data_demo_F.txt not found. It is part of the authors' replication\n"
            "package and is not committed here; pass --data or set SDM_DEMO_DATA.\n"
            "The committed results.json records the numbers from a prior run."
        )

    Y, units, years = load_panel(args.data)
    ti = units.index(TREATED)
    T0 = sum(y < INTERVENTION for y in years)
    T = len(years)
    donors = [
        j for j in range(len(units)) if j != ti and not np.isnan(Y[:, j]).any()
    ]
    yy, XX = Y[:, ti], Y[:, donors]
    print(f"treated={TREATED}  T0={T0}  T1={T - T0}  donors={len(donors)}")

    fit = sdm_fit(yy, XX, T0)
    print(f"\niterations to stop: {fit['iterations']}")
    print(f"Reg1 scope fall-throughs: {fit['scope_fallthroughs']}")
    print(
        f"rho as coded {fit['rho_coded'][-1]:.3e} vs Eq.(13) "
        f"{fit['rho_paper'][-1]:.3e} (threshold 1e-4)"
    )

    arms = {}
    for label, w in (
        ("SDM objective (sum w free)", fit["w_objective"]),
        ("SDM bookkeeping (Sbet)", fit["w_bookkeeping"]),
        ("SDM shipped (renormalised)", fit["w_shipped"]),
    ):
        alpha, _, pre, att = eq14(yy, XX, w, T0)
        arms[label] = dict(
            sum_w=float(w.sum()), intercept=alpha, pre_mspe=pre, att=att
        )

    yp, Xp = yy[:T0], XX[:T0]
    ones = np.ones((T0, 1))

    wb, _ = nnls(Xp, yp)
    cf_b = XX @ wb  # MSCb carries no intercept
    arms["MSCb exact"] = dict(
        sum_w=float(wb.sum()),
        intercept=0.0,
        pre_mspe=float(np.mean((yp - cf_b[:T0]) ** 2)),
        att=float(np.mean(yy[T0:] - cf_b[T0:])),
    )

    Dc = np.hstack([ones, Xp])
    lb = np.r_[-np.inf, np.zeros(Xp.shape[1])]
    bc = lsq_linear(
        Dc, yp, bounds=(lb, np.full(Dc.shape[1], np.inf)), tol=1e-12, max_iter=500
    ).x
    cf_c = bc[0] + XX @ bc[1:]
    arms["MSCc exact"] = dict(
        sum_w=float(bc[1:].sum()),
        intercept=float(bc[0]),
        pre_mspe=float(np.mean((yp - cf_c[:T0]) ** 2)),
        att=float(np.mean(yy[T0:] - cf_c[T0:])),
        active_donors=int((bc[1:] > 1e-8).sum()),
    )

    # The claim the spike turns on: mlsynth already fits SDM's fixed point.
    # Read it from the library, not from a stand-in solver.
    tssc = tssc_mscc(args.data, T0)
    if tssc is None:
        print("\nmlsynth unavailable; TSSC arm omitted")
    else:
        arms["TSSC(method='MSCc') via mlsynth"] = tssc

    simplex = simplex_fits(yp / SCALE, Xp / SCALE, ones)
    if simplex is None:
        print("\ncvxpy unavailable or solve failed; SC / MSCa omitted")
    else:
        for label, (a, w) in simplex.items():
            alpha = 0.0 if a is None else a * SCALE
            cf = alpha + XX @ w
            arms[f"{label} exact"] = dict(
                sum_w=float(w.sum()),
                intercept=float(alpha),
                pre_mspe=float(np.mean((yp - cf[:T0]) ** 2)),
                att=float(np.mean(yy[T0:] - cf[T0:])),
            )

    print(f"\n{'arm':<30} {'sum w':>8} {'pre-MSPE':>16} {'ATT':>12}")
    for label in sorted(arms, key=lambda k: arms[k]["pre_mspe"]):
        a = arms[label]
        print(
            f"{label:<30} {a['sum_w']:>8.4f} {a['pre_mspe']:>16,.0f} "
            f"{a['att']:>12,.0f}"
        )

    kept = {
        units[donors[j]]: round(float(fit["w_shipped"][j]), 4)
        for j in np.flatnonzero(fit["w_shipped"] > 0.01)
    }
    out = dict(
        treated=TREATED,
        intervention=INTERVENTION,
        T0=T0,
        T1=T - T0,
        n_donors=len(donors),
        iterations=fit["iterations"],
        scope_fallthroughs=fit["scope_fallthroughs"],
        rho_coded_final=fit["rho_coded"][-1],
        rho_paper_final=fit["rho_paper"][-1],
        scale_offset=fit["scale_offset"],
        donors_kept_shipped=kept,
        arms=arms,
    )
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
