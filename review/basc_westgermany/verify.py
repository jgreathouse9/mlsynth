"""Check every numerical claim in the BASC West Germany referee report.

Each claim is computed from the panel and compared against the value the report
states. Deterministic claims are held to a tight tolerance; MCMC claims are
given a band, since a chain reproduces to its seed and not to a decimal.

    python verify.py                # deterministic claims only, about a minute
    python verify.py --with-mcmc    # also reads the R diagnostic outputs
    python verify.py --json out.json

Exits non-zero if any claim fails, so it can gate a submission.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS = []


def check(name, got, expect, tol, note=""):
    """tol is absolute; expect may be a scalar or an (lo, hi) band."""
    if isinstance(expect, tuple):
        ok = expect[0] <= got <= expect[1]
        shown = f"[{expect[0]:g}, {expect[1]:g}]"
    else:
        ok = abs(got - expect) <= tol
        shown = f"{expect:g}"
    RESULTS.append({"claim": name, "reported": shown, "computed": round(float(got), 4),
                    "ok": bool(ok), "note": note})
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<52} report {shown:>14}   computed {got:>12.4f}")
    return ok


def panel(dta):
    d = pd.read_stata(dta)
    d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    return d


def arrays(d):
    from mlsynth.utils.datautils import dataprep
    p = dataprep(d, "country", "year", "gdp", "treat")
    y = np.asarray(p["y"]).flatten()
    X = np.asarray(p["donor_matrix"])
    return y, X, list(p["donor_names"]), int(p["pre_periods"])


# --------------------------------------------------------------------- toolkit
def estimators(d):
    from mlsynth import VanillaSC, CLUSTERSC, FDID, MVBBSC
    base = dict(df=d, unitid="country", time="year", outcome="gdp", treat="treat",
                display_graphs=False)
    print("\nToolkit fits on the ADH panel")
    v = VanillaSC({**base}).fit()
    check("VanillaSC pre-1990 RMSE", v.fit_diagnostics.rmse_pre, 60.8, 0.1)
    check("VanillaSC ATT", v.effects.att, -1297.5, 2.0)

    r = CLUSTERSC({**base, "method": "rpca", "rpca_method": "PCP"}).fit()
    r = r.sub_method_results["RPCA"] if getattr(r, "sub_method_results", None) else r
    check("CLUSTERSC RPCA-SC (fPCA-SYNTH) pre-RMSE", r.fit_diagnostics.rmse_pre, 88.6, 0.5)
    check("CLUSTERSC RPCA-SC ATT", r.effects.att, -1501, 15,
          "paper reports -1655 at its stated settings")

    c = CLUSTERSC({**base, "method": "PCR"}).fit()
    c = c.sub_method_results["PCR"] if getattr(c, "sub_method_results", None) else c
    check("CLUSTERSC PCR (ClusterSC) pre-RMSE", c.fit_diagnostics.rmse_pre, 98.0, 1.0)
    check("CLUSTERSC PCR ATT", c.effects.att, -2039, 25)

    f = FDID({**base}).fit()
    f = f.sub_method_results["FDID"] if getattr(f, "sub_method_results", None) else f
    check("FDID pre-1990 RMSE", f.fit_diagnostics.rmse_pre, 83.4, 1.0)

    m = MVBBSC({**base, "n_warmup": 500, "n_samples": 500, "n_chains": 4, "seed": 0}).fit()
    check("MVBBSC (B-MV) pre-1990 RMSE", m.fit_diagnostics.rmse_pre, 62.1, 1.5)
    check("MVBBSC (B-MV) ATT", m.effects.att, -2079, 60)


# ------------------------------------------------------- the paper's own Table 7
S_SCM = {"USA": .05, "UK": .05, "Austria": .05, "Belgium": .05, "Denmark": .05, "France": .04,
         "Italy": .04, "Netherlands": .04, "Norway": .05, "Switzerland": .04, "Japan": .04,
         "Greece": .03, "Portugal": .04, "Spain": .34, "Australia": .04, "New Zealand": .05}
BASC_W = {"USA": 0.00, "UK": 0.00, "Austria": 0.00, "Belgium": 0.00, "Denmark": 0.00,
          "France": 0.00, "Italy": 0.12, "Netherlands": 0.00, "Norway": 0.00,
          "Switzerland": 0.44, "Japan": 0.37, "Greece": 0.01, "Portugal": 0.05,
          "Spain": 0.00, "Australia": 0.00, "New Zealand": 0.00}


def table7(y, X, names, npre):
    pre = np.arange(len(y)) < npre
    print("\nThe paper's Table 7, using only its published weights")
    w = np.array([S_SCM[n] for n in names])
    syn = X @ w
    check("Table 7 s-SCM pre-1990 RMSE", np.sqrt(np.mean((y[pre] - syn[pre]) ** 2)), 1925.5, 1.0)
    check("Table 7 s-SCM implied ATT", (y[~pre] - syn[~pre]).mean(), 2891.5, 2.0,
          "paper's text reports -159")
    u = np.full(len(names), 1 / len(names))
    check("uniform weights pre-1990 RMSE", np.sqrt(np.mean((y[pre] - (X @ u)[pre]) ** 2)),
          1289.3, 1.0)

    rng = np.random.default_rng(0)
    rs, ats = [], []
    for _ in range(4000):
        wp = np.clip(w + rng.uniform(-0.005, 0.005, w.size), 0, None); wp /= wp.sum()
        s = X @ wp
        rs.append(np.sqrt(np.mean((y[pre] - s[pre]) ** 2)))
        ats.append((y[~pre] - s[~pre]).mean())
    check("rounding sweep, min RMSE", min(rs), (1855, 1866), 0)
    check("rounding sweep, max RMSE", max(rs), (1996, 2007), 0)
    check("rounding sweep, ATT stays positive", float(min(ats) > 0), 1.0, 0,
          f"min ATT over 4000 draws {min(ats):.0f}")


# ----------------------------------------------------- the full-sample likelihood
def likelihood(y, X, names, npre):
    import cvxpy as cp
    pre = np.arange(len(y)) < npre
    D = (~pre).astype(float)
    print("\nWhat the full-sample likelihood does to the weights")

    def solve(mask, basis):
        w = cp.Variable(X.shape[1], nonneg=True)
        if basis is None:
            r = y[mask] - X[mask] @ w
        else:
            a = cp.Variable(basis.shape[1]); r = y[mask] - X[mask] @ w - basis[mask] @ a
        cp.Problem(cp.Minimize(cp.sum_squares(r)), [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
        return np.asarray(w.value).ravel(), (np.atleast_1d(a.value) if basis is not None
                                             else np.zeros(1))

    allm = np.ones(len(y), bool)
    rm = lambda v: float(np.sqrt(np.mean((y[pre] - (X @ v)[pre]) ** 2)))
    gap = lambda v: float((y[~pre] - (X @ v)[~pre]).mean())

    w_pre, _ = solve(pre, None)
    check("pre-period only: pre-1990 RMSE", rm(w_pre), 60.8, 0.1)
    check("pre-period only: effect", gap(w_pre), -1297, 2)

    w_const, a_const = solve(allm, D.reshape(-1, 1))
    check("all 44 years, one constant: pre-1990 RMSE", rm(w_const), 198.4, 0.5)
    check("all 44 years, one constant: constant", float(a_const[0]), -351, 3)
    check("all 44 years, one constant: L1 from pre-only", float(np.abs(w_const - w_pre).sum()),
          1.627, 0.02)

    w_sat, _ = solve(allm, np.eye(len(y))[:, ~pre])
    check("saturated effect: pre-1990 RMSE", rm(w_sat), 60.8, 0.1)
    check("saturated effect: L1 from pre-only", float(np.abs(w_sat - w_pre).sum()), 0.0, 1e-3,
          "the post-period carries no information about w")

    paper = np.array([BASC_W[n] for n in names])
    check("L1, paper's BASC column vs full-sample least squares",
          float(np.abs(paper - w_const).sum()), 0.041, 0.005)


# --------------------------------------------------------------- MCMC diagnostics
def mcmc(datadir):
    print("\nMCMC diagnostics (from the R outputs)")
    g = pd.read_csv(os.path.join(datadir, "gamma1_diagnostic.csv"))
    q = pd.read_csv(os.path.join(datadir, "effect_basis_diagnostic.csv"))
    dc = pd.read_csv(os.path.join(datadir, "selection_decomposition.csv"))

    row = g[(g.config == "basc") & (g.N == 2000) & (g.seed == 200)]
    if len(row):
        check("control, 2000/2000 seed 200: RMSE", row.rmse_pre.iloc[0], 169.45, 0.01)
        check("control, 2000/2000 seed 200: ATT", row.att_post.iloc[0], -582.50, 0.05)
    row = g[(g.config == "basc") & (g.N == 25000)]
    if len(row):
        check("control, 25000/25000: RMSE", row.rmse_pre.iloc[0], 195.516, 0.01)
        check("control, 25000/25000: ATT", row.att_post.iloc[0], -389.799, 0.05)

    s = g[(g.config == "g1_a1") & (g.N == 2000)]
    if len(s):
        check("gamma=1, alpha_u=1: RMSE across seeds", s.rmse_pre.max(), (185, 205), 0,
              "does not return to B-MV's 62")
    s = g[(g.config == "g1_a2.5") & (g.N == 2000)]
    if len(s):
        check("gamma=1, alpha_u=2.5: RMSE across seeds", s.rmse_pre.max(), (195, 215), 0)

    s = q[(q.config == "q2_linear") & (q.N == 2000)]
    if len(s):
        check("q=2 linear basis: RMSE across seeds", s.rmse_pre.max(), (125, 160), 0)
        check("q=2 linear basis: ATT across seeds", s.att_post.mean(), (-1500, -900), 0,
              "standard SCM returns -1297")

    d2 = dc[dc.N == 2000]
    if len(d2):
        check("q=2: fit flat whether selection is on or off",
              float(d2.rmse_pre.max() - d2.rmse_pre.min()), (0, 20), 0,
              "selection is neither buying nor costing fit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dta", default="basedata/repgermany.dta")
    ap.add_argument("--data", default="data")
    ap.add_argument("--with-mcmc", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    import mlsynth
    print(f"mlsynth {mlsynth.__version__}   panel {a.dta}")
    d = panel(a.dta)
    y, X, names, npre = arrays(d)
    print(f"panel: {len(y)} periods, {X.shape[1]} donors, {npre} pre-treatment")

    estimators(d)
    table7(y, X, names, npre)
    likelihood(y, X, names, npre)
    if a.with_mcmc:
        mcmc(a.data)

    failed = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} claims reproduce")
    if a.json:
        json.dump(RESULTS, open(a.json, "w"), indent=2)
    if failed:
        print("failed:")
        for r in failed:
            print(f"  {r['claim']}: report {r['reported']}, computed {r['computed']}")
    sys.exit(1 if failed else 0)
