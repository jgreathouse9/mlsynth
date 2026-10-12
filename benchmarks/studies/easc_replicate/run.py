"""EASC against the author's own code, run live, and against the paper's tables.

Reference: Rok Spruk (2026), "Synthetic Control under Out-of-Span Common
Shocks: Diagnosis, Exposure Balance and Correction", arXiv:2610.05535v1.

This is a replication spike, not an estimator and not a pinned case. It answers
one question before any build starts: does a port that ingests through
``dataprep`` reproduce the author's arithmetic cell for cell, and the paper's
printed tables to the precision they are printed?

Run::

    python -m benchmarks.studies.easc_replicate.run               # Path A, MC cross-check, MC at 100 reps
    python -m benchmarks.studies.easc_replicate.run --reps 500    # MC at the paper's 500 reps

The reference numbers come from ``benchmarks/reference/easc_iran/reference.json``,
captured by ``capture.py`` in that directory from a live run of the author's
replication package. Results are written to ``results/easc_replicate.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from . import activeset_frontier as AF
from . import oracle as O
from . import paper_values as P
from . import simulate as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PANEL = ROOT / "basedata" / "iran_sanctions_oil.csv"
REFERENCE = ROOT / "benchmarks" / "reference" / "easc_iran" / "reference.json"
OUT = HERE / "results" / "easc_replicate.json"
POOLS = {"base": S.BASE, "ext": S.BASE + ["AZE", "AGO"], "base+AGO": S.BASE + ["AGO"],
         "base+AZE": S.BASE + ["AZE"]}
# Exposure measures: (column, first year, last year), means over the window as in code/app.py.
EXPOSURE = {"oilrent_9611": ("oilrent", 1996, 2011), "oilrent_0211": ("oilrent", 2002, 2011),
            "oilrent_9605": ("oilrent", 1996, 2005), "vdem_9606": ("oil_share", 1996, 2006)}
# code/app.py's robustness specifications, in its order: (label, pool, exposure, k, controls).
ROBUSTNESS = [
    ("Oil rents 1996-2011 (baseline)", "base", "oilrent_9611", 2, True),
    ("Oil rents 2002-2011", "base", "oilrent_0211", 2, True),
    ("V-Dem oil income share", "base", "vdem_9606", 2, True),
    ("No pre-treatment growth controls in the exposure regression", "base", "oilrent_9611", 2, False),
    ("Three latent factors", "base", "oilrent_9611", 3, True),
    ("One latent factor", "base", "oilrent_9611", 1, True),
    ("Exposure-feasible pool (21)", "ext", "oilrent_9611", 2, True),
    ("Exposure-feasible pool, V-Dem share", "ext", "vdem_9606", 2, True),
    ("Exposure-feasible pool without Azerbaijan (20)", "base+AGO", "oilrent_9611", 2, True),
    ("Exposure-feasible pool without Angola (20)", "base+AZE", "oilrent_9611", 2, True),
    ("Exposure-feasible pool, oil rents 1996--2005", "ext", "oilrent_9605", 2, True),
    ("Exposure-feasible pool, three latent factors", "ext", "oilrent_9611", 3, True),
    ("Exposure-feasible pool, no growth controls", "ext", "oilrent_9611", 2, False),
]


# ------------------------------------------------------------------- ingestion


class Inputs:
    """One donor pool through ``dataprep``, which is the only pivot in the study."""

    def __init__(self, pool: Sequence[str], exposure: str = "oilrent_9611"):
        from mlsynth.utils.datautils import dataprep

        df = pd.read_csv(PANEL)
        prepped = dataprep(df[df.iso.isin(["IRN", *pool])], "iso", "year", "lgdppc", "sanctions")
        self.donors: List[str] = list(prepped["donor_names"])
        self.years = np.asarray(prepped["time_labels"], dtype=int)
        self.y1 = np.asarray(prepped["y"], dtype=float).ravel()
        self.Y0 = np.asarray(prepped["donor_matrix"], dtype=float)
        self.pre = np.arange(len(self.years)) < int(prepped["pre_periods"])
        self.base = int(np.where(self.pre)[0][-1])
        col, a, b = EXPOSURE[exposure]
        # Unit means over the window; the Dominican Republic and Panama report no oil
        # rents and are set to zero, as in code/app.py.
        s = df[df.year.between(a, b)].groupby("iso")[col].mean().reindex(["IRN", *self.donors]).fillna(0.0)
        self.s1, self.s0 = float(s.iloc[0]), s.iloc[1:].to_numpy(dtype=float)
        brent = df.drop_duplicates("year").set_index("year")["brent"]
        self.x = np.log(brent.loc[self.years].to_numpy(dtype=float))
        self.phases = {k: (self.years >= lo) & (self.years <= hi) for k, (lo, hi) in S.PHASES.items()}

    def phase_means(self, g: np.ndarray) -> Dict[str, float]:
        return {k: float(g[m].mean()) for k, m in self.phases.items()}


# --------------------------------------------------------------- comparisons


def _maxdiff(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _shown(v: float, d: int) -> str:
    s = f"{v:.{d}f}"
    return "0." + "0" * d if s == "-0." + "0" * d else s


def _display(rows: list, label: str, mine: float, printed: float, d: int) -> None:
    """Record whether ``mine`` prints as the paper's ``printed`` at ``d`` decimals."""
    rows.append(dict(quantity=label, ours=mine, printed=printed, decimals=d,
                     match=_shown(mine, d) == _shown(printed, d),
                     off_by_digits=round(abs(round(mine, d) - printed) * 10 ** d)))


# ---------------------------------------------------------------- Path A


def path_a(ref: Dict) -> Dict:
    t = ref["tables"]
    cross: Dict[str, float] = {}
    display: list = []
    fits = {}
    for key in ("base", "ext"):
        inp = Inputs(POOLS[key])
        fit = O.analyse(inp.y1, inp.Y0, inp.pre, inp.s1, inp.s0, inp.x, inp.base)
        fits[key] = (inp, fit)
        r = t[key]
        order = [inp.donors.index(d) for d in r["w_sc"]]
        cross[f"{key}: SC weights"] = _maxdiff(fit["w_sc"][order], list(r["w_sc"].values()))
        cross[f"{key}: exposure s0"] = _maxdiff(inp.s0[order], list(r["s0"].values()))
        for q in ("dmin", "ei_sc", "rmspe_sc", "r2", "r2_all", "ei_min", "rmspe_min", "ei_tuned"):
            cross[f"{key}: {q}"] = _maxdiff(fit[q], r[q])
        for q in ("kappa", "kappa_lat", "partial", "partial_all", "phi", "se_phi", "g_sc", "g_easc", "g_ebsc_min",
                  "g_easc_tuned"):
            cross[f"{key}: {q} (29 periods)"] = _maxdiff(fit[q], r[q])
        cross[f"{key}: frontier tolerances (41)"] = _maxdiff([p["tol"] for p in fit["path"]], [p["tol"] for p in r["path"]])
        cross[f"{key}: frontier EI (41)"] = _maxdiff([p["ei"] for p in fit["path"]], [p["ei"] for p in r["path"]])
        cross[f"{key}: frontier RMSPE (41)"] = _maxdiff([p["rmspe"] for p in fit["path"]], [p["rmspe"] for p in r["path"]])
        cross[f"{key}: frontier criterion (41)"] = _maxdiff([p["crit"] for p in fit["path"]], [p["crit"] for p in r["path"]])
        cross[f"{key}: frontier weights (41 x J)"] = max(
            _maxdiff(p["w"][order], list(q["w"].values())) for p, q in zip(fit["path"], r["path"]))
        cross[f"{key}: EASC* frontier point"] = abs(fit["best"]["frac"] - r["best_frac"])

        rows = O.phase_inference(inp.y1, inp.Y0, inp.pre, inp.s1, inp.s0, inp.x, inp.base, inp.phases,
                                 Ms=(0.0, 0.5, 1.0))
        ref_rows = {(q["est"], q["phase"]): q for q in t[f"{key}_inference"]}
        for q in ("tau", "sd", "sdm", "kbar", "pval", "Mstar", "Mstar_mad"):
            cross[f"{key}: inference {q} (2 estimators x 3 phases)"] = max(
                abs(m[q] - ref_rows[(m["est"], m["phase"])][q]) for m in rows)
        cross[f"{key}: inference CI bounds at M = 0, 0.5, 1"] = max(
            _maxdiff([m["ci"][M] for M in (0.0, 0.5, 1.0)],
                     [ref_rows[(m["est"], m["phase"])][c] for c in ("ci0", "ci05", "ci1")]) for m in rows)

        # Table 4.
        t4 = P.TABLE4[key]
        post_mask = inp.years >= 2012
        mine4 = {"s1": inp.s1, "s0_min": inp.s0.min(), "s0_max": inp.s0.max(), "dmin": fit["dmin"],
                 "ei_sc": fit["ei_sc"], "rmspe_sc": fit["rmspe_sc"], "rmspe_min": fit["rmspe_min"], "r2": fit["r2"],
                 "max_ell_pre": np.abs(fit["partial"][inp.pre]).max(),
                 "max_ell_post": np.abs(fit["partial"][post_mask]).max()}
        for q, vd in t4.items():
            if q in mine4:
                _display(display, f"Table 4 {key} {q}", float(mine4[q]), *vd)
        ell = fit["partial"]
        for ph, m in inp.phases.items():
            k_sc = O.phase_leverage(O.with_constant(fit["Fl"], inp.x), inp.pre, m)
            k_ea = float(np.sqrt(O.phase_leverage(O.with_constant(fit["Fl"]), inp.pre, m) ** 2 + ell[inp.base] ** 2))
            _display(display, f"Table 4 {key} kappa SC {ph}", k_sc, *t4["kappa_sc"][ph])
            _display(display, f"Table 4 {key} kappa EASC {ph}", k_ea, *t4["kappa_easc"][ph])

        # Table 5, panel A.
        gaps = {"SC": fit["g_sc"], "EBSC-min": fit["g_ebsc_min"], "EASC": fit["g_easc"], "EASC*": fit["g_easc_tuned"]}
        for est, vals in P.TABLE5_POINT[key].items():
            for ph, v in zip(P.PHASES, vals):
                _display(display, f"Table 5A {key} {est} {ph}", inp.phase_means(gaps[est])[ph], v, 3)
        # Table 5, panels B and C.
        for m in rows:
            pv = P.TABLE5_INFERENCE[(key, m["est"])]
            i = P.PHASES.index(m["phase"])
            for q in ("pval", "sd", "kbar", "Mstar", "Mstar_mad"):
                _display(display, f"Table 5 {key} {m['est']} {m['phase']} {q}", m[q], pv[q][i], P.INF_DECIMALS[q])
            for q, M in (("ci0", 0.0), ("ci05", 0.5)):
                for side in (0, 1):
                    _display(display, f"Table 5 {key} {m['est']} {m['phase']} {q}[{side}]", m["ci"][M][side],
                             pv[q][i][side], 2)
        for ph, (v, tv) in zip(P.PHASES, P.TABLE5_PHI[key]):
            m = inp.phases[ph]
            phib = float(fit["phi"][m].mean())
            se = O.jackknife_phase_se(fit["jk_phi"], m)
            _display(display, f"Table 5C {key} 100 x phi {ph}", 100 * phib, v, 2)
            _display(display, f"Table 5C {key} jackknife t {ph}", phib / se, tv, 1)
            cross[f"{key}: phase phi and jackknife se ({ph})"] = max(
                abs(phib - t["phi_phase"][key][ph][0]), abs(se - t["phi_phase"][key][ph][1]))

        # Influence: hat values and delete-one phase averages of phi (code/app_influence.py).
        hat = O.regression_hat_values(inp.Y0, inp.s0, inp.pre, inp.base)
        ref_inf = {row["donor"]: row for row in t["influence"][key]["rows"]}
        cross[f"{key}: hat values"] = max(abs(hat[inp.donors.index(d)] - row["hat"]) for d, row in ref_inf.items())
        worst = 0.0
        for j, d in enumerate(inp.donors):
            keep = [i for i in range(len(inp.donors)) if i != j]
            phid, _, _ = O.exposure_effects(inp.Y0[:, keep], inp.s0[keep], inp.pre, inp.base)
            for ph, m in inp.phases.items():
                worst = max(worst, abs(100 * phid[m].mean() - ref_inf[d][f"phi_{ph}"]))
        cross[f"{key}: delete-one-donor phi x 100"] = worst
        if key == "ext":
            for d, v in P.HAT_VALUES.items():
                _display(display, f"Section 6.3 hat value {d}", float(hat[inp.donors.index(d)]), v, 2)

        # Leverage and breakdown by number of latent factors (Online Appendix).
        sc_shock = next(m for m in rows if m["est"] == "SC" and m["phase"] == "shock")
        rmspe = fit["rmspe_sc"]
        for k in (1, 2, 3):
            Fl = O.donor_factors(inp.Y0, k, inp.s0)
            _, r2k = O.partial_leverage(inp.x, Fl, inp.pre)
            kk = {ph: O.phase_leverage(O.with_constant(Fl, inp.x), inp.pre, m) for ph, m in inp.phases.items()}
            ref_k = t["influence"]["leverage_k"][f"{key}_k{k}"]
            cross[f"{key}: leverage with k = {k}"] = max(abs(kk[ph] - ref_k[ph]) for ph in P.PHASES)
            r2p, kp, mp = P.TABLE_FACTORS[(key, k)]
            _display(display, f"Appendix factors {key} k={k} R2", r2k, r2p, 3)
            for ph, v in zip(P.PHASES, kp):
                _display(display, f"Appendix factors {key} k={k} kappa {ph}", kk[ph], v, 1)
            _display(display, f"Appendix factors {key} k={k} M* (SC, 2012-15)",
                     O.breakdown(sc_shock["tau"], kk["shock"], rmspe, sc_shock["sd"]), mp, 2)

        # The text-versus-code choice for EASC's correction variance.
        alt = O.phase_inference(inp.y1, inp.Y0, inp.pre, inp.s1, inp.s0, inp.x, inp.base, inp.phases,
                                easc_se="phase")
        fits[key] = (inp, fit, rows, alt)

    # Robustness (Table 7 and code/app.py's longer list).
    rob_ref = t["robustness"]
    rob_rows, worst = [], {}
    for (label, pool, expo, k, ctl), r in zip(ROBUSTNESS, rob_ref):
        inp = Inputs(POOLS[pool], expo)
        fit = O.analyse(inp.y1, inp.Y0, inp.pre, inp.s1, inp.s0, inp.x, inp.base, k=k, controls=ctl)
        mine = dict(ei_sc=fit["ei_sc"], ei_t=fit["ei_tuned"], rmspe_min=fit["rmspe_min"], dmin=fit["dmin"])
        for tag, g in (("sc", fit["g_sc"]), ("easc", fit["g_easc"]), ("easct", fit["g_easc_tuned"]),
                       ("ebmin", fit["g_ebsc_min"])):
            for ph, v in inp.phase_means(g).items():
                mine[f"{tag}_{ph}"] = v
        for q, v in mine.items():
            worst[q] = max(worst.get(q, 0.0), abs(v - r[q]))
        rob_rows.append(dict(spec=label, **mine))
    for q, v in worst.items():
        cross[f"robustness (13 specifications): {q}"] = v
    by_label = {row["spec"]: row for row in rob_rows}
    label_map = {"No pre-treatment growth controls": "No pre-treatment growth controls in the exposure regression",
                 "Exposure-feasible pool, oil rents 1996-2005": "Exposure-feasible pool, oil rents 1996--2005"}
    for label, pool, expo, ctl, ei, easc, ebmin in P.TABLE7:
        row = by_label[label_map.get(label, label)]
        _display(display, f"Table 7 {label}: EI(SC)", row["ei_sc"], ei, 1)
        for ph, v in zip(P.PHASES, easc):
            _display(display, f"Table 7 {label}: EASC {ph}", row[f"easc_{ph}"], v, 3)
        feasible = row["rmspe_min"] <= 0.1
        display.append(dict(quantity=f"Table 7 {label}: EBSC-min feasible", ours=feasible, printed=ebmin is not None,
                            match=feasible == (ebmin is not None)))
        if ebmin is not None:
            for ph, v in zip(P.PHASES, ebmin):
                _display(display, f"Table 7 {label}: EBSC-min {ph}", row[f"ebmin_{ph}"], v, 3)

    # Mutual consistency of the alternative EASC variance (text, not code).
    variance_choice = []
    for key in ("base", "ext"):
        _, _, rows, alt = fits[key]
        for m, a in zip(rows, alt):
            if m["est"] == "EASC":
                variance_choice.append(dict(pool=key, phase=m["phase"], sd_code=m["sd"], sd_text=a["sd"],
                                            Mstar_code=m["Mstar"], Mstar_text=a["Mstar"]))
    return dict(cross_validation=cross, display=display, easc_variance_choice=variance_choice)


# ---------------------------------------------------------------- Path B


def _rep(args):
    scn, rep = args
    return S.one_rep(S.calibrate(PANEL), scn, rep)


def path_b(ref: Dict, reps: int, designs: Iterable[str], procs: int) -> Dict:
    cal = S.calibrate(PANEL)
    mc = ref["tables"]["mc"]
    bitwise = 0
    for key, d in mc["draws"].items():
        scn, r = key.split("/")
        Y, _ = S.draw(cal, scn, np.random.default_rng(d["seed"]))
        bitwise += hashlib.sha256(np.ascontiguousarray(Y).tobytes()).hexdigest() == d["Y_sha256"]
    per_rep = {"err": 0.0, "inf": 0.0, "diag": 0.0, "coverage_flips": 0}
    for key, d in mc["one_rep"].items():
        scn, r = key.split("/")
        mine = S.one_rep(cal, scn, int(r))
        per_rep["err"] = max(per_rep["err"], max(abs(mine["err"][e][p] - d["err"][e][p])
                                                 for e in d["err"] for p in d["err"][e]))
        for k, v in d["inf"].items():
            m = mine["inf"][k]
            if isinstance(v, list):
                per_rep["inf"] = max(per_rep["inf"], abs(m[1] - v[1]))
                per_rep["coverage_flips"] += int(m[0] != v[0])
            else:
                per_rep["inf"] = max(per_rep["inf"], abs(m - v))
        per_rep["diag"] = max(per_rep["diag"], max(abs(mine["diag"][k] - v) for k, v in d["diag"].items()))

    jobs = [(scn, r) for scn in designs for r in range(reps)]
    t0 = time.time()
    with Pool(procs) as pool:
        out = pool.map(_rep, jobs, chunksize=10)
    summary = S.summarise(out)
    display = []
    for name, dct, d in (("bias", P.MC_BIAS, 3), ("rmse", P.MC_RMSE, 3)):
        for (scn, ph, est), v in dct.items():
            k = f"{scn}|{ph}|{est}"
            if k in summary[name]:
                _display(display, f"MC {name} {k}", summary[name][k], v, d)
    for name, dct, d in (("coverage", P.MC_COVERAGE, 2), ("length", P.MC_LENGTH, 3)):
        for (scn, ph, est, M), v in dct.items():
            k = f"{scn}|{ph}|{est}|{M}"
            if k in summary[name]:
                _display(display, f"MC {name} {k}", summary[name][k], v, d)
    return dict(draws_bitwise=f"{bitwise}/{len(mc['draws'])}", per_replication_maxdiff=per_rep,
                reps=reps, designs=list(designs), seconds=round(time.time() - t0, 1), summary=summary,
                display=display)


def solver_choice(ref: Dict) -> Dict:
    """The frontier on the active-set solver against the CLARABEL reference, both pools."""
    out = {}
    for key in ("base", "ext"):
        inp = Inputs(POOLS[key])
        a, A = inp.y1[inp.pre], inp.Y0[inp.pre]
        w_sc = O.sc_weights(a, A)
        path = ref["tables"][key]["path"]
        order = [inp.donors.index(d) for d in ref["tables"][key]["w_sc"]]
        t0 = time.time()
        pts = [AF.frontier_point(a, A, inp.s1, inp.s0, p["tol"], w_sc)[0] for p in path]
        secs = time.time() - t0
        t0 = time.time()
        O.frontier(a, A, inp.s1, inp.s0, n=len(path))
        secs_cl = time.time() - t0
        out[key] = dict(
            points=len(path),
            max_abs_rmspe=max(abs(float(np.sqrt(np.mean((a - A @ w) ** 2))) - p["rmspe"]) for w, p in zip(pts, path)),
            max_abs_ei=max(abs(inp.s1 - inp.s0 @ w - p["ei"]) for w, p in zip(pts, path)),
            max_abs_weight=max(float(np.abs(w[order] - np.array(list(p["w"].values()))).max()) for w, p in zip(pts, path)),
            seconds_active_set=round(secs, 2), seconds_clarabel=round(secs_cl, 2))
    return out


def _tally(display: list) -> Dict:
    n = len(display)
    hit = sum(bool(r["match"]) for r in display)
    off = [r for r in display if not r["match"]]
    return dict(cells=n, match=hit, mismatches=off)


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=100, help="Monte Carlo replications per design (paper: 500)")
    ap.add_argument("--designs", default=",".join(S.DESIGNS))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--skip-mc", action="store_true")
    a = ap.parse_args(argv)
    ref = json.loads(REFERENCE.read_text())
    res = dict(paper="arXiv:2610.05535v1", python=platform.python_version())
    res["path_a"] = path_a(ref)
    res["path_a"]["tally"] = _tally(res["path_a"]["display"])
    res["frontier_solver"] = solver_choice(ref)
    if not a.skip_mc:
        res["path_b"] = path_b(ref, a.reps, a.designs.split(","), a.procs)
        res["path_b"]["tally"] = _tally(res["path_b"]["display"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("Path A cross-validation (max |ours - reference|):")
    for k, v in res["path_a"]["cross_validation"].items():
        print(f"  {k:<62s} {v:.1e}")
    ta = res["path_a"]["tally"]
    print(f"Path A paper cells at printed precision: {ta['match']}/{ta['cells']}")
    for r in ta["mismatches"]:
        print(f"  MISMATCH {r['quantity']}: ours {r['ours']} printed {r['printed']}")
    for key, v in res["frontier_solver"].items():
        print(f"Frontier on the active-set solver, {key}: |RMSPE diff| {v['max_abs_rmspe']:.1e}, "
              f"|EI diff| {v['max_abs_ei']:.1e}, |w diff| {v['max_abs_weight']:.1e}, "
              f"{v['seconds_active_set']}s vs CLARABEL {v['seconds_clarabel']}s")
    if "path_b" in res:
        b = res["path_b"]
        print(f"Path B draws bitwise identical: {b['draws_bitwise']}; per-replication max diff {b['per_replication_maxdiff']}")
        tb = b["tally"]
        print(f"Path B paper cells at printed precision ({b['reps']} reps): {tb['match']}/{tb['cells']}")
        for r in tb["mismatches"]:
            print(f"  off {r['off_by_digits']} digit(s): {r['quantity']}: ours {r['ours']:.4f} printed {r['printed']}")
    return res


if __name__ == "__main__":
    main()
