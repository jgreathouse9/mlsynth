"""MMSCM against the authors' own Python, on the three panels of their paper.

Reference: Masahiro Kato and Akari Ohda (2026), "Asymptotically unbiased
synthetic control methods by moment matching", *Journal of Causal Inference*
14(1):20250008, DOI 10.1515/jci-2025-0008. Section 5.3 is the estimator,
Theorem 5.10 its convergence, Section 7.2 the three empirical applications.
Reference implementation: https://github.com/MasaKat0/mmscm (`mmscm.py`).

This is a replication spike, not an estimator and not a pinned case. It answers
three questions before any build starts:

1. does a port that ingests through ``dataprep`` reproduce the authors'
   arithmetic cell for cell?
2. is the Abadie baseline the paper compares against actually solved?
3. are the moment-matched weights identified?

The answers are yes, no, and only sometimes. See README.md.

Run::

    python -m benchmarks.studies.mmscm_replicate.run
    python -m benchmarks.studies.mmscm_replicate.run --covariates
    python -m benchmarks.studies.mmscm_replicate.run --starts 40
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.optimize

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference.clone_mmscm import import_mmscm
from mlsynth.utils.datautils import dataprep

from . import port

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "basedata"

#: Published synthetic-control weights, for the baseline comparison. Each comes
#: from a covariate-matched fit, so an outcome-only solve is not expected to
#: reproduce them exactly; they anchor which donors should carry the weight.
PUBLISHED = {
    "prop99": "Utah .334, Nevada .234, Montana .199, Colorado .164, "
              "Connecticut .069 (Abadie, Diamond & Hainmueller 2010)",
    "basque": "Catalonia .851, Madrid .149 (Abadie & Gardeazabal 2003)",
    "germany": "Austria .42, USA .22, Japan .16, Switzerland .11, "
               "Netherlands .09 (Abadie, Diamond & Hainmueller 2015)",
}

# Basque sector shares are observed in five odd years only (85 of 731 rows), and
# the authors' _data_setup mean-fills them, so they are left out of the covariate
# set here; --sector-block puts them back to show the fill changes little.
BASQUE_COVARIATES = ["school.illit", "school.prim", "school.med", "school.high",
                     "popdens", "invest"]
BASQUE_SECTORS = ["sec.agriculture", "sec.energy", "sec.industry",
                  "sec.construction", "sec.services.venta",
                  "sec.services.nonventa"]


def _panel(name: str, covariates: bool, sector_block: bool):
    """(frame, unit, time, outcome, treated, last_pre_period, covariate list).

    ``frame`` carries the outcome first: ``train_param`` raises
    ``treated_final[0]`` to powers, so whichever numeric column comes first is
    the one the polynomial moments are built from.
    """
    if name == "prop99":
        src = "california_panel.csv" if covariates else "smoking_data.csv"
        d = pd.read_csv(BASE / src)
        cov = ["retprice"] if covariates else []
        return d, "state", "year", "cigsale", "California", 1988, cov
    if name == "basque":
        d = pd.read_csv(BASE / "basque_data.csv")
        treated = next(r for r in d.regionname.unique() if "Vasco" in r)
        cov = (BASQUE_COVARIATES + (BASQUE_SECTORS if sector_block else [])
               ) if covariates else []
        return d, "regionname", "year", "gdpcap", treated, 1969, cov
    if name == "germany":
        src = "germany_augmented.csv" if covariates else "german_reunification.csv"
        d = pd.read_csv(BASE / src)
        cov = ["infrate", "trade", "industry"] if covariates else []
        return d, "country", "year", "gdp", "West Germany", 1989, cov
    raise SystemExit(f"unknown panel {name!r}")


def _inputs(name: str, covariates: bool, sector_block: bool):
    """Ingest through ``dataprep``, which is the only pandas touchpoint."""
    raw, uid, tid, out, treated, last_pre, cov = _panel(name, covariates, sector_block)
    d = raw.copy()
    d["D"] = ((d[uid] == treated) & (d[tid] > last_pre)).astype(int)
    prep = dataprep(d[[uid, tid, out, "D"]], uid, tid, out, "D")
    frame = d[[uid, tid, out] + cov].copy()
    return dict(
        prep=prep, frame=frame, uid=uid, tid=tid, out=out, treated=treated,
        last_pre=last_pre, n_cov=len(cov),
        T0=int(prep["pre_periods"]),
        y=np.asarray(prep["y"], float).ravel(),
        donors=np.asarray(prep["donor_matrix"], float).T,
        names=list(prep["donor_names"]),
        # dataprep sorts donors; the authors keep DataFrame order. The two
        # coincide for prop99 and germany and differ for basque, so every
        # weight vector crossing that boundary is reindexed through `reidx`.
        reidx=[[s for s in d[uid].unique() if s != treated].index(s)
               for s in prep["donor_names"]],
    )


def _raw_features(inp: Dict[str, Any], which: str) -> np.ndarray:
    """The pre-period feature block, outcome row first, from the ingested frame.

    ``dataprep`` carries the outcome; the covariate columns are read off the
    same frame and stacked beneath it, which is the layout ``_data_setup``
    builds and ``build_moments`` expects.
    """
    f, uid, tid, out = inp["frame"], inp["uid"], inp["tid"], inp["out"]
    cols = [out] + [c for c in f.columns if c not in (uid, tid, out)]
    T0, treated = inp["T0"], inp["treated"]
    pre = f[f[tid] <= inp["last_pre"]]
    num = pre[cols].apply(pd.to_numeric, errors="coerce")
    pre = pre.assign(**{c: num[c].fillna(num[c].mean()) for c in cols})
    if which == "treated":
        return pre[pre[uid] == treated][cols].to_numpy(float).T          # (K0, T0)
    return np.stack([pre[pre[uid] == u][cols].to_numpy(float).T
                     for u in inp["names"]], axis=0)                      # (J,K0,T0)


def _effects(w: np.ndarray, inp: Dict[str, Any]):
    y, D, T0 = inp["y"], inp["donors"], inp["T0"]
    tp, dp = y[:T0], D[:, :T0]
    cf = port.counterfactual(w, D) + port.bias_correction(w, tp, dp)
    return (float(np.sqrt(np.mean((y[:T0] - cf[:T0]) ** 2))),
            float(np.mean(y[T0:] - cf[T0:])))


def _authors_fit(inp: Dict[str, Any], method: str, poly: int = 2):
    """Run the authors' own class on the same frame.

    The reference is GPL-3.0 and cloned on demand, never vendored; see
    :mod:`benchmarks.reference.clone_mmscm`.
    """
    m = import_mmscm().MMSCM(
        inp["frame"], method, inp["uid"], inp["treated"], inp["out"],
        inp["tid"], inp["last_pre"], poly=poly)
    m.train_param()
    return m


def _simplex_solve(obj, J: int, w0: np.ndarray, ftol: float):
    return scipy.optimize.minimize(
        obj, w0, method="SLSQP", bounds=[(0.0, 1.0)] * J,
        constraints=({"type": "eq", "fun": lambda x: 1 - np.sum(x)},),
        options={"ftol": ftol, "maxiter": 2000})


def study(name: str, covariates: bool, sector_block: bool, starts: int,
          polys: List[int]) -> Dict[str, Any]:
    inp = _inputs(name, covariates, sector_block)
    J = inp["donors"].shape[0]
    T0, y, D = inp["T0"], inp["y"], inp["donors"]
    tp, dp = y[:T0], D[:, :T0]
    rec: Dict[str, Any] = {
        "panel": name, "donors": J, "pre_periods": T0,
        "covariates": inp["n_cov"], "published_weights": PUBLISHED[name],
    }

    # 1. does the port reproduce the authors' arithmetic?
    agree_w = agree_cf = 0.0
    for poly in polys:
        m = _authors_fit(inp, "MMSCM", poly=poly)
        # the reference's own scaled feature block, so the port sees the same
        # inputs: (K0, T) treated and (J, K0, T) donors, outcome row first.
        w_p, _, _ = port.fit(_raw_features(inp, "treated"),
                             _raw_features(inp, "donors"), poly=poly)
        _, cf_t = m.predict(bias=True)
        w_t = np.asarray(m.res.x, float)[inp["reidx"]]
        cf_p = port.counterfactual(w_p, D) + port.bias_correction(w_p, tp, dp)
        agree_w = max(agree_w, float(np.abs(w_p - w_t).max()))
        agree_cf = max(agree_cf, float(np.abs(cf_p - np.asarray(cf_t, float)).max()))
    rec["port_vs_reference"] = {"max_abs_weight_diff": agree_w,
                                "max_abs_counterfactual_diff": agree_cf}

    # 2. is the Abadie baseline the paper compares against actually solved?
    ab = _authors_fit(inp, "Abadie")
    tm = np.asarray(ab.treated_final, float)
    dm = np.stack([np.asarray(u, float).T for u in ab.untreated_final_list], axis=1)
    sse = lambda w: float(((tm - np.einsum("j,kjt->kt", w, dm)) ** 2).mean())
    w_them = np.asarray(ab.res.x, float)
    conv = _simplex_solve(sse, J, np.ones(J) / J, 1e-14)
    rm_t, att_t = _effects(w_them[inp["reidx"]], inp)
    rm_c, att_c = _effects(np.asarray(conv.x)[inp["reidx"]], inp)
    top = sorted(zip(np.asarray(conv.x)[inp["reidx"]], inp["names"]), reverse=True)[:5]
    rec["abadie_baseline"] = {
        "authors": {"iterations": int(ab.res.nit), "objective": sse(w_them),
                    "pre_rmse": rm_t, "att": att_t},
        "converged": {"iterations": int(conv.nit), "objective": sse(conv.x),
                      "pre_rmse": rm_c, "att": att_c,
                      "top_weights": {n: round(float(v), 4) for v, n in top}},
        "att_error_from_premature_stop": att_t - att_c,
        "att_error_as_share_of_converged": (att_t - att_c) / att_c if att_c else None,
    }

    # 3. are the moment-matched weights identified?
    ident = []
    for poly in polys:
        m = _authors_fit(inp, "MMSCM", poly=poly)
        tmm = np.asarray(m.treated_final, float)
        dmm = np.stack([np.asarray(u, float).T for u in m.untreated_final_list], axis=1)
        K = tmm.shape[0]
        v = np.ones(K) / K
        obj = lambda w: port.mmscm_loss(w, tmm, dmm, v)
        M = dmm.mean(axis=2)
        sv = np.linalg.svd(M, compute_uv=False)
        rank = int(np.linalg.matrix_rank(np.vstack([M, np.ones(J)]), tol=1e-10))
        rng = np.random.default_rng(0)
        atts = []
        for _ in range(starts):
            r = _simplex_solve(obj, J, rng.dirichlet(np.ones(J)), 1e-12)
            if r.success:
                atts.append(_effects(np.asarray(r.x)[inp["reidx"]], inp)[1])
        atts = np.asarray(atts)
        w_point = np.asarray(m.res.x, float)[inp["reidx"]]
        rm, at = _effects(w_point, inp)
        ident.append({
            "poly": poly, "moments": K, "rank": rank, "null_dim": J - rank,
            "condition_number": float(sv[0] / sv[sv > 1e-14][-1])
            if (sv > 1e-14).any() else None,
            "pre_rmse": rm, "att": at,
            "att_spread_over_starts": float(atts.max() - atts.min()) if atts.size else None,
        })
    rec["identification"] = ident
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", nargs="+", default=["prop99", "basque", "germany"])
    ap.add_argument("--covariates", action="store_true",
                    help="match the covariate series too, as the paper does")
    ap.add_argument("--sector-block", action="store_true",
                    help="Basque only: add the six sector shares, 88%% mean-filled")
    ap.add_argument("--starts", type=int, default=25,
                    help="random simplex starts for the identification probe")
    ap.add_argument("--polys", nargs="+", type=int, default=[2, 6, 11])
    ap.add_argument("--out", default=str(HERE / "results" / "mmscm_replicate.json"))
    args = ap.parse_args()

    out: Dict[str, Any] = {
        "reference": "Kato & Ohda (2026), 10.1515/jci-2025-0008",
        "reference_code": "https://github.com/MasaKat0/mmscm",
        "covariates": args.covariates, "sector_block": args.sector_block,
        "starts": args.starts, "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "panels": [],
    }
    for name in args.panels:
        rec = study(name, args.covariates, args.sector_block, args.starts, args.polys)
        out["panels"].append(rec)
        ab = rec["abadie_baseline"]
        print(f"\n{name}:  J={rec['donors']}  T0={rec['pre_periods']}  "
              f"covariates={rec['covariates']}")
        print(f"  port vs reference      max|dw| {rec['port_vs_reference']['max_abs_weight_diff']:.1e}"
              f"   max|dcf| {rec['port_vs_reference']['max_abs_counterfactual_diff']:.1e}")
        print(f"  Abadie, authors        nit {ab['authors']['iterations']:>3}  "
              f"obj {ab['authors']['objective']:.3e}  pre-RMSE {ab['authors']['pre_rmse']:9.4f}  "
              f"ATT {ab['authors']['att']:+11.4f}")
        print(f"  Abadie, converged      nit {ab['converged']['iterations']:>3}  "
              f"obj {ab['converged']['objective']:.3e}  pre-RMSE {ab['converged']['pre_rmse']:9.4f}  "
              f"ATT {ab['converged']['att']:+11.4f}")
        share = ab["att_error_as_share_of_converged"]
        print(f"  -> baseline error {ab['att_error_from_premature_stop']:+.4f}"
              + (f"  ({abs(share)*100:.0f}% of the converged ATT)" if share else ""))
        print(f"     converged weights   {ab['converged']['top_weights']}")
        print(f"     published           {rec['published_weights']}")
        for row in rec["identification"]:
            print(f"  MMSCM poly={row['poly']:<3} K={row['moments']:<3} "
                  f"null_dim={row['null_dim']:<3} cond={row['condition_number']:.1e}  "
                  f"pre-RMSE {row['pre_rmse']:9.4f}  ATT {row['att']:+11.4f}  "
                  f"spread {row['att_spread_over_starts']:.3e}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
