"""RRMA, demonstrated before it is built: Path A on Basque and the ablation.

Reference: Lv, J. and Li, J. (2026), "Rank-based Model Average Estimator for
Average Treatment Effects", arXiv:2610.15553v1.

A replication spike, not an estimator and not a pinned case. Two questions:

1. Path A. Does a port that ingests through ``dataprep`` reproduce the Basque
   application (Section 6, Table 4), and how much of that number is fixed by
   the method as written?
2. The ablation the paper does not run. On its own Example 1, how much of
   RRMA's advantage comes from the rank loss and how much from averaging over
   donor groups?

Run::

    python -m benchmarks.studies.rrma_ablation.run                 # both, 500 reps (~50 min on 4 cores)
    python -m benchmarks.studies.rrma_ablation.run --only basque   # Path A (~5 min)
    python -m benchmarks.studies.rrma_ablation.run --only mc --reps 100

Path A compares against ``benchmarks/reference/rrma_basque/reference.json``,
captured from ``rrma_m2.R`` by ``capture.py`` in that directory. Results are
written to ``results/rrma_ablation.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import lp as LP
from . import oracle as O
from . import paper_values as P
from . import simulate as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PANEL = ROOT / "basedata" / "basque_data.csv"
REFERENCE = ROOT / "benchmarks" / "reference" / "rrma_basque" / "reference.json"
OUT = HERE / "results" / "rrma_ablation.json"
BASQUE_NO = 17


# ------------------------------------------------------------------- Path A


def basque(key: str = "regionno"):
    """``(y, X, T1, donor labels)`` through ``dataprep``, keyed by ``key``."""
    from mlsynth.utils.datautils import dataprep

    df = pd.read_csv(PANEL)
    df["treat"] = ((df.regionno == BASQUE_NO) & (df.year >= 1975)).astype(int)
    prep = dataprep(df, key, "year", "gdpcap", "treat")
    return prep["y"], prep["donor_matrix"], int(prep["pre_periods"]), list(prep["donor_names"])


def _att(y, cf, T1) -> float:
    return float(np.mean(y[T1:] - cf[T1:]))


def _window_mallows_P(y, X, T1, groups):
    """Equation (11) with the Mallows weights recomputed on each training window."""
    val = O.validation_window(T1)
    Pm = np.empty((len(val), len(groups)))
    for k, t in enumerate(val):
        for s, g in enumerate(groups):
            h, _ = O.mallows_or_ones(X[:t][:, g])
            a, b = O.rank_fit(y[:t], X[:t][:, g], h)
            Pm[k, s] = a + X[t, g] @ b
    return Pm


def _no_tau(y_val, Pm, rng=None):
    """Section 2.4's weights with every pair weighted 1 (tau off)."""
    return LP.simplex_rank_lad(y_val, Pm, np.ones(len(y_val))), True


def seams(y, X, T1) -> Dict:
    """The m = 2 path against the R reference, quantity by quantity."""
    ref = json.loads(REFERENCE.read_text())
    order, pvals, ties = O.screen(y[:T1], X[:T1])
    path = O.candidate_paths(y, X, T1, [2], order=order)[2]
    ch = O.choose({2: path}, y, T1, O.weigh_pairwise)
    hbar = O.mallows(path.P)
    ref_order = [g for grp in ref["groups"] for g in grp]
    mx = lambda a, b: float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    return dict(
        order_identical=[int(v) for v in order] == ref_order,
        groups_identical=[list(map(int, g)) for g in path.groups] == ref["groups"],
        ties=ties,
        h=mx(path.hs, ref["h"]),
        P=mx(path.P, ref["P"]),
        hbar=mx(hbar, ref["hbar"]),
        w=mx(ch.w, ref["w"]),
        theta=mx([np.r_[a, b] for a, b in path.thetas], ref["theta"]),
        Q=abs(ch.Q[2] - ref["Q"]),
        att_python=_att(y, ch.counterfactual, T1), att_r=ref["att"],
        kendall_p_min_r=float(min(ref["kendall_p"])),
    )


def readings(y, X, T1) -> Dict[str, Dict]:
    """The m = 2 estimate under each reading the paper leaves open."""
    order, _, _ = O.screen(y[:T1], X[:T1])
    on = O.candidate_paths(y, X, T1, [2], order=order)
    off = O.candidate_paths(y, X, T1, [2], mallows_weights=False, order=order)
    p = on[2]
    win = {2: O.Path(2, p.groups, p.hs, _window_mallows_P(y, X, T1, p.groups), p.M, p.thetas)}
    rows = {
        "as specified (h over all pre-periods, tau on)": O.choose(on, y, T1, O.weigh_pairwise),
        "tau off": O.choose(on, y, T1, _no_tau),
        "Mallows b off (plain Wilcoxon)": O.choose(off, y, T1, O.weigh_pairwise),
        "b off and tau off": O.choose(off, y, T1, _no_tau),
        "h recomputed per training window": O.choose(win, y, T1, O.weigh_pairwise),
        "h per training window, tau off": O.choose(win, y, T1, _no_tau),
        "weights on levels (RRMA_level)": O.choose(on, y, T1, O.weigh_level),
    }
    return {k: dict(att=_att(y, c.counterfactual, T1), w=np.round(c.w, 4).tolist(),
                    support=(np.flatnonzero(c.w > 1e-8) + 1).tolist()) for k, c in rows.items()}


def _seed_job(args):
    """One seed of the full selection; returns ``((key, seed, m, att), counts)``."""
    return S.counting(_seed_fit, *args)


def _seed_fit(key, seed):
    y, X, T1, _ = basque(key)
    ch, _ = O.rrma(y, X, T1, P.BASQUE_M_RANGE, rng=np.random.default_rng(seed))
    return key, seed, ch.m, _att(y, ch.counterfactual, T1)


def _perm_job(seed):
    """One donor order at m = 2; returns ``(att, counts)``."""
    return S.counting(_perm_fit, seed)


def _perm_fit(seed):
    y, X, T1, _ = basque("regionno")
    perm = np.random.default_rng(seed).permutation(X.shape[1])
    ch = O.choose(O.candidate_paths(y, X[:, perm], T1, [2]), y, T1, O.weigh_pairwise)
    return _att(y, ch.counterfactual, T1)


def _boot_job(seed, B):
    """One block-bootstrap run; the returned dict carries the solver diagnostics."""
    out, counts = S.counting(_boot_fit, seed, B)
    return dict(out, **counts)


def _boot_fit(seed, B):
    y, X, T1, _ = basque("regionno")
    path = O.candidate_paths(y, X, T1, [2])[2]
    ch = O.choose({2: path}, y, T1, O.weigh_pairwise)
    var, parts = O.rrma_variance(y, X, T1, path.groups, ch.w, np.random.default_rng(seed), B)
    # The same draws without the factor T1 in Cov(sqrt(T1) theta): a hypothesis
    # for the paper's 0.1247, not a reading of its text.
    se_no_t1 = float(np.sqrt(parts["term_theta"] / T1 + parts["term_v"]))
    return dict(seed=seed, se=float(np.sqrt(var)), se_without_T1_factor=se_no_t1, **parts)


def path_a(workers: int, seeds: int, B: int, perms: int) -> Dict:
    y, X, T1, labels = basque("regionno")
    paper = P.implied_rrma()
    out = dict(T1=T1, T2=len(y) - T1, donors=[int(v) for v in labels], paper=paper)

    order, pvals, ties = O.screen(y[:T1], X[:T1])
    groups = [sorted((g + 1).tolist()) for g in O.partition(order, 2)]
    out["screen"] = dict(order=(order + 1).tolist(), ties=ties, groups_m2=groups,
                         groups_match_paper=groups == P.BASQUE_GROUPS)
    out["seams_vs_r"], n_seams = S.counting(seams, y, X, T1)
    out["readings_m2"], n_read = S.counting(readings, y, X, T1)

    with Pool(workers) as pool:
        sweep = pool.map(_seed_job, [(k, s) for k in ("regionno", "regionname") for s in range(seeds)])
        perms_out = pool.map(_perm_job, range(perms))
        boots = pool.starmap(_boot_job, [(s, B) for s in range(5)])
    sweep, n_sweep = [r for r, _ in sweep], _total([n for _, n in sweep])
    perm_att, n_perm = [a for a, _ in perms_out], _total([n for _, n in perms_out])
    for key in ("regionno", "regionname"):
        rows = [r for r in sweep if r[0] == key]
        att = np.array([r[3] for r in rows])
        out[f"seed_sweep_{key}"] = dict(
            seeds=len(rows), m_counts=dict(sorted(Counter(r[2] for r in rows).items())),
            att_min=float(att.min()), att_q25=float(np.quantile(att, .25)),
            att_median=float(np.median(att)), att_q75=float(np.quantile(att, .75)),
            att_max=float(att.max()),
            att_by_m={int(m): [float(att[[r[2] == m for r in rows]].min()),
                               float(att[[r[2] == m for r in rows]].max())]
                      for m in sorted({r[2] for r in rows})},
            within_0005_of_paper=float(np.mean(np.abs(att - paper["att"]) < 0.005)))
    y2, X2, T12, _ = basque("regionname")
    ch2, n_name = S.counting(
        lambda: O.choose(O.candidate_paths(y2, X2, T12, [2]), y2, T12, O.weigh_pairwise))
    out["regionname_m2_att"] = _att(y2, ch2.counterfactual, T12)
    pa = np.array(perm_att)
    out["donor_order_m2"] = dict(perms=perms, att_min=float(pa.min()), att_max=float(pa.max()),
                                 att_q05=float(np.quantile(pa, .05)), att_q95=float(np.quantile(pa, .95)),
                                 distinct=int(len(np.unique(np.round(pa, 6)))))
    out["inference_m2"] = boots
    # solver diagnostics (``simulate.counting``) by part of Path A
    out["solver"] = dict(seams=n_seams, readings=n_read, seed_sweeps=n_sweep, regionname=n_name,
                         donor_orders=n_perm,
                         bootstrap=_total([{k: b[k] for k in S.COUNTED} for b in boots]))
    return out


def _total(counts: List[Dict]) -> Dict:
    return {k: int(sum(c[k] for c in counts)) for k in S.COUNTED}


# ---------------------------------------------------------------- Monte Carlo


def _mc_job(args):
    """One replication; a failure comes back as a row naming it, not as a dead pool."""
    case, rep, T1, T2, N = args
    try:
        return S.one_rep(case, T1, T2, N, rep)
    except Exception:                                    # noqa: BLE001 - reported in the results
        return dict(case=case, rep=rep, error=traceback.format_exc(limit=4))


def summarise_case(rs: List[Dict], case: str, rng) -> Dict:
    """One case's arms, paired ratios, 2x2, diagnostics and the paper's values."""
    E = {a: np.array([r["err"][a] for r in rs]) for a in S.ARMS}
    d = [r["diag"] for r in rs]
    return dict(
        arms={a: _summ(E[a]) for a in S.ARMS},
        ratio_to_rrma={a: _ratio(E[a], E["RRMA"], rng) for a in S.ARMS if a != "RRMA"},
        two_by_two=dict(
            loss_given_one=_ratio(E["RANK_one"], E["OLS_one"], rng),
            loss_given_avg=_ratio(E["RRMA"], E["LS_avg"], rng),
            avg_given_rank=_ratio(E["RRMA"], E["RANK_one"], rng),
            avg_given_ls=_ratio(E["LS_avg"], E["OLS_one"], rng)),
        diag=dict(
            d_mean=float(np.mean([x["d"] for x in d])),
            share_with_ties=float(np.mean([x["ties"] > 0 for x in d])),
            m_rrma=dict(sorted(Counter(x["m_rrma"] for x in d).items())),
            strong_in_screen_mean=float(np.mean([x["strong_in_screen"] for x in d])),
            b_fallbacks=int(sum(x["b_fallbacks"] for x in d)),
            tau_fallbacks=int(sum(x["tau_fallbacks"] for x in d)),
            rank_one_mallows_failed=int(sum(not x["rank_one_mallows"] for x in d)),
            uncertified=int(sum(x["uncertified"] for x in d)),
            reps_with_uncertified=int(sum(x["uncertified"] > 0 for x in d)),
            unidentified=int(sum(x["unidentified"] for x in d)),
            reps_used=len(rs),
            rrma_seconds_mean=float(np.mean([r["sec"]["RRMA"] for r in rs]))),
        paper={arm: [vals[0], vals[1]] for arm, vals in P.TABLE1_T60[case].items()},
    )


def _summ(e: np.ndarray) -> Dict:
    sq = e * e
    return dict(bias=float(e.mean()), mae=float(np.abs(e).mean()), mse100=float(100 * sq.mean()),
                mse100_se=float(100 * sq.std(ddof=1) / np.sqrt(len(e))))


def _ratio(num: np.ndarray, den: np.ndarray, rng, B: int = 2000):
    """Paired MSE ratio with a percentile bootstrap over replications."""
    n = len(num)
    idx = rng.integers(0, n, size=(B, n))
    boots = (num[idx] ** 2).mean(axis=1) / (den[idx] ** 2).mean(axis=1)
    return [float((num ** 2).mean() / (den ** 2).mean()),
            float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def monte_carlo(workers: int, reps: int, T1: int = 60, T2: int = 30, N: int = 25) -> Dict:
    jobs = [(c, r, T1, T2, N) for c in S.CASES for r in range(reps)]
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        rows = pool.map(_mc_job, jobs, chunksize=1)
    wall = time.perf_counter() - t0
    errors = [dict(case=r["case"], rep=r["rep"], error=r["error"]) for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    rng = np.random.default_rng(20261018)
    out = dict(T1=T1, T2=T2, N=N, reps=reps, wall_seconds=wall, errors=errors, cases={})
    for case in S.CASES:
        out["cases"][case] = summarise_case([r for r in rows if r["case"] == case], case, rng)
    return out


# ---------------------------------------------------------------------- output


def _print(res: Dict) -> None:
    if "path_a" in res:
        a = res["path_a"]
        print("== Path A: Basque, keyed by region number")
        print(f"   screen order {a['screen']['order']}  ties {a['screen']['ties']}  "
              f"m=2 groups match paper: {a['screen']['groups_match_paper']}")
        s = a["seams_vs_r"]
        print(f"   vs R: order {s['order_identical']} groups {s['groups_identical']}  h {s['h']:.1e}  "
              f"P {s['P']:.1e}  hbar {s['hbar']:.1e}  w {s['w']:.1e}  theta {s['theta']:.1e}  "
              f"att {s['att_python']:.7f} / R {s['att_r']:.7f}")
        print(f"   paper implies att {a['paper']['att']:.4f}, se {a['paper']['se']:.4f}")
        for k, v in a["readings_m2"].items():
            print(f"   m=2 {k:48s} att {v['att']:+.4f} support {v['support']}")
        for key in ("regionno", "regionname"):
            sw = a[f"seed_sweep_{key}"]
            print(f"   seeds ({key}): m {sw['m_counts']}  att {sw['att_min']:+.3f} .. {sw['att_max']:+.3f} "
                  f"(median {sw['att_median']:+.4f})  within 0.005 of paper {sw['within_0005_of_paper']:.2f}")
        print(f"   m=2 keyed by region name: att {a['regionname_m2_att']:+.4f}")
        print(f"   solver diagnostics: {a.get('solver')}")
        do = a["donor_order_m2"]
        print(f"   m=2 under {do['perms']} donor orders: att {do['att_min']:+.4f} .. {do['att_max']:+.4f} "
              f"(5-95%: {do['att_q05']:+.4f} .. {do['att_q95']:+.4f}; {do['distinct']} distinct)")
        for b in a["inference_m2"]:
            print(f"   MBB seed {b['seed']}: se {b['se']:.4f}, without T1 {b.get('se_without_T1_factor', float('nan')):.4f} "
                  f"(theta term {b['term_theta']:.5f}, "
                  f"v term {b['term_v']:.5f}; Mallows fallbacks {b['boot_mallows_fallbacks']})")
    if "mc" in res:
        mc = res["mc"]
        print(f"== Monte Carlo: Example 1, T1={mc['T1']} T2={mc['T2']} N={mc['N']}, {mc['reps']} reps "
              f"({mc['wall_seconds'] / 60:.1f} min), {len(mc.get('errors', []))} failed replications")
        for err in mc.get("errors", []):
            print(f"   FAILED case {err['case']} rep {err['rep']}: {err['error'].strip().splitlines()[-1]}")
        for case, c in mc["cases"].items():
            print(f"-- case {case}: {c['diag'].get('reps_used')} reps, d {c['diag']['d_mean']:.1f}, "
                  f"m_rrma {c['diag']['m_rrma']}, ties in {c['diag']['share_with_ties']:.0%}, "
                  f"uncertified solves {c['diag'].get('uncertified')} "
                  f"(in {c['diag'].get('reps_with_uncertified')} reps), "
                  f"unidentified fits {c['diag'].get('unidentified')}")
            for arm in S.ARMS:
                v = c["arms"][arm]
                r = c["ratio_to_rrma"].get(arm)
                rs = f"  ratio {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]" if r else ""
                pp = c["paper"].get(arm)
                ps = f"  paper {pp[0][2]:.2f} / {pp[1][2]:.2f}" if pp else ""
                print(f"   {arm:11s} bias {v['bias']:+.3f} MAE {v['mae']:.3f} "
                      f"MSE {v['mse100']:6.2f} ({v['mse100_se']:.2f}){rs}{ps}")
            tt = c["two_by_two"]
            print("   2x2 MSE ratios: " + "  ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in tt.items()))


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=("basque", "mc"))
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.update(paper="arXiv:2610.15553v1", python=platform.python_version())
    def checkpoint():
        OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")

    if args.only in (None, "basque"):
        res["path_a"] = path_a(args.workers, args.seeds, args.boot, args.perms)
        checkpoint()
    if args.only in (None, "mc"):
        res["mc"] = monte_carlo(args.workers, args.reps)
        checkpoint()
    _print(res)
    return res


if __name__ == "__main__":
    main()
