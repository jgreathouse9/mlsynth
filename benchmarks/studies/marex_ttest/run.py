"""Size, power and interval length of the debiased t-test on MAREX designs.

    python -m benchmarks.studies.marex_ttest.run [draws]
    python -m benchmarks.studies.marex_ttest.run report

The draws are the ``marex_global_test_dependence`` study's (same regimes, same
``T0`` of 104 and 156 weeks, 12 post weeks, same seeds), so its ``shipped``
(eq. (17)) and ``cyclic_signed`` arms reappear here number for number beside
the t-test arms of ``ttest.ARMS``. Each draw solves the design once; an effect
``tau`` moves every fold estimate by ``tau`` and leaves the standard error, so
the t-test's p-values at every effect come from one fit
(``ttest.shifted_pvalue``). Rows go to ``results/runs.jsonl``; ``report``
writes ``results/summary.json`` and prints the tables.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import t as _t

from benchmarks.studies.marex_global_test_dependence import globaltest as gt
from benchmarks.studies.marex_global_test_dependence import panels

from . import ttest as tt

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent / "results"
H, T0S, TAUS, ALPHA = 12, (104, 156), (0.0, 0.5, 1.0, 2.0), 0.05


def one(task):
    regime, T0, draw = task
    rng = np.random.default_rng([panels.REGIMES.index(regime), T0, draw])
    Y = panels.panel(T0 + H, regime, rng)
    d = panels.design(Y, T0, H)
    row = dict(regime=regime, T0=T0, draw=draw, blank=d.blank)
    treated = d.w > 0
    for tau in TAUS:
        Yt = Y.copy()
        Yt[T0:, treated] += tau
        b, post = panels.contrast(Yt, d, H)
        row[f"shipped_{tau}"] = panels.shipped_pvalue(Yt, d, H)
        row[f"cyclic_signed_{tau}"] = gt.global_pvalue(b, post, "cyclic", stat="signed")
    for arm, res in tt.arms(Y, d, H).items():
        row[f"{arm}_att"], row[f"{arm}_se"] = res.att, res.se
    return row


def main(draws="2000"):
    from multiprocessing import Pool
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "runs.jsonl"
    done = set()
    if path.exists():
        done = {(r["regime"], r["T0"], r["draw"]) for r in map(json.loads, path.read_text().splitlines())}
    tasks = [(g, T0, i) for g in panels.REGIMES for T0 in T0S for i in range(int(draws))
             if (g, T0, i) not in done]
    print(f"{len(tasks)} draws to run, {len(done)} done", flush=True)
    with Pool(4) as pool, path.open("a") as fh:
        for k, row in enumerate(pool.imap_unordered(one, tasks, chunksize=8), 1):
            fh.write(json.dumps(row) + "\n")
            if k % 500 == 0:
                fh.flush()
                print(f"{k}/{len(tasks)}", flush=True)


def _arm_pvalues(rs, arm, tau):
    K = int(arm.rsplit("_", 1)[1])
    return np.array([tt.shifted_pvalue(tt.TTest(r[f"{arm}_att"], r[f"{arm}_se"],
                                                tt._tstat(r[f"{arm}_att"], r[f"{arm}_se"]),
                                                np.empty(0), 0, K), tau) for r in rs])


def _pvalues(rs, arm, tau):
    if arm in ("shipped", "cyclic_signed"):
        return np.array([r[f"{arm}_{tau}"] for r in rs])
    return _arm_pvalues(rs, arm, tau)


def size_adjusted_power(p_null, p_alt, alpha=ALPHA):
    """Rejections at the largest cutoff that rejects at most ``alpha`` of the
    null draws; NaN when every cutoff rejects more, as a discrete p-value whose
    smallest value is too common cannot be held to ``alpha``."""
    p_null, p_alt = np.asarray(p_null), np.asarray(p_alt)
    ok = [q for q in np.unique(p_null) if np.mean(p_null <= q) <= alpha]
    return float(np.mean(p_alt <= max(ok))) if ok else float("nan")


def report():
    rows = [json.loads(line) for line in (RESULTS / "runs.jsonl").read_text().splitlines()]
    out = []
    for regime in panels.REGIMES:
        for T0 in T0S:
            rs = [r for r in rows if r["regime"] == regime and r["T0"] == T0]
            if not rs:
                continue
            cell = dict(regime=regime, T0=T0, draws=len(rs), blank=rs[0]["blank"])
            for arm in ("shipped", "cyclic_signed"):
                for tau in TAUS:
                    cell[f"{arm}_{tau}"] = float(np.mean([r[f"{arm}_{tau}"] <= ALPHA for r in rs]))
            for arm in tt.ARMS:
                K = int(arm.rsplit("_", 1)[1])
                for tau in TAUS:
                    cell[f"{arm}_{tau}"] = float(np.mean(_arm_pvalues(rs, arm, tau) <= ALPHA))
                se = np.array([r[f"{arm}_se"] for r in rs])
                cell[f"{arm}_halfwidth"] = float(np.median(_t.ppf(1 - ALPHA / 2, K - 1) * se))
                cell[f"{arm}_bias"] = float(np.mean([r[f"{arm}_att"] for r in rs]))
            for arm in ("shipped", "cyclic_signed") + tt.ARMS:
                for tau in TAUS[1:]:
                    cell[f"{arm}_{tau}_adjusted"] = size_adjusted_power(
                        _pvalues(rs, arm, 0.0), _pvalues(rs, arm, tau))
            out.append(cell)
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    arms = ("shipped", "cyclic_signed") + tt.ARMS
    print("| Regime | T0 | blank | size 5%: " + " / ".join(arms) + " | power tau=1 | tau=2 |")
    print("|---|---|---|---|---|---|")
    for c in out:
        f = lambda tau: " / ".join(f"{100 * c[f'{a}_{tau}']:.1f}" for a in arms)  # noqa: E731
        print(f"| {c['regime']} | {c['T0']} | {c['blank']} | {f(0.0)} | {f(1.0)} | {f(2.0)} |")
    print("\n| Regime | T0 | size-adjusted power tau=1: " + " / ".join(arms) + " | tau=2 |")
    print("|---|---|---|---|")
    for c in out:
        g = lambda tau: " / ".join(  # noqa: E731
            "n/a" if np.isnan(c[f"{a}_{tau}_adjusted"]) else f"{100 * c[f'{a}_{tau}_adjusted']:.1f}"
            for a in arms)
        print(f"| {c['regime']} | {c['T0']} | {g(1.0)} | {g(2.0)} |")
    print("\n| Regime | T0 | median 95% half-width: " + " / ".join(tt.ARMS)
          + " | mean estimate at no effect |")
    print("|---|---|---|---|")
    for c in out:
        hw = " / ".join(f"{c[f'{a}_halfwidth']:.2f}" for a in tt.ARMS)
        bias = " / ".join(f"{c[f'{a}_bias']:+.3f}" for a in tt.ARMS)
        print(f"| {c['regime']} | {c['T0']} | {hw} | {bias} |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main(*sys.argv[1:])
