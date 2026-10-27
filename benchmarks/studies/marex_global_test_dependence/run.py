"""Size and power of MAREX's global test under serially dependent factors.

    python -m benchmarks.studies.marex_global_test_dependence.run [draws]
    python -m benchmarks.studies.marex_global_test_dependence.run report

Each draw takes one panel (``panels.panel``) of ``T0 + 12`` periods, solves the
MAREX design on its pre-period, adds a constant effect ``tau`` to the treated
units in the post periods (``tau`` in units of the errors' standard deviation,
so the contrast moves by ``tau``), and computes the global p-value three ways:
``shipped`` (MAREX's eq. (17) through ``compute_inference``), ``cyclic`` and
``cyclic_w`` (cyclic shifts weighted ``0.97`` to the distance from the post
window), ``windows`` (the post block against the windows of consecutive
blank periods), and ``subsets_signed`` and ``cyclic_signed``, the subsets and the
cyclic shifts with the absolute value of the mean contrast as the statistic, and
``cyclic_signed_w``, the latter with the recency weights. ``T0`` is 104 or 156 weeks, two and three years; the blank window is
MAREX's default, 31 and 46 periods. Rows go to ``results/runs.jsonl``; ``report``
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

from . import globaltest as gt
from . import panels

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent / "results"
H = 12
T0S = (104, 156)
TAUS = (0.0, 0.5, 1.0, 2.0)
RHO = 0.97
ALPHAS = (0.05, 0.10)
ARMS = ("shipped", "cyclic", "cyclic_w", "windows", "subsets_signed", "cyclic_signed",
        "cyclic_signed_w")


def one(task):
    regime, T0, draw = task
    rng = np.random.default_rng([panels.REGIMES.index(regime), T0, draw])
    Y = panels.panel(T0 + H, regime, rng)
    d = panels.design(Y, T0, H)
    row = dict(regime=regime, T0=T0, draw=draw, blank=d.blank)
    blank, _ = panels.contrast(Y, d, H)
    row["lag1"] = float(np.corrcoef(blank[:-1], blank[1:])[0, 1])
    treated = d.w > 0
    for tau in TAUS:
        Yt = Y.copy()
        Yt[T0:, treated] += tau
        b, post = panels.contrast(Yt, d, H)
        row[f"shipped_{tau}"] = panels.shipped_pvalue(Yt, d, H)
        row[f"cyclic_{tau}"] = gt.global_pvalue(b, post, "cyclic")
        row[f"cyclic_w_{tau}"] = gt.global_pvalue(b, post, "cyclic", rho=RHO)
        row[f"windows_{tau}"] = gt.global_pvalue(b, post, "windows")
        row[f"cyclic_signed_{tau}"] = gt.global_pvalue(b, post, "cyclic", stat="signed")
        row[f"cyclic_signed_w_{tau}"] = gt.global_pvalue(b, post, "cyclic", stat="signed", rho=RHO)
        row[f"subsets_signed_{tau}"] = gt.global_pvalue(b, post, "subsets", stat="signed",
                                                        rng=np.random.default_rng(42))
    return row


def main(draws="1000"):
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


def report():
    rows = [json.loads(line) for line in (RESULTS / "runs.jsonl").read_text().splitlines()]
    out = []
    for regime in panels.REGIMES:
        for T0 in T0S:
            rs = [r for r in rows if r["regime"] == regime and r["T0"] == T0]
            if not rs:
                continue
            cell = dict(regime=regime, T0=T0, draws=len(rs), blank=rs[0]["blank"],
                        lag1=float(np.mean([r["lag1"] for r in rs])))
            for arm in ARMS:
                for tau in TAUS:
                    p = np.array([r[f"{arm}_{tau}"] for r in rs])
                    for a in ALPHAS:
                        cell[f"{arm}_{tau}_{a}"] = float(np.mean(p <= a))
            out.append(cell)
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print("| Regime | T0 | blank | lag-1 | size 5%: " + " / ".join(ARMS) + " "
          "| size 10% | power tau=0.5 (5%) | tau=1 | tau=2 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in out:
        f = lambda arm, tau, a: f"{100 * c[f'{arm}_{tau}_{a}']:.1f}"  # noqa: E731
        trio = lambda tau, a: " / ".join(f(arm, tau, a) for arm in ARMS)  # noqa: E731
        print(f"| {c['regime']} | {c['T0']} | {c['blank']} | {c['lag1']:.2f} | {trio(0.0, 0.05)} | "
              f"{trio(0.0, 0.1)} | {trio(0.5, 0.05)} | {trio(1.0, 0.05)} | {trio(2.0, 0.05)} |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main(*sys.argv[1:])
