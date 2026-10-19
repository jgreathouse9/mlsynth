"""Capture R's values for the RRMA study's distribution functions.

``special.R`` beside this file evaluates ``qnorm``, ``qchisq`` and ``pchisq`` on
fixed grids, including every ``qchisq(qn / n, p)`` that ``MASS::cov.rob`` asks
for at the study's sizes, and ``cor.test(method = "kendall")`` on input vectors
this script writes: random draws on both sides of R's exact/normal switch at
n = 50, rounded draws with ties, and the Basque panel's treated unit against
each donor over 1955-1974.

Run::

    python benchmarks/reference/rrma_special/capture.py

It writes ``reference.json``, ``reference.out`` and ``provenance.json`` next to
this file.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = HERE / "special.R"
PANEL = ROOT / "basedata" / "basque_data.csv"


def _cases():
    rng = np.random.default_rng(20261018)
    cases = {}
    for n in (5, 8, 10, 15, 20, 30, 45, 49, 50, 60, 120):
        for k in range(3):
            x = rng.normal(size=n)
            y = 0.4 * x + rng.normal(size=n) if k else rng.normal(size=n)
            cases[f"rand_n{n:03d}_{k}"] = (x, y)
    for n in (10, 30, 60):
        for k in range(2):
            cases[f"ties_n{n:03d}_{k}"] = (np.round(rng.normal(size=n), 1), np.round(rng.normal(size=n), 1))
    df = pd.read_csv(PANEL)
    W = df.pivot(index="year", columns="regionno", values="gdpcap").sort_index().loc[:1974]
    for c in W.columns.drop(17):
        cases[f"basque_{c:02d}"] = (W[17].to_numpy(), W[c].to_numpy())
    return cases


def main() -> None:
    cases = _cases()
    with tempfile.TemporaryDirectory() as tmp:
        for name, (x, y) in cases.items():
            np.savetxt(Path(tmp) / f"kendall_{name}.tsv", np.column_stack([x, y]),
                       delimiter="\t", fmt="%.17g")
        res = subprocess.run(["Rscript", str(SCRIPT), tmp], capture_output=True, text=True, check=True)
    (HERE / "reference.out").write_text(res.stdout)
    ref = {"qnorm": [], "qchisq": [], "pchisq": [], "kendall": []}
    for line in res.stdout.splitlines():
        kind, *vals = line.split("\t")
        if kind == "qnorm":
            ref["qnorm"].append([float(vals[0]), float(vals[1])])
        elif kind == "qchisq":
            ref["qchisq"].append([float(v) for v in vals])
        elif kind == "pchisq":
            ref["pchisq"].append([float(v) for v in vals])
        elif kind == "kendall":
            name = vals[0][len("kendall_"):-len(".tsv")]
            x, y = cases[name]
            ref["kendall"].append(dict(name=name, x=x.tolist(), y=y.tolist(), p=float(vals[1]),
                                       method="exact" if vals[2] == "T" else "normal"))
        elif kind == "versions":
            ref["r"] = vals[0]
    (HERE / "reference.json").write_text(json.dumps(ref) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    prov = dict(generated_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                git_sha=git, platform=platform.platform(), r=ref.get("r"),
                command="python benchmarks/reference/rrma_special/capture.py",
                files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in (SCRIPT, PANEL)])
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print({k: len(v) for k, v in ref.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
