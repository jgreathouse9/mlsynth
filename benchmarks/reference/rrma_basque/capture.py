"""Capture the R reference run of RRMA on the Basque panel at m = 2.

Lv, J. and Li, J. (2026), "Rank-based Model Average Estimator for Average
Treatment Effects", arXiv:2610.15553v1. The authors released no code, so the
reference is ``rrma_m2.R`` beside this file: the m = 2 path of Algorithm 1
written in R with the functions the paper names (``cor.test``,
``MASS::cov.rob``, ``quantreg::rq``) and ``lpSolve`` for Section 2.4's linear
programme. See ``agents/agents_r_environment.md`` for installing R here; the
packages come from apt (``r-cran-mass``, ``r-cran-quantreg``,
``r-cran-lpsolve``).

Run::

    python benchmarks/reference/rrma_basque/capture.py

It writes ``reference.out`` (what R printed), ``reference.json`` (the same
values parsed) and ``provenance.json`` next to this file.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = HERE / "rrma_m2.R"
PANEL = ROOT / "basedata" / "basque_data.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        key, *vals = [v for v in line.rstrip("\t ").split("\t") if v != ""]
        vals = [v.strip() for v in vals]
        try:
            out[key] = [float(v) for v in vals]
        except ValueError:
            out[key] = vals
    for key in [k for k in out if k.startswith("group")]:
        out[key] = [int(v) - 1 for v in out[key]]            # R's 1-based columns
    S = len([k for k in out if k.startswith("group")])
    out["P"] = [out.pop(f"P{k}") for k in range(1, 1 + len([k for k in out if k[:1] == "P" and k[1:].isdigit()]))]
    out["h"] = [out.pop(f"h{s}") for s in range(1, S + 1)]
    out["theta"] = [out.pop(f"theta{s}") for s in range(1, S + 1)]
    out["groups"] = [out.pop(f"group{s}") for s in range(1, S + 1)]
    for key in ("att", "Q", "lp_status"):
        out[key] = out[key][0]
    return out


def main() -> None:
    res = subprocess.run(["Rscript", str(SCRIPT), str(PANEL)], capture_output=True,
                         text=True, check=True)
    (HERE / "reference.out").write_text(res.stdout)
    ref = _parse(res.stdout)
    (HERE / "reference.json").write_text(json.dumps(ref, indent=1) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    prov = dict(
        generated_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=git, platform=platform.platform(),
        r=ref["versions"][0], mass=ref["versions"][1], quantreg=ref["versions"][2],
        lpsolve=ref["versions"][3],
        command="python benchmarks/reference/rrma_basque/capture.py",
        files=[dict(path=str(p.relative_to(ROOT)), sha256=_sha(p)) for p in (SCRIPT, PANEL)],
    )
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"att {ref['att']:.10f}  w {ref['w']}")


if __name__ == "__main__":
    main()
