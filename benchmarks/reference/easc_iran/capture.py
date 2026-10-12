"""Capture the reference run of Spruk (2026) for the EASC replication study.

Rok Spruk, "Synthetic Control under Out-of-Span Common Shocks: Diagnosis,
Exposure Balance and Correction", arXiv:2610.05535v1. The reference is the
author's replication package (submitted to the Journal of Applied
Econometrics). It is not public yet and carries no licence, so it is not
vendored here: this script runs it from a local copy and records what it
printed, with a sha256 of every file it ran.

Run::

    python benchmarks/reference/easc_iran/capture.py --package <unzipped package>

The package is copied to a temporary directory first, because its scripts write
to ``../results``. Inside the copy the script runs the four application scripts
(``app.py``, ``app_inference.py``, ``app_influence.py``, ``phi_phase.py``)
unchanged, then imports ``mc.py`` to record the Monte Carlo draws and the
per-replication output of ``mc.one_rep`` on fixed seeds. It writes
``reference.json`` and ``provenance.json`` next to this file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
APP_SCRIPTS = ("app.py", "app_inference.py", "app_influence.py", "phi_phase.py")
HASHED = ("code/ebsc.py", "code/app.py", "code/app_inference.py", "code/app_influence.py",
          "code/phi_phase.py", "code/mc.py", "code/modern.py", "code/scm.py",
          "data/panel_2012.csv", "data/oilrents.csv", "data/brent.csv")
# Monte Carlo seeds follow mc.py: 20261002 + 1000 * (design index) + replication.
MC_DESIGNS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11")
MC_REPS = (0, 1, 2)
MC_ONE_REP = ("S0", "S2", "S3", "S9", "S11")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arr(v):
    import numpy as np
    return [float(z) for z in np.asarray(v, dtype=float).ravel()]


def _mc_worker(out_path: str) -> None:
    """Runs inside the package's ``code`` directory, with it on ``sys.path``."""
    import numpy as np
    import mc  # the author's simulator

    draws, reps = {}, {}
    for i, scn in enumerate(mc.SCEN):
        for r in MC_REPS:
            seed = 20261002 + 1000 * i + r
            Y, s, _ = mc.draw(scn, np.random.default_rng(seed))
            # A bitwise digest of the draw plus two summaries for diagnosing a mismatch.
            draws[f"{scn}/{r}"] = dict(seed=seed, s=_arr(s), Y_sum=float(Y.sum()), Y_treated=_arr(Y[:, 0]),
                                       Y_sha256=hashlib.sha256(np.ascontiguousarray(Y).tobytes()).hexdigest())
            if scn in MC_ONE_REP:
                res = mc.one_rep((scn, r, seed))
                keep = ("SC", "EASC", "EBSC-min", "EASC*")
                reps[f"{scn}/{r}"] = dict(
                    seed=seed,
                    err={e: res["err"][e] for e in keep},
                    inf={f"{k[0]}|{k[1]}|{k[2]}": (float(v) if not isinstance(v, tuple)
                                                   else [float(v[0]), float(v[1])])
                         for k, v in res["inf"].items() if k[0] in ("SC", "EASC")},
                    diag={k: float(v) for k, v in res["diag"].items()},
                )
    Path(out_path).write_text(json.dumps(dict(draws=draws, one_rep=reps)))


def _pool_block(r: dict, pool: list, years: list) -> dict:
    path = [dict(frac=p["frac"], tol=p["tol"], ei=p["ei"], rmspe=p["rmspe"],
                 crit=p.get("crit"), w=dict(zip(pool, _arr(p["w"])))) for p in r["path"]]
    return dict(
        s1=float(r["s1"]), s0=dict(zip(pool, _arr(r["s0"]))), dmin=float(r["dmin"]),
        w_sc=dict(zip(pool, _arr(r["w_sc"]))), rmspe_sc=float(r["rmspe_sc"]), ei_sc=float(r["ei_sc"]),
        r2=float(r["r2"]), r2_all=float(r["r2_all"]),
        years=years,
        kappa=_arr(r["kappa"]), kappa_lat=_arr(r["kappa_lat"]), partial=_arr(r["partial"]),
        partial_all=_arr(r["partial_all"]), phi=_arr(r["phi"]), se_phi=_arr(r["se_phi"]),
        g_sc=_arr(r["g_sc"]), g_easc=_arr(r["g_easc"]), g_ebsc_min=_arr(r["g_ebsc_min"]),
        g_easc_tuned=_arr(r["g_easc_tuned"]), ei_min=float(r["ei_min"]), rmspe_min=float(r["rmspe_min"]),
        ei_tuned=float(r["ei_tuned"]), best_frac=float(r["best"]["frac"]), path=path,
    )


def capture(package: Path) -> None:
    package = package.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "pkg"
        shutil.copytree(package / "code", work / "code")
        shutil.copytree(package / "data", work / "data")
        (work / "results").mkdir()
        code = work / "code"
        stdout = {}
        for script in APP_SCRIPTS:
            run = subprocess.run([sys.executable, script], cwd=code, capture_output=True, text=True, check=True)
            stdout[script] = run.stdout
        mc_out = work / "results" / "mc_capture.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mc-worker", str(mc_out)],
                       cwd=code, check=True, env={"PYTHONPATH": str(code), "PATH": "/usr/bin:/bin"})

        # The pickles were written by the author's code in this run, not shipped ones.
        main = pickle.loads((work / "results" / "app_main.pkl").read_bytes())
        inf = pickle.loads((work / "results" / "app_inference.pkl").read_bytes())
        influence = json.loads((work / "results" / "app_influence.json").read_text())
        phi_phase = json.loads((work / "results" / "phi_phase.json").read_text())
        mc = json.loads(mc_out.read_text())

    years = list(range(1996, 2025))
    pools = {"base": main["base"]["pool"], "ext": main["ext"]["pool"]}
    values, tables = {}, {}
    for key in ("base", "ext"):
        blk = _pool_block(main[key], pools[key], years)
        tables[key] = blk
        values[f"{key}_dmin"] = blk["dmin"]
        values[f"{key}_ei_sc"] = blk["ei_sc"]
        values[f"{key}_rmspe_sc"] = blk["rmspe_sc"]
        values[f"{key}_r2"] = blk["r2"]
        values[f"{key}_rmspe_min"] = blk["rmspe_min"]
        rows = []
        for _, row in inf[key]["table"].iterrows():
            rec = {c: (float(row[c]) if c not in ("est", "phase", "ci0", "ci05", "ci1") else row[c])
                   for c in ("est", "phase", "tau", "sd", "sdm", "kbar", "Mstar", "Mstar_mad", "pval",
                             "ci0", "ci05", "ci1")}
            for c in ("ci0", "ci05", "ci1"):
                rec[c] = [float(rec[c][0]), float(rec[c][1])]
            rows.append(rec)
            for c in ("tau", "sd", "kbar", "Mstar", "pval"):
                values[f"{key}_{row['est']}_{row['phase']}_{c}"] = float(row[c])
        tables[f"{key}_inference"] = rows
    tables["robustness"] = json.loads(main["rob"].to_json(orient="records"))
    tables["influence"] = influence
    tables["phi_phase"] = phi_phase
    tables["mc"] = mc

    (HERE / "reference.json").write_text(json.dumps(dict(values=values, tables=tables), indent=1) + "\n")
    (HERE / "reference.out").write_text("".join(f"== {s} ==\n{stdout[s]}\n" for s in APP_SCRIPTS))

    import numpy, scipy, pandas, cvxpy
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prov = dict(
        generated_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=git, platform=platform.platform(), python=platform.python_version(),
        numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__, cvxpy=cvxpy.__version__,
        command="python benchmarks/reference/easc_iran/capture.py --package <unzipped package>",
        package="esc-scm-replication (author's JAE submission package; not public at capture time)",
        package_files=[dict(path=p, sha256=_sha256(package / p)) for p in HASHED],
        basedata=dict(path="basedata/iran_sanctions_oil.csv",
                      sha256=_sha256(ROOT / "basedata" / "iran_sanctions_oil.csv")),
    )
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", type=Path)
    ap.add_argument("--mc-worker", metavar="OUT")
    a = ap.parse_args()
    if a.mc_worker:
        _mc_worker(a.mc_worker)
    elif a.package:
        capture(a.package)
    else:
        ap.error("--package is required")
