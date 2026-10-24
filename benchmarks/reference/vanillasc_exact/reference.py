#!/usr/bin/env python3
"""Live captured run of Abadie and Shang's replication package (``run_all.py``).

Abadie and Shang (2026), "Synthetic Control: An Exact Geometric Formulation",
publish the code behind every data-based table of the paper at
https://github.com/lychemon/synth-AS-replication. This script fetches it at a
pinned commit, with the companion ``synthx_package`` beside it (its Stata bridge
is the second solver ``verify_results.py`` compares against), runs
``code/scripts/run_all.py`` unchanged in a scratch copy, and prints every number
the run writes to ``output/``:

- Tables 1 and 5: the diagonal-class donor weights and validation losses of
  California and West Germany, and of California with the lagged-outcome years
  1975, 1980 and 1988 dropped from validation (the leakage-free case);
- Tables D.1 and D.2: the positive definite class, with the visible-facet count
  and the facet holding the optimum;
- the mean post-treatment gaps the paper reports (1989-2000, 1991-2003);
- Table 3: faces of the donor hull by dimension, attainable, visible and all;
- Tables 2, 4 and 6: the predictor weights that rationalise the diagonal
  optimum, diagonal (log-det maximiser, analytic centre, support) and positive
  definite (log-det maximiser, analytic centre, identified ranges). ``run_all.py``
  rounds these to four decimals; the same functions are also called here with
  ``decimals=None`` on the saved weights, keys ending in ``.full``, and the positive
  definite log-det maximiser and analytic centre are recorded entry by entry
  (``logdet_matrix[k,l]``, ``center_matrix[k,l]``, upper triangle).

The run's own check, ``output/verification.txt``, must report every paper number
reproduced, and the two panels in its ``data/`` must be byte-identical to
``basedata/abadie_shang_smoking.csv`` and ``basedata/abadie_shang_germany.csv``.
Nothing of theirs is redistributed apart from those two CC BY 4.0 panels; the
repositories are fetched into the gitignored ``benchmarks/reference/.cache``.

Prints the ``== REFERENCE VALUES ==`` block that ``generate.py`` parses, then
``== SESSION INFO ==`` and the verification report. Run from the repository
root::

    python benchmarks/reference/vanillasc_exact/reference.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reference._fetch import fetch_pinned_repo  # noqa: E402

REPOS = {
    "synth-AS-replication": ("https://github.com/lychemon/synth-AS-replication.git",
                             "791a3865fe3e9fd0b82a8121c0b338791b7ffa39"),
    "synthx_package": ("https://github.com/lychemon/synthx_package.git",
                       "a10a28c571a906ee76574bad9c5446c22c3a184f"),
}
DATA = {"smoking_data.csv": ROOT / "basedata" / "abadie_shang_smoking.csv",
        "german_reunification.csv": ROOT / "basedata" / "abadie_shang_germany.csv"}
CACHE = ROOT / "benchmarks" / "reference" / ".cache"
CASES = {"california_headline": "california", "california_leakage_free": "california_leakage_free",
         "germany": "germany"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch() -> Path:
    """Both repositories, side by side in a scratch directory, at the pinned commits."""
    work = Path(tempfile.mkdtemp(prefix="vanillasc_exact_"))
    for name, (url, sha) in REPOS.items():
        cached = CACHE / f"lychemon-{name}-{sha[:12]}"
        if not (cached / "README.md").exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            fetch_pinned_repo(url, sha, cached)
        shutil.copytree(cached, work / name, ignore=shutil.ignore_patterns(".git", "output"))
    for theirs, ours in DATA.items():
        if _sha256(work / "synth-AS-replication" / "data" / theirs) != _sha256(ours):
            raise SystemExit(f"data/{theirs} at the pinned commit differs from {ours.name}")
    return work


def _emit(key: str, value) -> None:
    print(f"{key}\t{float(value)!r}")


def _weights(prefix: str, weights: dict, donors: list) -> None:
    for d in donors:
        _emit(f"{prefix}[{d}]", weights.get(d, 0.0))


def main() -> int:
    work = _fetch()
    rep = work / "synth-AS-replication"
    run = subprocess.run([sys.executable, "code/scripts/run_all.py"], cwd=rep,
                         capture_output=True, text=True)
    out = rep / "output"
    verification = (out / "verification.txt").read_text() if (out / "verification.txt").exists() else ""
    if run.returncode != 0 or "every reported number reproduced" not in verification:
        sys.stderr.write(run.stdout[-3000:] + run.stderr[-3000:])
        return 1

    sys.path.insert(0, str(rep / "code" / "support"))
    import data_preparation as dp  # noqa: E402  (theirs, at the pinned commit)
    import predictor_weights as pw  # noqa: E402
    import treatment_effects as te  # noqa: E402

    def load(name):
        return json.loads((out / name).read_text())

    summary, pdj = load("diagonal_solver_summary.json"), load("pd_donor_weights.json")
    leak = load("california_diagonal_donor_weights_leakage_free.json")
    dgc, vst = load("diagonal_predictor_weights.json"), load("pd_predictor_weights_logdet.json")
    anc = load("pd_predictor_weights_analytic_center.json")
    ca = dp.build_california(str(rep / "data" / "smoking_data.csv"))
    ge = dp.build_germany(str(rep / "data" / "german_reunification.csv"))
    data = {"california": (ca, "California", range(1989, 2001)),
            "germany": (ge, "West Germany", range(1991, 2004))}

    print("== REFERENCE VALUES ==")
    for case, (d, treated, post) in data.items():
        diag_w = summary[case]["donor_weights"]
        pd_w = pdj[case]["pd_weights"]
        _weights(f"{case}.diag.w", diag_w, d["donors"])
        _emit(f"{case}.diag.loss", summary[case]["validation_loss"])
        _weights(f"{case}.pd.w", pd_w, d["donors"])
        _emit(f"{case}.pd.loss", pdj[case]["pd_loss"])
        _emit(f"{case}.pd.n_visible_facets", pdj[case]["n_visible_facets"])
        for donor in pdj[case]["facet_vertices"]:
            _emit(f"{case}.pd.facet[{donor}]", 1)
        for cls, w in (("diag", diag_w), ("pd", pd_w)):
            _emit(f"{case}.{cls}.mean_gap", te.mean_effect(
                d["years"], d["outcomes"][treated], te.synthetic_path(d, w), post))
        for kind in ("attainable", "visible", "total"):
            for dim, n in enumerate(summary[case][f"faces_{kind}_by_dim"]):
                _emit(f"{case}.faces.{kind}[{dim}]", n)
    _weights("california_leakage_free.diag.w", leak["donor_weights"], ca["donors"])
    _emit("california_leakage_free.diag.loss", leak["loss"])

    for case in pw.predictor_weight_cases(str(rep / "data"), str(out)):
        tag = CASES[case.tag]
        g = dgc[case.tag]
        for label in ("logdet", "analytic_center"):
            for name, v in g[label].items():
                _emit(f"{tag}.diag_v.{label}[{name}]", v)
        for name in g["support"]:
            _emit(f"{tag}.diag_v.support[{name}]", 1)
        _emit(f"{tag}.diag_v.n_inactive_donors", g["n_inactive_donors"])
        _emit(f"{tag}.diag_v.n_implicit_equalities", g["n_implicit_equalities"])
        full = pw.diag_centers(case.X0, case.X1, case.weights, case.predictor_names, decimals=None)
        for label in ("logdet", "analytic_center"):
            for name, v in full[label].items():
                _emit(f"{tag}.diag_v.{label}.full[{name}]", v)

        lv, av = vst[case.tag], anc[case.tag]
        for name, v in lv["diag"].items():
            _emit(f"{tag}.pd_v.logdet[{name}]", v)
            _emit(f"{tag}.pd_v.range_lo[{name}]", lv["ranges"][name][0])
            _emit(f"{tag}.pd_v.range_hi[{name}]", lv["ranges"][name][1])
            _emit(f"{tag}.pd_v.center[{name}]", av["diag"][name])
        _emit(f"{tag}.pd_v.logdet_value", lv["logdet"])
        _emit(f"{tag}.pd_v.logdet_min_eig", lv["min_eig"])
        _emit(f"{tag}.pd_v.center_min_eig", av["min_eig"])
        _emit(f"{tag}.pd_v.center_n_implicit_equalities", av["n_implicit_equalities"])
        Vl, fl = pw.compute_pd_logdet_and_ranges(case.X1, case.X0, case.weights, case.predictor_names,
                                                 decimals=None)
        Vc, fc = pw.analytic_center(case.X1, case.X0, case.weights, case.predictor_names, decimals=None)
        for name in fl["diag"]:
            _emit(f"{tag}.pd_v.logdet.full[{name}]", fl["diag"][name])
            _emit(f"{tag}.pd_v.range_lo.full[{name}]", fl["ranges"][name][0])
            _emit(f"{tag}.pd_v.range_hi.full[{name}]", fl["ranges"][name][1])
            _emit(f"{tag}.pd_v.center.full[{name}]", fc["diag"][name])
        for k, l in zip(*np.triu_indices(len(case.predictor_names))):   # the matrices themselves
            _emit(f"{tag}.pd_v.logdet_matrix[{k},{l}]", Vl[k, l])
            _emit(f"{tag}.pd_v.center_matrix[{k},{l}]", Vc[k, l])

    print("== SESSION INFO ==")
    print(f"python_{platform.python_version()}")
    for mod in ("numpy", "scipy", "cvxpy", "clarabel"):
        print(f"{mod}_{__import__(mod).__version__}")
    for name, (url, sha) in REPOS.items():
        print(f"{name}\t{url}\t{sha}")
    for theirs, ours in DATA.items():
        print(f"data/{theirs}\t{_sha256(ours)}\t= {ours.relative_to(ROOT)}")
    print("== VERIFICATION (output/verification.txt) ==")
    print(verification)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
