#!/usr/bin/env python3
"""Build the public Validation dashboard from the committed comparison corpus.

The goal this serves: mlsynth's accuracy should be *publicly, inspectably* proven
against the original authors' code -- on the docs site, linkable, honest about
how tight each match is, and impossible to silently drift because it is generated
from the same pinned reference bundles the CI asserts against.

Source of truth (no re-running estimators): each cross-validation case commits a
``benchmarks/reference/<case>/comparison.csv`` -- a metadata header (estimator,
config, reference implementation + version) plus ``quantity, mlsynth, reference,
abs_diff`` rows -- alongside a ``manifest.json`` (paper, dataset, path type) and
``provenance.json`` (package versions, data SHA-256, git SHA). This reads that
corpus and emits one web-native page:

    docs/validation.rst

grouped by estimator, one row per (estimator x reference) with an honest verdict
(exact / tight / close / documented), links out to the original codebase, the
dataset checksum, and the mlsynth case source. Run::

    python benchmarks/reference/build_validation.py            # all estimators
    python benchmarks/reference/build_validation.py VanillaSC  # one estimator
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmarks.reference.export_comparison import _read_comparison_csv  # noqa: E402

GH = "https://github.com/jgreathouse9/mlsynth/blob/main"

# Curated links from a reference implementation to its public home. Text is shown
# verbatim when a URL is not known; add entries as references are cross-checked.
KNOWN_REFS = {
    "R package Synth::synth": "https://CRAN.R-project.org/package=Synth",
    "R package MSCMT": "https://CRAN.R-project.org/package=MSCMT",
    "Python package scpi_pkg": "https://pypi.org/project/scpi-pkg/",
    "synth-inference/synthdid R (synthdid_estimate)":
        "https://github.com/synth-inference/synthdid",
    "susanathey/MCPanel R (mcnnm_cv, defaults)":
        "https://github.com/susanathey/MCPanel",
    "R package scinference (sc.cf t-test, live run, captured)":
        "https://github.com/kwuthrich/scinference",
}


def _verdict(rows: list) -> tuple:
    """Honest tightness band from the worst relative deviation across quantities.

    Relative deviation ``|Δ| / max(|reference|, 1)`` avoids blowing up on
    near-zero weights while staying meaningful for ATT/SSR-scale quantities.
    """
    worst = 0.0
    for r in rows:
        try:
            ref = abs(float(r["reference"]))
            rel = abs(float(r["abs_diff"])) / max(ref, 1.0)
        except (ValueError, TypeError):
            continue
        worst = max(worst, rel)
    if worst <= 1e-4:
        return "exact", worst
    if worst <= 2e-2:
        return "tight", worst
    if worst <= 1e-1:
        return "close", worst
    return "documented", worst


def _ref_link(impl: str) -> str:
    url = KNOWN_REFS.get(impl)
    return f"`{impl} <{url}>`__" if url else impl


def _bundle_meta(case: str) -> dict:
    out = {}
    man = REF_DIR / case / "manifest.json"
    prov = REF_DIR / case / "provenance.json"
    if man.exists():
        out.update(json.loads(man.read_text()))
    if prov.exists():
        p = json.loads(prov.read_text())
        data = p.get("data", [])
        if data:
            out["dataset"] = Path(data[0]["path"]).name
            out["dataset_sha"] = data[0].get("sha256", "")[:12]
    return out


def collect() -> dict:
    """Return {estimator: [case_record, ...]} from every committed comparison.csv."""
    by_est: dict = {}
    for csv_path in sorted(REF_DIR.glob("*/comparison.csv")):
        case = csv_path.parent.name
        parsed = _read_comparison_csv(csv_path)
        meta, rows = parsed["meta"], parsed["rows"]
        verdict, worst = _verdict(rows)
        max_abs = max((abs(float(r["abs_diff"])) for r in rows
                       if _is_num(r["abs_diff"])), default=float("nan"))
        bundle = _bundle_meta(case)
        rec = {
            "case": case,
            "estimator": meta.get("estimator", "?"),
            "reference_impl": meta.get("reference_impl", "?"),
            "reference_version": meta.get("reference_version", ""),
            "paper": bundle.get("paper", ""),
            "path_type": bundle.get("path_type", ""),
            "dataset": bundle.get("dataset", ""),
            "dataset_sha": bundle.get("dataset_sha", ""),
            "n_quantities": len(rows),
            "max_abs_diff": max_abs,
            "worst_rel": worst,
            "verdict": verdict,
            "rows": rows,
        }
        by_est.setdefault(rec["estimator"], []).append(rec)
    for recs in by_est.values():
        recs.sort(key=lambda r: r["case"])
    return by_est


def _is_num(x) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


_VERDICT_LABEL = {
    "exact": "exact — matches to display precision",
    "tight": "tight",
    "close": "close",
    "documented": "documented — see notes",
}


def to_rst(by_est: dict, only: str | None = None) -> str:
    L = [
        ".. _validation:",
        "",
        "Validation dashboard",
        "====================",
        "",
        "Every estimator in mlsynth is checked against the original authors' code,",
        "or the paper's reported numbers, on real data. This page is generated from",
        "the pinned reference bundles the test suite asserts against, so the numbers",
        "here cannot drift from what CI enforces. Each row links to the reference",
        "implementation, the dataset (with checksum), and the mlsynth case that runs",
        "the check. Verdicts are honest about tightness -- see the legend.",
        "",
        "Legend: **exact** (agreement to display precision), **tight** (worst",
        "relative deviation :math:`\\le 2\\%`), **close** (:math:`\\le 10\\%`), and",
        "**documented** (looser, with a stated reason on the estimator's replication",
        "page -- typically an intrinsically extrapolated or weakly-identified"
        " quantity).",
        "",
    ]
    ests = [only] if only else sorted(by_est)
    for est in ests:
        recs = by_est.get(est, [])
        if not recs:
            continue
        L += [est, "-" * max(len(est), 4), ""]
        L += [".. list-table::", "   :header-rows: 1",
              "   :widths: 22 30 16 12 10 10", ""]
        L += ["   * - Reference", "     - Dataset", "     - Quantities",
              "     - max \\|Δ\\|", "     - Verdict", "     - Case"]
        for r in recs:
            ds = f"``{r['dataset']}``" if r["dataset"] else "—"
            if r["dataset_sha"]:
                ds += f" ({r['dataset_sha']}…)"
            mad = "—" if r["max_abs_diff"] != r["max_abs_diff"] else f"{r['max_abs_diff']:.2g}"
            L += [
                f"   * - {_ref_link(r['reference_impl'])}",
                f"     - {ds}",
                f"     - {r['n_quantities']}",
                f"     - {mad}",
                f"     - {_VERDICT_LABEL[r['verdict']]}",
                f"     - `{r['case']} <{GH}/benchmarks/cases/{r['case']}.py>`__",
            ]
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    by_est = collect()
    rst = to_rst(by_est, only=only)
    out = ROOT / "docs" / "validation.rst"
    out.write_text(rst)
    n = len(by_est.get(only, [])) if only else sum(len(v) for v in by_est.values())
    print(f"wrote {out} ({n} cross-validation row(s)"
          f"{' for ' + only if only else ' across ' + str(len(by_est)) + ' estimators'})")


if __name__ == "__main__":
    main()
