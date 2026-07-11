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

from benchmarks.reference.export_comparison import (  # noqa: E402
    _read_comparison_csv,
    _missing_cases,
)

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


# A few cases record an internal function in their comparison() estimator field;
# map those to the estimator a reader recognises.
_ESTIMATOR_REMAP = {
    "ridge_augment_weights": "VanillaSC",   # ascm_kansas -- ridge-augmented VanillaSC
    "fit_en_scm": "LINF",                   # linf_* -- L-infinity SC
    "run_pda_multitreat": "PDA",            # pda_brexit -- Panel Data Approach
}


def _canon(est: str) -> str:
    """Canonical estimator label: strip the descriptive backend/config suffix the
    case authors append (``ClusterSC/PCR (run_pcr ...)`` -> ``ClusterSC``) and map
    the handful of function-name fields to the estimator a reader recognises."""
    base = (est or "?").split("(")[0].split("/")[0].strip() or "?"
    return _ESTIMATOR_REMAP.get(base, base)


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
            "estimator": _canon(meta.get("estimator", "?")),
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


def pending() -> list:
    """Cross-validation cases that define ``comparison()`` but have no committed
    ``comparison.csv`` yet -- captured by the daily action once their reference
    toolchain provisions. Listed so coverage is never silently understated."""
    out = []
    for case in sorted(_missing_cases()):
        man = REF_DIR / case / "manifest.json"
        impl = ""
        if man.exists():
            impl = json.loads(man.read_text()).get("reference_impl", "")
        out.append({"case": case, "reference_impl": impl})
    return out


_VERDICT_LABEL = {
    "exact": "exact — matches to display precision",
    "tight": "tight",
    "close": "close",
    "documented": "documented — see notes",
}


_ORDER = {"exact": 0, "tight": 1, "close": 2, "documented": 3}


def _slug(est: str) -> str:
    return "val-" + "".join(c.lower() if c.isalnum() else "-" for c in est)


def _verdict_mix(recs: list) -> str:
    """A compact honest summary like ``4 exact · 3 tight``."""
    counts = {}
    for r in recs:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return " · ".join(f"{counts[v]} {v}" for v in
                      sorted(counts, key=lambda v: _ORDER[v]))


def to_rst(by_est: dict, only: str | None = None, pend: list | None = None) -> str:
    ests = [only] if only else sorted(by_est)
    ests = [e for e in ests if by_est.get(e)]
    pend = pend or []
    n_checks = sum(len(by_est[e]) for e in ests)
    n_exact = sum(1 for e in ests for r in by_est[e] if r["verdict"] == "exact")
    n_tight = sum(1 for e in ests for r in by_est[e] if r["verdict"] == "tight")

    L = [
        ".. _validation:",
        "",
        "Validation dashboard",
        "====================",
        "",
        "Every estimator in mlsynth is checked against the original authors' code",
        "on real data. This page is generated from the pinned reference bundles the",
        "test suite asserts against, so the numbers here cannot drift from what CI",
        "enforces. Each row links to the reference implementation, the dataset (with",
        "checksum), and the mlsynth case that runs the check.",
        "",
        f"Coverage: **{n_checks} cross-validation checks** against original",
        f"implementations across **{len(ests)} estimators** -- "
        f"{n_exact} reproduce the reference to display precision, {n_tight} to",
        "within two percent."
        + (f" A further {len(pend)} are captured on the next daily run"
           " (see `Pending capture`_)." if (pend and not only) else "")
        + " Per-estimator paper replications (Path A / Path B) are catalogued in"
        " :doc:`replications`.",
        "",
        "Legend: **exact** (agreement to display precision), **tight** (worst",
        "relative deviation :math:`\\le 2\\%`), **close** (:math:`\\le 10\\%`), and",
        "**documented** (looser, with a stated reason on the estimator's replication",
        "page -- typically an intrinsically extrapolated or weakly-identified"
        " quantity).",
        "",
    ]

    # Tier 1 -- one row per estimator.
    if not only:
        L += ["Summary", "-------", "",
              ".. list-table::", "   :header-rows: 1",
              "   :widths: 26 14 44 16", ""]
        L += ["   * - Estimator", "     - Checks", "     - Agreement",
              "     - Worst max \\|Δ\\|"]
        for est in ests:
            recs = by_est[est]
            worst = max((r["max_abs_diff"] for r in recs
                         if r["max_abs_diff"] == r["max_abs_diff"]), default=float("nan"))
            mad = "—" if worst != worst else f"{worst:.2g}"
            L += [
                f"   * - :ref:`{est} <{_slug(est)}>`",
                f"     - {len(recs)}",
                f"     - {_verdict_mix(recs)}",
                f"     - {mad}",
            ]
        L.append("")

    # Tier 2 -- per estimator, one row per reference.
    for est in ests:
        recs = by_est[est]
        L += [f".. _{_slug(est)}:", "", est, "-" * max(len(est), 4), ""]
        L += [".. list-table::", "   :header-rows: 1",
              "   :widths: 22 28 8 12 14 16", ""]
        L += ["   * - Reference", "     - Dataset", "     - #",
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

    # Honesty: cross-validation cases whose comparison.csv is not captured yet.
    if pend and not only:
        L += ["Pending capture", "---------------", "",
              "These cross-validation cases are wired up but their reference had",
              "not been captured when this page was last generated; the daily",
              "action records them once its toolchain provisions.", "",
              ".. list-table::", "   :header-rows: 1", "   :widths: 30 50", ""]
        L += ["   * - Case", "     - Reference"]
        for p in pend:
            L += [f"   * - `{p['case']} <{GH}/benchmarks/cases/{p['case']}.py>`__",
                  f"     - {p['reference_impl'] or '—'}"]
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    by_est = collect()
    rst = to_rst(by_est, only=only, pend=pending())
    out = ROOT / "docs" / "validation.rst"
    out.write_text(rst)
    n = len(by_est.get(only, [])) if only else sum(len(v) for v in by_est.values())
    print(f"wrote {out} ({n} cross-validation row(s)"
          f"{' for ' + only if only else ' across ' + str(len(by_est)) + ' estimators'})")


if __name__ == "__main__":
    main()
