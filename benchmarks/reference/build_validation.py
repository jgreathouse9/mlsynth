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
    "qkrcks0218/SPSC R (single-proxy synthetic control)":
        "https://github.com/qkrcks0218/SPSC",
    "R package scinference (sc.cf t-test, live run, captured)":
        "https://github.com/kwuthrich/scinference",
    "R package bsynth (bayesianSynth, predictor_match=FALSE, live run)":
        "https://github.com/ignacio82/bsynth",
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
    "fit_en_scm": "RESCM",                  # linf_* -- the RESCM engine's L-infinity method
    "run_pda_multitreat": "PDA",            # pda_brexit -- Panel Data Approach
    # Labels that name the reference package or an engine method instead of the
    # mlsynth class. Left unmapped they look like estimators the package does
    # not export, and their checks go uncounted against coverage.
    "ClusterSC": "CLUSTERSC",   # srho1/ClusterSC, the reference for part of CLUSTERSC
    "LINF": "RESCM",            # the L-infinity method of the RESCM engine
    "SPSC": "PROXIMAL",         # single-proxy SC -- proximal_helpers.spsc
}


def _canon(est: str) -> str:
    """Canonical estimator label: strip the descriptive backend/config suffix the
    case authors append (``ClusterSC/PCR (run_pcr ...)`` -> ``ClusterSC``) and map
    the handful of function-name fields to the estimator a reader recognises."""
    base = est.split("(")[0].split("/")[0].strip()
    # A field may name a variant after the estimator ("TSSC MSCa"); the first
    # token is the estimator and the rest is the configuration.
    if " " in base and base not in _ESTIMATOR_REMAP:
        base = base.split()[0]
    return _ESTIMATOR_REMAP.get(base, base)


def _estimators_of(meta: dict, case: str) -> list:
    """Every estimator a bundle checks.

    A case may compare more than one -- ``brabander_brexit_table1`` runs seven
    estimators across four mlsynth classes -- so the field is a comma-separated
    list and the case appears under each. An empty field used to fall back to
    ``"?"``, which produced an unnamed dashboard row carrying real checks that
    no reader could attribute; that is a defect in the bundle, and is raised.
    """
    raw = (meta.get("estimator") or "").strip()
    if not raw:
        raise ValueError(
            f"benchmarks/reference/{case}/comparison.csv has no `estimator` in "
            f"its metadata header, so its checks cannot be attributed. Set "
            f"`mlsynth_call['estimator']` in the case's comparison() and "
            f"regenerate the bundle.")
    # Strip parenthetical config notes first: they routinely contain commas
    # ("ClusterSC/PCR (run_pcr OLS, fixed rank)"), and splitting on those
    # produces label fragments like "fixed rank)".
    import re as _re
    flat = _re.sub(r"\([^)]*\)", "", raw)
    parts = _re.split(r"[,+]", flat)
    return [e for e in (_canon(part) for part in parts) if e]


def _all_estimators() -> list:
    """Every estimator class mlsynth exports, so coverage is measured against
    the package instead of against the corpus that happens to exist."""
    import inspect
    import mlsynth
    return sorted(n for n in mlsynth.__all__
                  if inspect.isclass(getattr(mlsynth, n, None))
                  and hasattr(getattr(mlsynth, n), "fit"))


def _benchmarked() -> dict:
    """``{estimator: [case, ...]}`` over the registered benchmark cases.

    Matched on the class name, the ``<name>_helpers`` package and the
    ``estimators.<name>`` module, because a case can drive an estimator through
    its helpers without naming the class -- ``cfm`` does exactly that.
    """
    import re
    from benchmarks import registry
    out: dict = {}
    for name, mod in registry.CASES.items():
        f = ROOT / (mod.replace(".", "/") + ".py")
        if not f.exists():
            continue
        src = f.read_text()
        for e in _all_estimators():
            pats = (rf"\b{re.escape(e)}\b",
                    rf"\b{re.escape(e.lower())}_helpers\b",
                    rf"\bestimators\.{re.escape(e.lower())}\b")
            if any(re.search(pt, src) for pt in pats):
                out.setdefault(e, []).append(name)
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
        ests = _estimators_of(meta, case)
        rec = {
            "case": case,
            "estimator": ests[0],
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
        for e in ests:
            by_est.setdefault(e, []).append({**rec, "estimator": e})
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
    all_est = _all_estimators()
    bench = _benchmarked()
    xval = set(ests)
    n_covered = sum(1 for e in all_est if e in xval)
    bench_only = [e for e in all_est if e not in xval and e in bench]
    # An estimator can be cross-validated in prose without a registered case or
    # a captured bundle -- SCTA against augsynth on the Texas SB8 panel is the
    # example. That is validation, but it is not re-runnable, so it is reported
    # as its own state instead of being lumped in with the unguarded ones.
    reps = {f.stem for f in (ROOT / "docs" / "replications").glob("*.rst")}
    documented = [e for e in all_est
                  if e not in xval and e not in bench and e.lower() in reps]
    neither = [e for e in all_est if e not in xval and e not in bench
               and e.lower() not in reps]

    L = [
        ".. _validation:",
        "",
        "Validation dashboard",
        "====================",
        "",
        "mlsynth estimators are checked against the original authors' code on",
        "real data wherever a runnable reference exists. This page is generated",
        "from the pinned reference bundles the test suite asserts against, so the",
        "numbers here cannot drift from what CI enforces. Each row links to the",
        "reference implementation, the dataset (with checksum), and the mlsynth",
        "case that runs the check.",
        "",
        f"Coverage: **{n_checks} cross-validation checks** against original",
        f"implementations, covering **{n_covered} of {len(all_est)} estimators** -- "
        f"{n_exact} reproduce the reference to display precision, {n_tight} to",
        "within two percent."
        + (f" A further {len(pend)} are captured on the next daily run"
           " (see `Pending capture`_)." if (pend and not only) else "")
        + " Per-estimator paper replications (Path A / Path B) are catalogued in"
        " :doc:`replications`.",
        "",
        "What the denominator counts: exported estimator classes. That is",
        "auditable, and it is not the same as a count of methods. Several classes",
        "nest a family -- CLUSTERSC covers Robust SC, principal-component",
        "regression, a Bayesian variant and a convex one -- so one covered row can",
        "stand for several validated methods. A method can also be reachable from",
        "more than one class: TSSC fits the baseline VanillaSC synthetic control as",
        "one of its variants, so an uncovered class need not be an unvalidated",
        "method. Take the ratio as a bound on class-level coverage and the",
        "per-estimator sections below for what is actually checked.",
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

    # Estimators validated by a paper replication instead of a cell-by-cell
    # cross-check. They belong here because they are benchmarked, and this page
    # is otherwise read as the whole validation picture.
    if bench_only and not only:
        L += ["Benchmarked without a reference cross-check",
              "-------------------------------------------", "",
              "These estimators carry a durable benchmark case but no reference",
              "bundle, because no runnable reference implementation exists or its",
              "output cannot be captured. They are validated against the source",
              "paper instead -- see :doc:`replications` and each estimator's page.",
              "", ".. list-table::", "   :header-rows: 1", "   :widths: 22 78",
              "", "   * - Estimator", "     - Benchmark case(s)"]
        for e in bench_only:
            cases = " \u00b7 ".join(
                f"`{c} <{GH}/benchmarks/cases/{c}.py>`__" for c in sorted(bench[e]))
            L += [f"   * - {e}", f"     - {cases}"]
        L += [""]

    if documented and not only:
        L += ["Validated, but not re-runnable",
              "------------------------------", "",
              "These estimators are cross-validated against a reference on the",
              "paper's own data, and the comparison is written up, but no",
              "benchmark case or captured bundle exists -- so CI does not re-run",
              "it and a regression would not be caught here:", ""]
        for e in documented:
            L += [f"* {e} -- :doc:`replications/{e.lower()}`"]
        L += [""]

    if neither and not only:
        L += ["No durable benchmark", "--------------------", "",
              "Neither a reference bundle, a registered benchmark case, nor a",
              "replication page, so nothing guards these against regression: "
              + ", ".join(f"``{e}``" for e in neither) + ".", ""]

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
