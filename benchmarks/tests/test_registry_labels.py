"""Every benchmark case declares what it validates and what it runs on.

The registry used to carry that in a free-text comment and the docs page
carried it again in a section heading, and the two drifted: 52 cases were
commented ``Path B`` against 56 rows under the docs page's Monte Carlo
section, with only 42 in both. ``botosaru_ferman_covariates`` sat under Monte
Carlo while running on West German GDP, and 61 registered cases had no row on
the page at all. Neither surface could answer "which cases are simulations"
without someone reading 214 modules.

Two things caused the drift, and both are fixed by
:data:`benchmarks.registry.LABELS` instead of by correcting the text.

A case can establish more than one path. ``xu_gsynth_sims`` reproduces Table A5
and cross-validates against ``gsynth 1.0``; ``gsynth_xu_turnout`` reproduces
Table 2 and cross-validates against ``fect``. A single bucket forces whoever
adds the case to drop one, so ``paths`` is a set.

And the path does not say what the case runs on. ``gsynth_av_laws``
cross-validates on a real panel and ``xu_gsynth_sims`` cross-validates on a
simulated one, so "is this a simulation" is a second, independent question.
That is ``data``.

These tests are the gate: a new case cannot merge unlabelled, with a label
outside the vocabulary, or filed on the docs page under a path it does not
claim.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "benchmarks.rst"
GENERATOR = ROOT / "tools" / "gen_benchmark_index.py"

# Which docs section asserts which path. A section not listed here is prose and
# is not checked for membership.
SECTION_PATH = {
    "Path A": "A",
    "Path B": "B",
    "Path C": "C",
    "Cross-validation": "X",
}


@pytest.fixture(scope="module")
def registry():
    from benchmarks import registry as reg
    return reg


def test_every_case_is_labelled(registry):
    missing = sorted(set(registry.CASES) - set(registry.LABELS))
    assert not missing, (
        f"{len(missing)} registered case(s) carry no LABELS entry, so nothing "
        f"can sort or count them: {missing[:10]}"
    )


def test_no_label_without_a_case(registry):
    orphans = sorted(set(registry.LABELS) - set(registry.CASES))
    assert not orphans, (
        f"LABELS names a case the registry does not load: {orphans}"
    )


def test_paths_are_a_non_empty_subset_of_the_vocabulary(registry):
    bad = {
        name: sorted(label.paths)
        for name, label in registry.LABELS.items()
        if not label.paths or not set(label.paths) <= set(registry.PATH_NAMES)
    }
    assert not bad, (
        f"a case must claim at least one path from {sorted(registry.PATH_NAMES)}; "
        f"offenders: {bad}"
    )


def test_data_kind_is_in_the_vocabulary(registry):
    bad = {
        name: label.data
        for name, label in registry.LABELS.items()
        if label.data not in registry.DATA_KINDS
    }
    assert not bad, (
        f"data must be one of {registry.DATA_KINDS}; offenders: {bad}"
    )


def _docs_sections() -> dict[str, set[str]]:
    """Case names listed under each checked section of the docs page."""
    lines = DOCS.read_text().splitlines()
    heads = []
    for i, ln in enumerate(lines[:-1]):
        nxt = lines[i + 1].strip()
        if ln.strip() and nxt and set(nxt) <= set("-~^=") and len(nxt) >= len(ln.strip()):
            heads.append((i, ln.strip()))
    out: dict[str, set[str]] = {}
    for idx, (start, title) in enumerate(heads):
        key = next((k for k in SECTION_PATH if title.startswith(k)), None)
        if key is None:
            continue
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        out[key] = {
            m.group(1)
            for ln in lines[start:end]
            for m in [re.match(r"^\s*\* - ``([a-z0-9_]+)``\s*$", ln)]
            if m
        }
    return out


def test_docs_sections_only_list_cases_that_claim_that_path(registry):
    """The misfiling check: a row under Monte Carlo must claim path B."""
    wrong = []
    for section, names in _docs_sections().items():
        path = SECTION_PATH[section]
        for name in sorted(names):
            label = registry.LABELS.get(name)
            if label is None:
                wrong.append(f"{name}: on the page under {section}, not registered")
            elif path not in label.paths:
                wrong.append(
                    f"{name}: filed under {section} but claims "
                    f"{sorted(label.paths)}"
                )
    assert not wrong, "docs/benchmarks.rst files cases under paths they do not claim:\n  " + "\n  ".join(wrong)


def test_generated_index_is_current():
    """``tools/gen_benchmark_index.py`` output matches what is committed."""
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 0, (
        "docs/benchmarks.rst's generated index is stale; regenerate with\n"
        "    python tools/gen_benchmark_index.py\n\n" + proc.stdout[-3000:] + proc.stderr[-2000:]
    )


def test_helpers_agree_with_the_table(registry):
    """``by_path`` and ``by_data`` are views on LABELS, not a second copy."""
    for path in registry.PATH_NAMES:
        assert set(registry.by_path(path)) == {
            n for n, l in registry.LABELS.items() if path in l.paths
        }
    for kind in registry.DATA_KINDS:
        assert set(registry.by_data(kind)) == {
            n for n, l in registry.LABELS.items() if l.data == kind
        }


def test_comment_markers_do_not_contradict_the_table():
    """The drift mechanism, closed.

    Many registry comments still open with ``Path A:`` or ``Path B:``. The
    descriptions stay -- they say what a case does, which LABELS does not --
    but the marker inside them is a second copy of the label, and a second copy
    is what let the two surfaces disagree in the first place. The comment
    stays; contradicting the table does not.
    """
    from benchmarks import registry as reg

    src = (ROOT / "benchmarks" / "registry.py").read_text()
    entries = re.findall(
        r'^\s*"([a-z0-9_]+)":\s*"benchmarks\.cases\.[a-z0-9_]+",?\s*#(.*)$',
        src, re.M,
    )
    wrong = []
    for name, comment in entries:
        claimed = {m.upper() for m in re.findall(r"\bPath ([ABC])\b", comment)}
        if re.search(r"cross-val|cross-check", comment, re.I):
            claimed.add("X")
        label = reg.LABELS.get(name)
        if label is None:
            continue
        stray = claimed - set(label.paths)
        if stray:
            wrong.append(
                f"{name}: comment says {sorted(claimed)}, LABELS says "
                f"{sorted(label.paths)}"
            )
    assert not wrong, (
        "registry comments contradict LABELS (edit one or the other):\n  "
        + "\n  ".join(wrong)
    )
