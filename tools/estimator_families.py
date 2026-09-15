"""Read ``tools/estimator_families.toml`` and render the coverage table.

The table on ``docs/replications.rst`` reports, per family, how many estimators
mlsynth ships and how many carry a replication. Both counts were maintained by
hand and both went 42 estimators out of date without anything failing. They are
now derived from the data file, and ``tools/gen_coverage_table.py`` writes the
result into the page between its two markers.

Loading refuses a file that does not describe the library: an unknown family, a
duplicate family key, or an unverified estimator that was never assigned all
raise, so a malformed file fails where it is read and not several counts later.
"""
from __future__ import annotations

import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

DATA_FILE = Path(__file__).resolve().parent / "estimator_families.toml"

#: Family keys in the order their rows appear in the table.
FAMILY_ORDER = [
    "canonical",
    "decomp",
    "generalised",
    "hull",
    "highdim",
    "time",
    "bayesian",
    "staggered",
    "spillover",
    "missing",
    "endogeneity",
    "compositional",
    "nocontrols",
    "inference",
    "randomized",
    "forecasting",
    "privacy",
    "design",
]


@dataclass(frozen=True)
class Family:
    key: str
    title: str


@dataclass(frozen=True)
class Coverage:
    families: List[Family]
    assignments: Dict[str, str]
    unverified: Dict[str, str]


@dataclass(frozen=True)
class Row:
    family: Family
    verified: int
    in_family: int
    members: List[str]
    unverified: List[str]


def load_families(path: Path | None = None) -> Coverage:
    """Parse the data file, or raise saying what is wrong with it."""
    path = Path(path) if path is not None else DATA_FILE
    if not path.exists():
        raise FileNotFoundError(path)
    raw = tomllib.loads(path.read_text())

    families: List[Family] = []
    seen = set()
    for entry in raw.get("family", []):
        key = entry["key"]
        if key in seen:
            raise ValueError("duplicate family key %r in %s" % (key, path))
        seen.add(key)
        families.append(Family(key=key, title=entry["title"]))

    assignments: Mapping[str, str] = raw.get("estimators", {})
    unknown = {f for f in assignments.values() if f not in seen}
    if unknown:
        raise ValueError(
            "estimators assigned to undefined families %s in %s"
            % (sorted(unknown), path)
        )

    unverified: Mapping[str, str] = raw.get("unverified", {})
    stray = set(unverified) - set(assignments)
    if stray:
        raise ValueError(
            "unverified names %s are not assigned a family in %s"
            % (sorted(stray), path)
        )

    return Coverage(
        families=families,
        assignments=dict(assignments),
        unverified=dict(unverified),
    )


def coverage_rows(data: Coverage) -> List[Row]:
    """One row per family, in ``FAMILY_ORDER``, with its members counted."""
    rows = []
    for family in data.families:
        members = sorted(k for k, v in data.assignments.items() if v == family.key)
        bad = sorted(m for m in members if m in data.unverified)
        rows.append(
            Row(
                family=family,
                verified=len(members) - len(bad),
                in_family=len(members),
                members=members,
                unverified=bad,
            )
        )
    return rows


#: Cell text wraps at this width, so the generated page stays as readable in a
#: diff as the hand-written table it replaced.
CELL_WIDTH = 62


def _wrap(text: str) -> List[str]:
    # break_on_hyphens would split "non-public" across two table rows.
    return textwrap.wrap(text, width=CELL_WIDTH, break_on_hyphens=False) or [""]


def _status(row: Row, reasons: Mapping[str, str]) -> List[str]:
    """The status cell: what is covered, and what is not, wrapped to width."""
    if not row.unverified:
        return _wrap("Complete (%s)" % ", ".join(row.members))
    covered = [m for m in row.members if m not in row.unverified]
    lines = []
    if covered:
        lines += _wrap("%s verified;" % ", ".join(covered))
    for name in row.unverified:
        lines += _wrap("%s -- %s" % (name, reasons[name]))
    return lines


def render_table(data: Coverage) -> str:
    """Render the coverage table as an RST list-table."""
    rows = coverage_rows(data)

    out = [
        ".. list-table:: Verification coverage by family",
        "   :header-rows: 1",
        "   :widths: 26 10 10 54",
        "",
        "   * - Family",
        "     - Verified",
        "     - In family",
        "     - Status",
    ]
    for row in rows:
        out.append("   * - %s" % row.family.title)
        out.append("     - %d" % row.verified)
        out.append("     - %d" % row.in_family)
        status = _status(row, data.unverified)
        out.append("     - %s" % status[0])
        for extra in status[1:]:
            out.append("       %s" % extra)

    total_in = sum(r.in_family for r in rows)
    total_ok = sum(r.verified for r in rows)
    out += [
        "   * - Total",
        "     - %d" % total_ok,
        "     - %d" % total_in,
        "     - %s"
        % (
            "Every estimator carries a replication."
            if total_ok == total_in
            else "%d of %d estimators carry a replication."
            % (total_ok, total_in)
        ),
    ]
    return "\n".join(out) + "\n"
