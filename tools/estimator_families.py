"""Build the estimator-family coverage and render its table.

The table on ``docs/replications.rst`` reports, per family, how many estimators
mlsynth ships and how many carry a replication. Both counts were maintained by
hand and both went 42 estimators out of date without anything failing. They are
now derived from :mod:`tools.estimator_families_data`, and
``tools/gen_coverage_table.py`` writes the result into the page between its two
markers.

Building refuses data that does not describe the library: an unknown family, a
duplicate family key, or an unverified estimator that was never assigned all
raise, so bad data fails where it is read and not several counts later.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from tools.estimator_families_data import ESTIMATORS, FAMILIES, UNVERIFIED

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


def build_coverage(
    families: Sequence[Tuple[str, str]],
    assignments: Mapping[str, str],
    unverified: Mapping[str, str],
) -> Coverage:
    """Validate the three pieces and bind them together, or raise saying why.

    Kept separate from :func:`load_families` so the refusals can be exercised
    on inputs built in a test, with no file to write first.
    """
    defined: List[Family] = []
    seen = set()
    for key, title in families:
        if key in seen:
            raise ValueError("duplicate family key %r" % key)
        seen.add(key)
        defined.append(Family(key=key, title=title))

    unknown = {f for f in assignments.values() if f not in seen}
    if unknown:
        raise ValueError(
            "estimators assigned to undefined families: %s" % sorted(unknown)
        )

    stray = set(unverified) - set(assignments)
    if stray:
        raise ValueError(
            "unverified names are not assigned a family: %s" % sorted(stray)
        )

    return Coverage(
        families=defined,
        assignments=dict(assignments),
        unverified=dict(unverified),
    )


def load_families() -> Coverage:
    """The library's own coverage, from :mod:`tools.estimator_families_data`."""
    return build_coverage(FAMILIES, ESTIMATORS, UNVERIFIED)


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
