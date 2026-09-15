#!/usr/bin/env python3
"""Write the coverage table into docs/replications.rst.

    python tools/gen_coverage_table.py            # rewrite the page
    python tools/gen_coverage_table.py --check    # exit 1 if it is out of date

The table sits between ``.. coverage-table-start`` and ``.. coverage-table-end``
in the page; everything outside the markers is left alone. The source of the
counts is tools/estimator_families.toml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.estimator_families import load_families, render_table  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs" / "replications.rst"
START = ".. coverage-table-start"
END = ".. coverage-table-end"


def rewrite(text: str, table: str) -> str:
    """Replace the marked block, or raise if the markers are missing."""
    try:
        start = text.index(START)
        end = text.index(END)
    except ValueError as exc:
        raise SystemExit(
            "%s: missing %s / %s markers" % (DOCS, START, END)
        ) from exc
    if end < start:
        raise SystemExit("%s: %s appears before %s" % (DOCS, END, START))
    return text[:start] + START + "\n\n" + table + "\n" + text[end:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the page is out of date",
    )
    args = ap.parse_args()

    data = load_families()
    text = DOCS.read_text()
    updated = rewrite(text, render_table(data))

    if args.check:
        if updated != text:
            print(
                "docs/replications.rst is out of date -- run "
                "python tools/gen_coverage_table.py",
                file=sys.stderr,
            )
            return 1
        print("docs/replications.rst is up to date")
        return 0

    if updated != text:
        DOCS.write_text(updated)
        print("rewrote the coverage table in docs/replications.rst")
    else:
        print("docs/replications.rst already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
