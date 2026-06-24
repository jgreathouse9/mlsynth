"""Reference implementations and captured reference bundles for cross-validation
benchmarks.

Two complementary patterns live here:

- Reference *code* -- consumed directly when it ships under a permissive licence
  or as an installable package, or **cloned on demand** at a pinned commit when
  it is not redistributable (see ``clone_spsydid``). The Python case runs it live.

- Captured reference *bundles* -- a ``<case>/`` directory holding the exact
  reference script, its verbatim output, the parsed values the case pins
  against, and full provenance (tool versions, OS, git SHA, input-data
  checksums). A case pins its reference numbers by reading them via
  :func:`reference_value`, so the constant in ``EXPECTED`` and the captured run
  are the same object and cannot silently drift. Regenerate a bundle with
  ``python benchmarks/reference/generate.py <case>``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_REF_DIR = Path(__file__).resolve().parent


def load_reference(case: str) -> Dict[str, Any]:
    """Return the parsed ``reference.json`` for ``case`` (``{values, weights}``)."""
    path = _REF_DIR / case / "reference.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no reference bundle for '{case}'; generate it with "
            f"`python benchmarks/reference/generate.py {case}`"
        )
    return json.loads(path.read_text())


def reference_value(case: str, key: str) -> float:
    """Return a single pinned reference value ``case``/``key``."""
    return float(load_reference(case)["values"][key])
