"""Registry of benchmark cases. Map a short name to its module."""
from __future__ import annotations

import importlib

# name -> "benchmarks.cases.<module>"  (pure-Python unless noted needs_reference)
CASES = {
    "fdid_table5": "benchmarks.cases.fdid_table5",      # Path B: simulation
    "fdid_hongkong": "benchmarks.cases.fdid_hongkong",  # Path A: HK GDP empirical
    # "synth_prop99": "benchmarks.cases.synth_prop99",   # needs R (cross-validation)
}

NEEDS_REFERENCE = set()   # names whose case reads an R reference output


def load(name: str):
    mod = importlib.import_module(CASES[name])
    return mod
