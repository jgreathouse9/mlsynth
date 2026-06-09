"""Registry of benchmark cases. Map a short name to its module."""
from __future__ import annotations

import importlib

# name -> "benchmarks.cases.<module>"  (pure-Python unless noted needs_reference)
CASES = {
    "fdid_table5": "benchmarks.cases.fdid_table5",      # Path B: simulation
    "fdid_hongkong": "benchmarks.cases.fdid_hongkong",  # Path A: HK GDP empirical
    "sdid_prop99": "benchmarks.cases.sdid_prop99",      # cross-val vs causaltensor
    "mcnnm_prop99": "benchmarks.cases.mcnnm_prop99",    # cross-val vs causaltensor
    "spsydid_state_mc": "benchmarks.cases.spsydid_state_mc",  # cross-val vs authors' repo
    "clustersc_subgroups": "benchmarks.cases.clustersc_subgroups",      # Path B: ClusterSC vs RSC
    "clustersc_subgroups_ref": "benchmarks.cases.clustersc_subgroups_ref",  # cross-val vs authors' repo
    "clustersc_rpca_germany": "benchmarks.cases.clustersc_rpca_germany",  # Path A: RPCA-SC West Germany
    "rsc_synth_error": "benchmarks.cases.rsc_synth_error",      # Path B: RSC train≈gen error
    "rsc_shen_coverage": "benchmarks.cases.rsc_shen_coverage",  # cross-val: Shen CIs + coverage
    "lexscm_walmart": "benchmarks.cases.lexscm_walmart",        # Path A: Walmart placebo design
    "lexscm_design_mc": "benchmarks.cases.lexscm_design_mc",    # Path B: Abadie-Zhao design sim
    # "synth_prop99": "benchmarks.cases.synth_prop99",   # needs R (cross-validation)
}

# Names whose case reads an external R/MATLAB reference *dump*. The cross-checks
# against the pip-installable ``causaltensor`` and the on-demand SpSyDiD clone are
# NOT listed here: they run under the default ``--all`` and skip themselves
# (BenchmarkSkipped) when their optional dependency is absent.
NEEDS_REFERENCE = set()


def load(name: str):
    mod = importlib.import_module(CASES[name])
    return mod
