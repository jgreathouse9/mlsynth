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
    "seq_sdid_mc": "benchmarks.cases.seq_sdid_mc",            # Path B: SSDiD vs DiD coverage/RMSE
    "clustersc_subgroups": "benchmarks.cases.clustersc_subgroups",      # Path B: ClusterSC vs RSC
    "clustersc_subgroups_ref": "benchmarks.cases.clustersc_subgroups_ref",  # cross-val vs authors' repo
    "clustersc_rpca_germany": "benchmarks.cases.clustersc_rpca_germany",  # Path A: RPCA-SC West Germany
    "tssc_brooklyn": "benchmarks.cases.tssc_brooklyn",        # Path A: Brooklyn showroom (Li-Shankar)
    "tssc_figure2": "benchmarks.cases.tssc_figure2",          # Path B: Figure 2 MSE-ratio grid
    "sbc_germany": "benchmarks.cases.sbc_germany",            # Path A: SBC German reunification
    "sbc_mc": "benchmarks.cases.sbc_mc",                      # Path B: Shi-Xi-Xie MSE ratios
    "hsc_hongkong": "benchmarks.cases.hsc_hongkong",          # Path A: HSC HK handover
    "hsc_mc": "benchmarks.cases.hsc_mc",                      # Path B: HSC regime adaptation
    "rsc_synth_error": "benchmarks.cases.rsc_synth_error",      # Path B: RSC train≈gen error
    "rsc_shen_coverage": "benchmarks.cases.rsc_shen_coverage",  # cross-val: Shen CIs + coverage
    "lexscm_walmart": "benchmarks.cases.lexscm_walmart",        # Path A: Walmart placebo design
    "lexscm_design_mc": "benchmarks.cases.lexscm_design_mc",    # Path B: Abadie-Zhao design sim
    "scmo_germany": "benchmarks.cases.scmo_germany",            # Path A: Tian et al. West Germany balance
    "scmo_concatenated_mc": "benchmarks.cases.scmo_concatenated_mc",  # Path B: Tian Table 1 / Sun Sim1
    "scmo_averaged_mc": "benchmarks.cases.scmo_averaged_mc",    # Path B: Sun averaged regime geometry
    "rescm_brexit": "benchmarks.cases.rescm_brexit",            # Path A: SCM-relaxation Brexit/UK GDP
    "rescm_relax_ref": "benchmarks.cases.rescm_relax_ref",      # cross-val vs scmrelax (skips if absent)
    "rescm_relax_mc": "benchmarks.cases.rescm_relax_mc",        # Path B: latent-group MC, relaxations beat SCM
    "linf_crossval_ref": "benchmarks.cases.linf_crossval_ref",  # cross-val: LINF vs LinfinitySC (skips if absent)
    "linf_prop99": "benchmarks.cases.linf_prop99",              # Path A: dense L-inf vs sparse SC (Prop 99)
    "linf_sim": "benchmarks.cases.linf_sim",                    # Path B: L-inf vs SC (Wang-Xing-Ye Table 4)
    "sparse_sc_prop99": "benchmarks.cases.sparse_sc_prop99",    # Path A: L1 predictor selection (Prop 99)
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
