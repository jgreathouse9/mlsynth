"""Registry of benchmark cases. Map a short name to its module."""
from __future__ import annotations

import importlib

# name -> "benchmarks.cases.<module>"  (pure-Python unless noted needs_reference)
CASES = {
    "spcd_prop99": "benchmarks.cases.spcd_prop99",      # Path A: SPCD design vs random/SC on Prop 99 (Lu et al. 2022)
    "syndes_bls": "benchmarks.cases.syndes_bls",        # Path B: Doudchenko et al. 2021 Monte Carlo (BLS unemployment)
    "si_prop99": "benchmarks.cases.si_prop99",          # cross-val vs Agarwal-Shah-Shen 2026 authors' code (Prop 99)
    "snn_prop99": "benchmarks.cases.snn_prop99",        # cross-val vs deshen24/syntheticNN (Prop 99)
    "ppscm_paglayan": "benchmarks.cases.ppscm_paglayan",  # cross-val vs augsynth::multisynth (jackknife + bootstrap SEs)
    "rolldid_lw": "benchmarks.cases.rolldid_lw",        # Path A: Lee-Wooldridge Prop99 + castle
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
    "marex_walmart": "benchmarks.cases.marex_walmart",          # Path A: MAREX Walmart placebo design (Abadie-Zhao SCDesign, 10-store subset)
    "scmo_germany": "benchmarks.cases.scmo_germany",            # Path A: Tian et al. West Germany balance
    "scmo_concatenated_mc": "benchmarks.cases.scmo_concatenated_mc",  # Path B: Tian Table 1 / Sun Sim1
    "scmo_averaged_mc": "benchmarks.cases.scmo_averaged_mc",    # Path B: Sun averaged regime geometry
    "rescm_brexit": "benchmarks.cases.rescm_brexit",            # Path A: SCM-relaxation Brexit/UK GDP (2016Q3)
    "rescm_brexit_2020": "benchmarks.cases.rescm_brexit_2020",  # Path A: SCM-relaxation Brexit robustness (2020Q1)
    "rescm_relax_ref": "benchmarks.cases.rescm_relax_ref",      # cross-val vs scmrelax (skips if absent)
    "rescm_relax_mc": "benchmarks.cases.rescm_relax_mc",        # Path B: latent-group MC, relaxations beat SCM
    "linf_crossval_ref": "benchmarks.cases.linf_crossval_ref",  # cross-val: LINF vs LinfinitySC (skips if absent)
    "linf_prop99": "benchmarks.cases.linf_prop99",              # Path A: dense L-inf vs sparse SC (Prop 99)
    "linf_sim": "benchmarks.cases.linf_sim",                    # Path B: L-inf vs SC (Wang-Xing-Ye Table 4)
    "sparse_sc_prop99": "benchmarks.cases.sparse_sc_prop99",    # Path A: L1 predictor selection (Prop 99)
    "nsc_prop99": "benchmarks.cases.nsc_prop99",                # cross-val vs Tian's NSC.R (Prop 99 Table 2)
    "nsc_mc": "benchmarks.cases.nsc_mc",                        # Path B: nonlinear coverage + error-shrinks-with-J
    "vanillasc_prop99": "benchmarks.cases.vanillasc_prop99",  # Path A: canonical ADH 2010 Prop 99
    "cwz_ttest": "benchmarks.cases.cwz_ttest",                # Path A: CWZ 2025 Table 5 carbon-tax debiased t-test
    "cwz_mc": "benchmarks.cases.cwz_mc",                      # Path B: CWZ 2025 Table 3 application-based Monte Carlo
    "masc_basque": "benchmarks.cases.masc_basque",            # Path A: MASC Basque/ETA (KMPT Sec 5)
    "tasc_mc": "benchmarks.cases.tasc_mc",                    # Path B: TASC vs SC state-space ablation (Rho et al.)
    "fscm_prop99": "benchmarks.cases.fscm_prop99",            # Path A: forward-selected SC (Prop 99)
    "pda_hongkong": "benchmarks.cases.pda_hongkong",          # Path A: PDA methods on HK CEPA (Shi-Wang App E.1)
    "pda_table1": "benchmarks.cases.pda_table1",              # Path B: Shi-Huang Table 1 fs-vs-LASSO size/power geometry
    "pda_lasso_sim": "benchmarks.cases.pda_lasso_sim",        # Path B: Li-Bell Table 2 LASSO-PDA OOS prediction (N>T1)
    "pda_l2_sim": "benchmarks.cases.pda_l2_sim",              # Path B: Shi-Wang Table 2 L2-relaxation size/power
    "pda_luxurywatch": "benchmarks.cases.pda_luxurywatch",    # Path A: Shi-Huang China luxury-watch fsPDA (prewhitened-NW)
    "pda_ppi": "benchmarks.cases.pda_ppi",                    # Path A: Shi-Wang China PPI L2-relaxation (real-estate policy)
    "pda_brexit": "benchmarks.cases.pda_brexit",              # Path A: Shi-Wang Brexit multi-treated-units L2-relaxation
    "pda_pi_coverage": "benchmarks.cases.pda_pi_coverage",    # Path B: Jiang et al. 2025 prediction-interval coverage (Tables 2-5)
    "mlsc_bottmer": "benchmarks.cases.mlsc_bottmer",          # cross-val vs Bottmer's mlSC_estimator (skips if absent)
    "proximal_panic1907": "benchmarks.cases.proximal_panic1907",  # cross-val vs freshtaste/proximal (Panic 1907 Table 3)
    "spsc_ifem_mc": "benchmarks.cases.spsc_ifem_mc",          # Path B: SPSC IFEM recovery + DT-vs-NoDT coverage (Park-Tchetgen)
    "dr_proximal_mc": "benchmarks.cases.dr_proximal_mc",      # Path B: DR/PIPW recovery + double-robustness (Qiu et al. normal DGP)
    "proximal_surrogates_mc": "benchmarks.cases.proximal_surrogates_mc",  # Path B: PI/PIS/PIPost vs SC under trending factor (Liu et al.)
    "ssc_guanajuato": "benchmarks.cases.ssc_guanajuato",      # cross-val vs jcao0/staggered_synthetic_control (criminality Sec 4)
    "spillsynth_prop99": "benchmarks.cases.spillsynth_prop99",  # cross-val vs jcao0/synthetic-control-spillover (Cao-Dowd Prop 99)
    "spillsynth_iscm_germany": "benchmarks.cases.spillsynth_iscm_germany",  # Path A: inclusive SCM German reunification (Di Stefano-Mellace)
    "spillsynth_iscm_xval": "benchmarks.cases.spillsynth_iscm_xval",  # cross-val vs Melnychuk-Andrii/Spillover-SCM (inclusive SCM German)
    "spillsynth_grossi_germany": "benchmarks.cases.spillsynth_grossi_germany",  # Path A: grossi direct+spillover German reunification (Grossi et al.)
    "spillsynth_iterative_germany": "benchmarks.cases.spillsynth_iterative_germany",  # Path A: iterative waterfall SCM German reunification (Melnychuk)
    "spillsynth_sar_mc": "benchmarks.cases.spillsynth_sar_mc",  # Path B: SAR spillover recovery + SCM nesting (Sakaguchi-Tagawa)
    "spotsynth_real_data": "benchmarks.cases.spotsynth_real_data",  # SPOTSYNTH donor-spillover screening: Germany/California/Basque (Fig 6) + detection (Fig 2) + debias (Fig 4)
    "ctsc_powell_mc": "benchmarks.cases.ctsc_powell_mc",      # Path B: CTSC vs two-way FE bias (Powell 2022 Table 1)
    "siv_syria_mc": "benchmarks.cases.siv_syria_mc",          # Path B: SIV vs 2SLS-TWFE bias (Gulek-Vives Table 1)
    "fma_coverage_mc": "benchmarks.cases.fma_coverage_mc",      # Path B: FMA asymptotic-CI coverage robust to variance (Li-Sonnier)
    "pangeo_supergeo_mc": "benchmarks.cases.pangeo_supergeo_mc",  # Path B: PANGEO trajectory match vs scalar (Chen et al.)
    "shc_recovery_mc": "benchmarks.cases.shc_recovery_mc",      # Path B: SHC latent-confounder recovery (Chen-Yang-Yang Sec 3.1)
    "dscar_beijing": "benchmarks.cases.dscar_beijing",      # Path A: DSCAR Beijing PM2.5 alerts (Zheng-Chen)
    "msqrt_sim": "benchmarks.cases.msqrt_sim",                # Path B: MSQRT unbiasedness + RMSE noise-floor (Shen-Song-Abadie Sec 6)
    "dsc_dube": "benchmarks.cases.dsc_dube",                  # Path A: DSC distributional SC on Dube minimum-wage (Gunsilius/DiSCo vignette)
    "ascm_kansas": "benchmarks.cases.ascm_kansas",            # cross-val vs augsynth: Kansas ridge-ASCM ladder (SCM/ridge/covariate/residualized)
    "augsynth_calibrated": "benchmarks.cases.augsynth_calibrated",  # Path B: ASCM near-nominal coverage + bias reduction (BMR 2021 Sec 7)
    "geolift_walkthrough": "benchmarks.cases.geolift",  # cross-val vs GeoLift/augsynth: GeoLift_Walkthrough full summary, base + ridge-augmented (ATT/lift/incremental/L2/scaled-L2/%improve/bias/weights/conformal)
    "geolift_augsynth_ref": "benchmarks.cases.geolift_augsynth_ref",  # cross-val vs LIVE augsynth (Rscript): lambda/weights/ATT (skips if absent)
    "geolift_cpic": "benchmarks.cases.geolift_cpic",  # cross-val vs GeoLiftMarketSelection: CPIC investment value-for-value
    "geolift_marketselection": "benchmarks.cases.geolift_marketselection",  # cross-val vs GeoLiftMarketSelection: pooled N=2-5 BestMarkets top-5 ranking (rank/investment/MDE/abs_lift_in_zero)
    "geolift_marketselection_ref": "benchmarks.cases.geolift_marketselection_ref",  # cross-val vs LIVE GeoLiftMarketSelection (Rscript): BestMarkets top-5 (skips if absent)
    "geolift_multicell": "benchmarks.cases.geolift_multicell",  # cross-val vs augsynth: multi-cell per-cell ATT + donor exclusion
    "microsynth_seattle": "benchmarks.cases.microsynth_seattle",  # cross-val vs R microsynth panel method (Seattle DMI)
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
