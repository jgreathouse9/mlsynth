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
    "ppscm_paglayan_covs": "benchmarks.cases.ppscm_paglayan_covs",  # cross-val vs augsynth::multisynth Sec 5.2 (auxiliary covariates)
    "rolldid_lw": "benchmarks.cases.rolldid_lw",        # Path A: Lee-Wooldridge Prop99 + castle
    "fdid_table5": "benchmarks.cases.fdid_table5",      # Path B: simulation
    "fdid_hongkong": "benchmarks.cases.fdid_hongkong",  # Path A: HK GDP empirical
    "cfm": "benchmarks.cases.cfm",                      # Path A: Bai-Wang 2026 Prop99 + German reunification
    "cscipca_mc": "benchmarks.cases.cscipca_mc",        # Path B: CSC-IPCA bias shrinks in observed-covariate share + beats extrapolating SC (Wang 2024 eq-13 DGP)
    "cscipca_brexit": "benchmarks.cases.cscipca_brexit",  # Path A: Wang 2024 Brexit->UK FDI, per-year ATT (2017/18/19 = -7.8/-12.9/-18.3)
    "medsc_prop99": "benchmarks.cases.medsc_prop99",    # Path A: Mellace-Pasquini 2022 Prop99 mediation, cross-world direct effect (1995/2000 = -16.8/-18.0) + negative growing indirect price channel
    "sdid_prop99": "benchmarks.cases.sdid_prop99",      # cross-val vs authors' synthdid R (Prop 99)
    "mcnnm_prop99": "benchmarks.cases.mcnnm_prop99",    # cross-val vs authors' MCPanel R (Prop 99)
    "spsydid_state_mc": "benchmarks.cases.spsydid_state_mc",  # cross-val vs authors' repo
    "spsydid_lawa_diff": "benchmarks.cases.spsydid_lawa_diff",  # differential cross-val vs authors' functions_ssdid on the real Arizona LAWA CPS panel (SpSyDiD.fit() ATT + spillover agree to solver tolerance under canonical convention)
    "seq_sdid_mc": "benchmarks.cases.seq_sdid_mc",            # Path B: SSDiD vs DiD coverage/RMSE
    "clustersc_subgroups": "benchmarks.cases.clustersc_subgroups",      # Path B: ClusterSC vs RSC
    "clustersc_subgroups_ref": "benchmarks.cases.clustersc_subgroups_ref",  # cross-val vs authors' repo
    "clustersc_rpca_germany": "benchmarks.cases.clustersc_rpca_germany",  # cross-val vs Bayani's RPCA-SC code (West Germany reunification, value-for-value)
    "tssc_brooklyn": "benchmarks.cases.tssc_brooklyn",        # Path A: Brooklyn showroom (Li-Shankar)
    "tssc_figure2": "benchmarks.cases.tssc_figure2",          # Path B: Figure 2 MSE-ratio grid
    "sbc_germany": "benchmarks.cases.sbc_germany",            # Path A: SBC German reunification
    "sbc_hongkong": "benchmarks.cases.sbc_hongkong",          # cross-val vs authors' SBC_HK.R (HK handover): detrend exact, mlsynth cyclical SSE < ipop
    "sbc_mc": "benchmarks.cases.sbc_mc",                      # Path B: Shi-Xi-Xie MSE ratios
    "hsc_hongkong": "benchmarks.cases.hsc_hongkong",          # Path A: HSC HK handover
    "hsc_mc": "benchmarks.cases.hsc_mc",                      # Path B: HSC regime adaptation
    "rsc_synth_error": "benchmarks.cases.rsc_synth_error",      # Path B: RSC train≈gen error
    "rsc_shen_coverage": "benchmarks.cases.rsc_shen_coverage",  # cross-val: Shen CIs + coverage
    "pcr_rsc_ref": "benchmarks.cases.pcr_rsc_ref",              # cross-val: mlsynth PCR vs original RSC (jehangiramjad/tslib, Prop 99)
    "lexscm_walmart": "benchmarks.cases.lexscm_walmart",        # Path A: Walmart placebo design
    "lexscm_design_mc": "benchmarks.cases.lexscm_design_mc",    # Path B: Abadie-Zhao design sim
    "marex_walmart": "benchmarks.cases.marex_walmart",          # Path A: MAREX Walmart placebo design (Abadie-Zhao SCDesign, 10-store subset)
    "scmo_germany": "benchmarks.cases.scmo_germany",            # Path A: Tian et al. West Germany balance
    "scmo_concatenated_mc": "benchmarks.cases.scmo_concatenated_mc",  # Path B: Tian Table 1 / Sun Sim1
    "scmo_averaged_mc": "benchmarks.cases.scmo_averaged_mc",    # Path B: Sun averaged regime geometry
    "rescm_brexit": "benchmarks.cases.rescm_brexit",            # Path A: SCM-relaxation Brexit/UK GDP (2016Q3)
    "rescm_brexit_2020": "benchmarks.cases.rescm_brexit_2020",  # Path A: SCM-relaxation Brexit robustness (2020Q1)
    "rescm_relax_ref": "benchmarks.cases.rescm_relax_ref",      # cross-val vs scmrelax toy panel (skips if absent)
    "rescm_balanced_gdp": "benchmarks.cases.rescm_balanced_gdp",  # cross-val vs scmrelax on authors' balanced-GDP Brexit panel (UK 2016Q3; skips if absent)
    "rescm_relax_mc": "benchmarks.cases.rescm_relax_mc",        # Path B: latent-group MC, relaxations beat SCM
    "linf_crossval_ref": "benchmarks.cases.linf_crossval_ref",  # cross-val: LINF vs LinfinitySC (skips if absent)
    "linf_prop99": "benchmarks.cases.linf_prop99",              # Path A: dense L-inf vs sparse SC (Prop 99)
    "linf_sim": "benchmarks.cases.linf_sim",                    # Path B: L-inf vs SC (Wang-Xing-Ye Table 4)
    "sparse_sc_prop99": "benchmarks.cases.sparse_sc_prop99",    # Path A: L1 predictor selection (Prop 99)
    "nsc_prop99": "benchmarks.cases.nsc_prop99",                # cross-val vs Tian's NSC.R (Prop 99 Table 2)
    "nsc_mc": "benchmarks.cases.nsc_mc",                        # Path B: nonlinear coverage + error-shrinks-with-J
    "vanillasc_prop99": "benchmarks.cases.vanillasc_prop99",  # Path A: canonical ADH 2010 Prop 99
    "lto_refined_placebo": "benchmarks.cases.lto_refined_placebo",  # cross-val vs authors' LTO code (Sudijono-Lei): leave-two-out refined placebo p-value on Prop 99 + West Germany + Basque, value-for-value
    "cwz_ttest": "benchmarks.cases.cwz_ttest",                # Path A: CWZ 2025 Table 5 carbon-tax debiased t-test
    "cwz_mc": "benchmarks.cases.cwz_mc",                      # Path B: CWZ 2025 Table 3 application-based Monte Carlo
    "masc_basque": "benchmarks.cases.masc_basque",            # Path A: MASC Basque/ETA (KMPT Sec 5)
    "masc_crossval": "benchmarks.cases.masc_crossval",        # cross-val vs authors' own R MASC (maxkllgg/masc, nogurobi) on Basque, value-for-value
    "src_basque": "benchmarks.cases.src_basque",              # cross-val vs R Code_SMC + Path A: SRC Basque/ETA (Zhu 2023)
    "bscm_china_watches": "benchmarks.cases.bscm_china_watches",  # cross-val vs reference Stan horseshoe + FSPDA (Shi-Huang) on China anti-corruption watches, p>n (Kim-Lee-Gupta 2020)
    "bvss_watches": "benchmarks.cases.bvss_watches",              # cross-val vs authors' own two-coordinate Gibbs (Xu-Zhou 2025) on China anti-corruption watches, p>n: engine exact + posterior ATT within MC error
    "bfsc_germany": "benchmarks.cases.bfsc_germany",                # cross-val vs author appendix Stan (corr 0.999999) + Path A: West Germany reunification (Pinkney 2021)
    "bfsc_prop99": "benchmarks.cases.bfsc_prop99",                  # cross-val vs author appendix Stan run LIVE via rstan (Prop 99, California 1989) -- needs [bayes] + rstan
    "mscmt_basque": "benchmarks.cases.mscmt_basque",          # cross-val vs R MSCMT: AG Basque, fit_window=(1960,1969)
    "malo_prop99": "benchmarks.cases.malo_prop99",            # Path A: Malo et al. 2024 Table 1 bilevel optimum (Prop 99)
    "malo_basque": "benchmarks.cases.malo_basque",            # cross-val vs scm.corner: AG Basque bilevel optimum, beats MSCMT
    "tasc_mc": "benchmarks.cases.tasc_mc",                    # Path B: TASC vs SC state-space ablation (Rho et al.)
    "tasc_prop99": "benchmarks.cases.tasc_prop99",            # cross-val vs authors' TimeAwareSC (srho1/tasc) on Prop 99 (d=2)
    "fscm_prop99": "benchmarks.cases.fscm_prop99",            # Path A: forward-selected SC (Prop 99)
    "pda_hongkong": "benchmarks.cases.pda_hongkong",          # Path A: PDA methods on HK CEPA (Shi-Wang App E.1)
    "pda_hcw_hongkong": "benchmarks.cases.pda_hcw_hongkong",  # Path A: original HCW best-subset on HK sovereignty (Table XVI/XVII, vs pampe)
    "pda_table1": "benchmarks.cases.pda_table1",              # Path B: Shi-Huang Table 1 fs-vs-LASSO size/power geometry
    "pda_lasso_sim": "benchmarks.cases.pda_lasso_sim",        # Path B: Li-Bell Table 2 LASSO-PDA OOS prediction (N>T1)
    "pda_l2_sim": "benchmarks.cases.pda_l2_sim",              # Path B: Shi-Wang Table 2 L2-relaxation size/power
    "pda_luxurywatch": "benchmarks.cases.pda_luxurywatch",    # Path A: Shi-Huang China luxury-watch fsPDA (prewhitened-NW)
    "pda_ppi": "benchmarks.cases.pda_ppi",                    # Path A: Shi-Wang China PPI L2-relaxation (real-estate policy)
    "pda_brexit": "benchmarks.cases.pda_brexit",              # Path A: Shi-Wang Brexit multi-treated-units L2-relaxation
    "pda_pi_coverage": "benchmarks.cases.pda_pi_coverage",    # Path B: Jiang et al. 2025 prediction-interval coverage (Tables 2-5)
    "mlsc_bottmer": "benchmarks.cases.mlsc_bottmer",          # cross-val vs Bottmer's mlSC_estimator (skips if absent)
    "proximal_panic1907": "benchmarks.cases.proximal_panic1907",  # cross-val vs freshtaste/proximal (Panic 1907 Table 3)
    "proximal_germany_oid": "benchmarks.cases.proximal_germany_oid",  # cross-val vs authors' manuscript code (Shi et al. 2026 JASA): over-identified PI (PIOID) on German reunification, ATT + GMM CI value-for-value
    "proximal_oid_mc": "benchmarks.cases.proximal_oid_mc",    # Path B: PIOID recovery + coverage + beats naive SC under error-in-variables donors (Shi et al. 2026 linear DGP; M=15k array core + full-.fit() equivalence guard)
    "pioid_overid_jtest": "benchmarks.cases.pioid_overid_jtest",  # Path B: PIOID Hansen J over-id test size+power on the authors' linear IFEM DGP (shixu0830/SyntheticControl)
    "spsc_ifem_mc": "benchmarks.cases.spsc_ifem_mc",          # Path B: SPSC IFEM recovery + DT-vs-NoDT coverage (Park-Tchetgen)
    "spsc_prop99": "benchmarks.cases.spsc_prop99",            # Path A/X: SPSC California (Prop 99) linear effect path vs qkrcks0218/SPSC
    "spsc_panic": "benchmarks.cases.spsc_panic",              # Path A/X: SPSC averaged-treated Panic of 1907 vs qkrcks0218/SPSC
    "dpsc_prop99": "benchmarks.cases.dpsc_prop99",            # cross-val vs srho1/dpsc (differentially private SC, Prop 99, bit-for-bit both mechanisms)
    "scd_cps": "benchmarks.cases.scd_cps",                    # cross-val vs base-R SCD reference (Rincon-Song 2026): Arizona LAWA CPS weights + effect path + RC SE + confidence set, value-for-value
    "scul_prop99": "benchmarks.cases.scul_prop99",            # Path A/X: SCUL California (Prop 99) lasso SC vs hollina/scul
    "dr_proximal_mc": "benchmarks.cases.dr_proximal_mc",      # Path B: DR/PIPW recovery + double-robustness (Qiu et al. normal DGP)
    "dr_proximal_brazil": "benchmarks.cases.dr_proximal_brazil",  # cross-val vs LIVE R (authors' analysis.Rmd commit 3bcb5ec): over-identified DR-OID, Brazil vaccine/pneumonia
    "brazil_vaccine_scm_vs_proximal": "benchmarks.cases.brazil_vaccine_scm_vs_proximal",  # cross-val vs LIVE R: standard SC (VanillaSC) vs proximal (DR-OID h/DR), Brazil vaccine/pneumonia contrast
    "proximal_surrogates_mc": "benchmarks.cases.proximal_surrogates_mc",  # Path B: PI/PIS/PIPost vs SC under trending factor (Liu et al.)
    "ssc_guanajuato": "benchmarks.cases.ssc_guanajuato",      # cross-val vs jcao0/staggered_synthetic_control (criminality Sec 4)
    "spillsynth_prop99": "benchmarks.cases.spillsynth_prop99",  # cross-val vs jcao0/synthetic-control-spillover (Cao-Dowd Prop 99)
    "spillsynth_iscm_germany": "benchmarks.cases.spillsynth_iscm_germany",  # Path A: inclusive SCM German reunification (Di Stefano-Mellace)
    "spillsynth_iscm_xval": "benchmarks.cases.spillsynth_iscm_xval",  # cross-val vs Melnychuk-Andrii/Spillover-SCM (inclusive SCM German)
    "spillsynth_grossi_germany": "benchmarks.cases.spillsynth_grossi_germany",  # Path A: grossi direct+spillover German reunification (Grossi et al.)
    "spillsynth_iterative_germany": "benchmarks.cases.spillsynth_iterative_germany",  # Path A: iterative waterfall SCM German reunification (Melnychuk)
    "spillsynth_sar_mc": "benchmarks.cases.spillsynth_sar_mc",  # Path B: SAR spillover recovery + SCM nesting (Sakaguchi-Tagawa)
    "spillsynth_prop99_sar": "benchmarks.cases.spillsynth_prop99_sar",  # cross-val vs Mendez/Sakaguchi-Tagawa California Prop 99 SAR tutorial (bare rho 4dp + ATT + Nevada spillover; full rho weakly identified)
    "spillsynth_sudan": "benchmarks.cases.spillsynth_sudan",  # cross-val vs Sakaguchi-Tagawa Rcpp SAR (2011 Sudan secession, empirical)
    "spotsynth_real_data": "benchmarks.cases.spotsynth_real_data",  # SPOTSYNTH donor-spillover screening: Germany/California/Basque (Fig 6) + detection (Fig 2) + debias (Fig 4)
    "spotsynth_panic1907": "benchmarks.cases.spotsynth_panic1907",  # cross-method: SPOTSYNTH debias vs PROXIMAL PI on Panic 1907 + TCA screen + systemic-shock limit
    "ctsc_powell_mc": "benchmarks.cases.ctsc_powell_mc",      # Path B: CTSC vs two-way FE bias (Powell 2022 Table 1)
    "siv_syria_mc": "benchmarks.cases.siv_syria_mc",          # Path B: SIV vs 2SLS-TWFE bias (Gulek-Vives Table 1)
    "orthsc_carbontax": "benchmarks.cases.orthsc_carbontax",  # Path A: ORTHSC Fry carbon-tax ATT/p/K/CI (Andersson 2019 data, vs live R)
    "orthsc_size_power": "benchmarks.cases.orthsc_size_power",  # Path B: ORTHSC fixed-smoothing t-test size control + power (Fry Tables 1-2)
    "th_prop99": "benchmarks.cases.th_prop99",  # Path A: Spoelstra et al. 2025 Table 1 left-TH SDID (Prop 99)
    "gmmsce_carbontax": "benchmarks.cases.gmmsce_carbontax",  # cross-val vs Fry GMM-SCE.R GMMSC (carbon tax, J-statistic + optimality)
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
    "pensynth_prop99": "benchmarks.cases.pensynth_prop99",  # cross-val vs LIVE pensynth wsoll1 (Rscript+LowRankQP) on Prop 99 penalized SC (skips if absent)
    "geolift_cpic": "benchmarks.cases.geolift_cpic",  # cross-val vs GeoLiftMarketSelection: CPIC investment value-for-value
    "geolift_marketselection": "benchmarks.cases.geolift_marketselection",  # cross-val vs GeoLiftMarketSelection: pooled N=2-5 BestMarkets top-5 ranking (rank/investment/MDE/abs_lift_in_zero)
    "geolift_marketselection_ref": "benchmarks.cases.geolift_marketselection_ref",  # cross-val vs LIVE GeoLiftMarketSelection (Rscript): BestMarkets top-5 (skips if absent)
    "geolift_multicell": "benchmarks.cases.geolift_multicell",  # cross-val vs augsynth: multi-cell per-cell ATT + donor exclusion
    "microsynth_seattle": "benchmarks.cases.microsynth_seattle",  # cross-val vs R microsynth panel method (Seattle DMI)
    "scpi_staggered": "benchmarks.cases.scpi_staggered",  # cross-val vs scpi: staggered point estimates (Germany)
    "scpi_staggered_pi": "benchmarks.cases.scpi_staggered_pi",  # cross-val vs scpi: staggered TSUA prediction intervals (Germany)
    "scpi_staggered_covariate": "benchmarks.cases.scpi_staggered_covariate",  # cross-val vs scpi: covariate (multi-feature) staggered illustration (Germany)
    "scpi_germany_pi": "benchmarks.cases.scpi_germany_pi",  # cross-val vs scpi: single-unit CFT-2021 prediction intervals, levels + cointegrated (German reunification)
    "scpi_ridge_germany": "benchmarks.cases.scpi_ridge_germany",  # cross-val vs scpi: ridge-constraint Q/lambda/df via CLUSTERSC RSC (.fit()), Amjad et al. 2018 (German reunification)
    "vanillasc_carbontax": "benchmarks.cases.vanillasc_carbontax",
    "beast_prop99": "benchmarks.cases.beast_prop99",  # cross-val vs authors R (jeremylhour): BEAST immunized ATT path on Prop 99 (basic covariate regime)
    "eiv_coverage_mc": "benchmarks.cases.eiv_coverage_mc",  # Path B: Hirshberg 2021 error-in-variables SC interval coverage (low-rank DGP)  # Path A: Andersson 2019 Swedish carbon tax ATT/2005-gap, malo + mscmt backends (paper predictor spec)
    "synth_prop99": "benchmarks.cases.synth_prop99",   # cross-val vs original R Synth solver (Prop 99 outcome-only); skips if R/Synth absent
    "cmbsts_vignette": "benchmarks.cases.cmbsts_vignette",  # cross-val vs R CausalMBSTS: multivariate BSTS vignette (trend+cycle)
    "cmbsts_supermarket": "benchmarks.cases.cmbsts_supermarket",  # Path A + cross-val vs R CausalMBSTS: Menchetti-Bojinov Table 3 (1-month horizon, pairs 4/7/10)
    "propsc_spain": "benchmarks.cases.propsc_spain",  # Path A + cross-val vs R propsdid: Bogatyrev-Stoetzer Table 2 (common-weights SDID, party vote shares sum to zero)
}

# Names whose case reads an external R/MATLAB reference *dump*. Cross-checks that
# clone a reference on demand (e.g. the SpSyDiD clone) are NOT listed here: they
# run under the default ``--all`` and skip themselves (BenchmarkSkipped) when
# their optional dependency is absent.
NEEDS_REFERENCE = set()


def load(name: str):
    mod = importlib.import_module(CASES[name])
    return mod
