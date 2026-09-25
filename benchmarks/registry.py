"""Registry of benchmark cases. Map a short name to its module."""
from __future__ import annotations

import importlib
from typing import NamedTuple

# name -> "benchmarks.cases.<module>"  (pure-Python unless noted needs_reference)
CASES = {
    "spcd_prop99": "benchmarks.cases.spcd_prop99",      # Path A: SPCD design vs random/SC on Prop 99 (Lu et al. 2022)
    "syndes_bls": "benchmarks.cases.syndes_bls",        # Path B: Doudchenko et al. 2021 Monte Carlo (BLS unemployment)
    "syndes_exact_vs_mip": "benchmarks.cases.syndes_exact_vs_mip",  # solver cross-check: the two-way treated-set search vs SCIP proving optimality on the BLS panel
    "si_prop99": "benchmarks.cases.si_prop99",          # cross-val vs Agarwal-Shah-Shen 2026 authors' code (Prop 99)
    "snn_prop99": "benchmarks.cases.snn_prop99",        # cross-val vs deshen24/syntheticNN (Prop 99)
    "ppscm_paglayan": "benchmarks.cases.ppscm_paglayan",  # cross-val vs augsynth::multisynth (jackknife + bootstrap SEs)
    "ppscm_paglayan_covs": "benchmarks.cases.ppscm_paglayan_covs",  # cross-val vs augsynth::multisynth Sec 5.2 (auxiliary covariates)
    "ppscm_cs_real_panels": "benchmarks.cases.ppscm_cs_real_panels",  # cross-val vs diff-diff CallawaySantAnna on mpdta / castle_doctrine / walmart
    "ronczewski_cannabis": "benchmarks.cases.ronczewski_cannabis",  # Path A: cannabis/alcohol, PPSCM+SDID+GSYNTH vs augsynth/did/synthdid/gsynth
    "ppscm_bfr_mc": "benchmarks.cases.ppscm_bfr_mc",  # Path B: BFR sharp-null designs (ATT coverage both methods + cumulative band)
    "rolldid_lw": "benchmarks.cases.rolldid_lw",        # Path A: Lee-Wooldridge Prop99 + castle
    "fdid_table5": "benchmarks.cases.fdid_table5",      # Path B: simulation
    "fdid_hongkong": "benchmarks.cases.fdid_hongkong",  # Path A: HK GDP empirical
    "fdid_selection_mc": "benchmarks.cases.fdid_selection_mc",  # Path C: forward-selection consistency (Li 2023 Prop 2.2 / D.1) -- Pr(U_hat = U*) climbs 0.00 -> 0.77 over T1 = 25 -> 1600, plus Lemma B.1's uniform sqrt(log N / T1) rate over all 2^N - 1 subsets
    "fdid_normality_mc": "benchmarks.cases.fdid_normality_mc",  # Path C: ATT asymptotic normality (Li 2023 Prop 2.1) -- dispersion falls to 1 and coverage climbs to nominal where Assumption 4 holds; where 4(ii) fails the paper's statistic settles at sqrt(1 + T2/T1) and mlsynth's finite-sample SE converges anyway
    "fdid_serial_correlation_mc": "benchmarks.cases.fdid_serial_correlation_mc",  # Path C: where Li 2023 Prop 2.1 stops applying -- the SE uses the marginal residual variance where a block mean needs the long-run variance, so under an AR(1) residual 95% coverage falls 0.94 -> 0.53 as rho goes 0 -> 0.9, and a closed-form long-run-variance prediction accounts for the whole 2.79x dispersion blow-up; inference="hac" prices the autocovariances in and holds coverage at 0.92-0.95 across the same range
    "twsf_coverage_mc": "benchmarks.cases.twsf_coverage_mc",  # Path B: Shen 2026 TWSF section 7.1 -- sigma=0 exact recovery, plug-in variance calibration, and coverage at nominal 90%; also pins the spectral reason coverage falls short on small panels
    "cfm": "benchmarks.cases.cfm",                      # Path A: Bai-Wang 2026 Prop99 + German reunification
    "cscipca_mc": "benchmarks.cases.cscipca_mc",        # Path B: CSC-IPCA bias shrinks in observed-covariate share + beats extrapolating SC (Wang 2024 eq-13 DGP)
    "cscipca_brexit": "benchmarks.cases.cscipca_brexit",  # Path A: Wang 2024 Brexit->UK FDI, per-year ATT (2017/18/19 = -7.8/-12.9/-18.3)
    "medsc_prop99": "benchmarks.cases.medsc_prop99",    # Path A: Mellace-Pasquini 2022 Prop99 mediation, cross-world direct effect (1995/2000 = -16.8/-18.0) + negative growing indirect price channel
    "sdid_prop99": "benchmarks.cases.sdid_prop99",
    "sdid_euets": "benchmarks.cases.sdid_euets",  # Path A: Basaglia-Grunau-Drupp 2024 PNAS EU ETS co-benefits, the SDID robustness half -- three pollutants reproduce the authors' Stata sdid log to 3e-4 once the covariate row rule matches; records that Stata's projected fits beta on never-treated units only while Kranz (and mlsynth) use every untreated row      # cross-val vs authors' synthdid R (Prop 99)
    "sdid_ddd_hpv": "benchmarks.cases.sdid_ddd_hpv",    # Path A: SDID synthetic triple difference (Zhuang 2024) on Virginia HPV mandate (Feldman-Semprini 2026); SC-DDD +1.559 / naive SC-DD +0.252 vs Stata sdid
    "mcnnm_prop99": "benchmarks.cases.mcnnm_prop99",    # cross-val vs authors' MCPanel R (Prop 99)
    "lpca_kansas": "benchmarks.cases.lpca_kansas",      # cross-val vs Feng's own R (Kansas tax cut)
    "lpca_mc": "benchmarks.cases.lpca_mc",              # cross-val vs Feng's own R (Section 5 designs)
    "spsydid_state_mc": "benchmarks.cases.spsydid_state_mc",  # cross-val vs authors' repo
    "spsydid_lawa_diff": "benchmarks.cases.spsydid_lawa_diff",  # differential cross-val vs authors' functions_ssdid on the real Arizona LAWA CPS panel (SpSyDiD.fit() ATT + spillover agree to solver tolerance under canonical convention)
    "seq_sdid_mc": "benchmarks.cases.seq_sdid_mc",
    "geox_augsynth_geolift": "benchmarks.cases.geox_augsynth_geolift",  # cross-val vs R GeoLiftMarketSelection: the augsynth engine reproduces the published BestMarkets top five (rank, MDE, investment, abs_lift_in_zero)
    "geox_mc": "benchmarks.cases.geox_mc",              # design calibration (original method, no external referent): size at the null, out-of-sample power at the reported MDE, and the winner's-curse gap on the selected region
    "geox_augsynth_recast": "benchmarks.cases.geox_augsynth_recast",  # cross-val vs getrecast/geolift-simulation-study: the augsynth engine reproduces their published GeoLift bias and false-positive rate across all four stress scenarios, including the +3.2pp outlier signature
    "geox_sdid_equivalence": "benchmarks.cases.geox_sdid_equivalence",  # differential cross-val, mlsynth vs mlsynth: with the region forced, GEOX's readout equals SDID(...).fit() on Prop 99 to solver noise -- ATT and every donor weight, over six treated units and seven design-knob settings
    "clustersc_subgroups": "benchmarks.cases.clustersc_subgroups",      # Path B: ClusterSC vs RSC
    "clustersc_subgroups_ref": "benchmarks.cases.clustersc_subgroups_ref",  # cross-val vs authors' repo
    "clustersc_rpca_germany": "benchmarks.cases.clustersc_rpca_germany",  # cross-val vs Bayani's RPCA-SC code (West Germany reunification, value-for-value)
    "fgrc_denoise_behavior": "benchmarks.cases.fgrc_denoise_behavior",  # Path C: what rpca_method="FGRC" does to the donor matrix on Basque/Germany/Prop99 -- fgrc_keep="cluster" strips the between-donor spread and collapses the fit onto one donor on all three; records that FGRC does not beat cv-PCP on pre-fit anywhere
    "fgrc_grc_crossval": "benchmarks.cases.fgrc_grc_crossval",  # cross-val vs Yamamoto's own R grc package: the GRC objective value-for-value on the reference's own solution (2e-15), and the port's ALS never worse from the same start
    "fgrc_toy_subspace": "benchmarks.cases.fgrc_toy_subspace",  # Path B: fGRC subspace separation recovers cluster structure invisible to k-means (Yamamoto-Hwang GRC.Rd toy example)
    "cast_aca": "benchmarks.cases.cast_aca",  # Path A + cross-val vs authors' CAST-panel (Xia-Yan-Wainwright 2025): ACA Medicaid expansion, entrywise point estimates value-for-value + Table 1; skips without the package/data
    "rrsc_reference": "benchmarks.cases.rrsc_reference",  # cross-val: mlsynth RRSC vs reference R (He-Li-Shi-Miao 2026), both regimes value-for-value; skips without R
    "tssc_brooklyn": "benchmarks.cases.tssc_brooklyn",        # Path A: Brooklyn showroom (Li-Shankar)
    "tssc_figure2": "benchmarks.cases.tssc_figure2",          # Path B: Figure 2 MSE-ratio grid
    "tssc_tables2_5": "benchmarks.cases.tssc_tables2_5",       # Path B + cross-val vs the authors' MATLAB under Octave: Li-Shankar Tables 2-5, the Step-1 restriction tests. Size is nominal at every level under DGP1 (0.050/0.100/0.208 at 5/10/20%), each DGP fires only the test it violates, and the four variant ATTs match core Octave's qp to 1e-4 on shared panels
    "sbc_germany": "benchmarks.cases.sbc_germany",            # Path A: SBC German reunification
    "sbc_hongkong": "benchmarks.cases.sbc_hongkong",          # cross-val vs authors' SBC_HK.R (HK handover): detrend exact, mlsynth cyclical SSE < ipop
    "sbc_mc": "benchmarks.cases.sbc_mc",                      # Path B: Shi-Xi-Xie MSE ratios
    "hsc_hongkong": "benchmarks.cases.hsc_hongkong",          # Path A: HSC HK handover
    "hsc_mc": "benchmarks.cases.hsc_mc",                      # Path B: HSC regime adaptation
    "rsc_synth_error": "benchmarks.cases.rsc_synth_error",      # Path B: RSC train≈gen error
    "rsc_rank_condition_mc": "benchmarks.cases.rsc_rank_condition_mc",  # Path C: Amjad-Shah-Shen Thm 6 (rank(M-) = rank(M) is what lets a pre-period relation extrapolate -- exact to 2e-15 when it holds, fails on every design when it does not, costing RSC 14x post-period RMSE) + Thm 3's Goldilocks tradeoff in the singular-value threshold + Section 4.3's reading of Thms 3 and 7 across the ridge penalty eta, where the described exchange (worse pre-period fit for better post-period accuracy) appears at neither threshold: eta buys nothing where the rank is right and improves both windows where it is too permissive
    "rsc_shen_coverage": "benchmarks.cases.rsc_shen_coverage",  # cross-val: Shen CIs + coverage
    "pcr_rsc_ref": "benchmarks.cases.pcr_rsc_ref",              # cross-val: mlsynth PCR vs original RSC (jehangiramjad/tslib, Prop 99)
    "bayesian_rsc_ref": "benchmarks.cases.bayesian_rsc_ref",    # cross-val: mlsynth Bayesian RSC posterior vs SucreRouge/synth_control (Prop 99)
    "lexscm_walmart": "benchmarks.cases.lexscm_walmart",        # Path A: Walmart placebo design
    "lexscm_design_mc": "benchmarks.cases.lexscm_design_mc",    # Path B: Abadie-Zhao design sim
    "marex_walmart": "benchmarks.cases.marex_walmart",
    "marex_section5_mc": "benchmarks.cases.marex_section5_mc",  # Path B: Abadie-Zhao Section 5 / Table 2 simulation (MAE, RMSE, ||w||_0 by cardinality) + the weakly-targeted design family          # Path A: MAREX Walmart placebo design (Abadie-Zhao SCDesign, 10-store subset)
    "marex_scdesign_sim": "benchmarks.cases.marex_scdesign_sim",  # cross-val vs SCDesign's own cardinality-constrained design on the Section 5 simulation panels (captured R run, open quadprog, no Gurobi)
    "marex_table3": "benchmarks.cases.marex_table3",  # Path B: Abadie-Zhao Table 3 -- MAREX computes the SC column on the authors' panels and beats every published alternative at every cardinality
    "scmo_germany": "benchmarks.cases.scmo_germany",            # Path A: Tian et al. West Germany balance
    "scmo_concatenated_mc": "benchmarks.cases.scmo_concatenated_mc",  # Path B: Tian Table 1 / Sun Sim1
    "scmo_averaged_mc": "benchmarks.cases.scmo_averaged_mc",    # Path B: Sun averaged regime geometry
    "scmo_demeaned_mc": "benchmarks.cases.scmo_demeaned_mc",    # Path B: Tian Online Appendix Table B.1 (demeaned matching + permutation test size)
    "scmo_covid_sweden": "benchmarks.cases.scmo_covid_sweden",  # Path A: Tian Online Appendix B.3 -- Sweden's NPIs, Table B.3 weights + effect magnitudes + significance pattern
    "scta_ibex_xval": "benchmarks.cases.scta_ibex_xval",        # cross-val: SCTA vs an independent build of Sun-Ben-Michael-Feller Sec. 2 solved by cvxpy/CLARABEL (ibex monthly day-ahead price, ES treated), plain + ridge-augmented
    "scta_texas_sb8": "benchmarks.cases.scta_texas_sb8",        # Path A + cross-val vs augsynth 0.2.0 on the authors' Texas SB8 panel: pins the nu = K*year_wt^2 knob mapping (augsynth weights the objective by V^2), the demeaning-basis residual, and the Figure 1 frontier
    "rescm_brexit": "benchmarks.cases.rescm_brexit",            # Path A: SCM-relaxation Brexit/UK GDP (2016Q3)
    "rescm_brexit_2020": "benchmarks.cases.rescm_brexit_2020",  # Path A: SCM-relaxation Brexit robustness (2020Q1)
    "brabander_brexit_table1": "benchmarks.cases.brabander_brexit_table1",      # Path A: de Brabander et al. 2025 Table 1, all 14 cells (SC/DSC/SDID i-iii/MASC/ASCM, 2016Q3, no covariates)
    "brabander_brexit_insample": "benchmarks.cases.brabander_brexit_insample",  # Path A: de Brabander et al. 2025 Table 7, in-sample placebo across 20 periods (RMSE/MAB/MedAB)
    "brabander_mc": "benchmarks.cases.brabander_mc",                            # Path B: de Brabander et al. 2025 Sec. 5 Monte Carlo -- per-replication cross-val vs synthdid on the authors' DGP + Table 9 bias geometry
    "rescm_relax_ref": "benchmarks.cases.rescm_relax_ref",      # cross-val vs scmrelax toy panel (skips if absent)
    "rescm_balanced_gdp": "benchmarks.cases.rescm_balanced_gdp",  # cross-val vs scmrelax on authors' balanced-GDP Brexit panel (UK 2016Q3; skips if absent)
    "rescm_relax_mc": "benchmarks.cases.rescm_relax_mc",        # Path B: latent-group MC, relaxations beat SCM
    "rescm_relax_behavior": "benchmarks.cases.rescm_relax_behavior",  # Path B + property: Liao-Shi-Zheng (2026) Tables 1-2 as behaviour -- the relaxation spreads weight within groups where SCM concentrates (L1 distance to the oracle weights 0.21x SCM's), the objective ordering holds in all three panels including the paper's entropy-over-L2 crossover at K > r, and the exact-1/J collapse rate is pinned as a regression guard on the tau grid
    "linf_crossval_ref": "benchmarks.cases.linf_crossval_ref",  # cross-val: LINF vs LinfinitySC (skips if absent)
    "linf_prop99": "benchmarks.cases.linf_prop99",              # Path A: dense L-inf vs sparse SC (Prop 99)
    "linf_sim": "benchmarks.cases.linf_sim",                    # Path B: L-inf vs SC (Wang-Xing-Ye Table 4)
    "sparse_sc_prop99": "benchmarks.cases.sparse_sc_prop99",    # Path A: L1 predictor selection (Prop 99)
    "cscm_viszero": "benchmarks.cases.cscm_viszero",            # cross-val vs Bonander's CSCM R (Vision Zero): SCM->Finland, rate ratio, sum-of-weights
    "nsc_prop99": "benchmarks.cases.nsc_prop99",                # cross-val vs Tian's NSC.R (Prop 99 Table 2)
    "nsc_mc": "benchmarks.cases.nsc_mc",                        # Path B: nonlinear coverage + error-shrinks-with-J
    "vanillasc_prop99": "benchmarks.cases.vanillasc_prop99",  # Path A: canonical ADH 2010 Prop 99
    "dmlfm_germany": "benchmarks.cases.dmlfm_germany",  # cross-val vs pinned pblasso 1.0.8 (Pang, Liu & Xu 2022, German reunification): design objects exact, covariate scaling exact, ATT on a mean across seeds since the sampler spread is wide
    "pang_liu_xu_sims": "benchmarks.cases.pang_liu_xu_sims",  # Path B + cross-val: Pang, Liu & Xu (2022) Tables A6/A7 single-treated-unit cells, plus mlsynth GSYNTH/DMLFM against gsynth 1.0 and pblasso 1.0.8 on shared R-drawn panels -- the designs generate no effect, so their bias column is the mean estimate and their coverage column is coverage of zero
    "vanillasc_olympics": "benchmarks.cases.vanillasc_olympics",  # Path A (Yoneoka et al. 2022 BMJ Open, Tokyo 2020 Olympics -> COVID cases: 143072/89210 cumulative exact) + cross-val vs pinned tidysynth 0.2.0; records that the authors' donor weights are no longer reproducible (0.183) while the p-value is
    "ibex_dap": "benchmarks.cases.ibex_dap",                  # cross-val vs mharoruiz/ibex scinference/lsei SC: Iberian exception day-ahead price (Haro Ruiz-Schult-Wunder 2024), weights value-for-value
    "secession_scm": "benchmarks.cases.secession_scm",       # Path A: Schulte et al. 2026 lost-autonomy triggers -> secessionist surge (Catalonia 2010 / Faroe 1994), tracks authors' SyntheticControlMethods synthetic
    "lto_refined_placebo": "benchmarks.cases.lto_refined_placebo",  # cross-val vs authors' LTO code (Sudijono-Lei): leave-two-out refined placebo p-value on Prop 99 + West Germany + Basque, value-for-value
    "cwz_ttest": "benchmarks.cases.cwz_ttest",                # Path A: CWZ 2026 Table 5 carbon-tax debiased t-test
    "cwz_conformal": "benchmarks.cases.cwz_conformal",    # cross-val vs scinference conformal (CWZ 2021 JASA Sec 5 application)
    "cwz_conformal_mc": "benchmarks.cases.cwz_conformal_mc",  # Path B: CWZ 2021 JASA Sec 4 size, live against the authors' simulation design
    "cwz_conformal_nonstationary": "benchmarks.cases.cwz_conformal_nonstationary",  # Path B: CWZ 2021 supplement Tables I.2/I.4 -- under trending factors the conformal test stops being exact for a misspecified SC (size 0.98 at DGP3, T0=100, against 0.10 with stationary factors), plus Figure I.2 power against the closed-form oracle bound
    "cwz_ttest_mc": "benchmarks.cases.cwz_ttest_mc",          # Path B: CWZ Table 3, live against the authors' calibrated design
    "cwz_rae": "benchmarks.cases.cwz_rae",                    # Path B: CWZ Table 1 relative efficiency, the formula behind ttest_K="auto"
    "cwz_mc": "benchmarks.cases.cwz_mc",                      # Path B: CWZ 2026 Table 3 application-based Monte Carlo
    "masc_basque": "benchmarks.cases.masc_basque",            # Path A: MASC Basque/ETA (KMPT Sec 5)
    "masc_crossval": "benchmarks.cases.masc_crossval",        # cross-val vs authors' own R MASC (maxkllgg/masc, nogurobi) on Basque, value-for-value
    "src_basque": "benchmarks.cases.src_basque",
    "ferman_demeaned_basque": "benchmarks.cases.ferman_demeaned_basque",  # cross-val vs Ferman-Pinto (2021) demeaned SC in their own R (quadprog), Basque/ETA 1975: MSCa == demeaned SC value-for-value (LIVE Rscript)              # cross-val vs R Code_SMC + Path A: SRC Basque/ETA (Zhu 2023)
    "drosc_basque": "benchmarks.cases.drosc_basque",  # cross-val vs authors' own R DRoSC (Koo & Guo 2026, helpers.R + limSolve::lsei) run LIVE via Rscript on Basque: worst-case estimand tau(lambda) + lambda=0 weights value-for-value (~1e-7); skips if R/limSolve absent
    "bscm_china_watches": "benchmarks.cases.bscm_china_watches",  # cross-val vs reference Stan horseshoe + FSPDA (Shi-Huang) on China anti-corruption watches, p>n (Kim-Lee-Gupta 2020)
    "bvss_watches": "benchmarks.cases.bvss_watches",              # cross-val vs authors' own two-coordinate Gibbs (Xu-Zhou 2025) on China anti-corruption watches, p>n: engine exact + posterior ATT within MC error
    "bfsc_germany": "benchmarks.cases.bfsc_germany",                # cross-val vs author appendix Stan (corr 0.999999) + Path A: West Germany reunification (Pinkney 2021)
    "bfsc_prop99": "benchmarks.cases.bfsc_prop99",                  # cross-val vs author appendix Stan run LIVE via rstan (Prop 99, California 1989) -- needs [bayes] + rstan
    "mvbbsc_germany": "benchmarks.cases.mvbbsc_germany",            # cross-val vs authors' bsynth package (rstan) + Path A: West Germany reunification (Martinez & Vives-i-Bastida 2024)
    "geox_mvbbsc_equivalence": "benchmarks.cases.geox_mvbbsc_equivalence",  # cross-val: GEOX engine="mvbbsc" is the MVBBSC estimator run through the seam (West Germany, identity to the last bit)
    "mtgp_california": "benchmarks.cases.mtgp_california",          # cross-val vs replication-package Stan run LIVE via rstan (California APPS, homicide rates 1997-2018, treated 2007) -- needs [bayes] + rstan
    "bpscs_synthetic": "benchmarks.cases.bpscs_synthetic",           # self-contained: BPSCS effect recovery + distance-based shrinkage on a simulated spatial-spillover panel (GPL reference not shipped) -- needs [bayes]
    "mscmt_basque": "benchmarks.cases.mscmt_basque",          # cross-val vs R MSCMT: AG Basque, fit_window=(1960,1969)
    "mscmt_solver": "benchmarks.cases.mscmt_solver",          # cross-val vs cvxpy: the MSCMT inner simplex solver, exactness + work bounds
    "lamba_tigers": "benchmarks.cases.lamba_tigers",          # cross-val vs tidysynth 0.2.0: Lamba et al. 2023 tiger reserves, staggered per-reserve SCM with zone-restricted donor pools
    "malo_prop99": "benchmarks.cases.malo_prop99",            # Path A: Malo et al. 2024 Table 1 bilevel optimum (Prop 99)
    "malo_basque": "benchmarks.cases.malo_basque",            # cross-val vs scm.corner: AG Basque bilevel optimum, beats MSCMT
    "tasc_mc": "benchmarks.cases.tasc_mc",                    # Path B: TASC vs SC state-space ablation (Rho et al.)
    "tasc_prop99": "benchmarks.cases.tasc_prop99",            # cross-val vs authors' TimeAwareSC (srho1/tasc) on Prop 99 (d=2)
    "fscm_prop99": "benchmarks.cases.fscm_prop99",            # Path A: forward-selected SC (Prop 99)
    "gpits": "benchmarks.cases.gpits",                        # Path A: GP-ITS (Heller, no donors)
    "botosaru_ferman_covariates": "benchmarks.cases.botosaru_ferman_covariates",  # Path A: Botosaru-Ferman 2019 Table 1 -- a no-covariate SC matches West German GDP to 0.02% and misses inflation by 92%
    "xu_gsynth_properties": "benchmarks.cases.xu_gsynth_properties",  # Path B: Xu (2017) Table A1 bias/SD/RMSE at T0=15, Nco=40, plus cross-val vs gsynth 1.0 on sim_TN.R and sim_coverage.R -- identical ATT at a given rank, and the Algorithm 2 bootstrap covers at nominal
    "xu_gsynth_sims": "benchmarks.cases.xu_gsynth_sims",  # Path B: Xu (2017) Table A5 rank recovery on sim_factor.R (0.801/0.921/0.896/0.895 at Ntr=5), plus cross-val vs the gsynth 1.0 the archive ships -- same estimator to solver precision at a shared rank, and every rank disagreement is the 0.1%-vs-1% CV guard
    "shi_fine_grained_sc": "benchmarks.cases.shi_fine_grained_sc",  # Path B + property: Shi et al. 2022 Table 2 exactly, the |S| blow-up under OLS, and why the simplex fails differently (convex-hull membership, not |S|)
    "pda_hongkong": "benchmarks.cases.pda_hongkong",          # Path A: PDA methods on HK CEPA (Shi-Wang App E.1)
    "pda_hcw_hongkong": "benchmarks.cases.pda_hcw_hongkong",  # Path A: original HCW best-subset on HK sovereignty (Table XVI/XVII, vs pampe)
    "pda_table1": "benchmarks.cases.pda_table1",              # Path B: mlsynth's default PDA path on the Shi-Huang Table-1 design
    "arco_retail": "benchmarks.cases.arco_retail",                # Path A: Masini-Medeiros 2021 Table 5a -- WLASSO + partial resampling on their Brazilian retail price experiment, value-for-value
    "arco_resampling_mc": "benchmarks.cases.arco_resampling_mc",  # Path B: Masini-Medeiros 2021 Tables 2-3 size -- True/Oracle arms reproduce, LASSO arm over-rejects 2.2x, with the five checks that localise it
    "fspda_dense_mc": "benchmarks.cases.fspda_dense_mc",      # cross-val vs fsPDA FS()/lasso.BIC() on their own dense-MC panels
    "fspda_sparse_mc": "benchmarks.cases.fspda_sparse_mc",    # cross-val vs fsPDA fs()/lasso_ic()/oracle() on their three sparse DGPs
    "fspda_table1": "benchmarks.cases.fspda_table1",          # Path B: all 108 cells of Shi-Huang Table 1, vs the paper and vs their own code
    "pda_lasso_sim": "benchmarks.cases.pda_lasso_sim",        # Path B: Li-Bell Table 2 LASSO-PDA OOS prediction (N>T1)
    "pda_l2_sim": "benchmarks.cases.pda_l2_sim",              # Path B: Shi-Wang Table 2 L2-relaxation size/power
    "pda_luxurywatch": "benchmarks.cases.pda_luxurywatch",    # Path A: Shi-Huang China luxury-watch fsPDA (prewhitened-NW)
    "pda_ppi": "benchmarks.cases.pda_ppi",                    # Path A: Shi-Wang China PPI L2-relaxation (real-estate policy)
    "pda_brexit": "benchmarks.cases.pda_brexit",              # Path A: Shi-Wang Brexit multi-treated-units L2-relaxation
    "pda_pi_coverage": "benchmarks.cases.pda_pi_coverage",    # Path B: Jiang et al. 2025 prediction-interval coverage (Tables 2-5)
    "wan_pda_vs_scm": "benchmarks.cases.wan_pda_vs_scm",              # Path B: Wan-Xie-Hsiao 2018 Table 2 Design 6a, PDA vs SCM over 5 (J,T0) cells x both aggregation rules
    "wan_pda_vs_scm_ref": "benchmarks.cases.wan_pda_vs_scm_ref",      # cross-val vs pampe/Synth on all six Wan-Xie-Hsiao design variants (paired on shared R-generated panels)
    "pda_wheeler_lassosynth": "benchmarks.cases.pda_wheeler_lassosynth",  # cross-val vs Wheeler's LassoSynth: the resampled cumulative band reduces to his construction at block=1
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
    "esc_prop99": "benchmarks.cases.esc_prop99",              # Path A: ESC California (Prop 99) reproduces classic SC path + informative PI
    "esc_saopaulo": "benchmarks.cases.esc_saopaulo",          # Path A: ESC Sao Paulo homicides (T0=9) short-panel interval on lives saved
    "dr_proximal_mc": "benchmarks.cases.dr_proximal_mc",      # Path B: DR/PIPW recovery + double-robustness (Qiu et al. normal DGP)
    "dr_proximal_scenarios": "benchmarks.cases.dr_proximal_scenarios",  # cross-val vs DR_Proximal_SC correct.DR/correct.q on all 7 just-identified scenarios
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
    "dsc_dube": "benchmarks.cases.dsc_dube",
    "dsc_mc": "benchmarks.cases.dsc_mc",                      # Path B: DSC asymptotics MC (Zhang-Zhang-Zhang 2026 Sec 5.1, Figures 1-2 digitised) -- risk ratio -> 1 and weight error shrinking in M, at J = 20 and 50
    "dsc_disco_xval": "benchmarks.cases.dsc_disco_xval",  # cross-val vs Davidvandijcke/DiSCos on the Dube panel; settles issue #304
    "disco_tenure": "benchmarks.cases.disco_tenure",  # cross-val vs the disco Stata Journal published weights (deterministic reference; the R package's are a Monte Carlo draw)                  # Path A: DSC distributional SC on Dube minimum-wage (Gunsilius/DiSCo vignette)
    "dtwsc_basque": "benchmarks.cases.dtwsc_basque",          # Cross-validation: DTWSC warp vs the conflictlab/dsc R package on Basque
    "ascm_kansas": "benchmarks.cases.ascm_kansas",            # cross-val vs augsynth: Kansas ridge-ASCM ladder (SCM/ridge/covariate/residualized)
    "ascm_mixtape": "benchmarks.cases.ascm_mixtape",          # cross-val vs live augsynth 0.2.0: Cunningham Mixtape studies -- Prop 99 (well-posed) and Texas prisons (interpolating fit, runaway ridge CV)
    "wied_nj_minwage": "benchmarks.cases.wied_nj_minwage",  # Path A: Wied (2026) NJ minimum wage, DRSC conditional distributional effects
    "wine_tennessee": "benchmarks.cases.wine_tennessee",  # Path A: Sun et al. (2025 AJAE) Tennessee wine reform, SCM + SDID(optimized); Study 1 unreplicable (NielsenIQ)
    "ascm_jackknife_plus": "benchmarks.cases.ascm_jackknife_plus",  # cross-val vs augsynth inf_type="jackknife+": per-period bounds on Kansas, both branches
    "augsynth_calibrated": "benchmarks.cases.augsynth_calibrated",  # Path B: ASCM near-nominal coverage + bias reduction (BMR 2021 Sec 7)
    "gsynth_xu_turnout": "benchmarks.cases.gsynth_xu_turnout",  # Path A (Xu 2017 PA Table 2 cols 3-4) + cross-val vs live fect 2.4.5: GSYNTH on the EDR/turnout panel, r=0..5 grid and Algorithm 1
    "gsynth_av_laws": "benchmarks.cases.gsynth_av_laws",  # cross-val vs pinned gsynth 1.2.1 (Lang et al. 2026 age-verification laws): 96-fit outcome x force x rank grid, Algorithm 1 criterion and selected rank, published Table 2 loose
    "xu_gsynth_vs_scm": "benchmarks.cases.xu_gsynth_vs_scm",  # Path B: Xu (2017) Table A4 -- the factor model against the convex hull. Synth's bias climbs 0.71/1.33/1.63/2.13 as the loading supports separate while GSC stays in the third decimal at any rank; also records that the archive's sim_adh.R needs p=0 to reproduce its own table
    "bilgel_turkey_lockdown": "benchmarks.cases.bilgel_turkey_lockdown",  # Path A (Bilgel 2022 EctJ Table 3 col.1): PPSCM vs multisynth nu=0.5, six mobility outcomes, Turkey lockdowns
    "song_ml_ascm": "benchmarks.cases.song_ml_ascm",          # Path A (Song et al. 2023 published main_result.csv, loose) + cross-val vs live augsynth 0.2.0 (tight): China clean winter heating, 30 stratified cells
    "pensynth_prop99": "benchmarks.cases.pensynth_prop99",  # cross-val vs LIVE pensynth wsoll1 (Rscript+LowRankQP) on Prop 99 penalized SC (skips if absent)
    "microsynth_seattle": "benchmarks.cases.microsynth_seattle",  # cross-val vs R microsynth panel method (Seattle DMI)
    "microsynth_baltimore": "benchmarks.cases.microsynth_baltimore",  # cross-val vs R microsynth panel method (Baltimore BCIC, Lawrence et al. 2026): identified quantities exact, counterfactual under-identified (max-ESS vs LowRankQP)
    "scpi_staggered": "benchmarks.cases.scpi_staggered",  # cross-val vs scpi: staggered point estimates (Germany)
    "scpi_staggered_pi": "benchmarks.cases.scpi_staggered_pi",  # cross-val vs scpi: staggered TSUA prediction intervals (Germany)
    "scpi_staggered_covariate": "benchmarks.cases.scpi_staggered_covariate",  # cross-val vs scpi: covariate (multi-feature) staggered illustration (Germany)
    "scpi_germany_pi": "benchmarks.cases.scpi_germany_pi",  # cross-val vs scpi: single-unit CFT-2021 prediction intervals, levels + cointegrated (German reunification)
    "scpi_ridge_germany": "benchmarks.cases.scpi_ridge_germany",  # cross-val vs scpi: ridge-constraint Q/lambda/df via CLUSTERSC RSC (.fit()), Amjad et al. 2018 (German reunification)
    "vanillasc_carbontax": "benchmarks.cases.vanillasc_carbontax",
    "beast_prop99": "benchmarks.cases.beast_prop99",  # cross-val vs authors R (jeremylhour): BEAST immunized ATT path on Prop 99 (basic covariate regime)
    "eiv_coverage_mc": "benchmarks.cases.eiv_coverage_mc",  # Path B: Hirshberg 2021 error-in-variables SC interval coverage (low-rank DGP)  # Path A: Andersson 2019 Swedish carbon tax ATT/2005-gap, malo + mscmt backends (paper predictor spec)
    "synth_prop99": "benchmarks.cases.synth_prop99",   # cross-val vs original R Synth solver (Prop 99 outcome-only); skips if R/Synth absent
    "synth_jhai_prop99": "benchmarks.cases.synth_jhai_prop99",  # cross-val vs Hainmueller j-hai/Synth 1.2.0 (Prop 99 ADH spec): weights/ATT + split-conformal band value-for-value
    "ferman_manyperiods": "benchmarks.cases.ferman_manyperiods",  # Path B: Ferman 2021 JASA Table 1 -- VanillaSC recovers factor structure as J,T0 grow (E[mu01]->1, se(alpha) shrinks vs OLS grows); mlsynth == R solve.QP value-for-value
    "ferman_pinto_mc": "benchmarks.cases.ferman_pinto_mc",  # Path B + cross-val: Ferman-Pinto 2021 QE Table 1 MC (CPS-calibrated factor model) -- VanillaSC(SC)/TSSC-MSCa(demeaned SC) reproduce Panel A/B bias + theory; == authors' quadprog QPs value-for-value on identical panels (LIVE Rscript)
    "cmbsts_vignette": "benchmarks.cases.cmbsts_vignette",  # cross-val vs R CausalMBSTS: multivariate BSTS vignette (trend+cycle)
    "cmbsts_supermarket": "benchmarks.cases.cmbsts_supermarket",  # Path A + cross-val vs R CausalMBSTS: Menchetti-Bojinov Table 3 (1-month horizon, pairs 4/7/10)
    "propsc_spain": "benchmarks.cases.propsc_spain",  # Path A + cross-val vs R propsdid: Bogatyrev-Stoetzer Table 2 (common-weights SDID, party vote shares sum to zero)
    "compsc_pennsylvania": "benchmarks.cases.compsc_pennsylvania",  # Path A: Boussim 2026 Pennsylvania AEPS -- Table 1 weights, all of Table 2, and the sec 6.4 placebo (p=0.111) reproduced from public EIA generation data
    "compsc_pennsylvania_r": "benchmarks.cases.compsc_pennsylvania_r",  # cross-val vs the author's csc_replication.R (quadprog) run live: all ten donor weights, every Table-2 cell, and the 42-donor placebo agree to solver tolerance
    "fsc_okano": "benchmarks.cases.fsc_okano",  # Path A (reference port): Okano-Kurisu 2026 functional SC -- all three applications, both fits each, and Tables 1-3 reproduced exactly from the authors' data
    "fsc_estimator": "benchmarks.cases.fsc_estimator",  # Path A: mlsynth.FSC itself on the same three applications -- fertility exact, and the mortality/service divergences measured and pinned
    "vanillasc_xval_references": "benchmarks.cases.vanillasc_xval_references",  # cross-val vs Synth (uniform V) + tidysynth (ADH spec): placebo rank/p-value agreement, plus recorded solver-quality gaps
    "wiltshire_walmart": "benchmarks.cases.wiltshire_walmart",
    "conformal_inversion_prop99": "benchmarks.cases.conformal_inversion_prop99",  # cross-val vs Facure's "Conformal Inference for Synthetic Controls" notebook, transcribed: the CWZ block-permutation p-value agrees value-for-value on the Prop 99 panel, and the cumulative band that inverts it is pinned under both searches -- the accepted set is two islands, so the bisecting search excludes zero where the grid search (and the p-value itself) accepts it  # Path A (geometry, not cells): Wiltshire 2023 sec 4.2 stacked SCM on 566 Walmart counties -- the paper's prose claims (pre-fit, no effect at entry, decline from e=2, large negative at e=5) plus the base-period indexing identity; magnitudes not claimed, see docs/replications/stackedsc.rst
    "conformal_window_count": "benchmarks.cases.conformal_window_count",  # design calibration (no external referent): the cumulative band's coverage is bounded by the number of calibration WINDOWS, not periods -- 0.86 at m=3, 0.94 at m=26, with exchangeability and normality granted throughout 
    "illenberger_rtm": "benchmarks.cases.illenberger_rtm",  # Path B: Illenberger-Small-Shaw 2020 Tables 1-2 -- regression to the mean inflates the SC placebo test to 0.51 at nominal 0.05 under the paper's level-matching spec and 0.41 under VanillaSC's path matching, while unmatched DiD holds 0.05
    "pcr_shen_estimator_coverage": "benchmarks.cases.pcr_shen_estimator_coverage",  # design calibration (no external referent): coverage of the intervals shen_inference actually returns, on the paper's own DGP -- per period the shipped path is calibrated for all three variance estimators, but the multi-period ATT interval is the library's own construction and its VT arm falls from 0.95 at one post-period to 0.46 at ten, understated by almost exactly the sqrt(T1) it assumes
    "ppscm_geo_conformal_coverage": "benchmarks.cases.ppscm_geo_conformal_coverage",  # design calibration (no external referent): PPSCM's per-unit cumulative band on a synthetic top-30 geo panel covers at the rank its order statistic implies -- 10 windows at h=8 cannot reach 95% at any width, 21 at h=4 can, and the gap to the exchangeable prediction is zero in both
}

# Names whose case reads an external R/MATLAB reference *dump*. Cross-checks that
# clone a reference on demand (e.g. the SpSyDiD clone) are NOT listed here: they
# run under the default ``--all`` and skip themselves (BenchmarkSkipped) when
# their optional dependency is absent.
NEEDS_REFERENCE = set()


def load(name: str):
    mod = importlib.import_module(CASES[name])
    return mod

# ---------------------------------------------------------------------------
# What each case validates, and what it runs on.
#
# ``paths`` is a set, because a case can establish more than one thing at once:
# ``gsynth_xu_turnout`` reproduces Xu (2017) Table 2 *and* cross-validates
# against a live ``fect`` run. The old single-label comment forced a choice, so
# whichever half was written down was the half that survived.
#
#   A  the paper's empirical result, on the authors' data
#   B  the paper's Monte Carlo or simulation table
#   C  a theoretical property, or a design calibration with no external
#      referent -- the case asserts something about the method itself
#   X  cross-validation against an authoritative reference implementation
#
# ``data`` is independent of ``paths``, because the path does not say what the
# case runs on: ``gsynth_av_laws`` cross-validates on a real panel and
# ``cwz_conformal_mc`` cross-validates on a generated one.
#
#   simulated  every panel the case fits comes from a data-generating process
#   empirical  every panel comes from a dataset on disk
#   both       the case has arms of each kind, or calibrates a DGP from a real
#              panel and then fits the draws
#
# A captured reference dump under ``benchmarks/reference/`` says nothing about
# this axis -- many of them hold simulated panels.
#
# benchmarks/tests/test_registry_labels.py is the gate: a case that is missing
# here, carries a value outside the vocabulary, or is filed on the docs page
# under a path it does not claim, fails CI.
# ---------------------------------------------------------------------------

PATH_NAMES = {
    "A": "empirical replication",
    "B": "Monte Carlo / simulation",
    "C": "theoretical property or design calibration",
    "X": "cross-validation against a reference implementation",
}

DATA_KINDS = ("simulated", "empirical", "both")


class Label(NamedTuple):
    """What a benchmark case establishes, and what it runs on."""

    paths: frozenset          # a non-empty subset of PATH_NAMES
    data: str                 # one of DATA_KINDS


# name -> (paths, data)
_RAW: dict[str, tuple[str, str]] = {
    "arco_resampling_mc":             ("B", "simulated"),
    "arco_retail":                    ("A", "empirical"),
    "ascm_jackknife_plus":            ("X", "empirical"),
    "ascm_kansas":                    ("X", "empirical"),
    "ascm_mixtape":                   ("X", "empirical"),
    "augsynth_calibrated":            ("B", "both"),
    "bayesian_rsc_ref":               ("X", "empirical"),
    "beast_prop99":                   ("X", "empirical"),
    "bfsc_germany":                   ("AX", "empirical"),
    "bfsc_prop99":                    ("X", "empirical"),
    "bilgel_turkey_lockdown":         ("A", "empirical"),
    "botosaru_ferman_covariates":     ("A", "empirical"),
    "bpscs_synthetic":                ("C", "simulated"),
    "brabander_brexit_insample":      ("A", "empirical"),
    "brabander_brexit_table1":        ("A", "empirical"),
    "brabander_mc":                   ("BX", "simulated"),
    "brazil_vaccine_scm_vs_proximal": ("X", "empirical"),
    "bscm_china_watches":             ("X", "empirical"),
    "bvss_watches":                   ("X", "empirical"),
    "cast_aca":                       ("AX", "empirical"),
    "cfm":                            ("A", "empirical"),
    "clustersc_rpca_germany":         ("X", "empirical"),
    "clustersc_subgroups":            ("B", "simulated"),
    "clustersc_subgroups_ref":        ("X", "simulated"),
    "cmbsts_supermarket":             ("AX", "empirical"),
    "cmbsts_vignette":                ("X", "both"),
    "compsc_pennsylvania":            ("A", "empirical"),
    "compsc_pennsylvania_r":          ("X", "empirical"),
    "conformal_inversion_prop99":     ("AX", "empirical"),
    "conformal_window_count":         ("C", "simulated"),
    "cscipca_brexit":                 ("A", "empirical"),
    "cscipca_mc":                     ("B", "simulated"),
    "cscm_viszero":                   ("X", "empirical"),
    "ctsc_powell_mc":                 ("B", "simulated"),
    "cwz_conformal":                  ("X", "empirical"),
    "cwz_conformal_mc":               ("B", "simulated"),
    "cwz_conformal_nonstationary":    ("B", "simulated"),
    "cwz_mc":                         ("B", "both"),
    "cwz_rae":                        ("B", "simulated"),
    "cwz_ttest":                      ("A", "empirical"),
    "cwz_ttest_mc":                   ("B", "simulated"),
    "disco_tenure":                   ("AX", "empirical"),
    "dmlfm_germany":                  ("X", "empirical"),
    "dpsc_prop99":                    ("X", "both"),
    "dr_proximal_brazil":             ("X", "empirical"),
    "dr_proximal_mc":                 ("B", "simulated"),
    "dr_proximal_scenarios":          ("X", "simulated"),
    "drosc_basque":                   ("X", "empirical"),
    "dsc_disco_xval":                 ("X", "empirical"),
    "dsc_dube":                       ("A", "empirical"),
    "dsc_mc":                         ("B", "simulated"),
    "dscar_beijing":                  ("A", "empirical"),
    "dtwsc_basque":                   ("X", "empirical"),
    "eiv_coverage_mc":                ("AB", "simulated"),
    "esc_prop99":                     ("A", "empirical"),
    "esc_saopaulo":                   ("A", "empirical"),
    "fdid_hongkong":                  ("AX", "empirical"),
    "fdid_normality_mc":              ("C", "simulated"),
    "fdid_selection_mc":              ("C", "simulated"),
    "fdid_serial_correlation_mc":     ("C", "simulated"),
    "fdid_table5":                    ("B", "simulated"),
    "ferman_demeaned_basque":         ("AX", "empirical"),
    "ferman_manyperiods":             ("B", "simulated"),
    "ferman_pinto_mc":                ("BX", "simulated"),
    "fgrc_denoise_behavior":          ("C", "empirical"),
    "fgrc_grc_crossval":              ("X", "simulated"),
    "fgrc_toy_subspace":              ("B", "simulated"),
    "fma_coverage_mc":                ("B", "simulated"),
    "fsc_estimator":                  ("A", "empirical"),
    "fsc_okano":                      ("A", "empirical"),
    "fscm_prop99":                    ("A", "empirical"),
    "fspda_dense_mc":                 ("X", "simulated"),
    "fspda_sparse_mc":                ("X", "empirical"),
    "fspda_table1":                   ("B", "simulated"),
    "geox_augsynth_geolift":          ("X", "empirical"),
    "geox_augsynth_recast":           ("BX", "simulated"),
    "geox_mc":                        ("C", "both"),
    "geox_mvbbsc_equivalence":        ("X", "empirical"),
    "geox_sdid_equivalence":          ("X", "empirical"),
    "gmmsce_carbontax":               ("X", "empirical"),
    "gpits":                          ("A", "empirical"),
    "gsynth_av_laws":                 ("AX", "empirical"),
    "gsynth_xu_turnout":              ("AX", "empirical"),
    "hsc_hongkong":                   ("A", "empirical"),
    "hsc_mc":                         ("B", "simulated"),
    "ibex_dap":                       ("X", "empirical"),
    "illenberger_rtm":                ("B", "simulated"),
    "lamba_tigers":                   ("X", "empirical"),
    "lexscm_design_mc":               ("B", "simulated"),
    "lexscm_walmart":                 ("A", "empirical"),
    "linf_crossval_ref":              ("X", "simulated"),
    "linf_prop99":                    ("A", "empirical"),
    "linf_sim":                       ("B", "simulated"),
    "lpca_kansas":                    ("X", "empirical"),
    "lpca_mc":                        ("X", "simulated"),
    "lto_refined_placebo":            ("X", "empirical"),
    "malo_basque":                    ("X", "empirical"),
    "malo_prop99":                    ("A", "empirical"),
    "marex_scdesign_sim":             ("X", "simulated"),
    "marex_section5_mc":              ("AB", "simulated"),
    "marex_table3":                   ("B", "simulated"),
    "marex_walmart":                  ("AX", "empirical"),
    "masc_basque":                    ("A", "empirical"),
    "masc_crossval":                  ("X", "empirical"),
    "mcnnm_prop99":                   ("X", "empirical"),
    "medsc_prop99":                   ("A", "empirical"),
    "microsynth_baltimore":           ("X", "empirical"),
    "microsynth_seattle":             ("X", "empirical"),
    "mlsc_bottmer":                   ("X", "simulated"),
    "mscmt_basque":                   ("X", "empirical"),
    "mscmt_solver":                   ("X", "both"),
    "msqrt_sim":                      ("B", "simulated"),
    "mtgp_california":                ("X", "empirical"),
    "mvbbsc_germany":                 ("AX", "empirical"),
    "nsc_mc":                         ("B", "simulated"),
    "nsc_prop99":                     ("AX", "empirical"),
    "orthsc_carbontax":               ("A", "empirical"),
    "orthsc_size_power":              ("B", "simulated"),
    "pang_liu_xu_sims":               ("BX", "simulated"),
    "pangeo_supergeo_mc":             ("B", "simulated"),
    "pcr_rsc_ref":                    ("X", "empirical"),
    "pcr_shen_estimator_coverage":    ("C", "both"),
    "pda_brexit":                     ("A", "empirical"),
    "pda_hcw_hongkong":               ("A", "empirical"),
    "pda_hongkong":                   ("A", "empirical"),
    "pda_l2_sim":                     ("B", "simulated"),
    "pda_lasso_sim":                  ("B", "simulated"),
    "pda_luxurywatch":                ("A", "empirical"),
    "pda_pi_coverage":                ("B", "simulated"),
    "pda_ppi":                        ("A", "empirical"),
    "pda_table1":                     ("B", "simulated"),
    "pda_wheeler_lassosynth":         ("X", "simulated"),
    "pensynth_prop99":                ("X", "empirical"),
    "pioid_overid_jtest":             ("B", "simulated"),
    "ppscm_bfr_mc":                   ("B", "simulated"),
    "ppscm_cs_real_panels":           ("X", "empirical"),
    "ppscm_geo_conformal_coverage":   ("C", "simulated"),
    "ppscm_paglayan":                 ("X", "empirical"),
    "ppscm_paglayan_covs":            ("X", "empirical"),
    "propsc_spain":                   ("AX", "empirical"),
    "proximal_germany_oid":           ("X", "empirical"),
    "proximal_oid_mc":                ("B", "simulated"),
    "proximal_panic1907":             ("AX", "empirical"),
    "proximal_surrogates_mc":         ("B", "simulated"),
    "rescm_balanced_gdp":             ("X", "both"),
    "rescm_brexit":                   ("A", "empirical"),
    "rescm_brexit_2020":              ("A", "empirical"),
    "rescm_relax_behavior":           ("BC", "simulated"),
    "rescm_relax_mc":                 ("B", "simulated"),
    "rescm_relax_ref":                ("X", "simulated"),
    "rolldid_lw":                     ("A", "empirical"),
    "ronczewski_cannabis":            ("A", "empirical"),
    "rrsc_reference":                 ("X", "simulated"),
    "rsc_rank_condition_mc":          ("C", "simulated"),
    "rsc_shen_coverage":              ("X", "simulated"),
    "rsc_synth_error":                ("B", "simulated"),
    "sbc_germany":                    ("A", "empirical"),
    "sbc_hongkong":                   ("X", "empirical"),
    "sbc_mc":                         ("B", "simulated"),
    "scd_cps":                        ("X", "empirical"),
    "scmo_averaged_mc":               ("B", "simulated"),
    "scmo_concatenated_mc":           ("B", "simulated"),
    "scmo_covid_sweden":              ("A", "empirical"),
    "scmo_demeaned_mc":               ("B", "simulated"),
    "scmo_germany":                   ("A", "empirical"),
    "scpi_germany_pi":                ("X", "empirical"),
    "scpi_ridge_germany":             ("X", "empirical"),
    "scpi_staggered":                 ("X", "empirical"),
    "scpi_staggered_covariate":       ("X", "empirical"),
    "scpi_staggered_pi":              ("X", "empirical"),
    "scta_ibex_xval":                 ("X", "empirical"),
    "scta_texas_sb8":                 ("AX", "empirical"),
    "scul_prop99":                    ("A", "empirical"),
    "sdid_ddd_hpv":                   ("A", "empirical"),
    "sdid_euets":                     ("AX", "empirical"),
    "sdid_prop99":                    ("X", "empirical"),
    "secession_scm":                  ("A", "empirical"),
    "seq_sdid_mc":                    ("B", "both"),
    "shc_recovery_mc":                ("B", "simulated"),
    "shi_fine_grained_sc":            ("B", "simulated"),
    "si_prop99":                      ("X", "empirical"),
    "siv_syria_mc":                   ("B", "simulated"),
    "snn_prop99":                     ("X", "empirical"),
    "song_ml_ascm":                   ("AX", "empirical"),
    "sparse_sc_prop99":               ("A", "empirical"),
    "spcd_prop99":                    ("A", "both"),
    "spillsynth_grossi_germany":      ("A", "empirical"),
    "spillsynth_iscm_germany":        ("A", "empirical"),
    "spillsynth_iscm_xval":           ("X", "empirical"),
    "spillsynth_iterative_germany":   ("A", "empirical"),
    "spillsynth_prop99":              ("X", "empirical"),
    "spillsynth_prop99_sar":          ("X", "empirical"),
    "spillsynth_sar_mc":              ("B", "simulated"),
    "spillsynth_sudan":               ("X", "empirical"),
    "spotsynth_panic1907":            ("C", "empirical"),
    "spotsynth_real_data":            ("A", "both"),
    "spsc_ifem_mc":                   ("B", "simulated"),
    "spsc_panic":                     ("A", "empirical"),
    "spsc_prop99":                    ("A", "empirical"),
    "spsydid_lawa_diff":              ("X", "empirical"),
    "spsydid_state_mc":               ("X", "both"),
    "src_basque":                     ("AX", "empirical"),
    "ssc_guanajuato":                 ("X", "empirical"),
    "syndes_bls":                     ("B", "both"),
    "syndes_exact_vs_mip":            ("X", "both"),
    "synth_jhai_prop99":              ("X", "empirical"),
    "synth_prop99":                   ("X", "empirical"),
    "tasc_mc":                        ("B", "simulated"),
    "tasc_prop99":                    ("X", "empirical"),
    "th_prop99":                      ("A", "empirical"),
    "tssc_brooklyn":                  ("A", "empirical"),
    "tssc_figure2":                   ("B", "simulated"),
    "tssc_tables2_5":                 ("BX", "simulated"),
    "twsf_coverage_mc":               ("B", "simulated"),
    "vanillasc_carbontax":            ("A", "empirical"),
    "vanillasc_olympics":             ("AX", "empirical"),
    "vanillasc_prop99":               ("A", "empirical"),
    "vanillasc_xval_references":      ("X", "empirical"),
    "wan_pda_vs_scm":                 ("B", "simulated"),
    "wan_pda_vs_scm_ref":             ("X", "both"),
    "wied_nj_minwage":                ("A", "empirical"),
    "wiltshire_walmart":              ("A", "empirical"),
    "wine_tennessee":                 ("A", "empirical"),
    "xu_gsynth_properties":           ("BX", "simulated"),
    "xu_gsynth_sims":                 ("BX", "simulated"),
    "xu_gsynth_vs_scm":               ("BX", "simulated"),
}

LABELS: dict[str, Label] = {
    name: Label(frozenset(paths), data) for name, (paths, data) in _RAW.items()
}


def paths_of(name: str) -> frozenset:
    """The set of validation paths ``name`` establishes."""
    return LABELS[name].paths


def data_of(name: str) -> str:
    """Whether ``name`` runs on simulated data, empirical data, or both."""
    return LABELS[name].data


def by_path(path: str) -> list:
    """Every case claiming ``path``, sorted. ``by_path("B")`` is the simulations."""
    return sorted(n for n, label in LABELS.items() if path in label.paths)


def by_data(kind: str) -> list:
    """Every case whose data is ``kind``, sorted."""
    return sorted(n for n, label in LABELS.items() if label.data == kind)
