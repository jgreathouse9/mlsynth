.. _benchmarks:

Benchmarks
==========

Every estimator in mlsynth ships with at least one *durable benchmark*: a
self-contained case under ``benchmarks/cases/`` that re-runs a published result
(or a reference implementation) and asserts the headline numbers against a fixed
tolerance. Where the :doc:`replications` page tells the *story* of each
validation in prose, this page documents the *machinery* -- the runnable cases
that guard against regressions as the library changes.

Each case is a small module exposing ``run()`` (which returns a dict of metrics,
driving everything through mlsynth's public API) and ``EXPECTED`` (a map from
metric to a ``(value, tolerance)`` pair). The driver compares the two and a case
that cannot find its data or an optional reference dependency raises
``BenchmarkSkipped`` rather than failing.

Running them
------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --all            # every pure-Python case
   python benchmarks/run_benchmarks.py --case cwz_ttest  # one case
   python benchmarks/run_benchmarks.py --with-reference  # also R / external cross-checks

The registry of cases lives in ``benchmarks/registry.py`` (the source of truth);
the catalogue below is grouped by validation path.

Validation paths
----------------

* **Path A** -- reproduce the source paper's empirical result on the original
  authors' data.
* **Path B** -- reproduce the paper's Monte Carlo / simulation table.
* **Cross-validation** -- match an authoritative reference implementation
  (an R/MATLAB package or the authors' own code); these skip themselves when
  the optional dependency is absent.

Path A — empirical replications
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Case
     - Validates
   * - ``clustersc_rpca_germany``
     - RPCA-SC West Germany
   * - ``cwz_ttest``
     - CWZ 2025 Table 5 carbon-tax debiased t-test
   * - ``dsc_dube``
     - DSC distributional SC on Dube minimum-wage (Gunsilius/DiSCo vignette)
   * - ``dscar_beijing``
     - DSCAR Beijing PM2.5 alerts (Zheng-Chen)
   * - ``fdid_hongkong``
     - HK GDP empirical
   * - ``fscm_prop99``
     - forward-selected SC (Prop 99)
   * - ``hsc_hongkong``
     - HSC HK handover
   * - ``lexscm_walmart``
     - Walmart placebo design
   * - ``linf_prop99``
     - dense L-inf vs sparse SC (Prop 99)
   * - ``marex_walmart``
     - MAREX Walmart placebo design (Abadie-Zhao SCDesign, 10-store subset)
   * - ``masc_basque``
     - MASC Basque/ETA (KMPT Sec 5)
   * - ``pda_brexit``
     - Shi-Wang Brexit multi-treated-units L2-relaxation
   * - ``pda_hongkong``
     - PDA methods on HK CEPA (Shi-Wang App E.1)
   * - ``pda_luxurywatch``
     - Shi-Huang China luxury-watch fsPDA (prewhitened-NW)
   * - ``pda_ppi``
     - Shi-Wang China PPI L2-relaxation (real-estate policy)
   * - ``rescm_brexit``
     - SCM-relaxation Brexit/UK GDP
   * - ``rolldid_lw``
     - Lee-Wooldridge Prop99 + castle
   * - ``sbc_germany``
     - SBC German reunification
   * - ``scmo_germany``
     - Tian et al. West Germany balance
   * - ``sparse_sc_prop99``
     - L1 predictor selection (Prop 99)
   * - ``spcd_prop99``
     - SPCD design vs random/SC on Prop 99 (Lu et al. 2022)
   * - ``spillsynth_grossi_germany``
     - grossi direct+spillover German reunification (Grossi et al.)
   * - ``spillsynth_iscm_germany``
     - inclusive SCM German reunification (Di Stefano-Mellace)
   * - ``spillsynth_iterative_germany``
     - iterative waterfall SCM German reunification (Melnychuk)
   * - ``spotsynth_real_data``
     - SPOTSYNTH donor-spillover screening: Germany/California/Basque (Fig 6) + detection (Fig 2) + debias (Fig 4)
   * - ``tssc_brooklyn``
     - Brooklyn showroom (Li-Shankar)
   * - ``vanillasc_prop99``
     - canonical ADH 2010 Prop 99

Path B — Monte Carlo / simulation
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Case
     - Validates
   * - ``augsynth_calibrated``
     - ASCM near-nominal coverage + bias reduction (BMR 2021 Sec 7)
   * - ``clustersc_subgroups``
     - ClusterSC vs RSC
   * - ``ctsc_powell_mc``
     - CTSC vs two-way FE bias (Powell 2022 Table 1)
   * - ``cwz_mc``
     - CWZ 2025 Table 3 application-based Monte Carlo
   * - ``dr_proximal_mc``
     - DR/PIPW recovery + double-robustness (Qiu et al. normal DGP)
   * - ``fdid_table5``
     - simulation
   * - ``fma_coverage_mc``
     - FMA asymptotic-CI coverage robust to variance (Li-Sonnier)
   * - ``hsc_mc``
     - HSC regime adaptation
   * - ``lexscm_design_mc``
     - Abadie-Zhao design sim
   * - ``linf_sim``
     - L-inf vs SC (Wang-Xing-Ye Table 4)
   * - ``msqrt_sim``
     - MSQRT unbiasedness + RMSE noise-floor (Shen-Song-Abadie Sec 6)
   * - ``nsc_mc``
     - nonlinear coverage + error-shrinks-with-J
   * - ``pangeo_supergeo_mc``
     - PANGEO trajectory match vs scalar (Chen et al.)
   * - ``pda_l2_sim``
     - Shi-Wang Table 2 L2-relaxation size/power
   * - ``pda_lasso_sim``
     - Li-Bell Table 2 LASSO-PDA OOS prediction (N>T1)
   * - ``pda_table1``
     - Shi-Huang Table 1 fs-vs-LASSO size/power geometry
   * - ``proximal_surrogates_mc``
     - PI/PIS/PIPost vs SC under trending factor (Liu et al.)
   * - ``rescm_relax_mc``
     - latent-group MC, relaxations beat SCM
   * - ``rsc_synth_error``
     - RSC train≈gen error
   * - ``sbc_mc``
     - Shi-Xi-Xie MSE ratios
   * - ``scmo_averaged_mc``
     - Sun averaged regime geometry
   * - ``scmo_concatenated_mc``
     - Tian Table 1 / Sun Sim1
   * - ``seq_sdid_mc``
     - SSDiD vs DiD coverage/RMSE
   * - ``shc_recovery_mc``
     - SHC latent-confounder recovery (Chen-Yang-Yang Sec 3.1)
   * - ``siv_syria_mc``
     - SIV vs 2SLS-TWFE bias (Gulek-Vives Table 1)
   * - ``spillsynth_sar_mc``
     - SAR spillover recovery + SCM nesting (Sakaguchi-Tagawa)
   * - ``spsc_ifem_mc``
     - SPSC IFEM recovery + DT-vs-NoDT coverage (Park-Tchetgen)
   * - ``syndes_bls``
     - Doudchenko et al. 2021 Monte Carlo (BLS unemployment)
   * - ``tasc_mc``
     - TASC vs SC state-space ablation (Rho et al.)
   * - ``tssc_figure2``
     - Figure 2 MSE-ratio grid

Cross-validation against reference implementations
--------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Case
     - Validates
   * - ``ascm_kansas``
     - vs augsynth: Kansas ridge-ASCM ladder (SCM/ridge/covariate/residualized)
   * - ``clustersc_subgroups_ref``
     - vs authors' repo
   * - ``geolift_augsynth_ref``
     - vs LIVE augsynth (Rscript): lambda/weights/ATT (skips if absent)
   * - ``geolift_cpic``
     - vs GeoLiftMarketSelection: CPIC investment value-for-value
   * - ``geolift_multicell``
     - vs augsynth: multi-cell per-cell ATT + donor exclusion
   * - ``geolift_walkthrough``
     - vs GeoLift/augsynth: GeoLift_Walkthrough realized report (fixedeff ASCM + conformal)
   * - ``linf_crossval_ref``
     - LINF vs LinfinitySC (skips if absent)
   * - ``mcnnm_prop99``
     - vs causaltensor
   * - ``microsynth_seattle``
     - vs R microsynth panel method (Seattle DMI)
   * - ``mlsc_bottmer``
     - vs Bottmer's mlSC_estimator (skips if absent)
   * - ``nsc_prop99``
     - vs Tian's NSC.R (Prop 99 Table 2)
   * - ``ppscm_paglayan``
     - vs augsynth::multisynth (jackknife + bootstrap SEs)
   * - ``proximal_panic1907``
     - vs freshtaste/proximal (Panic 1907 Table 3)
   * - ``rescm_relax_ref``
     - vs scmrelax (skips if absent)
   * - ``rsc_shen_coverage``
     - Shen CIs + coverage
   * - ``sdid_prop99``
     - vs causaltensor
   * - ``si_prop99``
     - vs Agarwal-Shah-Shen 2026 authors' code (Prop 99)
   * - ``snn_prop99``
     - vs deshen24/syntheticNN (Prop 99)
   * - ``spillsynth_iscm_xval``
     - vs Melnychuk-Andrii/Spillover-SCM (inclusive SCM German)
   * - ``spillsynth_prop99``
     - vs jcao0/synthetic-control-spillover (Cao-Dowd Prop 99)
   * - ``spsydid_state_mc``
     - vs authors' repo
   * - ``ssc_guanajuato``
     - vs jcao0/staggered_synthetic_control (criminality Sec 4)
