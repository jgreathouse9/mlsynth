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
``BenchmarkSkipped`` instead of failing.

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
   * - ``brabander_brexit_table1``
     - de Brabander et al. (2025) Table 1: seven estimators on the Brexit referendum (SC, DSC, SDID under three panel conventions, MASC, ASCM) at two dates, all fourteen cells
   * - ``brabander_brexit_insample``
     - de Brabander et al. (2025) Table 7: the in-sample placebo across twenty pre-Brexit quarters that ranks those seven, all twenty-one cells
   * - ``clustersc_rpca_germany``
     - RPCA-SC West Germany
   * - ``cwz_ttest``
     - CWZ 2025 Table 5 carbon-tax debiased t-test
   * - ``dsc_dube``
     - DSC distributional SC on Dube minimum-wage (Gunsilius/DiSCo vignette)
   * - ``fsc_okano``
     - Okano-Kurisu (2026) functional SC, all three applications from the authors' data: pre-treatment fits for the plain and ridge-augmented estimator in each (fertility curves 0.1259/0.0687, age-at-death distributions 0.2092/0.0634, trade covariance matrices 39.3429/20.0639) plus every weight of Tables 1-3, reproduced exactly by a port of the authors' R code
   * - ``fsc_estimator``
     - :doc:`fsc` itself on the same three applications: fertility exact (0.1259/0.0687), mortality exact on the plain leg and 0.0630 against 0.0634 on the augmented one, and the trade application under the Frobenius isometry Example 3 specifies, not the authors' plain vech. The two divergences are corrections and are pinned with their measured size; also pins the scale-free penalty search
   * - ``bilgel_turkey_lockdown``
     - Bilgel (2022, Econometrics Journal) Table 3 column 1, from the author's
       replication package: all six mobility ATTs for Turkey's Covid-19 lockdowns
       under PPSCM at ``nu = 0.5``, carried against two reference bases -- a
       commit-pinned ``augsynth`` 0.2.0 run of the author's own ``multisynth``
       call (tight: 1.5e-4 on the ATTs, 4.1e-4 across 102 event-time points) and
       the printed table (loose: within half its last printed digit). Live
       ``augsynth`` reproduces the table, so the two references agree and the
       live rows bind
   * - ``gsynth_xu_turnout``
     - Xu (2017, Political Analysis) Table 2 columns (3) and (4), on the
       author's own EDR/turnout panel: both ATTs (5.13, 4.90) and both
       covariate coefficients (0.15, -1.05) reproduce within half the last
       printed digit, and against a commit-pinned ``fect`` 2.4.5 run the
       agreement is 7.7e-14 on the ATT, 4.8e-11 on the coefficients and 8.9e-14
       across 396 event-time points, over every rank from zero to five on both
       specifications. Algorithm 1's criterion is pinned alongside, because the
       ATT is not monotone in the rank and the rule that picks it decides the
       headline number
   * - ``gsynth_av_laws``
     - Lang et al. (2026) state age-verification laws, from the authors' own
       replication package: four Google Trends outcomes by four ``force``
       settings by ranks zero to five, 96 fits, against a commit-pinned
       ``gsynth`` 1.2.1 -- the version the authors ran, and the last release
       before the package became a shell over ``fect``. The overall ATT, the
       paper's estimand window and the pre-treatment MAE all agree to 6.4e-14,
       Algorithm 1's criterion to 1.3e-12 over the same 96 cells, and its
       selected rank in all sixteen outcome-by-force combinations. Table 2 is
       carried loosely alongside: three of four ATTs within 0.15, the fourth
       0.70 out because the published rank is not the one the paper's own
       cross-validation selects
   * - ``song_ml_ascm``
     - Song et al. (2023) clean winter heating in China, the ridge-ASCM half of their ML-ASCM: 30 stratified cells of their 1024-fit design, carried against two reference bases -- a live augsynth 0.2.0 run (tight) and the authors' published ``main_result.csv`` (loose, with the drift between the two pinned as its own row)
   * - ``dscar_beijing``
     - DSCAR Beijing PM2.5 alerts (Zheng-Chen)
   * - ``fdid_hongkong``
     - HK GDP empirical
   * - ``fscm_prop99``
     - forward-selected SC (Prop 99)
   * - ``hsc_hongkong``
     - HSC HK handover
   * - ``ibex_dap``
     - VanillaSC vs mharoruiz/ibex scinference/lsei SC: Iberian exception day-ahead price, weights value-for-value (Haro Ruiz-Schult-Wunder 2024)
   * - ``lexscm_walmart``
     - Walmart placebo design
   * - ``linf_prop99``
     - dense L-inf vs sparse SC (Prop 99)
   * - ``marex_walmart``
     - MAREX Walmart placebo design vs live SCDesign (Abadie-Zhao, full 45-store panel + covariates, open quadprog, no Gurobi)
   * - ``marex_section5_mc``
     - MAREX vs Abadie-Zhao Section 5 / Table 2 on panels captured from the authors' DGP: effect path, MAE, RMSE and treated count by cardinality, plus the weakly-targeted design family
   * - ``marex_scdesign_sim``
     - MAREX vs SCDesign's own cardinality-constrained design on the Section 5 simulation panels (captured R run, open quadprog, no Gurobi)
   * - ``marex_table3``
     - MAREX computes Abadie-Zhao Table 3's SC column on the authors' panels and beats every published randomized alternative at every cardinality
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
     - SCM-relaxation Brexit/UK GDP (2016Q3)
   * - ``rescm_brexit_2020``
     - SCM-relaxation Brexit robustness (2020Q1)
   * - ``rolldid_lw``
     - Lee-Wooldridge Prop99 + castle
   * - ``sbc_germany``
     - SBC German reunification
   * - ``scmo_germany``
     - Tian et al. West Germany balance
   * - ``scpi_staggered``
     - scpi staggered point estimates, Germany (Cattaneo et al. 2025)
   * - ``scpi_staggered_pi``
     - scpi staggered TSUA prediction intervals, Germany (Cattaneo et al. 2025)
   * - ``scpi_staggered_covariate``
     - scpi covariate (multi-feature) staggered illustration, Germany (Cattaneo et al. 2025)
   * - ``scpi_germany_pi``
     - scpi single-unit CFT-2021 prediction intervals, German reunification: levels + cointegrated bands + weights
   * - ``secession_scm``
     - VanillaSC reproduces Schulte et al. (2026) lost-autonomy SCM: post-trigger secessionist surge, Catalonia 2010 / Faroe 1994 (tracks authors' SyntheticControlMethods synthetic)
   * - ``sdid_euets``
     - Basaglia, Grunau and Drupp (2024, PNAS) EU ETS co-benefits, the synthetic
       DiD robustness half, from the authors' own replication package: all three
       pollutants (SO2, PM2.5, NOx) reproduce the five-decimal values in their
       committed Stata log to 3e-4 once the covariate projection uses the
       reference's row rule. The page records why that qualifier is needed --
       Stata's ``projected`` fits the covariate coefficients on never-treated
       units only, while Kranz (2022), whom it cites, and mlsynth use every
       untreated row; the gap is 0.055 on PM2.5 and is pinned in its own row
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
   * - ``vanillasc_olympics``
     - Yoneoka et al. (2022, BMJ Open) Tokyo 2020 Olympics and COVID-19
       incidence, from the authors' own replication package: the cumulative
       143,072 observed against 89,210 counterfactual (+60.4%) reproduce to the
       digit and the Fisher p-value exactly, with the estimand window recovered
       from the script. Carried against a second base -- a commit-pinned
       ``tidysynth`` 0.2.0 run of the authors' script -- which does not return
       their committed donor weights (0.183 apart, Germany 0.547 against 0.730)
       while returning the same p-value at all three intervention timings. The
       weights are recorded, not gated; mlsynth matches the inference exactly at
       a six-fold better pre-treatment fit

Path B — Monte Carlo / simulation
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Case
     - Validates
   * - ``brabander_mc``
     - de Brabander et al. (2025) Section 5 Monte Carlo: per-replication cross-validation against synthdid on 48 panels drawn by the authors' own DGP, plus the Table 9 finding that demeaning cuts the factor-driven bias about fivefold
   * - ``augsynth_calibrated``
     - ASCM near-nominal coverage + bias reduction (BMR 2021 Sec 7)
   * - ``clustersc_subgroups``
     - ClusterSC vs RSC
   * - ``fgrc_toy_subspace``
     - fGRC subspace separation recovers cluster structure invisible to k-means (Yamamoto-Hwang ``GRC.Rd`` toy example)
   * - ``ctsc_powell_mc``
     - CTSC vs two-way FE bias (Powell 2022 Table 1)
   * - ``cwz_mc``
     - CWZ 2025 Table 3 application-based Monte Carlo
   * - ``dr_proximal_scenarios``
     - DR_Proximal_SC ``correct.DR`` / ``correct.q`` across their scenario directory
   * - ``dr_proximal_mc``
     - DR/PIPW recovery + double-robustness (Qiu et al. normal DGP)
   * - ``dsc_mc``
     - Zhang, Zhang & Zhang (2026) Section 5.1, the asymptotics Monte Carlo behind the Algorithm 1 :doc:`dsc` implements. Targets digitised from the paper's vector-PDF figures, since Section 5 prints no tables. Both theorems' geometry reproduces across sixteen points -- risk ratio falling to 1 in M, weight error falling in M, the larger donor pool converging more slowly -- and the risk ratio matches the published figure to 0.0017 at every cell with M >= 200. The two cells at M = 50 sit above it by 0.006 and 0.028, and the weight-error curve is steeper in the paper; both distances are reported, and :doc:`replications/dsc_mc` says why
   * - ``ferman_manyperiods``
     - VanillaSC recovers the factor structure as J, T0 grow (Ferman 2021 JASA Table 1): weight on the treated factor group → 1, se(α) shrinks while OLS's grows; mlsynth == R ``solve.QP`` value-for-value
   * - ``ferman_pinto_mc``
     - Ferman-Pinto 2021 QE Table 1 MC (CPS-calibrated factor model): VanillaSC (SC) and TSSC MSCa (demeaned SC) reproduce the Panel A/B bias + the efficiency-over-DID and break-panel findings; SC/demeaned == the authors' ``quadprog`` QPs value-for-value on identical panels (live Rscript)
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
   * - ``fspda_dense_mc``
     - fsPDA ``FS()`` / ``lasso.BIC()`` / ``scm.R`` cell by cell on their own dense-MC panels
   * - ``fspda_sparse_mc``
     - fsPDA ``fs()`` / ``lasso_ic()`` / ``oracle()`` on their three sparse DGPs
   * - ``pda_l2_sim``
     - Shi-Wang Table 2 L2-relaxation size/power
   * - ``pda_lasso_sim``
     - Li-Bell Table 2 LASSO-PDA OOS prediction (N>T1)
   * - ``pda_pi_coverage``
     - Jiang et al. 2025 prediction-interval coverage (Tables 2-5)
   * - ``fspda_table1``
     - all 108 cells of Shi-Huang Table 1, vs the paper and their own code
   * - ``pda_table1``
     - mlsynth's default PDA path on the Table-1 design
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
   * - ``orthsc_carbontax``
     - ORTHSC carbon-tax ATT/p/K/CI (Fry; Andersson 2019 data, vs live R)
   * - ``vanillasc_carbontax``
     - VanillaSC malo + mscmt reproduce Andersson (2019) carbon-tax ATT/2005-gap (paper predictor spec)
   * - ``wiltshire_walmart``
     - STACKEDSC on Wiltshire (2023) Section 4.2: 566 Walmart counties in six cohorts against 39 never-treated donors. Geometry, not cells -- the paper's prose claims (excellent pre-fit, no effect at entry, decline from the following year, large negative at five years), the base-period indexing identity at 6e-16, and the per-cohort batching. Its Table 4 magnitudes are not claimed, and :doc:`replications/stackedsc` says why
   * - ``eiv_coverage_mc``
     - Hirshberg (2021) error-in-variables SC prediction-interval coverage on a low-rank DGP
   * - ``orthsc_size_power``
     - ORTHSC fixed-smoothing t-test size control + power (Fry Tables 1-2)
   * - ``spillsynth_sar_mc``
     - SAR spillover recovery + SCM nesting (Sakaguchi-Tagawa)
   * - ``spillsynth_prop99_sar``
     - SAR Bayesian spatial SC vs Mendez California Prop 99 tutorial (bare rho 4dp + ATT + Nevada spillover; full rho weakly identified)
   * - ``spsc_ifem_mc``
     - SPSC IFEM recovery + DT-vs-NoDT coverage (Park-Tchetgen)
   * - ``syndes_bls``
     - Doudchenko et al. 2021 Monte Carlo (BLS unemployment)
   * - ``syndes_exact_vs_mip``
     - SYNDES two-way treated-set search vs SCIP proving optimality on the same program (BLS panel)
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
   * - ``dsc_disco_xval``
     - DSC against the authors' ``DiSCos`` R package on the Dube panel, in both feasible sets. The reference draws its quadrature points with ``runif``, so a single run is not a target; this scores mlsynth against the mean of 40 seeds at M = 10,000 and reads the across-seed spread as the yardstick. Max donor-weight gap 0.0079 (simplex) and 0.0103 (sum-to-one) against reference seed standard deviations of 0.0160 and 0.0300 -- closer to the reference's centre than the reference is to itself, which settles issue #304
   * - ``disco_tenure``
     - vs the ``disco`` Stata Journal published results: the tenure example's top-5 donor weights (to 5e-5) and its quantile-effects table (to 5e-4), plus the M-converged readings so the published m(100) values are not mistaken for the estimand
   * - ``ascm_jackknife_plus``
     - vs augsynth ``inf_type="jackknife+"`` on Kansas: the per-drop held-out errors and counterfactuals, the pointwise bounds under both the default and conservative branches, and the post-period average interval
   * - ``drosc_basque``
     - DROSC vs authors' R ``DRoSC`` (Koo & Guo 2026, ``limSolve::lsei``) on Basque: worst-case estimand τ(λ) and λ=0 weights value-for-value across the robustness sweep
   * - ``propsc_spain``
     - vs LIVE propsdid (Rscript): Bogatyrev-Stoetzer Table 2 common-weights SDID on party vote shares (skips if absent)
   * - ``vanillasc_xval_references``
     - vs Synth (uniform custom.v) and tidysynth (ADH spec) on Prop 99: placebo rank and p-value agree exactly (1/39); also records where mlsynth attains a lower value of Synth's own objective and where ipop fails outright
   * - ``compsc_pennsylvania``
     - Path A: Boussim (2026) Pennsylvania AEPS from public EIA data -- Table 1 donor weights and both RMSPEs, all eleven years of Table 2, and the section 6.4 placebo (p = 0.111); also pins the clr-vs-alr geometry correction
   * - ``clustersc_subgroups_ref``
     - vs authors' repo
   * - ``cast_aca``
     - vs the authors' ``CAST-panel`` package (Xia-Yan-Wainwright 2025) on the ACA Medicaid expansion: entrywise point estimates value-for-value, Table 1 ATET and population totals, and both significance-count regimes (see the replication page for the standard-error correction)
   * - ``rrsc_reference``
     - vs LIVE reference R (He-Li-Shi-Miao 2026): RRSC large-N and fixed-N regimes value-for-value on a synthetic interference panel (skips if R absent)
   * - ``pensynth_prop99``
     - vs LIVE pensynth wsoll1 (Rscript+LowRankQP): penalized SC weights/ATT on Prop 99 (skips if absent)
   * - ``linf_crossval_ref``
     - LINF vs LinfinitySC (skips if absent)
   * - ``mcnnm_prop99``
     - vs authors' MCPanel R (mcnnm_cv; ATT + California counterfactual path)
   * - ``microsynth_seattle``
     - vs R microsynth panel method (Seattle DMI)
   * - ``mlsc_bottmer``
     - vs Bottmer's mlSC_estimator (skips if absent)
   * - ``mvbbsc_germany``
     - vs authors' bsynth R package (rstan): posterior counterfactual + credible bands + ATT, West Germany reunification (Martinez & Vives-i-Bastida)
   * - ``nsc_prop99``
     - vs Tian's NSC.R (Prop 99 Table 2)
   * - ``ppscm_paglayan``
     - vs augsynth::multisynth (jackknife + bootstrap SEs)
   * - ``dr_proximal_brazil``
     - vs live R (authors' analysis.Rmd, commit 3bcb5ec): over-identified DR-OID, Brazil vaccine/pneumonia
   * - ``brazil_vaccine_scm_vs_proximal``
     - vs live R (same script): standard SC (VanillaSC) vs proximal (DR-OID h/DR) contrast, Brazil vaccine/pneumonia
   * - ``proximal_panic1907``
     - vs freshtaste/proximal (Panic 1907 Table 3)
   * - ``rescm_relax_ref``
     - vs scmrelax (skips if absent)
   * - ``rsc_shen_coverage``
     - Shen CIs + coverage
   * - ``ferman_demeaned_basque``
     - TSSC MSCa == Ferman-Pinto (2021) demeaned SC, value-for-value vs their R quadprog (live Rscript), Basque/ETA 1975
   * - ``sdid_prop99``
     - vs authors' synthdid R (synthdid_estimate; SDID/DiD/SC on Prop 99)
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
   * - ``mscmt_solver``
     - vs cvxpy (CLARABEL) on the MSCMT inner simplex program: the batched active set never finishes above the interior-point optimum across the Basque candidate weightings, and its work bounds are pinned as iteration counts, which are machine independent where wall-clock is not. Also pins what the default ``mscmt_tol`` costs the estimate against an exhaustive search

The captured reference corpus
-----------------------------

For many of the cross-validation cases above, the reference is not a number
transcribed from a paper or a package that has to be re-installed every time the
suite runs. It is a captured artifact: the original authors' code, the exact
command that ran it, the verbatim output, and a record of the environment that
produced it, all committed under ``benchmarks/reference/<case>/``. mlsynth's
result is then pinned to that captured output, so the comparison is reproducible
offline and the reference value cannot silently drift from what the authors'
code actually produces.

This section documents that machinery in detail.

Anatomy of a bundle
~~~~~~~~~~~~~~~~~~~~~

A captured bundle is a directory ``benchmarks/reference/<case>/`` containing:

* ``manifest.json`` -- the bundle's contract. It records the ``case`` name, a
  human ``title``, the ``paper`` being validated, a ``reference_impl`` string
  naming the exact code that was run, the ``path_type`` (Path A / Path B /
  cross-validation), the ``command`` that regenerates the bundle, and the list
  of input ``data`` files. The ``command`` is run verbatim, so it can be an
  ``Rscript`` invocation, a ``python`` script, or anything else that prints the
  expected output block.
* ``reference.R`` or ``reference.py`` -- the runnable reference. It drives the
  authors' code on the case's data at the matched settings and prints two
  blocks: a ``== REFERENCE VALUES ==`` block of ``key<TAB>value`` lines (and
  ``weight<TAB>label<TAB>value`` rows for weight vectors), and a
  ``== SESSION INFO ==`` block of tool and package versions.
* The authors' code itself, vendored alongside (for example ``Fun_FDID.R``,
  ``scm.corner.R``, or a ``vendor/`` subdirectory of the minimal modules
  needed), together with any small input data the run requires (for example
  ``GDP.csv``). A ``NOTICE`` file records provenance and licensing -- and where
  an upstream repository ships no license, only the minimal subset needed to run
  the reference is vendored, for provenance, not redistribution.
* ``reference.out`` -- the verbatim captured stdout of the run, kept as the
  human-readable evidence of what the authors' code printed.
* ``reference.json`` -- the parsed result, a mapping ``{"values": {...}}`` that
  the test harness reads.
* ``provenance.json`` -- a record of the run: a UTC ``generated_at`` timestamp,
  the ``git_sha`` of the repository at capture time, the ``platform``, the
  ``command``, the input ``data`` with SHA-256 checksums, and the interpreter
  and package versions (for example ``r_version`` and the loaded ``packages``).
* ``comparison.csv`` -- the side-by-side table of mlsynth against the reference,
  one row per quantity, with the absolute difference (described under
  :ref:`benchmarks-comparison-tables`).

How a live cross-validation is built
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each live cross-validation follows the same recipe, designed to isolate the one
thing being tested -- whether mlsynth and the authors' code compute the same
quantity -- from everything that would otherwise confound it.

#. Run the authors' code, not a paraphrase of it. The reference fetches or
   vendors the upstream implementation and calls it directly, on the same input
   data the mlsynth case uses.
#. Match the settings that are free to differ. Estimators expose tuning choices
   (a penalty level, a number of retained singular values, a transformation
   window, an EM initialisation). The reference and the mlsynth call are driven
   at the same values so that any remaining difference is attributable to the
   implementation, not the configuration. Where a method's own tuning differs
   from a paper's by construction -- for example a time-respecting
   cross-validation against a future-leaking K-fold -- the cross-validation
   pins the solve at a single fixed setting (where the program is a unique
   optimisation), not the tuned end-to-end number, and the tuned number
   is kept as a separate, clearly labelled pin.
#. Capture the output with provenance. ``benchmarks/reference/generate.py`` runs
   the manifest ``command``, parses the ``== REFERENCE VALUES ==`` block into
   ``reference.json``, stores the verbatim ``reference.out``, and writes
   ``provenance.json`` with the checksums and versions above.
#. Pin mlsynth to the captured values. The case reads the captured numbers with
   :func:`benchmarks.reference.reference_value` (or ``load_reference``) and uses
   them as the ``EXPECTED`` targets, so the constant in the test and the
   captured run are the same object -- they cannot diverge without the bundle
   being regenerated.

.. _benchmarks-comparison-tables:

Comparison tables
~~~~~~~~~~~~~~~~~~

Every bundle with a ``comparison()`` writes a ``comparison.csv``: a metadata
header (the case title, the reference implementation, the generation timestamp
and versions) followed by one row per quantity with columns ``quantity``,
``mlsynth``, ``reference`` and ``abs_diff``. The public, web-native rollup of
the whole corpus is the :doc:`validation` dashboard, generated from these CSVs.
Regenerate both with

.. code-block:: bash

   python benchmarks/reference/export_comparison.py --all
   python benchmarks/reference/build_validation.py

What running the authors' code surfaced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Executing the reference, instead of comparing against a printed table, turned
up things a number-for-number comparison could not have. In each case the
discrepancy was traced to its cause, and mlsynth was found to be at the
correct optimum.

* Synthetic Business Cycle (``sbc_germany``). The Hamilton detrending and trend
  forecast match the authors' ``Germany.R`` to about :math:`10^{-8}`, but the
  cycle-matching weight solve diverged. The program is strictly convex and well
  conditioned, so its optimum is unique; four independent solvers (mlsynth's
  and three from cvxpy) agree on it, while the authors' ``Synth::synth`` ipop
  solver lands about :math:`2.6\%` short and does not improve when its
  tolerances are tightened. Running the code also revealed that the shipped wide
  CSV permutes its donor column labels. The full account is on the dedicated
  page :doc:`replications/sbc`.
* Time-Aware Synthetic Control (``tasc_prop99``). Because TASC fits a
  state-space model by a non-convex EM, the two implementations can converge to
  different local optima. Comparing the fitted pre-period log-likelihoods --
  computed identically on both fits -- showed mlsynth's optimum is the better
  one, so the small counterfactual difference is local-optima spread, not an
  error; ``tasc_loglik_advantage`` is pinned as a guard.
* PCR against the original Robust Synthetic Control library (``pcr_rsc_ref``).
  Both implementations solve hard-singular-value-thresholding plus regression,
  but tslib forms the rank-:math:`k` subspace from the stacked donor-and-treated
  matrix while mlsynth de-noises the donor matrix alone (the Amjad-Shah-Shen
  convention). Each is exact for its own convention, and the small gap is
  documented.
* L-infinity synthetic control (``linf_prop99``). With more donors than
  pre-periods the :math:`\ell_\infty`-minimising weight vector is genuinely
  non-unique, so individual weights are not identified. The case cross-validates
  the quantities that are -- the effect path, the pre-fit, the dense
  weight signature, and the effect estimate -- and a multi-solver check confirms
  mlsynth sits at or below the reference's objective.

Regenerating a bundle
~~~~~~~~~~~~~~~~~~~~~~~

A bundle is rebuilt from its manifest with

.. code-block:: bash

   python benchmarks/reference/generate.py <case>

which re-runs the captured ``command``, refreshes ``reference.out`` /
``reference.json`` / ``provenance.json``, and so re-stamps the environment and
checksums. Regeneration requires whatever the reference needs (an R toolchain
and the named packages, or the relevant Python dependency); when that toolchain
is absent the corresponding case raises ``BenchmarkSkipped`` at suite time
instead of failing, and the committed bundle remains the offline record.
