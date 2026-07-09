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
     - MAREX Walmart placebo design vs live SCDesign (Abadie-Zhao, full 45-store panel + covariates, open quadprog, no Gurobi)
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
   * - ``pda_pi_coverage``
     - Jiang et al. 2025 prediction-interval coverage (Tables 2-5)
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
   * - ``orthsc_carbontax``
     - ORTHSC carbon-tax ATT/p/K/CI (Fry; Andersson 2019 data, vs live R)
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
   * - ``propsc_spain``
     - vs LIVE propsdid (Rscript): Bogatyrev-Stoetzer Table 2 common-weights SDID on party vote shares (skips if absent)
   * - ``clustersc_subgroups_ref``
     - vs authors' repo
   * - ``geolift_augsynth_ref``
     - vs LIVE augsynth (Rscript): lambda/weights/ATT (skips if absent)
   * - ``pensynth_prop99``
     - vs LIVE pensynth wsoll1 (Rscript+LowRankQP): penalized SC weights/ATT on Prop 99 (skips if absent)
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
   * - ``dr_proximal_brazil``
     - vs live R (authors' analysis.Rmd, commit 3bcb5ec): over-identified DR-OID, Brazil vaccine/pneumonia
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
  the reference is vendored, for provenance rather than redistribution.
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
   optimisation) rather than the tuned end-to-end number, and the tuned number
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
``mlsynth``, ``reference`` and ``abs_diff``. A combined workbook,
``benchmarks/reference/comparisons.xlsx``, collects a summary sheet (one row per
case, with its largest absolute difference) plus a metadata-stamped sheet per
case, so the whole corpus can be scanned at a glance. Both are rebuilt by

.. code-block:: bash

   python benchmarks/reference/export_comparison.py --all

What running the authors' code surfaced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Executing the reference rather than trusting a printed table repeatedly turned
up things a number-for-number comparison could not have. In each case the
discrepancy was traced to its cause rather than absorbed into a loose
tolerance, and mlsynth was found to be at the correct optimum.

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
  convention). Each is exact for its own convention; the small gap is
  documented rather than tuned away.
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
rather than failing, and the committed bundle remains the offline record.
