.. _validation:

Validation dashboard
====================

Every estimator in mlsynth is checked against the original authors' code
on real data. This page is generated from the pinned reference bundles the
test suite asserts against, so the numbers here cannot drift from what CI
enforces. Each row links to the reference implementation, the dataset (with
checksum), and the mlsynth case that runs the check.

Coverage: **48 cross-validation checks** against original
implementations across **27 estimators** -- 18 reproduce the reference to display precision, 19 to
within two percent. A further 2 are captured on the next daily run (see `Pending capture`_). Per-estimator paper replications (Path A / Path B) are catalogued in :doc:`replications`.

Legend: **exact** (agreement to display precision), **tight** (worst
relative deviation :math:`\le 2\%`), **close** (:math:`\le 10\%`), and
**documented** (looser, with a stated reason on the estimator's replication
page -- typically an intrinsically extrapolated or weakly-identified quantity).

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 26 14 44 16

   * - Estimator
     - Checks
     - Agreement
     - Worst max \|Δ\|
   * - :ref:`BEAST <val-beast>`
     - 1
     - 1 tight
     - 0.15
   * - :ref:`BFSC <val-bfsc>`
     - 1
     - 1 close
     - 1
   * - :ref:`ClusterSC <val-clustersc>`
     - 2
     - 1 exact · 1 tight
     - 0.036
   * - :ref:`FDID <val-fdid>`
     - 1
     - 1 exact
     - 0.00032
   * - :ref:`GEOLIFT <val-geolift>`
     - 2
     - 2 exact
     - 0.026
   * - :ref:`LINF <val-linf>`
     - 2
     - 1 tight · 1 close
     - 0.39
   * - :ref:`MAREX <val-marex>`
     - 1
     - 1 tight
     - 0.016
   * - :ref:`MCNNM <val-mcnnm>`
     - 1
     - 1 tight
     - 0.81
   * - :ref:`MLSC <val-mlsc>`
     - 1
     - 1 exact
     - 1e-06
   * - :ref:`MicroSynth <val-microsynth>`
     - 1
     - 1 tight
     - 0.097
   * - :ref:`NSC <val-nsc>`
     - 1
     - 1 close
     - 1.9
   * - :ref:`ORTHSC <val-orthsc>`
     - 2
     - 1 exact · 1 close
     - 0.037
   * - :ref:`PDA <val-pda>`
     - 5
     - 4 exact · 1 close
     - 0.056
   * - :ref:`PPSCM <val-ppscm>`
     - 1
     - 1 tight
     - 0.0022
   * - :ref:`PROXIMAL <val-proximal>`
     - 1
     - 1 tight
     - 0.014
   * - :ref:`RESCM <val-rescm>`
     - 2
     - 2 tight
     - 0.0013
   * - :ref:`ROLLDID <val-rolldid>`
     - 1
     - 1 exact
     - 0
   * - :ref:`SCMO <val-scmo>`
     - 1
     - 1 tight
     - 0.011
   * - :ref:`SCUL <val-scul>`
     - 1
     - 1 tight
     - 0.14
   * - :ref:`SDID <val-sdid>`
     - 1
     - 1 tight
     - 0.0016
   * - :ref:`SI <val-si>`
     - 1
     - 1 exact
     - 0
   * - :ref:`SNN <val-snn>`
     - 1
     - 1 exact
     - 0
   * - :ref:`SPILLSYNTH <val-spillsynth>`
     - 4
     - 1 exact · 1 tight · 1 close · 1 documented
     - 7.6
   * - :ref:`SSC <val-ssc>`
     - 1
     - 1 tight
     - 0.001
   * - :ref:`SpSyDiD <val-spsydid>`
     - 1
     - 1 close
     - 0.094
   * - :ref:`TASC <val-tasc>`
     - 1
     - 1 documented
     - 25
   * - :ref:`VanillaSC <val-vanillasc>`
     - 10
     - 4 exact · 4 tight · 2 close
     - 0.11

.. _val-beast:

BEAST
-----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - jeremylhour/alternative-synthetic-control-sparsity R (CalibrationLasso/OrthogonalityReg/ImmunizedATT)
     - ``augmented_cali_long.csv`` (974ae6ad6ab7…)
     - 13
     - 0.15
     - tight
     - `beast_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/beast_prop99.py>`__

.. _val-bfsc:

BFSC
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - author appendix Stan (via Rscript + rstan)
     - —
     - 3
     - 1
     - close
     - `bfsc_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/bfsc_prop99.py>`__

.. _val-clustersc:

ClusterSC
---------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - jehangiramjad/tslib RobustSyntheticControl (live run, captured), modelType='svd', kSingularValuesToKeep=3
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 8
     - 0.036
     - tight
     - `pcr_rsc_ref <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pcr_rsc_ref.py>`__
   * - deshen24/panel-data-regressions var.var_est (homoskedastic + jackknife)
     - —
     - 6
     - 0
     - exact — matches to display precision
     - `rsc_shen_coverage <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/rsc_shen_coverage.py>`__

.. _val-fdid:

FDID
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - Kathleen T. Li's Fun_FDID.R (MKSC replication, live run, captured)
     - ``GDP.csv`` (487c7007cad0…)
     - 7
     - 0.00032
     - exact — matches to display precision
     - `fdid_hongkong <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fdid_hongkong.py>`__

.. _val-geolift:

GEOLIFT
-------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - R package augsynth (via Rscript)
     - —
     - 15
     - 0.026
     - exact — matches to display precision
     - `geolift_augsynth_ref <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/geolift_augsynth_ref.py>`__
   * - R GeoLiftMarketSelection (via Rscript)
     - —
     - 15
     - 0.005
     - exact — matches to display precision
     - `geolift_marketselection_ref <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/geolift_marketselection_ref.py>`__

.. _val-linf:

LINF
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - LinfinitySC our(method='inf'|'l1-inf') (Wang, Xing & Ye 2025), https://github.com/BioAlgs/LinfinitySC
     - —
     - 40
     - 0.00041
     - tight
     - `linf_crossval_ref <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/linf_crossval_ref.py>`__
   * - LinfinitySC our(method='inf') (Wang, Xing & Ye 2025), https://github.com/BioAlgs/LinfinitySC, lambda via param_selector(method='inf', n_folds=10)
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 43
     - 0.39
     - close
     - `linf_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/linf_prop99.py>`__

.. _val-marex:

MAREX
-----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - jinglongzhao2/SCDesign (cardinality-K design, open quadprog, live run)
     - ``walmart_weekly_sales_covariates.csv`` (906fb3cd9e2f…)
     - 7
     - 0.016
     - tight
     - `marex_walmart <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/marex_walmart.py>`__

.. _val-mcnnm:

MCNNM
-----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - `susanathey/MCPanel R (mcnnm_cv, defaults) <https://github.com/susanathey/MCPanel>`__
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 13
     - 0.81
     - tight
     - `mcnnm_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/mcnnm_prop99.py>`__

.. _val-mlsc:

MLSC
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - leabottmer/multi-level-sc-estimator (mlSC_estimator, cvxpy+SCS)
     - —
     - 4
     - 1e-06
     - exact — matches to display precision
     - `mlsc_bottmer <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/mlsc_bottmer.py>`__

.. _val-microsynth:

MicroSynth
----------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - R package microsynth (via Rscript)
     - —
     - 6
     - 0.097
     - tight
     - `microsynth_seattle <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/microsynth_seattle.py>`__

.. _val-nsc:

NSC
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - Tian (2023) NSC.R (vendored, live run, captured), a*=0.3, b*=0.7
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 23
     - 1.9
     - close
     - `nsc_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/nsc_prop99.py>`__

.. _val-orthsc:

ORTHSC
------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - Fry GMM-SCE.R GMMSC() (R, live run, captured)
     - ``carbontax_fullsample_data.dta.txt`` (0df075d00fcc…)
     - 15
     - 0.037
     - close
     - `gmmsce_carbontax <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/gmmsce_carbontax.py>`__
   * - Fry OrthogonalizedSyntheticControl (R, live run, captured)
     - ``carbontax_fullsample_data.dta.txt`` (0df075d00fcc…)
     - 5
     - 0
     - exact — matches to display precision
     - `orthsc_carbontax <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/orthsc_carbontax.py>`__

.. _val-pda:

PDA
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - Authors' Fun/L2relax.R (ishwang1/L2relax-PDA), per UK firm, reproduced via cvxpy/ECOS, live run captured
     - ``brexit_long.parquet`` (1e5997075c1e…)
     - 4
     - 1e-06
     - exact — matches to display precision
     - `pda_brexit <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pda_brexit.py>`__
   * - R package pampe (pampe(), live run, captured)
     - ``HongKong.csv`` (ad5b35ff563a…)
     - 6
     - 3.6e-05
     - exact — matches to display precision
     - `pda_hcw_hongkong <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pda_hcw_hongkong.py>`__
   * - Authors' Fun/L2relax.R (ishwang1/L2relax-PDA) reproduced via cvxpy/ECOS, live run captured
     - ``HongKong.csv`` (ad5b35ff563a…)
     - 24
     - 0
     - exact — matches to display precision
     - `pda_hongkong <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pda_hongkong.py>`__
   * - Shi & Huang fsPDA application script (zhentaoshi/fsPDA, live run, captured)
     - ``china_watches_long.csv`` (1ce8146af9a9…)
     - 4
     - 0.056
     - close
     - `pda_luxurywatch <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pda_luxurywatch.py>`__
   * - Authors' Fun/L2relax.R (ishwang1/L2relax-PDA) reproduced via cvxpy/ECOS, live run captured
     - ``china_ppi_long.csv`` (cc4cda27e17b…)
     - 64
     - 1e-06
     - exact — matches to display precision
     - `pda_ppi <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pda_ppi.py>`__

.. _val-ppscm:

PPSCM
-----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - R augsynth::multisynth (live run, captured)
     - ``Teachingaugsynth.scv`` (59573a2dd46f…)
     - 49
     - 0.0022
     - tight
     - `ppscm_paglayan <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ppscm_paglayan.py>`__

.. _val-proximal:

PROXIMAL
--------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - authors' proximal code (freshtaste/proximal, cloned)
     - —
     - 3
     - 0.014
     - tight
     - `proximal_panic1907 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/proximal_panic1907.py>`__

.. _val-rescm:

RESCM
-----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - scmrelax L2RelaxationCV (Liao-Shi-Zheng; github.com/metricshilab/scmrelax = github.com/YapengZheng/Relaxed_SC; MOSEK->CLARABEL; live run, captured)
     - ``balanced_gdp.csv`` (26fee37d55d9…)
     - 6
     - 0.0013
     - tight
     - `rescm_balanced_gdp <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/rescm_balanced_gdp.py>`__
   * - scmrelax L2RelaxationCV (Liao-Shi-Zheng; github.com/metricshilab/scmrelax = github.com/YapengZheng/Relaxed_SC; MOSEK->CLARABEL; live run, captured)
     - —
     - 10
     - 0.00036
     - tight
     - `rescm_relax_ref <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/rescm_relax_ref.py>`__

.. _val-rolldid:

ROLLDID
-------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - lwdid.lwdid (Lee & Wooldridge DiD, live run, captured): prop99 common-timing (d, post, vce=None); castle staggered (gvar, control_group='never_treated', aggregate='overall', vce=None/hc3)
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 9
     - 0
     - exact — matches to display precision
     - `rolldid_lw <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/rolldid_lw.py>`__

.. _val-scmo:

SCMO
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - Tian-Lee-Panchenko Germany.R (fn_W solve.QP, live run, captured)
     - ``repgermany.csv`` (61a624e307e6…)
     - 6
     - 0.011
     - tight
     - `scmo_germany <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scmo_germany.py>`__

.. _val-scul:

SCUL
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - authors' SCUL() (R, via Rscript + glmnet)
     - —
     - 3
     - 0.14
     - tight
     - `scul_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scul_prop99.py>`__

.. _val-sdid:

SDID
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - `synth-inference/synthdid R (synthdid_estimate) <https://github.com/synth-inference/synthdid>`__
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 1
     - 0.0016
     - tight
     - `sdid_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/sdid_prop99.py>`__

.. _val-si:

SI
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - authors' SI code (INFORMS opre.2025.1590.cd), vendored benchmarks/reference/synth_iv_OR25
     - —
     - 20
     - 0
     - exact — matches to display precision
     - `si_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/si_prop99.py>`__

.. _val-snn:

SNN
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - deshen24/syntheticNN (live run, captured), SyntheticNearestNeighbors(n_neighbors=1)
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 14
     - 0
     - exact — matches to display precision
     - `snn_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/snn_prop99.py>`__

.. _val-spillsynth:

SPILLSYNTH
----------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - Melnychuk-Andrii/Spillover-SCM inclusive SCM (scm_weights/runInclusiveSCM), transcribed to NumPy
     - —
     - 4
     - 7.6
     - tight
     - `spillsynth_iscm_xval <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/spillsynth_iscm_xval.py>`__
   * - jcao0/synthetic-control-spillover MATLAB spillover.csv (CA row)
     - —
     - 13
     - 5.7e-05
     - exact — matches to display precision
     - `spillsynth_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/spillsynth_prop99.py>`__
   * - Mendez tutorial Rcpp sc_spillover (cmg777)
     - ``california_panel.csv`` (9d6a73e21f1a…)
     - 4
     - 3.4
     - documented — see notes
     - `spillsynth_prop99_sar <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/spillsynth_prop99_sar.py>`__
   * - Sakaguchi-Tagawa RcppArmadillo sc_spillover (method=sar, live run on the nonproprietary panel, captured)
     - ``sudan_panel.csv`` (722471a42b6c…)
     - 5
     - 0.41
     - close
     - `spillsynth_sudan <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/spillsynth_sudan.py>`__

.. _val-ssc:

SSC
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - jcao0/staggered_synthetic_control (committed results_ssc.csv / Table1_eigenvalue.csv)
     - —
     - 364
     - 0.001
     - tight
     - `ssc_guanajuato <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ssc_guanajuato.py>`__

.. _val-spsydid:

SpSyDiD
-------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - authors' SDID weight functions (serenini/spatial_SDID functions_ssdid) + the notebook's spatial WLS, via benchmarks.reference.spsydid_ref
     - —
     - 20
     - 0.094
     - close
     - `spsydid_state_mc <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/spsydid_state_mc.py>`__

.. _val-tasc:

TASC
----

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - srho1/tasc TimeAwareSC (live run, captured; em_pre, naive init, set_seed(1))
     - ``smoking_data.csv`` (a13dd4d5d6e4…)
     - 15
     - 25
     - documented — see notes
     - `tasc_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/tasc_prop99.py>`__

.. _val-vanillasc:

VanillaSC
---------

.. list-table::
   :header-rows: 1
   :widths: 22 28 8 12 14 16

   * - Reference
     - Dataset
     - #
     - max \|Δ\|
     - Verdict
     - Case
   * - R package augsynth (live run, Kansas study)
     - ``kansas_ascm.csv`` (b026c651760c…)
     - 8
     - 0.0091
     - tight
     - `ascm_kansas <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ascm_kansas.py>`__
   * - `R package scinference (sc.cf t-test, live run, captured) <https://github.com/kwuthrich/scinference>`__
     - ``carbontax_data.dta`` (815787c1e448…)
     - 3
     - 0
     - exact — matches to display precision
     - `cwz_ttest <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cwz_ttest.py>`__
   * - Malo et al. scm.corner (SCM-Debug, live run, captured)
     - ``basque_mscmt.csv`` (3aca35dc9b55…)
     - 3
     - 0.00048
     - tight
     - `malo_basque <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/malo_basque.py>`__
   * - Malo et al. scm.corner (SCM-Debug, live run, captured)
     - ``augmented_cali_long.csv`` (974ae6ad6ab7…)
     - 6
     - 0.0048
     - tight
     - `malo_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/malo_prop99.py>`__
   * - R package MSCMT (live run, captured)
     - ``basque_mscmt.csv`` (3aca35dc9b55…)
     - 4
     - 4e-05
     - exact — matches to display precision
     - `mscmt_basque <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/mscmt_basque.py>`__
   * - authors' wsoll1 (R, via Rscript + LowRankQP)
     - —
     - 3
     - 0.00098
     - exact — matches to display precision
     - `pensynth_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pensynth_prop99.py>`__
   * - scpi_pkg scdata(cointegrated_data=True)+scpi CI_all_gaussian
     - ``scpi_germany.csv`` (10b150fbcc2c…)
     - 13
     - 0.11
     - close
     - `scpi_germany_pi <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scpi_germany_pi.py>`__
   * - `Python package scpi_pkg <https://pypi.org/project/scpi-pkg/>`__
     - —
     - 15
     - 4.3e-05
     - exact — matches to display precision
     - `scpi_staggered <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scpi_staggered.py>`__
   * - `R package Synth::synth <https://CRAN.R-project.org/package=Synth>`__
     - ``california_panel.csv`` (9d6a73e21f1a…)
     - 8
     - 0.02
     - tight
     - `synth_prop99 <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/synth_prop99.py>`__
   * - Andersson (2019) AEJ:EP 11(4), Section III reported values
     - ``carbontax_data.dta`` (815787c1e448…)
     - 4
     - 0.028
     - close
     - `vanillasc_carbontax <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/vanillasc_carbontax.py>`__

Pending capture
---------------

These cross-validation cases are wired up but their reference had
not been captured when this page was last generated; the daily
action records them once its toolchain provisions.

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Case
     - Reference
   * - `dr_proximal_brazil <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/dr_proximal_brazil.py>`__
     - —
   * - `propsc_spain <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/propsc_spain.py>`__
     - —

