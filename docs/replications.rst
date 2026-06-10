.. _replications:

Replications
============

Every estimator in mlsynth is implemented from a specific published
source paper, and most of them carry an explicit replication section
in their documentation -- either reproducing an empirical result on
the original authors' dataset ("Path A"), reproducing a Monte Carlo
from the paper's simulation section ("Path B"), or matching the
output of an authoritative reference implementation
("cross-validation").

This page catalogues those replications. Thirty-five of the
thirty-six estimators are currently fully verified; ISCM carries a
one-draw illustration only (its paper relies on a non-public panel
and provides no Monte Carlo to reproduce).

* The :ref:`replications-canonical` section covers the workhorse
  SC variants you would reach for first.
* The :ref:`replications-decomp` section covers
  decomposition-first methods that match on a cycle or spectral
  component, not raw outcomes.
* The :ref:`replications-generalised` section covers estimators
  that generalise the classical setup along one axis -- a different
  estimand, treatment type, or unit scale.
* The :ref:`replications-hull` section covers estimators that
  relax the convex-hull assumption.
* The :ref:`replications-highdim` section covers high-dimensional
  donor pools.
* The :ref:`replications-time` section covers state-space and
  factor-model methods.
* The :ref:`replications-staggered`, :ref:`replications-missing`,
  and :ref:`replications-endogeneity` sections cover the
  remaining observational families.
* The :ref:`replications-design` section covers experimental
  design.

Each entry below links to the full replication code in the source
documentation page and calls out the headline number that
demonstrates the match. Replications are progressively being broken out
into **dedicated pages** (one per source paper) collected in the toctree
below; the catalogue entries link to a dedicated page where one exists.

.. toctree::
   :hidden:
   :caption: Dedicated replication pages

   replications/fdid
   replications/tssc
   replications/vanillasc
   replications/sparse_sc
   replications/nsc
   replications/sdid
   replications/mcnnm
   replications/spsydid
   replications/seq_sdid
   replications/clustersc
   replications/lexscm
   replications/scmo
   replications/rescm
   replications/linf

.. _replications-canonical:

Canonical workhorses
--------------------

* :doc:`tssc` -- Li & Shankar (2024) Two-Step Synthetic Control.
  **Path A:** ATT (+1131.97) and pre-RMSE (434.43) on the
  Brooklyn-showroom panel reproduced to three decimals; Step 1's
  recommendation matches the paper's MSC(b). **Path B:** Figure 2
  MSE-ratio grid -- all 16 cells of the
  :math:`(T_1, T_2)` sweep fall below :math:`1.0`, range
  :math:`[0.039, 0.889]`, matching the paper's geometry.
  → dedicated page: :doc:`replications/tssc`; durable cases
  ``tssc_brooklyn`` (Path A) and ``tssc_figure2`` (Path B).
* :doc:`vanillasc` -- Standard SC (ADH 2010/2015; Abadie-Gardeazabal 2003).
  **Path A:** the three canonical studies on their original datasets --
  Prop 99 (Utah/Nevada/Montana/Colorado/Connecticut, ATT :math:`\approx -19`
  packs), German reunification (Austria-dominant donor pool), Basque
  (Cataluna :math:`\approx 0.8` + Madrid :math:`\approx 0.2`, ATT
  :math:`\approx -0.68`); plus the Lei-Sudijono (2025) leave-two-out Table-1
  p-value relations. The Prop 99 fit reproduces ADH 2010 Table 2 with the full
  predictor spec (Utah 0.335, Nevada 0.236, Montana 0.202, Colorado 0.160,
  Connecticut 0.068, each within 0.005 of the paper; durable:
  ``vanillasc_prop99``).
  → dedicated page: :doc:`replications/vanillasc`.
* :doc:`masc` -- Kellogg, Mogstad, Pouliot & Torgovitsky (2020)
  matching + synthetic control. **Path A:** the KMPT Section-5
  Basque Country / ETA-terrorism study on ``basque_jasa.csv``,
  reproduced **value for value** when run as the authors did
  (MSCMT/``synth`` SC + covariate matching): the CV selects pure SC
  (:math:`\hat\varphi = 0`, as in KMPT), Cataluna + Madrid carry all
  the weight (1.00 vs 0.85 + 0.15), pre-RMSE :math:`\$89` vs
  :math:`\$94`, and ATT :math:`-\$585` vs the paper's
  :math:`-\$580`. The ``sc_backend`` and ``match_on`` toggles expose
  both optimiser/matching choices (durable: ``masc_basque``).
* :doc:`fdid` -- Li (2024) Forward Difference-in-Differences.
  **Path A:** the author's public Hong Kong GDP companion replication
  reproduced cell by cell (FDID ATT :math:`0.0254`, :math:`53.84\%`,
  pre-:math:`R^2 = 0.843`, 9 of 24 controls). **Path B:** Table 5 PMSE grid
  across DGP1-DGP4 at four :math:`(T_1, T_2)` configurations; cell
  :math:`(48, 24)` yields :math:`\mathrm{PMSE} = 0.084` against the paper's
  :math:`0.082`.
  → dedicated page: :doc:`replications/fdid`.

.. _replications-decomp:

Decomposition-first synthetic control
-------------------------------------

* :doc:`hsc` -- Harmonic Synthetic Control. **Path A:** the 1997
  Hong Kong handover -- 2003 effect :math:`-\$1{,}902` against
  the paper's :math:`-\$1{,}900`; Korea weight 0.18 and Germany
  weight 0.14 match the paper (with ``ridge="sdid"`` + a refined
  rho grid; durable: ``hsc_hongkong``). **Path B:** Liu-Xu regime-adaptation
  grid with ARIMA(1,1,0) trends -- HSC tracks the oracle-best fixed
  method in both regimes (rho adapts; durable: ``hsc_mc``).
* :doc:`sbc` -- Synthetic Business Cycle. **Path A:** German
  reunification -- Greece weight 0.44, Netherlands 0.37,
  :math:`\widehat{\mathrm{ATT}} \approx -952` (durable:
  ``sbc_germany``); also reproduces the
  Hong Kong handover. **Path B:** Shi-Xi-Xie Models 1-3
  nonstationary DGP, post-MSE ratio < 1 in every model and highest
  (SC competitive) under cointegration (durable: ``sbc_mc``).

.. _replications-generalised:

Generalising the estimand, treatment, or unit
---------------------------------------------

* :doc:`scmo` -- multi-outcome SC, both variants.
  **Path A:** Tian-Lee-Panchenko (2026) German reunification (nine
  pre-1989 indicators) -- the concatenated synthetic reproduces their
  Table 2 balance cell by cell (synthetic 1989 GDP per capita
  :math:`19029.8`; CPI :math:`3.1`; trade :math:`59.1`; tax
  :math:`34.1`), pre-RMSE :math:`= 110` (durable: ``scmo_germany``).
  **Path B (concatenated):** TLP Table 1 == Sun et al. ``Simulation1.R``
  -- bias falls and pre-fit rises with the outcome count :math:`K` across
  :math:`T_0 \in \{1, 5, 10\}` (durable: ``scmo_concatenated_mc``).
  **Path B (averaged):** Sun-Ben-Michael-Feller (2025) Appendix-D regime
  contrast -- averaging beats the separate SC under a common factor and
  hurts under purely idiosyncratic factors (durable: ``scmo_averaged_mc``).
  → dedicated page: :doc:`replications/scmo`.
* :doc:`rescm` -- SCM-relaxation (Liao, Shi & Zheng 2026).
  **Path A:** Brexit / UK real GDP -- the L2 relaxation's cumulative
  GDP loss is :math:`\approx 4.0\%` (paper 3.85%, treatment 2016Q3),
  dense weights on the major EU economies that classic SC drops
  (durable: ``rescm_brexit``). **Cross-validation:** mlsynth's L2
  relaxation matches the authors' ``scmrelax`` package cell by cell at a
  matched :math:`\tau` (donor-weight :math:`L_1` distance
  :math:`\approx 0.0014`; durable: ``rescm_relax_ref``).
  **Path B:** the Section-5 latent-group Monte Carlo -- with
  :math:`J \gg T_0`, the L2 relaxation's out-of-sample error against the
  oracle counterfactual is :math:`\approx 0.43` of classic SC's (median;
  paper's Table 1 :math:`\approx 0.15`--:math:`0.53`), beating SC in
  :math:`\approx 73\%` of reps (durable: ``rescm_relax_mc``).
  → dedicated page: :doc:`replications/rescm`.
* :doc:`rescm` -- L-infinity-norm SC (Wang, Xing & Ye 2025;
  ``LINF`` / ``L1LINF``). **Cross-validation:** mlsynth's L-infinity
  engine matches the authors' ``LinfinitySC`` code cell by cell in the
  unique :math:`T_0>J` regime (donor-weight :math:`L_1` distance
  :math:`\approx 0.0019`; durable: ``linf_crossval_ref``). **Path A:**
  Proposition 99 -- dense weighting across all 38 donors vs classic SC's
  6 (durable: ``linf_prop99``). **Path B:** the Section 5 two-factor
  Monte Carlo -- L-infinity beats sparse SC in the dense DGPs, ``L1LINF``
  in the sparse one (durable: ``linf_sim``).
  → dedicated page: :doc:`replications/linf`.
* :doc:`ctsc` -- Continuous-Treatment SC. **Path B:** Section 5 /
  Table 1 Monte Carlo across Models 1-4 with factor structure and
  true average effect :math:`= 0`; CTSC bias :math:`\approx 0.00`
  against the paper's :math:`0.005`-:math:`0.011` range, vs.
  fixed-effects bias :math:`\approx 0.80`.
* :doc:`dsc` -- Distributional SC. **Path B:** Gunsilius (2023)
  Figure 4 reproduction -- the 2-Wasserstein-squared distance
  collapses at :math:`J = 30` against the rough fit at
  :math:`J = 4`, and a location-shift planted effect of
  :math:`+1.5` is recovered.
* :doc:`si` -- Synthetic Interventions. **Path A:** California
  Proposition 99 multi-arm counterfactual reproduces the paper's
  reported control/tax-only/full-program values 75.8 / 57.5 / 59.1.
  **Path B:** Gavish-Donoho :math:`k = 5` for control and
  :math:`k = 1` for the tax / program arms, matching the paper.
* :doc:`microsynth` -- User-Level SC. **Path B:** Section 5
  contamination-recovery study, 200 reps with true lift
  :math:`+0.05`; MicroSynth bias :math:`+0.0028` vs. ITT bias
  :math:`-0.0181` and naive TOT bias :math:`+0.0291`.

.. _replications-hull:

Convex-hull relaxation
----------------------

* :doc:`iscm` -- Imperfect Synthetic Controls. **Example:**
  one-factor panel with planted effect 3.0 recovered as
  :math:`\widehat{\mathrm{ATT}} \approx 3.0` and a small outside-
  hull fit metric, illustrating the all-units-as-synthetic-
  controls construction. *Full Monte Carlo replication queued.*
* :doc:`nsc` -- Nonlinear SC. **Cross-validation:** matched to
  Tian's (2023) own R implementation on the Proposition 99 panel
  (Table 2) -- per-donor NSC weights track the author's to
  correlation 0.989 (max :math:`|\Delta| = 0.024`), recovering the
  same signed donor pool, with an average effect :math:`-19.1` and
  an effect path (:math:`-9.1/-22.6/-27.0` in 1990/1995/2000)
  matching the paper's :math:`-9.5/-24.5/-28.7`. **Path B:** the
  nonlinear-outcome Monte Carlo (Section 4, Table 1) -- on the
  :math:`r = 2` panel NSC's 95% CI covers near nominal
  (:math:`0.93/0.97` at :math:`J = 25/50`, paper
  :math:`0.935/0.950`) and the error shrinks as the donor pool
  grows. → dedicated page: :doc:`replications/nsc`; durable cases
  ``nsc_prop99``, ``nsc_mc``.

.. _replications-highdim:

High-dimensional donor pools
----------------------------

* :doc:`bvss` -- Xu & Zhou (2025) Bayesian SC with soft simplex.
  **Path A:** China anti-corruption study on luxury-watch imports
  (87 commodity-category donors) -- ATT = -0.020 (95% credible
  interval :math:`[-0.033, -0.003]`) matches the paper's
  :math:`-0.021\ ([-0.032, -0.008])`.
* :doc:`clustersc` -- Cluster-then-pool SC. **Path A (PCR-SC):**
  California Proposition 99 -- rank-1 PCR
  :math:`\widehat{\mathrm{ATT}}` in the :math:`[-19, -24]` range
  matching the classical ADH baseline. **Path A (RPCA-SC):** West
  German reunification -- Norway weight 0.485, France 0.354,
  pre-RMSE :math:`\approx 89` matching Bayani's reference figures
  (durable: ``clustersc_rpca_germany``).
  **Path B:** Amjad-Shah-Shen periodicity DGP plus a two-factor
  DGP for missing-data robustness. **Path B (subgroups,
  cross-validated):** Rho et al. (2025) ClusterSC -- in the
  high-dimensional-subgroup regime (pooled rank :math:`>T_0`)
  ClusterSC beats whole-pool RSC at every noise level (MSE down
  :math:`60.8\% / 43.2\% / 24.3\%` at :math:`\sigma = 0.10/0.25/0.40`),
  and the authors' own code reproduces its :math:`\approx 50\%`
  headline on its own DGP. **Path B (RSC, Amjad-Shah-Shen 2018):**
  the PCR-SC training error approximates its generalization error
  (gen/train :math:`\approx 1.0\text{-}1.15`) across noise, the RSC
  Table-1 finding. **Cross-validation (Shen et al. CIs):** mlsynth's
  variance code matches deshen24/panel-data-regressions to machine
  precision, and the doubly-robust CI attains :math:`\approx 95\%`
  coverage for all three estimands where a single-source CI drops to
  :math:`63\%`.
  → dedicated page: :doc:`replications/clustersc`.
* :doc:`mlsc` -- Bottmer (2024) multi-level SC. **Path A:**
  matched to the reference implementation on the README DGP --
  :math:`\widehat{\mathrm{ATT}} = +0.011930`,
  :math:`\lambda = 1.970185`, equal to six decimals.
  **Path B:** 200 independent draws,
  :math:`\max |\Delta| = 5.76 \times 10^{-4}`.
* :doc:`pda` -- Shi & Huang (2023) panel data approach.
  **Path A:** Hong Kong economic integration (HCW 24-economy
  panel, quarterly GDP growth) -- all three methods recover a
  positive, highly significant CEPA effect, with the
  L2-relaxation estimate (2.48%, :math:`t \approx 7.7`)
  reproducing Shi & Wang's Appendix-E.1 headline of
  :math:`+2.65\%` (:math:`t = 8.35`); LASSO and forward
  selection bracket it at 3.3% and 3.9% (durable:
  ``pda_hongkong``). **Path B:** Table 1
  forward-selection vs. LASSO at 300 reps -- size = 0.047 vs.
  the paper's :math:`\approx 0.05`; power = 0.98 at D5 vs.
  :math:`1.0`.
* :doc:`fscm` -- Cerulli (2024) forward-selected SC. **Path A:**
  Proposition 99 --
  :math:`\widehat{\mathrm{ATT}} = -20.15`,
  pre-:math:`R^2 = 0.970`, CV RMSPE :math:`= 1.605` vs. the
  full-pool baseline 2.916 (durable: ``fscm_prop99``).
* :doc:`sparse_sc` -- Sparse SC. **Path A:** California
  Proposition 99 with an over-rich augmented predictor set, with
  conformal inference -- the L1 penalty prunes to 6 of 33 predictors
  and the effect lands at :math:`-17.9` packs (95% CI
  :math:`[-21.3, -15.4]`) on the ADH donor pool (durable:
  ``sparse_sc_prop99``). See the
  :doc:`dedicated page <replications/sparse_sc>`.

.. _replications-time:

Time-aware and factor models
----------------------------

* :doc:`fma` -- Li & Sonnier (2023) factor model approach.
  **Path B:** coverage study under DGP1 (stationary) and DGP2
  (non-stationary), 1000 reps; coverage at :math:`M = 1000`
  equals 0.947 against the nominal 0.95.
* :doc:`tasc` -- Rho et al. (2026) time-aware SC. **Path A:**
  Proposition 99 -- pre-RMSE 0.767, ATT -16.793, gap of -24
  packs by 2000 against the paper's Figure 10 gap of -25 to -30.
  **Path B:** the Section-5.2 :math:`(q, r)` state-space ablation
  (Figures 3-4) -- TASC has the smaller median counterfactual RMSE
  than vanilla SC in **all four** regimes (ratio 0.84-0.93),
  strongest under high observation noise, the paper's headline
  (durable: ``tasc_mc``).

.. _replications-staggered:

Staggered adoption
------------------

* :doc:`sdid` -- Arkhangelsky et al. (2021) synthetic
  difference-in-differences. **Path A:** California Proposition
  99 -- :math:`\widehat{\mathrm{ATT}} = -15.605` matches the
  paper's :math:`-15.6` (and the ``synthdid`` R-package's
  :math:`-15.604`) to three significant figures; placebo SE
  :math:`7.58` against the paper's :math:`8.4`.
  **Cross-validation:** matched to ``causaltensor.SDID`` on the same
  matrix to :math:`|\Delta| = 3.1\times 10^{-3}`.
  → dedicated page: :doc:`replications/sdid`.
* :doc:`spsydid` -- Spatial Synthetic-DiD. **Path B
  (cross-validation):** the authors' State-Level Monte Carlo
  (serenini/spatial_SDID) run per-rep against their own algorithm --
  per-rep ATT correlation 0.996, mean ATT bias :math:`\approx 0.02`
  for both implementations (the paper's headline unbiasedness).
  → dedicated page: :doc:`replications/spsydid`.
* :doc:`spotsynth` -- O'Riordan & Gilligan-Lee (2025) spillover
  detection for donor selection. **Path B:** the Figure 2 bias study
  on the Appendix B DGP -- a synthetic control on all donors is biased
  (:math:`\approx +1.6`), one on valid donors is unbiased, and the S1 /
  S2 screens recover most of the gap, degrading with noise. **Path A
  (semi-synthetic):** the Figure 6 demonstrations -- a planted
  noisy-proxy donor that biases California tobacco (:math:`-1.4` vs the
  canonical :math:`-20.5`) and German reunification toward zero is
  flagged and excluded, restoring the effect; both run through
  ``SPOTSYNTH.fit()``.
* :doc:`ppscm` -- Ben-Michael, Feller & Rothstein (2022)
  partially pooled SC. **Cross-validation:** matched end-to-end
  to the ``augsynth`` R-package vignette (:math:`\nu = 0.2607`,
  average ATT :math:`= -0.011`) to four decimals.
* :doc:`ssc` -- Cao, Lu & Wu (2026) staggered synthetic control.
  **Path A:** the Guanajuato police-reform application (Section 4;
  :math:`N = 33`, 10 staggered adopters) -- event-time ATT estimates
  match the authors' reference output for all seven outcomes, to
  :math:`\approx 10^{-4}` for the homicide (:math:`T_0 = 174`) and
  theft rates and :math:`\approx 10^{-3}` for the annual cartel
  outcomes, with end-of-sample bands present/``NaN`` exactly as in
  the reference. **Path B:** the paper's staggered AR(1)-factor DGP
  (Section 3) -- the event-time ATT path :math:`\tau = 1 + e` is
  recovered, using all units (including not-yet-treated) as donors.
* :doc:`seq_sdid` -- Arkhangelsky & Samkov (2025) Sequential
  Synthetic DiD. **Path B:** the Section-5.2.2 calibrated-panel
  Monte Carlo (Table 1) reconstructed from the paper's description
  (the authors' CPS log-wage panel is not public) -- under an
  interactive-fixed-effects violation of parallel trends, standard
  DiD's 95% CI coverage collapses (:math:`\approx 0.45`, paper
  :math:`\approx 0.70`) while Sequential SDiD stays near nominal
  (:math:`0.945`) with roughly five-times-smaller bias and lower
  RMSE -- the paper's "DiD severely biased, Sequential SDiD reliable"
  headline (durable: ``seq_sdid_mc``). A noiseless rank-one IFE
  corollary pins exact machine-precision recovery for every
  donor-balanced cohort.
  → dedicated page: :doc:`replications/seq_sdid`.

.. _replications-missing:

Missing data
------------

* :doc:`mcnnm` -- Athey, Bayati, Doudchenko, Imbens & Khosravi
  (2021) matrix-completion SC. **Path A:** Proposition 99 --
  :math:`\widehat{\mathrm{ATT}} \approx -20` packs per capita,
  consistent with classical SC, FSCM and SNN estimates on the
  same panel. **Cross-validation:** matched to
  ``causaltensor``'s MC-NNM ATT (-19.83 vs -20.27, :math:`\approx`
  2%; estimand-level, given the differing FE sub-solvers).
  → dedicated page: :doc:`replications/mcnnm`.
* :doc:`snn` -- Synthetic Nearest Neighbours.
  **Cross-validation:** Proposition 99 --
  :math:`\widehat{\mathrm{ATT}} \approx -19` packs/capita,
  matching the Abadie-Diamond-Hainmueller (2010) reference
  baseline.
* :doc:`rmsi` -- Agarwal, Choi & Yuan (2026) robust matrix
  estimation with side information. **Path A:** Proposition 99
  with the Abadie covariates as side information --
  :math:`\widehat{\mathrm{ATT}} \approx -21` packs/capita
  (widening to :math:`\approx -32` by 2000), matching the ADH
  baseline. **Path B:** the paper's four-component MNAR Monte
  Carlo (Section 5.1) -- RMSI's missing-block AMSE is lower than
  the no-side-information baseline, the paper's central finding.

.. _replications-endogeneity:

Identification under endogeneity
--------------------------------

* :doc:`siv` -- Gulek & Vives (2024) synthetic IV. **Path A:**
  Autor-Dorn-Hanson China-shock 2SLS (Table 3) reproduced
  exactly on the public replication archive -- coefficients
  :math:`-0.888 / -0.718 / -0.746` against
  :math:`-0.89 / -0.72 / -0.75`. **Path B:** Section 6 Syrian-
  calibrated Monte Carlo across :math:`r \in \{0.5, 0.7, 0.9\}`
  at 200 reps; TSLS-TWFE bias 0.111 / 0.228 / 0.387 matches the
  paper's 0.111 / 0.218 / 0.360 essentially exactly.
* :doc:`proximal` -- Proximal SC. **Path A:** Panic of 1907 on
  the Trust panel, reproduced across all six proximal variants
  -- PI :math:`-1.148` vs. paper :math:`-1.138`; PIS
  :math:`-1.148` vs. :math:`-1.134`; PIPost :math:`-1.220` vs.
  :math:`-1.220`. **Path B:** Single-Proxy SC (SPSC) plus
  Doubly-Robust and PIPW Monte Carlo --
  SPSC-DT :math:`\widehat{\mathrm{ATT}} = -0.815` against
  :math:`-0.816`.

.. _replications-design:

Experimental design
-------------------

* :doc:`syndes` -- Abadie & Zhao (2025) synthetic design.
  **Path B:** Section 5 ablation across the three design types
  (``per_unit`` / ``two_way_global`` / ``one_way_global``) at 40
  reps under AR(1) factors; ``per_unit`` design RMSE = 0.098
  against a random-DiM baseline of 0.982, power = 0.50 at
  MDE = 0.157.
* :doc:`marex` -- Abadie & Zhao (2025) market-exclusion design.
  **Path A:** Walmart 45-store weekly-sales placebo design --
  pre-fit RMSE 2.2%, placebo ATT :math:`-1.0%`, p-value 0.937
  against the paper's 0.933. **Path B:** Lin-factor DGP recovers
  the treatment effect across the MAE-vs-effect-scale sweep.
* :doc:`pangeo` -- Parallel-Trends Supergeo Design. **Path B:**
  seasonal-sales panel with up-trend, down-trend, and cyclical
  trajectories; PANGEO RMSE :math:`\approx 0.2` against scalar-
  matching RMSE :math:`\approx 6` at :math:`\tau = 4` (a
  :math:`30\times` improvement); pre-period parallelism
  :math:`R^2 \in [0.90, 0.98]`.
* :doc:`spcd` -- Synthetic Principal Component Design.
  **Path B:** Lu-Li-Ying-Blanchet linear-factor model with
  :math:`\tau = 1`, :math:`\sigma = 1` -- SPCD mean
  :math:`\widehat{\mathrm{ATT}} \approx 1.0`, RMSE
  :math:`\approx 0.4` against the random-design baseline RMSE
  :math:`\approx 3.9` (a :math:`10\times` improvement).
* :doc:`lexscm` -- Lexicographic synthetic experimental design
  (Abadie-Zhao 2026; Vives-i-Bastida 2022). **Path A:** the Walmart
  45-store placebo design -- pre-fit RMSE :math:`\approx 2.7\%` of
  mean sales, placebo effect :math:`\approx 0.9\%`, permutation
  p :math:`\approx 0.63` (CI covers zero), the paper's "no spurious
  effect" result. **Path B:** the Abadie-Zhao Section-5 linear-factor
  design simulation (exact params: :math:`J=15`, :math:`T_E=20`) -- the
  design recovers the planted effect with MAE/scale :math:`\approx 0.24`
  for the single-treated-unit design falling to :math:`\approx 0.16` at
  :math:`m=2` (Table-2 monotonicity).
  → dedicated page: :doc:`replications/lexscm`.

Coverage summary
----------------

.. list-table:: Verification coverage by family
   :header-rows: 1
   :widths: 32 14 14 40

   * - Family
     - Verified
     - In family
     - Status
   * - Canonical workhorses
     - 2
     - 2
     - Complete (TSSC, FDID)
   * - Decomposition-first
     - 2
     - 2
     - Complete (HSC, SBC)
   * - Generalised estimand / treatment / unit
     - 5
     - 5
     - Complete (SCMO, CTSC, DSC, SI, MicroSynth)
   * - Convex-hull relaxation
     - 1
     - 2
     - NSC ✓ (cross-validated vs author's R + Path-B MC);
       ISCM one-draw only (non-public panel, no MC)
   * - High-dimensional donors
     - 7
     - 7
     - Complete (BVSS, CLUSTERSC, MLSC, PDA, RESCM, FSCM,
       SPARSE_SC)
   * - Time-aware / factor models
     - 2
     - 2
     - Complete (FMA, TASC)
   * - Staggered adoption
     - 5
     - 5
     - Complete (SDID, SpSyDiD, PPSCM, SSC, SEQ_SDID)
   * - Spillover-aware (donor screening)
     - 1
     - 1
     - Complete (SPOTSYNTH; SpSyDiD counted under staggered)
   * - Missing data
     - 3
     - 3
     - Complete (MCNNM, SNN, RMSI)
   * - Identification under endogeneity
     - 2
     - 2
     - Complete (SIV, PROXIMAL)
   * - Experimental design
     - 5
     - 5
     - Complete (LEXSCM, MAREX, SYNDES, PANGEO, SPCD)

Of mlsynth's 36 estimators, 35 (97%) carry a strong or solid
replication against their source paper or against an
authoritative reference implementation. Only ISCM remains a
one-draw illustration -- its paper relies on a non-public panel
and provides no Monte Carlo to reproduce.

Contributing a replication
--------------------------

If you have used an mlsynth estimator and replicated a paper's
number in the course of your work, please open an issue or pull
request on `GitHub
<https://github.com/jgreathouse9/mlsynth/issues>`_. The two
requirements are:

1. The replication target is a published number -- a row of a
   table, a point on a figure, or a key statistic in the prose.
2. The replication is runnable from a copy of the documentation
   snippet plus the source dataset (CSV, R-package data, or a
   simulator we ship).

A reference Path A entry is :doc:`tssc`'s Brooklyn-showroom
section; a reference Path B entry is :doc:`tasc`'s Section 5
grid; a reference cross-validation entry is :doc:`ppscm`'s
``augsynth``-vignette comparison.
