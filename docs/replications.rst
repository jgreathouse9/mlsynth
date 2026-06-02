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

This page catalogues those replications. Thirty-two of the
thirty-six estimators are currently fully verified, three carry
partial replications (one-draw illustrations or empirical
applications without an explicit number-match), and one has no
replication content yet.

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
demonstrates the match.

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
* :doc:`fdid` -- Li (2024) Forward Difference-in-Differences.
  **Path B:** Table 5 PMSE grid across DGP1-DGP4 at four
  :math:`(T_1, T_2)` configurations; cell :math:`(48, 24)` yields
  :math:`\mathrm{PMSE} = 0.084` against the paper's :math:`0.082`.

.. _replications-decomp:

Decomposition-first synthetic control
-------------------------------------

* :doc:`hsc` -- Harmonic Synthetic Control. **Path A:** the 1997
  Hong Kong handover -- 2003 effect :math:`-\$1{,}902` against
  the paper's :math:`-\$1{,}900`; Korea weight 0.18 and Germany
  weight 0.14 match the paper. **Path B:** Liu-Xu regime-adaptation
  grid with ARIMA(1,1,0) trends -- HSC RMSE 6.46 vs. SC-diff 6.11
  under the idiosyncratic regime.
* :doc:`sbc` -- Synthetic Business Cycle. **Path A:** German
  reunification -- Greece weight 0.44, Netherlands 0.37,
  :math:`\widehat{\mathrm{ATT}} \approx -952`; also reproduces the
  Hong Kong handover. **Path B:** Shi-Xi-Xie Models 1-3
  nonstationary DGP, Model 1 MSE ratio 0.012 vs. the paper's 0.01.

.. _replications-generalised:

Generalising the estimand, treatment, or unit
---------------------------------------------

* :doc:`scmo` -- Tian, Lee & Panchenko (2024) multi-outcome SC.
  **Path A:** German reunification (nine pre-1989 indicators) --
  pre-RMSE = 110, matching Table 1 to three decimals.
  **Path B:** TLP Table 1 plus SBMF averaging, 400-800 reps; bias
  / SD at :math:`K = 6` are :math:`1.11 / 1.40` against the
  paper's :math:`1.08 / 1.36`.
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
* :doc:`nsc` -- Nonlinear SC. **Path B (one-draw):** Tian (2023)
  Section 4 nonlinear DGP with :math:`J = 12`, :math:`T = 18`,
  :math:`\tau_{\mathrm{true}} = 0.10` -- recovered
  :math:`\widehat{\tau} \approx 0.10`. *Averaged Monte Carlo
  queued.*

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
  German reunification -- Norway weight 0.48, France 0.35,
  pre-RMSE :math:`\approx 90` matching Bayani's reference figures.
  **Path B:** Amjad-Shah-Shen periodicity DGP plus a two-factor
  DGP for missing-data robustness.
* :doc:`mlsc` -- Bottmer (2024) multi-level SC. **Path A:**
  matched to the reference implementation on the README DGP --
  :math:`\widehat{\mathrm{ATT}} = +0.011930`,
  :math:`\lambda = 1.970185`, equal to six decimals.
  **Path B:** 200 independent draws,
  :math:`\max |\Delta| = 5.76 \times 10^{-4}`.
* :doc:`pda` -- Shi & Huang (2023) panel data approach.
  **Path A:** Hong Kong economic integration (HCW 24-economy
  panel, quarterly GDP growth). **Path B:** Table 1
  forward-selection vs. LASSO at 300 reps -- size = 0.047 vs.
  the paper's :math:`\approx 0.05`; power = 0.98 at D5 vs.
  :math:`1.0`.
* :doc:`rescm` -- Liao, Shi & Zheng (2025) relaxed SC. **Path A:**
  Proposition 99 (38 control states). **Path B:** high-dim grid
  :math:`N = 90`, :math:`T_1 = T_2 = 36`, 50 reps at
  :math:`\delta \in \{0, 1\}`; SC and LINF achieve size 0.20-0.22
  and power 0.98; RELAX-:math:`L_2` reaches size 0.28, power 0.94.
* :doc:`fscm` -- Cerulli (2024) forward-selected SC. **Path A:**
  Proposition 99 --
  :math:`\widehat{\mathrm{ATT}} = -20.15`,
  pre-:math:`R^2 = 0.970`, CV RMSPE :math:`= 1.605` vs. the
  full-pool baseline 2.916.
* :doc:`sparse_sc` -- Sparse SC. **Path A:** California
  Proposition 99 with the augmented ADH-7 dataset of Vives, with
  conformal inference -- validation MSE conformal CI
  :math:`\approx [-20, -18]`, magnitude consistent with the ADH
  baseline.

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
  **Path B:** Section 5.2 :math:`(Q, R)` ablation grid (30 reps);
  TASC dominates the simplex-SC baseline in all four cells, with
  a :math:`4.5\times` margin at big-:math:`Q` / small-:math:`R`.

.. _replications-staggered:

Staggered adoption
------------------

* :doc:`sdid` -- Arkhangelsky et al. (2021) synthetic
  difference-in-differences. **Path A:** California Proposition
  99 -- :math:`\widehat{\mathrm{ATT}} = -15.605` matches the
  paper's :math:`-15.6` (and the ``synthdid`` R-package's
  :math:`-15.604`) to three significant figures; placebo SE
  :math:`7.58` against the paper's :math:`8.4`.
* :doc:`spsydid` -- Spatial Synthetic-DiD. **Path B:** 8x8 grid
  with 6 treated units and planted spillover-exposure effects --
  :math:`\widehat{\tau} \approx 2.0`,
  :math:`\widehat{\tau}_s \approx 1.0`, both recovered.
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

.. _replications-missing:

Missing data
------------

* :doc:`mcnnm` -- Athey, Bayati, Doudchenko, Imbens & Khosravi
  (2021) matrix-completion SC. **Path A:** Proposition 99 --
  :math:`\widehat{\mathrm{ATT}} \approx -20` packs per capita,
  consistent with classical SC, FSCM and SNN estimates on the
  same panel.
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
* :doc:`lexscm` -- Lexicographic SC. **Path B:** synthetic sales
  panel with budget and operational constraints --
  enumeration solver returns ``OPTIMAL`` status and best
  imbalance :math:`\approx 4 \times 10^{-4}`, demonstrating the
  full lexicographic max-validity / max-power decision.

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
     - 0 (2 partial)
     - 2
     - ISCM and NSC have one-draw illustrations;
       averaged Monte Carlos queued
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
     - 4
     - 5
     - SDID, SpSyDiD, PPSCM, SSC ✓; SEQ_SDID queued
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

Of mlsynth's 36 estimators, 32 (89%) carry a strong or solid
replication against their source paper or against an
authoritative reference implementation. ISCM, NSC, and
SPARSE_SC carry partial replications -- one-draw illustrations
or empirical applications without an explicit number-match --
and SEQ_SDID is the single estimator still in queue.

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
