.. _replication-clustersc:

ClusterSC — Synthetic Control with Donor Selection (Rho et al. 2025)
====================================================================

:Estimator: :doc:`../clustersc` — :class:`mlsynth.CLUSTERSC`
:Source: Rho, S., Tang, A., Bergam, N., Cummings, R., & Misra, V. (2025),
   *"ClusterSC: Advancing Synthetic Control with Donor Selection,"* arXiv:2503.21629.
:Replication type: **Path B** — the paper's synthetic Monte Carlo (Section 6.1),
   exercising both estimator modes, with a **cross-validation** of the authors'
   reference code against their own headline.
:Status: **Verified** — mlsynth reproduces the paper's central claim in the
   high-dimensional-subgroup regime; the authors' code reproduces its ~50%
   headline on its own DGP.

Two estimators for the price of one
-----------------------------------

mlsynth's single ``CLUSTERSC`` estimator covers both halves of the paper:

* ``clustering=False`` is plain **RSC / PCR-SC** (Amjad-Shah-Shen 2018:
  HSVT-denoise the donor pool to rank ``r``, then OLS) — the paper's *benchmark*.
* ``clustering=True`` is **ClusterSC** — k-means the donors in SVD-feature space
  (Algorithm 3), keep the target's subgroup, then run RSC on it (Algorithm 4).

So one benchmark validates both the baseline and the contribution.

The regime that matters
-----------------------

The paper motivates ClusterSC by the **curse of dimensionality**: with many
donors, the whole-pool regression lives in a high-dimensional, noisy space.
mlsynth's RSC, however, denoises the *pre-period* donor matrix to a fixed rank
*before* regressing, so it is already robust to the raw donor count ``n`` — and
on the paper's own two-subgroup DGP (rank 3 + 3) clustering buys mlsynth almost
nothing, because its rank-6 whole-pool fit already captures the structure.

The lever that genuinely exercises the paper's argument is the **pooled signal
rank**. With ``K`` well-separated subgroups of rank ``r``, the pooled donor
matrix has rank ``K·r``; once ``K·r`` exceeds the pre-period length ``T0`` the
whole-pool fit *must* under-denoise, while each subgroup stays low-rank and
well-conditioned. We use ``K=6`` subgroups of rank 3 with ``T0=8`` (pooled rank
18 ≫ 8). In this regime mlsynth's ClusterSC clearly beats its whole-pool RSC at
every noise level (placebo DGP, true effect 0, so post-period MSE is pure
prediction error):

.. list-table:: Median post-period test MSE (mlsynth, 25 placebo targets, seed 0)
   :header-rows: 1
   :widths: 18 22 22 20

   * - Noise σ
     - RSC (whole pool)
     - ClusterSC
     - MSE reduction
   * - 0.10
     - 0.0299
     - 0.0117
     - **60.8%**
   * - 0.25
     - 0.1161
     - 0.0659
     - **43.2%**
   * - 0.40
     - 0.2495
     - 0.1888
     - **24.3%**

The durable check is ``benchmarks/cases/clustersc_subgroups.py``::

   python benchmarks/run_benchmarks.py --case clustersc_subgroups

It asserts ``clustering_wins_all == 1`` (ClusterSC beats RSC at every noise) plus
a positive floor on the smallest gain. The DGP is a faithful ``K``-subgroup
generalisation of the authors' two-subgroup sine mixture, in
:func:`mlsynth.utils.clustersc_helpers.simulation.simulate_subgroup_panel`.

Cross-validation against the authors' code
------------------------------------------

mlsynth and the authors' reference implementation
(https://github.com/srho1/ClusterSC) make different, individually-valid
modelling choices:

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Step
     - mlsynth (paper Algorithm 3/4)
     - reference code
   * - Clustering features
     - rank-``r`` truncated ``UΣ_r`` of the **pre-period** donors
     - **full, untruncated** ``UΣ`` of the whole panel
   * - Denoising
     - pre-period only (no post leakage)
     - full panel
   * - Weight fit
     - OLS via pseudo-inverse, **no intercept**
     - OLS **with intercept**

Because of this, the two are strongest in *different* regimes — clustering helps
mlsynth where the pooled rank exceeds ``T0``, and helps the reference on its own
two-subgroup DGP (where its full-panel-plus-intercept RSC baseline is weaker). A
per-target numeric cross-validation between the two implementations is therefore
**not meaningful**, and we do not assert one.

What *is* a clean cross-check is the **authors' code against the authors' paper**:
``benchmarks/cases/clustersc_subgroups_ref.py`` clones the reference repo (pinned
commit, MIT-licensed, imported not vendored) and runs its RSC / ClusterSC on its
own ``generate_sine_dataset_A`` / ``_B`` DGP, confirming the paper's headline that
clustering substantially lowers test MSE (single seed, 30 targets):

.. list-table:: Median test-MSE reduction, authors' code on authors' DGP
   :header-rows: 1
   :widths: 18 26

   * - Noise σ
     - MSE reduction
   * - 0.10
     - 38.8%
   * - 0.25
     - 71.3%
   * - 0.40
     - 70.6%

(The paper reports ~50% at n=1000 over 500 reps; the single-seed median over 30
targets is noisier but unambiguously large and positive at every noise level.)

Run it with::

   pip install kneed scikit-learn
   python benchmarks/run_benchmarks.py --case clustersc_subgroups_ref

It skips gracefully when the repo cannot be cloned or ``syclib`` / ``kneed`` are
unavailable.

RSC pre/post test error (Amjad-Shah-Shen 2018)
----------------------------------------------

The PCR-SC path is also benchmarked against the **Robust Synthetic Control**
paper (Amjad, Shah & Shen 2018, JMLR 19:1-51) that underpins it. Section 5.3,
Table 1 reports that on a low-rank latent-variable panel the **pre-intervention
MSE (training error) approximates the post-intervention MSE (generalization
error)** -- so the in-sample pre-fit honestly predicts out-of-sample forecast
accuracy. Both errors are taken against the *true* (noise-free) mean, which the
DGP exposes; mlsynth's RSC (``CLUSTERSC`` with ``clustering=False``) reproduces
this at every noise level:

.. list-table:: PCR-SC error vs the true mean (N=100, T=2000, T0=1600, rank 3)
   :header-rows: 1
   :widths: 16 22 22 18

   * - Noise σ
     - Training (pre) MSE
     - Generalization (post) MSE
     - gen / train
   * - 3.1
     - 0.176
     - 0.202
     - 1.15
   * - 1.3
     - 0.041
     - 0.044
     - 1.08
   * - 0.4
     - 0.0043
     - 0.0044
     - 1.03

The ratio sits just above 1 throughout, matching the paper's "training error
approximates generalization error" finding; the absolute magnitudes depend on
the (paper-underspecified) truncation rank, so the durable check
(``benchmarks/cases/rsc_synth_error.py``) pins the *ratio* and noise-monotonicity
rather than Table 1's exact cells. The DGP is
:func:`mlsynth.utils.clustersc_helpers.simulation.simulate_rsc_panel`.

Confidence-interval coverage (Shen et al.)
------------------------------------------

mlsynth's frequentist PCR-SC confidence intervals port the variance estimators of
Shen et al.'s *Same Root Different Leaves* (the
https://github.com/deshen24/panel-data-regressions reference). Two cross-checks
in ``benchmarks/cases/rsc_shen_coverage.py``:

* **Variance cross-validation.** On identical resampled inputs, mlsynth's
  ``_var_homo`` / ``_var_jack`` equal the reference ``var.py`` to machine
  precision (max :math:`|\Delta| \approx 4\times 10^{-16}` / exactly 0).
* **Coverage validity.** Reproducing the repo's ``simulation.py`` Monte Carlo
  (calibrated to Prop 99), the **doubly-robust** variance is approximately valid
  for *all three* estimands, while a single-source variance under-covers the
  estimand it is not built for:

  .. list-table:: 95%-CI coverage (seed 0, 500 reps)
     :header-rows: 1
     :widths: 26 18 18 18

     * - Variance
       - μ_hz
       - μ_vt
       - μ_dr
     * - doubly robust (DR)
       - 0.95
       - 1.00
       - 0.92
     * - vertical only (VT)
       - **0.63**
       - 0.93
       - 0.58

  The DR variance keeps coverage near nominal everywhere; the vertical-only
  variance covers the horizontal estimand just 63% of the time -- the paper's
  motivation for the doubly-robust construction. Run with
  ``python benchmarks/run_benchmarks.py --case rsc_shen_coverage`` (skips if the
  repo cannot be cloned).

References
----------

Shen, D., Ding, P., Sekhon, J., & Yu, B. (2023). "Same Root Different Leaves:
Time Series and Cross-Sectional Methods in Panel Data." *Econometrica*
91(6):2125-2154.

Rho, S., Tang, A., Bergam, N., Cummings, R., & Misra, V. (2025). "ClusterSC:
Advancing Synthetic Control with Donor Selection." *arXiv:2503.21629.*

Amjad, M., Shah, D., & Shen, D. (2018). "Robust Synthetic Control." *Journal of
Machine Learning Research* 19(22):1-51.
