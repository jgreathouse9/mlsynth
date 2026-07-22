.. _replication-mvbbsc:

MVBBSC -- Bayesian Synthetic Control (Martinez & Vives-i-Bastida 2024)
======================================================================

:Estimator: :doc:`../mvbbsc` -- :class:`mlsynth.MVBBSC`
:Source: Martinez, I. and Vives-i-Bastida, J. (2024), *"Bayesian and
   Frequentist Inference for Synthetic Controls,"* arXiv:2206.01779
   [MVBBSC2024]_.
:Replication type: cross-validation against the authors' own reference
   implementation, the ``bsynth`` R package, on the German reunification panel;
   plus Path A, the German reunification study.
:Status: verified -- the NumPyro posterior matches the ``bsynth`` (Stan/rstan)
   posterior to Monte-Carlo error, in both the point estimates and the credible
   bands.

Validation strategy
-------------------

Martinez and Vives-i-Bastida (2024) ship their method as the ``bsynth`` R
package, which fits the model in Stan via ``rstan``. mlsynth's MVBBSC is a
clean-room port of the method -- the same generative model (uniform-Dirichlet
simplex weights, ``HalfNormal`` scale, Gaussian likelihood on the
pre-period-standardized series) sampled with NUTS in NumPyro -- so ``bsynth`` is
the ground truth. We fit both on the identical panel and compare the posterior
counterfactual, the ATT, and the credible bands.

The reference is the ``bsynth`` model ``bayesianSynth`` with
``predictor_match = FALSE`` (outcome-only, no covariates, no Gaussian-process
term), ``ci_width = 0.95``, and four chains of 1000 warm-up + 1000 draws. The
mlsynth fit uses the same chain configuration and seed on
``basedata/german_reunification.csv`` (West Germany treated, 16 OECD donors,
1960--2003, reunification in 1990).

Cross-validation -- point estimates and bands
---------------------------------------------

Both samplers agree to Monte-Carlo error:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Quantity
     - MVBBSC (NumPyro)
     - bsynth (rstan)
   * - pre-1990 in-sample RMSE
     - 62.2
     - 62.2
   * - mean post-1990 ATT
     - :math:`-2078`
     - :math:`-2075`
   * - mean 95% band width, pre-1990
     - 300.4
     - 300.9
   * - mean 95% band width, post-1990
     - 1162
     - 1155

Year by year the 95% credible-band widths agree to within a few percent (ratios
:math:`0.95`--:math:`1.07`), the residual being pure Monte-Carlo error between
two independent NUTS runs. The band is well calibrated: in-sample coverage of the
observed West Germany series is :math:`96.7\%` against the nominal :math:`95\%`,
and it has the right shape -- tight in the pre-period where the donors pin the
fit (:math:`\sim 300`), fanning out through the post-period as the counterfactual
extrapolates (to :math:`\sim 2400` by 2003).

Path A -- the reunification result
----------------------------------

On its own terms MVBBSC recovers the classical reunification finding. The mean
post-1990 effect near :math:`-2080` PPP-USD per capita is about a :math:`7.5\%`
decline in West German per-capita GDP -- the magnitude the authors themselves
report in the text -- with the effect widening after 2000 and a credible band
that grows with the horizon.

Reproducing
-----------

The durable case ``benchmarks/cases/mvbbsc_germany.py`` pins the reproducible
headline quantities of a fresh NumPyro fit (the negative effect near the
:math:`7.5\%` magnitude, a close pre-period fit, an ordered credible band, and
converged NUTS). The reference ``bsynth`` numbers were produced with the R script
under ``benchmarks/R/mvbbsc_bsynth_ref.R`` (install pinned via
``benchmarks/R/install_bsynth.sh``); because it needs R and rstan it is not run
in CI, but the script is committed so the cross-check is reproducible.
