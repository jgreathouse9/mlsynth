.. _replication-mtgp:

MTGP -- Multitask Gaussian Process Synthetic Control (Ben-Michael et al. 2023)
==============================================================================

:Estimator: :doc:`../mtgp` -- :class:`mlsynth.MTGP`
:Source: Ben-Michael, E., Arbour, D., Feller, A., Franks, A., & Raphael, S.
   (2023), *"Estimating the effects of a California gun control program with
   multitask Gaussian processes,"* Annals of Applied Statistics 17(2), 985--1016
   [MTGP2023]_.
:Replication type: cross-validation against the authors' own reference (the
   replication package's Stan program) on the paper's California panel.
:Status: verified -- the NumPyro posterior matches the Stan posterior
   cell-for-cell (to Monte-Carlo error).

Validation strategy
-------------------

Ben-Michael et al. (2023) ship their model as a Stan program in the paper's
replication package. MTGP ports that program to NumPyro (both are NUTS
samplers), so the Stan program is the ground truth. We transcribe the Gaussian
(``normal.stan``) model verbatim, run it via ``rstan`` on the study's California
panel, and compare the posterior counterfactual of California to the NumPyro
estimator on the identical data.

The panel is the paper's own: gun-homicide rates per 100,000 across the 50
states, 1997--2018 (22 years), with California treated in 2007 -- the year the
Armed and Prohibited Persons System (APPS) began removing firearms from
prohibited owners. Population enters the noise model, so a small state's noisy
rate is downweighted relative to a large one.

Cross-validation -- cell for cell
---------------------------------

Both samplers use the same rank-5 coregionalization, squared-exponential time
kernels for the global trend and the factors, population-scaled Gaussian noise,
and matched NUTS budgets. The posterior counterfactual of California agrees:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Quantity
     - NumPyro vs Stan
   * - counterfactual, posterior mean (all 22 years)
     - max :math:`|\Delta|` = 0.2% of level; corr 0.99993
   * - mean post-2007 ATT (obs -- counterfactual)
     - :math:`-1.029` vs :math:`-1.031` per 100,000 (agree to 0.002)
   * - length-scales / noise scale
     - :math:`\rho_f, \rho_g, \sigma` agree to ~2%

The residual is the Monte-Carlo difference between two independent NUTS runs, not
a specification gap. A coregionalization model is rotation/sign non-identified,
so the raw loadings do not mix (Stan flags this too); the identified quantities
-- the counterfactual, the ATT, and the length-scales -- mix cleanly and match.

The one build note that matters
-------------------------------

The GP Cholesky needs double precision. JAX defaults to ``float32``, whose
machine epsilon (:math:`\approx 10^{-7}`) is larger than the :math:`10^{-9}`
jitter added to the kernel diagonal, so in single precision the jitter is lost
and the Cholesky factorizes a numerically indefinite matrix -- the sampler then
produces garbage (negative rates, near-zero correlation with Stan). The fix is a
single line, ``numpyro.enable_x64()``, called in the estimator's backend shim
before the model is built. With ``float64`` the port matches Stan to
correlation :math:`0.99993` and R-hat :math:`1.001`--:math:`1.005` on the
identified quantities. This is the standard trap for GP models on JAX and is the
reason the estimator hard-enables x64 rather than leaving it to the caller.

Why NumPyro, not pure numpy
---------------------------

The model marginalizes a Gaussian process whose kernel hyperparameters
(length-scales, marginal scales) are themselves sampled, so the geometry is a
correlated, heavy-tailed hierarchy -- exactly what a hand-written Gibbs sampler
struggles with (the kernel conditionals go ill-conditioned as the length-scales
move). NUTS handles it, so MTGP uses NumPyro behind the ``[bayes]`` optional
dependency, following :doc:`../bfsc` and :doc:`../spotsynth`. The durable case is
``benchmarks/cases/mtgp_california.py``; it feeds the identical California panel
to both :class:`mlsynth.MTGP` and the compiled Stan reference and reports the
cell-for-cell agreement, requiring the ``[bayes]`` extra for mlsynth and
``rstan`` for the reference, and skipping gracefully when either is absent.
